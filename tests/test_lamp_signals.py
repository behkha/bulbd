"""Signals: alert pulses, the Focus timer and the sunrise alarm - the flow
builders, the hold-and-resume path in a running session, the pomodoro state
machine, the alarm schedule across midnight and the desk endpoints."""
import datetime as dt
import io
import json
import threading
import time

import pytest

import lamp_signals as ls
import music_bulb_dash as dash
from music_bulb import Link, Sender
from test_restore import ScriptedSock


def steps_of(expr):
    nums = [int(x) for x in expr.split(",")]
    assert len(nums) % 4 == 0
    return [tuple(nums[i:i + 4]) for i in range(0, len(nums), 4)]


# ------------------------------------------------------------------ alerts
@pytest.mark.parametrize("pulses", [1, 2, 3, 5])
@pytest.mark.parametrize("speed", list(ls.SPEEDS))
def test_alert_flow_pulses_the_colour_and_fits_the_bulb(pulses, speed):
    count, expr, ms = ls.alert_flow(0x2FB457, pulses, speed)
    steps = steps_of(expr)
    assert count == len(steps) == 2 * pulses <= ls.MAX_STEPS
    half = ls.SPEEDS[speed]
    assert steps[:2] == [(half, 1, 0x2FB457, 100), (half, 1, 0x2FB457, 1)]
    assert ms == sum(s[0] for s in steps) == 2 * half * pulses


def test_a_flow_that_ends_on_white_adds_one_step_and_its_time():
    count, expr, ms = ls.flow_then_white(ls.alert_flow(0xFF0000, 1), 5000, 90)
    assert count == 3 and steps_of(expr)[-1] == (600, 2, 5000, 90) and ms == 700 + 600


@pytest.mark.parametrize("body, want", [
    ({}, (0xE5352B, 2, "normal")),
    ({"color": "#2F7DE1", "pulses": 5, "speed": "fast"}, (0x2F7DE1, 5, "fast")),
    ({"color": "2fb457"}, (0x2FB457, 2, "normal")),
    ({"color": "Purple", "pulses": 1}, (0xA24BE0, 1, "normal")),
    ({"color": "#123", "x": 1}, None), ({"color": "#000000"}, None), ({"color": 5}, None),
    ({"pulses": 0}, None), ({"pulses": 6}, None), ({"pulses": True}, None),
    ({"pulses": 2.0}, None), ({"speed": "warp"}, None), ([], None),
])
def test_an_alert_takes_only_what_a_pulse_needs(body, want):
    picked, error = ls.clean_alert(body)
    assert picked == want and bool(error) == (want is None)


def scripted(power):
    sock = ScriptedSock(props=[power])
    return sock, (lambda addr, timeout: sock)


def test_an_alert_on_a_lit_lamp_is_start_cf_with_recover():
    sock, connect = scripted("on")
    flow = ls.alert_flow(0xFF0000, 2)
    ls.alert_lamp("10.0.0.9", flow, connect=connect)
    assert sock.sent == [("get_prop", ["power"]), ("start_cf", [4, 0, flow[1]])]
    assert sock.closed


def test_an_alert_on_a_lamp_that_is_off_lights_it_with_set_scene():
    # start_cf is ignored by an off color5; set_scene cf with action 0 lights
    # it for the pulse and leaves it off again (measured on the lamp)
    sock, connect = scripted("off")
    flow = ls.alert_flow(0xFF0000, 1)
    ls.alert_lamp("10.0.0.9", flow, connect=connect)
    assert sock.sent[-1] == ("set_scene", ["cf", 2, 0, flow[1]]) and sock.closed


# ----------------------------------------------- the hold in a running session
class MusicBulb:
    """A bulb in music mode: frames and flows, each with the time it landed."""

    def __init__(self):
        self.log = []
        self.lock = threading.Lock()

    def set_hsv(self, h, s, v, effect, duration):
        with self.lock:
            self.log.append(((h, s, v), time.monotonic()))

    def send_command(self, method, params):
        with self.lock:
            self.log.append((method, time.monotonic()))


@pytest.fixture
def running():
    made = []

    def make():
        bulb = MusicBulb()
        s = Sender(bulb)
        s.start()
        made.append(s)
        return bulb, s

    yield make
    for s in made:
        s.stop()


def wait_for(cond, secs=2.0):
    end = time.monotonic() + secs
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_frames_wait_while_the_pulse_runs_then_the_newest_goes_out(running):
    bulb, s = running()
    s.put((10, 90, 50), False)
    assert wait_for(lambda: len(bulb.log) == 1)
    t0 = time.monotonic()
    s.flow([4, 0, "50,1,255,100,50,1,255,1,50,1,255,100,50,1,255,1"], 200)
    for v in range(20, 40):                     # the loop keeps putting frames
        s.put((20, 90, v), False)
        time.sleep(0.005)
    assert wait_for(lambda: len(bulb.log) == 3)
    time.sleep(0.1)
    (first, _), (flow, t_flow), (frame, t_frame) = bulb.log
    assert first == (10, 90, 50) and flow == "start_cf"
    assert frame == (20, 90, 39)                # only the newest held frame
    assert t_frame - t0 >= 0.2 + Sender.HOLD_MARGIN - 0.01
    assert len(bulb.log) == 3


def test_after_the_pulse_frames_flow_freely_again(running):
    bulb, s = running()
    s.flow([2, 0, "50,1,255,100,50,1,255,1"], 60)
    assert wait_for(lambda: len(bulb.log) == 1)
    time.sleep(0.2)
    s.put((1, 2, 3), False)
    assert wait_for(lambda: len(bulb.log) == 2)
    assert bulb.log[1][0] == (1, 2, 3) and bulb.log[1][1] - bulb.log[0][1] >= 0.14


def test_a_pulse_with_no_frames_after_it_sends_nothing_more(running):
    bulb, s = running()
    s.flow([2, 0, "50,1,255,100,50,1,255,1"], 50)
    time.sleep(0.3)
    assert [m for m, _ in bulb.log] == ["start_cf"]


def test_stop_during_a_pulse_ends_the_sender_at_once(running):
    bulb, s = running()
    s.flow([2, 0, "5000,1,255,100,5000,1,255,1"], 10000)
    s.put((1, 1, 1), False)
    t0 = time.monotonic()
    s.stop()
    assert not s.is_alive() and time.monotonic() - t0 < 1.0


def test_link_flow_needs_a_live_sender():
    link = Link("10.0.0.9")
    assert link.flow([2, 0, "x"], 100) is False
    link.sender = Sender(MusicBulb())
    link.sender.lost.set()
    assert link.flow([2, 0, "x"], 100) is False


# ----------------------------------------------------------------- sunrise
def test_sunrise_is_one_flow_from_dim_warm_to_bright_cool():
    count, expr, ms = ls.sunrise_flow()
    steps = steps_of(expr)
    assert count == len(steps) <= ls.MAX_STEPS
    assert steps[0] == (50, 2, 1700, 1) and steps[-1][2:] == (6500, 100)
    assert all(s[1] == 2 for s in steps)
    cts = [s[2] for s in steps]
    brights = [s[3] for s in steps]
    assert cts == sorted(cts) and brights == sorted(brights)
    assert sum(s[0] for s in steps) == ms and abs(ms - 30 * 60 * 1000) <= 100


def test_a_shortened_ramp_keeps_its_shape():
    _, expr, ms = ls.sunrise_flow(60_000)
    assert abs(ms - 60_050) <= 10 and len(steps_of(expr)) == 6


D = dt.datetime


@pytest.mark.parametrize("now, at, want", [
    (D(2026, 11, 14, 6, 0), 7 * 60, D(2026, 11, 14, 7, 0)),
    (D(2026, 11, 14, 7, 0), 7 * 60, D(2026, 11, 15, 7, 0)),           # strictly after
    (D(2026, 11, 14, 23, 50), 10, D(2026, 11, 15, 0, 10)),            # across midnight
    (D(2026, 11, 14, 23, 50), 23 * 60 + 40, D(2026, 11, 15, 23, 40)),
    (D(2026, 12, 31, 23, 59, 30), 0, D(2027, 1, 1, 0, 0)),            # across the year
])
def test_next_fire(now, at, want):
    assert ls.next_fire(now, at) == want


def test_the_alarm_fires_once_at_its_time_across_midnight():
    plan, at = ls.AlarmPlan(), 5               # 00:05
    now = D(2026, 11, 14, 23, 58)
    fired = []
    while now < D(2026, 11, 15, 0, 30):
        fire, wait = plan.step(now, True, at)
        if fire:
            fired.append(now)
        assert 0 < wait <= ls.CHECK_S
        now += dt.timedelta(seconds=wait)
    assert len(fired) == 1 and D(2026, 11, 15, 0, 5) <= fired[0] < D(2026, 11, 15, 0, 5, 1)
    assert plan.target == D(2026, 11, 16, 0, 5)


def test_a_mac_that_wakes_late_still_gets_its_sunrise_but_not_hours_late():
    plan = ls.AlarmPlan()
    plan.step(D(2026, 11, 14, 23, 0), True, 7 * 60)
    assert plan.step(D(2026, 11, 15, 7, 12), True, 7 * 60)[0] is True     # slept past it
    plan = ls.AlarmPlan()
    plan.step(D(2026, 11, 14, 23, 0), True, 7 * 60)
    assert plan.step(D(2026, 11, 15, 9, 0), True, 7 * 60)[0] is False
    assert plan.target == D(2026, 11, 16, 7, 0)


def test_an_alarm_that_is_off_sleeps_until_a_change_and_a_new_time_is_taken():
    plan = ls.AlarmPlan()
    assert plan.step(D(2026, 11, 14, 6, 0), False, 420) == (False, None)
    plan.step(D(2026, 11, 14, 6, 0), True, 420)
    assert plan.target == D(2026, 11, 14, 7, 0)
    plan.step(D(2026, 11, 14, 6, 1), True, 6 * 60 + 30)
    assert plan.target == D(2026, 11, 14, 6, 30)


# ------------------------------------------------------------------- Focus
def test_focus_walks_work_break_work():
    f = ls.Focus()
    assert f.wait(0) is None and f.view(0)["phase"] == "idle"
    assert f.start(0, 1500) is True
    assert (f.phase, f.round, f.wait(100)) == ("work", 1, 1400)
    assert not f.due(1499) and f.due(1500)
    assert f.advance(1500, 1500, 300) == "break" and f.wait(1500) == 300
    assert f.advance(1800, 1200, 300) == "work" and f.round == 2 and f.wait(1800) == 1200
    assert f.view(1900) == {"phase": "work", "paused": False, "round": 2, "left": 1100}


def test_focus_pause_keeps_the_time_left_and_resume_carries_on():
    f = ls.Focus()
    f.start(0, 1500)
    f.pause(600)
    assert f.wait(5000) is None and not f.due(99999)
    assert f.view(9999)["left"] == 900 and f.view(9999)["paused"]
    assert f.start(10_000, 1500) is False                       # a resume, not a new block
    assert f.wait(10_000) == 900 and f.round == 1


def test_focus_reset_goes_idle():
    f = ls.Focus()
    f.start(0, 60)
    f.reset()
    assert not f.active and f.wait(0) is None and f.view(0)["left"] is None


# ------------------------------------------------------------ desk: Signals
class FakeLink:
    def __init__(self, ok=True):
        self.ok, self.flows = ok, []

    def flow(self, params, ms):
        self.flows.append((params, ms))
        return self.ok


def make(link=None, **saved):
    state = dash.State(dict({"ip": "10.0.0.9"}, **saved), store=None)
    lamp, pulses = [], []
    clock = [100.0]
    sig = ls.Signals(state, link=lambda: link,
                     lamp=lambda ip, cmds: lamp.append((ip, cmds)),
                     pulse=lambda ip, flow: pulses.append((ip, flow)),
                     read=lambda ip: {"power": "off", "bright": 30, "color_mode": 2,
                                      "ct": 2700, "rgb": None, "hue": None, "sat": None},
                     clock=lambda: clock[0])
    return sig, state, lamp, pulses, clock


def test_with_a_session_the_alert_goes_through_its_sender():
    link = FakeLink()
    sig, state, lamp, pulses, _ = make(link)
    code, body = sig.alert(0x2F7DE1, 3, "fast")
    count, expr, ms = ls.alert_flow(0x2F7DE1, 3, "fast")
    assert code == 200 and body["via"] == "session"
    assert link.flows == [([count, 0, expr], ms)] and not pulses


def test_without_a_session_the_alert_is_one_plain_connection():
    sig, state, lamp, pulses, _ = make(None)
    code, body = sig.alert(0xFF0000, 1, "normal")
    assert code == 200 and body["via"] == "lamp" and pulses == [("10.0.0.9", ls.alert_flow(0xFF0000, 1))]


def test_one_alert_at_a_time():
    sig, state, lamp, pulses, clock = make(None)
    assert sig.alert(0xFF0000, 2)[0] == 200
    clock[0] += 1.0
    assert sig.alert(0xFF0000, 2)[0] == 409
    clock[0] += 0.5
    assert sig.alert(0xFF0000, 2)[0] == 200


def test_an_alert_waits_for_a_session_that_is_still_connecting():
    sig, state, lamp, pulses, _ = make(FakeLink(ok=False))
    state.set(running=True)
    assert sig.alert(0xFF0000)[0] == 409 and not pulses


def test_an_unreachable_bulb_is_a_502():
    sig, state, *_ = make(None)
    sig.pulse = lambda ip, flow: (_ for _ in ()).throw(OSError("refused"))
    code, body = sig.alert(0xFF0000)
    assert code == 502 and "refused" in body["error"]


def test_focus_takes_the_lamp_and_reset_puts_it_back():
    sig, state, lamp, _, clock = make(None)
    code, body = sig.focus_op("start")
    assert code == 200 and body["focus"]["phase"] == "work"
    (ip, [(method, params)]), = lamp
    assert method == "set_scene" and params[:3] == ["cf", 3, 1]
    assert steps_of(params[3])[-1][1:] == (2, *ls.WORK_WHITE)
    sig.focus_op("reset")
    assert lamp[-1][1][-1] == ("set_power", ["off", "smooth", 300])
    assert sig.view()["focus"]["phase"] == "idle"


def test_focus_runs_on_a_stopped_desk_only_and_a_session_ends_it_first():
    sig, state, lamp, _, _ = make(None)
    state.set(running=True)
    assert sig.focus_op("start")[0] == 409 and not lamp
    state.set(running=False)
    sig.focus_op("start")
    sig.session_started()
    assert not sig.focus.active and lamp[-1][1][0][0] == "set_ct_abx"


def test_the_focus_loop_changes_phase_on_its_own_timer():
    sig, state, lamp, _, clock = make(None, focus_work=5, focus_break=1)
    sig.start()
    sig.focus_op("start")
    clock[0] += 5 * 60
    sig._focus_wake.set()
    assert wait_for(lambda: len(lamp) == 2)
    method, params = lamp[1][1][0]
    assert method == "start_cf" and params[1] == 1
    assert steps_of(params[2])[-1][1:] == (2, *ls.BREAK_WHITE)
    assert sig.view()["focus"]["phase"] == "break"


def test_the_sunrise_is_skipped_while_a_session_or_focus_has_the_lamp():
    sig, state, lamp, _, _ = make(None)
    state.set(running=True)
    assert sig.fire_sunrise() is False
    state.set(running=False)
    sig.focus_op("start")
    assert sig.fire_sunrise() is False
    sig.focus_op("reset")
    lamp.clear()
    assert sig.fire_sunrise(60_000) is True
    method, params = lamp[0][1][0]
    assert method == "set_scene" and params[0] == "cf" and params[2] == 1
    assert sig.view()["sunrise_ramping"] is True
    sig.sunrise_stop()
    assert lamp[-1][1] == [("stop_cf", [])] and sig.view()["sunrise_ramping"] is False


# ------------------------------------------------------------ desk: settings
def test_signal_settings_are_clamped_saved_and_not_part_of_a_look():
    out = dash.clean_patch({"focus_work": 500, "focus_break": 0, "sunrise_at": 1500,
                            "sunrise_on": "yes"}, {"monitors": []})
    assert out == {"focus_work": 90, "focus_break": 1, "sunrise_at": 1439}
    assert dash.clean_patch({"sunrise_on": True}, {"monitors": []}) == {"sunrise_on": True}
    for key in ("focus_work", "focus_break", "sunrise_at", "sunrise_on"):
        assert key in dash.SAVED_KEYS and key not in dash.LOOK_KEYS
    state = dash.State(None, store=None).snapshot()
    assert (state["focus_work"], state["focus_break"], state["sunrise_at"],
            state["sunrise_on"]) == (25, 5, 420, False)
    assert "focus" not in dash.SAVED_KEYS


# ------------------------------------------------------------ desk: endpoints
def post(state, signals, path, payload):
    h = dash.Handler.__new__(dash.Handler)
    h.state, h.signals = state, signals
    body = json.dumps(payload).encode()
    h.headers = {"Content-Length": str(len(body)), "Host": "localhost:8765"}
    h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
    h.path, h.command, h.request_version = path, "POST", "HTTP/1.1"
    h.requestline, h.client_address, h.close_connection = "", ("127.0.0.1", 0), False
    h.do_POST()
    head, _, out = h.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(out) if out.startswith(b"{") else out


def test_alert_endpoint_validates_then_pulses():
    sig, state, lamp, pulses, _ = make(None)
    assert post(state, sig, "/api/alert", {"color": "#zzzzzz"})[0] == 400
    assert not pulses
    code, body = post(state, sig, "/api/alert", {"color": "green", "pulses": 3, "speed": "slow"})
    assert code == 200 and body["via"] == "lamp" and pulses[0][1] == ls.alert_flow(0x2FB457, 3, "slow")


def test_focus_and_sunrise_endpoints():
    sig, state, lamp, _, _ = make(None)
    assert post(state, sig, "/api/focus", {"op": "dance"})[0] == 400
    assert post(state, sig, "/api/focus", {"op": "start"})[0] == 200
    assert post(state, sig, "/api/focus", {"op": "pause"})[1]["focus"]["paused"] is True
    assert post(state, sig, "/api/sunrise", {"op": "go"})[0] == 400
    assert post(state, sig, "/api/sunrise", {"op": "stop"})[0] == 200


def test_start_ends_focus_before_the_session_and_alarm_changes_wake_the_alarm():
    sig, state, lamp, _, _ = make(None)
    sig.focus_op("start")
    post(state, sig, "/api/control", {"running": True})
    assert not sig.focus.active and state.snapshot()["running"]
    sig._alarm_wake.clear()
    post(state, sig, "/api/control", {"sunrise_on": True, "sunrise_at": 390})
    assert sig._alarm_wake.is_set() and state.snapshot()["sunrise_at"] == 390
