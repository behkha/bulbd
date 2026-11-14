"""What the lamp does on its own: the flows, the sleep timer, what Stop leaves
on the lamp and the power-on look - with the bulb replaced by fakes."""
import socket
import time

import pytest

import bulb_alone as alone
import music_bulb
import music_bulb_dash as dash
from music_bulb import PALETTES, White, release_bulb
from test_desk import FakeLink, make_state, post, run_session
from test_restore import WARM, ScriptedSock, StoppedBulb, no_new_connection


def steps(cmd):
    """("start_cf", [count, action, expr]) -> count, action, [[ms, mode, value, bright]]."""
    method, (count, action, expr) = cmd
    assert method == "start_cf"
    nums = [int(x) for x in expr.split(",")]
    assert len(nums) % 4 == 0
    return count, action, [nums[i:i + 4] for i in range(0, len(nums), 4)]


# ------------------------------------------------------------------ flows
def test_rgb_int():
    assert alone.rgb_int(0, 100) == 0xFF0000
    assert alone.rgb_int(120, 100) == 0x00FF00
    assert alone.rgb_int(240, 0) == 0xFFFFFF
    assert alone.rgb_int(360 + 240, 100) == 0x0000FF


def test_flow_clamps_steps_and_brightness():
    count, action, s = steps(alone.flow([(10, 1, 0xFF, 0), (900, 2, 2700, 250)], count=2, action=1))
    assert (count, action) == (2, 1)
    assert s == [[50, 1, 0xFF, 1], [900, 2, 2700, 100]]


def test_candle_flickers_warm_white_for_ever():
    count, action, s = steps(alone.candle_flow())
    assert count == 0 and action == 0
    assert {mode for _, mode, _, _ in s} == {alone.MODE_CT}
    assert {v for _, _, v, _ in s} == {alone.CANDLE_CT}
    assert len({b for *_, b in s}) > 3                    # it flickers


def test_breathe_goes_in_and_out_on_white():
    count, action, s = steps(alone.breathe_flow())
    assert count == 0 and [m for _, m, _, _ in s] == [alone.MODE_CT] * 2
    assert s[0][3] > s[1][3]


def test_drift_walks_the_palette_anchors():
    anchors = PALETTES["lagoon"]
    count, action, s = steps(alone.drift_flow(anchors))
    assert count == 0 and len(s) == len(anchors)
    assert [v for _, _, v, _ in s] == [alone.rgb_int(h, sat) for h, sat in anchors]
    assert all(ms == alone.DRIFT_STEP_MS and mode == alone.MODE_RGB for ms, mode, _, _ in s)


def test_drift_with_one_anchor_still_makes_a_flow():
    _, _, s = steps(alone.drift_flow([(30, 80)]))
    assert len(s) == 2


@pytest.mark.parametrize("name", ["candle", "drift", "breathe"])
def test_each_idle_look_is_one_flow(name):
    cmds = alone.idle_commands(name, PALETTES["ember"])
    assert len(cmds) == 1 and cmds[0][0] == "start_cf"


@pytest.mark.parametrize("name", ["restore", "bogus", None])
def test_restore_and_unknown_looks_send_nothing(name):
    assert alone.idle_commands(name) == []


# ------------------------------------------------------------- sleep flow
@pytest.mark.parametrize("saved, step", [
    (WARM, [2, 2700, 30]),
    (dict(WARM, color_mode=1, rgb=0x3366FF), [1, 0x3366FF, 30]),
    (dict(WARM, color_mode=3, hue=120, sat=100), [1, 0x00FF00, 30]),
    (dict(WARM, power="off"), None),
    (None, None),
])
def test_look_step(saved, step):
    assert alone.look_step(saved) == step


def test_fade_holds_then_dims_to_one_percent_then_switches_off():
    count, action, s = steps(alone.fade_flow([1, 0x3366FF, 70], 30 * 60))
    assert action == 2 and count == len(s) == 2
    assert s[0] == [20 * 60000, 1, 0x3366FF, 70]
    assert s[1] == [alone.SLEEP_FADE_S * 1000, 1, 0x3366FF, 1]
    assert sum(ms for ms, *_ in s) == 30 * 60000           # ends with the bulb's timer


def test_a_timer_shorter_than_the_fade_is_all_fade():
    count, action, s = steps(alone.fade_flow([2, 2700, 40], 120))
    assert count == 1 and action == 2 and s == [[120000, 2, 2700, 1]]


def test_sleep_commands():
    assert alone.sleep_commands(30) == [("cron_add", [0, 30])]
    cmds = alone.sleep_commands(30, [2, 2700, 40])
    assert cmds[0] == ("cron_add", [0, 30]) and cmds[1][0] == "start_cf"


def test_cancel_never_uses_stop_cf_which_would_switch_off():
    assert alone.cancel_commands() == [("cron_del", [0])]
    cmds = alone.cancel_commands([2, 2700, 40])
    assert "stop_cf" not in [m for m, _ in cmds]
    count, action, s = steps(cmds[1])
    assert (count, action) == (1, 1) and s[0][1:] == [2, 2700, 40]


@pytest.mark.parametrize("v, out", [(30, 30), (29.6, 30), (0, None), (1441, None),
                                    (True, None), ("30", None), (float("nan"), None)])
def test_clean_minutes(v, out):
    assert alone.clean_minutes(v) == out


def test_clean_sleep():
    now = 1_000_000.0
    good = {"at": now + 600, "step": [2, 2700, 40]}
    assert alone.clean_sleep(good, now) == good
    assert alone.clean_sleep({"at": now + 600, "step": [9, 1, 1]}, now) == {"at": now + 600, "step": None}
    assert alone.clean_sleep({"at": now - 1}, now) is None              # already over
    assert alone.clean_sleep({"at": now + 10 * 86400}, now) is None     # beyond what the bulb takes
    assert alone.clean_sleep("soon", now) is None


def test_dimming():
    assert alone.dim_factor(None) == 1.0
    assert alone.dim_factor(alone.SLEEP_FADE_S + 5) == 1.0
    assert alone.dim_factor(alone.SLEEP_FADE_S / 2) == pytest.approx(0.5)
    assert alone.dim_factor(0) == 0.01
    assert alone.dim_frame((10, 80, 60), 0.5) == (10, 80, 30)
    assert alone.dim_frame((10, 80, 1), 0.01) == (10, 80, 1)
    assert alone.dim_frame(White(4000, 40), 0.25) == White(4000, 10)


def test_dimmed_link_scales_frames_and_passes_the_rest_through():
    class L:
        lost = False
        got = None

        def put(self, frame, sudden, duration=40):
            self.got = (frame, sudden, duration)

    link = L()
    d = alone.Dimmed(link, lambda: 0.5)
    d.put((0, 100, 80), True, duration=30)
    assert link.got == ((0, 100, 40), True, 30)
    d.lost = True                                          # a loop marks the link lost
    assert link.lost is True and d.lost is True


# ---------------------------------------------------------------- release
def test_then_follows_the_restore_on_the_same_connection(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", no_new_connection)
    control = ScriptedSock()
    flow = alone.candle_flow()
    assert release_bulb(StoppedBulb(control), restore=True, then=[flow]) is True
    assert [m for m, _ in control.sent] == ["set_ct_abx", "set_bright", "start_cf"]
    assert control.closed


def test_then_alone_replaces_the_restore(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", no_new_connection)
    control = ScriptedSock()
    assert release_bulb(StoppedBulb(control), restore=False, then=[alone.breathe_flow()]) is False
    assert [m for m, _ in control.sent] == ["start_cf"] and control.closed


def test_the_sender_is_joined_before_anything_is_sent(monkeypatch):
    order = []

    class Sender:
        lost = type("E", (), {"is_set": lambda self: False})()

        def stop(self):
            order.append("joined")

    link = music_bulb.Link("192.0.2.1")
    link.sender, link.bulb = Sender(), object()
    monkeypatch.setattr(music_bulb, "release_bulb",
                        lambda bulb, restore, then=(): order.append(("sent", list(then))) or False)
    link.close(restore=False, then=[alone.candle_flow()])
    assert order[0] == "joined" and order[1][0] == "sent" and order[1][1][0][0] == "start_cf"


# --------------------------------------------------------- desk settings
def test_idle_and_sleep_are_saved_settings():
    assert "idle" in dash.SAVED_KEYS and "sleep" in dash.SAVED_KEYS
    assert "idle" not in dash.LOOK_KEYS                     # the room, not the look
    assert make_state().snapshot()["idle"] == "restore"     # restore stays the default


@pytest.mark.parametrize("v, kept", [("candle", "candle"), ("drift", "drift"), ("breathe", "breathe"),
                                     ("restore", "restore"), ("disco", "gone"), (3, "gone")])
def test_idle_is_validated(v, kept):
    assert dash.clean_patch({"idle": v}, {"monitors": []}).get("idle", "gone") == kept


def test_the_page_sets_idle_but_never_sleep():
    st = make_state(ip="192.168.1.2")
    post(st, {"idle": "candle", "sleep": {"at": time.time() + 600, "step": None}})
    snap = st.snapshot()
    assert snap["idle"] == "candle" and snap["sleep"] is None


def test_a_saved_timer_loads_until_it_has_run_out():
    ahead = {"at": time.time() + 600, "step": [2, 2700, 40]}
    assert make_state(sleep=ahead).snapshot()["sleep"] == ahead
    assert make_state(sleep={"at": time.time() - 5}).snapshot()["sleep"] is None


# ------------------------------------------------------ Stop, per look
@pytest.fixture
def engine(monkeypatch):
    FakeLink.made, FakeLink.plan = [], {}
    monkeypatch.setattr(dash, "Link", FakeLink)
    monkeypatch.setattr(dash, "screen_permission", lambda: True)
    st = make_state(ip="192.168.1.2")
    eng = dash.Engine(st)
    eng.sleep.call = lambda ip, batches: []
    return st, eng


def stop_once(st, eng, saved=None):
    def loop(st_, link):
        if saved is not None:
            link.saved = saved
        st_.set(running=False)
    run_session(st, eng, loop)
    return FakeLink.made[0]


def test_restore_is_the_default_stop(engine):
    st, eng = engine
    link = stop_once(st, eng)
    assert link.closed is True and link.then == []


@pytest.mark.parametrize("name", ["candle", "breathe"])
def test_an_idle_look_replaces_the_restore(engine, name):
    st, eng = engine
    st.set(idle=name)
    link = stop_once(st, eng)
    assert link.closed is False and link.then == alone.idle_commands(name)


def test_drift_walks_the_selected_palette(engine):
    st, eng = engine
    st.set(idle="drift", palette="lagoon")
    link = stop_once(st, eng)
    assert link.closed is False and link.then == [alone.drift_flow(PALETTES["lagoon"])]


def test_drift_on_auto_walks_the_palette_it_was_on(engine):
    st, eng = engine
    st.set(idle="drift", palette="auto", active_palette="orchid")
    assert stop_once(st, eng).then == [alone.drift_flow(PALETTES["orchid"])]


def test_drift_takes_the_users_own_palette(engine):
    st, eng = engine
    mine = [[10, 90], [200, 70], [300, 60]]
    st.set(idle="drift", palette="Mine", palettes=[{"name": "Mine", "anchors": mine}])
    assert stop_once(st, eng).then == [alone.drift_flow([tuple(a) for a in mine])]


def test_no_restore_flag_leaves_the_last_frame(engine):
    st, eng = engine
    eng.restore = False
    link = stop_once(st, eng)
    assert link.closed is False and link.then == []


def test_restore_with_a_timer_hands_the_fade_to_the_bulb(engine):
    st, eng = engine
    st.set(sleep={"at": time.time() + 1800, "step": None})
    link = stop_once(st, eng, saved=dict(WARM))
    assert link.closed is True and len(link.then) == 1
    count, action, s = steps(link.then[0])
    assert action == 2 and s[-1][1:] == [2, 2700, 1]
    assert sum(ms for ms, *_ in s) == pytest.approx(1800 * 1000, abs=3000)
    assert st.snapshot()["sleep"]["step"] == [2, 2700, 30]      # Cancel puts this back


def test_a_lamp_restored_to_off_drops_the_timer(engine):
    st, eng = engine
    st.set(sleep={"at": time.time() + 1800, "step": None})
    link = stop_once(st, eng, saved=dict(WARM, power="off"))
    assert link.closed is True and link.then == [("cron_del", [0])]
    assert st.snapshot()["sleep"] is None


def test_an_idle_look_keeps_running_under_a_timer(engine):
    st, eng = engine
    st.set(idle="candle", sleep={"at": time.time() + 1800, "step": None})
    link = stop_once(st, eng, saved=dict(WARM))
    assert link.then == [alone.candle_flow()]
    assert st.snapshot()["sleep"] is not None               # the bulb still switches it off


def test_a_timer_running_out_ends_the_session_dark(engine):
    st, eng = engine
    st.set(idle="candle")

    def loop(st_, link):
        st_.set(sleep={"at": time.time() + 600, "step": None})
        eng.sleep._fire()                                    # the timer runs out
        assert not st_.snapshot()["running"]
    run_session(st, eng, loop)
    link = FakeLink.made[0]
    assert link.closed is False and link.then == [("set_power", ["off", "smooth", 500])]
    assert st.snapshot()["sleep"] is None and not eng.sleep.fired.is_set()


def test_the_loops_get_a_dimmed_link(engine):
    st, eng = engine
    got = []

    def loop(st_, link):
        got.append(link)
        st_.set(running=False)
    run_session(st, eng, loop)
    assert isinstance(got[0], alone.Dimmed)


# ------------------------------------------------------------ the timer
class Bulb:
    """lamp_call batches as the SleepTimer sends them."""

    def __init__(self, props=("on", "70", "1", "3368703", "2700", "0", "100", "0"), fail=None):
        self.props, self.fail, self.sent = list(props), fail, []

    def __call__(self, ip, batches):
        out = []
        for cmds in batches:
            cmds = cmds(out) if callable(cmds) else cmds
            if self.fail:
                raise self.fail
            self.sent.append(cmds)
            out.append([self.props if m == "get_prop" else ["ok"] for m, _ in cmds])
        return out


def timer(bulb, **saved):
    st = make_state(ip="192.168.1.2", **saved)
    return st, dash.SleepTimer(st, call=bulb)


def test_set_while_stopped_reads_the_lamp_then_sets_timer_and_fade():
    bulb = Bulb()
    st, t = timer(bulb)
    try:
        assert t.start(30) == ""
        assert bulb.sent[0][0][0] == "get_prop"
        assert bulb.sent[1] == alone.sleep_commands(30, [1, 3368703, 70])
        cur = st.snapshot()["sleep"]
        assert cur["step"] == [1, 3368703, 70] and 1795 < cur["at"] - time.time() <= 1800
        assert 1795 < t.left() <= 1800
    finally:
        t._disarm()


def test_set_over_an_idle_look_only_counts():
    bulb = Bulb(props=("on", "50", "2", "0", "2000", "0", "0", "1"))       # flowing
    st, t = timer(bulb)
    try:
        assert t.start(15) == ""
        assert bulb.sent[1] == [("cron_add", [0, 15])]
        assert st.snapshot()["sleep"]["step"] is None
    finally:
        t._disarm()


def test_set_during_a_session_only_counts():
    bulb = Bulb()
    st, t = timer(bulb)
    st.set(running=True)
    try:
        assert t.start(60) == ""
        assert bulb.sent == [[("cron_add", [0, 60])]]
    finally:
        t._disarm()


def test_a_lamp_that_is_off_says_so():
    bulb = Bulb(fail=RuntimeError("cron_add: {'code': -5006, 'message': 'illegal request'}"))
    st, t = timer(bulb)
    assert t.start(30) == "The lamp is off."
    assert st.snapshot()["sleep"] is None


def test_a_silent_bulb_says_so():
    st, t = timer(Bulb(fail=TimeoutError("get_prop: no answer")))
    assert t.start(30) == "The bulb did not answer."


def test_cancel_clears_the_timer_and_puts_the_faded_look_back():
    bulb = Bulb()
    st, t = timer(bulb, sleep={"at": time.time() + 600, "step": [2, 2700, 40]})
    assert t._timer is not None                              # armed from the saved file
    assert t.cancel() == ""
    assert bulb.sent == [alone.cancel_commands([2, 2700, 40])]
    assert st.snapshot()["sleep"] is None and t._timer is None


def test_running_out_while_stopped_only_clears_the_desk():
    st, t = timer(Bulb(), sleep={"at": time.time() + 600, "step": None})
    t._disarm()
    t._fire()
    assert st.snapshot()["sleep"] is None and not t.fired.is_set()


def test_sleep_endpoint(monkeypatch):
    bulb = Bulb()
    st = make_state(ip="192.168.1.2")
    eng = type("E", (), {})()
    eng.sleep = dash.SleepTimer(st, call=bulb)
    monkeypatch.setattr(dash.Handler, "engine", eng, raising=False)
    try:
        code, body = post(st, {"minutes": 30}, path="/api/sleep")
        assert code == 200 and st.snapshot()["sleep"]
        assert post(st, {"minutes": 9999}, path="/api/sleep")[0] == 400
        code, _ = post(st, {"minutes": 0}, path="/api/sleep")
        assert code == 200 and st.snapshot()["sleep"] is None
    finally:
        eng.sleep._disarm()


def test_power_on_look_endpoint(monkeypatch):
    sent = []
    monkeypatch.setattr(music_bulb, "lamp_call", lambda ip, cmds, **kw: sent.append((ip, cmds)) or [["ok"]])
    st = make_state(ip="192.168.1.2")
    code, _ = post(st, {}, path="/api/default")
    assert code == 200 and sent == [("192.168.1.2", [("set_default", [])])]
    assert post(make_state(), {}, path="/api/default")[0] == 400       # no bulb set up
