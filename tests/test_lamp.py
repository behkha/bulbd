"""Lamp mode: the desk as a plain lamp while no session runs.

Every action is one short connection that is closed at once, Follow the day
wakes only while it is on and connects only when its value moved, and a
temperature set by hand wins until the next morning.
"""
import contextlib
import io
import json
import threading
import time
import urllib.error

import pytest

import music_bulb
import music_bulb_dash as dash
from music_bulb import lamp_commands, lamp_link, lamp_main, lamp_send, describe_lamp
from test_restore import ScriptedSock


# ------------------------------------------------------------ the helper
def test_actions_are_the_commands_that_need_no_current_value():
    assert lamp_commands("toggle") == [("toggle", [])]
    assert lamp_commands("up") == [("adjust_bright", [10, 400])]
    assert lamp_commands("down") == [("adjust_bright", [-10, 400])]
    assert lamp_commands("on") == [("set_power", ["on", "smooth", 400])]
    assert lamp_commands("off") == [("set_power", ["off", "smooth", 400])]


def test_temperature_and_brightness_are_clamped():
    assert lamp_commands(ct=9000, bright=0) == [("set_ct_abx", [6500, "smooth", 400]),
                                                ("set_bright", [1, "smooth", 400])]
    assert lamp_commands(ct=1000, bright=250)[0][1][0] == 2700
    assert lamp_commands(ct=1000, bright=250)[1][1][0] == 100
    with pytest.raises(ValueError):
        lamp_commands("explode")


def test_lamp_send_is_one_connection_that_reads_back_and_closes():
    socks = []

    def connect(addr, timeout):
        socks.append(ScriptedSock(props=["on", "62", "2", "255", "3400", "0", "100"]))
        return socks[-1]

    lamp = lamp_send("10.0.0.9", lamp_commands("up"), connect=connect)
    assert len(socks) == 1 and socks[0].closed
    assert [m for m, _ in socks[0].sent] == ["adjust_bright", "get_prop"]
    assert socks[0].nudges == 2                        # the firmware's extra byte, every time
    assert lamp == {"power": "on", "bright": 62, "color_mode": 2, "rgb": 255, "ct": 3400,
                    "hue": 0, "sat": 100}
    assert describe_lamp(lamp) == "on 3400 K 62%"


def test_lamp_link_closes_even_when_the_exchange_fails():
    sock = ScriptedSock(silent=True)
    with pytest.raises(TimeoutError):
        with lamp_link("10.0.0.9", timeout=0.05, connect=lambda a, t: sock) as call:
            call([("toggle", [])])
    assert sock.closed


def test_describe_lamp():
    assert describe_lamp({"power": "off"}) == "off"
    assert describe_lamp({"power": "on", "color_mode": 1, "bright": 5}) == "on colour 5%"
    assert describe_lamp(None) == "no answer"


# -------------------------------------------------------- the day curve
def at(h, m=0):
    return h * 60 + m


def test_curve_is_warm_at_night_and_cool_by_day():
    assert dash.day_kelvin(at(3)) == 2700
    assert dash.day_kelvin(at(6, 59)) == 2700
    assert dash.day_kelvin(at(8)) == 5000
    assert dash.day_kelvin(at(13)) == 5000
    assert dash.day_kelvin(at(20)) == 5000
    assert dash.day_kelvin(at(22)) == 2700
    assert dash.day_kelvin(at(23, 30)) == 2700


def test_curve_ramps_smoothly_in_steps():
    morning = [dash.day_kelvin(at(7, m)) for m in range(0, 61)]
    evening = [dash.day_kelvin(at(20, m)) for m in range(0, 121)]
    assert morning == sorted(morning) and evening == sorted(evening, reverse=True)
    assert 3500 <= dash.day_kelvin(at(7, 30)) <= 4200          # halfway up
    assert 3500 <= dash.day_kelvin(at(21)) <= 4200             # halfway down
    assert all(k % dash.DAY_STEP == 0 for k in morning + evening)


def test_curve_honours_its_settings_and_wraps_midnight():
    assert dash.day_kelvin(at(12), day_k=6500, night_k=3000) == 6500
    assert dash.day_kelvin(at(2), day_k=6500, night_k=3000) == 3000
    # a night owl: day from 13:00, evening at 23:30, so it is still cooling at 00:30
    assert dash.day_kelvin(at(0, 30), wake=at(13), dusk=at(23, 30)) not in (2700, 5000)
    assert dash.day_kelvin(at(2), wake=at(13), dusk=at(23, 30)) == 2700
    # a day too short for both ramps never reaches day_k, and never jumps
    short = [dash.day_kelvin(m, wake=at(12), dusk=at(12, 30)) for m in range(at(11), at(15))]
    assert max(short) < 5000 and max(abs(a - b) for a, b in zip(short, short[1:])) <= 100


def test_next_resume_is_the_coming_four_oclock():
    def local(h, m=0, day=14):
        return time.mktime((2026, 11, day, h, m, 0, 0, 0, -1))

    assert dash.next_resume(local(2)) == local(4)
    assert dash.next_resume(local(4)) == local(4, day=15)
    assert dash.next_resume(local(21, 30)) == local(4, day=15)


# ------------------------------------------------------- settings keys
def test_lamp_settings_are_cleaned_and_saved():
    out = dash.clean_patch({"follow_day": True, "day_k": 9999, "night_k": 100,
                            "wake": 422, "dusk": 2000, "day_held": 1.7e9}, {"monitors": []})
    assert out == {"follow_day": True, "day_k": 6500, "night_k": 2700,
                   "wake": 420, "dusk": 1425, "day_held": 1.7e9}
    bad = dash.clean_patch({"follow_day": "yes", "day_k": "cold", "wake": True,
                            "day_held": -3}, {"monitors": []})
    assert not set(bad) & set(dash.LAMP_DEFAULTS)
    assert set(dash.LAMP_DEFAULTS) <= set(dash.SAVED_KEYS)
    assert "lamp" not in dash.SAVED_KEYS and "lamp_error" not in dash.SAVED_KEYS


def test_a_file_without_the_lamp_keys_loads_the_defaults():
    snap = dash.State({"ip": "10.0.0.9"}, store=None).snapshot()
    assert {k: snap[k] for k in dash.LAMP_DEFAULTS} == dash.LAMP_DEFAULTS
    snap = dash.State({"follow_day": True, "wake": 360}, store=None).snapshot()
    assert snap["follow_day"] is True and snap["wake"] == 360


# ------------------------------------------------------------ the fakes
class FakeLamp:
    """The bulb's control socket as lamp_link() hands it out: counts the
    connections and answers like the color5 (look changes ignored while off,
    adjust_bright as measured)."""

    def __init__(self, power="on", bright=60, mode=2, ct=4000):
        self.st = {"power": power, "bright": bright, "color_mode": mode, "ct": ct,
                   "rgb": 255, "hue": 0, "sat": 100, "flowing": 0}
        self.connections = 0
        self.sent = []
        self.fail = None
        self.good_ips = set()                          # answer here even while failing
        self.ips = []
        self.raise_on = None                           # a method that fails mid-exchange

    @contextlib.contextmanager
    def link(self, ip):
        if self.fail and ip not in self.good_ips:
            raise self.fail
        self.connections += 1
        self.ips.append(ip)
        yield self.call

    def call(self, cmds):
        out = []
        for method, params in cmds:
            if method == self.raise_on:
                raise TimeoutError(f"{method}: no answer")
            self.sent.append((method, params))
            st = self.st
            if method == "get_prop":
                out.append([str(st[p]) for p in params])
                continue
            if method == "set_power":
                st["power"] = params[0]
            elif method == "toggle":
                st["power"] = "off" if st["power"] == "on" else "on"
            elif method == "adjust_bright":
                if st["power"] != "on":
                    st.update(power="on", bright=params[0])
                else:
                    st["bright"] = max(1, min(100, st["bright"] + params[0]))
            elif st["power"] == "on" and method == "set_ct_abx":
                st.update(ct=params[0], color_mode=2)
            elif st["power"] == "on" and method == "set_bright":
                st["bright"] = params[0]
            out.append(["ok"])
        return out

    def methods(self):
        return [m for m, _ in self.sent]


class Clock:
    def __init__(self, h, m=0):
        self.t = time.mktime((2026, 11, 14, h, m, 0, 0, 0, -1))

    def __call__(self):
        return self.t

    def advance(self, minutes):
        self.t += minutes * 60


def lamp_mode(clock=None, fake=None, **saved):
    saved.setdefault("ip", "10.0.0.9")
    st = dash.State(saved, store=None)
    fake = fake or FakeLamp()
    idle = threading.Event()
    idle.set()
    lm = dash.LampMode(st, idle=idle, link=fake.link, clock=clock or Clock(13),
                       find=lambda did, ip: None)
    return st, lm, fake


# ------------------------------------------------------ Follow the day
def test_off_costs_nothing():
    st, lm, fake = lamp_mode()
    assert lm.tick() == "idle" and fake.connections == 0


def test_first_tick_sends_and_then_only_a_change_connects():
    clock = Clock(13)
    st, lm, fake = lamp_mode(clock, follow_day=True)
    assert lm.tick() == "sent"
    assert fake.st["ct"] == 5000 and ("set_ct_abx", [5000, "smooth", dash.DAY_FADE_MS]) in fake.sent
    clock.advance(5)
    assert lm.tick() == "same" and fake.connections == 1      # midday: nothing moved
    clock.t = Clock(21)()                                     # mid-evening
    assert lm.tick() == "sent" and fake.connections == 2
    assert 2700 < fake.st["ct"] < 5000
    assert st.snapshot()["lamp"]["ct"] == fake.st["ct"]


def test_an_off_lamp_is_left_off_and_corrected_once_it_is_on():
    st, lm, fake = lamp_mode(follow_day=True, fake=FakeLamp(power="off", ct=2700))
    assert lm.tick() == "off" and "set_ct_abx" not in fake.methods()
    assert lm.tick() == "off"                                  # keeps looking while off
    fake.st["power"] = "on"                                    # the wall switch
    assert lm.tick() == "sent" and fake.st["ct"] == 5000
    assert lm.tick() == "same"


def test_a_temperature_it_did_not_send_holds_it_until_morning():
    clock = Clock(13)
    st, lm, fake = lamp_mode(clock, follow_day=True)
    assert lm.tick() == "sent"
    fake.st["ct"] = 3000                                       # changed from another app
    clock.t = Clock(21)()
    assert lm.tick() == "hand"
    assert fake.st["ct"] == 3000                               # not fought
    held = st.snapshot()["day_held"]
    assert held == dash.next_resume(clock())
    clock.t = Clock(23)()
    before = fake.connections
    assert lm.tick() == "held" and fake.connections == before  # no connection while held
    clock.t = held + 60                                        # 04:01
    assert lm.tick() == "sent" and st.snapshot()["day_held"] is None
    assert fake.st["ct"] == 2700


def test_a_colour_set_by_hand_also_wins():
    clock = Clock(13)
    st, lm, fake = lamp_mode(clock, follow_day=True)
    lm.tick()
    fake.st["color_mode"] = 1
    clock.t = Clock(21)()
    assert lm.tick() == "hand"


def test_it_never_touches_a_running_session():
    st, lm, fake = lamp_mode(follow_day=True)
    st.set(running=True)
    assert lm.tick() == "idle" and fake.connections == 0


def test_temperature_from_the_desk_holds_and_ticking_again_resumes():
    clock = Clock(13)
    st, lm, fake = lamp_mode(clock, follow_day=True)
    lm.tick()
    lm.act(ct=3100)
    assert st.snapshot()["day_held"] == dash.next_resume(clock())
    assert lm.tick() == "held" and fake.st["ct"] == 3100
    code, _ = post(st, {"follow_day": True}, lm)               # the switch ticked again
    assert code == 200 and st.snapshot()["day_held"] is None
    assert lm.tick() == "sent" and fake.st["ct"] == 5000


def test_brightness_from_the_desk_does_not_hold():
    st, lm, fake = lamp_mode(follow_day=True)
    lm.tick()
    lm.act(bright=20)
    assert st.snapshot()["day_held"] is None and fake.st["bright"] == 20


def test_turning_on_while_following_brings_the_curves_white():
    st, lm, fake = lamp_mode(Clock(22, 30), follow_day=True, fake=FakeLamp(power="off", ct=5000))
    lamp = lm.act("toggle")
    assert lamp["power"] == "on" and lamp["ct"] == 2700
    assert fake.connections == 1                               # read, on, white: one connection
    assert fake.methods() == ["get_prop", "set_power", "set_ct_abx", "get_prop"]
    assert lm.act("toggle")["power"] == "off"


def test_toggle_without_following_is_the_bulbs_own_toggle():
    st, lm, fake = lamp_mode()
    lm.act("toggle")
    assert fake.methods() == ["toggle", "get_prop"]


def test_the_thread_sleeps_while_off_and_ticks_at_once_when_switched_on():
    st, lm, fake = lamp_mode()
    lm.start()
    time.sleep(0.1)
    assert fake.connections == 0
    code, _ = post(st, {"follow_day": True}, lm)
    deadline = time.monotonic() + 2
    while fake.connections == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert code == 200 and fake.connections == 1


# --------------------------------------------------------- the endpoint
def call(state, lm, method, path, body=None):
    h = dash.Handler.__new__(dash.Handler)
    h.state, h.lamp = state, lm
    raw = b"" if body is None else json.dumps(body).encode()
    h.headers = {"Content-Length": str(len(raw)), "Host": "localhost:8765"}
    h.rfile, h.wfile = io.BytesIO(raw), io.BytesIO()
    h.path, h.command, h.request_version = path, method, "HTTP/1.1"
    h.requestline, h.client_address, h.close_connection = "", ("127.0.0.1", 0), False
    (h.do_GET if method == "GET" else h.do_POST)()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(payload or b"null")


def post(state, patch, lm, path="/api/control"):
    code, body = call(state, lm, "POST", path, patch)
    return code, body


def test_endpoint_reads_and_changes_the_lamp():
    st, lm, fake = lamp_mode()
    code, body = call(st, lm, "GET", "/api/lamp")
    assert code == 200 and body["lamp"]["ct"] == 4000 and fake.methods() == ["get_prop"]
    code, body = call(st, lm, "POST", "/api/lamp", {"ct": 3300, "bright": 45})
    assert code == 200 and body["lamp"]["ct"] == 3300 and body["lamp"]["bright"] == 45
    assert fake.connections == 2


def test_endpoint_refuses_nonsense_and_explains_a_missing_bulb():
    st, lm, fake = lamp_mode()
    assert call(st, lm, "POST", "/api/lamp", {"op": "dance"})[0] == 400
    assert call(st, lm, "POST", "/api/lamp", {"ct": "warm"})[0] == 400
    fake.fail = TimeoutError("timed out")
    code, body = call(st, lm, "POST", "/api/lamp", {"op": "up"})
    assert code == 502 and body["error"] == "The bulb did not answer at 10.0.0.9."
    assert st.snapshot()["lamp_error"] == body["error"]


def test_while_a_session_runs_only_toggle_is_taken_and_it_ends_dark():
    st, lm, fake = lamp_mode()
    st.set(running=True)
    code, body = call(st, lm, "GET", "/api/lamp")
    assert code == 409 and fake.connections == 0
    assert call(st, lm, "POST", "/api/lamp", {"op": "up"})[0] == 409
    code, body = call(st, lm, "POST", "/api/lamp", {"op": "toggle"})
    assert code == 200 and st.snapshot()["running"] is False
    assert body["lamp"]["power"] == "off" and fake.methods()[0] == "set_power"


def test_a_new_address_forgets_the_old_reading():
    st, lm, fake = lamp_mode(follow_day=True)
    lm.tick()
    post(st, {"ip": "10.0.0.20"}, lm)
    snap = st.snapshot()
    assert snap["lamp"] is None and lm.sent is None


def test_the_page_cannot_set_the_hold():
    st, lm, fake = lamp_mode()
    post(st, {"day_held": 2e9}, lm)
    assert st.snapshot()["day_held"] is None


# ------------------------------------------------------------- the CLI
class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_cli_goes_through_a_running_desk(capsys):
    seen = []

    def urlopen(req, timeout):
        seen.append((req.full_url, json.loads(req.data)))
        return Resp(json.dumps({"ok": True, "lamp": {"power": "on", "color_mode": 2,
                                                     "ct": 2700, "bright": 70}}).encode())

    assert lamp_main(["up"], urlopen=urlopen, send=None) == 0
    assert seen == [("http://127.0.0.1:8765/api/lamp", {"op": "up"})]
    assert capsys.readouterr().out.strip() == "on 2700 K 70%"


def test_cli_talks_to_the_bulb_when_no_desk_runs(tmp_path, capsys):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"version": 1, "settings": {"ip": "10.0.0.9"}}))
    sent = []

    def refused(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    def send(ip, cmds):
        sent.append((ip, cmds))
        return {"power": "off"}

    assert lamp_main(["toggle", "--settings", str(path)], urlopen=refused, send=send) == 0
    assert sent == [("10.0.0.9", [("toggle", [])])]
    assert capsys.readouterr().out.strip() == "off"


def test_cli_reports_a_desk_refusal_and_a_missing_address(tmp_path, capsys):
    def busy(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 409, "busy", {},
                                     io.BytesIO(b'{"error": "a session is running"}'))

    assert lamp_main(["up"], urlopen=busy) == 1
    assert "a session is running" in capsys.readouterr().err
    assert lamp_main(["up", "--desk", "", "--settings", str(tmp_path / "none.json")]) == 1
    assert "no bulb address" in capsys.readouterr().err


def test_bulbd_lamp_is_routed_before_the_session_flags(monkeypatch):
    got = []
    monkeypatch.setattr(music_bulb, "lamp_main", lambda argv: got.append(argv) or 0)
    monkeypatch.setattr("sys.argv", ["bulbd", "lamp", "down", "--ip", "10.0.0.9"])
    with pytest.raises(SystemExit) as exc:
        music_bulb.main()
    assert exc.value.code == 0 and got == [["down", "--ip", "10.0.0.9"]]


# ------------------------------------------------- review round: races
def in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except Exception as exc:
            out["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_nothing_goes_out_while_a_stopping_session_restores():
    st, lm, fake = lamp_mode(follow_day=True)
    lm.idle.clear()                                   # Stop pressed, restore under way
    for n, fn in enumerate((lambda: lm.act("up"), lm.tick, lm.read)):
        lm.sent = None                                # so the tick has something to send
        t, out = in_thread(fn)
        time.sleep(0.1)
        assert t.is_alive() and fake.connections == n
        lm.idle.set()
        t.join(2)
        assert "error" not in out and fake.connections == n + 1
        lm.idle.clear()


def test_a_command_waiting_on_the_lock_is_refused_if_a_session_started():
    st, lm, fake = lamp_mode()
    lm.lock.acquire()                                 # the engine opening the bulb
    t, out = in_thread(lambda: lm.act("up"))
    time.sleep(0.1)
    st.set(running=True)
    lm.lock.release()
    t.join(2)
    assert isinstance(out.get("error"), dash.LampBusy) and fake.connections == 0
    t, out = in_thread(lm.tick)
    t.join(2)
    assert out["value"] == "idle"


def test_the_engine_holds_the_lamp_lock_while_it_opens_and_closes(monkeypatch):
    held = []

    class Link:
        ip, lost, saved = "10.0.0.9", False, {"power": "on"}

        def __init__(self, *a, **k):
            pass

        def open(self):
            held.append(("open", eng.lamp_lock.locked()))
            return True

        def close(self, restore=False, then=()):
            held.append(("close", eng.lamp_lock.locked()))

    monkeypatch.setattr(dash, "Link", Link)
    st = dash.State({"ip": "10.0.0.9"}, store=None)
    eng = dash.Engine(st)
    ended = []
    eng.on_session_end = lambda: ended.append(True)
    eng._audio_loop = lambda st_, link: st_.set(running=False)
    st.set(running=True)
    eng._session()
    assert held == [("open", True), ("close", True)] and ended == [True]
    assert not eng.lamp_lock.locked()


class SharedLampSock(ScriptedSock):
    """A control socket over one shared lamp dict, applying what it is sent."""

    def __init__(self, lamp):
        super().__init__()
        self.lamp = lamp

    def sendall(self, data):
        if not data.strip():
            self.nudges += 1
            return
        msg = json.loads(data.decode())
        method, params = msg["method"], msg["params"]
        self.sent.append((method, params))
        L = self.lamp
        if method == "get_prop":
            result = [str(L[p]) for p in params]
        else:
            result = ["ok"]
            if method == "set_power":
                L["power"] = params[0]
            elif method == "set_ct_abx":
                L.update(ct=params[0], color_mode=2)
            elif method == "set_rgb":
                L.update(rgb=params[0], color_mode=1)
            elif method == "set_bright":
                L["bright"] = params[0]
        self.inbox += (json.dumps({"id": msg["id"], "result": result}) + "\r\n").encode()


def test_a_session_after_lamp_use_saves_and_restores_what_the_controls_left(monkeypatch):
    import socket
    import yeelight

    lamp = {"power": "off", "bright": 80, "color_mode": 1, "rgb": 255, "ct": 6500,
            "hue": 0, "sat": 100}
    # the lamp controls: on, 3100 K, 35%, each on a plain connection
    plain = lambda addr, timeout: SharedLampSock(lamp)
    lamp_send("10.0.0.9", lamp_commands("on"), connect=plain)
    lamp_send("10.0.0.9", lamp_commands(ct=3100, bright=35), connect=plain)
    left = dict(lamp)
    assert (left["power"], left["ct"], left["bright"], left["color_mode"]) == ("on", 3100, 35, 2)

    class Bulb:
        def __init__(self, ip, **kw):
            self._ip = ip
            self._Bulb__socket = None

        @property
        def _socket(self):
            if self._Bulb__socket is None:
                self._Bulb__socket = SharedLampSock(lamp)
            return self._Bulb__socket

        def turn_on(self):
            lamp["power"] = "on"

        def start_music(self):
            pass

        def stop_music(self):
            self._Bulb__socket = SharedLampSock(lamp)   # the fresh control connection

    monkeypatch.setattr(yeelight, "Bulb", Bulb)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("extra slot"))
    bulb = music_bulb.open_bulb("10.0.0.9", remember=True)
    assert bulb.bulbd_saved == left
    lamp.update(color_mode=1, rgb=0xFF00FF, bright=100)  # the session's light
    assert music_bulb.release_bulb(bulb, restore=True)
    look = ("power", "color_mode", "ct", "bright")       # rgb is moot on a white lamp
    assert {k: lamp[k] for k in look} == {k: left[k] for k in look}


def test_a_reading_inside_its_own_fade_is_not_a_hand_change():
    clock = Clock(21)
    st, lm, fake = lamp_mode(clock, follow_day=True)
    assert lm.tick() == "sent"
    fake.st["ct"] = (fake.st["ct"] + 5000) // 2          # still gliding there
    post(st, {"day_k": 6000}, lm)                         # a quick second edit
    clock.advance(0.02)                                   # ~1 s later
    assert lm.tick() == "sent" and st.snapshot()["day_held"] is None
    fake.st["ct"] = 3000                                  # well after the fade: a hand
    clock.t = Clock(21, 20)()
    assert lm.tick() == "hand"


def test_a_failed_send_is_not_remembered_as_sent():
    st, lm, fake = lamp_mode(Clock(22, 30), follow_day=True, fake=FakeLamp(power="off", ct=5000))
    fake.raise_on = "set_ct_abx"
    with pytest.raises(TimeoutError):
        lm.act("on")
    assert lm.sent is None
    fake.raise_on = None
    with pytest.raises(TimeoutError):                     # the tick fails the same way
        fake.raise_on = "set_ct_abx"
        lm.tick()
    assert lm.sent is None
    fake.raise_on = None
    assert lm.tick() == "sent" and fake.st["ct"] == 2700


def test_after_a_session_the_curve_is_reasserted():
    clock = Clock(22, 30)
    st, lm, fake = lamp_mode(clock, follow_day=True)
    assert lm.tick() == "sent"
    fake.st.update(color_mode=1)                          # --no-restore left the session's colour
    lm.session_ended()
    assert lm.wake.is_set()
    assert lm.tick() == "sent" and fake.st["color_mode"] == 2
    assert st.snapshot()["day_held"] is None


def test_a_temperature_sent_to_an_off_lamp_holds_nothing():
    st, lm, fake = lamp_mode(follow_day=True, fake=FakeLamp(power="off"))
    lm.act(ct=3000)
    assert st.snapshot()["day_held"] is None


def test_wake_and_dusk_never_cross():
    snap = {"monitors": [], "wake": 420, "dusk": 1200}
    assert dash.clean_patch({"dusk": 300}, snap) == {"wake": 420, "dusk": 540}
    assert dash.clean_patch({"wake": 1300}, snap) == {"wake": 1080, "dusk": 1200}
    assert dash.clean_patch({"wake": 1400, "dusk": 10}, snap) == {"wake": 1305, "dusk": 1425}
    assert dash.clean_patch({"dusk": 30}, dict(snap, wake=0)) == {"wake": 0, "dusk": 120}
    for patch in ({"dusk": 300}, {"wake": 1300}, {"wake": 1400, "dusk": 10}):
        out = dict(snap, **dash.clean_patch(patch, snap))
        assert out["dusk"] - out["wake"] >= dash.MIN_DAY
        assert dash.day_kelvin((out["wake"] + out["dusk"]) // 2 + 30,
                               wake=out["wake"], dusk=out["dusk"]) > 2700


def test_a_moved_bulb_is_searched_for_once_and_followed():
    st, lm, fake = lamp_mode(did=0x1E5A1F80)
    searches = []
    lm.find = lambda did, ip: searches.append((did, ip)) or "10.0.0.33"
    fake.fail, fake.good_ips = TimeoutError("timed out"), {"10.0.0.33"}
    lamp = lm.act("up")
    assert searches == [(0x1E5A1F80, "10.0.0.9")] and fake.ips == ["10.0.0.33"]
    assert st.snapshot()["ip"] == "10.0.0.33" and lamp["power"] == "on"
    assert st.snapshot()["lamp"] == lamp


def test_an_unreachable_bulb_is_searched_for_only_once():
    st, lm, fake = lamp_mode(follow_day=True, did=0x1E5A1F80)
    searches = []
    lm.find = lambda did, ip: searches.append(ip)
    fake.fail = TimeoutError("timed out")
    for _ in range(3):
        with pytest.raises(TimeoutError):
            lm.tick()
    assert searches == ["10.0.0.9"]
    fake.fail = None
    assert lm.tick() == "sent"
    fake.fail = TimeoutError("timed out")                 # a new unreachable spell
    with pytest.raises(TimeoutError):
        lm.act("up")
    assert searches == ["10.0.0.9", "10.0.0.9"]


def test_no_search_without_a_known_id():
    st, lm, fake = lamp_mode()
    lm.find = lambda did, ip: pytest.fail("searched")
    fake.fail = TimeoutError("timed out")
    with pytest.raises(TimeoutError):
        lm.act("up")


def test_cli_with_an_address_still_goes_through_a_running_desk():
    seen = []

    def urlopen(req, timeout):
        seen.append(req.full_url)
        return Resp(b'{"ok": true, "lamp": {"power": "off"}}')

    assert lamp_main(["toggle", "--ip", "10.0.0.9"], urlopen=urlopen,
                     send=lambda *a: pytest.fail("sent beside the desk")) == 0
    assert seen == ["http://127.0.0.1:8765/api/lamp"]


def test_cli_never_sends_again_after_a_desk_timeout(capsys):
    import socket

    def slow(req, timeout):
        raise urllib.error.URLError(socket.timeout("timed out"))

    def hung(req, timeout):
        raise TimeoutError("timed out")

    for urlopen in (slow, hung):
        assert lamp_main(["toggle", "--ip", "10.0.0.9"], urlopen=urlopen,
                         send=lambda *a: pytest.fail("toggled twice")) == 1
        assert "did not answer" in capsys.readouterr().err


# ------------------------------------- beside the other things on the lamp
def test_a_flow_on_the_bulb_is_left_running_and_the_curve_comes_back_after():
    """An On stop look, a sleep fade or a pulse is a flow: a set would end it."""
    clock = Clock(13)
    st, lm, fake = lamp_mode(clock, follow_day=True)
    fake.st["flowing"] = 1
    assert lm.tick() == "flow" and "set_ct_abx" not in fake.methods()
    assert st.snapshot()["day_held"] is None                  # not a hand change
    clock.advance(5)
    assert lm.tick() == "flow" and "set_ct_abx" not in fake.methods()
    fake.st["flowing"] = 0
    fake.st.update(ct=6500)                                   # what the flow left
    assert lm.tick() == "sent" and fake.st["ct"] == 5000
    assert st.snapshot()["day_held"] is None


def test_the_curve_stands_back_while_focus_or_the_sunrise_has_the_lamp():
    clock = Clock(13)
    st, lm, fake = lamp_mode(clock, follow_day=True)
    assert lm.tick() == "sent"
    busy = [True]
    lm.yield_to = lambda: busy[0]
    fake.st.update(ct=6000)                                   # the Focus white
    clock.t = Clock(21)()
    assert lm.tick() == "idle" and fake.connections == 1      # no connection at all
    busy[0] = False
    assert lm.tick() == "sent"                                # and no "hand" hold
    assert st.snapshot()["day_held"] is None


def test_a_request_from_another_website_is_refused():
    st, lm, fake = lamp_mode()

    def ask(method, path, body=None, **headers):
        h = dash.Handler.__new__(dash.Handler)
        h.state, h.lamp = st, lm
        raw = b"" if body is None else json.dumps(body).encode()
        h.headers = {"Content-Length": str(len(raw)), "Host": "localhost:8765", **headers}
        h.rfile, h.wfile = io.BytesIO(raw), io.BytesIO()
        h.path, h.command, h.request_version = path, method, "HTTP/1.1"
        h.requestline, h.client_address, h.close_connection = "", ("127.0.0.1", 0), False
        (h.do_GET if method == "GET" else h.do_POST)()
        return int(h.wfile.getvalue().split()[1])

    toggle = {"op": "toggle"}
    # the page itself, and a script with no browser headers
    assert ask("POST", "/api/lamp", toggle, Origin="http://localhost:8765") == 200
    assert ask("POST", "/api/lamp", toggle, **{"Sec-Fetch-Site": "same-origin"}) == 200
    assert ask("POST", "/api/lamp", toggle) == 200
    sent = len(fake.sent)
    # another site, another port on this machine, a sandboxed frame, an <img>
    assert ask("POST", "/api/lamp", toggle, Origin="https://example.com") == 403
    assert ask("POST", "/api/lamp", toggle, Origin="http://localhost:3000") == 403
    assert ask("POST", "/api/lamp", toggle, Origin="null") == 403
    assert ask("POST", "/api/control", {"running": False},
               **{"Sec-Fetch-Site": "cross-site"}) == 403
    assert ask("GET", "/api/lamp", **{"Sec-Fetch-Site": "cross-site"}) == 403
    assert len(fake.sent) == sent                             # nothing reached the lamp
