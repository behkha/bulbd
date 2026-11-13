"""Auto mode: what it picks, how it holds through gaps, what watching the
unused input costs, and how the desk shows and switches it."""
import threading
import time

import numpy as np
import pytest

import music_bulb_dash as dash
from music_bulb import AutoPicker, MotionMeter
from signals import solid

LOUD, QUIET = 0.05, 0.0


def drive(picker, secs, sound, moving, t0, dt=0.1, probe=2.0):
    """Feed `secs` of the same situation. -> (time reached, [(t, mode)])."""
    t, picks, next_probe = t0, [], t0
    while t < t0 + secs - 1e-9:
        picker.feed_sound(LOUD if sound else QUIET, t)
        if t >= next_probe:
            picker.feed_motion(moving, t)
            next_probe = t + probe
        picks.append((t, picker.pick(t)))
        t += dt
    return t, picks


def modes(picks):
    out = []
    for _, m in picks:
        if not out or out[-1] != m:
            out.append(m)
    return out


# ---------------------------------------------------------------- the table
@pytest.mark.parametrize("sound, moving, want", [
    (True, True, "fusion"), (True, False, "music"), (False, True, "screen"), (False, False, "screen")])
def test_each_situation_has_its_mode(sound, moving, want):
    p = AutoPicker(now=0.0)
    _, picks = drive(p, 20, sound, moving, 0.0)
    assert picks[-1][1] == want


def test_without_screen_recording_it_is_only_ever_music():
    p = AutoPicker(screen_ok=False, now=0.0)
    seen = set()
    for sound, moving in [(True, True), (False, True), (False, False), (True, False)]:
        _, picks = drive(p, 20, sound, moving, p.since + 20)
        seen |= {m for _, m in picks}
    assert seen == {"music"}


def test_the_first_pick_lands_after_a_few_seconds():
    p = AutoPicker(now=0.0)
    assert p.mode == "fusion"                                 # starts hearing and seeing
    _, picks = drive(p, 6, True, False, 0.0)
    switch = next(t for t, m in picks if m == "music")
    assert AutoPicker.SETTLE <= switch < 4.0


# --------------------------------------------------------------- hysteresis
def test_a_gap_between_tracks_does_not_leave_music():
    p = AutoPicker(now=0.0)
    t, a = drive(p, 30, True, False, 0.0)
    t, b = drive(p, 4, False, False, t)                       # the gap
    t, c = drive(p, 30, True, False, t)
    assert modes(a + b + c)[1:] == ["music"]


def test_long_silence_hands_over_to_the_picture():
    p = AutoPicker(now=0.0)
    t, a = drive(p, 30, True, False, 0.0)
    t, b = drive(p, 20, False, False, t)
    assert modes(a + b)[-2:] == ["music", "screen"]
    left = next(tt for tt, m in b if m == "screen") - b[0][0]
    assert AutoPicker.SOUND_OFF <= left < AutoPicker.SOUND_OFF + 0.5


def test_a_paused_film_keeps_its_mode_then_lets_go():
    p = AutoPicker(now=0.0)
    t, a = drive(p, 30, True, True, 0.0)                      # a film with sound
    t, b = drive(p, 10, True, False, t)                       # paused, the room still talking
    assert {m for _, m in b} == {"fusion"}
    t, c = drive(p, 20, True, False, t)
    assert c[-1][1] == "music"


def test_a_notification_ping_is_not_sound():
    p = AutoPicker(now=0.0)
    t, a = drive(p, 30, False, True, 0.0)
    for _ in range(5):                                        # 1 s pings, 6 s apart
        t, b = drive(p, 1, True, True, t)
        t, c = drive(p, 5, False, True, t)
        a += b + c
    assert modes(a)[1:] == ["screen"]


def test_one_moving_probe_is_not_a_film():
    p = AutoPicker(now=0.0)
    t, a = drive(p, 20, True, False, 0.0)
    p.feed_motion(True, t)                                    # a window switch
    t, b = drive(p, 20, True, False, t + 0.1)
    assert modes(a + b)[1:] == ["music"]


def test_a_pick_holds_at_least_the_hold_time():
    p = AutoPicker(now=0.0)
    t, a = drive(p, 15, True, False, 0.0)
    t, b = drive(p, 3, False, True, t)                        # a short burst of the other
    t, c = drive(p, 3, True, False, t)
    changes = [tt for (tt, m), (_, prev) in zip((a + b + c)[1:], a + b + c) if m != prev]
    assert all(t2 - t1 >= AutoPicker.HOLD - 1e-6 for t1, t2 in zip(changes, changes[1:]))


# ------------------------------------------------------------------ motion
def test_motion_meter_compares_one_frame_per_probe():
    m = MotionMeter()
    grey, bright = solid((.4, .4, .4)), solid((.8, .8, .8))
    assert m.feed(grey, 0.0) is None                          # nothing to compare yet
    assert m.feed(bright, 0.5) is None                        # between probes
    assert m.feed(bright, 2.0) is True
    assert m.feed(bright, 4.0) is False


def test_a_cursor_sized_change_is_not_motion():
    m = MotionMeter()
    a = solid((.3, .3, .3))
    b = a.copy(); b[10:14, 10:12] = 1.0                       # a blinking caret
    m.feed(a, 0.0)
    assert m.feed(b, 2.0) is False


def test_a_video_in_a_window_is_motion():
    rng = np.random.default_rng(0)
    m = MotionMeter()
    a = solid((.3, .3, .3))
    b = a.copy(); b[20:60, 40:120] = rng.random((40, 80, 3))   # 12% of the screen
    m.feed(a, 0.0)
    assert m.feed(b, 2.0) is True


# ------------------------------------------------------------------ probes
class FakeGrab:
    def __init__(self):
        self.n, self.closed = 0, False

    def frame(self):
        self.n += 1
        return solid((self.n % 2, 0, 0))                      # every grab differs

    def close(self):
        self.closed = True


def test_the_screen_probe_in_music_mode_is_cheap(monkeypatch):
    monkeypatch.setattr(MotionMeter, "PROBE", 0.05)
    grab, picker = FakeGrab(), AutoPicker(now=time.monotonic())
    probe = dash.MotionProbe(grab, picker)
    probe.start()
    time.sleep(0.5)
    probe.close()
    assert 5 <= grab.n <= 12 and grab.closed                 # one grab per PROBE
    assert picker.moving


def test_the_sound_probe_in_screen_mode_reads_only_the_level():
    class Rec:
        def __init__(self):
            self.blocks = 0

        def recorder(self, **kw):
            rec = self

            class Ctx:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def record(self, numframes):
                    rec.blocks += 1
                    time.sleep(0.01)
                    return np.full((numframes, 2), 0.1, dtype=np.float32)

            return Ctx()

    rec = Rec()
    picker = AutoPicker(now=time.monotonic() - 10)
    probe = dash.SoundProbe(rec, picker)
    probe.start()
    t0 = time.monotonic()
    while picker._heard is None and time.monotonic() - t0 < 2:
        time.sleep(0.01)
    probe.close()
    assert picker._heard is not None and not probe.is_alive()


# -------------------------------------------------------------------- desk
def test_auto_is_a_saved_switch():
    assert "auto" in dash.SAVED_KEYS
    assert dash.clean_patch({"auto": True}, {"monitors": []}) == {"auto": True}
    assert dash.clean_patch({"auto": "yes"}, {"monitors": []}) == {}
    assert dash.clean_patch({"mode": "auto"}, {"monitors": []}) == {}


def test_turning_auto_off_keeps_the_mode_it_was_running():
    from test_desk import post
    st = dash.State({"auto": True, "mode": "music"})
    st.set(auto_mode="fusion")
    post(st, {"auto": False})
    snap = st.snapshot()
    assert snap["auto"] is False and snap["mode"] == "fusion"


def test_picking_a_mode_turns_auto_off():
    from test_desk import post
    st = dash.State({"auto": True})
    post(st, {"mode": "screen", "auto": False})
    assert st.snapshot()["mode"] == "screen" and not st.snapshot()["auto"]


def test_switching_auto_ends_the_running_loop():
    eng = dash.Engine(dash.State({"auto": False, "mode": "music"}))
    cfg = eng.state.snapshot()
    assert eng._mode(cfg) == "music"
    cfg["auto"] = True
    assert eng._mode(cfg) is None                             # rebuild with a picker
    eng._watch(cfg, start="music")
    assert eng._mode(cfg) == "music" and eng.picker.mode == "music"


def test_the_session_runs_what_auto_picks(monkeypatch):
    from test_desk import FakeLink
    FakeLink.made, FakeLink.plan = [], {}
    monkeypatch.setattr(dash, "Link", FakeLink)
    monkeypatch.setattr(dash, "screen_permission", lambda: True)
    st = dash.State({"ip": "192.168.1.2", "auto": True, "mode": "music"})
    eng = dash.Engine(st)
    ran = []

    def loop(name):
        def run(st_, link):
            ran.append(name)
            if len(ran) == 1:
                eng.picker.mode = "screen"                    # the picker changed its mind
            else:
                st_.set(running=False)
        return run

    eng._audio_loop, eng._screen_loop, eng._fusion_loop = loop("music"), loop("screen"), loop("fusion")
    st.set(running=True)
    t = threading.Thread(target=eng._session, daemon=True)
    t.start()
    t.join(3)
    assert ran == ["fusion", "screen"]                        # starts hearing and seeing
    assert st.snapshot()["auto_mode"] == "screen"
