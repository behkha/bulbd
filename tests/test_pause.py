"""Pause light: a picture held still raises a soft warm white, whatever its
brightness; the first real motion hands back to the picture; a dark scene
that moves never counts; a moved control does not cut it short."""
import threading
import time

import numpy as np
import pytest

import music_bulb_dash as dash
from music_bulb import SWITCH_MS, CapturePacer, PauseLight, White, ramp_ms
from signals import H, W, solid


def run(frames, delay=5.0, fps=10, quiet=True):
    """Feed frames through CapturePacer and PauseLight the way the screen
    loop does. -> [(t, on, entered, left)]"""
    pacer, pause, out = CapturePacer(), PauseLight(delay), []
    for i, frame in enumerate(frames):
        now = i / fps
        pacer.period(frame, fps, now)
        on = pause.feed(pacer.moved_at, now, quiet)
        out.append((now, on, pause.entered, pause.left))
    return out


def still(rgb, secs, fps=10):
    return [solid(rgb)] * int(secs * fps)


def dark_motion(secs, fps=10, seed=0):
    """A dark scene that moves: dim noise that changes every frame."""
    rng = np.random.default_rng(seed)
    return [np.clip(0.06 + rng.normal(0, 0.05, (H, W, 3)), 0, 1).astype(np.float32)
            for _ in range(int(secs * fps))]


def first_on(rows):
    return next((t for t, on, *_ in rows if on), None)


# ------------------------------------------------------------- stillness, not darkness
@pytest.mark.parametrize("label, rgb", [("black paused film", (0, 0, 0)),
                                        ("dark paused film", (.05, .03, .02)),
                                        ("bright slide", (1, 1, 1)),
                                        ("colour slide", (.1, .3, .9))])
def test_a_still_picture_raises_it_after_the_delay(label, rgb):
    rows = run(still(rgb, 12))
    assert first_on(rows) == pytest.approx(5.0, abs=0.11)
    assert sum(entered for *_, entered, _ in rows) == 1


def test_a_dark_scene_that_moves_never_does():
    assert first_on(run(dark_motion(20))) is None


def test_off_by_default():
    assert first_on(run(still((0, 0, 0), 20), delay=0)) is None


def test_the_first_real_motion_hands_back():
    frames = still((.2, .2, .2), 8) + [solid((.8, .2, .1))] + still((.8, .2, .1), 2)
    rows = run(frames)
    on = [t for t, o, *_ in rows if o]
    assert on and on[-1] == pytest.approx(7.9)          # off on the frame that moved
    assert [t for t, _, _, left in rows if left] == [pytest.approx(8.0)]


def test_a_cursor_blink_is_not_motion():
    blink = solid((.1, .1, .1))
    blink[60:63, 100:101] = 1.0                         # 3 px of 26000
    frames = [solid((.1, .1, .1)) if i % 5 else blink for i in range(120)]
    assert first_on(run(frames)) == pytest.approx(5.0, abs=0.11)


def test_a_moved_control_does_not_cut_it_short():
    pacer, pause = CapturePacer(), PauseLight(2.0)
    frame = solid((0, 0, 0))
    for i in range(40):
        now = i / 10
        if i == 30:
            pacer.reset()                               # the desk's `changed`
        pacer.period(frame, 10, now)
        pause.feed(pacer.moved_at, now)
        if i >= 20:
            assert pause.on, f"off at {now}"


def test_in_fusion_the_sound_must_be_gone_too():
    assert first_on(run(still((0, 0, 0), 12), quiet=False)) is None


# ------------------------------------------------------------- the light
def test_it_is_a_soft_warm_white_whatever_the_picture():
    f = PauseLight(5).frame()
    assert isinstance(f, White) and f.ct == 2700 and 8 <= f.bright <= 16


def test_it_fades_in_slowly_and_out_quickly():
    pause = PauseLight(1)
    pause.feed(0.0, 2.0)
    assert pause.entered and pause.ramp(pause.frame(), (20, 60, 40), 0.1) == PauseLight.RISE_MS
    pause.feed(2.5, 2.6)
    assert pause.left and pause.ramp((20, 60, 40), pause.frame(), 0.1) >= SWITCH_MS
    pause.feed(2.5, 2.7)
    assert pause.ramp((20, 60, 41), (20, 60, 40), 0.1) == ramp_ms((20, 60, 41), (20, 60, 40), 0.1)


# ------------------------------------------------------------- the desk
def test_pause_is_a_saved_setting_and_paused_is_not():
    snap = dash.State().snapshot()
    assert snap["pause"] == 0 and snap["paused"] is False
    assert dash.clean_patch({"pause": 500}, snap) == {"pause": 120}
    assert dash.clean_patch({"pause": -3}, snap) == {"pause": 0}
    assert dash.clean_patch({"pause": "8"}, snap) == {}
    assert "pause" in dash.SAVED_KEYS and "paused" not in dash.SAVED_KEYS
    # the viewer's habit, like Beat sync: applying Cinema must not switch it off
    assert "pause" not in dash.LOOK_KEYS


class StillCapture:
    def __init__(self, *a, **k):
        self.backend, self.pushes, self.box = "sck", True, {"width": 200, "height": 130}
        self.moving = False
        self.n = 0

    def frame(self):
        self.n += 1
        if self.moving:
            return solid((0.8, 0.1, 0.1) if self.n % 2 else (0.1, 0.1, 0.8))
        return solid((0.02, 0.02, 0.02))

    def pace(self, started, base, period, wake=None):
        time.sleep(0.01)
        return False

    def set_fps(self, fps):
        pass

    def close(self):
        pass


class Recorder:
    lost = False

    def __init__(self):
        self.sent = []

    def put(self, frame, sudden, duration=40):
        self.sent.append((time.monotonic(), frame, sudden, duration))


@pytest.fixture
def screen_desk(monkeypatch):
    grabs = []

    def make(*a, **k):
        grabs.append(StillCapture())
        return grabs[-1]
    monkeypatch.setattr(dash, "Capture", make)
    monkeypatch.setattr(dash, "screen_permission", lambda: True)
    st = dash.State({"ip": "192.168.1.2", "mode": "screen", "pause": 1})
    st.set(running=True)
    return st, dash.Engine(st, capture="sck"), grabs


def drive(st, eng, rec, secs):
    t = threading.Thread(target=eng._screen_loop, args=(st, rec), daemon=True)
    t.start()
    return t


def test_the_screen_loop_raises_it_on_a_still_picture_and_drops_it_on_motion(screen_desk):
    st, eng, grabs = screen_desk
    rec = Recorder()
    t = drive(st, eng, rec, 0)
    time.sleep(1.4)
    assert st.snapshot()["paused"] is True
    whites = [(f, d) for _, f, s, d in rec.sent if f == PauseLight(1).frame()]
    assert whites and whites[0][1] == PauseLight.RISE_MS
    snap = st.snapshot()
    assert snap["white"] and snap["ct"] == 2700 and snap["val"] == PauseLight.LEVEL
    grabs[0].moving = True
    time.sleep(0.3)
    st.set(running=False)
    t.join(2)
    after = [f for _, f, s, d in rec.sent[len(whites):] if not isinstance(f, White)]
    assert after                                        # back on the picture
    assert st.snapshot()["paused"] is False


def test_the_screen_loop_never_pauses_a_moving_picture(screen_desk):
    st, eng, grabs = screen_desk
    rec = Recorder()
    t = drive(st, eng, rec, 0)
    time.sleep(0.05)
    grabs[0].moving = True
    time.sleep(1.4)
    seen = st.snapshot()["paused"]
    st.set(running=False)
    t.join(2)
    assert seen is False
    assert PauseLight(1).frame() not in [f for _, f, *_ in rec.sent]
