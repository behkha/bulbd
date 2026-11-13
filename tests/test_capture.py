"""Screen capture backends: the ScreenCaptureKit stream (with a fake Kit, so
this runs anywhere without pyobjc), choosing and falling back to mss, the
loops' pacing, and that no stream outlives what opened it."""
import threading
import time

import numpy as np
import pytest

import music_bulb
import music_bulb_dash as dash
import sck_capture
from music_bulb import Capture, CapturePacer, ScreenGrabber, pace
from sck_capture import (COMPLETE, IDLE, StreamDied, StreamGrabber, bgra_to_frame,
                         find_display, frame_size)
from signals import solid

BUILTIN = {"left": 0, "top": 0, "width": 1800, "height": 1169}
EXTERNAL = {"left": -392, "top": -1080, "width": 2560, "height": 1080}


# ------------------------------------------------------------------ pixels
def padded(rgb, pad):
    """An (h, w, 3) uint8 picture as a BGRA buffer with `pad` junk bytes per row."""
    h, w, _ = rgb.shape
    bgra = np.concatenate([rgb[..., ::-1], np.full((h, w, 1), 255, np.uint8)], axis=2)
    rows = np.concatenate([bgra.reshape(h, w * 4),
                           np.full((h, pad), 0xAB, np.uint8)], axis=1)
    return rows.tobytes(), w * 4 + pad


def test_a_padded_bgra_buffer_becomes_the_frame_the_engines_expect():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    buf, bpr = padded(rgb, 36)                          # 28 bytes of pixels, 64 a row
    f = bgra_to_frame(buf, 7, 5, bpr)
    assert f.shape == (5, 7, 3) and f.dtype == np.float32
    assert np.allclose(f, rgb / 255.0)                  # RGB order, padding dropped


def test_the_frame_is_a_copy_the_buffer_can_be_unlocked_after():
    rgb = np.zeros((2, 2, 3), np.uint8)
    raw = bytearray(padded(rgb, 0)[0])
    f = bgra_to_frame(raw, 2, 2, 8)
    raw[:] = b"\xff" * len(raw)
    assert f.max() == 0.0


def test_a_buffer_without_padding_on_its_last_row_still_reads():
    rgb = np.full((3, 4, 3), 200, np.uint8)
    buf, bpr = padded(rgb, 16)
    f = bgra_to_frame(buf[:-16], 4, 3, bpr)
    assert np.allclose(f, 200 / 255.0)


@pytest.mark.parametrize("w, h, bpr, size", [(4, 3, 15, 48), (4, 3, 16, 40), (0, 3, 16, 48)])
def test_a_buffer_that_does_not_fit_its_geometry_is_refused(w, h, bpr, size):
    with pytest.raises(ValueError):
        bgra_to_frame(bytes(size), w, h, bpr)


@pytest.mark.parametrize("box_w, box_h", [(1800, 1169), (2560, 1080), (300, 200), (100, 64)])
def test_the_stream_is_asked_for_the_shape_mss_would_give(box_w, box_h):
    stride = max(1, int(box_w // 192))
    want = np.zeros((box_h, box_w))[::stride, ::stride].shape
    assert frame_size(box_w, box_h, 192) == (want[1], want[0])


# ----------------------------------------------------------------- display
def test_the_display_is_found_by_its_frame_or_by_where_the_box_is():
    ds = [("ext", (-392, -1080, 2560, 1080)), ("lcd", (0, 0, 1800, 1169))]
    assert find_display(ds, BUILTIN)[0] == "lcd"
    assert find_display(ds, EXTERNAL)[0] == "ext"
    assert find_display(ds, {"left": 100, "top": -500, "width": 50, "height": 50})[0] == "ext"
    with pytest.raises(RuntimeError):
        find_display(ds, {"left": 5000, "top": 0, "width": 10, "height": 10})


# --------------------------------------------------------------- fake kit
class FakeKit:
    """Stands in for sck_capture.Kit: records the calls, and lets a test push
    samples and stop the stream as ScreenCaptureKit's thread would."""

    def __init__(self, first=True, start_error=None):
        self.first, self.start_error = first, start_error
        self.started, self.stopped, self.configured = [], 0, []
        self.on_sample = self.on_stop = None

    def displays(self):
        return [("ext", (-392, -1080, 2560, 1080)), ("lcd", (0, 0, 1800, 1169))]

    def start(self, display, rect, size, fps, on_sample, on_stop):
        if self.start_error:
            raise self.start_error
        self.started.append((display, rect, size, fps))
        self.on_sample, self.on_stop = on_sample, on_stop
        self.size = size
        if self.first:
            self.push(0.5)
        return {"stream": display}

    def push(self, level=None, status=COMPLETE):
        """One sample: a picture of grey `level`, or an idle one."""
        self.on_sample(("sample", status, level), 0)

    def read(self, sample, convert):
        _, status, level = sample
        if status != COMPLETE:
            return status, None
        w, h = self.size
        buf, bpr = padded(np.full((h, w, 3), int(level * 255), np.uint8), 8)
        return status, convert(buf, w, h, bpr)

    def color_space(self, sample):
        return "Color LCD"

    def configure(self, handle, fps):
        self.configured.append(fps)

    def stop(self, handle):
        self.stopped += 1
        return True


def test_a_region_is_captured_in_the_display_s_own_points():
    kit = FakeKit()
    g = StreamGrabber(EXTERNAL, (-392 + 100, -1080 + 50, 640, 360), 192, 10, kit=kit)
    display, rect, size, fps = kit.started[0]
    assert display == "ext" and rect == (100, 50, 640, 360)
    assert size == frame_size(640, 360, 192) and fps == 10
    assert g.box == {"left": -292, "top": -1030, "width": 640, "height": 360}
    g.close()


def test_a_whole_display_has_no_source_rect():
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    assert kit.started[0][:3] == ("lcd", None, (200, 130))
    f = g.frame()
    assert f.shape == (130, 200, 3) and np.allclose(f, 127 / 255.0)
    g.close()


def test_idle_samples_count_but_change_nothing():
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    first = g.frame()
    seq = g.seq
    for _ in range(5):
        kit.push(status=IDLE)
    assert g.idle == 5 and g.seq == seq and g.frame() is first
    kit.push(0.9)
    assert g.delivered == 2 and np.allclose(g.frame(), 229 / 255.0)
    g.close()


def test_a_stream_that_sends_no_picture_is_stopped_and_refused():
    kit = FakeKit(first=False)
    with pytest.raises(RuntimeError):
        StreamGrabber(BUILTIN, None, 192, 10, kit=kit, first_frame=0.05)
    assert kit.stopped == 1


def test_a_stream_that_stops_on_its_own_raises_on_the_next_frame():
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    kit.on_stop("display removed")
    with pytest.raises(StreamDied):
        g.frame()
    g.close()


def test_closing_stops_the_stream_once_however_often_it_is_called():
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    g.close()
    g.close()
    assert kit.stopped == 1


def test_streams_left_open_are_stopped_at_exit():
    kits = [FakeKit(), FakeKit()]
    grabs = [StreamGrabber(BUILTIN, None, 192, 10, kit=k) for k in kits]
    sck_capture.close_all()
    assert [k.stopped for k in kits] == [1, 1]
    assert not any(g in sck_capture._LIVE for g in grabs)


def test_a_new_frame_rate_reconfigures_the_stream_only_when_it_changes():
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    g.set_fps(10)
    g.set_fps(20)
    g.set_fps(20)
    assert kit.configured == [20]
    g.close()


# ------------------------------------------------------------------ pacing
def test_a_new_picture_ends_the_wait_early():
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    g.frame()
    wake = threading.Event()
    threading.Timer(0.05, kit.push, args=(0.2,)).start()
    t = time.monotonic()
    woke = g.pace(t, 0.01, 2.0, wake)
    assert not woke and time.monotonic() - t < 0.5
    g.close()


def test_a_moved_control_ends_the_wait_and_says_so():
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    g.frame()
    wake = threading.Event()
    threading.Timer(0.05, wake.set).start()
    t = time.monotonic()
    assert g.pace(t, 0.01, 2.0, wake) is True
    assert time.monotonic() - t < 0.5 and not wake.is_set()
    g.close()


def test_a_still_picture_waits_the_pacer_s_period_and_no_less_than_base():
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    g.frame()
    for _ in range(3):
        kit.push(status=IDLE)
    t = time.monotonic()
    assert g.pace(t, 0.02, 0.15, threading.Event()) is False
    assert 0.14 <= time.monotonic() - t < 0.4
    kit.push(0.3)                                       # a picture already waiting:
    t = time.monotonic()                                # ...only base is waited out
    g.pace(t, 0.05, 1.0, threading.Event())
    assert 0.045 <= time.monotonic() - t < 0.3
    g.close()


def test_mss_pacing_is_unchanged():
    wake = threading.Event()
    t = time.monotonic()
    assert pace(object(), t, 0.02, 0.1, wake) is False
    assert 0.09 <= time.monotonic() - t < 0.3
    threading.Timer(0.03, wake.set).start()
    t = time.monotonic()
    assert pace(object(), t, 0.01, 2.0, wake) is True
    assert time.monotonic() - t < 0.5 and not wake.is_set()


def test_on_a_stream_the_stillness_step_down_is_the_same():
    """A still picture reaches the 5 and 2 fps steps on cached frames: the
    pacer sees the same frame each tick, as it would from mss grabs."""
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    p, now, periods = CapturePacer(), 0.0, []
    for _ in range(200):
        periods.append(p.period(g.frame(), 10, now))
        now += periods[-1]
    assert periods[0] == 0.1 and 0.2 in periods and periods[-1] == 0.5
    g.close()


# ---------------------------------------------------- choosing the backend
class FakeMss:
    made = []

    def __init__(self, monitor=1, region=None, width=192):
        self.box = dict(BUILTIN)
        self.closed = False
        FakeMss.made.append(self)

    pushes = False

    def frame(self):
        return solid((0.1, 0.2, 0.3))

    def close(self):
        self.closed = True


@pytest.fixture
def no_mss(monkeypatch):
    FakeMss.made = []
    monkeypatch.setattr(music_bulb, "ScreenGrabber", FakeMss)
    monkeypatch.setattr(music_bulb, "monitor_box", lambda m: dict(BUILTIN))


def stream_with(kit):
    return lambda display, region, width, fps: StreamGrabber(display, region, width, fps, kit=kit)


def test_mss_never_tries_the_stream(no_mss):
    kit = FakeKit()
    c = Capture(1, None, 192, 10, "mss", stream=stream_with(kit))
    assert c.backend == "mss" and not kit.started and not c.pushes


def test_sck_uses_the_stream(no_mss):
    kit = FakeKit()
    c = Capture(1, None, 192, 10, "sck", stream=stream_with(kit))
    assert c.backend == "sck" and c.pushes and c.frame().shape == (130, 200, 3)
    c.close()
    assert kit.stopped == 1


def test_auto_follows_the_measured_default(no_mss, monkeypatch):
    for default in ("sck", "mss"):
        monkeypatch.setattr(music_bulb, "AUTO_CAPTURE", default)
        c = Capture(1, None, 192, 10, "auto", stream=stream_with(FakeKit()))
        assert c.backend == default
        c.close()


def test_a_stream_that_cannot_start_leaves_mss_and_says_why(no_mss):
    kit = FakeKit(start_error=RuntimeError("stream did not start: declined"))
    c = Capture(1, None, 192, 10, "sck", stream=stream_with(kit))
    assert c.backend == "mss" and "declined" in c.note and c.frame() is not None


def test_without_pyobjc_or_off_macos_capture_is_mss(no_mss, monkeypatch):
    def kit_missing():
        raise ImportError("No module named 'ScreenCaptureKit'")
    monkeypatch.setattr(sck_capture, "Kit", kit_missing)
    c = Capture(1, None, 192, 10, "sck")
    assert c.backend == "mss" and "ScreenCaptureKit" in c.note


def test_a_stream_that_dies_mid_session_is_swapped_for_mss(no_mss):
    kit = FakeKit()
    c = Capture(1, None, 192, 10, "sck", stream=stream_with(kit))
    c.frame()
    kit.on_stop("permission revoked")
    assert c.pace(time.monotonic(), 0.0, 5.0, threading.Event()) is False  # no hang
    f = c.frame()
    assert c.backend == "mss" and np.allclose(f[0, 0], (0.1, 0.2, 0.3))
    assert kit.stopped == 1 and "revoked" in c.note and not c.pushes


def test_the_flag_is_on_both_command_lines():
    import inspect
    assert '"--capture", choices=CAPTURE_BACKENDS' in inspect.getsource(music_bulb.main)
    assert '"--capture", choices=CAPTURE_BACKENDS' in inspect.getsource(dash.main)


# ------------------------------------------------------------ desk loops
class FakeCapture:
    made = []

    def __init__(self, monitor=1, region=None, width=192, fps=10, backend="auto"):
        self.backend, self.pushes, self.closed, self.n = "sck", True, 0, 0
        self.box = dict(BUILTIN)
        self.fail_after = None
        FakeCapture.made.append(self)

    def frame(self):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise RuntimeError("display gone")
        return solid((0.8, 0.1, 0.1) if self.n % 2 else (0.1, 0.1, 0.8))

    def pace(self, started, base, period, wake=None):
        time.sleep(0.005)
        return False

    def set_fps(self, fps):
        pass

    def close(self):
        self.closed += 1


class Sender:
    lost = False

    def put(self, *a, **k):
        pass


def run_loop(loop, st, until, timeout=3.0):
    errors = []

    def body():
        try:
            loop(st, Sender())
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=body, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not FakeCapture.made:
        time.sleep(0.01)
    while time.monotonic() < deadline and FakeCapture.made[-1].n < 3:
        time.sleep(0.01)
    until(st)
    t.join(timeout)
    assert not t.is_alive()
    return errors


@pytest.fixture
def desk(monkeypatch):
    FakeCapture.made = []
    monkeypatch.setattr(dash, "Capture", FakeCapture)
    monkeypatch.setattr(dash, "screen_permission", lambda: True)
    st = dash.State({"ip": "192.168.1.2", "mode": "screen"})
    st.set(running=True)
    return st, dash.Engine(st, capture="sck")


@pytest.mark.parametrize("end", [
    lambda st: st.set(running=False),                    # Stop
    lambda st: st.set(mode="music"),                     # away from the screen
])
def test_the_screen_loop_always_closes_its_capture(desk, end):
    st, eng = desk
    assert run_loop(eng._screen_loop, st, end) == []
    assert FakeCapture.made[0].closed == 1
    assert st.snapshot()["capture"] is None


def test_the_screen_loop_closes_its_capture_on_an_error(desk):
    st, eng = desk

    def fail(st):
        FakeCapture.made[0].fail_after = 0
    errors = run_loop(eng._screen_loop, st, fail)
    assert errors and FakeCapture.made[0].closed == 1


def test_the_state_says_which_capture_runs(desk):
    st, eng = desk
    seen = []

    def stop(st):
        seen.append(st.snapshot()["capture"])
        st.set(running=False)
    run_loop(eng._screen_loop, st, stop)
    assert seen == ["sck"]


def test_the_fusion_thread_closes_its_capture(desk, monkeypatch):
    st, eng = desk
    from music_bulb import ScreenColorEngine, ScreenColorThread
    grab = FakeCapture()
    th = ScreenColorThread(grab, ScreenColorEngine(), fps=30)
    th.start()
    time.sleep(0.05)
    th.close()
    th.join(1)
    assert grab.closed == 1 and not th.is_alive()


def test_the_screen_loop_waits_on_the_stream_not_a_timer(desk, monkeypatch):
    """With a stream the loop hands its wait to the capture, passing the
    desk's `changed` event so a moved control still ends it."""
    st, eng = desk
    waits = []

    def spy(self, started, base, period, wake=None):
        waits.append(wake)
        time.sleep(0.005)
        return False
    monkeypatch.setattr(FakeCapture, "pace", spy)
    run_loop(eng._screen_loop, st, lambda st: st.set(running=False))
    assert waits and all(w is st.changed for w in waits)


def test_a_change_too_small_to_count_as_motion_does_not_wake_the_loop():
    """A blinking cursor must not hold a still picture at the full rate: the
    frame is kept, but the wait runs to the pacer's period."""
    kit = FakeKit()
    g = StreamGrabber(BUILTIN, None, 192, 10, kit=kit)
    g.frame()
    threading.Timer(0.03, kit.push, args=(0.51,)).start()    # 0.01: under PIXEL
    t = time.monotonic()
    assert g.pace(t, 0.01, 0.2, threading.Event()) is False
    assert time.monotonic() - t >= 0.19
    assert np.allclose(g.frame(), 130 / 255.0) and g.delivered == 2
    g.close()
