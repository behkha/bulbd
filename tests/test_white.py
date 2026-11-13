"""True white for grey scenes: the decision, the temperature, the brightness
match, the deadband and the exact frames on the wire."""
import numpy as np
import pytest

from music_bulb import (SWITCH_MS, WHITE_GAIN, ScreenColorEngine, White, crop_letterbox,
                        neutral_kelvin, ramp_ms, rgb_to_hsv, same_light, screen_frame,
                        send_frame, white_frame)
from signals import W, patch, solid

yeelight = pytest.importorskip("yeelight")


def run(frames, fps=10, eng=None):
    """Feed frames the way screen mode does. -> (engine, [frame sent per tick])."""
    eng = eng or ScreenColorEngine()
    out = []
    for f in frames:
        hue, sat, val, _cut = eng.tick(1 / fps, *eng.measure(crop_letterbox(f)))
        v = int(round(float(np.clip(3 + 97 * val, 1, 100))))
        out.append(screen_frame(eng, hue, sat, v))
    return eng, out


def hold(frame, secs=4.0, fps=10):
    return [frame] * int(secs * fps)


def accent(frac, colour=(0, .9, 0), grey=(.55, .55, .55)):
    """A grey picture with a vivid block covering `frac` of it."""
    return patch(grey, colour, frac)


# ------------------------------------------------------------ the decision
@pytest.mark.parametrize("rgb", [(1, 1, 1), (.5, .5, .5), (.2, .2, .2), (.62, .6, .56)])
def test_a_colourless_scene_goes_to_the_white_leds(rgb):
    eng, out = run(hold(solid(rgb)))
    assert eng.white and isinstance(out[-1], White)


@pytest.mark.parametrize("rgb", [(1, 0, 0), (0, 0, 1), (.9, .7, .6)])
def test_a_colour_scene_stays_on_the_colour_leds(rgb):
    eng, out = run(hold(solid(rgb)))
    assert not eng.white and not isinstance(out[-1], White)


def test_a_tiny_accent_on_grey_is_white_and_a_real_one_is_colour():
    assert run(hold(accent(.01)))[0].white
    assert not run(hold(accent(.05)))[0].white


def test_a_pale_scene_is_white_and_a_pastel_keeps_its_colour():
    assert run(hold(solid((1, .9, .8))))[0].white               # cream, s 0.20
    assert not run(hold(solid((.9, .7, .6))))[0].white          # pastel, s 0.33


def test_a_black_frame_is_white_at_the_bottom_of_the_range():
    eng, out = run(hold(solid((0, 0, 0))))
    assert eng.white and out[-1].bright == 1


def test_the_boundary_does_not_flap():
    """A scene whose accent wanders back and forth across the line every
    second or so, with grain, must not switch LEDs back and forth."""
    rng = np.random.default_rng(3)
    eng = ScreenColorEngine()
    run(hold(accent(.022), 3), eng=eng)
    frames = []
    for i in range(300):                                   # 30 s
        frac = .022 + .006 * np.sin(i / 3.0)               # 1.6% .. 2.8% of the picture
        f = accent(frac) + rng.normal(0, .02, (130, W, 3)).astype(np.float32)
        frames.append(np.clip(f, 0, 1))
    _, out = run(frames, eng=eng)
    kinds = [isinstance(o, White) for o in out]
    switches = sum(a != b for a, b in zip(kinds, kinds[1:]))
    assert switches <= 1


def test_it_does_switch_once_the_scene_really_changes():
    eng = ScreenColorEngine()
    _, a = run(hold(solid((.5, .5, .5))), eng=eng)
    _, b = run(hold(solid((0, .3, 1))), eng=eng)
    _, c = run(hold(solid((.5, .5, .5))), eng=eng)
    assert isinstance(a[-1], White) and not isinstance(b[-1], White) and isinstance(c[-1], White)


# ----------------------------------------------------------- temperature
def kelvin(rgb):
    f = solid(rgb)
    _, s, v = rgb_to_hsv(f)
    return neutral_kelvin(f, s, v)


def test_a_white_page_is_d65():
    assert kelvin((1, 1, 1)) == pytest.approx(6500, abs=60)


def test_warm_greys_come_out_warmer_than_neutral_ones():
    warm, warmer = kelvin((.6, .55, .5)), kelvin((.6, .52, .44))
    assert 2700 <= warmer < warm < 6000


def test_cooler_than_the_display_is_clamped_to_its_white():
    assert kelvin((.8, .85, 1.0)) == 6500


def test_an_accent_does_not_pull_the_temperature():
    f = accent(.2, colour=(1, .2, 0))
    _, s, v = rgb_to_hsv(f)
    assert neutral_kelvin(f, s, v) == pytest.approx(kelvin((.55, .55, .55)), abs=50)


def test_a_dark_frame_has_no_temperature_and_keeps_the_last_one():
    assert kelvin((.05, .05, .05)) is None
    eng, _ = run(hold(solid((.6, .55, .5)), 3))
    before = eng.ct
    run(hold(solid((0, 0, 0)), 2), eng=eng)
    assert eng.ct == pytest.approx(before, abs=1)


def test_the_temperature_follows_the_picture_and_lands_on_a_cut():
    eng, _ = run(hold(solid((.9, .9, .9)), 3))
    assert eng.ct > 6000
    _, s, v = rgb_to_hsv(solid((.35, .33, .3)))
    target = neutral_kelvin(solid((.35, .33, .3)), s, v)
    assert target < 6000
    run([solid((.35, .33, .3))], eng=eng)                 # a 0.5 luma drop: a cut
    assert eng.ct == pytest.approx(target, abs=1)


# -------------------------------------------------- brightness and switching
def test_white_is_sent_at_the_matched_brightness():
    assert white_frame(4120, 80) == White(4100, round(80 * WHITE_GAIN))
    assert white_frame(6500, 1).bright == 1


def test_the_switch_crossfades_and_ordinary_frames_do_not():
    w, c = White(5000, 30), (200, 60, 75)
    assert ramp_ms(w, c, .1) == SWITCH_MS and ramp_ms(c, w, .1) == SWITCH_MS
    assert ramp_ms(c, (201, 60, 75), .1) == 100 and ramp_ms(w, White(5050, 30), .04) == 40


def test_deadband_understands_white():
    assert same_light(White(5000, 30), White(5050, 30))
    assert not same_light(White(5000, 30), White(5100, 30))
    assert not same_light(White(5000, 30), White(5000, 31))
    assert not same_light(White(5000, 30), (0, 0, 30))        # a switch always goes out
    assert not same_light((0, 0, 30), White(5000, 30))
    assert same_light((10, 50, 50), (11, 51, 51))


def test_level_stays_put_across_the_switch():
    """The telemetry level is on the colour scale either side, so the page's
    field does not jump when the lamp changes LEDs."""
    from music_bulb_dash import light_telemetry
    t_white = light_telemetry(White(4000, 24), 30.0, 60)
    t_colour = light_telemetry((30, 40, 60), 30.0, 60)
    assert t_white["val"] == t_colour["val"] == 60
    assert t_white["white"] and t_white["ct"] == 4000 and not t_colour["white"]


# ------------------------------------------------------------- the wire
class FakeSock:
    def __init__(self):
        self.sent = []

    def send(self, data):
        import json
        if data.strip():
            self.sent.append(json.loads(data.decode().strip()))
        return len(data)

    def close(self):
        pass


@pytest.fixture
def wire():
    b = yeelight.Bulb("192.0.2.1", effect="sudden", duration=0, auto_on=True)
    sock = FakeSock()
    b._music_mode = True
    b._Bulb__socket = sock
    return b, sock


@pytest.mark.parametrize("frame, sudden, duration, expr", [
    (White(3400, 25), False, 100, "100, 2, 3400, 25"),        # screen at 10 fps
    (White(6500, 40), False, 300, "300, 2, 6500, 40"),        # crossfade into white
    (White(5000, 12), True, 0, "50, 2, 5000, 12"),            # a cut lands in 50 ms
    (White(2700, 1), False, 10, "30, 2, 2700, 1"),            # never under 30 ms
])
def test_white_goes_out_as_one_colour_temperature_flow(wire, frame, sudden, duration, expr):
    b, sock = wire
    send_frame(b, frame, sudden, duration)
    assert sock.sent == [{"id": sock.sent[0]["id"], "method": "start_cf", "params": [1, 1, expr]}]


def test_colour_frames_are_unchanged(wire):
    b, sock = wire
    send_frame(b, (26, 92, 80), False, 40)
    assert sock.sent[0]["params"] == [1, 1, "40, 1, 16742932, 80"]


def test_the_fusion_thread_publishes_white(monkeypatch):
    from music_bulb import ScreenColorThread

    class Grab:
        def frame(self):
            return solid((.5, .5, .5))

        def close(self):
            pass

    eng = ScreenColorEngine()
    th = ScreenColorThread(Grab(), eng, fps=30)
    th.start()
    import time
    deadline = time.monotonic() + 3
    while not th.white()[0] and time.monotonic() < deadline:
        time.sleep(0.02)
    th.close()
    th.join(2)
    white, ct = th.white()
    assert white and 6000 < ct <= 6500
    assert isinstance(screen_frame(th, 30, 18, 50), White)
