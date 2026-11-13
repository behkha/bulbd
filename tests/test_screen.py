"""Screen colour: hue, saturation, brightness, scene cuts, pacer, helpers."""
import numpy as np
import pytest

from music_bulb import CapturePacer, ScreenColorEngine, crop_letterbox, rgb_to_hsv
from signals import H, W, drift_frame, hue_gap, noisy_two_colour, patch, settle, solid, to_bulb


def two_thirds():
    return patch((0, 0, 1), (1, 0, 0), .6)


# ----------------------------------------------------- solid frames, 4 s each
@pytest.mark.parametrize("name, rgb, hue, val", [
    ("red", (1, 0, 0), 0, 33), ("green", (0, 1, 0), 120, 78), ("blue", (0, 0, 1), 240, 17),
    ("yellow", (1, 1, 0), 60, 95), ("dark red", (.2, 0, 0), 0, 12)])
def test_vivid_solid_frame(name, rgb, hue, val):
    (h, s, v), hue_m, _ = settle(solid(rgb))
    assert abs(hue_gap(h, hue)) <= 2 and s >= 98 and abs(v - val) <= 3
    assert hue_m == pytest.approx(hue, abs=1)


@pytest.mark.parametrize("name, rgb, val", [("white", (1, 1, 1), 100), ("mid grey", (.5, .5, .5), 61),
                                            ("black", (0, 0, 0), 3)])
def test_colourless_frame_is_near_white_at_its_own_brightness(name, rgb, val):
    (h, s, v), hue_m, _ = settle(solid(rgb))
    assert hue_m is None
    assert abs(s - 18) <= 3 and abs(h - 30) <= 3 and abs(v - val) <= 3


def test_pastel_keeps_its_hue_at_the_saturation_floor():
    (h, s, v), _, _ = settle(solid((.9, .7, .6)))
    assert abs(hue_gap(h, 20)) <= 3 and 52 <= s <= 60 and abs(v - 80) <= 3


def test_majority_colour_wins():
    (h, s, v), _, _ = settle(two_thirds())
    assert abs(hue_gap(h, 0)) <= 2 and s >= 98


def test_letterbox_bars_do_not_dim_the_lamp():
    film = solid((0, 0, 0)); film[20:110, :] = (1, 0, 0)
    assert settle(film)[0] == settle(solid((1, 0, 0)))[0]
    assert crop_letterbox(film).shape == (90, W, 3)


def test_crop_letterbox_keeps_a_black_frame():
    assert crop_letterbox(solid((0, 0, 0))).shape == (H, W, 3)


def test_small_vivid_window_on_grey_still_tints_the_lamp():
    f = solid((.1, .1, .1)); f[50:80, 80:120] = (0, 1, 0)
    (h, s, v), _, _ = settle(f)
    assert abs(hue_gap(h, 120)) <= 2 and s >= 98 and abs(v - 24) <= 4


@pytest.mark.parametrize("share", [0.50, 0.52, 0.55, 0.60])
def test_near_even_split_with_grain_does_not_flicker(share):
    rng = np.random.default_rng(1)
    eng = ScreenColorEngine(); tops, hues = [], []
    for _ in range(100):
        hm, sm, lu = eng.measure(noisy_two_colour(rng, share))
        hues.append(eng.tick(.1, hm, sm, lu)[0]); tops.append(hm)
    flips = sum(1 for a, b in zip(tops, tops[1:]) if abs(hue_gap(a, b)) > 60)
    assert flips == 0
    assert max(abs(hue_gap(h, hues[50])) for h in hues[50:]) <= 5


# ------------------------------------------------- saturation vs share of light
@pytest.mark.parametrize("name, frame, lo, hi", [
    ("full green", solid((0, 1, 0)), 98, 100),
    ("5% green on black", patch((0, 0, 0), (0, 1, 0), .05), 98, 100),
    ("5% green on dark grey", patch((.1, .1, .1), (0, 1, 0), .05), 98, 100),
    ("5% green on mid grey", patch((.5, .5, .5), (0, 1, 0), .05), 45, 70),
    ("5% green on white", patch((1, 1, 1), (0, 1, 0), .05), 30, 50),
    ("faint orange accent on dark UI", patch((.13, .13, .13), (.9, .5, .3), .0015), 15, 25),
    ("25% green on mid grey", patch((.5, .5, .5), (0, 1, 0), .25), 98, 100),
])
def test_saturation_follows_the_colours_share_of_the_light(name, frame, lo, hi):
    (_, s, _), _, _ = settle(frame, secs=5)
    assert lo <= s <= hi, f"{name}: sat {s}"


def test_saturation_falls_as_the_grey_gets_brighter():
    sats = [settle(patch(bg, (0, 1, 0), .05), secs=5)[0][1]
            for bg in ((0, 0, 0), (.1, .1, .1), (.5, .5, .5), (1, 1, 1))]
    assert sats == sorted(sats, reverse=True) and sats[0] - sats[-1] > 50


def test_share_is_reported():
    assert settle(solid((0, 1, 0)))[2] == pytest.approx(1.0, abs=0.01)
    assert settle(patch((.5, .5, .5), (0, 1, 0), .05))[2] == pytest.approx(0.10, abs=0.03)


# ---------------------------------------------------------------- scene cuts
CUTS = [("red", (1, 0, 0), "green", (0, 1, 0)), ("red", (1, 0, 0), "blue", (0, 0, 1)),
        ("orange", (1, .5, 0), "teal", (0, .6, .6)), ("dark blue", (0, 0, .5), "white", (1, 1, 1)),
        ("green", (0, 1, 0), "black", (0, 0, 0))]


@pytest.mark.parametrize("a_name, a, c_name, c", CUTS)
def test_hard_cut_is_detected_and_lands_in_one_frame(a_name, a, c_name, c):
    fps = 10
    eng = ScreenColorEngine()
    for _ in range(40):
        eng.tick(1 / fps, *eng.measure(solid(a)))
    ref = ScreenColorEngine()                                  # where this scene settles on its own
    for _ in range(80):
        end = ref.tick(1 / fps, *ref.measure(solid(c)))
    hue, sat, val, cut = eng.tick(1 / fps, *eng.measure(solid(c)))
    assert cut
    assert abs(hue_gap(hue, end[0])) < 10 or end[1] < 25
    assert abs(sat - end[1]) < 5 and abs(val - end[2]) * 97 < 5


def cuts_over(frames):
    eng = ScreenColorEngine()
    return sum(int(eng.tick(.1, *eng.measure(f))[3]) for f in frames)


def test_grain_is_not_a_cut():
    rng = np.random.default_rng(2)
    assert cuts_over(noisy_two_colour(rng) for _ in range(100)) == 0


def test_slow_pan_is_not_a_cut():
    def frame(i):
        f = solid((1, .1, 0)); f[:, :(i * 2) % W] = (0, .3, 1)
        return f
    assert cuts_over(frame(i) for i in range(1, 100)) == 0     # i=0 and the wrap at 100 are real jumps


def test_slow_colour_drift_is_not_a_cut():
    assert cuts_over(drift_frame(i) for i in range(100)) == 0


def test_measure_is_stateful_so_a_frame_is_not_measured_twice():
    """measure() remembers the previous frame; measuring one frame twice reads as 'nothing moved'."""
    eng = ScreenColorEngine()
    eng.measure(solid((1, 0, 0)))
    eng.measure(solid((0, 1, 0)))
    assert eng.changed > 0.99
    eng.measure(solid((0, 1, 0)))
    assert eng.changed == 0.0


# --------------------------------------------------------------- CapturePacer
def drive(pacer, frames, fps=10):
    """Feed frames at the pace the pacer asks for; -> periods."""
    now, out = 0.0, []
    for f in frames:
        p = pacer.period(f, fps, now)
        out.append(p)
        now += p
    return out


def test_pacer_runs_flat_out_on_a_moving_picture():
    frames = [solid((i % 2, 0, 0)) for i in range(40)]
    assert set(drive(CapturePacer(), frames)) == {0.1}


def test_pacer_steps_down_on_a_still_picture():
    periods = drive(CapturePacer(), [solid((.3, .3, .3))] * 200)
    assert periods[0] == 0.1
    assert 0.2 in periods and periods[-1] == 0.5
    assert periods.index(0.2) * 0.1 == pytest.approx(1.5, abs=0.15)      # 5 fps after 1.5 s
    assert periods == sorted(periods)                                     # only ever slower


def test_pacer_returns_to_full_rate_on_the_first_changed_frame():
    p = CapturePacer()
    drive(p, [solid((.3, .3, .3))] * 150)
    assert p.period(solid((.3, .3, .3)), 10, 100.0) == 0.5
    assert p.period(solid((.9, .3, .3)), 10, 100.5) == 0.1


def test_pacer_ignores_a_tiny_change():
    f = solid((.3, .3, .3)); g = f.copy(); g[:2, :5] = 1.0                # 10 of 26000 pixels
    assert drive(CapturePacer(), [f, g] * 50)[-1] > 0.1


def test_pacer_counts_a_slow_fade_as_motion():
    frames = [solid((.2 + .01 * i, .2, .2)) for i in range(60)]           # each frame is a hair different
    assert set(drive(CapturePacer(), frames)) == {0.1}


def test_pacer_never_beats_the_requested_rate_and_reset_restores_it():
    p = CapturePacer()
    assert drive(p, [solid((.3, .3, .3))] * 3, fps=30)[0] == pytest.approx(1 / 30)
    drive(p, [solid((.3, .3, .3))] * 150)
    p.reset()
    assert p.period(solid((.3, .3, .3)), 10, 200.0) == 0.1


# --------------------------------------------------------------- colour maths
def test_rgb_to_hsv_matches_colorsys():
    import colorsys
    rng = np.random.default_rng(5)
    px = rng.random((200, 3)).astype(np.float32)
    h, s, v = rgb_to_hsv(px)
    for (r, g, b), hh, ss, vv in zip(px, h, s, v):
        eh, es, ev = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        assert hh == pytest.approx(eh * 360, abs=0.01) and ss == pytest.approx(es, abs=1e-4)
        assert vv == pytest.approx(ev, abs=1e-5)


def test_rgb_to_hsv_greys_have_no_hue():
    h, s, v = rgb_to_hsv(np.array([[.4, .4, .4], [0, 0, 0]], dtype=np.float32))
    assert list(h) == [0, 0] and list(s) == [0, 0] and v[0] == pytest.approx(.4)
