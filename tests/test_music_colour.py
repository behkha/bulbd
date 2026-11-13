"""Palette state machine, fade sweeps, brightness, driven by synthetic audio."""
import numpy as np
import pytest

from music_bulb import PALETTES, ColorEngine, hue_lerp
from signals import SR, hue_gap, run_music, track

EMBER = [a[0] for a in PALETTES["ember"]]


@pytest.fixture(scope="module")
def steady():
    pcm, _ = track(120)
    return run_music(pcm, palette="ember")


# ------------------------------------------------------------------- steps
def test_colour_steps_every_two_beats(steady):
    gaps = np.diff([t for t, _, _ in steady["steps"]])
    assert len(gaps) >= 30
    assert gaps.min() >= 0.94                      # two beats at 120 BPM, less 60 ms of slack
    assert 0.95 <= np.median(gaps) <= 1.1


@pytest.mark.parametrize("bpm", [90, 150])
def test_step_gap_follows_tempo(bpm):
    """Two beats apart, but never closer than 0.9 s (so 150 BPM is held to 0.9, not 0.8)."""
    pcm, _ = track(bpm)
    gaps = np.diff([t for t, _, _ in run_music(pcm, palette="ember")["steps"] if t > 8])
    floor = max(120.0 / bpm - 0.06, 0.9)
    assert gaps.min() >= floor - 0.02
    assert floor <= np.median(gaps) <= floor + 0.4


def test_steps_walk_the_anchors_in_order(steady):
    steps = steady["steps"]
    assert steps[0][1] == EMBER[0]
    for k, (_, a, c) in enumerate(steps):
        assert a == EMBER[k % 4] and c == EMBER[(k + 1) % 4]


def test_hue_sits_on_anchors(steady):
    rows = [r for r in steady["rows"] if r[0] > 5]
    dist = np.array([min(abs(hue_gap(r[1], a)) for a in EMBER) for r in rows])
    assert (dist < 2).mean() > 0.6
    assert dist.max() < 12                          # never drifts between anchors


def test_pinned_palette_never_switches():
    p1, _ = track(120, secs=40)
    p2, _ = track(120, secs=40, hat=1.5, pad=0.4, seed=3)
    out = run_music(np.concatenate([p1, p2]), palette="ember")
    assert {r[6] for r in out["rows"]} == {"ember"}


def test_palette_switches_once_on_a_section_change():
    p1, _ = track(120, secs=40)
    p2, _ = track(120, secs=40, hat=1.5, pad=0.4, seed=3)    # kick-only -> kick + bright noise
    rows = run_music(np.concatenate([p1, p2]))["rows"]
    changes = [(a[0], b[6]) for a, b in zip(rows, rows[1:]) if a[6] != b[6]]
    assert len(changes) == 1
    t, pal = changes[0]
    assert 40 <= t <= 46 and pal != "ember"


# -------------------------------------------------------------------- fade
@pytest.fixture(scope="module")
def faded():
    pcm, _ = track(120)
    return run_music(pcm, palette="ember", fade=1.5)


def test_fade_steps_are_held_a_sweep_apart(faded):
    gaps = np.diff([t for t, _, _ in faded["steps"]])
    assert len(gaps) >= 15 and gaps.min() >= 1.5 - 0.03


def test_fade_sweep_length(faded):
    rows = np.array([(r[0], r[1]) for r in faded["rows"]])
    t0, a, c = faded["steps"][3]
    seg = rows[(rows[:, 0] >= t0) & (rows[:, 0] <= t0 + 1.6)]
    arc = abs(hue_gap(c, a))
    done = np.array([abs(hue_gap(h, a)) / arc for h in seg[:, 1]])
    t50 = seg[np.argmax(done >= .5), 0] - t0
    t99 = seg[np.argmax(done >= .99), 0] - t0
    assert 0.6 <= t50 <= 0.95                       # smoothstep: half way at half time
    assert 1.2 <= t99 <= 1.55


def test_fade_makes_no_hard_sets_mid_sweep(faded):
    sends = faded["sends"]
    jumps = [abs(hue_gap(b[1][0], a[1][0])) for a, b in zip(sends, sends[1:])]
    assert max(jumps) <= 6
    for t0, _, _ in faded["steps"]:
        assert not [s for s in sends if t0 + 0.05 < s[0] < t0 + 1.45 and s[2]]


def test_without_fade_the_hue_snaps_towards_the_anchor():
    ce = ColorEngine("ember")
    ce.step(2.0, 0.5)
    assert ce.target_hue == 26 and ce.fade_t is None
    assert abs(hue_gap(ce.hue, 6)) > 8              # part of the way at once, for punch


def test_hue_lerp_takes_the_short_arc():
    assert hue_lerp(350, 10, 0.5) == pytest.approx(0.0)
    assert hue_lerp(10, 350, 0.5) == pytest.approx(0.0)
    assert hue_lerp(0, 120, 0.25) == pytest.approx(30.0)


# -------------------------------------------------------------- brightness
def test_silence_breathes_in_its_range():
    out = run_music(np.zeros((SR * 12, 2), dtype=np.float32))
    b = np.array([r[3] for r in out["rows"] if r[0] > 3])
    assert 0.05 <= b.min() < 0.08 and 0.15 <= b.max() <= 0.17
    assert b.max() - b.min() > 0.07                 # it actually moves


def test_beats_swing_the_brightness():
    pcm, _ = track(120)
    out = run_music(pcm, palette="ember")
    b = np.array([r[3] for r in out["rows"] if 20 < r[0] < 30])
    assert b.min() <= 0.40 and b.max() >= 0.75
    vals = [s[1][2] for s in out["sends"] if 20 < s[0] < 30]
    assert min(vals) <= 25 and max(vals) >= 60


def test_send_rate_stays_under_the_led_limit():
    pcm, _ = track(120)
    out = run_music(pcm, palette="ember")
    n = len([s for s in out["sends"] if s[0] > 5])
    assert n / 35 <= 25.5                           # --fps default 25; LEDs resolve ~30
