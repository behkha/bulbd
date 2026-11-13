"""Onset detection: silence on sustained sound, and kicks still caught."""
import numpy as np
import pytest

from signals import GATE, SR, STRONG, analyse, caught, stereo, sustained, track


def rates(onsets, secs=24.0, after=6.0):
    late = [s for t, s in onsets if t > after]
    return (sum(1 for s in late if s > GATE) / secs, sum(1 for s in late if s > STRONG) / secs)


# ------------------------------------------------- no beats on sustained sound
@pytest.mark.parametrize("name", ["tone-55", "tone-110", "tone-440", "tone-1760", "tone-5000",
                                  "chord", "vibrato"])
def test_steady_note_has_no_onsets(name):
    _, onsets, _ = analyse(stereo(sustained(name)))
    assert [t for t, _ in onsets if t > 6] == []


@pytest.mark.parametrize("hz", [40, 45, 55, 65, 80])
def test_held_bass_note_has_no_beats(hz):
    t = np.arange(SR * 30) / SR
    _, onsets, _ = analyse(stereo(0.3 * np.sin(2 * np.pi * hz * t)))
    assert rates(onsets) == (0.0, 0.0)


# a few weak ones are allowed in steady noise (and in a wobbling amplitude),
# but nothing strong enough to step the colour, and no steady drumming
@pytest.mark.parametrize("name, gate_max", [("white", 0.6), ("hiss", 0.6), ("pink", 0.4),
                                            ("tremolo", 0.8)])
def test_steady_noise_stays_quiet(name, gate_max):
    _, onsets, _ = analyse(stereo(sustained(name)))
    passing, strong = rates(onsets)
    assert passing <= gate_max
    assert strong <= 0.15


def test_steady_tone_holds_brightness_flat():
    _, _, bright = analyse(stereo(sustained("tone-440", secs=12)))
    b = np.array([v for t, v in bright if t > 6])
    assert b.max() - b.min() < 0.03


# ------------------------------------------------------ real beats still caught
@pytest.mark.parametrize("label, kw, min_caught", [
    ("kick and hat", {}, 69),
    ("34 dB quieter", dict(kick=.02, hat=.005, pad=.001), 69),
    ("noise bed 0.05", dict(noise=.05), 68),
    ("noise bed 0.15", dict(noise=.15), 67),
    ("noise bed 0.30", dict(noise=.30), 67),
    ("kick snare hat", dict(snare=.6), 66),
    ("loud pad 0.5", dict(pad=.5), 68),
])
def test_kicks_are_caught(label, kw, min_caught):
    pcm, kicks = track(120, **kw)
    _, onsets, _ = analyse(pcm)
    got, want = caught(onsets, kicks)
    assert want == 69
    assert got >= min_caught, f"{label}: {got}/{want}"


@pytest.mark.parametrize("label, kw", [("kick and hat", {}), ("noise bed 0.15", dict(noise=.15)),
                                       ("kick snare hat", dict(snare=.6)), ("loud pad 0.5", dict(pad=.5))])
def test_no_onsets_between_kicks_and_hats(label, kw):
    pcm, kicks = track(120, **kw)
    _, onsets, _ = analyse(pcm)
    stray = [t for t, s in onsets if t > 5 and s > GATE
             and not any(abs(t - k) < .07 or abs(t - k - .25) < .07 for k in kicks)]
    assert stray == []


def test_beats_return_soon_after_a_loud_to_quiet_change():
    loud, _ = track(120, secs=30)
    quiet, qk = track(120, secs=30, kick=.05, hat=.0125, pad=.0025)   # 26 dB down
    _, onsets, _ = analyse(np.concatenate([loud, quiet]))
    first = next(t - 30 for t, s in onsets if t > 30.2 and s > GATE)
    assert first < 1.2
    got, want = caught(onsets, qk, after=31.0, offset=30.0)
    assert want == 57 and got >= 55


def test_strong_onsets_are_kick_sized():
    """Colour steps follow strong onsets; a kick-and-hat track must produce them."""
    pcm, kicks = track(120)
    _, onsets, _ = analyse(pcm)
    strong = [t for t, s in onsets if t > 5 and s > STRONG]
    assert len(strong) >= 30
