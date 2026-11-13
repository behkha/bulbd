"""Tempo estimate across the grid, for each drum pattern."""
import pytest

from signals import analyse, track

PATTERNS = {"kick-only": dict(hat=0), "kick-hat": {}, "kick-loud-hat": dict(hat=1.0),
            "kick-snare-hat": dict(snare=.6)}
BPMS = (66, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176)


@pytest.mark.parametrize("bpm", BPMS)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_tempo_within_four_percent(pattern, bpm):
    pcm, _ = track(bpm, **PATTERNS[pattern])
    an, _, _ = analyse(pcm)
    est = 60.0 / an.beat
    assert abs(est - bpm) / bpm <= 0.04, f"{pattern}: {bpm} -> {est:.1f}"
