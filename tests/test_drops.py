"""Drops and build-ups: found after a breakdown, silent on steady sound, a
groove that never stops, and a gap of silence between tracks."""
import numpy as np
import pytest

from music_bulb import (BLOCK, MAX_STROBE, Analyser, DropWatch, Flow, flow_ms, strobe_flow,
                        swell_flow)
from signals import SR, _drums, stereo, sustained, track


def events(pcm):
    """-> [(t, "build" | "drop")]"""
    an, dw, out = Analyser(), DropWatch(), []
    for i in range(0, len(pcm) - BLOCK, BLOCK):
        e = dw.feed(an.feed(pcm[i:i + BLOCK]))
        if e:
            out.append((round(an.t, 2), e))
    return out


def pad(secs, level=0.05, hz=220):
    t = np.arange(int(secs * SR)) / SR
    return stereo(0.5 * level * np.sin(2 * np.pi * hz * t))


def roll(secs, bpm=128, level=0.6, start=1, end=4, seed=0):
    """A snare roll over a pad that speeds up from `start` to `end` hits per
    beat and gets louder: the usual build-up."""
    _, _, sn = _drums(seed)
    beat = 60.0 / bpm
    n = int(secs * SR)
    x = np.zeros(n, dtype=np.float32)
    t, quarter = 0.0, secs / 4
    while t < secs - 0.1:
        k = min(int(t / quarter), 3)
        per = [start, 2, end, end][k]
        s = int(t * SR)
        amp = level * (0.3 + 0.7 * t / secs)
        x[s:s + len(sn)] += (amp * sn)[:n - s]
        t += beat / per
    return stereo(0.5 * x) + pad(secs)


def song(*parts):
    return np.concatenate(parts)


GROOVE = track(128, secs=16)[0]
AGAIN = track(128, secs=10)[0]
GROOVE_START = 16.0


# ------------------------------------------------------------ found
def test_the_beat_coming_back_after_a_breakdown_is_a_drop():
    ev = events(song(GROOVE, pad(10), AGAIN))
    drops = [t for t, e in ev if e == "drop"]
    assert len(drops) == 1
    assert 26.0 <= drops[0] < 26.15                # on the first kick back
    assert [e for t, e in ev if e == "build"] == []   # a still pad is no build-up


def test_a_snare_roll_before_the_drop_is_a_build_up():
    ev = events(song(GROOVE, pad(4), roll(8), AGAIN))
    builds = [t for t, e in ev if e == "build"]
    drops = [t for t, e in ev if e == "drop"]
    assert len(builds) == 1 and 20.0 < builds[0] < 28.0
    assert len(drops) == 1 and 28.0 <= drops[0] < 28.15


def test_a_riser_that_only_gets_louder_is_a_build_up():
    t = np.arange(int(6 * SR)) / SR
    rng = np.random.default_rng(1)
    riser = stereo(0.5 * (0.004 * 10 ** (1.8 * t / 6)) * rng.normal(0, 1, len(t)))
    ev = events(song(GROOVE, pad(3), riser + pad(6), AGAIN))
    assert [e for _, e in ev] == ["build", "drop"]


@pytest.mark.parametrize("bpm", [90, 128, 150, 174])
def test_drops_are_found_at_any_tempo(bpm):
    a, b = track(bpm, secs=14)[0], track(bpm, secs=8)[0]
    ev = events(song(a, pad(8), b))
    assert [e for _, e in ev] == ["drop"]
    assert 22.0 <= ev[0][0] < 22.15


# ------------------------------------------------------------ silent
@pytest.mark.parametrize("name", ["tone-55", "tone-110", "tone-440", "tone-5000", "chord",
                                  "vibrato", "tremolo", "white", "hiss", "pink"])
def test_steady_sound_gives_nothing(name):
    assert events(stereo(sustained(name))) == []


@pytest.mark.parametrize("label, kw", [("kick and hat", {}), ("kick snare hat", dict(snare=.6)),
                                       ("noise bed", dict(noise=.15)), ("loud pad", dict(pad=.5))])
def test_a_groove_that_never_stops_gives_nothing(label, kw):
    assert events(track(128, secs=40, **kw)[0]) == []


def test_a_held_chord_in_the_breakdown_is_no_build_up():
    chord = stereo(sustained("chord", secs=10))
    ev = events(song(GROOVE, chord, AGAIN))
    assert [e for _, e in ev] == ["drop"]


def test_a_gap_of_silence_between_tracks_is_no_drop():
    ev = events(song(GROOVE, np.zeros((int(8 * SR), 2), np.float32), AGAIN))
    assert ev == []


def test_a_short_stop_is_no_drop():
    ev = events(song(GROOVE, pad(3), AGAIN))
    assert ev == []


def test_the_first_beats_of_a_track_are_no_drop():
    ev = events(song(pad(8), AGAIN))
    assert ev == []


# ------------------------------------------------------------ hats through the gap
def hats(secs, bpm=128, per_beat=2, level=0.25, seed=0):
    """Hats (and a light snare on 2 and 4) with no kick: the gap of a kick
    dropout, a fill or a kickless verse, where the rest keeps playing."""
    _, hh, sn = _drums(seed)
    beat = 60.0 / bpm
    n = int(secs * SR)
    x = np.zeros(n, dtype=np.float32)
    k = 0
    while k * beat / per_beat < secs - 0.05:
        s = int(k * beat / per_beat * SR)
        x[s:s + len(hh)] += (level * hh)[:n - s]
        if k % (2 * per_beat) == per_beat:
            x[s:s + len(sn)] += (0.3 * sn)[:n - s]
        k += 1
    return stereo(0.5 * x) + pad(secs)


BAR = 4 * 60 / 128


@pytest.mark.parametrize("per_beat", [2, 4])
def test_a_one_bar_kick_dropout_with_the_hats_going_on_gives_nothing(per_beat):
    ev = events(song(GROOVE, hats(BAR, per_beat=per_beat), AGAIN))
    assert ev == []


def test_a_kick_dropout_every_eighth_bar_gives_nothing():
    parts = []
    for _ in range(5):                                  # 5 x (7 bars + a bar without kick)
        parts += [track(128, secs=7 * BAR)[0], hats(BAR, per_beat=4)]
    assert events(song(*parts)) == []


@pytest.mark.parametrize("per_beat", [2, 4])
def test_a_kickless_verse_is_no_build_up(per_beat):
    ev = events(song(GROOVE, hats(12, per_beat=per_beat), AGAIN))
    assert [e for _, e in ev if e == "build"] == []


def test_a_fill_is_no_build_up_and_no_drop():
    # half a bar of 16ths speeding to 32nds where the kick was, then the groove
    ev = events(song(GROOVE, hats(BAR / 2, per_beat=4), hats(BAR / 2, per_beat=8), AGAIN))
    assert ev == []


def test_a_swell_ends_when_the_beat_is_back():
    from music_bulb import DropFx
    fx, an = DropFx(), Analyser()
    pcm = song(GROOVE, hats(BAR, per_beat=4), AGAIN)
    started = ended = None
    for i in range(0, len(pcm) - BLOCK, BLOCK):
        fx.feed(an.feed(pcm[i:i + BLOCK]))
        if started is None and an.t >= 16.3:
            fx.event = "build"                           # as if a swell had started here
            assert fx.take(an.t, (20, 80, 40)) is not None
            started = an.t
        elif started is not None and not fx.holding(an.t):
            ended = an.t
            break
    assert ended is not None and ended < 16.0 + BAR + 0.3   # on the first kick back


def test_a_build_during_a_swell_does_not_extend_it():
    from music_bulb import DropFx
    fx = DropFx()
    fx.event = "build"
    fx.take(0.0, (20, 80, 40))
    until = fx.until
    fx.event = "build"
    assert fx.take(3.0, (20, 80, 40)) is None and fx.until == until


# ------------------------------------------------------------ the flows
def test_the_strobe_never_passes_three_a_second():
    f = strobe_flow(200, 90, 20, rate=10, flashes=4)
    assert isinstance(f, Flow) and f.count == 8
    assert flow_ms(f) >= 4 / MAX_STROBE - 1e-6
    assert all(ms >= 50 for ms, *_ in f.steps)          # no step the bulb would stretch


def test_the_strobe_flashes_to_full_and_falls_back_to_the_level():
    f = strobe_flow(0, 100, 25, rate=2, flashes=3)
    assert f.steps[0] == (50, 1, 0xFF0000, 100)
    assert f.steps[1] == (450, 1, 0xFF0000, 25)
    assert flow_ms(f) == pytest.approx(1.5)


def test_white_flows_use_the_white_leds():
    assert swell_flow(0, 0, white=4000).steps[0][1:3] == (2, 4000)
    assert strobe_flow(0, 0, 10, white=6500).steps[0][1:3] == (2, 6500)


# ------------------------------------------------------------ the glue
from music_bulb import DropFx, SWELL_HOLD, White  # noqa: E402


def primed(event):
    fx = DropFx()
    fx.event = event
    return fx


def test_a_drop_sends_one_strobe_and_holds_the_loop_until_it_ends():
    fx = primed("drop")
    flow = fx.take(10.0, (200, 90, 30))
    assert flow.count == 8 and flow.steps[0][3] == 100 and flow.steps[1][3] == 30
    assert fx.take(10.1, (200, 90, 30)) is None           # once per event
    assert fx.holding(10.0 + flow_ms(flow) - 0.01)
    assert not fx.holding(10.0 + flow_ms(flow) + 0.01)
    assert fx.handback and fx.ramp(40) == DropFx.HANDBACK_MS and fx.ramp(40) == 40


def test_a_drop_cuts_a_swell_short_but_a_swell_never_cuts_a_strobe():
    fx = primed("build")
    assert fx.take(0.0, (10, 80, 40)).steps[0][0] == 2500
    assert fx.holding(SWELL_HOLD - 0.1)
    fx.event = "drop"
    assert fx.take(1.0, (10, 80, 40)).count == 8
    fx.event = "build"
    assert fx.take(1.2, (10, 80, 40)) is None


def test_switched_off_it_sends_nothing_and_forgets_the_event():
    fx = primed("drop")
    assert fx.take(0.0, (0, 100, 50), on=False) is None
    assert fx.take(0.1, (0, 100, 50)) is None and not fx.holding(0.1)


def test_on_a_white_scene_it_flashes_the_white_leds_at_their_matched_full():
    flow = primed("drop").take(0.0, White(3500, 12))
    assert flow.steps[0] == (50, 2, 3500, 40) and flow.steps[1][1:] == (2, 3500, 12)
