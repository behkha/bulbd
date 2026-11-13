"""Beat sync: a flash sent early lands on the beat; one flash per beat; no
phantom beats once the music stops or holds; a tempo change is followed; the
colour still follows what was heard. Landing = send + the Wi-Fi hop and half
the bulb's ramp, modelled as 60 ms."""
import numpy as np
import pytest

from music_bulb import BeatClock
from signals import SR, run_music, stereo, sustained, track

LAND = 0.060                  # one-way Wi-Fi (~30 ms) + half the 50 ms ramp, minus nothing
LEAD = 0.080                  # the desk's default


def landings(flashes, kicks, after=8.0, until=None, delay=LAND):
    """For each kick in range: the flashes landing within 25% of a period of
    it (the off-beat hats sit at 50%, and land 76 ms later than they sound).
    -> (errors of the nearest one, count of kicks with two or more, predicted share)"""
    errs, doubles, pred, n = [], 0, 0, 0
    for i, k in enumerate(kicks):
        if k < after or (until is not None and k > until):
            continue
        period = kicks[i] - kicks[i - 1] if i else kicks[1] - kicks[0]
        n += 1
        near = sorted(((t + delay - k, p) for t, p in flashes
                       if abs(t + delay - k) < 0.25 * period), key=lambda e: abs(e[0]))
        if len(near) > 1:
            doubles += 1
        if near:
            errs.append(near[0][0])
            pred += near[0][1]
    return np.array(errs), doubles, pred / max(n, 1)


def concat(*parts):
    """Join (pcm, kicks) pieces end to end. -> (pcm, kicks shifted to match)."""
    pcm, kicks, t0 = [], [], 0.0
    for p, k in parts:
        pcm.append(p)
        kicks += [x + t0 for x in k]
        t0 += len(p) / SR
    return np.concatenate(pcm), tuple(kicks)


# ----------------------------------------------------------------- timing
def test_compensation_moves_the_flash_onto_the_beat():
    pcm, kicks = track(120, secs=30)
    late, _, _ = landings(run_music(pcm, lead=0.0)["flashes"], kicks)
    synced, doubles, pred = landings(run_music(pcm, lead=LEAD)["flashes"], kicks)
    assert np.median(late) > 0.065                         # today: 15 ms to hear + 60 to land
    assert abs(np.median(synced)) < 0.015                  # with 80 ms lead: on the beat
    assert np.percentile(synced, 75) - np.percentile(synced, 25) < 0.025
    assert doubles == 0 and pred > 0.95


@pytest.mark.parametrize("lead", [0.03, 0.08, 0.15])
def test_the_lead_moves_the_flash_by_that_much(lead):
    pcm, kicks = track(120, secs=30)
    base, _, _ = landings(run_music(pcm, lead=0.0)["flashes"], kicks)
    moved, _, _ = landings(run_music(pcm, lead=lead)["flashes"], kicks)
    assert np.median(base) - np.median(moved) == pytest.approx(lead, abs=0.013)   # half a block


@pytest.mark.parametrize("bpm", [80, 120, 150, 172])
@pytest.mark.parametrize("pattern", [{}, dict(snare=.6), dict(hat=1.0), dict(noise=.15)])
def test_one_flash_per_beat(bpm, pattern):
    pcm, kicks = track(bpm, secs=30, **pattern)
    errs, doubles, pred = landings(run_music(pcm, lead=LEAD)["flashes"], kicks)
    assert doubles == 0
    assert pred > 0.9 and len(errs) >= 0.95 * sum(1 for k in kicks if k >= 8.0)


def test_off_beat_hats_still_flash_reactively_without_a_second_beat_flash():
    pcm, kicks = track(120, secs=30, hat=1.0)
    r = run_music(pcm, lead=LEAD)
    beat = 0.5
    off = [t for t, p in r["flashes"] if t > 8 and not p
           and min(abs(t - k - beat / 2) for k in kicks) < 0.07]
    assert len(off) > 20                                   # the hats are still there
    _, doubles, _ = landings(r["flashes"], kicks)
    assert doubles == 0


def test_fusion_gets_the_same_timing():
    pcm, kicks = track(120, secs=30)
    errs, doubles, pred = landings(run_music(pcm, lead=LEAD, colour=False)["flashes"], kicks)
    assert abs(np.median(errs)) < 0.015 and doubles == 0 and pred > 0.95


# ------------------------------------------------------ no phantom beats
@pytest.mark.parametrize("tail", ["silence", "tone-440", "chord", "white"])
def test_predicted_flashes_stop_within_a_beat_of_the_music(tail):
    """After the last kick at most one more beat is flashed early - the one
    that was already due. A sustained sound that starts on the grid (the
    chord here) is itself heard as one more beat, so it may allow one more."""
    music, kicks = track(120, secs=20)
    rest = np.zeros(SR * 10) if tail == "silence" else sustained(tail, secs=10)
    pcm, _ = concat((music, kicks), (stereo(0.5 * rest), ()))
    r = run_music(pcm, lead=LEAD)
    last, period = kicks[-1], 0.5
    after = [t for t, p in r["flashes"] if p and t > last + 0.05]
    beats = 1 if tail == "silence" else 2
    assert len(after) <= beats and all(t < last + beats * period for t in after)


@pytest.mark.parametrize("name", ["tone-55", "chord", "vibrato", "white", "pink"])
def test_sustained_sound_alone_never_predicts(name):
    r = run_music(stereo(sustained(name, secs=20)), lead=LEAD)
    assert [t for t, p in r["flashes"] if p] == []


def test_a_break_hands_back_to_reactive_and_relocks():
    a, ka = track(120, secs=15)
    b, kb = track(120, secs=15)
    pcm, kicks = concat((a, ka), (stereo(np.zeros(SR * 3)), ()), (b, kb))
    r = run_music(pcm, lead=LEAD)
    second = [k for k in kicks if k > 18]
    errs, doubles, pred = landings(r["flashes"], second, after=second[0] + 6)
    assert doubles == 0 and pred > 0.9


# ------------------------------------------------------------ tempo change
def test_a_tempo_change_is_followed():
    pcm, kicks = concat(track(120, secs=20), track(132, secs=25))
    r = run_music(pcm, lead=LEAD)
    errs, doubles, pred = landings(r["flashes"], kicks, after=35.0)
    assert doubles == 0
    assert pred > 0.8 and abs(np.median(errs)) < 0.02


def test_the_change_itself_falls_back_rather_than_flashing_off_beat():
    pcm, kicks = concat(track(120, secs=20), track(132, secs=25))
    r = run_music(pcm, lead=LEAD)
    stray = [t for t, p in r["flashes"] if p and 20 < t < 35
             and min(abs(t + LAND - k) for k in kicks) > 0.06]
    assert len(stray) <= 2


# ------------------------------------------------------------------ colour
def test_colour_steps_are_still_the_heard_accents():
    pcm, _ = track(120, secs=30)
    plain = run_music(pcm, lead=0.0)["steps"]
    synced = run_music(pcm, lead=LEAD)
    assert abs(len(synced["steps"]) - len(plain)) <= 2
    fires = [t for t, p in synced["flashes"] if p]
    late = [t for t, *_ in synced["steps"] if t > 10]
    assert late and all(min(abs(t - f) for f in fires) < 1e-6 for t in late)   # land with a flash


def test_no_colour_step_after_the_music_stops():
    music, kicks = track(120, secs=20)
    pcm, _ = concat((music, kicks), (stereo(np.zeros(SR * 8)), ()))
    steps = run_music(pcm, lead=LEAD)["steps"]
    assert [t for t, *_ in steps if t > kicks[-1] + 0.6] == []


def test_lead_zero_is_todays_behaviour():
    pcm, _ = track(120, secs=20)
    r = run_music(pcm, lead=0.0)
    assert not any(p for _, p in r["flashes"])
    clock = BeatClock()
    assert clock.frame({"t": 1.0, "beat": 0.5, "onset": True, "strength": 1.0,
                        "low": 1.0, "flux": 1.0}, 0.0) == (None, False)


def test_lead_is_a_clamped_saved_setting():
    import music_bulb_dash as dash
    assert "lead" in dash.SAVED_KEYS
    assert dash.clean_patch({"lead": 400}, {"monitors": []}) == {"lead": 250}
    assert dash.clean_patch({"lead": -5}, {"monitors": []}) == {"lead": 0}
    assert dash.clean_patch({"lead": 72.4}, {"monitors": []}) == {"lead": 72}
