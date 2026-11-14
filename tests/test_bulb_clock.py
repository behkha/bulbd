"""Bulb-timed beats: the same predicted beats, but each goes out ~60 ms early
as one flow the bulb times itself, so the flash no longer waits for the
block nearest the beat. Off by default; on, it must keep every Beat sync
rule: one flash per beat, nothing past a beat after the music, colour steps
only on heard accents, and no send of ours inside the flow."""
import numpy as np
import pytest

from music_bulb import (BLOCK, BULB_AHEAD, LED_GAP, Analyser, BeatSync, ColorEngine, Flow,
                        LevelEngine, White, beat_flow, frame_step, hsv_rgb, same_light)
from signals import SR, run_music, stereo, sustained, track
from test_sync import LAND, LEAD, concat, landings


def run_timed(pcm, lead=LEAD, fps=25, colour=True):
    """music mode's loop with bulb-timed beats, as the desk runs it.
    -> dict: `flashes` (t the flash lands, predicted), `sends` (t, frame),
    `steps` (t of each colour step), `flows` (t sent, hold, Flow)."""
    an, lv = Analyser(), LevelEngine()
    ce = ColorEngine(None)
    sync = BeatSync(timed=True)
    flashes, sends, steps, flows = [], [], [], []
    last_sent, last, hold_until, prev = -1.0, None, -1.0, ce.target_hue
    for i in range(0, len(pcm) - BLOCK, BLOCK):
        f = an.feed(pcm[i:i + BLOCK])
        fired = sync.feed(f, lv, an.dt, lead, ce if colour else None)
        ce.maybe_switch(f["t"], f["bands"])
        hue, sat = ce.tick(an.dt, f["treble"], f["t"])
        if ce.target_hue != prev:
            steps.append(f["t"])
            prev = ce.target_hue
        now = f["t"]
        if now < hold_until:
            continue                               # the flow is the bulb's until it lands
        # a flow's wait changes nothing on the LEDs: the gap counts to its flash
        if now - last_sent < (LED_GAP - sync.hold if fired else 1.0 / fps):
            continue
        val = int(round(np.clip(4 + 96 * (lv.brightness ** 1.9), 1, 100)))
        hsv = (int(round(hue)) % 360, int(round(sat)), val)
        if fired:
            peak = (hsv[0], hsv[1], int(round(np.clip(4 + 96 * sync.flash ** 1.9, 1, 100))))
            flow = beat_flow(sync.hold, last or hsv, peak)
            flows.append((now, sync.hold, flow))
            sends.append((now, flow))
            flashes.append((now + sync.hold, True))
            hold_until = now + sync.hold + 0.05
            last_sent, last = now, peak
            continue
        if lv.flashed:
            flashes.append((now, False))
        if same_light(hsv, last, hue_tol=2):
            continue
        sends.append((now, hsv))
        last_sent, last = now, hsv
    return dict(flashes=flashes, sends=sends, steps=steps, flows=flows)


# ----------------------------------------------------------------- off
def test_off_by_default_and_off_is_unchanged():
    assert BeatSync().timed is False
    sync = BeatSync()
    f = {"t": 1.0, "beat": 0.5, "onset": False, "strength": 0.0, "low": 0.0, "flux": 0.0,
         "rms": 0.1, "energy": 0.5}
    sync.feed(f, LevelEngine(), BLOCK / SR, LEAD)
    assert sync.clock.ahead == 0.0 and sync.hold == 0.0 and sync.flash is None


# ----------------------------------------------------------------- timing
def test_the_flash_lands_on_the_beat_and_tighter_than_block_pacing():
    pcm, kicks = track(120, secs=30)
    block, _, _ = landings(run_music(pcm, lead=LEAD)["flashes"], kicks)
    timed, doubles, pred = landings(run_timed(pcm)["flashes"], kicks)
    assert abs(np.median(timed)) < 0.015
    assert doubles == 0 and pred > 0.95
    iqr = lambda e: np.percentile(e, 75) - np.percentile(e, 25)
    assert iqr(timed) <= iqr(block)
    assert np.std(timed) < np.std(block)


def test_each_flow_waits_48_to_72_ms_then_flashes_in_50():
    r = run_timed(track(120, secs=20)[0])
    assert r["flows"]
    for t, hold, flow in r["flows"]:
        assert BULB_AHEAD - BLOCK / SR / 2 - 1e-6 <= hold <= BULB_AHEAD + BLOCK / SR / 2 + 1e-6
        assert isinstance(flow, Flow) and flow.count == 2 and flow.action == 1
        assert flow.steps[0][0] == max(50, round(hold * 1000)) and flow.steps[1][0] == 50


def test_nothing_of_ours_is_sent_inside_a_flow_or_too_fast():
    r = run_timed(track(120, secs=20)[0])
    sends = r["sends"]
    for (t0, a), (t1, b) in zip(sends, sends[1:]):
        lands = t1 + (b.steps[0][0] / 1000 if isinstance(b, Flow) else 0.0)
        assert lands - t0 >= LED_GAP - 1e-9
        if isinstance(a, Flow):
            assert t1 >= t0 + (a.steps[0][0] + 50) / 1000 - BLOCK / SR
    span = sends[-1][0] - sends[0][0]
    assert len(sends) / span <= 30


@pytest.mark.parametrize("bpm", [80, 120, 150, 172])
@pytest.mark.parametrize("pattern", [{}, dict(snare=.6), dict(hat=1.0), dict(noise=.15)])
def test_one_flash_per_beat(bpm, pattern):
    pcm, kicks = track(bpm, secs=30, **pattern)
    errs, doubles, pred = landings(run_timed(pcm)["flashes"], kicks)
    assert doubles == 0
    assert pred > 0.9 and len(errs) >= 0.95 * sum(1 for k in kicks if k >= 8.0)


# ----------------------------------------------------------------- no phantoms
@pytest.mark.parametrize("tail", ["silence", "tone-440", "chord", "white"])
def test_predicted_flashes_stop_within_a_beat_of_the_music(tail):
    music, kicks = track(120, secs=20)
    rest = np.zeros(SR * 10) if tail == "silence" else sustained(tail, secs=10)
    pcm, _ = concat((music, kicks), (stereo(0.5 * rest), ()))
    r = run_timed(pcm)
    last, period = kicks[-1], 0.5
    after = [t for t, p in r["flashes"] if p and t > last + 0.05]
    beats = 1 if tail == "silence" else 2
    assert len(after) <= beats and all(t < last + beats * period for t in after)


@pytest.mark.parametrize("name", ["tone-55", "chord", "vibrato", "white", "pink"])
def test_sustained_sound_alone_never_predicts(name):
    r = run_timed(stereo(sustained(name, secs=20)))
    assert r["flows"] == []


# ----------------------------------------------------------------- colour
def test_colour_steps_are_still_the_heard_accents_and_land_with_a_flash():
    pcm, _ = track(120, secs=30)
    plain = run_music(pcm, lead=0.0)["steps"]
    r = run_timed(pcm)
    assert abs(len(r["steps"]) - len(plain)) <= 2
    sent = [t for t, *_ in r["flows"]]
    late = [t for t in r["steps"] if t > 10]
    assert late and all(min(abs(t - s) for s in sent) < 1e-6 for t in late)
    # the new colour shows only in the flash: the wait holds the old one
    for t, hold, flow in r["flows"]:
        if t in late:
            assert flow.steps[0][2] != flow.steps[1][2]


# ----------------------------------------------------------------- frames
def test_flow_steps_carry_colour_and_white():
    assert frame_step((0, 100, 40), 60) == (60, 1, 0xFF0000, 40)
    assert frame_step(White(4000, 20), 50) == (50, 2, 4000, 20)
    assert hsv_rgb(240, 100) == 0x0000FF and hsv_rgb(0, 0) == 0xFFFFFF


def test_bulb_clock_is_a_saved_setting_off_by_default():
    import music_bulb_dash as dash
    snap = dash.State().snapshot()
    assert snap["bulb_clock"] is False
    assert "bulb_clock" in dash.SAVED_KEYS and "bulb_clock" not in dash.LOOK_KEYS
    assert dash.clean_patch({"bulb_clock": True}, snap) == {"bulb_clock": True}
    assert dash.clean_patch({"bulb_clock": 1}, snap) == {}
