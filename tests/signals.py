"""Synthetic audio and video for the offline tests, plus the loops that drive
the engines the way the real front ends do. Everything is seeded.
"""
import colorsys
from functools import lru_cache

import numpy as np

from music_bulb import (BLOCK, LED_GAP, SAMPLERATE, Analyser, BeatSync, ColorEngine, LevelEngine,
                        ScreenColorEngine, crop_letterbox, same_light)

SR = SAMPLERATE
GATE = 0.12          # LevelEngine ignores onsets weaker than this (0.30 in calm mode)
STRONG = 0.45        # ...and steps the colour on onsets stronger than this
TOL = 0.07           # a kick counts as caught when an onset lands within 70 ms of it


# ------------------------------------------------------------------ audio
def stereo(x):
    return np.stack([x, x], 1).astype(np.float32)


@lru_cache(maxsize=None)
def _drums(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(.18 * SR)) / SR
    kick = (np.sin(2 * np.pi * (55 + 60 * np.exp(-t * 30)) * t) * np.exp(-t * 22)).astype(np.float32)
    n = int(.04 * SR)
    hat = (rng.normal(0, 1, n) * np.exp(-np.arange(n) / (SR * .008))).astype(np.float32)
    n = int(.12 * SR)
    snare = (rng.normal(0, 1, n) * np.exp(-np.arange(n) / (SR * .03))).astype(np.float32)
    return kick, hat, snare


@lru_cache(maxsize=None)
def track(bpm, secs=40, kick=1.0, hat=0.25, snare=0.0, pad=0.05, noise=0.0, seed=0):
    """Four-on-the-floor kicks, hats on the off-beat, a snare on every second
    beat, a 220 Hz pad and an optional noise bed. -> (stereo pcm, kick times).
    The result is cached and shared: do not modify it."""
    rng = np.random.default_rng(seed)
    kk, hh, sn = _drums(seed)
    n = int(secs * SR)
    x = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm
    kicks = []
    i = 0
    while i * beat < secs - .3:
        s = int(i * beat * SR)
        x[s:s + len(kk)] += kick * kk
        kicks.append(i * beat)
        if hat:
            s2 = int((i + .5) * beat * SR)
            x[s2:s2 + len(hh)] += hat * hh
        if snare and i % 2 == 1:
            x[s:s + len(sn)] += snare * sn
        i += 1
    x += pad * np.sin(2 * np.pi * 220 * np.arange(n) / SR).astype(np.float32)
    if noise:
        x += noise * rng.normal(0, 1, n).astype(np.float32)
    pcm = stereo(0.5 * x)
    pcm.flags.writeable = False
    return pcm, tuple(kicks)


def sustained(name, secs=30):
    """Steady sound with no events in it."""
    n = SR * secs
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)
    if name.startswith("tone-"):
        return 0.3 * np.sin(2 * np.pi * float(name[5:]) * t)
    if name == "chord":
        return 0.15 * sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330))
    if name == "vibrato":                      # 440 Hz, +-6 Hz at 5.5 Hz
        return 0.3 * np.sin(2 * np.pi * 440 * t + (6 / 5.5) * np.sin(2 * np.pi * 5.5 * t))
    if name == "tremolo":                      # amplitude wobble, 20% at 5 Hz
        return 0.3 * np.sin(2 * np.pi * 330 * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))
    if name == "white":
        return 0.05 * rng.normal(0, 1, n)
    if name == "hiss":
        return 0.005 * rng.normal(0, 1, n)
    if name == "pink":
        return np.cumsum(rng.normal(0, 1, n)) * 0.0005 + 0.02 * rng.normal(0, 1, n)
    raise KeyError(name)


def analyse(pcm, calm=False):
    """Analyser + LevelEngine over a whole signal.
    -> (analyser, [(t, strength)] onsets, [(t, brightness)])"""
    an, lv = Analyser(), LevelEngine(calm=calm)
    onsets, bright = [], []
    for i in range(0, len(pcm) - BLOCK, BLOCK):
        f = an.feed(pcm[i:i + BLOCK])
        if f["onset"]:
            onsets.append((f["t"], f["strength"]))
        lv.feed(f, an.dt)
        bright.append((f["t"], lv.brightness))
    return an, onsets, bright


def caught(onsets, kicks, after=5.0, offset=0.0, gate=GATE):
    """-> (kicks caught, kicks expected) for kicks later than `after` s."""
    want = [k + offset for k in kicks if k + offset > after]
    got = sum(1 for k in want if any(abs(t - k) < TOL and s > gate for t, s in onsets))
    return got, len(want)


def run_music(pcm, palette=None, calm=False, fade=0.0, fps=25, lead=0.0, colour=True):
    """The audio loop of music mode without the audio device and the bulb.
    `lead` is beat sync in seconds (0 = off); colour=False runs it the way
    fusion mode does, brightness only.

    -> dict with `rows` (t, hue, sat, brightness, hit, beat, palette per
    frame), `onsets`, `steps` (t, from_hue, to_hue per colour step), `sends`
    (t, (h, s, v), sudden) as music mode would hand them to Sender, and
    `flashes` (t, predicted) for every frame that started a flash.
    """
    an, lv, ce = Analyser(), LevelEngine(calm=calm), ColorEngine(palette, calm=calm, fade=fade)
    sync = BeatSync()
    rows, onsets, steps, sends, flashes = [], [], [], [], []
    last_sent, last_hsv, prev_target = -1.0, None, ce.target_hue
    for i in range(0, len(pcm) - BLOCK, BLOCK):
        f = an.feed(pcm[i:i + BLOCK])
        if f["onset"]:
            onsets.append((f["t"], f["strength"]))
        fired = sync.feed(f, lv, an.dt, lead, ce if colour else None)
        if lv.flashed:
            flashes.append((f["t"], fired))
        ce.maybe_switch(f["t"], f["bands"])
        hue, sat = ce.tick(an.dt, f["treble"], f["t"])
        if ce.target_hue != prev_target:
            steps.append((f["t"], prev_target, ce.target_hue))
            prev_target = ce.target_hue
        rows.append((f["t"], hue, sat, lv.brightness, lv.hit, f["beat"], ce.pal))
        now = f["t"]                         # the loop is paced by record(): audio time == wall time
        if now - last_sent < (LED_GAP if fired else 1.0 / fps):
            continue
        val = int(round(np.clip(4 + 96 * (lv.brightness ** 1.9), 1, 100)))
        hsv = (int(round(hue)) % 360, int(round(sat)), val)
        if not fired and same_light(hsv, last_hsv, hue_tol=2):
            continue
        sends.append((now, hsv, lv.hit > 0.35 and ce.fade_t is None))
        last_sent, last_hsv = now, hsv
    return dict(rows=rows, onsets=onsets, steps=steps, sends=sends, flashes=flashes)


def hue_gap(a, b):
    """Signed shortest distance a - b on the colour wheel, in degrees."""
    return ((a - b + 180.0) % 360.0) - 180.0


# ------------------------------------------------------------------ video
H, W = 130, 200


def solid(rgb):
    return np.tile(np.array(rgb, dtype=np.float32), (H, W, 1))


def patch(bg, colour, frac):
    """`bg` with `colour` over the left `frac` of the columns."""
    f = solid(bg)
    f[:, :int(round(W * frac))] = colour
    return f


def to_bulb(hue, sat, val):
    """What the screen loop sends: integer hue, sat and value 1..100."""
    return round(hue) % 360, round(sat), int(round(float(np.clip(3 + 97 * val, 1, 100))))


def settle(frame, secs=4.0, fps=10):
    """Feed one frame for `secs` the way screen mode does.
    -> (bulb hsv, measured hue or None, share of light)"""
    eng = ScreenColorEngine()
    for _ in range(int(secs * fps)):
        hue_m, sat_m, luma = eng.measure(crop_letterbox(frame))
        hue, sat, val, _cut = eng.tick(1.0 / fps, hue_m, sat_m, luma)
    return to_bulb(hue, sat, val), hue_m, eng.share


def noisy_two_colour(rng, share=0.5):
    f = solid((1, .1, 0))
    f[:, int(W * share):] = (0, .3, 1)
    return np.clip(f + rng.normal(0, .03, f.shape).astype(np.float32), 0, 1)


def drift_frame(i):
    """Hue creeping 3 degrees a frame."""
    return solid(colorsys.hsv_to_rgb(((i * 3) % 360) / 360, 1, .8))
