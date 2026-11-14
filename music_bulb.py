#!/usr/bin/env python3
"""
music_bulb.py - v2 - make a Xiaomi Mi Smart LED Bulb (White & Color) react to audio.

Signal chain:
    PCM -> log-spaced bands -> percentile normalisation
        -> spectral-flux onset detection (adaptive threshold)
        -> tempo estimate (autocorrelation of the onset envelope)
        -> beat-synced hit envelopes + slow sustain bed
        -> palette state machine (steps on accents, switches on sections;
           --fade sweeps between anchors instead of snapping)
        -> gamma-corrected render -> non-blocking sender thread -> bulb

<bulb-ip> is whatever address your own router gave the bulb; there is no
default. The dashboard's cloud deck (music_bulb_dash.py) lists it along with
the token, which is the one route that works before the LAN API is open.
Afterwards `yeelight.discover_bulbs()` finds it too.

Setup (once, unlocks the LAN API on Mi-branded bulbs):
    python music_bulb.py --enable-lan --ip <bulb-ip> --token <32-hex>

Run:
    python music_bulb.py --ip <bulb-ip>
    python music_bulb.py --ip <bulb-ip> --palette lagoon --calm

Smooth colour mode - same beat-driven palette steps, but the hue sweeps
between two anchors over a fixed time instead of snapping to the next one:
    python music_bulb.py --ip <bulb-ip> --fade 1.5

Fusion mode - the screen picks the colour, the audio drives the brightness:
    python music_bulb.py --ip <bulb-ip> --fusion

Screen mode - ambient light that follows the dominant colour on screen,
for films rather than music:
    python music_bulb.py --ip <bulb-ip> --screen
    python music_bulb.py --ip <bulb-ip> --screen --monitor 2 --calm

    Screen frame -> letterbox crop -> weighted hue histogram
        -> dominant hue + saturation -> scene-cut detector
        -> time-smoothed render -> the same sender thread -> bulb

    macOS needs Screen Recording permission for whichever app runs this,
    otherwise every frame captures the desktop wallpaper instead.

Install:
    pip install yeelight numpy soundcard python-miio mss
"""

import argparse
import contextlib
import json
import math
import queue
import socket
import sys
import threading
import time
from collections import deque, namedtuple

import numpy as np

SAMPLERATE = 44100
BLOCK = 1024                      # ~23 ms analysis hop


# ---------------------------------------------------------------------------
# palettes - hue/saturation anchors, walked in order
# ---------------------------------------------------------------------------
PALETTES = {
    "ember":    [(6, 96), (26, 92), (44, 86), (350, 92)],
    "lagoon":   [(188, 90), (204, 84), (162, 82), (222, 92)],
    "orchid":   [(300, 90), (274, 94), (326, 84), (256, 92)],
    "citrus":   [(50, 92), (74, 84), (30, 96), (96, 80)],
    "nocturne": [(230, 92), (252, 88), (210, 94), (270, 82)],
    "peach":    [(14, 78), (34, 70), (340, 74), (8, 84)],
}


def hue_lerp(a, b, k):
    """Interpolate hue along the shorter arc of the wheel."""
    d = ((b - a + 180.0) % 360.0) - 180.0
    return (a + d * k) % 360.0


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
class Analyser:
    BAND_FLOOR = 0.02                 # -34 dB under a band's own recent peak
    MIX_FLOOR = 0.003                 # -50 dB under the loudest band's peak...
    MIX_FLOOR_LOW = 0.01              # ...-40 dB for the bands by DC (see feed)
    TEMPO_BANDS = 4                   # 40-390 Hz, where the beat lives

    def __init__(self, sr=SAMPLERATE, block=BLOCK, n_bands=10):
        self.dt = block / sr
        self.window = np.hanning(block).astype(np.float32)

        freqs = np.fft.rfftfreq(block, 1.0 / sr)
        edges = np.geomspace(40.0, 12000.0, n_bands + 1)
        self.masks = []
        for i in range(n_bands):
            m = (freqs >= edges[i]) & (freqs < edges[i + 1])
            if not m.any():                       # narrow low bands can be empty
                m = np.zeros_like(m)
                m[np.argmin(np.abs(freqs - edges[i]))] = True
            self.masks.append(m)
        self.n_bands = n_bands
        # Bands of one or two bins sit next to DC, where a steady note higher
        # up leaks in from both its positive and negative image. The two beat
        # against each other, so the band swings 10 dB with nothing playing
        # in it, and needs the higher floor.
        self.by_dc = np.array([m.sum() < 4 for m in self.masks])
        self.mix_floor = np.where(self.by_dc, self.MIX_FLOOR_LOW,
                                  self.MIX_FLOOR).astype(np.float32)

        # beats live low, so weight the flux sum towards the bottom
        self.flux_w = np.linspace(1.6, 0.5, n_bands).astype(np.float32)

        self.prev_log = np.zeros(n_bands, dtype=np.float32)
        self.recent = deque(maxlen=3)             # last band levels, ~70 ms
        self.prev_flux = 0.0
        self.peaks = np.full(n_bands, 1e-6, dtype=np.float32)   # slow-decay band peaks
        self.lag = None                           # autocorrelation peak in use
        self.blur = np.array([1, 2, 3, 2, 1], dtype=np.float32) / 9.0
        self.levels = deque(maxlen=220)           # ~5 s, for percentile AGC
        self.flux_hist = deque(maxlen=64)         # ~1.5 s, for the threshold
        self.env = deque(maxlen=180)              # ~4 s, for tempo
        self.lo = np.zeros(n_bands, dtype=np.float32)
        self.hi = np.ones(n_bands, dtype=np.float32)

        self.t = 0.0
        self.frame = 0
        self.last_onset = -10.0
        self.beat = 0.5                           # seconds per beat
        self.sensitivity = 1.4                    # onset threshold, in sigmas

    # -- tempo from the autocorrelation of the onset envelope ---------------
    def _update_tempo(self):
        if len(self.env) < self.env.maxlen:
            return
        x = np.fromiter(self.env, dtype=np.float32)
        # An onset is one frame wide and a beat is rarely a whole number of
        # frames, so without a blur the true lag falls between two samples
        # and loses to whichever multiple happens to land on one.
        x = np.convolve(x, self.blur, mode="same")
        x -= x.mean()
        n = len(x)
        # divide by the overlap, or short lags win just by having more of it
        ac = np.correlate(x, x, mode="full")[n - 1:] / np.arange(n, 0, -1)
        lo = max(1, int(round(0.33 / self.dt)))              # 180 BPM
        hi = min(n - 2, int(round(0.92 / self.dt)))          # 65 BPM
        if hi <= lo:
            return
        seg = ac[lo:hi + 1]
        top = float(seg.max())
        if top <= 0.0:
            return                                           # nothing periodic
        # A beat correlates at every multiple of itself, and a backbeat or a
        # loud off-beat makes two of those peaks nearly equal. Stay on the
        # peak chosen last time if it is still a strong one; otherwise take
        # the shortest strong one. Without the first rule the estimate flips
        # between two readings and, being slewed, settles on neither.
        peaks = [i for i in range(lo, hi + 1)
                 if ac[i] >= 0.75 * top and ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]]
        if not peaks:
            return
        cur = self.lag
        held = [i for i in peaks if abs(i - cur) <= 0.08 * cur] if cur else []
        k = min(held, key=lambda i: abs(i - cur)) if held else peaks[0]
        self.lag = k
        a, b, c = float(ac[k - 1]), float(ac[k]), float(ac[k + 1])
        curve = a - 2.0 * b + c
        lag = k + (0.5 * (a - c) / curve if curve < 0.0 else 0.0)   # sub-frame peak
        self.beat += (lag * self.dt - self.beat) * 0.15      # slew, don't jump

    def feed(self, pcm):
        self.t += self.dt
        self.frame += 1
        mono = pcm.mean(axis=1) if pcm.ndim > 1 else pcm
        rms = float(np.sqrt(np.mean(np.square(mono))))

        spec = np.abs(np.fft.rfft(mono * self.window))
        raw = np.array([spec[m].mean() for m in self.masks], dtype=np.float32)
        # Bands are compared as ratios, but each one only down to 34 dB
        # under its own recent peak, and none below 40-50 dB under the loudest.
        # Under those floors a band holds hiss and window leakage, and its
        # frame-to-frame wobble is as big, as a ratio, as a real hit. The
        # floor is per band because music tilts: a hi-hat sits tens of dB
        # under the kick and must still count.
        self.peaks = np.maximum(raw, self.peaks * 0.997)
        log = np.log(raw + self.BAND_FLOOR * self.peaks
                     + self.mix_floor * float(self.peaks.max()) + 1e-7)

        # --- onset: half-wave-rectified spectral flux vs adaptive threshold
        # The bands by DC have to rise above their last three frames, not
        # just the last one. A held bass note beats against its own mirror
        # image there and the band ripples every three or four frames;
        # against one previous frame each ripple reads as a rise, against the
        # recent maximum it does not. A kick clears either reference. The
        # other bands keep the single frame: with the maximum, steady noise
        # turns into rare spikes over a floor of zero, which is what beats
        # look like to the threshold below.
        ref = self.prev_log
        if self.recent:
            ref = np.where(self.by_dc, np.max(self.recent, axis=0), ref)
        rise = self.flux_w * np.maximum(log - ref, 0.0)
        flux = float(rise.sum())
        self.recent.append(log)
        self.prev_log = log
        self.flux_hist.append(flux)
        # tempo listens to the low bands only: an off-beat hi-hat is an onset
        # worth a flash, but it would make the half beat look like the beat
        self.env.append(float(rise[:self.TEMPO_BANDS].sum()))

        onset, strength = False, 0.0
        if len(self.flux_hist) == self.flux_hist.maxlen:
            h = np.fromiter(self.flux_hist, dtype=np.float32)
            med = float(np.median(h))
            # Three bars to clear. The usual spread; a multiple of the typical
            # frame, which steady noise never reaches however its spread works
            # out; and a floor, so a held note with no events has no beats.
            s = self.sensitivity
            thr = max(med + s * float(h.std()), med * (1.0 + 2.0 * s), 0.35 * s)
            rising = flux > self.prev_flux
            if flux > thr and rising and (self.t - self.last_onset) > 0.13:
                onset = True
                strength = float(np.clip((flux - thr) / (thr + 1e-6), 0.0, 1.0))
                self.last_onset = self.t
        self.prev_flux = flux

        if self.frame % 10 == 0:
            self._update_tempo()

        # --- per-band percentile normalisation (robust to quiet masters)
        self.levels.append(raw)
        if self.frame % 8 == 0 and len(self.levels) > 40:
            arr = np.asarray(self.levels)
            self.lo = np.percentile(arr, 15, axis=0).astype(np.float32)
            self.hi = np.percentile(arr, 92, axis=0).astype(np.float32)
        norm = np.clip((raw - self.lo) / np.maximum(self.hi - self.lo, 1e-9), 0, 1)

        return {
            "bands": norm,
            "low": float(self.env[-1]),           # the flux's rise in the beat bands
            "flux": flux,
            "bass": float(norm[:3].mean()),
            "mid": float(norm[3:7].mean()),
            "treble": float(norm[7:].mean()),
            "energy": float(np.clip(np.dot(norm, self.flux_w) / self.flux_w.sum(), 0, 1)),
            "onset": onset,
            "strength": strength,
            "rms": rms,
            "beat": self.beat,
            "t": self.t,
        }


# ---------------------------------------------------------------------------
# colour state machine
# ---------------------------------------------------------------------------
class ColorEngine:
    """Palette walker with two ways of getting from one anchor to the next.

    fade == 0 (default) is the punchy behaviour: an accent snaps most of the
    way to the next anchor at once, so the colour reads as a hit. fade > 0
    is the smooth mode - the same discrete anchors, but the hue sweeps the
    short arc between them over `fade` seconds on an eased ramp, so red to
    blue passes through the wheel instead of cutting. Steps are held apart
    by at least `fade` so one sweep lands before the next begins.
    """

    def __init__(self, palette=None, calm=False, fade=0.0, palettes=None):
        # every palette it may use, the user's own included; the desk swaps
        # this mapping when one is added, changed or deleted
        self.palettes = dict(palettes or PALETTES)
        self.pal = palette if palette in self.palettes else next(iter(self.palettes))
        self.locked = palette is not None
        self.calm = calm
        self.fade = float(fade)
        self.idx = 0
        h, s = self.palettes[self.pal][0]
        self.hue = self.target_hue = float(h)
        self.sat = self.target_sat = float(s)
        self.from_hue = self.hue
        self.from_sat = self.sat
        self.fade_t = None                        # start of the current sweep
        self.clock = 0.0
        self.last_step = 0.0
        self.last_switch = 0.0
        self.slow = None
        self.fast = None

    def retarget(self, h, s, t=None):
        """Aim at a new anchor; in smooth mode, start a timed sweep to it."""
        self.target_hue, self.target_sat = float(h), float(s)
        if self.fade > 0.0:
            self.from_hue, self.from_sat = self.hue, self.sat
            self.fade_t = self.clock if t is None else t

    def step(self, t, beat):
        """Move to the next palette anchor - at most once every two beats."""
        # 60 ms of slack: the hit two beats on arrives a frame early as often
        # as late, and an exact gap would turn every second one away
        gap = max(2.0 * beat - 0.06, 0.9) * (2.2 if self.calm else 1.0)
        if self.fade > 0.0:
            gap = max(gap, self.fade)             # let a sweep land first
        if t - self.last_step < gap:
            return
        anchors = self.anchors()
        self.idx = (self.idx + 1) % len(anchors)
        h, s = anchors[self.idx]
        self.retarget(h, s, t)
        if self.fade <= 0.0:
            # snap part of the way for punch, glide the remainder
            self.hue = hue_lerp(self.hue, self.target_hue,
                                0.25 if self.calm else 0.55)
        self.last_step = t

    def anchors(self):
        """The current palette's anchors; one deleted under it falls back
        to the first palette there is."""
        if self.pal not in self.palettes:
            self.pal, self.idx = next(iter(self.palettes)), 0
        return self.palettes[self.pal]

    def maybe_switch(self, t, bands):
        """Change palette when the spectral profile shifts - verse to chorus."""
        if self.locked:
            return None
        self.slow = bands.copy() if self.slow is None else self.slow + (bands - self.slow) * 0.004
        self.fast = bands.copy() if self.fast is None else self.fast + (bands - self.fast) * 0.06
        if t - self.last_switch < 24.0:
            return None
        if float(np.abs(self.fast - self.slow).mean()) < 0.17:
            return None
        choices = [n for n in self.palettes if n != self.pal]
        if not choices:
            return None
        self.pal = choices[int(t * 7) % len(choices)]
        self.idx = 0
        self.retarget(*self.palettes[self.pal][0], t=t)
        self.last_switch = t
        self.slow = bands.copy()
        return self.pal

    def tick(self, dt, treble, t=None):
        self.clock = self.clock + dt if t is None else float(t)
        if self.fade <= 0.0:
            self.fade_t = None

        if self.fade_t is not None:
            p = min((self.clock - self.fade_t) / self.fade, 1.0)
            p = p * p * (3.0 - 2.0 * p)               # ease both ends
            self.hue = hue_lerp(self.from_hue, self.target_hue, p)
            base = self.from_sat + (self.target_sat - self.from_sat) * p
            if p >= 1.0:
                self.fade_t = None
        else:
            self.hue = hue_lerp(self.hue, self.target_hue,
                                1.0 - math.exp(-dt / 0.16))
            base = self.target_sat

        want = base - 14.0 * treble                   # bleach a little on peaks
        self.sat += (want - self.sat) * (1.0 - math.exp(-dt / 0.25))
        return self.hue, float(np.clip(self.sat, 45.0, 100.0))


# ---------------------------------------------------------------------------
# audio -> brightness
# ---------------------------------------------------------------------------
class LevelEngine:
    """Analyser frames -> brightness 0..1, plus a flag on strong accents.

    Instant attack, timed release: the flash fills the beat and then clears,
    which reads as light rather than as a VU meter. Silence breathes on a
    7 s cycle so the bulb never looks broken. Music mode and fusion mode
    both drive brightness from this; only what the colour does differs.
    """

    def __init__(self, calm=False, punch=None):
        self.calm = calm
        self.punch = punch                # None = follow calm
        self.hit = 0.0
        self.sustain = 0.0
        self.brightness = 0.0
        self.silent_since = None
        self.flashed = False

    def feed(self, f, dt, predicted=None, claimed=False):
        """-> True when this frame is a strong accent (worth a colour step).

        `predicted` is the strength of a beat BeatClock flashed early on this
        frame; `claimed` says this frame's onset is that beat arriving, so it
        adds no second flash - though it still counts as an accent, because
        the colour follows what was really heard. `flashed` is True on a
        frame that started a flash."""
        punch = self.punch if self.punch is not None else (0.45 if self.calm else 0.78)
        gate = 0.30 if self.calm else 0.12
        accent = False
        self.flashed = False

        if f["rms"] < 0.004:
            self.silent_since = self.silent_since or f["t"]
        else:
            self.silent_since = None

        if self.silent_since and f["t"] - self.silent_since > 1.2:
            phase = 0.5 + 0.5 * math.sin(2 * math.pi * f["t"] / 7.0)
            level = 0.06 + 0.10 * phase
            self.hit = self.sustain = 0.0
        else:
            if predicted is not None and predicted > gate:
                self.hit = max(self.hit, 0.35 + 0.65 * predicted)
                self.flashed = True
            if f["onset"] and f["strength"] > gate:
                if not claimed:
                    self.hit = max(self.hit, 0.35 + 0.65 * f["strength"])
                    self.flashed = True
                accent = f["strength"] > 0.45
            # decay tied to tempo: the flash fills the beat, then clears
            tau = float(np.clip(0.42 * f["beat"], 0.09, 0.40))
            self.hit *= math.exp(-dt / (tau * (1.8 if self.calm else 1.0)))
            self.sustain += (f["energy"] - self.sustain) * (1.0 - math.exp(-dt / 0.35))
            level = float(np.clip(0.10 + 0.42 * self.sustain + punch * self.hit,
                                  0.0, 1.0))

        if level > self.brightness:
            self.brightness = level
        else:
            self.brightness += (level - self.brightness) * (1.0 - math.exp(-dt / 0.11))
        return accent


# ---------------------------------------------------------------------------
# beat prediction - send the flash early so it lands on the beat
# ---------------------------------------------------------------------------
class BeatClock:
    """Predicts the next beat once the tempo can be trusted.

    A flash sent when the beat is heard lands late: the onset is seen at the
    end of its 23 ms block, then the Wi-Fi hop and the bulb's 50 ms ramp
    follow. With a steady tempo the next beat is known in advance, so it can
    be flashed `lead` seconds before it would be detected.

    The grid is the last beat heard plus the period. Only onsets that are
    mostly low-band rise count as beats (a hat is an onset, not the beat);
    each one on the grid nudges it (phase and period, a small two-term loop)
    and lengthens the streak. A beat is predicted only after LOCK beats in a
    row, and only the one right after a beat that was heard: when the music
    stops or turns into a held passage, at most one predicted flash follows
    the last real beat, inside one period. Any onset that arrives within the
    window of a beat already flashed is that beat, and is claimed, so a beat
    gives one flash, never two. Off the grid, or with no lead, everything
    stays reactive, as before this existed.

    Times are the analyser's audio time, which the loops run in step with.
    Firing happens on the block nearest the target, so a flash is within
    half a block (11.6 ms) of where it was aimed; the network's own jitter
    (round trips of 23-106 ms measured) is several times that.
    """

    LOCK = 4                  # beats on the grid in a row before predicting
    WINDOW = 0.2              # share of a period around a grid beat that is it
    PHASE_GAIN = 0.35
    PERIOD_GAIN = 0.1
    LOW_SHARE = 0.4           # share of an onset's flux in the beat bands
    RESYNC = 0.08             # follow the analyser's tempo past this mismatch

    def __init__(self, dt=BLOCK / SAMPLERATE):
        self.dt = dt
        self.reset()

    def reset(self):
        self.last = None              # time of the last beat heard, on the grid
        self.period = None
        self.streak = 0
        self.fired = None             # grid time of the beat flashed early
        self.offgrid = 0
        self.strengths = deque(maxlen=6)

    @property
    def trusted(self):
        return self.streak >= self.LOCK and self.period is not None

    def frame(self, f, lead):
        """One analyser frame. -> (strength of a beat to flash now or None,
        whether this frame's onset is a beat already flashed)."""
        t, beat = f["t"], f["beat"]
        if lead <= 0.0:
            self.reset()
            return None, False
        if self.period is None or abs(self.period - beat) > self.RESYNC * beat:
            self.period = beat               # the analyser's tempo moved: take it

        claimed = False
        if f["onset"]:
            claimed = (self.fired is not None
                       and abs(t - self.fired) <= self.WINDOW * self.period)
            if f["low"] >= self.LOW_SHARE * max(f["flux"], 1e-9):
                self._heard(t, f["strength"])

        # a run of missed beats loses the grid
        if self.last is not None and t - self.last > 2.5 * self.period:
            self.streak = 0

        fire = None
        if self.trusted:
            nxt = self.last + self.period
            if self.fired != nxt and nxt - lead - 0.5 * self.dt <= t < nxt:
                self.fired = nxt
                fire = float(np.median(self.strengths)) if self.strengths else 0.5
        return fire, claimed

    def _heard(self, t, strength):
        if self.last is None:
            self.last, self.streak = t, 1
            return
        p = self.period
        k = round((t - self.last) / p)
        err = t - (self.last + k * p)
        if k >= 1 and abs(err) <= self.WINDOW * p:
            self.last = self.last + k * p + self.PHASE_GAIN * err
            self.period = p + self.PERIOD_GAIN * err / k
            self.streak = self.streak + 1 if k <= 2 else 1
            self.offgrid = 0
            self.strengths.append(strength)
        elif k >= 1 or abs(err) > self.WINDOW * p:
            # a beat-like onset off the grid: a syncopated kick is forgiven
            # once, two in a row mean the grid is wrong
            self.offgrid += 1
            if not self.trusted or self.offgrid >= 2:
                self.last, self.streak, self.offgrid = t, 1, 0
                self.strengths.clear()


class BeatSync:
    """What the audio loops (music and fusion) do with each analyser frame:
    BeatClock, then the brightness, then - in music mode - the colour.

    The colour rule does not move: a palette step is still triggered only by
    an accent that was really heard. What prediction changes is when it
    lands. An accent heard on a beat that was already flashed early arms the
    step, and the step goes out with the next predicted flash, on that beat,
    instead of trailing the flash it belongs to by the lead time. Without a
    trusted grid it steps at once, as it always did."""

    def __init__(self):
        self.clock = BeatClock()
        self.pending = False

    def feed(self, f, lvl, dt, lead, color=None):
        """-> True if this frame flashed a predicted beat."""
        fire, claimed = self.clock.frame(f, lead)
        accent = lvl.feed(f, dt, predicted=fire, claimed=claimed)
        if color is not None:
            if accent and claimed:
                self.pending = True
            elif accent:
                color.step(f["t"], f["beat"])
            if self.pending and (fire is not None or not self.clock.trusted):
                color.step(f["t"], f["beat"])
                self.pending = False
        return fire is not None and lvl.flashed


LED_GAP = 1.0 / 30                    # the fastest the LEDs can show a change


# ---------------------------------------------------------------------------
# screen capture - ambient colour that follows the display
# ---------------------------------------------------------------------------
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def rgb_to_hsv(rgb):
    """Vectorised RGB(0..1) -> (hue 0..360, sat 0..1, value 0..1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    d = mx - rgb.min(axis=-1)

    h = np.zeros_like(mx)
    nz = d > 1e-6
    ri = nz & (mx == r)
    gi = nz & (mx == g) & ~ri
    bi = nz & ~ri & ~gi
    h[ri] = ((g[ri] - b[ri]) / d[ri]) % 6.0
    h[gi] = (b[gi] - r[gi]) / d[gi] + 2.0
    h[bi] = (r[bi] - g[bi]) / d[bi] + 4.0

    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h * 60.0, s, mx


def crop_letterbox(rgb, thresh=0.06):
    """Trim the black bars around a film so they cannot dim the lamp."""
    luma = rgb @ _LUMA
    rows = luma.max(axis=1) > thresh
    cols = luma.max(axis=0) > thresh
    if not rows.any() or not cols.any():
        return rgb                                  # fully black frame
    r0 = int(np.argmax(rows))
    r1 = len(rows) - int(np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = len(cols) - int(np.argmax(cols[::-1]))
    return rgb[r0:r1, c0:c1]


def screen_permission():
    """True if this process may capture the screen, None if not macOS.

    Without the grant macOS hands back the desktop wallpaper instead of an
    error, so the lamp just sits on one colour forever - worth catching.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util
        cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
        cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        return bool(cg.CGPreflightScreenCaptureAccess())
    except Exception:
        return None


def monitor_box(monitor):
    """mss's {left, top, width, height} of a display, in global points."""
    import mss
    with mss.MSS() as sct:
        mons = sct.monitors
    if not 1 <= monitor < len(mons):
        raise RuntimeError(f"no monitor {monitor}; this Mac has "
                           f"{len(mons) - 1}. Try --list-monitors")
    return dict(mons[monitor])


def pace(grab, started, base, period, wake=None):
    """Wait from `started` until the next frame is due: not before `base`,
    and on a still picture until `period`. A grabber that pushes frames
    (a ScreenCaptureKit stream) ends the wait as soon as the picture
    changes. -> True if `wake` ended it - a control moved - so the caller
    can put its pacer back to the full rate. `wake` is cleared either way."""
    own = getattr(grab, "pace", None)
    if own is not None:
        return own(started, base, period, wake)
    slack = started + base - time.monotonic()
    if slack > 0:
        time.sleep(slack)
    woke = False
    left = started + period - time.monotonic()
    if left > 0:
        if wake is None:
            time.sleep(left)
        else:
            woke = wake.wait(left)
    if wake is not None:
        wake.clear()
    return woke


class ScreenGrabber:
    """Downsampled screen frames as float RGB in 0..1, one mss grab each."""

    pushes = False

    def __init__(self, monitor=1, region=None, width=192):
        import mss
        if region:
            x, y, w, h = region
            self.box = {"left": x, "top": y, "width": w, "height": h}
        else:
            self.box = monitor_box(monitor)
        self._sct = mss.MSS()
        # stride-sample on the way out: the grab is the cost, not the maths
        self.stride = max(1, int(self.box["width"] // max(width, 16)))

    def frame(self):
        a = np.asarray(self._sct.grab(self.box))            # BGRA
        a = a[::self.stride, ::self.stride, 2::-1]          # -> RGB
        return a.astype(np.float32) / 255.0

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass


# Which backend "auto" means. tools/bench_capture.py measured the two on the
# same film: the stream cost ~5% of a core over idle (Python, WindowServer,
# tccd, trustd, syspolicyd, replayd together), one mss grab per frame ~26%.
# Without pyobjc, off macOS, or if the stream fails, "auto" is still mss.
AUTO_CAPTURE = "sck"
CAPTURE_BACKENDS = ("auto", "sck", "mss")


class Capture:
    """The screen through a ScreenCaptureKit stream or mss, whichever is
    asked for and works - with ScreenGrabber's contract either way.

    `backend` is "auto" (AUTO_CAPTURE), "sck" or "mss". A stream that cannot
    start - no pyobjc, not macOS, an old macOS, a display it does not list -
    leaves mss in its place, and one that dies mid-session is swapped for
    mss on the next frame, so the session carries on. `backend` says which
    one is in use, `note` why the stream is not.
    """

    def __init__(self, monitor=1, region=None, width=192, fps=10, backend="auto",
                 stream=None):
        self.monitor, self.region, self.width = monitor, region, width
        self.backend, self.note = "mss", ""
        self._impl = None
        want = AUTO_CAPTURE if backend == "auto" else backend
        if want == "sck":
            try:
                if stream is None:
                    from sck_capture import StreamGrabber as stream
                self._impl = stream(monitor_box(monitor), region, width, fps)
                self.backend = "sck"
            except Exception as exc:
                self.note = f"{type(exc).__name__}: {exc}"
        if self._impl is None:
            self._impl = ScreenGrabber(monitor, region, width)

    @property
    def pushes(self):
        return self._impl.pushes

    @property
    def box(self):
        return self._impl.box

    def _fall_back(self, exc):
        with contextlib.suppress(Exception):
            self._impl.close()
        self.note = f"{type(exc).__name__}: {exc}"
        self._impl = ScreenGrabber(self.monitor, self.region, self.width)
        self.backend = "mss"

    def frame(self):
        try:
            return self._impl.frame()
        except Exception as exc:
            if self.backend != "sck":
                raise
            self._fall_back(exc)
            return self._impl.frame()

    def pace(self, started, base, period, wake=None):
        return pace(self._impl, started, base, period, wake)

    def set_fps(self, fps):
        setter = getattr(self._impl, "set_fps", None)
        if setter is not None:
            setter(fps)

    def close(self):
        self._impl.close()


class CapturePacer:
    """How long to wait before the next grab; backs off while the picture holds.

    A grab costs the same ~20 ms whatever its size, and most of that is
    WindowServer compositing plus a TCC permission check in system daemons,
    so the only real saving is making fewer of them. Each frame is compared
    with the last one that moved (not the previous one, so a slow fade still
    adds up to motion); once nothing has moved for a while the rate steps
    down, and the first frame that differs puts it straight back.
    """

    PIXEL = 0.04                      # per-channel change that counts for a pixel
    SHARE = 0.002                     # share of pixels that must change
    STEPS = ((12.0, 2.0), (1.5, 5.0))  # (seconds still, fps cap), longest first

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget the stillness - the next frames run at the full rate."""
        self.ref = None
        self.still_since = None

    def period(self, frame, fps, now):
        """-> seconds from this grab to the next one."""
        ref = self.ref
        if (ref is None or ref.shape != frame.shape
                or float((np.abs(frame - ref).max(axis=2) > self.PIXEL).mean())
                > self.SHARE):
            self.ref, self.still_since = frame, None
        elif self.still_since is None:
            self.still_since = now

        fps = max(float(fps), 1.0)
        if self.still_since is not None:
            held = now - self.still_since
            for after, cap in self.STEPS:
                if held >= after:
                    fps = min(fps, cap)
                    break
        return 1.0 / fps


def neutral_kelvin(rgb, s, v, limits=(2700, 6500)):
    """Colour temperature of the picture's neutral light, or None if none is lit.

    Lit pixels short of vivid (saturation under 0.3, so an accent does not
    pull it) are averaged in linear light and put through McCamy's
    approximation. A white page on an sRGB display is D65, 6500 K - the
    temperature bias lighting is set to - and a warm grey film scene comes
    out lower. Clamped to `limits`: past them the scene has a colour."""
    keep = (v > 0.09) & (s < 0.3)
    if int(keep.sum()) < 24:
        return None
    lin = np.where(rgb[keep] <= 0.04045, rgb[keep] / 12.92,
                   ((rgb[keep] + 0.055) / 1.055) ** 2.4)
    r, g, b = (float(c) for c in lin.mean(axis=0))
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    tot = X + Y + Z
    if tot <= 1e-9:
        return None
    x, y = X / tot, Y / tot
    if abs(0.1858 - y) < 1e-6:
        return None
    n = (x - 0.3320) / (0.1858 - y)
    cct = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
    return float(np.clip(cct, *limits))


# A frame for the white LEDs: colour temperature in kelvin and the bulb's
# brightness number. Colour frames stay plain (hue, sat, value) tuples.
White = namedtuple("White", "ct bright")

# The white LEDs give far more light than the RGB ones at the same brightness
# number (the RGB ones also lose the share a near-grey colour leaves dark), so
# a white frame is sent at this fraction of the colour frame's number to land
# at the same level and not pop at the switch. Fitted on the lamp: see
# tools/check_white.py.
WHITE_GAIN = 0.4


def white_frame(ct, level):
    """The White frame for a colour temperature and a 1..100 colour-scale
    level: kelvin to the nearest 50, brightness matched to the RGB LEDs."""
    return White(int(round(ct / 50.0)) * 50, max(1, int(round(level * WHITE_GAIN))))


def same_light(a, b, hue_tol=3):
    """True if frame `b` would look the same as `a` - the deadband that keeps
    sends the LEDs could not show off the wire. A switch between the white and
    the colour LEDs is never the same."""
    if a is None or b is None or isinstance(a, White) != isinstance(b, White):
        return False
    if isinstance(a, White):
        return a.bright == b.bright and abs(a.ct - b.ct) < 100
    return (abs(a[2] - b[2]) < 2 and abs(a[1] - b[1]) < 3
            and abs(((a[0] - b[0] + 180) % 360) - 180) < hue_tol)


def screen_frame(engine, hue, sat, level):
    """What the screen and fusion loops send for one tick of `engine`: a
    White frame while it calls the scene colourless, else (hue, sat, level).
    `level` is the 1..100 brightness on the colour LEDs' scale. `engine` may
    also be a ScreenColorThread, whose white() is its last published frame."""
    white, ct = engine.white() if callable(engine.white) else (engine.white, engine.ct)
    if white:
        return white_frame(ct, level)
    return (int(round(hue)) % 360, int(round(sat)), int(level))


SWITCH_MS = 300                       # ramp across a colour <-> white switch


def ramp_ms(frame, last, period):
    """The bulb-side ramp for a smooth send: one frame gap, so consecutive
    frames join up, but a switch between the colour and the white LEDs
    crossfades for longer - a 100 ms jump between two sets of LEDs reads as a
    pop even when their levels match."""
    ms = int(np.clip(period * 1000, 30, 150))
    if last is not None and isinstance(frame, White) != isinstance(last, White):
        return max(ms, SWITCH_MS)
    return ms


class ScreenColorEngine:
    """Dominant hue by weighted hue histogram, then smoothed over time.

    A plain mean over the frame turns any two-colour scene into grey, so
    vivid pixels vote into 10-degree hue bins and the winning bin's circular
    mean becomes the target. Unlike the music path this drifts continuously -
    an ambient lamp tracking a film should not step in palette jumps.
    """

    N_BINS = 36
    GREY_SAT = 18.0                   # saturation of a colourless scene
    FULL_SHARE = 0.2                  # share of the light that earns full colour
    MOVED = 0.12                      # per-channel change that counts for a pixel
    # A scene this close to colourless goes to the white LEDs, and comes back
    # to colour only well past it: the gap is what keeps a scene that sits on
    # the line from flapping between the two sets of LEDs. For a vivid
    # accent on grey: white while it carries under about 2.5% of the light,
    # colour again once it carries about 5%.
    WHITE_IN, WHITE_OUT = 28.0, 38.0
    # A picture whose main colour is this pale is near-colourless too - a
    # warm-graded grey, a cream wall - and reads truer as warm white than as
    # that hue pushed up to the colour floor. Entered under 0.20, left over 0.26.
    PALE_IN, PALE_OUT = 0.20, 0.26
    CT_RANGE = (2700, 6500)           # warm lamp light .. the display's own D65

    def __init__(self, min_sat=55.0, sat_boost=1.35, gamma=0.75, calm=False):
        self.min_sat = float(min_sat)
        self.sat_boost = float(sat_boost)
        self.gamma = float(gamma)
        self.calm = calm
        self.hue = 30.0
        self.sat = float(min_sat)
        self.val = 0.2
        self.prev_luma = None
        self.kern = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        self.hist = np.zeros(self.N_BINS, dtype=np.float32)   # last frame, for UI
        self._prev = None                 # last frame, to tell a cut from motion
        self.changed = 0.0                # share of pixels that moved since it
        self.share = 0.0                  # share of the light the hue carries
        self.white = False                # send this scene on the white LEDs
        self.pale = False                 # its colour is too pale to count
        self.ct = float(self.CT_RANGE[1])  # its colour temperature, kelvin
        self._ct_m = None                 # this frame's measured temperature

    def measure(self, rgb):
        """One frame -> (hue or None if colourless, sat 0..1, luma 0..1)."""
        prev, self._prev = self._prev, rgb
        self.changed = (
            float((np.abs(rgb - prev).max(axis=-1) > self.MOVED).mean())
            if prev is not None and prev.shape == rgb.shape else 0.0)

        h, s, v = rgb_to_hsv(rgb)
        luma = float((rgb @ _LUMA).mean())
        self.share = 0.0
        self._ct_m = neutral_kelvin(rgb, s, v, self.CT_RANGE)

        keep = (v > 0.09) & (s > 0.14)              # drop crushed and grey
        if int(keep.sum()) < 24:
            self.hist[:] = 0.0
            return None, 0.0, luma

        hk, sk, vk = h[keep], s[keep], v[keep]
        w = (sk ** 1.5) * vk                        # vivid and lit pixels win

        bins = (hk * (self.N_BINS / 360.0)).astype(np.int32) % self.N_BINS
        hist = np.bincount(bins, weights=w,
                           minlength=self.N_BINS).astype(np.float32)
        # wrap the ends before smoothing so red does not lose to the seam
        hist = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]),
                           self.kern, mode="valid")
        self.hist = hist
        top = int(np.argmax(hist))

        off = ((bins - top + self.N_BINS // 2) % self.N_BINS) - self.N_BINS // 2
        near = np.abs(off) <= 1
        ww = w[near]
        if float(ww.sum()) <= 0.0:
            return None, 0.0, luma

        # How much of the picture's light this colour is. A small vivid patch
        # on a neutral screen should tint the lamp, not paint it.
        self.share = float(vk[near].sum() / max(float(v.sum()), 1e-9))

        ang = np.radians(hk[near])
        hue = float(np.degrees(np.arctan2(float((np.sin(ang) * ww).sum()),
                                          float((np.cos(ang) * ww).sum()))) % 360.0)
        return hue, float((sk[near] * ww).sum() / ww.sum()), luma

    def tick(self, dt, hue_m, sat_m, luma):
        """-> (hue, sat 0..100, value 0..1, scene_cut)."""
        # A cut is a jump in brightness, or most of the picture changing at
        # once while the colour lands far from where the lamp is. The second
        # test catches a cut between two scenes that are equally bright; the
        # pixel share keeps it from firing when the picture holds and only
        # the winning hue bin flips.
        far = hue_m is not None and abs(
            ((hue_m - self.hue + 180.0) % 360.0) - 180.0) > 40.0
        cut = ((self.prev_luma is not None and abs(luma - self.prev_luma) > 0.16)
               or (self.changed > 0.5 and far))
        self.prev_luma = luma

        t_hue = 0.10 if cut else (0.90 if self.calm else 0.45)
        t_sat = 0.12 if cut else (1.10 if self.calm else 0.60)
        t_val = 0.08 if cut else (0.45 if self.calm else 0.22)

        if hue_m is not None:
            self.pale = sat_m < (self.PALE_OUT if self.pale else self.PALE_IN)
        if hue_m is None or self.pale:
            want_sat = self.GREY_SAT                # grey scene -> near white
            if hue_m is not None:
                self.hue = hue_m if cut else hue_lerp(
                    self.hue, hue_m, 1.0 - math.exp(-dt / t_hue))
        else:
            self.hue = hue_m if cut else hue_lerp(
                self.hue, hue_m, 1.0 - math.exp(-dt / t_hue))
            full = float(np.clip(sat_m * 100.0 * self.sat_boost,
                                 self.min_sat, 100.0))
            # fade towards grey as the colour's share of the light shrinks
            k = min(self.share / self.FULL_SHARE, 1.0)
            want_sat = self.GREY_SAT + (full - self.GREY_SAT) * k

        want_val = float(np.clip(luma, 0.0, 1.0)) ** self.gamma
        if cut:                                     # a cut lands, it does not ease
            self.sat, self.val = want_sat, want_val
        else:
            self.sat += (want_sat - self.sat) * (1.0 - math.exp(-dt / t_sat))
            self.val += (want_val - self.val) * (1.0 - math.exp(-dt / t_val))

        # white: the temperature follows the picture's neutral light on the
        # saturation's clock, and lands with it on a cut
        if self._ct_m is not None:
            self.ct = self._ct_m if cut else self.ct + (self._ct_m - self.ct) * (
                1.0 - math.exp(-dt / t_sat))
        self.white = self.sat < (self.WHITE_OUT if self.white else self.WHITE_IN)
        return self.hue, float(np.clip(self.sat, 0.0, 100.0)), self.val, cut


class ScreenColorThread(threading.Thread):
    """Screen capture + hue extraction on a thread of its own.

    Fusion mode has to service two clocks at once. The audio loop must call
    record() every ~23 ms or the input buffer backs up and onsets arrive
    late; a full-display grab costs ~26 ms, so doing both in one loop would
    guarantee that drift. The grab lives here instead and publishes the
    latest colour, which the audio loop samples whenever it is ready.

    `cut` is sticky: a scene cut raised between two reads is held until the
    reader takes it, so a cut can never fall into the gap between sends.

    The grab rate follows CapturePacer, so a still picture is read a few
    times a second instead of `fps` times. `poke()` ends that idle wait, and
    a stream (Capture on ScreenCaptureKit) ends it with a new picture.
    """

    def __init__(self, grab, engine, fps=10, on_frame=None):
        super().__init__(daemon=True)
        self.grab = grab
        self.engine = engine
        self.fps = fps
        self.on_frame = on_frame          # (frame, now) after each grab - Auto's motion
        self.alive = True
        self.error = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pacer = CapturePacer()
        self._hue = engine.hue
        self._sat = engine.sat
        self._val = engine.val
        self._white, self._ct = engine.white, engine.ct
        self._cut = False

    def run(self):
        last = time.monotonic()
        while self.alive:
            now = time.monotonic()
            dt, last = now - last, now
            try:
                frame = self.grab.frame()
                hue, sat, val, cut = self.engine.tick(
                    dt, *self.engine.measure(crop_letterbox(frame)))
            except Exception as exc:                # a display that went away
                self.error = exc
                self.alive = False
                return
            with self._lock:
                self._hue, self._sat, self._val = hue, sat, val
                self._white, self._ct = self.engine.white, self.engine.ct
                self._cut = self._cut or cut
            if self.on_frame is not None:
                self.on_frame(frame, now)

            setter = getattr(self.grab, "set_fps", None)
            if setter is not None:
                setter(self.fps)
            # idle stretch, interruptible by poke() or a new picture
            pace(self.grab, now, 1.0 / max(self.fps, 1),
                 self._pacer.period(frame, self.fps, now), self._wake)

    def read(self):
        """-> (hue, sat 0..100, value 0..1, cut) - clears the pending cut."""
        with self._lock:
            cut, self._cut = self._cut, False
            return self._hue, self._sat, self._val, cut

    def white(self):
        """-> (white, kelvin) as of the last frame read."""
        with self._lock:
            return self._white, self._ct

    def poke(self):
        """A setting changed: read the screen at the full rate again."""
        self._pacer.reset()
        self._wake.set()

    def close(self):
        self.alive = False
        self._wake.set()
        self.grab.close()


# ---------------------------------------------------------------------------
# auto mode - pick music, screen or fusion from what is happening
# ---------------------------------------------------------------------------
class MotionMeter:
    """Whether the picture is moving, from one frame every PROBE seconds.

    Two frames PROBE apart are compared; it is moving when more than SHARE
    of the pixels changed. A film or a video in a window clears that easily,
    a blinking cursor or a ticking clock does not. Frames in between are
    ignored, so the screen loops can feed it every frame for free and the
    probe in music mode needs only one grab per PROBE."""

    PROBE = 2.0
    SHARE = 0.01
    PIXEL = 0.04

    def __init__(self):
        self.ref, self.at = None, None

    def feed(self, frame, now):
        """-> True/False on a comparison, None between them."""
        if self.ref is not None and now - self.at < self.PROBE * 0.9:
            return None
        ref, self.ref, self.at = self.ref, frame, now
        if ref is None or ref.shape != frame.shape:
            return None
        return float((np.abs(frame - ref).max(axis=-1) > self.PIXEL).mean()) > self.SHARE


class AutoPicker:
    """Chooses the mode from sound and a moving picture.

        sound + moving picture  -> fusion
        sound only              -> music
        moving picture only     -> screen
        neither                 -> screen: a still picture is read twice a
                                   second and the lamp holds its colour;
                                   a paused film keeps its light
        no Screen Recording     -> music, always

    It has to hold still through the ordinary gaps. Sound counts after
    SOUND_ON seconds of it and is let go after SOUND_OFF of silence (the gap
    between two tracks is a few seconds). A picture counts as moving after
    two comparisons in a row say so, and as still after MOTION_OFF seconds
    without one (a paused film). A pick stays at least HOLD seconds, except
    the first, which lands after SETTLE seconds of watching.
    """

    SOUND_ON, SOUND_OFF = 2.0, 10.0
    MOTION_OFF = 15.0
    HOLD, SETTLE = 10.0, 3.0
    QUIET = 0.004                       # rms: LevelEngine's own silence

    def __init__(self, screen_ok=True, now=0.0, start=None):
        self.screen_ok = screen_ok
        self.mode = start or ("fusion" if screen_ok else "music")
        self.since = now - self.HOLD + self.SETTLE
        self.sound = self.moving = False
        self._heard = self._quiet = None
        self._runs, self._last_motion = 0, None
        self._lock = threading.Lock()

    def feed_sound(self, rms, now):
        with self._lock:
            if rms >= self.QUIET:
                self._quiet = None
                self._heard = self._heard if self._heard is not None else now
                if now - self._heard >= self.SOUND_ON:
                    self.sound = True
            else:
                self._heard = None
                self._quiet = self._quiet if self._quiet is not None else now
                if now - self._quiet >= self.SOUND_OFF:
                    self.sound = False

    def feed_motion(self, moving, now):
        if moving is None:
            return
        with self._lock:
            if moving:
                self._runs += 1
                self._last_motion = now
                if self._runs >= 2:
                    self.moving = True
            else:
                self._runs = 0
                if self._last_motion is None or now - self._last_motion >= self.MOTION_OFF:
                    self.moving = False

    def want(self):
        if not self.screen_ok:
            return "music"
        if self.sound:
            return "fusion" if self.moving else "music"
        return "screen"

    def pick(self, now):
        """-> the mode to run now."""
        with self._lock:
            want = self.want()
            if want != self.mode and now - self.since >= self.HOLD:
                self.mode, self.since = want, now
            return self.mode


# ---------------------------------------------------------------------------
# sender thread - keeps network latency out of the audio loop
# ---------------------------------------------------------------------------
class Sender(threading.Thread):
    """Puts frames on the wire off the audio and screen loops' clocks.

    A failed send ends the sender: the error is kept in `error`, `lost` is
    set, and whoever owns the bulb (Link) decides whether to reconnect. It
    used to retry forever on the same Bulb, whose socket yeelight had already
    thrown away, and each retry opened another connection to a bulb that was
    struggling.

    One slot holds the newest frame not yet sent (latest wins) and another
    a flow (an alert); the thread sleeps on a condition between them, so a
    stopped desk does not wake. While a flow runs the frame slot is kept,
    not sent: the flow ends with the bulb back on the last frame sent, then
    the newest frame put meanwhile goes out - or, if none was, the last one
    again, which also finishes a smooth ramp the flow cut short - and the
    frame after that waits LED_GAP."""

    HOLD_MARGIN = 0.08          # s: the flow starts ~half an RTT after it is sent

    def __init__(self, bulb):
        super().__init__(daemon=True)
        self.bulb = bulb
        self.alive = True
        self.lost = threading.Event()
        self.error = None
        self._cv = threading.Condition()
        self._frame = None                         # newest unsent (frame, sudden, duration)
        self._flow = None                          # (params, ms)
        self._last = None                          # the last frame on the wire
        self.hold_until = 0.0                      # frames wait until then
        self._gap_until = 0.0                      # spacing after a released frame

    def put(self, hsv, sudden, duration=40):
        """Queue one frame. `duration` is the bulb-side ramp for smooth sends;
        set it to the gap between frames so consecutive colours join up."""
        with self._cv:
            self._frame = (hsv, sudden, duration)  # drop the stale frame
            self._cv.notify()

    def flow(self, params, ms):
        """Run a `start_cf` flow (params = [count, action, expression]) that
        lasts `ms`, and hold the loop's frames until it has ended. With end
        action 0 the bulb itself returns to the last frame first, so the loop
        carries on where it was."""
        with self._cv:
            self._flow = (params, ms)
            self._cv.notify()

    def stop(self, timeout=2.0):
        """End the thread, waking it if it sleeps."""
        with self._cv:
            self.alive = False
            self._cv.notify()
        # wait out a send already in flight: a set_hsv landing after
        # release_bulb() would paint over the colour it just restored
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)

    def _next(self):
        """Wait for something to do. -> ("flow", (params, ms)), ("frame",
        item), or None to end. Called with the condition held."""
        while self.alive:
            if self._flow is not None:
                flow, self._flow = self._flow, None
                return "flow", flow
            now = time.monotonic()
            if now < self.hold_until:
                self._cv.wait(self.hold_until - now)
                continue
            if self.hold_until:                    # the flow is over: release
                self.hold_until = 0.0
                item = self._frame or self._last
                self._frame = None
                if item is not None:
                    self._gap_until = now + LED_GAP
                    return "frame", item
                continue
            if self._frame is not None:
                if now < self._gap_until:
                    self._cv.wait(self._gap_until - now)
                    continue
                item, self._frame = self._frame, None
                return "frame", item
            self._cv.wait()                        # no wake-ups between frames
        return None

    def run(self):
        while True:
            with self._cv:
                job = self._next()
            if job is None:
                break
            kind, what = job
            try:
                if kind == "flow":
                    params, ms = what
                    self.bulb.send_command("start_cf", params)
                    with self._cv:
                        self.hold_until = time.monotonic() + ms / 1000.0 + self.HOLD_MARGIN
                else:
                    send_frame(self.bulb, *what)
                    self._last = what
            except Exception as exc:
                self.error = exc
                self.lost.set()
                break


def send_frame(bulb, frame, sudden, duration):
    """One frame onto a bulb in music mode.

    Colour goes through yeelight's set_hsv, which becomes a one-step flow in
    RGB mode. A White frame is the same flow in colour-temperature mode (2);
    `sudden` is the 50 ms step yeelight uses for colour."""
    if isinstance(frame, White):
        ms = 50 if sudden else max(int(duration), 30)
        bulb.send_command("start_cf", [1, 1, f"{ms}, 2, {frame.ct}, {frame.bright}"])
        return
    h, s, v = frame
    bulb.set_hsv(h, s, v, effect="sudden" if sudden else "smooth",
                 duration=0 if sudden else duration)


# What the lamp was showing before a session took it, so stopping can put it
# back. The exchange is our own rather than Bulb.send_command, which loops
# forever on a socket the bulb has closed cleanly (recv keeps returning b""),
# and nothing may hang a stop. It rides on the session's own connections
# (pass `sock`), because the bulb has only a handful of slots and frees a
# closed one slowly: a read and a restore should not cost two more.
LAMP_PROPS = ("power", "bright", "color_mode", "rgb", "ct", "hue", "sat")
LAMP_PORT = 55443


def lamp_call(ip, commands, timeout=2.0, connect=socket.create_connection,
              sock=None):
    """Send [(method, params), ...] in order and return each command's
    result list. Raises on the first error; the whole exchange is bounded by
    `timeout`. With `sock` it uses that open connection and leaves it open
    for its owner; without, it opens one of its own and always closes it."""
    deadline = time.monotonic() + timeout
    own = sock is None
    if own:
        sock = connect((ip, LAMP_PORT), timeout)
    before = None if own else sock.gettimeout()
    try:
        buf = b""
        results = []
        for n, (method, params) in enumerate(commands, 1):
            sock.sendall((json.dumps({"id": n, "method": method,
                                      "params": params},
                                     separators=(",", ":")) + "\r\n").encode())
            # This firmware does not act on a command until another byte
            # follows it: without this the bulb accepts the connection and
            # says nothing, which reads exactly like wedged slots. yeelight
            # sends the same lone space (its issue #61).
            sock.sendall(b" ")
            while True:
                line, sep, rest = buf.partition(b"\r\n")
                if not sep:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError(f"{method}: no answer")
                    sock.settimeout(left)
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError(f"{method}: bulb closed the connection")
                    buf += chunk
                    continue
                buf = rest
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("method") == "props":       # a state push, not our reply
                    continue
                if msg.get("id") != n:
                    continue
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                results.append(msg.get("result"))
                break
        return results
    finally:
        try:
            if own:
                sock.close()
            else:
                sock.settimeout(before)
        except Exception:
            pass


def read_lamp(ip, **kw):
    """The lamp's power, brightness and colour as ints (power as "on"/"off"),
    or None if the bulb did not say."""
    (values,) = lamp_call(ip, [("get_prop", list(LAMP_PROPS))], **kw)
    raw = dict(zip(LAMP_PROPS, values or []))

    def num(key):
        try:
            return int(raw.get(key))
        except (TypeError, ValueError):
            return None

    if raw.get("power") not in ("on", "off"):
        return None
    saved = {"power": raw["power"]}
    saved.update({k: num(k) for k in LAMP_PROPS[1:]})
    return saved


def restore_commands(saved, ms=300):
    """The commands that put `saved` back: colour first (so a lamp that was
    off comes back on in its own colour next time), then brightness, then
    power off if it was off."""
    cmds = []
    mode = saved.get("color_mode")
    if mode == 1 and saved.get("rgb") is not None:
        cmds.append(("set_rgb", [max(0, min(saved["rgb"], 0xFFFFFF)), "smooth", ms]))
    elif mode == 2 and saved.get("ct"):
        cmds.append(("set_ct_abx", [max(1700, min(saved["ct"], 6500)), "smooth", ms]))
    elif mode == 3 and saved.get("hue") is not None:
        cmds.append(("set_hsv", [saved["hue"] % 360,
                                 max(0, min(saved.get("sat") or 0, 100)), "smooth", ms]))
    if saved.get("bright"):
        cmds.append(("set_bright", [max(1, min(saved["bright"], 100)), "smooth", ms]))
    if saved.get("power") == "off":
        cmds.append(("set_power", ["off", "smooth", ms]))
    return cmds


def _stage(exc, name):
    """Tag an exception with the step of open_bulb() it came from, which is
    what classify() reads to explain it."""
    if getattr(exc, "bulbd_stage", None) is None:
        try:
            exc.bulbd_stage = name
        except Exception:
            pass
    return exc


def _harden(sock):
    """Make a dead music connection fail within seconds, not minutes.

    A send to a bulb that lost power sits in the kernel's buffer while TCP
    retransmits for minutes; nothing raises, so nothing reconnects. A send
    timeout and (on macOS) a cap on how long unacknowledged data may wait
    turn that into an error the sender sees."""
    try:
        sock.settimeout(2.0)
        if sys.platform == "darwin":
            sock.setsockopt(socket.IPPROTO_TCP, 0x80, 6)     # TCP_RXT_CONNDROPTIME
        elif hasattr(socket, "TCP_USER_TIMEOUT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 6000)
    except Exception:
        pass


def open_bulb(ip, attempts=3, on_attempt=None, remember=True, memo=None):
    """Connect and enter music mode, retrying a bulb that is still busy.

    A color5 accepts only a handful of LAN connections at once, so a slot
    freed a moment ago may not be free yet; each failed attempt is released
    before the next one rather than left dangling. Retries take a few
    seconds, so `on_attempt(n, total, error)` reports progress - without it
    a caller showing "connecting" just looks hung.

    Every attempt first asks the lamp for its state with lamp_call(), on the
    connection the session is about to use: bounded, where yeelight's own
    first command can wait forever on a bulb that closed the socket. With
    `remember` the answer is kept on the bulb as `bulbd_saved`, for
    release_bulb(bulb, restore=True). It is kept once: a retry must not
    mistake what a failed attempt left behind for the lamp's own look. Pass a
    dict as `memo` to carry that first reading across separate calls (Link
    does). An answer that makes no sense only means nothing is restored.

    A failure is raised as it came, tagged `bulbd_stage` = connect, query
    or music, for classify().
    """
    from yeelight import Bulb

    memo = {} if memo is None else memo
    last = None
    for i in range(attempts):
        if on_attempt:
            on_attempt(i + 1, attempts, last)
        bulb = Bulb(ip, effect="sudden", duration=0, auto_on=True)
        bulb.bulbd_saved = None
        try:
            try:
                sock = bulb._socket
            except Exception as exc:
                raise _stage(exc, "connect")
            try:
                # a bulb that will not answer this will not answer turn_on
                # either, so a timeout here fails the attempt like any other
                reading = read_lamp(ip, sock=sock)
            except Exception as exc:
                raise _stage(exc, "query")
            if remember and "saved" not in memo:
                memo["saved"] = reading
            bulb.bulbd_saved = memo.get("saved") if remember else None
            try:
                bulb.turn_on()
                bulb.start_music()
            except Exception as exc:
                raise _stage(exc, "music")
            _harden(getattr(bulb, "_Bulb__socket", None) or sock)
            return bulb
        except Exception as exc:
            last = exc
            release_bulb(bulb)
            if i < attempts - 1:
                time.sleep(1.5 * (i + 1))
    raise last


def release_bulb(bulb, restore=False):
    """Hand the bulb's LAN connection back, and optionally its old look.

    yeelight.Bulb holds one persistent socket and never closes it on error;
    stop_music() closes it but then sends `set_music 0`, which lazily opens
    a *fresh* socket that leaks if that send also fails. Every leaked socket
    burns one of the bulb's few connection slots until it times them out
    minutes later, so a few failed sessions lock the bulb out entirely.

    `restore=True` puts back what open_bulb() read (a dict from read_lamp()
    works too). It goes out on the connection stop_music() has just opened,
    before that is closed, so stopping costs the bulb no extra slot. A
    failure is reported, never raised: stopping always finishes.
    """
    saved = getattr(bulb, "bulbd_saved", None) if restore is True else restore
    try:
        bulb.stop_music()
    except Exception:
        pass
    sock = getattr(bulb, "_Bulb__socket", None)

    restored = False
    if saved:
        try:
            if sock is not None:
                lamp_call(bulb._ip, restore_commands(saved), sock=sock)
            else:
                # stop_music() never reached the bulb: one bounded try on a
                # fresh connection, which also has to end music mode
                lamp_call(bulb._ip, [("set_music", [0])] + restore_commands(saved))
            restored = True
        except Exception as exc:
            print(f"could not restore the lamp: {exc}", file=sys.stderr)

    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
        try:
            bulb._Bulb__socket = None
        except Exception:
            pass
    return restored


def abandon_bulb(bulb):
    """Close a dropped link's socket without sending anything.

    release_bulb() would send `set_music 0` on a fresh connection, which to a
    bulb that has just vanished is a 5 s connect timeout and, to one that is
    struggling, one more slot taken."""
    sock = getattr(bulb, "_Bulb__socket", None)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
    try:
        bulb._Bulb__socket = None
    except Exception:
        pass


# ---------------------------------------------------------------------------
# finding the bulb again - the miio hello
# ---------------------------------------------------------------------------
# Every miio device answers a 32-byte hello on UDP 54321 with its device id,
# even during a music session, and it costs no TCP slot. SSDP discovery
# returns nothing on the Mac this was written on; the hello to the subnet's
# broadcast address is answered (255.255.255.255 is refused by macOS).
MIIO_PORT = 54321
HELLO = bytes.fromhex("21310020" + "ff" * 28)


def _udp():
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def hello_id(data):
    """The device id in a hello answer, or None if `data` is not one."""
    if len(data) < 32 or data[:2] != b"\x21\x31":
        return None
    return int.from_bytes(data[8:12], "big")


def _collect(sock, deadline, want=None):
    """Read hello answers until `deadline`. -> {address: device id}, or as
    soon as `want` (a device id) answers, {its address: want}."""
    seen = {}
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return seen
        try:
            sock.settimeout(left)
            data, addr = sock.recvfrom(1024)
        except OSError:                            # timeout, or the net went away
            return seen
        did = hello_id(data)
        if did is None:
            continue
        seen[addr[0]] = did
        if want is not None and did == want:
            return {addr[0]: did}


def miio_hello(ip, timeout=0.8, tries=2, udp=_udp):
    """Ask `ip` for its miio device id. -> int, or None if nothing answered."""
    try:
        s = udp()
    except OSError:
        return None
    try:
        for _ in range(tries):
            s.sendto(HELLO, (ip, MIIO_PORT))
            seen = _collect(s, time.monotonic() + timeout)
            if ip in seen:
                return seen[ip]
        return None
    except OSError:
        return None
    finally:
        s.close()


def local_net(near_ip, udp=_udp):
    """(this machine's address towards `near_ip`, that LAN's broadcast).

    A connected UDP socket sends nothing but tells which interface the route
    uses. The broadcast address is read from ifconfig where it can be, else
    taken as the /24's - which is what a home router hands out."""
    s = udp()
    try:
        s.connect((near_ip, MIIO_PORT))
        local = s.getsockname()[0]
    finally:
        s.close()
    bcast = None
    try:
        import subprocess
        out = subprocess.run(["ifconfig"], capture_output=True, text=True,
                             timeout=1.0).stdout
        import re
        m = re.search(r"inet %s\b[^\n]*?broadcast (\d+\.\d+\.\d+\.\d+)"
                      % re.escape(local), out)
        bcast = m.group(1) if m else None
    except Exception:
        pass
    return local, bcast or ".".join(local.split(".")[:3] + ["255"])


def find_bulb(did, near_ip, timeout=1.5, sweep=True, udp=_udp, net=local_net):
    """Where the device with miio id `did` is now, or None.

    One hello to the broadcast address of the LAN `near_ip` was on; if no
    answer carries the id, one unicast hello to every other address of that
    /24, paced over about a quarter of a second, for routers that drop
    broadcasts. Bounded: a few seconds at most, and nothing here opens a TCP
    connection."""
    try:
        local, bcast = net(near_ip, udp)
        s = udp()
    except OSError:
        return None
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(HELLO, (bcast, MIIO_PORT))
        for addr, got in _collect(s, time.monotonic() + timeout, want=did).items():
            if got == did:
                return addr
        if not sweep:
            return None
        prefix = local.rsplit(".", 1)[0]
        for n in range(1, 255):
            addr = f"{prefix}.{n}"
            if addr != local:
                s.sendto(HELLO, (addr, MIIO_PORT))
                if n % 32 == 0:
                    time.sleep(0.03)
        for addr, got in _collect(s, time.monotonic() + timeout, want=did).items():
            if got == did:
                return addr
        return None
    except OSError:
        return None
    finally:
        s.close()


# ---------------------------------------------------------------------------
# why it cannot connect, in plain words
# ---------------------------------------------------------------------------
class BulbUnreachable(RuntimeError):
    """The bulb could not be opened, with the reason a person can act on.

    `kind` is one of no_device, not_bulb, lan_off, busy, blocked, other;
    `action` is "setup" when the fix is in the desk's Setup deck."""

    def __init__(self, kind, ip, detail="", searched=False, lost=False):
        self.kind, self.ip, self.detail, self.searched = kind, ip, detail, searched
        self.action = "setup" if kind in ("not_bulb", "lan_off") else None
        text = {
            "no_device": f"Nothing answers at {ip}"
                         + (", and the bulb is not answering anywhere else on "
                            "this network" if searched else "")
                         + ". Check that it has power.",
            "not_bulb": f"{ip} is another device, not the bulb. Find the "
                        f"bulb's address in Setup.",
            "lan_off": "The bulb is there, but LAN control is off. Turn it on "
                       "with Enable LAN in Setup.",
            "busy": "The bulb is there but takes no more connections. Wait a "
                    "minute and press Start; if that fails, switch it off and "
                    "on at the wall.",
            "blocked": "The bulb could not connect back to this computer, which "
                       "music mode needs. Allow incoming connections for Python "
                       "in the firewall.",
        }.get(kind, f"Could not open the bulb at {ip}: {str(detail).rstrip('.')}.")
        if lost:
            text = "Lost the bulb and could not get it back. " + text
        super().__init__(text)


def classify(exc, answered, known):
    """Which BulbUnreachable kind a failed open_bulb() was.

    `answered`: a miio hello came back from the address. `known`: it came
    back with the id of a bulb this desk has opened before, so LAN control
    was on. A refused port on a bulb that has worked is far more likely full
    slots than a switched-off API; on a bulb never opened, the reverse."""
    stage = getattr(exc, "bulbd_stage", None)
    if stage == "connect":
        if isinstance(exc, ConnectionRefusedError):
            if not answered:
                return "not_bulb"
            return "busy" if known else "lan_off"
        return "busy" if answered else "no_device"
    if stage == "query":
        return "busy"
    if stage == "music":
        return "blocked" if isinstance(exc, (socket.timeout, TimeoutError)) else "busy"
    return "other"


class Link:
    """One bulb from Start to Stop: open it, notice a drop, get it back.

    Every connection still goes through open_bulb() and release_bulb(). Each
    try starts with a miio hello (UDP, no slot). If the address no longer
    answers with this bulb's id, find_bulb() looks for the id on the LAN and
    the link moves there; `moved(ip, did)` tells the owner, which saves it.
    The id is learned on the first successful open. A failure is raised as
    BulbUnreachable, already explained.

    A first open retries only what time can fix (busy slots). After a drop,
    recover() waits RETRY seconds before each try and returns False the
    moment `stopped` is set. Nothing here adds a connection to a healthy
    session, and the waits keep a struggling bulb from being hammered.

    `status(word, notice)` reports progress: a short word for the top bar
    and None or (kind, text) for the note.
    """

    START_WAITS = (1.5, 3.0)                 # between the tries of a first open
    RETRY = (1, 2, 4, 8, 15, 30)             # before each try after a drop

    def __init__(self, ip, did=None, remember=True, status=None, stopped=None,
                 moved=None):
        self.ip, self.did, self.remember = ip, did, remember
        self.status = status or (lambda word, notice=None: None)
        self.stopped = stopped or threading.Event()
        self.moved = moved or (lambda ip, did: None)
        self.bulb = None
        self.sender = None
        self.memo = {}                       # the lamp's look, read once at Start
        self.moved_note = None

    # -- the pieces, looked up late so tests can swap them ----------------
    def _hello(self, ip):
        return miio_hello(ip)

    def _find(self, did, ip):
        return find_bulb(did, ip)

    def _open(self, ip):
        return open_bulb(ip, attempts=1, remember=self.remember, memo=self.memo)

    # ---------------------------------------------------------------------
    @property
    def saved(self):
        return self.memo.get("saved")

    @property
    def lost(self):
        return self.sender is not None and self.sender.lost.is_set()

    def put(self, *a, **kw):
        if self.sender is not None:
            self.sender.put(*a, **kw)

    def flow(self, params, ms):
        """An alert on the session's own connection. -> False with no sender."""
        sender = self.sender
        if sender is None or sender.lost.is_set():
            return False
        sender.flow(params, ms)
        return True

    def _try(self):
        """One hello, maybe a search, one open. -> bulb or BulbUnreachable."""
        self.moved_note = None
        seen = self._hello(self.ip)
        if self.did is not None and seen != self.did:
            gone = seen is None
            self.status("searching", ("warn", (
                f"Nothing answers at {self.ip}." if gone
                else f"{self.ip} is another device now.")
                + " Looking for the bulb on this network."))
            found = self._find(self.did, self.ip)
            if not found:
                raise BulbUnreachable("no_device" if gone else "not_bulb",
                                      self.ip, searched=True)
            seen = self.did
            if found != self.ip:
                old, self.ip = self.ip, found
                self.moved(found, self.did)
                self.moved_note = ("ok", f"The bulb moved from {old} to {found}.")
            # found at its own address: the first hello was simply lost, which
            # this Wi-Fi does often enough - carry on rather than call it gone
            self.status("connecting", self.moved_note)
        try:
            bulb = self._open(self.ip)
        except Exception as exc:
            known = self.did is not None and seen == self.did
            raise BulbUnreachable(classify(exc, seen is not None, known),
                                  self.ip, detail=exc) from exc
        if seen is not None and seen != self.did:
            self.did = seen
            self.moved(self.ip, seen)
        return bulb

    def _start(self, bulb):
        bulb.bulbd_saved = self.saved
        self.bulb = bulb
        self.sender = Sender(bulb)
        self.sender.start()

    def _wait(self, secs, tick=None):
        """Sleep `secs` unless stopped; `tick(left)` once a second. -> stopped."""
        end = time.monotonic() + secs
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return self.stopped.is_set()
            if tick:
                tick(int(math.ceil(left)))
            if self.stopped.wait(min(left, 1.0)):
                return True

    def open(self):
        """Connect for a new session. -> True, or False if stopped first."""
        waits = (0.0,) + tuple(self.START_WAITS)
        for i, wait in enumerate(waits):
            if wait and self._wait(wait):
                return False
            if self.stopped.is_set():
                return False
            self.status("connecting" if i == 0 else f"connecting {i + 1}/{len(waits)}")
            try:
                bulb = self._try()
            except BulbUnreachable as exc:
                if exc.kind not in ("busy", "other") or i == len(waits) - 1:
                    raise
                continue
            self._start(bulb)
            return True
        return False                                          # pragma: no cover

    def recover(self):
        """After a drop: let go of the dead link, then try again on RETRY.
        -> True when back, False when stopped. Raises BulbUnreachable when
        every try failed."""
        self._drop()
        last, n = None, len(self.RETRY)
        for i, wait in enumerate(self.RETRY, 1):
            word = f"reconnecting {i}/{n}"
            if self._wait(wait, lambda left: self.status(word, (
                    "warn", f"Lost the bulb. Trying again in {left} s."))):
                return False
            self.status(word, ("warn", "Lost the bulb. Trying again now."))
            try:
                bulb = self._try()
            except BulbUnreachable as exc:
                last = exc
                continue
            self._start(bulb)
            self.status("connected", self.moved_note)
            return True
        raise BulbUnreachable(last.kind, last.ip, detail=last.detail,
                              searched=last.searched, lost=True)

    def _drop(self):
        if self.sender is not None:
            self.sender.stop(timeout=2.5)
            self.sender = None
        bulb, self.bulb = self.bulb, None
        if bulb is not None:
            abandon_bulb(bulb)

    def close(self, restore=False):
        """End the session: put the lamp back if asked and it can be reached.
        -> True if it was restored."""
        lost = self.lost
        if self.sender is not None:
            self.sender.stop()
            self.sender = None
        bulb, self.bulb = self.bulb, None
        if bulb is None:
            return False
        if lost:
            abandon_bulb(bulb)
            return False
        return release_bulb(bulb, restore=(self.saved or False) if restore else False)


# ---------------------------------------------------------------------------
def enable_lan(ip, token):
    from miio import Yeelight
    bulb = Yeelight(ip, token)
    print("before:", bulb.status())
    bulb.set_developer_mode(True)
    time.sleep(1)
    print("after: ", bulb.status())
    print("\nDeveloper mode on. Port 55443 should now be open.")


def list_devices():
    import soundcard as sc
    try:
        mics = sc.all_microphones(include_loopback=True)
    except TypeError:
        mics = sc.all_microphones()
    print("Capture devices:")
    for m in mics:
        print("   ", m.name)
    print("\nOutput devices:")
    for s in sc.all_speakers():
        print("   ", s.name)
    try:
        print("\nDefault input:", sc.default_microphone().name)
    except Exception:
        pass


def list_monitors():
    import mss
    with mss.MSS() as sct:
        print("Monitors:")
        for i, m in enumerate(sct.monitors):
            tag = "all displays" if i == 0 else f"display {i}"
            print(f"    {i}  {tag:13s} {m['width']}x{m['height']} "
                  f"at ({m['left']},{m['top']})")


VIRTUAL_INPUTS = ("blackhole", "loopback", "soundflower")


def _output_feeds_a_cable(speaker):
    """True if the system output is a virtual cable or a multi-output device.

    Those are the two setups in which a virtual input such as BlackHole
    carries what the Mac is playing; with a plain output it is silent.
    """
    name = speaker.name.lower()
    if any(k in name for k in VIRTUAL_INPUTS + ("multi-output", "aggregate")):
        return True
    try:                                # the device may have been renamed
        from soundcard import coreaudio
        kind = int(coreaudio._CoreAudio.get_property(
            speaker.id, 0x7472616E, "UInt32")[0])         # 'tran'
        return kind in (0x67727570, 0x76697274)           # 'grup', 'virt'
    except Exception:
        return False


def pick_recorder(mic_name=None, quiet=False):
    import soundcard as sc
    if mic_name:
        for attempt in (lambda: sc.get_microphone(mic_name, include_loopback=True),
                        lambda: sc.get_microphone(mic_name)):
            try:
                return attempt()
            except Exception:
                continue
        names = ", ".join(m.name for m in sc.all_microphones()) or "none found"
        raise RuntimeError(f"no capture device matching {mic_name!r}. "
                           f"Available: {names}")
    try:
        return sc.get_microphone(str(sc.default_speaker().name), include_loopback=True)
    except Exception:
        pass
    # macOS has no loopback of an output. If the output is routed into a
    # virtual cable, that cable is the clean feed - better than a microphone
    # hearing the room.
    try:
        cables = [m for m in sc.all_microphones()
                  if any(k in m.name.lower() for k in VIRTUAL_INPUTS)]
        if cables and _output_feeds_a_cable(sc.default_speaker()):
            return cables[0]
    except Exception:
        pass
    if not quiet:
        print("No loopback device found; using the microphone.", file=sys.stderr)
    return sc.default_microphone()


def cli_link(args):
    """Open the bulb for a CLI run, printing what the desk would show."""
    said = []

    def say(word, notice=None):
        # a reconnect counts down once a second; print each step once
        if notice and not notice[1].startswith("Lost the bulb. Trying again in") \
                and notice[1] not in said[-1:]:
            said.append(notice[1])
            print(notice[1], file=sys.stderr)

    link = Link(args.ip, remember=args.restore, status=say)
    link.open()
    return link


def run(args):
    recorder = pick_recorder(args.mic)
    link = cli_link(args)
    sender = link
    print(f"Music mode active on {link.ip}. Ctrl-C to stop.")

    analyser = Analyser()
    color = ColorEngine(args.palette, calm=args.calm, fade=args.fade)

    dt = analyser.dt
    min_gap = 1.0 / args.fps
    lvl = LevelEngine(calm=args.calm)
    sync = BeatSync()
    last_sent = 0.0
    last_hsv = None

    try:
        with recorder.recorder(samplerate=SAMPLERATE, channels=2) as rec:
            while True:
                if link.lost:
                    link.recover()
                    last_hsv = None
                f = analyser.feed(rec.record(numframes=BLOCK))
                fired = sync.feed(f, lvl, dt, args.lead / 1000.0, color)

                color.maybe_switch(f["t"], f["bands"])
                hue, sat = color.tick(dt, f["treble"], f["t"])

                now = time.monotonic()
                # a predicted beat goes out on its block, not the next free one
                if now - last_sent < (LED_GAP if fired else min_gap):
                    continue

                # perceptual gamma: a linear duty cycle looks washed out
                val = int(round(np.clip(
                    4 + 96 * (lvl.brightness ** args.gamma), 1, 100)))
                hsv = (int(round(hue)) % 360, int(round(sat)), val)

                if not fired and same_light(hsv, last_hsv, hue_tol=2):
                    continue

                # a sweep in progress must never be cut into by a hard set,
                # or the fade turns back into the jump it exists to replace
                fading = color.fade_t is not None
                sender.put(hsv, sudden=lvl.hit > 0.35 and not fading,
                           duration=int(np.clip(min_gap * 1000, 30, 150)))
                last_sent, last_hsv = now, hsv
    finally:
        link.close(restore=args.restore)


def run_screen(args):
    if screen_permission() is False:
        print("No Screen Recording permission: every frame would come back as "
              "the desktop wallpaper.\nGrant it in System Settings > Privacy "
              "& Security > Screen Recording, then restart this app.",
              file=sys.stderr)
        return

    grab = Capture(args.monitor, args.screen_region, args.screen_width,
                   args.screen_fps, args.capture)
    link = cli_link(args)
    sender = link
    print(f"Screen mode active on {link.ip}, monitor {args.monitor}, "
          f"{args.screen_fps} fps, {capture_label(grab)}. Ctrl-C to stop.")

    color = ScreenColorEngine(min_sat=args.min_sat, sat_boost=args.sat_boost,
                              gamma=args.screen_gamma, calm=args.calm)

    pacer = CapturePacer()
    last = time.monotonic()
    last_hsv = None

    try:
        while True:
            if link.lost:
                link.recover()
                last_hsv = None
            now = time.monotonic()
            dt, last = now - last, now

            frame = grab.frame()
            hue, sat, val, cut = color.tick(
                dt, *color.measure(crop_letterbox(frame)))
            period = pacer.period(frame, args.screen_fps, now)

            v = int(round(float(np.clip(3 + 97 * val, 1, 100))))
            light = screen_frame(color, hue, sat, v)

            # skip sends the LEDs could not resolve anyway
            if cut or not same_light(light, last_hsv):
                # ramp for one frame gap so consecutive colours join up
                sender.put(light, sudden=cut,
                           duration=ramp_ms(light, last_hsv, period))
                last_hsv = light

            grab.pace(now, 1.0 / args.screen_fps, period)
    finally:
        link.close(restore=args.restore)
        grab.close()


def run_fusion(args):
    """Colour from the screen, brightness from the audio."""
    if screen_permission() is False:
        print("No Screen Recording permission: every frame would come back as "
              "the desktop wallpaper.\nGrant it in System Settings > Privacy "
              "& Security > Screen Recording, then restart this app.",
              file=sys.stderr)
        return

    grab = Capture(args.monitor, args.screen_region, args.screen_width,
                   args.screen_fps, args.capture)
    color = ScreenColorEngine(min_sat=args.min_sat, sat_boost=args.sat_boost,
                              gamma=args.screen_gamma, calm=args.calm)
    screen = ScreenColorThread(grab, color, fps=args.screen_fps)
    recorder = pick_recorder(args.mic)
    link = cli_link(args)
    sender = link
    print(f"Fusion mode active on {link.ip}: colour from monitor "
          f"{args.monitor} ({capture_label(grab)}), brightness from audio. "
          f"Ctrl-C to stop.")

    analyser = Analyser()
    lvl = LevelEngine(calm=args.calm)
    sync = BeatSync()
    screen.start()

    dt = analyser.dt
    min_gap = 1.0 / args.fps
    last_sent = 0.0
    last_hsv = None

    try:
        with recorder.recorder(samplerate=SAMPLERATE, channels=2) as rec:
            while True:
                if link.lost:
                    link.recover()
                    last_hsv = None
                # the audio loop sets the pace; accents move brightness only,
                # because here the display owns the colour
                fired = sync.feed(analyser.feed(rec.record(numframes=BLOCK)),
                                  lvl, dt, args.lead / 1000.0)

                if screen.error:
                    raise RuntimeError(f"screen capture stopped: {screen.error}")

                now = time.monotonic()
                if now - last_sent < (LED_GAP if fired else min_gap):
                    continue

                hue, sat, _val, cut = screen.read()
                val = int(round(np.clip(
                    4 + 96 * (lvl.brightness ** args.gamma), 1, 100)))
                light = screen_frame(screen, hue, sat, val)

                if not (cut or fired) and same_light(light, last_hsv):
                    continue

                sender.put(light, sudden=cut or lvl.hit > 0.35,
                           duration=ramp_ms(light, last_hsv, min_gap))
                last_sent, last_hsv = now, light
    finally:
        screen.close()
        link.close(restore=args.restore)


def run_alert(args):
    """--alert: one pulse. A running desk is asked first (POST /api/alert),
    because only the desk can hold a session's frames while the pulse runs;
    with no desk answering, one short plain connection to --ip."""
    import urllib.error
    import urllib.request

    from lamp_signals import alert_flow, alert_lamp, clean_alert

    body = {"color": args.alert, "pulses": args.pulses, "speed": args.speed}
    picked, error = clean_alert(body)
    if error:
        sys.exit(f"--alert: {error}")
    if args.desk:
        req = urllib.request.Request(args.desk.rstrip("/") + "/api/alert",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=4) as r:
                reply = json.loads(r.read() or b"{}")
            print(f"pulsed through the desk ({reply.get('via')})")
            return
        except urllib.error.HTTPError as exc:
            try:
                reason = json.loads(exc.read()).get("error")
            except Exception:
                reason = exc.reason
            # a desk that has no bulb yet cannot pulse one; --ip can
            if not (args.ip and exc.code == 409 and reason == "no bulb address"):
                sys.exit(f"the desk refused the alert: {reason}")
        except (urllib.error.URLError, OSError):
            pass                                  # no desk: go to the lamp
    if not args.ip:
        sys.exit("no desk answered at --desk; pass --ip to reach the bulb directly")
    try:
        alert_lamp(args.ip, alert_flow(*picked))
    except Exception as exc:
        sys.exit(f"the bulb did not take the alert: {exc}")
    print(f"pulsed {args.ip}")


def capture_label(grab):
    """"ScreenCaptureKit" / "mss", and why not the stream if it was wanted."""
    if grab.backend == "sck":
        return "ScreenCaptureKit"
    return f"mss ({grab.note})" if grab.note else "mss"


def parse_region(text):
    try:
        x, y, w, h = (int(n) for n in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected X,Y,W,H in pixels")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return x, y, w, h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip")
    p.add_argument("--token", help="only for --enable-lan")
    p.add_argument("--enable-lan", action="store_true")
    p.add_argument("--mic", help="exact audio device name to capture")
    p.add_argument("--list-devices", action="store_true",
                   help="print every capture and output device, then exit")
    p.add_argument("--palette", choices=sorted(PALETTES), default=None,
                   help="pin one palette instead of switching on section changes")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--gamma", type=float, default=1.9,
                   help="higher = more contrast between quiet and loud")
    p.add_argument("--calm", action="store_true",
                   help="gentler: softer hits, longer decays, slower colour steps")
    p.add_argument("--no-restore", dest="restore", action="store_false",
                   help="leave the lamp on the last colour sent when this "
                        "exits, instead of putting back what it showed before")
    p.add_argument("--lead", type=int, default=80, metavar="MS",
                   help="beat sync: once the tempo holds, send each flash this "
                        "many ms early so it lands on the beat over Wi-Fi "
                        "(0 = flash when the beat is heard; default 80)")
    p.add_argument("--fade", type=float, default=0.0, metavar="SECONDS",
                   help="smooth colour mode: sweep between palette anchors "
                        "over this many seconds instead of snapping "
                        "(0 = snap, the default; try 1.5)")

    a = p.add_argument_group("alert")
    a.add_argument("--alert", metavar="COLOR",
                   help="pulse the lamp in this colour (#rrggbb or red, orange, "
                        "green, blue, purple, white), then put back what it showed")
    a.add_argument("--pulses", type=int, default=2, help="1-5 (default 2)")
    a.add_argument("--speed", choices=("slow", "normal", "fast"), default="normal")
    a.add_argument("--desk", default="http://127.0.0.1:8765", metavar="URL",
                   help="a running desk to send the alert through (default "
                        "%(default)s; '' to always go to --ip)")

    g = p.add_argument_group("screen mode")
    g.add_argument("--screen", action="store_true",
                   help="follow the dominant screen colour instead of audio")
    g.add_argument("--fusion", action="store_true",
                   help="both at once: the screen picks the colour, the audio "
                        "drives the brightness")
    g.add_argument("--list-monitors", action="store_true",
                   help="print every display with its index, then exit")
    g.add_argument("--monitor", type=int, default=1,
                   help="display index from --list-monitors (default: 1)")
    g.add_argument("--screen-region", type=parse_region, metavar="X,Y,W,H",
                   help="capture only this box - use it to skip player chrome")
    g.add_argument("--screen-fps", type=int, default=10,
                   help="capture rate while the picture moves, max 30 "
                        "(default: 10); a still picture is read less often")
    g.add_argument("--capture", choices=CAPTURE_BACKENDS, default="auto",
                   help="screen capture: a ScreenCaptureKit stream (sck, needs "
                        "the 'sck' extra), one mss grab per frame (mss), or "
                        f"auto (= {AUTO_CAPTURE}); a stream that fails falls "
                        "back to mss")
    g.add_argument("--screen-width", type=int, default=192,
                   help="pixels the frame is sampled down to (default: 192)")
    g.add_argument("--screen-gamma", type=float, default=0.75,
                   help="below 1 lifts dark scenes (default: 0.75)")
    g.add_argument("--min-sat", type=float, default=55.0,
                   help="saturation floor for colour scenes (default: 55)")
    g.add_argument("--sat-boost", type=float, default=1.35,
                   help="multiplier on measured saturation (default: 1.35)")

    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.list_monitors:
        list_monitors()
        return

    if args.fusion and args.screen:
        p.error("--fusion already includes the screen; drop --screen")

    if not 0 <= args.lead <= 250:
        p.error("--lead must be between 0 and 250 ms")

    if not 0.0 <= args.fade <= 10.0:
        p.error("--fade must be between 0 and 10 seconds")

    if not 1 <= args.fps <= 30:
        p.error("--fps must be between 1 and 30; the LEDs cannot resolve "
                "faster")

    if not 1 <= args.screen_fps <= 30:
        p.error("--screen-fps must be between 1 and 30; the LEDs cannot "
                "resolve faster")

    if args.enable_lan:
        if not (args.ip and args.token):
            p.error("--enable-lan needs --ip and --token")
        enable_lan(args.ip, args.token)
        return

    if args.alert:
        run_alert(args)
        return

    if not args.ip:
        p.error("--ip is required")

    try:
        if args.fusion:
            run_fusion(args)
        elif args.screen:
            run_screen(args)
        else:
            run(args)
    except KeyboardInterrupt:
        print("\nbye")
    except BulbUnreachable as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
