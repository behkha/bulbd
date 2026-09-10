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
import math
import queue
import sys
import threading
import time
from collections import deque

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

        # beats live low, so weight the flux sum towards the bottom
        self.flux_w = np.linspace(1.6, 0.5, n_bands).astype(np.float32)

        self.prev_log = np.zeros(n_bands, dtype=np.float32)
        self.prev_flux = 0.0
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
        x -= x.mean()
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        lo = max(1, int(round(0.33 / self.dt)))              # 180 BPM
        hi = min(len(ac) - 1, int(round(0.92 / self.dt)))    # 65 BPM
        if hi <= lo:
            return
        est = (lo + int(np.argmax(ac[lo:hi]))) * self.dt
        self.beat += (est - self.beat) * 0.15                # slew, don't jump

    def feed(self, pcm):
        self.t += self.dt
        self.frame += 1
        mono = pcm.mean(axis=1) if pcm.ndim > 1 else pcm
        rms = float(np.sqrt(np.mean(np.square(mono))))

        spec = np.abs(np.fft.rfft(mono * self.window))
        raw = np.array([spec[m].mean() for m in self.masks], dtype=np.float32)
        log = np.log1p(raw * 200.0)

        # --- onset: half-wave-rectified spectral flux vs adaptive threshold
        flux = float(np.dot(self.flux_w, np.maximum(log - self.prev_log, 0.0)))
        self.prev_log = log
        self.flux_hist.append(flux)
        self.env.append(flux)

        onset, strength = False, 0.0
        if len(self.flux_hist) == self.flux_hist.maxlen:
            h = np.fromiter(self.flux_hist, dtype=np.float32)
            thr = float(np.median(h) + self.sensitivity * h.std()) + 1e-4
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

    def __init__(self, palette=None, calm=False, fade=0.0):
        self.names = list(PALETTES)
        self.pal = palette if palette in PALETTES else self.names[0]
        self.locked = palette is not None
        self.calm = calm
        self.fade = float(fade)
        self.idx = 0
        h, s = PALETTES[self.pal][0]
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
        gap = max(2.0 * beat, 0.9) * (2.2 if self.calm else 1.0)
        if self.fade > 0.0:
            gap = max(gap, self.fade)             # let a sweep land first
        if t - self.last_step < gap:
            return
        anchors = PALETTES[self.pal]
        self.idx = (self.idx + 1) % len(anchors)
        h, s = anchors[self.idx]
        self.retarget(h, s, t)
        if self.fade <= 0.0:
            # snap part of the way for punch, glide the remainder
            self.hue = hue_lerp(self.hue, self.target_hue,
                                0.25 if self.calm else 0.55)
        self.last_step = t

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
        choices = [n for n in self.names if n != self.pal]
        self.pal = choices[int(t * 7) % len(choices)]
        self.idx = 0
        self.retarget(*PALETTES[self.pal][0], t=t)
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

    def feed(self, f, dt):
        """-> True when this frame is a strong accent (worth a colour step)."""
        punch = self.punch if self.punch is not None else (0.45 if self.calm else 0.78)
        gate = 0.30 if self.calm else 0.12
        accent = False

        if f["rms"] < 0.004:
            self.silent_since = self.silent_since or f["t"]
        else:
            self.silent_since = None

        if self.silent_since and f["t"] - self.silent_since > 1.2:
            phase = 0.5 + 0.5 * math.sin(2 * math.pi * f["t"] / 7.0)
            level = 0.06 + 0.10 * phase
            self.hit = self.sustain = 0.0
        else:
            if f["onset"] and f["strength"] > gate:
                self.hit = max(self.hit, 0.35 + 0.65 * f["strength"])
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


class ScreenGrabber:
    """Downsampled screen frames as float RGB in 0..1."""

    def __init__(self, monitor=1, region=None, width=192):
        import mss
        self._sct = mss.MSS()
        mons = self._sct.monitors
        if region:
            x, y, w, h = region
            self.box = {"left": x, "top": y, "width": w, "height": h}
        else:
            if not 1 <= monitor < len(mons):
                raise RuntimeError(f"no monitor {monitor}; this Mac has "
                                   f"{len(mons) - 1}. Try --list-monitors")
            self.box = mons[monitor]
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


class ScreenColorEngine:
    """Dominant hue by weighted hue histogram, then smoothed over time.

    A plain mean over the frame turns any two-colour scene into grey, so
    vivid pixels vote into 10-degree hue bins and the winning bin's circular
    mean becomes the target. Unlike the music path this drifts continuously -
    an ambient lamp tracking a film should not step in palette jumps.
    """

    N_BINS = 36

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

    def measure(self, rgb):
        """One frame -> (hue or None if colourless, sat 0..1, luma 0..1)."""
        h, s, v = rgb_to_hsv(rgb)
        luma = float((rgb @ _LUMA).mean())

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

        ang = np.radians(hk[near])
        hue = float(np.degrees(np.arctan2(float((np.sin(ang) * ww).sum()),
                                          float((np.cos(ang) * ww).sum()))) % 360.0)
        return hue, float((sk[near] * ww).sum() / ww.sum()), luma

    def tick(self, dt, hue_m, sat_m, luma):
        """-> (hue, sat 0..100, value 0..1, scene_cut)."""
        cut = self.prev_luma is not None and abs(luma - self.prev_luma) > 0.16
        self.prev_luma = luma

        t_hue = 0.10 if cut else (0.90 if self.calm else 0.45)
        t_sat = 0.12 if cut else (1.10 if self.calm else 0.60)
        t_val = 0.08 if cut else (0.45 if self.calm else 0.22)

        if hue_m is None:
            want_sat = 18.0                         # grey scene -> near white
        else:
            self.hue = hue_lerp(self.hue, hue_m, 1.0 - math.exp(-dt / t_hue))
            want_sat = float(np.clip(sat_m * 100.0 * self.sat_boost,
                                     self.min_sat, 100.0))

        self.sat += (want_sat - self.sat) * (1.0 - math.exp(-dt / t_sat))
        want_val = float(np.clip(luma, 0.0, 1.0)) ** self.gamma
        self.val += (want_val - self.val) * (1.0 - math.exp(-dt / t_val))
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
    """

    def __init__(self, grab, engine, fps=18):
        super().__init__(daemon=True)
        self.grab = grab
        self.engine = engine
        self.fps = fps
        self.alive = True
        self.error = None
        self._lock = threading.Lock()
        self._hue = engine.hue
        self._sat = engine.sat
        self._val = engine.val
        self._cut = False

    def run(self):
        last = time.monotonic()
        while self.alive:
            now = time.monotonic()
            dt, last = now - last, now
            try:
                hue, sat, val, cut = self.engine.tick(
                    dt, *self.engine.measure(crop_letterbox(self.grab.frame())))
            except Exception as exc:                # a display that went away
                self.error = exc
                self.alive = False
                return
            with self._lock:
                self._hue, self._sat, self._val = hue, sat, val
                self._cut = self._cut or cut
            slack = 1.0 / max(self.fps, 1) - (time.monotonic() - now)
            if slack > 0:
                time.sleep(slack)

    def read(self):
        """-> (hue, sat 0..100, value 0..1, cut) - clears the pending cut."""
        with self._lock:
            cut, self._cut = self._cut, False
            return self._hue, self._sat, self._val, cut

    def close(self):
        self.alive = False
        self.grab.close()


# ---------------------------------------------------------------------------
# sender thread - keeps network latency out of the audio loop
# ---------------------------------------------------------------------------
class Sender(threading.Thread):
    def __init__(self, bulb):
        super().__init__(daemon=True)
        self.bulb = bulb
        self.box = queue.Queue(maxsize=1)          # latest value wins
        self.alive = True

    def put(self, hsv, sudden, duration=40):
        """Queue one frame. `duration` is the bulb-side ramp for smooth sends;
        set it to the gap between frames so consecutive colours join up."""
        try:
            self.box.get_nowait()                  # drop the stale frame
        except queue.Empty:
            pass
        try:
            self.box.put_nowait((hsv, sudden, duration))
        except queue.Full:
            pass

    def run(self):
        while self.alive:
            try:
                (h, s, v), sudden, duration = self.box.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.bulb.set_hsv(h, s, v,
                                  effect="sudden" if sudden else "smooth",
                                  duration=0 if sudden else duration)
            except Exception as exc:
                print("link dropped, reopening music mode:", exc, file=sys.stderr)
                time.sleep(0.4)
                try:
                    self.bulb.stop_music()
                except Exception:
                    pass
                try:
                    self.bulb.start_music()
                except Exception:
                    time.sleep(1.0)


def open_bulb(ip, attempts=3, on_attempt=None):
    """Connect and enter music mode, retrying a bulb that is still busy.

    A color5 accepts only a handful of LAN connections at once, so a slot
    freed a moment ago may not be free yet; each failed attempt is released
    before the next one rather than left dangling. Retries take a few
    seconds, so `on_attempt(n, total, error)` reports progress - without it
    a caller showing "connecting" just looks hung.
    """
    from yeelight import Bulb

    last = None
    for i in range(attempts):
        if on_attempt:
            on_attempt(i + 1, attempts, last)
        bulb = Bulb(ip, effect="sudden", duration=0, auto_on=True)
        try:
            bulb.turn_on()
            bulb.start_music()
            return bulb
        except Exception as exc:
            last = exc
            release_bulb(bulb)
            if i < attempts - 1:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"could not open music mode on {ip} after {attempts} "
                       f"tries: {last}. If the bulb answers discovery but not "
                       f"port 55443, its LAN slots are wedged - power-cycle it")


def release_bulb(bulb):
    """Hand the bulb's LAN connection back.

    yeelight.Bulb holds one persistent socket and never closes it on error;
    stop_music() closes it but then sends `set_music 0`, which lazily opens
    a *fresh* socket that leaks if that send also fails. Every leaked socket
    burns one of the bulb's few connection slots until it times them out
    minutes later, so a few failed sessions lock the bulb out entirely.
    """
    try:
        bulb.stop_music()
    except Exception:
        pass
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


def pick_recorder(mic_name=None):
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
        print("No loopback device found; using the microphone.", file=sys.stderr)
        return sc.default_microphone()


def run(args):
    bulb = open_bulb(args.ip)
    print(f"Music mode active on {args.ip}. Ctrl-C to stop.")

    analyser = Analyser()
    color = ColorEngine(args.palette, calm=args.calm, fade=args.fade)
    sender = Sender(bulb)
    sender.start()
    recorder = pick_recorder(args.mic)

    dt = analyser.dt
    min_gap = 1.0 / args.fps
    lvl = LevelEngine(calm=args.calm)
    last_sent = 0.0
    last_hsv = None

    try:
        with recorder.recorder(samplerate=SAMPLERATE, channels=2) as rec:
            while True:
                f = analyser.feed(rec.record(numframes=BLOCK))

                if lvl.feed(f, dt):
                    color.step(f["t"], f["beat"])

                color.maybe_switch(f["t"], f["bands"])
                hue, sat = color.tick(dt, f["treble"], f["t"])

                now = time.monotonic()
                if now - last_sent < min_gap:
                    continue

                # perceptual gamma: a linear duty cycle looks washed out
                val = int(round(np.clip(
                    4 + 96 * (lvl.brightness ** args.gamma), 1, 100)))
                hsv = (int(round(hue)) % 360, int(round(sat)), val)

                if last_hsv and abs(hsv[2] - last_hsv[2]) < 2 \
                        and abs(hsv[1] - last_hsv[1]) < 3 \
                        and abs(((hsv[0] - last_hsv[0] + 180) % 360) - 180) < 2:
                    continue

                # a sweep in progress must never be cut into by a hard set,
                # or the fade turns back into the jump it exists to replace
                fading = color.fade_t is not None
                sender.put(hsv, sudden=lvl.hit > 0.35 and not fading,
                           duration=int(np.clip(min_gap * 1000, 30, 150)))
                last_sent, last_hsv = now, hsv
    finally:
        sender.alive = False
        release_bulb(bulb)


def run_screen(args):
    if screen_permission() is False:
        print("No Screen Recording permission: every frame would come back as "
              "the desktop wallpaper.\nGrant it in System Settings > Privacy "
              "& Security > Screen Recording, then restart this app.",
              file=sys.stderr)
        return

    grab = ScreenGrabber(args.monitor, args.screen_region, args.screen_width)
    bulb = open_bulb(args.ip)
    print(f"Screen mode active on {args.ip}, monitor {args.monitor}, "
          f"{args.screen_fps} fps. Ctrl-C to stop.")

    color = ScreenColorEngine(min_sat=args.min_sat, sat_boost=args.sat_boost,
                              gamma=args.screen_gamma, calm=args.calm)
    sender = Sender(bulb)
    sender.start()

    period = 1.0 / args.screen_fps
    last = time.monotonic()
    last_hsv = None

    try:
        while True:
            now = time.monotonic()
            dt, last = now - last, now

            hue, sat, val, cut = color.tick(
                dt, *color.measure(crop_letterbox(grab.frame())))

            v = int(round(float(np.clip(3 + 97 * val, 1, 100))))
            hsv = (int(round(hue)) % 360, int(round(sat)), v)

            # skip sends the LEDs could not resolve anyway
            settled = (last_hsv and not cut
                       and abs(hsv[2] - last_hsv[2]) < 2
                       and abs(hsv[1] - last_hsv[1]) < 3
                       and abs(((hsv[0] - last_hsv[0] + 180) % 360) - 180) < 3)
            if not settled:
                sender.put(hsv, sudden=cut)
                last_hsv = hsv

            slack = period - (time.monotonic() - now)
            if slack > 0:
                time.sleep(slack)
    finally:
        sender.alive = False
        release_bulb(bulb)
        grab.close()


def run_fusion(args):
    """Colour from the screen, brightness from the audio."""
    if screen_permission() is False:
        print("No Screen Recording permission: every frame would come back as "
              "the desktop wallpaper.\nGrant it in System Settings > Privacy "
              "& Security > Screen Recording, then restart this app.",
              file=sys.stderr)
        return

    grab = ScreenGrabber(args.monitor, args.screen_region, args.screen_width)
    color = ScreenColorEngine(min_sat=args.min_sat, sat_boost=args.sat_boost,
                              gamma=args.screen_gamma, calm=args.calm)
    screen = ScreenColorThread(grab, color, fps=args.screen_fps)
    bulb = open_bulb(args.ip)
    print(f"Fusion mode active on {args.ip}: colour from monitor "
          f"{args.monitor}, brightness from audio. Ctrl-C to stop.")

    analyser = Analyser()
    lvl = LevelEngine(calm=args.calm)
    recorder = pick_recorder(args.mic)
    sender = Sender(bulb)
    sender.start()
    screen.start()

    dt = analyser.dt
    min_gap = 1.0 / args.fps
    last_sent = 0.0
    last_hsv = None

    try:
        with recorder.recorder(samplerate=SAMPLERATE, channels=2) as rec:
            while True:
                # the audio loop sets the pace; accents move brightness only,
                # because here the display owns the colour
                lvl.feed(analyser.feed(rec.record(numframes=BLOCK)), dt)

                if screen.error:
                    raise RuntimeError(f"screen capture stopped: {screen.error}")

                now = time.monotonic()
                if now - last_sent < min_gap:
                    continue

                hue, sat, _val, cut = screen.read()
                val = int(round(np.clip(
                    4 + 96 * (lvl.brightness ** args.gamma), 1, 100)))
                hsv = (int(round(hue)) % 360, int(round(sat)), val)

                settled = (last_hsv and not cut
                           and abs(hsv[2] - last_hsv[2]) < 2
                           and abs(hsv[1] - last_hsv[1]) < 3
                           and abs(((hsv[0] - last_hsv[0] + 180) % 360) - 180) < 3)
                if settled:
                    continue

                sender.put(hsv, sudden=cut or lvl.hit > 0.35,
                           duration=int(np.clip(min_gap * 1000, 30, 150)))
                last_sent, last_hsv = now, hsv
    finally:
        sender.alive = False
        screen.close()
        release_bulb(bulb)


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
    p.add_argument("--fade", type=float, default=0.0, metavar="SECONDS",
                   help="smooth colour mode: sweep between palette anchors "
                        "over this many seconds instead of snapping "
                        "(0 = snap, the default; try 1.5)")

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
    g.add_argument("--screen-fps", type=int, default=18,
                   help="capture rate, max 30 (default: 18)")
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

    if not 0.0 <= args.fade <= 10.0:
        p.error("--fade must be between 0 and 10 seconds")

    if not 1 <= args.screen_fps <= 30:
        p.error("--screen-fps must be between 1 and 30; the LEDs cannot "
                "resolve faster")

    if args.enable_lan:
        if not (args.ip and args.token):
            p.error("--enable-lan needs --ip and --token")
        enable_lan(args.ip, args.token)
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


if __name__ == "__main__":
    main()
