"""Check Beat sync on a REAL bulb: does the lead move the flash the right way?

    python tools/check_sync.py <bulb-ip> [--leads 0,80,200] [--secs 12]

Plays a 120 BPM kick-and-hat track on the default output (your speakers),
feeds the same samples to the analyser the desk uses, and flashes the lamp
red on the beat - the music-mode chain with BeatSync, through Link. The
lead changes every --secs seconds and is printed as it changes.

Expect: at 0 the flash trails each kick a little; at 80 the flash and the
kick coincide; at 200 the flash clearly comes before the kick. If 0 already
looks on time, your speakers are slow and a lead near 0-40 suits you; if 80
still trails, raise it on the desk's Beat sync fader.

Takes the bulb's connection slots like a desk session: stop the desk's
session first (POST /api/control {"running": false}) and start it again
afterwards. Ctrl-C puts the lamp back.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from music_bulb import (BLOCK, LED_GAP, SAMPLERATE, Analyser, BeatSync, BulbUnreachable,  # noqa: E402
                        LevelEngine, Link, same_light)

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("ip")
ap.add_argument("--leads", default="0,80,200", help="ms, comma-separated (default 0,80,200)")
ap.add_argument("--secs", type=float, default=12.0, help="seconds per lead (default 12)")
args = ap.parse_args()
leads = [int(x) for x in args.leads.split(",")]

# the test suite's kick (a falling sine) and an off-beat hat
sr, beat = SAMPLERATE, 0.5
secs = args.secs * len(leads) + 6.0               # 6 s for the tempo to lock first
t = np.arange(int(0.18 * sr)) / sr
kick = (np.sin(2 * np.pi * (55 + 60 * np.exp(-t * 30)) * t) * np.exp(-t * 22)).astype(np.float32)
rng = np.random.default_rng(0)
n = int(0.04 * sr)
hat = (rng.normal(0, 1, n) * np.exp(-np.arange(n) / (sr * 0.008))).astype(np.float32) * 0.25
x = np.zeros(int(secs * sr), dtype=np.float32)
for i in range(int(secs / beat)):
    s = int(i * beat * sr)
    x[s:s + len(kick)] += kick[:len(x) - s]
    s2 = int((i + 0.5) * beat * sr)
    x[s2:s2 + len(hat)] += hat[:max(0, len(x) - s2)]
pcm = np.stack([0.5 * x, 0.5 * x], 1)

import soundcard as sc  # noqa: E402

link = Link(args.ip, status=lambda w, n=None: n and print(" ", n[1]))
try:
    link.open()
except BulbUnreachable as exc:
    sys.exit(f"could not open: {exc}")

an, lvl, sync = Analyser(), LevelEngine(), BeatSync()
last_sent, last = -1.0, None
shown = None
try:
    # the device's own buffer: a Multi-Output device takes at most 512
    with sc.default_speaker().player(samplerate=sr, channels=2, blocksize=512) as player:
        for i in range(0, len(pcm) - BLOCK, BLOCK):
            block = pcm[i:i + BLOCK]
            player.play(block)
            f = an.feed(block)
            phase = max(0.0, f["t"] - 6.0)
            lead = leads[min(int(phase // args.secs), len(leads) - 1)]
            if lead != shown:
                print(f"lead {lead} ms" + ("  (tempo locking...)" if f["t"] < 6 else ""))
                shown = lead
            fired = sync.feed(f, lvl, an.dt, lead / 1000.0)
            if link.lost:
                link.recover()
            if f["t"] - last_sent < (LED_GAP if fired else 0.04):
                continue
            frame = (0, 100, int(round(np.clip(4 + 96 * lvl.brightness ** 1.9, 1, 100))))
            if not fired and same_light(frame, last, hue_tol=2):
                continue
            link.put(frame, lvl.hit > 0.35, 40)
            last_sent, last = f["t"], frame
except KeyboardInterrupt:
    pass
finally:
    print("restored:", link.close(restore=True))
