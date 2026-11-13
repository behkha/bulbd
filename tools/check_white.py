"""Judge white frames on a REAL bulb.  Not part of the suite.

    python tools/check_white.py <bulb-ip> match [--level 60]
    python tools/check_white.py <bulb-ip> sweep

Takes the bulb's connection slots like a desk session: stop the desk's
session first (POST /api/control {"running": false}) and start it again
afterwards. Ctrl-C puts the lamp back.

match: the lamp alternates every 3 s between what a grey scene used to send
(hue 30, saturation 18 on the colour LEDs, at --level) and the white LEDs at
6500 K, trying one brightness gain per pair - 0.25, 0.32, 0.40, 0.50, 0.63.
The printed gain whose white looks as bright as the colour step before it is
the one for WHITE_GAIN in music_bulb.py (0.40 now). Each switch uses the
same 300 ms crossfade as the desk; it should not pop.

sweep: white at the same matched level from 2700 K to 6500 K and back, one
step a second, then the scene-cut jump (50 ms) between 3000 K and 6500 K.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import music_bulb  # noqa: E402
from music_bulb import SWITCH_MS, BulbUnreachable, Link, White  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("ip")
ap.add_argument("what", choices=("match", "sweep"))
ap.add_argument("--level", type=int, default=60, help="colour-scale level 1..100 (default 60)")
args = ap.parse_args()

link = Link(args.ip, status=lambda w, n=None: n and print(" ", n[1]))
try:
    link.open()
except BulbUnreachable as exc:
    sys.exit(f"could not open: {exc}")


def show(frame, label, hold, sudden=False, ms=SWITCH_MS):
    if link.lost:
        link.recover()
    link.put(frame, sudden, ms)
    print(f"  {label}")
    time.sleep(hold)


try:
    grey = (30, 18, args.level)
    if args.what == "match":
        for gain in (0.25, 0.32, 0.40, 0.50, 0.63):
            show(grey, f"colour LEDs  hue 30 sat 18 at {args.level}", 3)
            bright = max(1, round(args.level * gain))
            show(White(6500, bright), f"white LEDs   6500 K at {bright}  (gain {gain:.2f})", 3)
    else:
        level = max(1, round(args.level * music_bulb.WHITE_GAIN))
        temps = list(range(2700, 6501, 400))
        for k in temps + temps[::-1]:
            show(White(k, level), f"{k} K at {level}", 1.0, ms=900)
        for _ in range(3):
            show(White(3000, level), "cut to 3000 K", 1.5, sudden=True)
            show(White(6500, level), "cut to 6500 K", 1.5, sudden=True)
except KeyboardInterrupt:
    pass
finally:
    print("restored:", link.close(restore=True))
