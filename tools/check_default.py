"""Set the REAL lamp's power-on look (set_default).  Not part of the suite.

    python tools/check_default.py <bulb-ip>                       # what it shows now
    python tools/check_default.py <bulb-ip> --ct 2700 --bright 40
    python tools/check_default.py <bulb-ip> --rgb ff8a2a --bright 60

The lamp keeps this look in its own memory and comes on with it whenever the
wall switch powers it. The old one cannot be read back, so this asks first.
Stop the desk's session first (it takes the lamp's connection slots). One
short connection; with --ct or --rgb the lamp switches to that look first and
stays on it.

Then check it: switch the lamp off at the wall, wait 5 s, switch it on. It
should come on in the saved look.
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from music_bulb import lamp_call, read_lamp  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("ip")
g = ap.add_mutually_exclusive_group()
g.add_argument("--ct", type=int, help="white, 1700-6500 K")
g.add_argument("--rgb", help="colour as RRGGBB")
ap.add_argument("--bright", type=int, default=None, help="1-100")
ap.add_argument("--yes", action="store_true", help="do not ask")
args = ap.parse_args()

cmds = []
if args.ct is not None:
    cmds.append(("set_ct_abx", [max(1700, min(args.ct, 6500)), "smooth", 300]))
elif args.rgb:
    cmds.append(("set_rgb", [int(args.rgb.lstrip("#"), 16) & 0xFFFFFF, "smooth", 300]))
if args.bright is not None:
    cmds.append(("set_bright", [max(1, min(args.bright, 100)), "smooth", 300]))

try:
    now = read_lamp(args.ip)
except Exception as exc:
    sys.exit(f"the bulb did not answer: {exc}")
print("the lamp shows:", now)
if not now or now["power"] != "on":
    sys.exit("the lamp is off: switch it on first (the bulb refuses set_default while off)")
if not args.yes and input("Save this as the power-on look? The old one is lost. [y/N] ").strip().lower() != "y":
    sys.exit("nothing changed")
try:
    lamp_call(args.ip, cmds + [("set_default", [])], timeout=3.0)
except Exception as exc:
    sys.exit(f"not saved: {exc}")
print("saved:", read_lamp(args.ip))
print("now switch it off at the wall, wait 5 s, and on: it should come on like this")
