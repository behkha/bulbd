"""Drive a REAL bulb and read back what it shows.  Not part of the test suite.

WARNING: this changes the colour and brightness of the lamp at the address you
pass, opens its music mode, and takes one of its few LAN connection slots. Do
not run it while the desk or `music_bulb.py` is driving the same bulb.

    python tools/check_bulb.py <bulb-ip>

It sends known smooth/sudden commands and compares them with what the bulb
reports over SSDP (which does not touch the control socket).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yeelight import discover_bulbs  # noqa: E402
from music_bulb import open_bulb, release_bulb  # noqa: E402

ap = argparse.ArgumentParser(description="Drive a real bulb and read back what it shows.")
ap.add_argument("ip", help="the bulb's address on your LAN")
IP = ap.parse_args().ip


def state():
    for _ in range(3):
        for b in discover_bulbs(timeout=1):
            if b["ip"] == IP:
                c = b["capabilities"]
                rgb = int(c.get("rgb", 0))
                return {"power": c.get("power"), "mode": c.get("color_mode"), "rgb": f"#{rgb:06x}",
                        "bright": int(c.get("bright", -1)), "hue": int(c.get("hue", -1)),
                        "sat": int(c.get("sat", -1))}
    return None


def show(label, want, st):
    print(f"  {label:44s} sent {want}  bulb reports {st}")


print("before:", state())
bulb = open_bulb(IP)
try:
    time.sleep(0.5)
    for dur in (30, 33, 40, 49, 50, 100):
        bulb.set_hsv(240, 100, 60, effect="sudden", duration=0)        # baseline: blue 60
        time.sleep(0.6)
        base = state()
        bulb.set_hsv(0, 100, 30, effect="smooth", duration=dur)       # test: red 30
        time.sleep(0.7)
        after = state()
        took = after and after["rgb"] != base["rgb"]
        print(f"  smooth dur={dur:3d} ms: baseline {base['rgb']}/{base['bright']} -> after {after['rgb']}/{after['bright']}"
              f"   {'APPLIED' if took else 'IGNORED'}")

    print("-- colour accuracy (sudden)")
    for hsv in ((26, 92, 80), (120, 100, 50), (216, 68, 28), (300, 45, 100), (0, 0, 1)):
        bulb.set_hsv(*hsv, effect="sudden", duration=0)
        time.sleep(0.7)
        show("", hsv, state())

    print("-- stream: 2 s of smooth frames, then what the bulb ends on")
    for label, gap, dur in (("21.5/s, dur 40 (music default)", 0.0464, 40), ("21.5/s, dur 50", 0.0464, 50),
                            ("10/s, dur 100 (screen default)", 0.1, 100)):
        bulb.set_hsv(240, 100, 60, effect="sudden", duration=0)
        time.sleep(0.5)
        n = int(2.0 / gap)
        for i in range(n):
            bulb.set_hsv(int(300 * (i + 1) / n), 100, 20 + int(60 * (i + 1) / n), effect="smooth", duration=dur)
            time.sleep(gap)
        time.sleep(0.6)
        show(label, (300, 100, 80), state())
finally:
    release_bulb(bulb)
    time.sleep(0.5)
    print("after release:", state())
