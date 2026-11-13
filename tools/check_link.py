"""Check the self-healing connection against a REAL bulb.  Not part of the suite.

    python tools/check_link.py <bulb-ip> hello        # id over UDP; no TCP slot
    python tools/check_link.py <bulb-ip> find         # broadcast, then sweep, by that id
    python tools/check_link.py <bulb-ip> diagnose     # one open through Link, explained
    python tools/check_link.py <bulb-ip> reconnect    # drive the lamp; power-cycle it

`hello` and `find` only send UDP hellos and are always safe. `diagnose` and
`reconnect` take the bulb's connection slots like a desk session: stop the
desk's session first (POST /api/control {"running": false}) and start it again
afterwards.

reconnect: the lamp cycles slowly through four colours. Switch the bulb off at
the wall for about 5 s and on again. Expect "Lost the bulb", a countdown,
"reconnecting n/6", then "connected" and the colours carrying on within about
15 s of power returning. Ctrl-C ends it and puts the lamp back as it was.
"""
import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from music_bulb import BulbUnreachable, Link, find_bulb, local_net, miio_hello  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("ip")
ap.add_argument("what", choices=("hello", "find", "diagnose", "reconnect"))
args = ap.parse_args()


def say(word, notice=None):
    print(f"  [{time.strftime('%H:%M:%S')}] {word:16s} {notice[1] if notice else ''}")


did = miio_hello(args.ip)
print(f"hello {args.ip}: " + (f"device id {did:08x}" if did is not None else "no answer"))
if args.what == "hello":
    sys.exit(0 if did is not None else 1)

if args.what == "find":
    if did is None:
        sys.exit("nothing to look for: the address did not answer the hello")
    print("this machine, broadcast:", *local_net(args.ip))
    for sweep in (False, True):
        t0 = time.monotonic()
        where = find_bulb(did, args.ip, sweep=sweep)
        print(f"  {'sweep' if sweep else 'broadcast'}: {where} in {time.monotonic() - t0:.2f} s")
    sys.exit(0)

link = Link(args.ip, did=did, status=say)
try:
    link.open()
except BulbUnreachable as exc:
    print(f"could not open ({exc.kind}): {exc}")
    sys.exit(1)
print(f"opened {link.ip}; the lamp's look at start: {link.saved}")
if args.what == "diagnose":
    link.close(restore=True)
    sys.exit(0)

colours = itertools.cycle([(0, 100, 60), (120, 100, 60), (240, 100, 60), (40, 100, 60)])
try:
    while True:
        if link.lost:
            print("send failed:", link.sender.error)
            link.recover()
        link.put(next(colours), False, 300)
        time.sleep(0.8)
except KeyboardInterrupt:
    pass
except BulbUnreachable as exc:
    print(f"gave up ({exc.kind}): {exc}")
finally:
    print("restored:", link.close(restore=True))
