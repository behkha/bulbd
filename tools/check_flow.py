"""How does a REAL bulb run colour flows (start_cf) inside music mode?

    python tools/check_flow.py <bulb-ip> all            # every test, ~6 min
    python tools/check_flow.py <bulb-ip> count stop     # just these

Tests, each one music session through Link of well under a minute:
    count    does a 6-step flow run to its count and then stop, on the last step?
    min      the shortest step the bulb honours (8 steps of 50/30/20/10 ms)
    replace  a one-step start_cf, then a plain set_rgb, sent into an endless
             flow: does the flow stop and stay stopped on the new colour?
    stop     stop_cf into an endless flow
    tick     the bulb's resolution: a 2-step flow whose first step grows 0-25 ms
    timing   how steady the end of a flow is, short (2 steps) against long
             (24 steps): per-step jitter would make the long one wander more

What can be seen. The music socket is one-way and, in music mode, the bulb
pushes no `props`, so the lamp is watched on one more plain connection
(`Watcher`) by polling get_prop. `rgb` does not move during a flow - it
shows the last step once the flow ends - but `flowing` does go 1 -> 0 when it
ends, so every test here times the END of a flow. A poll is answered in
~13 ms (median), so an end is pinned to about +-7 ms by polling back to back
around where it is expected.

The bulb allows 60 commands a minute per connection ("client quota
exceeded" after that), so each test spends at most ~55 polls and `all`
waits a minute between tests with the lamp released.

Costs the bulb one connection beyond a desk session's; each session
restores the lamp. Stop the desk's session first. Ctrl-C is safe.
"""
import argparse
import json
import socket
import statistics
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from music_bulb import LAMP_PORT, BulbUnreachable, Link  # noqa: E402

RED, BLUE, GREEN = 0xFF0000, 0x0000FF, 0x00FF00
QUOTA_WAIT = 62.0                     # s between tests: the per-minute quota refills


class Watcher:
    """A plain LAN connection that answers polls (and records any push)."""

    def __init__(self, ip):
        self.sock = socket.create_connection((ip, LAMP_PORT), 3)
        self.pushes = []
        self.replies = {}
        self.cv = threading.Condition()
        self.n = 1000
        self.spent = 0
        self.alive = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b""
        while self.alive:
            try:
                chunk = self.sock.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            now = time.monotonic()
            buf += chunk
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                with self.cv:
                    if msg.get("method") == "props":
                        self.pushes.append((now, msg["params"]))
                    else:
                        self.replies[msg.get("id")] = (now, msg)
                        self.cv.notify_all()

    def get(self, *props, timeout=0.6):
        """-> (time the bulb answered, as the midpoint of the round trip,
        {prop: value}) or (None, error)."""
        with self.cv:
            self.n += 1
            n = self.n
        self.spent += 1
        t0 = time.monotonic()
        # the firmware acts on a command only once another byte follows it
        self.sock.sendall((json.dumps({"id": n, "method": "get_prop", "params": list(props)})
                           + "\r\n").encode() + b" ")
        with self.cv:
            self.cv.wait_for(lambda: n in self.replies, timeout)
            t1, msg = self.replies.pop(n, (None, {"error": "timeout"}))
        if t1 is None or not isinstance(msg.get("result"), list):
            return None, msg.get("error")
        return (t0 + t1) / 2, dict(zip(props, msg["result"]))

    def end_of_flow(self, sent, expect, spacing=0.0, limit=12, margin=0.035):
        """Wait for `flowing` to read 1 (the start_cf left at `sent` and may
        not have reached the bulb yet), then poll from `expect` - `margin`
        seconds after `sent` until it reads 0. -> (end time = midway between
        the last 1 and the first 0, its +- error, polls), or None for the
        time when it was not seen running or ending."""
        polls = 0
        time.sleep(max(0.0, sent + 0.025 - time.monotonic()))
        last_on = None
        for _ in range(3):
            t, vals = self.get("flowing")
            polls += 1
            if t is not None and vals.get("flowing") == "1":
                last_on = t
                break
        if last_on is None:
            return None, None, polls
        time.sleep(max(0.0, sent + expect - margin - time.monotonic()))
        for _ in range(limit):
            t, vals = self.get("flowing")
            polls += 1
            if t is None:
                continue
            if vals.get("flowing") == "0":
                return (last_on + t) / 2, (t - last_on) / 2, polls
            last_on = t
            if spacing:
                time.sleep(spacing)
        return None, None, polls

    def close(self):
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass


def flow(bulb, count, action, steps):
    """start_cf on the music socket. steps = [(ms, mode, value, bright)].
    -> the time it left this machine."""
    expr = ", ".join(f"{ms}, {mode}, {val}, {br}" for ms, mode, val, br in steps)
    bulb.send_command("start_cf", [count, action, expr])
    return time.monotonic()


def rgb_steps(ms, colours, bright=60):
    return [(ms, 1, c, bright) for c in colours]


def ms(x):
    return None if x is None else round(1000 * x, 1)


def settle(w, secs=0.4):
    time.sleep(secs)


# --------------------------------------------------------------------------
def t_count(link, w):
    """6 steps of 300 ms: ends at ~1800 ms, on the 6th step's colour (blue)."""
    out = []
    for _ in range(2):
        t0 = flow(link.bulb, 6, 1, rgb_steps(300, [RED, BLUE]))
        mid, err, n = w.end_of_flow(t0, 1.8, margin=0.08)
        _, vals = w.get("rgb", "flowing")
        out.append({"end_ms": ms(mid and mid - t0), "err_ms": ms(err), "rgb": vals.get("rgb")
                    if isinstance(vals, dict) else vals})
        print(f"count: 6 x 300 ms ended at +{out[-1]['end_ms']} ms (+-{out[-1]['err_ms']}),"
              f" rgb then {out[-1]['rgb']} (blue = {BLUE})")
        settle(w)
    return out


def t_min(link, w):
    """8 steps of `step` ms: a bulb that stretches short steps takes longer
    than 8 x step; one that honours them ends on time."""
    out = {}
    for step in (50, 20, 10):
        t0 = flow(link.bulb, 8, 1, rgb_steps(step, [RED, BLUE]))
        mid, err, n = w.end_of_flow(t0, 0.04, spacing=0.02, limit=12, margin=0.0)
        out[step] = {"end_ms": ms(mid and mid - t0), "err_ms": ms(err)}
        print(f"min: 8 x {step} ms (asked {8 * step} ms) ended at +{out[step]['end_ms']} ms"
              f" (+-{out[step]['err_ms']}), {n} polls")
        time.sleep(0.3)
    return out


def _after(w, t, keys=("flowing", "rgb"), polls=3, gap=0.3):
    seen = []
    time.sleep(max(0.0, t - time.monotonic()))
    for _ in range(polls):
        _, vals = w.get(*keys)
        seen.append(vals)
        time.sleep(gap)
    return seen


def t_replace(link, w):
    bulb = link.bulb
    out = {}
    flow(bulb, 0, 1, rgb_steps(400, [RED, BLUE]))          # endless
    time.sleep(1.0)
    t1 = flow(bulb, 1, 1, [(50, 1, GREEN, 60)])             # a frame, as send_frame does
    out["start_cf"] = _after(w, t1 + 0.15)
    flow(bulb, 0, 1, rgb_steps(400, [RED, BLUE]))
    time.sleep(1.0)
    bulb.send_command("set_rgb", [GREEN, "sudden", 0])
    out["set_rgb"] = _after(w, time.monotonic() + 0.15)
    for k, v in out.items():
        print(f"replace by {k}: {v}  (want flowing 0 and rgb {GREEN} each time)")
    return out


def t_stop(link, w):
    bulb = link.bulb
    flow(bulb, 0, 1, rgb_steps(400, [RED, BLUE]))
    time.sleep(1.0)
    bulb.send_command("stop_cf", [])
    seen = _after(w, time.monotonic() + 0.15)
    print(f"stop_cf: {seen}  (want flowing 0, rgb held)")
    return seen


def t_tick(link, w):
    """A bulb-timed beat as the desk sends it: wait `hold` ms, then a 50 ms
    flash. With holds of 50 and 80 ms alternated, the ends should sit 30 ms
    apart and each set tight, if the bulb times the wait to the ms."""
    out = {50: [], 80: []}
    for _ in range(5):
        for hold in (50, 80):
            t0 = flow(link.bulb, 2, 1, [(hold, 1, RED, 30), (50, 1, BLUE, 60)])
            mid, err, n = w.end_of_flow(t0, (hold + 50) / 1000, limit=3, margin=0.02)
            out[hold].append((ms(mid and mid - t0), ms(err)))
            time.sleep(0.25)
    for hold, rows in out.items():
        ends = [e for e, _ in rows if e is not None]
        sd = round(statistics.pstdev(ends), 1) if len(ends) > 1 else None
        print(f"tick: {hold} + 50 ms ended at {rows} ms (end, +-), sd {sd}")
    return out


def t_timing(link, w):
    """Short and long flows, alternated. Each end is pinned to about +-7 ms;
    across trials its spread holds our send's Wi-Fi jitter and the poll's
    for both. Jitter in the bulb's steps would add up over 24 of them and
    widen the long flow's spread; a clock off true shows as the long one's
    end drifting from 24 x 50."""
    plans = {"short 2 x 250": (2, 250, 0.035, 0), "long 24 x 50": (24, 50, 0.09, 0.015)}
    out = {k: [] for k in plans}
    for _ in range(3):
        for name, (n, step, margin, spacing) in plans.items():
            t0 = flow(link.bulb, n, 1, rgb_steps(step, [RED, BLUE]))
            mid, err, polls = w.end_of_flow(t0, n * step / 1000, spacing=spacing,
                                            limit=9, margin=margin)
            out[name].append((ms(mid and mid - t0), ms(err)))
            time.sleep(0.3)
    for name, rows in out.items():
        n, step = plans[name][:2]
        ends = [e for e, _ in rows if e is not None]
        sd = round(statistics.pstdev(ends), 1) if len(ends) > 1 else None
        print(f"timing {name} (asked {n * step} ms): ends {ends} ms, sd {sd},"
              f" sighting +-{[x for _, x in rows]}")
    return out


def t_rtt(link, w):
    """The Wi-Fi under our own sends: 50 get_prop round trips, 80 ms apart."""
    rtts = []
    for _ in range(50):
        t0 = time.monotonic()
        t, vals = w.get("power")
        if t is not None:
            rtts.append(2 * (t - t0))
        time.sleep(0.08)
    rtts.sort()
    q = lambda p: round(1000 * rtts[min(len(rtts) - 1, int(p * len(rtts)))], 1)
    out = {"n": len(rtts), "p50": q(.5), "p90": q(.9), "max": q(1.0),
           "sd": round(1000 * statistics.pstdev(rtts), 1)}
    print(f"rtt: {out} ms")
    return out


TESTS = {"count": t_count, "min": t_min, "replace": t_replace, "stop": t_stop,
         "tick": t_tick, "timing": t_timing, "rtt": t_rtt}


def session(ip, name, memo):
    link = Link(ip, status=lambda word, note=None: note and print(" ", note[1]))
    link.memo = memo                     # every test restores the look before the first
    try:
        if not link.open():
            sys.exit("stopped")
    except BulbUnreachable as exc:
        sys.exit(f"could not open: {exc}")
    link.sender.stop()                   # every send here is raw, on this thread
    link.sender = None
    w = None
    try:
        w = Watcher(ip)
        result = TESTS[name](link, w)
        print(f"  ({w.spent} polls)")
        return result
    finally:
        if w:
            w.close()
        link.close(restore=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ip")
    ap.add_argument("test", nargs="+", choices=(*TESTS, "all"))
    ap.add_argument("--json", help="also write the results here")
    args = ap.parse_args()
    names = list(TESTS) if "all" in args.test else args.test
    results, memo = {}, {}
    try:
        for i, name in enumerate(names):
            if i:
                time.sleep(QUOTA_WAIT)
            results[name] = session(args.ip, name, memo)
    except KeyboardInterrupt:
        pass
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
