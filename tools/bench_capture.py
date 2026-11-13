#!/usr/bin/env python3
"""
Screen capture cost: mss vs a ScreenCaptureKit stream, no bulb involved.

Each backend runs in a child process through the real colour pipeline - the
CLI's screen loop without the sender: Capture -> crop_letterbox ->
ScreenColorEngine -> screen_frame, paced by CapturePacer - on a moving and
on a still picture. Meanwhile `top -l` samples the CPU of the child,
WindowServer, tccd, trustd, syspolicyd and replayd. A baseline with no
capture is measured the same way and shown, so load from anything else
(another capture, a playing film) can be told apart.

    python3 tools/bench_capture.py                       # finds the boxes itself
    python3 tools/bench_capture.py --secs 30 --moving 2 --still 1:1500,950,300,200
    python3 tools/bench_capture.py --compare 20          # colour agreement only

A box is MONITOR or MONITOR:X,Y,W,H (X, Y in that display's own points).
Without --moving / --still, each display and its quarters are watched for
a few seconds with mss and the most and least changing are used.

--compare runs both backends side by side on the same box and reports how
far apart the lamp's hue, saturation and brightness end up, plus a
full-resolution pixel check that the two read the same colour space.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                                   # noqa: E402

import music_bulb as mb                                              # noqa: E402

DAEMONS = ("WindowServer", "tccd", "trustd", "syspolicyd", "replayd")


# ---------------------------------------------------------------------------
def parse_box(text):
    mon, _, rest = text.partition(":")
    region = tuple(int(v) for v in rest.split(",")) if rest else None
    if region is not None and len(region) != 4:
        raise argparse.ArgumentTypeError("expected MONITOR or MONITOR:X,Y,W,H")
    return int(mon), region


def absolute(monitor, region):
    """A display-local region -> the absolute box Capture takes."""
    if region is None:
        return None
    m = mb.monitor_box(monitor)
    x, y, w, h = region
    return (m["left"] + x, m["top"] + y, w, h)


def describe(box):
    mon, region = box
    return f"display {mon}" + (f" {region[2]}x{region[3]} at {region[0]},{region[1]}"
                               if region else " (whole)")


def find_boxes(secs=4.0):
    """-> (moving box, still box) from what is on screen right now."""
    import mss
    with mss.MSS() as sct:
        mons = sct.monitors[1:]
    cands = []
    for i, m in enumerate(mons, start=1):
        cands.append((i, None))
        hw, hh = m["width"] // 2, m["height"] // 2
        for qx in (0, hw):
            for qy in (0, hh):
                cands.append((i, (qx, qy, hw, hh)))
    grabs = {c: mb.ScreenGrabber(c[0], absolute(*c)) for c in cands}
    first = {c: g.frame() for c, g in grabs.items()}
    moved = {c: 0.0 for c in cands}
    t_end = time.monotonic() + secs
    while time.monotonic() < t_end:
        time.sleep(0.25)
        for c, g in grabs.items():
            f = g.frame()
            moved[c] = max(moved[c], float((np.abs(f - first[c]).max(axis=2) > 0.04).mean()))
    for g in grabs.values():
        g.close()
    moving = max(cands, key=lambda c: (moved[c], c[1] is None))
    still = min(cands, key=lambda c: (moved[c], c[1] is not None))
    return moving, still, moved


# ---------------------------------------------------------------- the worker
def worker(args):
    """One backend through the colour pipeline for --secs, then a JSON line."""
    import resource
    grab = mb.Capture(args.monitor, absolute(args.monitor, args.region), 192,
                      args.fps, args.worker)
    color = mb.ScreenColorEngine()
    pacer = mb.CapturePacer()
    print(json.dumps({"ready": True, "backend": grab.backend, "note": grab.note,
                      "pid": os.getpid()}), flush=True)
    sys.stdin.readline()                       # the parent starts top, then says go
    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.monotonic()
    ticks, last = 0, t0
    while time.monotonic() - t0 < args.secs:
        now = time.monotonic()
        dt, last = now - last, now
        frame = grab.frame()
        hue, sat, val, cut = color.tick(dt, *color.measure(mb.crop_letterbox(frame)))
        mb.screen_frame(color, hue, sat, int(3 + 97 * val))
        ticks += 1
        grab.pace(now, 1.0 / args.fps, pacer.period(frame, args.fps, now))
    r1, wall = resource.getrusage(resource.RUSAGE_SELF), time.monotonic() - t0
    impl = grab._impl
    out = {"backend": grab.backend, "ticks": ticks, "wall": wall,
           "cpu": (r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime) / wall,
           "frames": getattr(impl, "delivered", ticks),
           "idle": getattr(impl, "idle", 0)}
    grab.close()
    print(json.dumps(out), flush=True)


# ---------------------------------------------------------------- measuring
def top_sample(secs):
    """CPU per process over `secs` from top's second sample. -> ({pid: %},
    {command: summed %})."""
    out = subprocess.run(["top", "-l", "2", "-s", str(int(secs)), "-stats", "pid,command,cpu",
                          "-o", "cpu", "-n", "400"], capture_output=True, text=True).stdout
    part = out.split("PID")[-1].splitlines()[1:]
    by_pid, by_cmd = {}, {}
    for line in part:
        m = re.match(r"\s*(\d+)\s+(.*?)\s+([\d.]+)\s*$", line)
        if not m:
            continue
        pid, cmd, cpu = int(m.group(1)), m.group(2).strip(), float(m.group(3))
        by_pid[pid] = cpu
        by_cmd[cmd] = by_cmd.get(cmd, 0.0) + cpu
    return by_pid, by_cmd


def run_backend(backend, box, secs, fps):
    mon, region = box
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", backend,
           "--monitor", str(mon), "--secs", str(secs), "--fps", str(fps)]
    if region:
        cmd += ["--region", ",".join(map(str, region))]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    hello = json.loads(p.stdout.readline())
    time.sleep(2.0)                                       # past the start-up
    result = {}

    def sample():
        result["top"] = top_sample(secs - 1)

    t = threading.Thread(target=sample)
    p.stdin.write("go\n")
    p.stdin.flush()
    t.start()
    done = json.loads(p.stdout.readline())
    t.join()
    p.wait(10)
    by_pid, by_cmd = result["top"]
    row = {d: by_cmd.get(d, 0.0) for d in DAEMONS}
    row.update(python=by_pid.get(hello["pid"], 0.0), self=done["cpu"] * 100,
               frames=done["frames"], ticks=done["ticks"], idle=done["idle"],
               backend=done["backend"], note=hello["note"], wall=done["wall"])
    return row


def baseline(secs):
    _, by_cmd = top_sample(secs)
    row = {d: by_cmd.get(d, 0.0) for d in DAEMONS}
    row.update(python=0.0, self=0.0, frames=0, ticks=0, idle=0, backend="none", note="")
    return row


def table(rows):
    head = ("picture", "backend", "frames", "ticks", "Python", "WindowServer", "tccd", "trustd",
            "syspolicyd", "replayd", "total")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, r in rows:
        total = r["self"] + sum(r[d] for d in DAEMONS)
        frames = f"{r['frames']}" + (f" (+{r['idle']} idle)" if r["idle"] else "")
        print(f"| {name} | {r['backend']} | {frames} | {r['ticks']} | {r['self']:.1f} | "
              + " | ".join(f"{r[d]:.1f}" for d in DAEMONS) + f" | {total:.1f} |")


# ------------------------------------------------------------------ colour
def compare(box, secs, fps):
    """Both backends on the same box at once; the lamp's numbers side by side."""
    mon, region = box
    reg = absolute(mon, region)
    sck = mb.Capture(mon, reg, 192, fps, "sck")
    if sck.backend != "sck":
        print(f"no stream: {sck.note}")
        return
    mss_ = mb.Capture(mon, reg, 192, fps, "mss")
    engines = (mb.ScreenColorEngine(), mb.ScreenColorEngine())
    rows, t0, last = [], time.monotonic(), time.monotonic()
    pacer = mb.CapturePacer()
    while time.monotonic() - t0 < secs:
        now = time.monotonic()
        dt, last = now - last, now
        fs = (sck.frame(), mss_.frame())
        meas = [e.measure(mb.crop_letterbox(f)) for e, f in zip(engines, fs)]
        outs = [e.tick(dt, *m) for e, m in zip(engines, meas)]
        rows.append((meas, outs))
        sck.pace(now, 1.0 / fps, pacer.period(fs[0], fps, now))
    # full resolution, a still moment: are the pixels the same colours?
    w = mb.monitor_box(mon)["width"] if reg is None else reg[2]
    full_s = mb.Capture(mon, reg, w, fps, "sck")
    full_m = mb.ScreenGrabber(mon, reg, w)
    a, b = full_s.frame(), full_m.frame()
    full_s.close()
    full_m.close()
    sck.close()
    mss_.close()

    def circ(x, y):
        return abs(((x - y + 180.0) % 360.0) - 180.0)

    dh = [circ(o[0][0], o[1][0]) for _, o in rows if min(o[0][1], o[1][1]) >= 30]
    ds = [abs(o[0][1] - o[1][1]) for _, o in rows]
    dv = [abs(o[0][2] - o[1][2]) * 100 for _, o in rows]
    cuts = sum(o[0][3] != o[1][3] for _, o in rows)
    wh = sum(1 for _ in rows)

    def stats(xs):
        if not xs:
            return "-"
        return f"median {np.median(xs):.1f}, p95 {np.percentile(xs, 95):.1f}, max {max(xs):.1f}"

    print(f"\ncolour on {describe(box)}, {wh} paired ticks over {secs:.0f} s")
    print(f"  hue (deg, both >= 30 sat, {len(dh)} ticks): {stats(dh)}")
    print(f"  saturation (0..100): {stats(ds)}")
    print(f"  brightness (0..100): {stats(dv)}")
    print(f"  ticks where only one saw a cut: {cuts}")
    if a.shape == b.shape:
        d = np.abs(a - b) * 255
        print(f"  full-res pixels {a.shape[1]}x{a.shape[0]}: mean |diff| per channel "
              f"{d.reshape(-1, 3).mean(0).round(2).tolist()} (0..255), "
              f"{(d.max(axis=2) > 2).mean() * 100:.1f}% of pixels off by more than 2")
    else:
        print(f"  full-res shapes differ: {a.shape} vs {b.shape}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--secs", type=float, default=20.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--moving", type=parse_box)
    p.add_argument("--still", type=parse_box)
    p.add_argument("--backends", default="mss,sck")
    p.add_argument("--compare", type=float, metavar="SECS",
                   help="only compare the colour the two backends give")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--worker", choices=("sck", "mss"), help=argparse.SUPPRESS)
    p.add_argument("--monitor", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--region", type=lambda s: tuple(int(v) for v in s.split(",")),
                   help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        worker(args)
        return

    if args.moving is None or args.still is None:
        print("watching the screen for a moving and a still box...", flush=True)
        moving, still, moved = find_boxes()
        args.moving = args.moving or moving
        args.still = args.still or still
        print(f"  moving: {describe(args.moving)} ({moved.get(args.moving, 0) * 100:.1f}% "
              f"of pixels changed); still: {describe(args.still)} "
              f"({moved.get(args.still, 0) * 100:.1f}%)")

    if args.compare:
        for box in (args.moving, args.still):
            compare(box, args.compare, args.fps)
        return

    rows = []
    if not args.no_baseline:
        print(f"baseline, {args.secs:.0f} s...", flush=True)
        rows.append(("nothing", baseline(args.secs)))
    for name, box in (("moving", args.moving), ("still", args.still)):
        for backend in args.backends.split(","):
            print(f"{name} / {backend}, {args.secs:.0f} s...", flush=True)
            r = run_backend(backend, box, args.secs, args.fps)
            if r["backend"] != backend:
                print(f"  {backend} unavailable: {r['note']}")
            rows.append((name, r))
    print(f"\n% of one core, {args.secs:.0f} s each, {args.fps} fps; moving = "
          f"{describe(args.moving)}, still = {describe(args.still)}\n")
    table(rows)


if __name__ == "__main__":
    main()
