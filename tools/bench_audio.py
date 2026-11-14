"""What an audio input costs, across every process that pays for it.

    python tools/bench_audio.py                       # system audio vs the devices
    python tools/bench_audio.py --inputs "System audio" --secs 30

Each input runs in a child for --secs through the session's own audio path
(`record(BLOCK)` -> `Analyser.feed`, no bulb). The child and the daemons an
audio capture leans on are read before and after: CPU time from `ps` (% of
one core) and context switches a second from `top` - how often the process
was woken, which is what a battery pays for beyond the CPU figure. A
baseline with nothing running comes first and "idle" is a child that only
sleeps; read the daemons against the baseline, not as absolutes, and expect
WindowServer to swing by several points on its own.

No bulb, no network.
"""
import argparse
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DAEMONS = ("replayd", "coreaudiod", "WindowServer")


def worker(name, secs):
    import warnings
    warnings.simplefilter("ignore")
    if name == "idle":
        time.sleep(secs)
        return
    import music_bulb as m
    an = m.Analyser()
    with m.pick_recorder(name, quiet=True).recorder(samplerate=m.SAMPLERATE,
                                                    channels=2) as rec:
        end = time.monotonic() + secs
        while time.monotonic() < end:
            an.feed(rec.record(numframes=m.BLOCK))


def snapshot(child):
    """-> {name: (cpu seconds, context switches)} for the child and DAEMONS."""
    cpu, pids = {}, {}
    out = subprocess.run(["ps", "-axo", "pid=,time=,comm="],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(?:(\d+):)?(\d+):([\d.]+)\s+(.*)$", line)
        if not m:
            continue
        pid, name = int(m.group(1)), m.group(5).rsplit("/", 1)[-1]
        key = "child" if pid == child else name if name in DAEMONS else None
        if key:
            secs = int(m.group(2) or 0) * 3600 + int(m.group(3)) * 60 + float(m.group(4))
            cpu[key] = cpu.get(key, 0.0) + secs
            pids[pid] = key
    args = ["top", "-l", "1", "-stats", "pid,csw"]
    for pid in pids:
        args += ["-pid", str(pid)]
    csw = {}
    for pid, n in re.findall(r"^\s*(\d+)\s+(\d+)", subprocess.run(
            args, capture_output=True, text=True).stdout, re.M):
        key = pids.get(int(pid))
        if key:
            csw[key] = csw.get(key, 0) + int(n)
    return {k: (cpu[k], csw.get(k, 0)) for k in cpu}


def measure(name, secs):
    """One row: {process: (% of a core, context switches a second)}."""
    child = None
    if name:
        child = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                  "--worker", name, "--secs", str(secs + 10)])
        time.sleep(4)                           # imports and the stream's start
    pid = child.pid if child else -1
    a, t0 = snapshot(pid), time.monotonic()
    time.sleep(secs)
    b, dt = snapshot(pid), time.monotonic() - t0
    if child:
        child.terminate()
        child.wait()
    return {k: (100 * (b[k][0] - a[k][0]) / dt, (b[k][1] - a[k][1]) / dt)
            for k in b if k in a}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--secs", type=int, default=20)
    p.add_argument("--inputs", default=None,
                   help="comma-separated input names (default: System audio "
                        "and every capture device)")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        return worker(args.worker, args.secs)

    if args.inputs:
        names = [n.strip() for n in args.inputs.split(",")]
    else:
        import warnings
        warnings.simplefilter("ignore")
        import music_bulb_dash as dash
        names = dash.audio_inputs()["inputs"]
    cols = ("child",) + DAEMONS
    print(f"{args.secs} s each; cells are % of a core / context switches a second\n")
    print(f"{'':24s}" + "".join(f"{c:>14s}" for c in cols))
    for label, name in [("baseline", None), ("idle", "idle")] + [(n, n) for n in names]:
        row = measure(name, args.secs)
        print(f"{label[:23]:24s}" + "".join(
            f"{row.get(c, (0.0, 0.0))[0]:7.2f}/{row.get(c, (0.0, 0.0))[1]:6.0f}"
            for c in cols), flush=True)


if __name__ == "__main__":
    main()
