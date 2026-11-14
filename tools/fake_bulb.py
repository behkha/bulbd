"""A stand-in color5 on this machine, for trying the desk without a lamp.

    python tools/fake_bulb.py                    # a healthy bulb at 127.0.0.1
    python tools/fake_bulb.py --mode refuse      # port closed: "LAN control is off"
    python tools/fake_bulb.py --mode silent      # takes the connection, never answers: "busy"
    python tools/fake_bulb.py --drop-after 20    # healthy, then vanishes for --gone seconds

Point a desk at it with its own settings file:
    python music_bulb_dash.py --port 8768 --ip 127.0.0.1 --settings /tmp/x.json

It answers the miio hello on UDP 54321 (127.0.0.1 only), the LAN API on TCP
55443 (get_prop and the set_* commands the restore uses) and music mode
(set_music: it connects back and reads the start_cf frames). Each frame it
receives is printed, so white frames and early flashes can be seen.

Colour flows behave as the color5 was measured to: start_cf is ignored by a
lamp that is off, set_scene cf lights it; end action 0 puts back the state
from before the flow, 1 stays on the last step, 2 turns off; more than 16
steps is refused with -5001; get_prop reports the pre-flow state and
`flowing` 1 while one runs; stop_cf leaves the lamp on the current step.
"""
import argparse
import json
import socket
import threading
import time

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--mode", choices=("ok", "refuse", "silent"), default="ok")
ap.add_argument("--drop-after", type=float, default=0.0, metavar="S",
                help="close everything after this many seconds of music mode")
ap.add_argument("--gone", type=float, default=6.0, metavar="S",
                help="how long it stays away after --drop-after (default 6)")
ap.add_argument("--did", type=lambda s: int(s, 16), default=0x12AB34CD)
ap.add_argument("--quiet", action="store_true", help="do not print every frame")
ap.add_argument("--port", type=int, default=55443,
                help="LAN API port (default 55443); another port lets a second "
                     "fake run beside one already on 55443, for a desk whose "
                     "bulb port is patched to match")
args = ap.parse_args()

state = {"power": "off", "bright": "30", "color_mode": "2", "rgb": "16711680",
         "ct": "2700", "hue": "0", "sat": "100", "flowing": "0", "music_on": "0"}
MAX_STEPS = 16
flow = {"timer": None, "end": None}
flow_lock = threading.Lock()


def parse_flow(count, expr):
    """-> [(ms, mode, value, bright)] or None if the firmware would refuse it."""
    try:
        nums = [int(x) for x in str(expr).split(",")]
    except ValueError:
        return None
    if len(nums) % 4 or not 1 <= len(nums) // 4 <= MAX_STEPS or int(count) < 0:
        return None
    return [tuple(nums[i:i + 4]) for i in range(0, len(nums), 4)]


def step_state(step):
    ms, mode, value, bright = step
    out = {"bright": str(max(1, min(bright, 100)))} if bright > 0 else {}
    if mode == 1:
        out.update(rgb=str(value), color_mode="1")
    elif mode == 2:
        out.update(ct=str(value), color_mode="2")
    return out


def run_flow(count, action, steps, light_it):
    """Start a flow; returns False if the lamp is off and nothing lights it."""
    if state["power"] != "on" and not light_it:
        return True                                  # "ok", and nothing happens
    with flow_lock:
        if flow["timer"]:
            flow["timer"].cancel()
        before = {k: state[k] for k in ("power", "bright", "color_mode", "rgb", "ct")}
        n = int(count) or len(steps)
        run = [steps[i % len(steps)] for i in range(n)]
        total = sum(s[0] for s in run) / 1000.0
        last = step_state(run[-1])
        state.update(power="on", flowing="1")

        def done():
            with flow_lock:
                state["flowing"] = "0"
                if int(action) == 0:
                    state.update(before)
                else:
                    state.update(last)
                    if int(action) == 2:
                        state["power"] = "off"
                flow["timer"] = flow["end"] = None
            print(f"flow over (action {action}):", {k: state[k] for k in before})

        flow["end"] = last
        flow["timer"] = threading.Timer(total, done)
        flow["timer"].daemon = True
        flow["timer"].start()
    print(f"flow: {n} steps, {total:.1f} s, action {action}")
    return True


def stop_flow():
    with flow_lock:
        if flow["timer"]:
            flow["timer"].cancel()
            state.update(flow["end"] or {})
        state["flowing"] = "0"
        flow["timer"] = flow["end"] = None
PROPS = ("power", "bright", "color_mode", "rgb", "ct", "hue", "sat")
away = threading.Event()
conns = []


def hello():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", 54321))
    except OSError:
        print("UDP 54321 is taken (another fake?): not answering the hello")
        return
    while True:
        data, addr = s.recvfrom(64)
        if data[:4] == bytes.fromhex("21310020") and not away.is_set():
            s.sendto(bytes.fromhex("21310020") + b"\0" * 4 + args.did.to_bytes(4, "big")
                     + int(time.time()).to_bytes(4, "big") + b"\xff" * 16, addr)


def music(host, port):
    c = socket.create_connection((host, port), 3)
    conns.append(c)
    print(f"music mode: connected back to {host}:{port}")
    started = time.monotonic()
    buf = b""
    try:
        while True:
            if args.drop_after and time.monotonic() - started > args.drop_after:
                print(f"dropping everything for {args.gone} s")
                away.set()
                for x in list(conns):
                    try:
                        x.shutdown(socket.SHUT_RDWR)
                        x.close()
                    except OSError:
                        pass
                time.sleep(args.gone)
                away.clear()
                print("back")
                return
            c.settimeout(0.5)
            try:
                chunk = c.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                return
            buf += chunk
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                if not args.quiet:
                    print(f"{time.monotonic():.3f}", msg["method"], msg["params"])
                p = msg.get("params", [])
                # a long flow in music mode is an alert (frames are one-step flows)
                if msg["method"] == "start_cf" and len(p) == 3 and int(p[0]) > 1:
                    steps = parse_flow(p[0], p[2])
                    if steps:
                        run_flow(p[0], p[1], steps, light_it=False)
    except OSError:
        return


def control(c):
    conns.append(c)
    buf = b""
    try:
        while True:
            chunk = c.recv(4096)
            if not chunk:
                return
            if args.mode == "silent":
                continue
            buf += chunk
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                method, params = msg["method"], msg.get("params", [])
                result = ["ok"]
                error = None
                if method in ("start_cf", "set_scene"):
                    cf = params if method == "start_cf" else params[1:]
                    steps = parse_flow(cf[0], cf[2]) if len(cf) == 3 and \
                        (method == "start_cf" or params[0] == "cf") else None
                    if method == "set_scene" and params and params[0] == "ct":
                        state.update(power="on", ct=str(params[1]), bright=str(params[2]),
                                     color_mode="2")
                    elif steps is None:
                        error = {"code": -5001, "message": "invalid params"}
                    else:
                        run_flow(cf[0], cf[1], steps, light_it=method == "set_scene")
                elif method == "stop_cf":
                    stop_flow()
                elif state["power"] != "on" and method in ("set_rgb", "set_ct_abx", "set_bright"):
                    pass                                     # an off lamp says ok and ignores it
                elif method == "get_prop":
                    result = [state.get(p, "") for p in params]
                elif method == "set_power":
                    state["power"] = params[0]
                elif method == "set_bright":
                    state["bright"] = str(params[0])
                elif method == "set_ct_abx":
                    state.update(ct=str(params[0]), color_mode="2")
                elif method == "set_rgb":
                    state.update(rgb=str(params[0]), color_mode="1")
                elif method == "set_music" and params and params[0] == 0:
                    state["music_on"] = "0"
                elif method == "set_music" and params and params[0] == 1:
                    state["power"] = "on"
                    state["music_on"] = "1"
                    threading.Thread(target=music, args=(params[1], params[2]), daemon=True).start()
                print("control:", method, params, error or "")
                reply = {"id": msg["id"], "error": error} if error else {"id": msg["id"], "result": result}
                c.sendall((json.dumps(reply) + "\r\n").encode())
    except OSError:
        return


threading.Thread(target=hello, daemon=True).start()
print(f"fake bulb {args.did:08x} on 127.0.0.1 ({args.mode})")
while True:
    if args.mode == "refuse":
        time.sleep(3600)
        continue
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(8)
    srv.settimeout(0.5)
    while not away.is_set():
        try:
            c, _ = srv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=control, args=(c,), daemon=True).start()
    srv.close()
    while away.is_set():
        time.sleep(0.2)
