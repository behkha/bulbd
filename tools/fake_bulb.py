"""A stand-in color5 on this machine, for trying the desk without a lamp.

    python tools/fake_bulb.py                    # a healthy bulb at 127.0.0.1
    python tools/fake_bulb.py --mode refuse      # port closed: "LAN control is off"
    python tools/fake_bulb.py --mode silent      # takes the connection, never answers: "busy"
    python tools/fake_bulb.py --drop-after 20    # healthy, then vanishes for --gone seconds

Point a desk at it with its own settings file:
    python music_bulb_dash.py --port 8768 --ip 127.0.0.1 --settings /tmp/x.json

It answers the miio hello on UDP 54321, the LAN API on TCP 55443 (get_prop
and the set_* commands the restore and lamp mode use) and music mode
(set_music: it connects back and reads the start_cf frames). Each frame it
receives is printed, so white frames and early flashes can be seen.

What the lamp does on its own follows the color5 as measured:
- Flows. start_cf is ignored by a lamp that is off, set_scene cf lights it.
  A finite flow runs its end action when its steps are done: 0 puts back
  the state from before the flow, 1 stays on the last step, 2 turns off.
  Count 0 runs until something replaces it. stop_cf runs the end action at
  once; a new flow, set_scene or a plain set replaces a flow without it.
  More than 16 steps is refused with -5001. get_prop reports the pre-flow
  state, `flowing` 1 and `flow_params` while one runs.
- The off timer. cron_add / cron_get / cron_del and get_prop delayoff; the
  timer really switches the lamp off (--minute S shortens a minute for
  trying it) and is refused with -5006 while the lamp is off.
- set_default is kept and printed. toggle and adjust_bright work as on the
  lamp; an off lamp answers "ok" to set_bright / set_ct_abx / set_rgb /
  set_hsv / set_adjust and changes nothing.

--host ::1 or --port N runs a second fake beside one on 127.0.0.1:55443.
"""
import argparse
import json
import math
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
ap.add_argument("--minute", type=float, default=60.0, metavar="S",
                help="seconds in a cron minute (default 60)")
ap.add_argument("--quiet", action="store_true", help="do not print every frame")
ap.add_argument("--host", default="127.0.0.1",
                help="address to listen on; ::1 runs a second fake beside one on "
                     "127.0.0.1 (point the desk at --ip localhost)")
ap.add_argument("--port", type=int, default=55443,
                help="LAN API port (default 55443); another port lets a second "
                     "fake run beside one already on 55443, for a desk whose "
                     "bulb port is patched to match")
args = ap.parse_args()
FAMILY = socket.AF_INET6 if ":" in args.host else socket.AF_INET

state = {"power": "off", "bright": "30", "color_mode": "2", "rgb": "16711680",
         "ct": "2700", "hue": "0", "sat": "100", "flowing": "0", "music_on": "0",
         "flow_params": ""}
LOOK = ("bright", "color_mode", "rgb", "ct", "hue", "sat")
# a color5 that is off answers "ok" to these and changes nothing
IGNORED_OFF = ("set_bright", "set_adjust", "set_ct_abx", "set_rgb", "set_hsv")
MAX_STEPS = 16
away = threading.Event()
conns = []
lock = threading.RLock()
flow = {"timer": None, "action": 0, "before": None, "steps": None, "at": 0.0}
cron = {"timer": None, "at": 0.0}
default = {}


# -- flows --------------------------------------------------------------------
def parse_flow(count, expr):
    """-> [(ms, mode, value, bright)] or None if the firmware would refuse it."""
    try:
        nums = [int(x) for x in str(expr).split(",")]
        count = int(count)
    except (TypeError, ValueError):
        return None
    if len(nums) % 4 or not 1 <= len(nums) // 4 <= MAX_STEPS or count < 0:
        return None
    return [tuple(nums[i:i + 4]) for i in range(0, len(nums), 4)]


def step_state(step):
    _, mode, value, bright = step
    out = {"bright": str(max(1, min(bright, 100)))} if bright > 0 else {}
    if mode == 1:
        out.update(rgb=str(value), color_mode="1")
    elif mode == 2:
        out.update(ct=str(value), color_mode="2")
    return out


def drop_flow():
    """Forget a running flow without its end action (something replaced it)."""
    with lock:
        if flow["timer"]:
            flow["timer"].cancel()
        flow.update(timer=None, steps=None, before=None)
        state.update(flowing="0", flow_params="")


def end_flow(now_step=None):
    """The flow's end action. `now_step` is where a stop_cf caught it."""
    with lock:
        if flow["steps"] is None:
            return
        action, before, steps = flow["action"], flow["before"], flow["steps"]
        drop_flow()
        if action == 0:
            state.update(before)
        else:
            state.update(step_state(now_step or steps[-1]))
            if action == 2:
                state["power"] = "off"
    print(f"flow over (action {action}):", {k: state[k] for k in ("power",) + LOOK[:4]})


def run_flow(count, action, steps, light_it):
    """Start a flow. A lamp that is off takes one only from set_scene."""
    with lock:
        if state["power"] != "on" and not light_it:
            return                               # "ok", and nothing happens
        before = {k: state[k] for k in ("power",) + LOOK}
        drop_flow()
        count = int(count)
        run = [steps[i % len(steps)] for i in range(count)] if count else list(steps)
        total = sum(s[0] for s in run) / 1000.0
        flow.update(action=int(action), before=before, steps=run, at=time.monotonic())
        state.update(power="on", flowing="1",
                     flow_params=f"{count},{action}," + ",".join(
                         str(x) for s in steps for x in s))
        if count:
            flow["timer"] = threading.Timer(total, end_flow)
            flow["timer"].daemon = True
            flow["timer"].start()
    print(f"flow: {len(run)} steps, " + (f"{total:.1f} s" if count else "forever")
          + f", action {action}")


def stop_flow():
    """stop_cf: the end action now, from the step the flow has reached."""
    with lock:
        steps = flow["steps"]
        if steps is None:
            return
        gone, here = (time.monotonic() - flow["at"]) * 1000.0, steps[-1]
        if flow["timer"] is None:                # endless: where in the loop
            gone %= max(1, sum(s[0] for s in steps))
        for s in steps:
            if gone < s[0]:
                here = s
                break
            gone -= s[0]
        end_flow(here)


# -- the off timer ------------------------------------------------------------
class Refused(Exception):
    code = -5006


def cron_fire():
    with lock:
        cron["timer"] = None
        drop_flow()
        state["power"] = "off"
    print("cron fired, lamp off")


def cron_left():
    with lock:
        if cron["timer"] is None:
            return 0
        return max(0, math.ceil((cron["at"] - time.monotonic()) / args.minute))


def cron_del():
    with lock:
        if cron["timer"]:
            cron["timer"].cancel()
        cron["timer"] = None


def cron_add(minutes):
    with lock:
        if state["power"] != "on":
            raise Refused
        cron_del()
        cron["at"] = time.monotonic() + minutes * args.minute
        cron["timer"] = threading.Timer(minutes * args.minute, cron_fire)
        cron["timer"].daemon = True
        cron["timer"].start()


def prop(name):
    if name == "delayoff":
        return str(cron_left())
    return state.get(name, "")


# -- the wire -----------------------------------------------------------------
def hello():
    s = socket.socket(FAMILY, socket.SOCK_DGRAM)
    try:
        s.bind((args.host, 54321))
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
    finally:
        state["music_on"] = "0"


def handle(method, params):
    """One LAN API command. -> result list. Raises Refused or ValueError."""
    on = state["power"] == "on"
    if method == "get_prop":
        return [prop(p) for p in params]
    if method in ("start_cf", "set_scene"):
        kind = "cf" if method == "start_cf" else (params[0] if params else None)
        cf = params if method == "start_cf" else params[1:]
        if kind == "cf":
            steps = parse_flow(cf[0], cf[2]) if len(cf) == 3 else None
            if steps is None:
                raise ValueError
            run_flow(cf[0], cf[1], steps, light_it=method == "set_scene")
        elif kind == "ct":
            drop_flow()
            state.update(power="on", ct=str(cf[0]), bright=str(cf[1]), color_mode="2")
        elif kind == "color":
            drop_flow()
            state.update(power="on", rgb=str(cf[0]), bright=str(cf[1]), color_mode="1")
        else:
            raise ValueError
    elif method == "stop_cf":
        stop_flow()
    elif method == "cron_add":
        cron_add(int(params[1]))
    elif method == "cron_get":
        left = cron_left()
        return [{"type": 0, "delay": left, "mix": 0}] if left else []
    elif method == "cron_del":
        cron_del()
    elif method == "set_default":
        default.update({k: state[k] for k in LOOK})
        print("power-on look saved", default)
    elif method == "set_power":
        if params[0] == "off":
            drop_flow()
        state["power"] = params[0]
    elif method == "toggle":
        if on:
            drop_flow()
        state["power"] = "off" if on else "on"
    elif method == "adjust_bright":
        # as measured on the color5: an off lamp comes on at the step, and
        # going under 1 switches it off
        if not on:
            if params[0] > 0:
                state.update(power="on", bright=str(min(100, params[0])))
        elif int(state["bright"]) + params[0] < 1:
            drop_flow()
            state["power"] = "off"
        else:
            state["bright"] = str(min(100, int(state["bright"]) + params[0]))
    elif method in IGNORED_OFF and not on:
        pass                                    # answered "ok", changes nothing
    elif method == "set_bright":
        state["bright"] = str(params[0])
    elif method == "set_ct_abx":
        drop_flow()
        state.update(ct=str(params[0]), color_mode="2")
    elif method == "set_rgb":
        drop_flow()
        state.update(rgb=str(params[0]), color_mode="1")
    elif method == "set_music" and params and params[0] == 0:
        state["music_on"] = "0"
    elif method == "set_music" and params and params[0] == 1:
        state.update(power="on", music_on="1")
        threading.Thread(target=music, args=(params[1], params[2]), daemon=True,
                         name="music").start()
    return ["ok"]


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
                try:
                    with lock:
                        reply = {"id": msg["id"], "result": handle(method, params)}
                    note = ""
                except Refused:
                    reply = {"id": msg["id"], "error": {"code": -5006,
                                                        "message": "illegal request"}}
                    note = "-> illegal request"
                except (ValueError, IndexError, TypeError):
                    reply = {"id": msg["id"], "error": {"code": -5001,
                                                        "message": "invalid params"}}
                    note = "-> invalid params"
                print("control:", method, params, note)
                c.sendall((json.dumps(reply) + "\r\n").encode())
    except OSError:
        return


threading.Thread(target=hello, daemon=True).start()
print(f"fake bulb {args.did:08x} on {args.host}:{args.port} ({args.mode})")
while True:
    if args.mode == "refuse":
        time.sleep(3600)
        continue
    srv = socket.socket(FAMILY, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
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
