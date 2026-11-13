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
args = ap.parse_args()

state = {"power": "off", "bright": "30", "color_mode": "2", "rgb": "16711680",
         "ct": "2700", "hue": "0", "sat": "100"}
PROPS = ("power", "bright", "color_mode", "rgb", "ct", "hue", "sat")
away = threading.Event()
conns = []


def hello():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 54321))
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
                if line and not args.quiet:
                    msg = json.loads(line)
                    print(f"{time.monotonic():.3f}", msg["method"], msg["params"])
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
                if method == "get_prop":
                    result = [state.get(p, "") for p in params]
                elif method == "set_power":
                    state["power"] = params[0]
                elif method == "set_bright":
                    state["bright"] = str(params[0])
                elif method == "set_ct_abx":
                    state.update(ct=str(params[0]), color_mode="2")
                elif method == "set_rgb":
                    state.update(rgb=str(params[0]), color_mode="1")
                elif method == "set_music" and params and params[0] == 1:
                    state["power"] = "on"
                    threading.Thread(target=music, args=(params[1], params[2]), daemon=True).start()
                print("control:", method, params)
                c.sendall((json.dumps({"id": msg["id"], "result": result}) + "\r\n").encode())
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
    srv.bind(("127.0.0.1", 55443))
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
