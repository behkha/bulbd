#!/usr/bin/env python3
"""
music_bulb_dash.py - a local control surface for music_bulb.py

Run it, open the URL it prints, and drive everything from the browser:
music mode or screen mode, palette, sensitivity, contrast, punch, display,
refresh rate - all live, with a preview of the colour the bulb is currently
being sent.

    python music_bulb_dash.py --ip <bulb-ip>

<bulb-ip> is whatever address your own router gave the bulb; there is no
default. You can also start with no --ip at all and set the address on the
cloud deck, which lists every device on the Xiaomi account with its LAN
address and its miio token.

Keep this file next to music_bulb.py; it imports the DSP from there.
The server binds to localhost only. Pass --host 0.0.0.0 to reach it from
your phone on the same network (there is no authentication, so only do
that on a network you trust).
"""

import argparse
import base64
import contextlib
import io
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from music_bulb import (PALETTES, SAMPLERATE, BLOCK, Analyser, ColorEngine,
                        LevelEngine, Sender, pick_recorder, enable_lan,
                        ScreenGrabber, ScreenColorEngine, ScreenColorThread,
                        CapturePacer, crop_letterbox, screen_permission,
                        open_bulb, release_bulb)

# The cloud half is optional: it pulls in requests/PIL/pycryptodome, and a
# desk that only drives an already-provisioned bulb should still start
# without them. Everything token-related checks `cloud` first.
try:
    import token_extractor as cloud
    CLOUD_ERROR = ""
except Exception as exc:                                    # pragma: no cover
    cloud = None
    CLOUD_ERROR = f"{type(exc).__name__}: {exc}"

HOSTNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62})$")


# ---------------------------------------------------------------------------
def screen_monitors():
    """[{i, w, h}] for every real display, or [] if mss is unavailable."""
    try:
        import mss
        with mss.MSS() as sct:
            return [{"i": i, "w": m["width"], "h": m["height"]}
                    for i, m in enumerate(sct.monitors) if i > 0]
    except Exception:
        return []


class State:
    def __init__(self, ip, mic):
        self._lock = threading.Lock()
        # the engine parks on `started` instead of polling for a session, and
        # a loop idling on a still picture ends its wait early on `changed`
        self.started = threading.Event()
        self.changed = threading.Event()
        mons = screen_monitors()
        self._d = {
            "ip": ip, "mic": mic,
            "running": False, "status": "idle",
            "mode": "music",
            "palette": "auto", "calm": False,
            "sensitivity": 1.4, "gamma": 1.9, "punch": 0.78, "fps": 25,
            "fade": 0.0,
            # screen mode
            "monitor": mons[0]["i"] if mons else 1,
            "screen_fps": 10, "screen_gamma": 0.75,
            "min_sat": 55.0, "sat_boost": 1.35, "screen_width": 192,
            "monitors": mons,
            "screen_ok": screen_permission() is not False,
            # telemetry
            "hue": 0, "sat": 0, "val": 0, "bpm": 0.0,
            "level": 0.0, "hit": 0.0, "bands": [0.0] * 10,
            "active_palette": "ember", "device": "-",
        }

    def snapshot(self):
        with self._lock:
            return dict(self._d)

    def set(self, **kw):
        with self._lock:
            self._d.update(kw)
            if "running" in kw:
                (self.started.set if kw["running"] else self.started.clear)()

    def control(self, **kw):
        """set(), for a change that came from the desk rather than telemetry."""
        self.set(**kw)
        self.changed.set()


# ---------------------------------------------------------------------------
def apply_palette(color, name):
    if name == "auto":
        color.locked = False
        return
    if color.pal != name or not color.locked:
        color.pal = name
        color.locked = True
        color.idx = 0
        color.retarget(*PALETTES[name][0])


class Engine(threading.Thread):
    daemon = True

    def __init__(self, state):
        super().__init__()
        self.state = state

    def run(self):
        while True:
            self.state.started.wait()           # parked while the desk is stopped
            try:
                self._session()
            except Exception as exc:
                self.state.set(running=False, status=f"error: {exc}")
                time.sleep(0.5)

    def _session(self):
        """Own the bulb for one run; the loops return on stop or mode change."""
        st = self.state
        cfg = st.snapshot()
        if not cfg["ip"]:
            st.set(running=False,
                   status="no bulb address - find one on the cloud deck")
            time.sleep(0.3)
            return
        st.set(status="connecting")

        bulb = open_bulb(cfg["ip"], on_attempt=lambda n, total, err: st.set(
            status=f"connecting {n}/{total}" + (f" - {err}" if err else "")))
        sender = Sender(bulb)
        sender.start()
        try:
            if cfg["mode"] == "screen":
                self._screen_loop(st, sender)
            elif cfg["mode"] == "fusion":
                self._fusion_loop(st, sender)
            else:
                self._audio_loop(st, sender)
        finally:
            sender.stop()
            release_bulb(bulb)
            st.set(status="stopped", level=0.0, hit=0.0, bands=[0.0] * 10)

    # -- screen mode --------------------------------------------------------
    def _screen_loop(self, st, sender):
        cfg = st.snapshot()
        if screen_permission() is False:
            raise RuntimeError("no Screen Recording permission - grant it in "
                               "System Settings > Privacy & Security")

        monitor = cfg["monitor"]
        grab = ScreenGrabber(monitor, None, cfg["screen_width"])
        color = ScreenColorEngine(min_sat=cfg["min_sat"],
                                  sat_boost=cfg["sat_boost"],
                                  gamma=cfg["screen_gamma"], calm=cfg["calm"])
        st.set(status="running", active_palette="screen", bpm=0.0,
               device=f"display {monitor} "
                      f"{grab.box['width']}x{grab.box['height']}")

        pacer = CapturePacer()
        last = time.monotonic()
        last_hsv = None
        cut_env = 0.0

        try:
            while True:
                cfg = st.snapshot()
                if not cfg["running"] or cfg["mode"] != "screen":
                    break
                if cfg["monitor"] != monitor:
                    break                       # restart the session on the new display

                color.calm = cfg["calm"]
                color.min_sat = float(cfg["min_sat"])
                color.sat_boost = float(cfg["sat_boost"])
                color.gamma = float(cfg["screen_gamma"])

                now = time.monotonic()
                dt, last = now - last, now
                frame = grab.frame()
                hue, sat, val, cut = color.tick(
                    dt, *color.measure(crop_letterbox(frame)))
                base = 1.0 / max(cfg["screen_fps"], 4)
                period = pacer.period(frame, max(cfg["screen_fps"], 4), now)

                v = int(round(float(np.clip(3 + 97 * val, 1, 100))))
                hsv = (int(round(hue)) % 360, int(round(sat)), v)

                # hue histogram, folded to 12 bars for the UI
                cut_env = max(cut_env * 0.72, 1.0 if cut else 0.0)
                hist = color.hist.reshape(12, 3).sum(axis=1)
                peak = float(hist.max())
                st.set(hue=hsv[0], sat=hsv[1], val=hsv[2],
                       level=round(float(val), 3), hit=round(cut_env, 3),
                       bands=[round(float(b) / peak, 3) for b in hist]
                             if peak > 0 else [0.0] * 12)

                settled = (last_hsv and not cut
                           and abs(hsv[2] - last_hsv[2]) < 2
                           and abs(hsv[1] - last_hsv[1]) < 3
                           and abs(((hsv[0] - last_hsv[0] + 180) % 360) - 180) < 3)
                if not settled:
                    # ramp for one frame gap so consecutive colours join up
                    sender.put(hsv, sudden=cut,
                               duration=int(np.clip(period * 1000, 30, 150)))
                    last_hsv = hsv

                slack = base - (time.monotonic() - now)
                if slack > 0:
                    time.sleep(slack)
                # still picture: wait out the longer gap, but a moved control
                # ends it so the desk never feels slow
                if period > base and st.changed.wait(period - base):
                    pacer.reset()
                st.changed.clear()
        finally:
            grab.close()

    # -- music mode ---------------------------------------------------------
    def _audio_loop(self, st, sender):
        cfg = st.snapshot()
        analyser = Analyser()
        color = ColorEngine(None, calm=cfg["calm"], fade=cfg["fade"])
        lvl = LevelEngine(calm=cfg["calm"], punch=cfg["punch"])
        recorder = pick_recorder(cfg["mic"])
        st.set(device=str(getattr(recorder, "name", recorder))[:48])
        dt = analyser.dt

        last_sent = 0.0
        last_hsv = None
        st.set(status="running")

        try:
            with recorder.recorder(samplerate=SAMPLERATE, channels=2) as rec:
                while True:
                    cfg = st.snapshot()
                    if not cfg["running"] or cfg["mode"] != "music":
                        break

                    analyser.sensitivity = cfg["sensitivity"]
                    color.calm = lvl.calm = cfg["calm"]
                    color.fade = float(cfg["fade"])
                    lvl.punch = cfg["punch"]
                    apply_palette(color, cfg["palette"])

                    f = analyser.feed(rec.record(numframes=BLOCK))
                    if lvl.feed(f, dt):
                        color.step(f["t"], f["beat"])

                    color.maybe_switch(f["t"], f["bands"])
                    hue, sat = color.tick(dt, f["treble"], f["t"])

                    now = time.monotonic()
                    if now - last_sent < 1.0 / max(cfg["fps"], 5):
                        continue

                    val = int(round(np.clip(
                        4 + 96 * (lvl.brightness ** cfg["gamma"]), 1, 100)))
                    hsv = (int(round(hue)) % 360, int(round(sat)), val)

                    st.set(hue=hsv[0], sat=hsv[1], val=hsv[2],
                           bpm=round(60.0 / max(f["beat"], 1e-3), 1),
                           level=round(lvl.brightness, 3),
                           hit=round(lvl.hit, 3),
                           bands=[round(float(b), 3) for b in f["bands"]],
                           active_palette=color.pal)

                    if last_hsv and abs(hsv[2] - last_hsv[2]) < 2 \
                            and abs(hsv[1] - last_hsv[1]) < 3 \
                            and abs(((hsv[0] - last_hsv[0] + 180) % 360) - 180) < 2:
                        continue

                    fading = color.fade_t is not None
                    sender.put(hsv, sudden=lvl.hit > 0.35 and not fading,
                               duration=int(np.clip(
                                   1000.0 / max(cfg["fps"], 5), 30, 150)))
                    last_sent, last_hsv = now, hsv
        finally:
            st.set(bands=[0.0] * 10)

    # -- fusion mode: colour from the display, brightness from the audio ----
    def _fusion_loop(self, st, sender):
        cfg = st.snapshot()
        if screen_permission() is False:
            raise RuntimeError("no Screen Recording permission - grant it in "
                               "System Settings > Privacy & Security")

        monitor = cfg["monitor"]
        analyser = Analyser()
        lvl = LevelEngine(calm=cfg["calm"], punch=cfg["punch"])
        color = ScreenColorEngine(min_sat=cfg["min_sat"],
                                  sat_boost=cfg["sat_boost"],
                                  gamma=cfg["screen_gamma"], calm=cfg["calm"])
        # the grab runs on its own thread: a ~26 ms capture inline would
        # starve rec.record() and drag the onsets behind the music
        screen = ScreenColorThread(ScreenGrabber(monitor, None,
                                                 cfg["screen_width"]),
                                   color, fps=cfg["screen_fps"])
        recorder = pick_recorder(cfg["mic"])
        dt = analyser.dt
        last_sent = 0.0
        last_hsv = None
        cut_env = 0.0

        st.set(status="running", active_palette="screen",
               device=f"display {monitor} + "
                      f"{str(getattr(recorder, 'name', recorder))[:32]}")
        screen.start()
        try:
            with recorder.recorder(samplerate=SAMPLERATE, channels=2) as rec:
                while True:
                    cfg = st.snapshot()
                    if not cfg["running"] or cfg["mode"] != "fusion":
                        break
                    if cfg["monitor"] != monitor:
                        break                   # restart on the new display
                    if screen.error:
                        raise RuntimeError(f"screen capture stopped: "
                                           f"{screen.error}")

                    analyser.sensitivity = cfg["sensitivity"]
                    lvl.calm = color.calm = cfg["calm"]
                    lvl.punch = cfg["punch"]
                    color.min_sat = float(cfg["min_sat"])
                    color.sat_boost = float(cfg["sat_boost"])
                    screen.fps = cfg["screen_fps"]
                    if st.changed.is_set():     # a control moved: full rate
                        st.changed.clear()
                        screen.poke()

                    f = analyser.feed(rec.record(numframes=BLOCK))
                    lvl.feed(f, dt)             # accents move brightness only

                    now = time.monotonic()
                    if now - last_sent < 1.0 / max(cfg["fps"], 5):
                        continue

                    hue, sat, _val, cut = screen.read()
                    val = int(round(np.clip(
                        4 + 96 * (lvl.brightness ** cfg["gamma"]), 1, 100)))
                    hsv = (int(round(hue)) % 360, int(round(sat)), val)

                    cut_env = max(cut_env * 0.72, 1.0 if cut else 0.0)
                    st.set(hue=hsv[0], sat=hsv[1], val=hsv[2],
                           bpm=round(60.0 / max(f["beat"], 1e-3), 1),
                           level=round(lvl.brightness, 3),
                           hit=round(max(lvl.hit, cut_env), 3),
                           bands=[round(float(b), 3) for b in f["bands"]])

                    settled = (last_hsv and not cut
                               and abs(hsv[2] - last_hsv[2]) < 2
                               and abs(hsv[1] - last_hsv[1]) < 3
                               and abs(((hsv[0] - last_hsv[0] + 180) % 360) - 180) < 3)
                    if settled:
                        continue

                    sender.put(hsv, sudden=cut or lvl.hit > 0.35,
                               duration=int(np.clip(
                                   1000.0 / max(cfg["fps"], 5), 30, 150)))
                    last_sent, last_hsv = now, hsv
        finally:
            screen.close()
            st.set(bands=[0.0] * 10)


# ---------------------------------------------------------------------------
def _data_url(image: bytes) -> str:
    """A data: URL the page can drop straight into <img>. The cloud serves
    captchas as JPEG and QR codes as PNG, so sniff rather than guess."""
    kind = "png" if image[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(image).decode()


class WebPrompter:
    """token_extractor.Prompter, answered by the browser instead of a tty."""

    # the CLI prints these in the clear; a page that stays open on a desk
    # should not, so the log masks them and the device table reveals one
    # row at a time on request
    SECRETS = {"TOKEN", "BLE KEY"}

    def __init__(self, session):
        self.session = session

    def note(self, text="", style="", tab=0):
        self.session.log(text, style, tab)

    def entry(self, key, value, tab=0):
        if not value:
            return
        shown = "*" * 16 if key in self.SECRETS else value
        self.session.log(f"{key + ':': <9}{shown}", "entry", tab)

    def ask(self, key, label, secret=False, hint=""):
        return self.session.request_answer(key, label, hint, secret)

    def show_image(self, kind, image, login_url=None):
        self.session.set_image(kind, _data_url(image), login_url)


class CloudSession:
    """One Xiaomi cloud login, driven from the browser.

    The login runs on its own thread. Whenever token_extractor needs an
    answer - password, captcha, 2FA code - the thread parks on an Event and
    the pending question turns up in the state the page polls; POST
    /api/cloud/answer hands the answer back and releases it. Nothing the
    user types is ever kept: the password lives in a local, the answer slot
    is cleared as soon as it is read.
    """

    PHASES = ("idle", "running", "prompt", "done", "error")
    MAX_LOG = 400

    def __init__(self):
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._cancel = threading.Event()
        self._answer = None
        self._thread = None
        with self._lock:
            self._clear()

    # -- state --------------------------------------------------------
    def _clear(self):
        self.phase = "idle"
        self.lines = []
        self.prompt = None
        self.image = None
        self.image_kind = None
        self.login_url = None
        self.devices = []
        self.error = ""
        self.lan = ""

    def snapshot(self):
        with self._lock:
            return {
                "phase": self.phase,
                "log": list(self.lines),
                "prompt": dict(self.prompt) if self.prompt else None,
                "image": self.image,
                "image_kind": self.image_kind,
                "login_url": self.login_url,
                "devices": list(self.devices),
                "error": self.error,
                "lan": self.lan,
            }

    def log(self, text="", style="", tab=0):
        with self._lock:
            self.lines.append({"t": text, "s": style, "i": tab})
            del self.lines[:-self.MAX_LOG]

    def set_image(self, kind, url, login_url=None):
        with self._lock:
            self.image, self.image_kind, self.login_url = url, kind, login_url

    # -- the question/answer handshake --------------------------------
    def request_answer(self, key, label, hint, secret):
        self._ready.clear()
        with self._lock:
            self._answer = None
            self.prompt = {"key": key, "label": label, "hint": hint, "secret": secret}
            self.phase = "prompt"
        # poll rather than wait forever, so a cancel from the page lands
        while not self._ready.wait(0.25):
            if self._cancel.is_set():
                raise cloud.LoginCancelled("cancelled")
        with self._lock:
            answer, self._answer = self._answer, None
            self.prompt = None
            self.phase = "running"
        if answer is None:
            raise cloud.LoginCancelled("cancelled")
        return answer

    def answer(self, value):
        with self._lock:
            if not self.prompt:
                return False
            self._answer = value
        self._ready.set()
        return True

    def cancel(self):
        if self._cancel.is_set():
            return
        self._cancel.set()
        self._ready.set()
        self.log("Aborting - a QR login only notices once its long poll "
                 "returns, up to 10 s.", "warn")

    # -- the run ------------------------------------------------------
    def start(self, method, username="", password="", server=""):
        with self._lock:
            if self.phase in ("running", "prompt"):
                return False, "a login is already running"
        self._cancel.clear()
        self._ready.clear()
        with self._lock:
            self._clear()
            self.phase = "running"
        self._thread = threading.Thread(
            target=self._run, args=(method, username, password, server), daemon=True)
        self._thread.start()
        return True, ""

    def _run(self, method, username, password, server):
        prompter = WebPrompter(self)
        try:
            if method == "qr":
                connector = cloud.QrCodeXiaomiCloudConnector(prompter)
                self.log("Waiting for the QR code to be scanned in the Mi Home app.", "info")
            else:
                connector = cloud.PasswordXiaomiCloudConnector(prompter, username, password)
            # one flag for both halves: the prompt handshake below and the
            # QR long poll inside the connector watch the same Event
            connector.cancel = self._cancel

            if not connector.login():
                self._fail("login refused - check the credentials and try again")
                return

            self.log("Logged in.", "ok")
            self.log()
            servers = [server] if server else list(cloud.SERVERS)
            self.log(f"Scanning {', '.join(servers)}...", "info")
            output = cloud.collect_devices(connector, servers, prompter)
            devices = cloud.flatten_devices(output)
            with self._lock:
                self.devices = devices
                self.phase = "done"
                self.prompt = None
                self.image = None
            self.log(f"{len(devices)} device(s) found.", "ok")
        except cloud.LoginCancelled:
            self.log("Cancelled.", "warn")
            with self._lock:
                self.phase = "idle"
                self.prompt = None
                self.image = None
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")

    def _fail(self, message):
        self.log(message, "error")
        with self._lock:
            self.phase = "error"
            self.error = message
            self.prompt = None
            self.image = None

    # -- one-shot helpers ---------------------------------------------
    def enable_lan(self, ip, token):
        """Flip developer mode on with the token just extracted. This is the
        step that opens port 55443, which is the only reason the desk needs
        the cloud at all."""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                enable_lan(ip, token)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.lan = message
            self.log(message, "error")
            return False, message
        for line in buf.getvalue().splitlines():
            if line.strip():
                self.log(line)
        message = f"developer mode on for {ip}; port 55443 should be open"
        with self._lock:
            self.lan = message
        self.log(message, "ok")
        return True, message


# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>bulbd</title>
<style>
/* Split: the live colour fills the left, controls sit in one column on the
   right; under 1000 px the colour folds into a band under the top bar.
   The only colour on the page is the light's own - chrome is neutral, the
   selected state is plain ink. Nothing animates on its own: every repaint
   comes from a poll that carried new state. */
:root{
  --bg:#f4f5f6; --sf:#ffffff; --sf2:#eceef0; --ink:#17181a; --dim:#686c73; --line:#dadde1;
  --ok:#1f8a4c; --warn:#a86a00; --err:#c9362c;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#141517; --sf:#1c1d20; --sf2:#242529; --ink:#e9eaec; --dim:#8d9199; --line:#2e3035;
  --ok:#5fc58a; --warn:#e0a948; --err:#ef6b62;
  color-scheme:dark;
}}
*{box-sizing:border-box}
[hidden]{display:none !important}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);
  font:14px/1.4 system-ui,-apple-system,"SF Pro Text","Segoe UI",sans-serif;
  font-variant-numeric:tabular-nums}
button,input,select{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
a{color:inherit}

/* ---------- frame ---------- */
.app{display:grid;grid-template-columns:1fr 520px;grid-template-rows:auto 1fr;
  grid-template-areas:"lamp top" "lamp side";min-height:100vh}
.lamp{grid-area:lamp;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;
  border-right:1px solid var(--line)}
.top{grid-area:top;display:flex;align-items:center;gap:18px;padding:14px 22px;min-height:56px}
.side{grid-area:side;padding:10px 22px 28px;display:grid;gap:20px;align-content:start}
body[data-deck="cloud"] .app{grid-template-columns:1fr;grid-template-areas:"top" "side"}
body[data-deck="cloud"] .lamp{display:none}
@media (max-width:999px){
  .app{grid-template-columns:1fr;grid-template-rows:auto;grid-template-areas:"top" "lamp" "side";min-height:0}
  .lamp{position:static;height:auto;border-right:0}
  .lamp .field{flex:0 0 190px}
  .side{padding-top:18px}
}

/* ---------- top bar ---------- */
.name{font-weight:680;font-size:15px;letter-spacing:-.01em}
.st{display:flex;align-items:center;gap:7px;color:var(--dim);font-size:13.5px;min-width:0}
.st i{flex:none;width:8px;height:8px;border-radius:50%;background:var(--dim)}
.st.on{color:var(--ink)} .st.on i{background:var(--ok)}
.st.busy i{background:var(--warn)}
.st.bad{color:var(--err)} .st.bad i{background:var(--err)}
#stxt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sp{flex:1}
.addr{flex:none;font-size:13.5px;color:var(--dim);background:transparent;border:1px solid var(--line);
  border-radius:7px;padding:5px 10px;cursor:pointer}
.addr b{color:var(--ink);font-weight:500}
.addr:hover{border-color:var(--dim)}
.addr.need{color:var(--bg);background:var(--ink);border-color:var(--ink);font-weight:600}
body[data-deck="cloud"] #addr, body[data-deck="cloud"] #status,
body[data-deck="runtime"] #done{display:none}

/* ---------- live colour ---------- */
.field{position:relative;flex:1;min-height:0;background:var(--sf2)}
.ro{position:absolute;left:20px;bottom:14px;right:20px;display:flex;flex-wrap:wrap;gap:4px 18px;
  font-size:13.5px;font-weight:500;opacity:.82}
.ro .pal{text-transform:capitalize}
#tape{display:block;width:100%;height:16px}
.spec{display:flex;align-items:flex-end;gap:3px;height:50px;padding:6px 20px 10px;border-bottom:1px solid var(--line)}
.spec i{flex:1;height:3px;border-radius:2px 2px 0 0;background:var(--line)}

/* ---------- controls ---------- */
.transport{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.seg{display:inline-flex;background:var(--sf2);border-radius:9px;padding:3px;gap:2px}
.seg button{border:0;background:transparent;padding:6px 15px;border-radius:7px;color:var(--dim);cursor:pointer}
.seg button:hover{color:var(--ink)}
.seg button[aria-pressed="true"]{background:var(--sf);color:var(--ink);font-weight:600;box-shadow:0 0 0 1px var(--line)}
.go{border:0;padding:8px 20px;border-radius:9px;background:var(--ink);color:var(--bg);font-weight:650;cursor:pointer}
.go[data-on="true"]{background:transparent;color:var(--ink);box-shadow:inset 0 0 0 1.5px var(--ink)}
.chk{display:flex;align-items:center;gap:8px;cursor:pointer}
.chk input{width:16px;height:16px;margin:0;accent-color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{display:flex;align-items:center;gap:8px;padding:6px 11px 6px 7px;border-radius:8px;border:0;cursor:pointer;
  background:var(--sf);box-shadow:inset 0 0 0 1px var(--line);font-size:13.5px;text-transform:capitalize}
.chip s{display:block;width:30px;height:12px;border-radius:3px}
.chip:hover{box-shadow:inset 0 0 0 1px var(--dim)}
.chip[aria-pressed="true"]{box-shadow:inset 0 0 0 1.5px var(--ink);font-weight:600}
.rows{display:grid;gap:4px}
.r{display:grid;grid-template-columns:150px 1fr 54px;gap:16px;align-items:center;min-height:34px}
.r output{text-align:right;color:var(--dim);font-size:13.5px}
@media (max-width:520px){.r{grid-template-columns:110px 1fr 46px;gap:10px}}
input[type=range]{--p:50%;-webkit-appearance:none;appearance:none;width:100%;height:20px;margin:0;background:transparent;cursor:pointer}
input[type=range]::-webkit-slider-runnable-track{height:4px;border-radius:2px;
  background:linear-gradient(90deg,var(--dim) var(--p),var(--line) var(--p))}
input[type=range]::-moz-range-track{height:4px;border-radius:2px;background:var(--line)}
input[type=range]::-moz-range-progress{height:4px;border-radius:2px;background:var(--dim)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;
  background:var(--ink);margin-top:-6px;border:0}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:var(--ink);border:0}
.in{display:grid;gap:6px;font-size:13.5px}
.in>span,.in>label{color:var(--dim)}
select,input.tx{border:1px solid var(--line);background:var(--sf);border-radius:8px;padding:7px 11px;
  min-height:36px;width:100%;font-size:14px}
input.tx::placeholder{color:var(--dim)}
.note{font-size:13.5px;margin:0}
.note:empty{display:none}
.warn{color:var(--warn)} .err{color:var(--err)} .ok{color:var(--ok)}
.line{display:flex;gap:8px}
.line>input{flex:1;min-width:0}
.btn{border:0;padding:8px 16px;border-radius:8px;background:var(--sf2);font-weight:550;white-space:nowrap;cursor:pointer}
.btn.pri{background:var(--ink);color:var(--bg)}
.btn:disabled{opacity:.4;cursor:default}
.mini{border:0;padding:4px 10px;border-radius:7px;background:transparent;box-shadow:inset 0 0 0 1px var(--line);
  font-size:13px;white-space:nowrap;cursor:pointer}
.mini:hover{box-shadow:inset 0 0 0 1px var(--dim)}
.mini:disabled{opacity:.5}

/* one desk, three modes - show only the controls that apply. Fusion draws
   from both sides, so it hides .no-fusion instead: the palette, the colour
   fade and the dark lift have nothing to act on when the display owns the
   hue and the audio owns the brightness. */
body[data-mode="music"] .only-screen,
body[data-mode="screen"] .only-music,
body[data-mode="fusion"] .no-fusion{display:none !important}
body[data-deck="runtime"] .deck-cloud,
body[data-deck="cloud"] .deck-runtime{display:none !important}

/* ---------- setup ---------- */
.setup{display:grid;grid-template-columns:380px 1fr;gap:24px;align-items:start}
.login{display:grid;gap:12px}
.log{background:var(--sf);border-radius:10px;padding:12px 14px;height:420px;overflow:auto;
  font:12.5px/1.65 ui-monospace,Menlo,monospace;color:var(--dim);white-space:pre-wrap;word-break:break-word}
.log .info{color:var(--ink)} .log .ok{color:var(--ok)} .log .warn{color:var(--warn)} .log .error{color:var(--err)}
@media (max-width:999px){.setup{grid-template-columns:1fr}.log{height:180px}}
.qr{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap}
.qr img{max-width:190px;background:#fff;padding:8px;border-radius:6px;image-rendering:pixelated}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;font-weight:500;color:var(--dim);padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:10px;border-bottom:1px solid var(--line);vertical-align:middle;white-space:nowrap}
td code{font:12.5px ui-monospace,Menlo,monospace;color:var(--dim);margin-right:6px}
td.acts{text-align:right}
td .mini+.mini{margin-left:6px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-left:8px;vertical-align:1px}
.empty{color:var(--dim);padding:12px 0;margin:0}
</style></head><body data-mode="music" data-deck="runtime">

<div class="app">
  <div class="top">
    <span class="name">bulbd</span>
    <span class="st" id="status"><i></i><span id="stxt">idle</span></span>
    <span class="sp"></span>
    <button class="addr" id="addr" type="button">Set up bulb</button>
    <button class="btn" id="done" type="button">Done</button>
  </div>

  <div class="lamp deck-runtime">
    <div class="field" id="field">
      <div class="ro" id="ro" hidden>
        <span id="rh"></span><span id="rs"></span><span id="rv"></span>
        <span class="only-music" id="rb"></span>
        <span class="only-music no-fusion pal" id="rp"></span>
        <span class="only-music" id="rd"></span>
      </div>
    </div>
    <canvas id="tape" hidden></canvas>
    <div class="spec" id="spec"></div>
  </div>

  <div class="side">
    <!-- ============================ RUNTIME ============================ -->
    <p class="note err deck-runtime" id="errnote"></p>
    <div class="transport deck-runtime">
      <span class="seg" id="modes">
        <button type="button" data-mode="music" aria-pressed="true">Music</button>
        <button type="button" data-mode="screen" aria-pressed="false">Screen</button>
        <button type="button" data-mode="fusion" aria-pressed="false">Fusion</button>
      </span>
      <span class="sp"></span>
      <label class="chk"><input type="checkbox" id="calm">Calm</label>
      <button class="go" id="go" type="button" data-on="false">Start</button>
    </div>

    <div class="chips deck-runtime only-music no-fusion" id="chips"></div>

    <div class="in deck-runtime only-screen">
      <label for="monitor">Display</label>
      <select id="monitor"></select>
      <p class="note warn" id="permnote"></p>
    </div>

    <div class="rows deck-runtime">
      <div class="r only-music"><label for="sens">Beat sensitivity</label>
        <input type="range" id="sens" min="0.6" max="2.6" step="0.05" value="1.4"><output id="v-sens"></output></div>
      <div class="r only-music"><label for="gamma">Contrast</label>
        <input type="range" id="gamma" min="1" max="3" step="0.05" value="1.9"><output id="v-gamma"></output></div>
      <div class="r only-music"><label for="punch">Punch</label>
        <input type="range" id="punch" min="0.2" max="1" step="0.02" value="0.78"><output id="v-punch"></output></div>
      <div class="r only-music"><label for="fps">Refresh rate</label>
        <input type="range" id="fps" min="8" max="30" step="1" value="25"><output id="v-fps"></output></div>
      <div class="r only-music no-fusion"><label for="fade">Colour fade</label>
        <input type="range" id="fade" min="0" max="6" step="0.1" value="0"><output id="v-fade"></output></div>
      <div class="r only-screen no-fusion"><label for="sgamma">Dark lift</label>
        <input type="range" id="sgamma" min="0.4" max="1.6" step="0.05" value="0.75"><output id="v-sgamma"></output></div>
      <div class="r only-screen"><label for="minsat">Colour floor</label>
        <input type="range" id="minsat" min="0" max="90" step="1" value="55"><output id="v-minsat"></output></div>
      <div class="r only-screen"><label for="sboost">Saturation boost</label>
        <input type="range" id="sboost" min="0.8" max="2" step="0.05" value="1.35"><output id="v-sboost"></output></div>
      <div class="r only-screen"><label for="sfps">Capture rate</label>
        <input type="range" id="sfps" min="4" max="30" step="1" value="10"><output id="v-sfps"></output></div>
    </div>

    <!-- ============================== SETUP ============================== -->
    <div class="setup deck-cloud">
      <div class="login">
        <div class="in">
          <label for="ip">Bulb address</label>
          <div class="line">
            <input class="tx" id="ip" autocomplete="off" spellcheck="false">
            <button class="btn" id="ipset" type="button">Set</button>
          </div>
        </div>
        <span class="seg" id="cmethods" style="justify-self:start;margin-top:6px">
          <button type="button" data-method="password" aria-pressed="true">Password</button>
          <button type="button" data-method="qr" aria-pressed="false">QR code</button>
        </span>
        <div class="login" id="cpw">
          <div class="in"><label for="cuser">Account</label>
            <input class="tx" id="cuser" autocomplete="username" spellcheck="false" placeholder="Email, phone or user ID"></div>
          <div class="in"><label for="cpass">Password</label>
            <input class="tx" id="cpass" type="password" autocomplete="current-password"></div>
        </div>
        <div class="in"><label for="cserver">Region</label><select id="cserver"></select></div>
        <p class="note warn" id="cloudnote"></p>
        <div class="line">
          <button class="btn pri" id="cstart" type="button">Log in</button>
          <button class="btn" id="ccancel" type="button" disabled>Abort</button>
        </div>
        <p class="note err" id="cerr"></p>
        <div class="qr" id="cimgwrap" hidden>
          <img id="cimg" alt="Verification image from the Xiaomi cloud">
          <p class="note" id="cqr"></p>
        </div>
        <div class="in" id="cprompt" hidden>
          <label id="cplabel" for="cpinput"></label>
          <div class="line">
            <input class="tx" id="cpinput" autocomplete="off" spellcheck="false">
            <button class="btn pri" id="cpsend" type="button">Send</button>
          </div>
        </div>
      </div>
      <div class="log" id="clog"><div class="info">ready</div></div>
    </div>

    <div class="deck-cloud">
      <p class="empty" id="cdevnote">Log in to list the devices on your Xiaomi account.</p>
      <div class="scroll"><table id="cdevtable" hidden>
        <thead><tr><th>Name</th><th>Model</th><th>Address</th><th>Token</th><th></th></tr></thead>
        <tbody id="cdevs"></tbody>
      </table></div>
      <p class="note" id="clan" style="padding-top:10px"></p>
    </div>
  </div>
</div>

<script>
const P = __PALETTES__;
const $ = id => document.getElementById(id);
let holdUntil = 0;   // don't let polling stomp a fader you're dragging

function hsv2rgb(h,s,v){
  h=((h%360)+360)%360; s/=100; v/=100;
  const c=v*s, x=c*(1-Math.abs((h/60)%2-1)), m=v-c;
  let r,g,b;
  if(h<60)[r,g,b]=[c,x,0]; else if(h<120)[r,g,b]=[x,c,0];
  else if(h<180)[r,g,b]=[0,c,x]; else if(h<240)[r,g,b]=[0,x,c];
  else if(h<300)[r,g,b]=[x,0,c]; else [r,g,b]=[c,0,x];
  return [(r+m)*255,(g+m)*255,(b+m)*255].map(Math.round);
}

// palette chips, each previewing its own anchors
const chips = $('chips');
[['auto',null]].concat(Object.keys(P).map(k=>[k,P[k]])).forEach(([name,anchors])=>{
  const b=document.createElement('button');
  b.className='chip'; b.type='button'; b.dataset.name=name;
  b.setAttribute('aria-pressed', name==='auto');
  const s=document.createElement('s');
  s.style.background = anchors
    ? 'linear-gradient(90deg,'+anchors.map(a=>`rgb(${hsv2rgb(a[0],a[1],92)})`).join(',')+')'
    : 'repeating-linear-gradient(90deg,var(--dim) 0 2px,transparent 2px 5px)';
  b.append(s, document.createTextNode(name));
  b.onclick=()=>{
    [...chips.children].forEach(c=>c.setAttribute('aria-pressed', c===b));
    send({palette:name});
  };
  chips.append(b);
});

// spectrum bars - 10 audio bands in music mode, 12 hue bins in screen mode
const spec=$('spec');
let bars=[];
function setBars(n){
  if(bars.length===n) return;
  spec.innerHTML='';
  for(let i=0;i<n;i++) spec.append(document.createElement('i'));
  bars=[...spec.children];
}
setBars(10);

// the last 30 s of colour, drawn only when a poll brings a new state
const tape=$('tape'), tctx=tape.getContext('2d'), TAPE_MS=30000;
let tbuf=[];
function drawTape(col){
  const now=performance.now();
  tbuf.push([now,col]);
  while(tbuf.length>1 && tbuf[1][0] < now-TAPE_MS) tbuf.shift();
  const dpr=devicePixelRatio||1, W=Math.round(tape.clientWidth*dpr), H=Math.round(tape.clientHeight*dpr);
  if(tape.width!==W || tape.height!==H){ tape.width=W; tape.height=H; }
  tctx.clearRect(0,0,W,H);
  const x=t=>(t-(now-TAPE_MS))/TAPE_MS*W;
  tbuf.forEach(([t,c],i)=>{
    const x0=Math.max(0,x(t)), x1=i+1<tbuf.length ? x(tbuf[i+1][0]) : W;
    tctx.fillStyle=c; tctx.fillRect(Math.floor(x0),0,Math.ceil(x1-x0)+1,H);
  });
}

async function send(patch){
  holdUntil = Date.now()+700;
  try{ await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},
       body:JSON.stringify(patch)}); }catch(e){ setStatus('server unreachable'); }
  poll();   // an idle desk polls slowly, so pick the change up now
}
async function post(path, body){
  const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
                             body:JSON.stringify(body||{})});
  return r.json().catch(()=>({ok:false,error:'bad response'}));
}

const KEYS = {sens:'sensitivity', gamma:'gamma', punch:'punch', fps:'fps',
              fade:'fade',
              sgamma:'screen_gamma', minsat:'min_sat', sboost:'sat_boost',
              sfps:'screen_fps'};
function fmt(id, v){
  v=+v;
  if(id==='fps' || id==='sfps') return v+'/s';
  if(id==='fade') return v ? v.toFixed(1)+' s' : 'Off';
  if(id==='minsat') return String(v);
  return v.toFixed(2);
}
function paint(id){
  const el=$(id);
  el.style.setProperty('--p', (el.value-el.min)/(el.max-el.min)*100+'%');
  $('v-'+id).textContent = fmt(id, el.value);
}
Object.keys(KEYS).forEach(id=>{
  const el=$(id);
  paint(id);
  el.addEventListener('input',()=>{ paint(id); send({[KEYS[id]]: +el.value}); });
});

function setStatus(text){
  const st=$('status');
  // a failure arrives with a paragraph of advice: the bar says "error" and
  // the advice goes in a note, where it has room to wrap
  const m = /^error:\s*([\s\S]*)$/.exec(text);
  $('stxt').textContent = m ? 'error' : text.replace('on the cloud deck','in Setup');
  $('errnote').textContent = m ? m[1].charAt(0).toUpperCase() + m[1].slice(1) : '';
  st.className = 'st' + (text==='running' ? ' on'
    : /^(connecting|starting)/.test(text) ? ' busy'
    : /^(error|no bulb|server)/.test(text) ? ' bad' : '');
}

[...$('modes').children].forEach(b=>{
  b.onclick=()=>{
    const m=b.dataset.mode;
    [...$('modes').children].forEach(c=>c.setAttribute('aria-pressed', c===b));
    document.body.dataset.mode = m;
    send({mode:m});
  };
});
function showDeck(name){
  const deck = name==='cloud' ? 'cloud' : 'runtime';
  document.body.dataset.deck = deck;
  if(deck==='cloud' && window.cloudUp) cloudPoll();   // not before its state exists
}
$('addr').onclick=()=>{ location.hash='cloud'; };
$('done').onclick=()=>{ location.hash='runtime'; };
// #cloud in the URL opens straight onto Setup, so a link can point at it
showDeck(location.hash.slice(1));
addEventListener('hashchange', ()=>showDeck(location.hash.slice(1)));

$('monitor').addEventListener('change',e=>send({monitor:+e.target.value}));
$('calm').addEventListener('change',e=>send({calm:e.target.checked}));
$('go').addEventListener('click',()=>{
  const on = $('go').dataset.on === 'true';
  send({running: !on});
});
$('ipset').addEventListener('click',()=>send({ip:$('ip').value.trim()}));
$('ip').addEventListener('keydown',e=>{ if(e.key==='Enter') send({ip:$('ip').value.trim()}); });

let monitorsBuilt=false;
let ipTouched=false;
$('ip').addEventListener('input',()=>{ipTouched=true;});
// Poll only as fast as there is something new to draw: the engine's own
// rate while a session runs, once a second on an idle desk, slower when the
// page is not on screen.
let busy=false, pollTimer=0, live=null, wasRunning=false;
function pollDelay(){
  if(document.hidden) return 2000;
  if(!live || !live.running) return 1000;
  return live.mode==='screen' ? Math.max(60, 1000/live.screen_fps) : 60;
}
async function poll(){
  if(busy) return; busy=true;
  clearTimeout(pollTimer);
  try{
    const s = await (await fetch('/api/state')).json();
    live = s;
    const [r,g,b] = hsv2rgb(s.hue, s.sat, s.val);
    const col = `rgb(${r},${g},${b})`;
    $('field').style.background = s.running ? col : '';
    $('ro').hidden = $('tape').hidden = !s.running;
    if(s.running){
      if(!wasRunning) tbuf=[];
      $('ro').style.color = (0.299*r+0.587*g+0.114*b) > 150 ? '#111' : '#fff';
      $('rh').textContent=s.hue+'°'; $('rs').textContent=s.sat+'%'; $('rv').textContent=s.val+'%';
      $('rb').textContent=s.bpm.toFixed(0)+' bpm'; $('rp').textContent=s.active_palette;
      $('rd').textContent=s.device;
      drawTape(col);
    }
    wasRunning = s.running;
    setStatus(s.status);
    const addr=$('addr');
    addr.classList.toggle('need', !s.ip);
    if(s.ip){ addr.innerHTML='Bulb <b></b>'; addr.lastChild.textContent=s.ip; }
    else addr.textContent='Set up bulb';
    $('go').dataset.on = s.running; $('go').textContent = s.running ? 'Stop' : 'Start';
    if(!ipTouched && document.activeElement !== $('ip')) $('ip').value = s.ip || '';

    const screenMode = s.mode === 'screen';
    setBars(s.bands.length);
    s.bands.forEach((v,i)=>{
      const bar=bars[i];
      bar.style.height = (3 + 34*v) + 'px';
      // in screen mode each bar owns a hue, so paint it with that hue
      bar.style.background = screenMode
        ? `hsl(${Math.round((i+0.5)*360/s.bands.length)},62%,${30+34*v}%)`
        : (s.running ? `rgba(${r},${g},${b},${0.35+0.65*v})` : '');
    });

    if(!monitorsBuilt && s.monitors.length){
      monitorsBuilt = true;
      $('monitor').innerHTML = s.monitors
        .map(m=>`<option value="${m.i}">Display ${m.i}, ${m.w} × ${m.h}</option>`).join('');
      $('monitor').value = s.monitor;
    }
    $('permnote').textContent = !s.monitors.length ? 'mss is not installed. Run pip install mss.'
      : !s.screen_ok ? 'No Screen Recording permission. Grant it in System Settings › Privacy & Security › Screen Recording.'
      : '';

    if(Date.now() > holdUntil){
      $('calm').checked = s.calm;
      [...chips.children].forEach(c=>c.setAttribute('aria-pressed', c.dataset.name===s.palette));
      [...$('modes').children].forEach(c=>c.setAttribute('aria-pressed', c.dataset.mode===s.mode));
      document.body.dataset.mode = s.mode;
      $('monitor').value = s.monitor;
      // faders follow the desk too, or a reload leaves them showing whatever
      // the browser restored rather than what is running
      Object.keys(KEYS).forEach(id=>{
        const el=$(id), v=s[KEYS[id]];
        if(document.activeElement===el || +el.value===v) return;
        el.value = v;
        paint(id);
      });
    }
  }catch(e){ live=null; setStatus('server unreachable'); }
  busy=false;
  pollTimer = setTimeout(poll, pollDelay());
}
poll();
document.addEventListener('visibilitychange', ()=>{ if(!document.hidden){ poll(); cloudPoll(); } });

/* ------------------------------ setup deck ------------------------------ */
let method='password', serversBuilt=false, logLen=-1, devKey='';
[...$('cmethods').children].forEach(b=>{
  b.onclick=()=>{
    method=b.dataset.method;
    [...$('cmethods').children].forEach(c=>c.setAttribute('aria-pressed', c===b));
    $('cpw').hidden = method==='qr';
  };
});
$('cstart').addEventListener('click', async ()=>{
  $('cstart').disabled = true;
  $('cerr').textContent = '';
  const r = await post('/api/cloud/start', {
    method, server:$('cserver').value,
    username:$('cuser').value.trim(), password:$('cpass').value});
  $('cpass').value='';
  if(!r.ok) $('cerr').textContent = r.error || 'could not start';
  cloudPoll();
});
$('ccancel').addEventListener('click', async ()=>{ await post('/api/cloud/cancel'); cloudPoll(); });
const answer = async ()=>{
  const v = $('cpinput').value;
  $('cpinput').value='';
  await post('/api/cloud/answer', {value:v});
  cloudPoll();
};
$('cpsend').addEventListener('click', answer);
$('cpinput').addEventListener('keydown', e=>{ if(e.key==='Enter') answer(); });

function mini(text){
  const b=document.createElement('button');
  b.className='mini'; b.type='button'; b.textContent=text;
  return b;
}
function renderDevices(devs){
  const key = devs.map(d=>d.did).join(',');
  if(key===devKey) return;
  devKey = key;
  $('cdevtable').hidden = devs.length===0;
  $('cdevnote').hidden = devs.length>0;
  $('cdevs').innerHTML='';
  const MASK='•'.repeat(16);
  devs.forEach(d=>{
    const tr=document.createElement('tr');
    const name=document.createElement('td');
    name.textContent = d.name || '(unnamed)';
    if(d.online){ const t=document.createElement('span'); t.className='dot';
                  t.title='online'; t.setAttribute('aria-label','online'); name.append(t); }
    const model=document.createElement('td'); model.textContent=d.model||'-';
    const ip=document.createElement('td'); ip.textContent=d.ip||'-';
    const tok=document.createElement('td');
    const code=document.createElement('code'); code.textContent=MASK;
    const reveal=mini('Show'), copy=mini('Copy');
    let shown=false;
    reveal.onclick=()=>{ shown=!shown;
      code.textContent = shown ? (d.token||'-') : MASK;
      reveal.textContent = shown ? 'Hide' : 'Show'; };
    copy.onclick=async()=>{ try{ await navigator.clipboard.writeText(d.token||'');
      copy.textContent='Copied'; setTimeout(()=>copy.textContent='Copy',1200);}catch(e){ shown=true;
      code.textContent=d.token||'-'; reveal.textContent='Hide'; } };
    tok.append(code, reveal, copy);
    const act=document.createElement('td'); act.className='acts';
    const use=mini('Use');
    use.onclick=()=>{ $('ip').value = d.ip||''; ipTouched=false; send({ip:d.ip||''}); };
    const lan=mini('Enable LAN');
    lan.onclick=async()=>{ lan.disabled=true; lan.textContent='Enabling...';
      const r = await post('/api/cloud/lan', {ip:d.ip, token:d.token});
      lan.disabled=false; lan.textContent='Enable LAN';
      $('clan').className = 'note ' + (r.ok?'ok':'err');
      $('clan').textContent = r.message||r.error||''; };
    act.append(use, lan);
    tr.append(name, model, ip, tok, act);
    $('cdevs').append(tr);
  });
}

// Setup only changes during a login, so poll it only while it is on
// screen, and quickly only while a login is in flight.
let cbusy=false, cloudTimer=0, cloudActive=false;
function cloudDelay(){
  if(document.hidden || document.body.dataset.deck!=='cloud') return 0;
  return cloudActive ? 500 : 3000;
}
async function cloudPoll(){
  if(cbusy) return; cbusy=true;
  clearTimeout(cloudTimer);
  try{
    const c = await (await fetch('/api/cloud/state')).json();
    cloudActive = (c.phase==='running' || c.phase==='prompt');
    if(!serversBuilt && c.servers.length){
      serversBuilt = true;
      $('cserver').innerHTML = '<option value="">All regions</option>'
        + c.servers.map(s=>`<option value="${s}">${s}</option>`).join('');
    }
    $('cloudnote').textContent = c.available ? '' : c.reason;
    $('cstart').disabled = !c.available || cloudActive;
    $('ccancel').disabled = !cloudActive;

    if(c.log.length !== logLen){
      logLen = c.log.length;
      const log=$('clog');
      log.innerHTML='';
      if(!c.log.length){ log.innerHTML='<div class="info">ready</div>'; }
      c.log.forEach(l=>{
        const d=document.createElement('div');
        d.className=l.s||''; d.textContent=(' '.repeat(l.i||0)+l.t) || ' ';
        log.append(d);
      });
      log.scrollTop = log.scrollHeight;
    }

    const p = c.prompt;
    $('cprompt').hidden = !p;
    if(p){
      $('cplabel').textContent = p.label + (p.hint ? ' ('+p.hint+')' : '');
      $('cpinput').type = p.secret ? 'password' : 'text';
      if(document.activeElement !== $('cpinput')) $('cpinput').focus();
    }
    $('cimgwrap').hidden = !c.image;
    if(c.image && $('cimg').src !== c.image){
      $('cimg').src = c.image;
      $('cqr').innerHTML = '';
      if(c.login_url){
        const a=document.createElement('a');
        a.href=c.login_url; a.target='_blank'; a.rel='noopener'; a.textContent='open the login link';
        $('cqr').append('Scan in Mi Home, or ', a, '.');
      }
    }
    renderDevices(c.devices);
  }catch(e){ /* the runtime poll already reports an unreachable server */ }
  cbusy=false;
  const wait = cloudDelay();
  if(wait) cloudTimer = setTimeout(cloudPoll, wait);
}
var cloudUp = true;
cloudPoll();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    state = None
    cloud = None
    # Cloud login means a Xiaomi password crosses this socket. The desk has
    # no authentication, so on anything but loopback the endpoints stay shut
    # unless the operator asked for them with --allow-remote-login.
    cloud_enabled = True
    cloud_reason = ""
    # keep-alive: the page talks to the desk many times a second, and
    # HTTP/1.0 would cost a new connection and a new thread for each one.
    # Every response carries Content-Length, which 1.1 needs. The timeout
    # reaps a connection the browser has stopped using.
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        self._send(code, json.dumps(payload), "application/json")

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return None

    def _cloud_ready(self):
        if cloud is None:
            return f"token extractor unavailable - {CLOUD_ERROR}"
        if not self.cloud_enabled:
            return self.cloud_reason
        return ""

    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self._send(200, json.dumps(self.state.snapshot()), "application/json")
        elif self.path.startswith("/api/cloud/state"):
            snap = self.cloud.snapshot()
            snap["available"] = not self._cloud_ready()
            snap["reason"] = self._cloud_ready()
            snap["servers"] = list(cloud.SERVERS) if cloud else []
            self._json(200, snap)
        elif self.path in ("/", "/index.html"):
            page = PAGE.replace("__PALETTES__", json.dumps(PALETTES))
            self._send(200, page, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path.startswith("/api/cloud/"):
            self._do_cloud()
            return
        if not self.path.startswith("/api/control"):
            self.close_connection = True        # its body was never read
            self._send(404, "not found", "text/plain")
            return
        patch = self._body()
        if patch is None:
            self._send(400, "bad json", "text/plain")
            return
        allowed = {"running", "palette", "calm", "sensitivity", "gamma",
                   "punch", "fps", "fade", "mode", "monitor", "screen_fps",
                   "screen_gamma", "min_sat", "sat_boost", "ip"}
        clean = {k: v for k, v in patch.items() if k in allowed}
        if "palette" in clean and clean["palette"] != "auto" and clean["palette"] not in PALETTES:
            clean.pop("palette")
        if clean.get("mode") not in (None, "music", "screen", "fusion"):
            clean.pop("mode")
        if "monitor" in clean:
            known = {m["i"] for m in self.state.snapshot()["monitors"]}
            if clean["monitor"] not in known:
                clean.pop("monitor")
        # the LEDs cannot resolve past ~30/s, so clamp rather than trust the UI
        for key, lo, hi in (("fps", 5, 30), ("screen_fps", 4, 30)):
            if key in clean:
                clean[key] = int(min(max(clean[key], lo), hi))
        if "fade" in clean:
            clean["fade"] = float(min(max(clean["fade"], 0.0), 6.0))
        if "ip" in clean:
            ip = str(clean["ip"]).strip()
            if not HOSTNAME_RE.match(ip):
                clean.pop("ip")
            else:
                clean["ip"] = ip
                # the session reads the address once, at connect time, so a
                # live run has to end before the new bulb can be opened
                if self.state.snapshot()["running"] and ip != self.state.snapshot()["ip"]:
                    clean["running"] = False
                    clean["status"] = f"bulb address is now {ip} - press Start"
        if "running" in clean:
            clean["status"] = "starting" if clean["running"] else "stopping"
        self.state.control(**clean)
        self._send(200, json.dumps({"ok": True}), "application/json")

    # -- cloud login --------------------------------------------------
    def _do_cloud(self):
        reason = self._cloud_ready()
        if reason:
            self._json(403, {"ok": False, "error": reason})
            return
        body = self._body()
        if body is None:
            self._send(400, "bad json", "text/plain")
            return
        tail = self.path[len("/api/cloud/"):].split("?")[0]

        if tail == "start":
            method = "qr" if body.get("method") == "qr" else "password"
            server = str(body.get("server", "") or "")
            if server not in ("", *cloud.SERVERS):
                server = ""
            ok, err = self.cloud.start(method, str(body.get("username", "") or ""),
                                       str(body.get("password", "") or ""), server)
            self._json(200 if ok else 409, {"ok": ok, "error": err})
        elif tail == "answer":
            ok = self.cloud.answer(str(body.get("value", "")))
            self._json(200, {"ok": ok})
        elif tail == "cancel":
            self.cloud.cancel()
            self._json(200, {"ok": True})
        elif tail == "lan":
            ip = str(body.get("ip", "") or "").strip()
            token = str(body.get("token", "") or "").strip()
            if not HOSTNAME_RE.match(ip) or len(token) != 32:
                self._json(400, {"ok": False,
                                 "error": "need a bulb address and a 32-character token"})
                return
            ok, message = self.cloud.enable_lan(ip, token)
            self._json(200 if ok else 500, {"ok": ok, "error": "" if ok else message,
                                            "message": message})
        else:
            self._send(404, "not found", "text/plain")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="", help="bulb address; can also be set "
                   "from the desk once the cloud login has found it")
    p.add_argument("--token", help="only for --enable-lan")
    p.add_argument("--enable-lan", action="store_true")
    p.add_argument("--mic", help="exact audio device name to capture")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--allow-remote-login", action="store_true",
                   help="permit the Xiaomi cloud login from a non-loopback "
                        "bind; the desk has no authentication, so only on a "
                        "network you trust")
    args = p.parse_args()

    if args.enable_lan:
        if not args.ip or not args.token:
            p.error("--enable-lan needs --ip and --token")
        enable_lan(args.ip, args.token)
        return

    state = State(args.ip, args.mic)
    Engine(state).start()
    Handler.state = state
    Handler.cloud = CloudSession()
    loopback = args.host in ("127.0.0.1", "localhost", "::1", "")
    Handler.cloud_enabled = loopback or args.allow_remote_login
    Handler.cloud_reason = ("" if Handler.cloud_enabled else
                            "cloud login is off because the desk is bound to "
                            f"{args.host} with no authentication - restart with "
                            "--allow-remote-login if that is what you want")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "localhost" if args.host == "127.0.0.1" else args.host
    print(f"Desk ready at http://{shown}:{args.port}  (bulb {args.ip})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
