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
import os
import re
import signal
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from music_bulb import (PALETTES, SAMPLERATE, BLOCK, Analyser, ColorEngine,
                        LevelEngine, pick_recorder, enable_lan,
                        SYSTEM_AUDIO, system_audio_blocked,
                        Capture, ScreenColorEngine, ScreenColorThread,
                        CapturePacer, crop_letterbox, screen_permission,
                        Link, BulbUnreachable, White, screen_frame, same_light,
                        ramp_ms, BeatSync, LED_GAP, AutoPicker, MotionMeter,
                        AUTO_CAPTURE, CAPTURE_BACKENDS, PauseLight,
                        DropFx, beat_flow)

import music_bulb
import palettes as pal_rules
import bulb_alone as alone
from palettes import PaletteError, clean_anchors
from lamp_signals import Signals, clean_alert

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
    """[{i, w, h, x, y}] for every real display, or [] if mss is unavailable.
    x, y is the display's origin in the desktop, which a region is offset by."""
    try:
        import mss
        with mss.MSS() as sct:
            return [{"i": i, "w": m["width"], "h": m["height"],
                     "x": m["left"], "y": m["top"]}
                    for i, m in enumerate(sct.monitors) if i > 0]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Every setting the desk can change goes through clean_patch(), whether it
# came from the page or from the settings file, so a hand-edited or stale
# file is held to the same limits as a POST.
NUMERIC = {                         # key: (low, high, type) - the faders' ranges
    "sensitivity": (0.6, 2.6, float), "gamma": (1.0, 3.0, float),
    "punch": (0.2, 1.0, float), "fps": (5, 30, int), "fade": (0.0, 6.0, float),
    "lead": (0, 250, int),          # beat sync, ms early; 0 = flash when heard
    "screen_fps": (4, 30, int), "screen_gamma": (0.4, 1.6, float),
    "min_sat": (0.0, 90.0, float), "sat_boost": (0.8, 2.0, float),
    "pause": (0, 120, int),         # pause light, s of a still picture; 0 = off
}
MODES = ("music", "screen", "fusion")
# What a preset holds: how the light looks. Never the address, input, display
# or area (they describe the room, not the look), never Beat sync, which
# is set once for the network and speakers, and never the pause light, a
# habit of the viewer's that picking Cinema must not switch off.
LOOK_KEYS = ("mode", "auto", "palette", "calm", "sensitivity", "gamma", "punch",
             "fps", "fade", "screen_fps", "screen_gamma", "min_sat", "sat_boost")
LOOK_DEFAULTS = {"mode": "music", "auto": False, "palette": "auto", "calm": False,
                 "sensitivity": 1.4, "gamma": 1.9, "punch": 0.78, "fps": 25,
                 "fade": 0.0, "screen_fps": 10, "screen_gamma": 0.75,
                 "min_sat": 55.0, "sat_boost": 1.35}
# Built-in looks. Each sets every look key, so applying one always gives the
# same light whatever came before; keys its mode does not use stay at the
# defaults above.
BUILTIN_PRESETS = [
    # dance music: more onsets pass (1.2), deeper dips between hits (2.3),
    # hits near full, colour steps snapping
    {"name": "Club", "look": dict(LOOK_DEFAULTS, sensitivity=1.2, gamma=2.3, punch=0.95)},
    # background music: only the clear beats (1.9), a flatter level (1.5),
    # soft hits, the calm decays, and the hue sweeping between anchors
    {"name": "Lounge", "look": dict(LOOK_DEFAULTS, calm=True, sensitivity=1.9, gamma=1.5,
                                    punch=0.45, fps=20, fade=2.5)},
    # films: slower follow (calm), dark scenes left darker (0.9), pastel
    # grading kept pastel (floor 40, boost 1.15)
    {"name": "Cinema", "look": dict(LOOK_DEFAULTS, mode="screen", calm=True, screen_gamma=0.9,
                                    min_sat=40.0, sat_boost=1.15)},
    # concert films and music videos: the picture's colour, the music's hits
    {"name": "Concert", "look": dict(LOOK_DEFAULTS, mode="fusion", gamma=2.0, punch=0.85,
                                     screen_fps=12)},
]
MAX_PRESETS = 24
PRESET_NAME_MAX = 32
MIN_REGION = 32                     # points; smaller than this is a mis-click
# Lamp mode's Follow the day: white moves with local time while no session
# runs. Morning and evening are minutes after midnight, on a 15-minute grid.
LAMP_NUMERIC = {"day_k": (2700, 6500, int), "night_k": (2700, 6500, int),
                "wake": (0, 1425, int), "dusk": (0, 1425, int)}
LAMP_DEFAULTS = {"follow_day": False, "day_k": 5000, "night_k": 2700,
                 "wake": 7 * 60, "dusk": 20 * 60,
                 # epoch seconds a hand-set temperature holds the curve off until
                 "day_held": None}
LAMP_SAVED = tuple(LAMP_DEFAULTS)
# what survives a restart. Never `running` (the lamp must not start on its
# own) and nothing from the cloud login (no secret touches disk).
SAVED_KEYS = ("ip", "did", "mic", "mode", "auto", "palette", "calm", *NUMERIC,
              "bulb_clock", "drops",
              "monitor", "region", "presets", "palettes", "idle", "sleep",
              *LAMP_SAVED)
# Signals (lamp_signals.py): Focus lengths in minutes and the sunrise time in
# minutes past midnight are clamped like the faders (but are not part of a
# preset's look); the sunrise switch is a bool. Whether Focus is running is never saved.
SIGNAL_NUMERIC = {"focus_work": (5, 90, int), "focus_break": (1, 30, int),
                  "sunrise_at": (0, 1439, int)}
SIGNAL_DEFAULTS = {"focus_work": 25, "focus_break": 5, "sunrise_at": 7 * 60,
                   "sunrise_on": False}
SAVED_KEYS += (*SIGNAL_NUMERIC, "sunrise_on")


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def clean_region(region, mon):
    """[x, y, w, h] in the display's own points, or None for the whole
    display. Raises ValueError for anything that does not fit `mon`."""
    if region is None:
        return None
    if mon is None or not isinstance(region, (list, tuple)) or len(region) != 4 \
            or not all(_number(v) for v in region):
        raise ValueError("region")
    x, y, w, h = (int(round(v)) for v in region)
    if x < 0 or y < 0 or w < MIN_REGION or h < MIN_REGION \
            or x + w > mon["w"] or y + h > mon["h"]:
        raise ValueError("region")
    if (x, y, w, h) == (0, 0, mon["w"], mon["h"]):
        return None
    return [x, y, w, h]


def clean_name(name):
    """A preset name as it will be kept, or None if it cannot be one."""
    if not isinstance(name, str):
        return None
    name = " ".join(name.split())
    if not name or len(name) > PRESET_NAME_MAX or not name.isprintable():
        return None
    return name


def clean_look(look, snap):
    """The look keys of `look` that are valid, through clean_patch."""
    if not isinstance(look, dict):
        return {}
    return clean_patch({k: look[k] for k in LOOK_KEYS if k in look}, snap)


def clean_presets(presets, snap):
    """The user's presets from a file or the desk: a list of {name, look},
    names unique (case aside) and not a built-in's, looks cleaned. Anything
    broken is dropped on its own."""
    if not isinstance(presets, list):
        return []
    taken = {p["name"].casefold() for p in BUILTIN_PRESETS}
    out = []
    for p in presets[:MAX_PRESETS]:
        if not isinstance(p, dict):
            continue
        name = clean_name(p.get("name"))
        look = clean_look(p.get("look"), snap)
        if name is None or name.casefold() in taken or not look:
            continue
        taken.add(name.casefold())
        out.append({"name": name, "look": look})
    return out


def find_preset(name, snap):
    for p in BUILTIN_PRESETS + list(snap.get("presets") or []):
        if p["name"] == name:
            return p
    return None


def preset_op(op, snap):
    """Save, rename or delete one of the user's presets.
    -> (new list of presets, error or "")."""
    mine = [dict(p) for p in snap.get("presets") or []]
    if not isinstance(op, dict):
        return mine, "bad request"
    kind, name = op.get("op"), clean_name(op.get("name"))
    builtin = {p["name"].casefold() for p in BUILTIN_PRESETS}
    index = next((i for i, p in enumerate(mine) if name and p["name"] == name), None)
    if kind == "save":
        if name is None:
            return mine, "Give the preset a name."
        if name.casefold() in builtin:
            taken = next(p["name"] for p in BUILTIN_PRESETS
                         if p["name"].casefold() == name.casefold())
            return mine, f"{taken} is a built-in preset. Choose another name."
        look = clean_look({k: snap[k] for k in LOOK_KEYS}, snap)
        if index is not None:
            mine[index] = {"name": name, "look": look}
        elif any(p["name"].casefold() == name.casefold() for p in mine):
            return mine, f"There is already a preset called {name}."
        elif len(mine) >= MAX_PRESETS:
            return mine, f"{MAX_PRESETS} presets is the most the desk keeps."
        else:
            mine.append({"name": name, "look": look})
        return mine, ""
    if index is None:
        return mine, "No such preset of yours."
    if kind == "delete":
        del mine[index]
        return mine, ""
    if kind == "rename":
        to = clean_name(op.get("to"))
        if to is None:
            return mine, "Give the preset a name."
        if to.casefold() in builtin or any(
                p["name"].casefold() == to.casefold() for i, p in enumerate(mine) if i != index):
            return mine, f"There is already a preset called {to}."
        mine[index] = {"name": to, "look": mine[index]["look"]}
        return mine, ""
    return mine, "bad request"


def all_palettes(snap):
    """{name: anchors} - the built-ins, then the user's own."""
    out = dict(PALETTES)
    for p in snap.get("palettes") or []:
        out[p["name"]] = [tuple(a) for a in p["anchors"]]
    return out


def _taken_palette(name, mine, skip=None):
    """The name `name` clashes with, or None. Case is ignored."""
    key = name.casefold()
    for other in (*PALETTES, *pal_rules.RESERVED):
        if other.casefold() == key:
            return other
    for i, p in enumerate(mine):
        if i != skip and p["name"].casefold() == key:
            return p["name"]
    return None


def clean_palette_name(name):
    name = clean_name(name)
    return name if name and len(name) <= pal_rules.NAME_MAX else None


def clean_palettes(palettes):
    """The user's palettes from a file: [{name, anchors}], each valid on its
    own, names unique and not a built-in's. Anything broken is dropped."""
    if not isinstance(palettes, list):
        return []
    out = []
    for p in palettes[:pal_rules.MAX_PALETTES]:
        if not isinstance(p, dict):
            continue
        name = clean_palette_name(p.get("name"))
        try:
            anchors = clean_anchors(p.get("anchors"))
        except PaletteError:
            continue
        if name is None or _taken_palette(name, out):
            continue
        out.append({"name": name, "anchors": [list(a) for a in anchors]})
    return out


def palette_op(op, snap):
    """Save, rename or delete one of the user's palettes.
    -> (patch for the state, error or "").

    A rename carries the selected palette and every preset that names it to
    the new name. A delete leaves them on "auto" - the switching between
    palettes - rather than on a palette that is gone."""
    mine = [dict(p) for p in snap.get("palettes") or []]
    if not isinstance(op, dict):
        return {}, "bad request"
    kind, name = op.get("op"), clean_palette_name(op.get("name"))
    index = next((i for i, p in enumerate(mine) if name and p["name"] == name), None)

    def follow(old, new):
        """Point the setting and the presets at `new` instead of `old`."""
        patch = {"palettes": mine}
        if snap.get("palette") == old:
            patch["palette"] = new
        presets = [dict(p, look=dict(p["look"], palette=new))
                   if p["look"].get("palette") == old else p
                   for p in snap.get("presets") or []]
        if presets != list(snap.get("presets") or []):
            patch["presets"] = presets
        return patch

    if kind == "save":
        if name is None:
            return {}, f"Give the palette a name of up to {pal_rules.NAME_MAX} characters."
        try:
            anchors = [list(a) for a in clean_anchors(op.get("anchors"))]
        except PaletteError as exc:
            return {}, str(exc)
        if index is not None:
            mine[index] = {"name": name, "anchors": anchors}
        else:
            taken = _taken_palette(name, mine)
            if taken:
                return {}, f"There is already a palette called {taken}."
            if len(mine) >= pal_rules.MAX_PALETTES:
                return {}, f"{pal_rules.MAX_PALETTES} palettes is the most the desk keeps."
            mine.append({"name": name, "anchors": anchors})
        return {"palettes": mine}, ""
    if index is None:
        return {}, "No such palette of yours."
    if kind == "delete":
        del mine[index]
        return follow(name, "auto"), ""
    if kind == "rename":
        to = clean_palette_name(op.get("to"))
        if to is None:
            return {}, f"Give the palette a name of up to {pal_rules.NAME_MAX} characters."
        taken = _taken_palette(to, mine, skip=index)
        if taken:
            return {}, f"There is already a palette called {taken}."
        mine[index] = {"name": to, "anchors": mine[index]["anchors"]}
        return follow(name, to), ""
    return {}, "bad request"


ART_SIDE = 128                      # px: the page sends a cover at most this big


def extract_palette(body):
    """A cover the page decoded and shrank -> (anchors or None, error).
    `body` is {w, h, rgb}: rgb is base64 of w*h RGB bytes, row by row. The
    page does the decoding, so the desk needs no image library and never
    fetches anything; None means the art has no colour to take."""
    if not isinstance(body, dict):
        return None, "bad request"
    w, h, data = body.get("w"), body.get("h"), body.get("rgb")
    if not (isinstance(w, int) and isinstance(h, int) and 1 <= w <= ART_SIDE
            and 1 <= h <= ART_SIDE and isinstance(data, str)):
        return None, "bad picture"
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError:
        return None, "bad picture"
    if len(raw) != w * h * 3:
        return None, "bad picture"
    return pal_rules.palette_from_image(
        np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)), ""


def clean_patch(patch, snap):
    """The subset of `patch` that is valid against the current state, with
    numbers clamped to the faders' ranges. Unknown keys are dropped."""
    if not isinstance(patch, dict):
        return {}
    out = {}
    # the user's palettes first: the palette setting and the presets' looks
    # are checked against them
    if "palettes" in patch:             # from the settings file; the page uses /api/palettes
        out["palettes"] = clean_palettes(patch["palettes"])
        snap = dict(snap, palettes=out["palettes"])
    for key, (lo, hi, kind) in (*NUMERIC.items(), *SIGNAL_NUMERIC.items()):
        v = patch.get(key)
        if _number(v):
            v = min(max(v, lo), hi)
            out[key] = int(round(v)) if kind is int else float(v)
    for key in ("running", "calm", "auto", "bulb_clock", "drops"):
        if isinstance(patch.get(key), bool):
            out[key] = patch[key]
    if patch.get("palette") == "auto" or (isinstance(patch.get("palette"), str)
                                          and patch["palette"] in all_palettes(snap)):
        out["palette"] = patch["palette"]
    if patch.get("mode") in MODES:
        out["mode"] = patch["mode"]
    if "ip" in patch:
        ip = str(patch["ip"] or "").strip()
        if HOSTNAME_RE.match(ip):
            out["ip"] = ip
    if "did" in patch:                  # the bulb's miio id, learned by Link
        did = patch["did"]
        if did is None or (_number(did) and int(did) == did and 0 <= did < 2 ** 32):
            out["did"] = None if did is None else int(did)
    if "mic" in patch:                  # None is the automatic choice
        mic = patch["mic"]
        if mic is None or mic == "":
            out["mic"] = None
        elif isinstance(mic, str) and len(mic) <= 200 and mic.isprintable():
            out["mic"] = mic
    mons = {m["i"]: m for m in snap.get("monitors", [])}
    v = patch.get("monitor")
    if _number(v) and int(v) == v and int(v) in mons:
        out["monitor"] = int(v)
        if out["monitor"] != snap.get("monitor"):
            out["region"] = None        # a box drawn on one display means nothing on another
    if "region" in patch:
        try:
            out["region"] = clean_region(
                patch["region"], mons.get(out.get("monitor", snap.get("monitor"))))
        except ValueError:
            pass
    if "presets" in patch:              # from the settings file; the page uses /api/presets
        out["presets"] = clean_presets(patch["presets"], snap)
    if patch.get("idle") in alone.IDLE_LOOKS:   # what Stop leaves on the lamp
        out["idle"] = patch["idle"]
    if "sleep" in patch:                # from the settings file; the page uses /api/sleep
        out["sleep"] = alone.clean_sleep(patch["sleep"])
    out.update(clean_lamp(patch, snap))
    if isinstance(patch.get("sunrise_on"), bool):
        out["sunrise_on"] = patch["sunrise_on"]
    return out


# lamp mode; the lamp CLI reads the same settings file, so the path lives there
import socket  # noqa: E402
from music_bulb import (settings_path, lamp_link, lamp_commands, lamp_state,  # noqa: E402
                        LAMP_PROPS, LAMP_FADE_MS, LAMP_OPS, find_bulb)


class SettingsStore:
    """The desk's settings file: read once at start, written after changes.

    A fader drag posts many patches a second, so writes wait for a 1 s lull
    (5 s at most) and go to a temp file that replaces the old one, so a
    crash mid-write leaves the previous file whole. Anything wrong with the
    file - missing, unreadable, not JSON, out of range - costs only the
    settings it affects; the desk always starts.
    """

    VERSION = 1

    def __init__(self, path, quiet=1.0, longest=5.0):
        self.path = path
        self.quiet, self.longest = quiet, longest
        self.state = None
        self._dirty = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._written = None
        self._thread = None

    def load(self):
        """The saved settings, unvalidated (State runs them through
        clean_patch), or {} if there is nothing usable."""
        if not self.path:
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            settings = data.get("settings") if isinstance(data, dict) else None
            if not isinstance(settings, dict):
                raise ValueError("no settings object")
        except FileNotFoundError:
            return {}
        except Exception as exc:
            print(f"ignoring unreadable settings file {self.path}: {exc}", file=sys.stderr)
            return {}
        return {k: v for k, v in settings.items() if k in SAVED_KEYS}

    def attach(self, state):
        self.state = state
        self._written = self._payload()      # what was loaded needs no rewrite
        if self.path:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def touch(self):
        self._dirty.set()

    def _payload(self):
        snap = self.state.snapshot()
        return {k: snap[k] for k in SAVED_KEYS}

    def _run(self):
        while not self._stop.is_set():
            self._dirty.wait()
            first = time.monotonic()
            while not self._stop.is_set():   # wait for a lull in the changes
                self._dirty.clear()
                left = self.longest - (time.monotonic() - first)
                if left <= 0 or not self._dirty.wait(min(self.quiet, left)):
                    break
            self.flush()

    def flush(self):
        if not self.path or self.state is None:
            return
        with self._lock:
            payload = self._payload()
            if payload == self._written:
                return
            folder = os.path.dirname(self.path)
            try:
                os.makedirs(folder, mode=0o700, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=folder)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump({"version": self.VERSION, "settings": payload}, f,
                                  indent=2, sort_keys=True)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, self.path)
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp)
                    raise
                self._written = payload
            except OSError as exc:
                print(f"could not save settings to {self.path}: {exc}", file=sys.stderr)

    def close(self):
        self._stop.set()
        self._dirty.set()
        self.flush()


class State:
    def __init__(self, saved=None, store=None, local=True):
        self._lock = threading.Lock()
        # the engine parks on `started` instead of polling for a session, and
        # a loop idling on a still picture ends its wait early on `changed`
        self.started = threading.Event()
        self.stopped = threading.Event()        # the inverse: ends a reconnect wait
        self.stopped.set()
        self.changed = threading.Event()
        self.store = store
        mons = screen_monitors()
        screen_ok = screen_permission() is not False
        self._d = {
            "ip": "", "did": None, "mic": None,
            "running": False, "status": "idle",
            # the note under the top bar: None or {kind, text, action}
            "notice": None,
            "mode": "music",
            # Auto picks the mode; `mode` keeps the last one chosen by hand
            "auto": False, "auto_mode": None,
            "palette": "auto", "calm": False,
            "sensitivity": 1.4, "gamma": 1.9, "punch": 0.78, "fps": 25,
            "fade": 0.0, "lead": 80,
            # bulb-timed beats (off = today's block-paced flashes); drop effects
            "bulb_clock": False, "drops": False,
            "presets": [],                  # the user's own; BUILTIN_PRESETS are fixed
            # what Stop leaves on the lamp (bulb_alone.IDLE_LOOKS), and the
            # bulb's own off timer: None or {at: epoch s, step: the faded look}
            "idle": "restore", "sleep": None,
            "palettes": [],                 # the user's own [{name, anchors}]; PALETTES are fixed
            # screen mode
            "monitor": mons[0]["i"] if mons else 1,
            "screen_fps": 10, "screen_gamma": 0.75,
            "min_sat": 55.0, "sat_boost": 1.35, "screen_width": 192,
            "pause": 0, "paused": False,
            "region": None,
            "monitors": mons,
            "screen_ok": screen_ok,
            # a still of the display for drawing the region; it is a picture
            # of the user's screen, so only a desk bound to loopback serves it
            "snapshot_ok": bool(local and mons and screen_ok),
            # telemetry
            "hue": 0, "sat": 0, "val": 0, "white": False, "ct": 0, "bpm": 0.0,
            "level": 0.0, "hit": 0.0, "bands": [0.0] * 10,
            "input_db": None, "signal": True,
            "active_palette": "ember", "device": "-",
            # the screen capture in use: "sck", "mss", or None when not capturing
            "capture": None,
        }
        # lamp mode: Follow the day, and the lamp as last read (never saved)
        self._d.update(LAMP_DEFAULTS, lamp=None, lamp_error=None)
        self._d.update(SIGNAL_DEFAULTS)
        if saved:
            self._d.update(clean_patch(saved, self._d))

    def snapshot(self):
        with self._lock:
            return dict(self._d)

    def set(self, **kw):
        with self._lock:
            self._d.update(kw)
            if "running" in kw:
                (self.started.set if kw["running"] else self.started.clear)()
                (self.stopped.clear if kw["running"] else self.stopped.set)()

    def control(self, **kw):
        """set(), for a change that came from the desk rather than telemetry."""
        self.set(**kw)
        self.changed.set()
        if self.store:
            self.store.touch()


class SignalWatch:
    """The input's level in dBFS, and whether it has gone quiet.

    A silent input looks exactly like a broken lamp - the bulb idles at a few
    percent - so the desk says so outright once the raw RMS has stayed under
    -70 dBFS for 1.5 s. BlackHole fed by nothing is exactly 0, a quiet room
    on a built-in microphone sits around -60.
    """

    FLOOR_DB = -70.0
    HOLD = 1.5

    def __init__(self):
        self.quiet_since = None
        self.peak = 0.0

    def feed(self, rms, now):
        self.peak = max(self.peak, rms)
        if 20 * math.log10(max(rms, 1e-7)) >= self.FLOOR_DB:
            self.quiet_since = None
        elif self.quiet_since is None:
            self.quiet_since = now

    def read(self, now):
        """(dB of the loudest block since the last read, signal present)."""
        db = round(max(20 * math.log10(max(self.peak, 1e-7)), -120.0), 1)
        self.peak = 0.0
        return db, not (self.quiet_since is not None
                        and now - self.quiet_since >= self.HOLD)


def open_recorder(name):
    """pick_recorder(), with the desk's wording when a saved input is gone."""
    try:
        return pick_recorder(name, quiet=True)
    except RuntimeError:
        # system audio says itself what it is missing
        if name and name.strip().lower() != SYSTEM_AUDIO.lower():
            raise RuntimeError(f"{name} is not connected. Choose another input.")
        raise


def audio_inputs():
    """What the input picker lists: every capture device, and the name the
    automatic choice resolves to right now."""
    names = []
    with contextlib.suppress(Exception):
        if system_audio_blocked() is None:
            names.append(SYSTEM_AUDIO)
    with contextlib.suppress(Exception):
        import soundcard as sc
        names += [m.name for m in sc.all_microphones() if m.name not in names]
    auto = ""
    with contextlib.suppress(Exception):
        auto = str(pick_recorder(None, quiet=True).name)
    return {"inputs": names, "auto": auto}


class MotionProbe(threading.Thread):
    """Music mode under Auto: is the picture moving? One grab every
    MotionMeter.PROBE seconds - a twentieth of screen mode's 10 a second -
    and half that after a minute of stillness, into the picker."""

    def __init__(self, grab, picker):
        super().__init__(daemon=True)
        self.grab, self.picker = grab, picker
        self.meter = MotionMeter()
        self.halt = threading.Event()

    def run(self):
        still_since = time.monotonic()
        wait = 0.0
        while not self.halt.wait(wait):
            now = time.monotonic()
            try:
                moving = self.meter.feed(self.grab.frame(), now)
            except Exception:
                return                          # display gone: the picker keeps its view
            self.picker.feed_motion(moving, now)
            if moving:
                still_since = now
            wait = MotionMeter.PROBE * (2 if now - still_since > 60 else 1)

    def close(self):
        self.halt.set()
        self.join(2)
        self.grab.close()


class SoundProbe(threading.Thread):
    """Screen mode under Auto: is there sound? Reads the input in 93 ms
    blocks and keeps only the level - no analysis, nothing sent."""

    def __init__(self, recorder, picker):
        super().__init__(daemon=True)
        self.recorder, self.picker = recorder, picker
        self.halt = threading.Event()

    def run(self):
        try:
            with self.recorder.recorder(samplerate=SAMPLERATE, channels=2) as rec:
                while not self.halt.is_set():
                    pcm = rec.record(numframes=4 * BLOCK)
                    self.picker.feed_sound(float(np.sqrt(np.mean(np.square(pcm)))),
                                           time.monotonic())
        except Exception:
            return                              # input gone: no sound to report

    def close(self):
        self.halt.set()
        self.join(2)


def region_box(cfg):
    """The absolute X, Y, W, H for ScreenGrabber, or None for the whole display."""
    region = cfg.get("region")
    if not region:
        return None
    mon = next((m for m in cfg["monitors"] if m["i"] == cfg["monitor"]), None)
    if mon is None:
        return None
    x, y, w, h = region
    return (mon["x"] + x, mon["y"] + y, w, h)


def display_snapshot(monitor, width=640):
    """One PNG still of a display, about `width` px wide, for drawing a region."""
    import mss
    import mss.tools
    with mss.MSS() as sct:
        mons = sct.monitors
        if not 1 <= monitor < len(mons):
            raise ValueError("no such display")
        shot = sct.grab(mons[monitor])
    a = np.asarray(shot)                                  # BGRA, device pixels
    step = max(1, int(math.ceil(a.shape[1] / width)))
    rgb = np.ascontiguousarray(a[::step, ::step, 2::-1])
    return mss.tools.to_png(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]))


# ---------------------------------------------------------------------------
def light_telemetry(light, hue, level):
    """What the page needs to paint the light: hue/sat/val on the colour
    LEDs' scale always, and for a White frame its temperature. `val` stays
    the matched level, not the white LEDs' own number, so the colour field
    does not jump when the lamp changes LEDs."""
    if isinstance(light, White):
        return {"hue": int(round(hue)) % 360, "sat": 0, "val": int(level),
                "white": True, "ct": light.ct}
    return {"hue": light[0], "sat": light[1], "val": light[2], "white": False, "ct": 0}


def apply_palette(color, name, palettes=None):
    """Point the music engine at the setting: a palette, or "auto". A new
    `palettes` mapping (one added, edited or deleted) is taken as it is."""
    if palettes is not None:
        color.palettes = palettes
    if name == "auto" or name not in color.palettes:
        color.locked = False
        return
    if color.pal != name or not color.locked:
        color.pal = name
        color.locked = True
        color.idx = 0
        color.retarget(*color.palettes[name][0])


def lamp_error(exc):
    """A lamp_call failure in the page's words."""
    if "-5006" in str(exc) or "illegal" in str(exc):
        return "The lamp is off."
    return "The bulb did not answer."


class SleepTimer:
    """The bulb's own off timer, as the desk sees it.

    The bulb counts (cron_add), so the lamp switches off on time whether the
    desk runs, quits or the Mac sleeps. The desk keeps `sleep` = {at, step}
    in its settings and arms one threading.Timer for `at`: no polling, a
    stopped desk sleeps until then. `at` is wall time: a timer whose `at`
    has passed is over, even if the threading.Timer (which does not count
    system sleep) has not fired yet.

    `step` is the desk's own fade: the look a fade flow on the bulb is
    dimming, set when one was sent and None otherwise. It, not the bulb's
    `flowing`, says whose flow is running, so a new time replaces our fade
    and Cancel ends it, while an idle look is left alone. A session's frames
    replace any flow, so a session clears it; on Stop, _leave() hands the
    fade back to the bulb or clears it.

    When the timer runs out during a session that began before `at`,
    `fired` makes that session end without the restore, which would switch
    the lamp back on. A session clears `fired` when it starts.

    Every read-modify-write of `sleep` and every lamp call holds `_lock`, so
    the HTTP threads, the Timer callback and the session's Stop see one
    order. Each arm gets a new generation; a callback from an older one
    does nothing.

    The bulb is reached with one short connection (`call`): during a session
    yeelight has closed the control socket and the music connection never
    answers, so a reply (the lamp is off, -5006) can only come back on a
    connection of its own. _lock keeps it to one at a time.
    """

    def __init__(self, state, call=None, gate=None):
        self.state = state
        self.call = call or self._call
        self.gate = gate                        # the desk's lamp lock
        self.fired = threading.Event()
        self.kick = threading.Event()           # wakes the session's fader
        self.session_from = None                # wall time the running session began
        self._timer = None
        self._gen = 0
        self._lock = threading.RLock()
        cur = state.snapshot()["sleep"]
        if cur:
            self._arm(cur["at"])

    @staticmethod
    def _call(ip, batches):
        """Each batch on one short connection, in turn. -> their results."""
        sock = socket.create_connection((ip, music_bulb.LAMP_PORT), 2)
        try:
            out = []
            for cmds in batches:
                cmds = cmds(out) if callable(cmds) else cmds
                out.append(music_bulb.lamp_call(ip, cmds, sock=sock) if cmds else [])
            return out
        finally:
            sock.close()

    def plain(self, running=False):
        """The lamp lock for a call made while no session runs; a session's
        own timer call goes beside it on a short connection of its own."""
        return contextlib.nullcontext() if running or self.gate is None else self.gate

    def current(self, now=None):
        """The timer if it is still ahead, else None."""
        cur = self.state.snapshot()["sleep"]
        now = time.time() if now is None else now
        return cur if cur and cur["at"] > now else None

    def left(self, now=None):
        now = time.time() if now is None else now
        cur = self.current(now)
        return None if cur is None else cur["at"] - now

    def _arm(self, at):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._gen += 1
            self._timer = threading.Timer(max(0.0, at - time.time()), self._fire, (self._gen,))
            self._timer.daemon = True
            self._timer.start()
        self.kick.set()

    def _disarm(self):
        with self._lock:
            self._gen += 1
            if self._timer:
                self._timer.cancel()
                self._timer = None
        self.kick.set()

    def _fire(self, gen=None):
        with self._lock:
            if gen is not None and gen != self._gen:
                return                          # disarmed or re-armed meanwhile
            cur = self.state.snapshot()["sleep"]
            if cur and cur["at"] > time.time() + 0.5:
                self._arm(cur["at"])            # early (clock moved): wait again
                return
            self._timer = None
            snap = self.state.snapshot()
            began = self.session_from
            if snap["running"] and cur and began is not None and began < cur["at"]:
                self.fired.set()
                self.state.control(sleep=None, running=False, status="sleeping", notice=None)
            else:
                # stopped, or a session started after the end time (the Mac
                # slept through it): the bulb already switched off
                self.state.control(sleep=None)

    def session_started(self):
        """A session begins: forget a timer that ran out unseen, drop the
        fade its frames are about to replace, and any stale `fired`."""
        with self._lock:
            self.fired.clear()
            self.session_from = time.time()
            cur = self.state.snapshot()["sleep"]
            if cur and cur["at"] <= self.session_from:
                self._disarm()
                self.state.control(sleep=None)
            elif cur and cur.get("step"):
                self.state.control(sleep=dict(cur, step=None))

    def session_ended(self):
        with self._lock:
            self.session_from = None
            self.fired.clear()

    def start(self, minutes):
        """-> error sentence or ''."""
        with self._lock:
            snap = self.state.snapshot()
            if not snap["ip"]:
                return "Set up the bulb first."
            cur = self.current()
            ours = cur.get("step") if cur else None
            step = None
            try:
                with self.plain(snap["running"]):
                    step = self._send_timer(snap, minutes, ours)
            except Exception as exc:
                return lamp_error(exc)
            at = time.time() + minutes * 60
            self.state.control(sleep={"at": at, "step": step})
            self._arm(at)
            return ""

    def _send_timer(self, snap, minutes, ours):
        """The timer onto the bulb. -> the look its fade dims, or None."""
        step = None
        if snap["running"]:
            # frames would replace a fade: the bulb only counts, the
            # session dims itself
            self.call(snap["ip"], [alone.sleep_commands(minutes)])
        elif ours:
            # our own fade is running: replace it, from the same look
            step = ours
            self.call(snap["ip"], [alone.sleep_commands(minutes, step)])
        else:
            props = list(music_bulb.LAMP_PROPS) + ["flowing"]

            def then(out):
                nonlocal step
                raw = dict(zip(props, out[0][0] or []))
                saved = {"power": raw.get("power")}
                for k in music_bulb.LAMP_PROPS[1:]:
                    with contextlib.suppress(TypeError, ValueError):
                        saved[k] = int(raw.get(k))
                # a flow we did not send is an idle look: it keeps
                # running and the timer alone switches it off
                if raw.get("flowing") != "1":
                    step = alone.look_step(saved)
                return alone.sleep_commands(minutes, step)

            self.call(snap["ip"], [[("get_prop", props)], then])
        return step

    def cancel(self):
        with self._lock:
            snap = self.state.snapshot()
            cur = snap["sleep"]
            self._disarm()
            self.state.control(sleep=None)
            if not snap["ip"]:
                return ""
            step = None if snap["running"] or not cur else cur.get("step")
            try:
                with self.plain(snap["running"]):
                    self.call(snap["ip"], [alone.cancel_commands(step)])
            except Exception as exc:
                return lamp_error(exc)
            return ""

    def handed_over(self, step):
        """Stop left the lamp: `step` is the fade it handed to the bulb, or
        None when the lamp shows something else (an idle look, the last
        frame, a look with no fade)."""
        with self._lock:
            cur = self.state.snapshot()["sleep"]
            if cur and cur.get("step") != step:
                self.state.control(sleep=dict(cur, step=step))


# ---------------------------------------------------------------------------
# lamp mode - the desk as a plain lamp while no session runs
# ---------------------------------------------------------------------------
DAY_TICK = 300                      # s between Follow the day's checks
FLOW_RETRY = 5                      # s: one early look after meeting a flow
DAY_STEP = 50                       # K: the curve moves in steps this big
DAY_UP = 60                         # min: night to day, from the morning time
DAY_DOWN = 120                      # min: day to night, from the evening time
DAY_FADE_MS = 3000                  # a curve step glides in rather than jumps
DAY_RESUME_HOUR = 4                 # a hand-set temperature holds until 04:00
GRID = 15                           # min: the strip's handles snap to this
MIN_DAY = 120                       # min: the evening at least this after the morning


def clean_lamp(patch, snap):
    """Follow the day's settings out of `patch`, clamped like the faders.
    The evening stays at least MIN_DAY after the morning on the same day:
    the one that moved gives way."""
    out = {}
    for key, (lo, hi, _) in LAMP_NUMERIC.items():
        v = patch.get(key)
        if _number(v):
            v = int(round(min(max(v, lo), hi)))
            if key in ("wake", "dusk"):
                v = min(int(round(v / GRID)) * GRID, hi)
            out[key] = v
    if "wake" in out or "dusk" in out:
        wake = out.get("wake", snap.get("wake", LAMP_DEFAULTS["wake"]))
        dusk = out.get("dusk", snap.get("dusk", LAMP_DEFAULTS["dusk"]))
        top = LAMP_NUMERIC["dusk"][1]
        if dusk - wake < MIN_DAY:
            if "dusk" in out:
                dusk = min(wake + MIN_DAY, top)
                wake = min(wake, dusk - MIN_DAY)
            else:
                wake = dusk - MIN_DAY
                if wake < 0:
                    wake, dusk = 0, MIN_DAY
            out.update(wake=wake, dusk=dusk)
    if isinstance(patch.get("follow_day"), bool):
        out["follow_day"] = patch["follow_day"]
    if "day_held" in patch:
        v = patch["day_held"]
        if v is None or (_number(v) and v > 0):
            out["day_held"] = v
    return out


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def day_kelvin(minute, day_k=5000, night_k=2700, wake=420, dusk=1200):
    """The white for `minute` after local midnight: night_k until `wake`, up
    to day_k over DAY_UP minutes, down again over DAY_DOWN from `dusk`, on a
    smoothstep, in DAY_STEP steps. An evening before the morning wraps past
    midnight; a day too short for both ramps peaks below day_k."""
    span = (dusk + DAY_DOWN - wake) % 1440
    into = (minute - wake) % 1440
    f = 0.0 if into >= span else min(_smooth(into / DAY_UP), _smooth((span - into) / DAY_DOWN))
    k = night_k + (day_k - night_k) * f
    return int(round(k / DAY_STEP)) * DAY_STEP


def day_target(snap, now):
    t = time.localtime(now)
    return day_kelvin(t.tm_hour * 60 + t.tm_min + t.tm_sec / 60, snap["day_k"],
                      snap["night_k"], snap["wake"], snap["dusk"])


def next_resume(now):
    """Epoch seconds of the next DAY_RESUME_HOUR:00, local time."""
    t = time.localtime(now)
    at = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, DAY_RESUME_HOUR, 0, 0, 0, 0, -1))
    if at <= now:
        t = time.localtime(now + 86400)
        at = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, DAY_RESUME_HOUR, 0, 0, 0, 0, -1))
    return at


def holding(snap, now):
    return bool(snap["day_held"]) and now < snap["day_held"]


class LampBusy(RuntimeError):
    """A lamp action the desk will not do while a session owns the bulb."""


def lamp_problem(exc, ip):
    if isinstance(exc, ConnectionRefusedError):
        return f"The bulb at {ip} refused the connection. LAN control may be off."
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return f"The bulb did not answer at {ip}."
    if isinstance(exc, OSError):
        return f"The bulb at {ip} cannot be reached ({exc.strerror or exc})."
    return f"The bulb at {ip} refused it: {exc}"


class LampMode(threading.Thread):
    """The lamp while the desk is stopped: actions from the page and the
    hotkey CLI, and Follow the day.

    Every exchange is one lamp_link() connection, closed at once, and goes
    through _turn(): it waits for a stopping session to finish its restore,
    then takes `lock` - which the engine also holds while a session opens
    and closes the bulb - and checks again that no session runs. So nothing
    lands while a session enters music mode, reads the lamp, or puts it back.

    Follow the day wakes every DAY_TICK seconds while it is on and not at
    all while it is off; it connects only when the curve's value moved since
    it last sent one, or when it last found the lamp off (so a lamp switched
    on at the wall is corrected within a tick). Manual wins: a temperature
    set here, or one it finds on the lamp that it did not send, holds the
    curve off until DAY_RESUME_HOUR; ticking the switch again resumes it at
    once. A reading taken while its own send is still fading is not judged."""

    daemon = True

    def __init__(self, state, idle=None, link=lamp_link, clock=time.time,
                 lock=None, find=find_bulb):
        super().__init__()
        self.state = state
        self.idle = idle                        # the engine's parked flag
        self.link = link
        self.clock = clock
        self.find = find
        # one exchange with the lamp at a time, shared with the engine
        self.lock = lock or threading.Lock()
        self.wake = threading.Event()
        self.sent = None                        # the curve value last sent
        self.fading_until = 0.0                 # clock() until that send has landed
        self.found_off = False
        self.searched = False                   # one search per unreachable spell
        self.flow_seen = False                  # the last tick met a flow
        # True while something else has the lamp on purpose (Focus, the
        # sunrise): the curve waits and is not mistaken for a hand change
        self.yield_to = lambda: False

    # -- the page, the CLI and the engine -------------------------------------
    def changed(self, clean):
        """A /api/control patch landed: re-tick when the curve's inputs moved."""
        if clean.get("follow_day") is True or clean.get("day_held", 0) is None:
            self.sent = None                    # resume: send without second-guessing
        if "ip" in clean:                       # another bulb, or the same one moved
            self.state.set(lamp=None, lamp_error=None)
            self.sent, self.found_off, self.searched = None, False, False
        if set(clean) & set(LAMP_DEFAULTS):
            self.wake.set()

    def session_ended(self):
        """A session let go of the bulb: whatever it left (a failed restore,
        --no-restore) is not a hand-set white, so the next tick reasserts."""
        self.sent, self.found_off = None, False
        self.wake.set()

    def _settle(self, wait=10.0):
        """Let a session that is stopping finish its restore first."""
        if self.idle is not None:
            self.idle.wait(wait)

    @contextlib.contextmanager
    def _turn(self):
        self._settle()
        with self.lock:
            snap = self.state.snapshot()
            if snap["running"]:
                raise LampBusy("a session is running")
            yield snap

    def _talk(self, snap, body):
        """Run body(call) on one connection to the lamp. A bulb that does
        not answer at its address, with a known id, is searched for once (as
        Link does for a moved lease); if it is elsewhere the address is saved
        and body runs there - nothing reached the old address, so nothing is
        sent twice."""
        ip = snap["ip"]
        try:
            with self.link(ip) as call:
                out = body(call)
        except OSError as exc:
            moved = None
            if snap["did"] is not None and not self.searched:
                self.searched = True
                moved = self.find(snap["did"], ip)
            if not moved or moved == ip:
                self.state.set(lamp_error=lamp_problem(exc, ip))
                raise
            self.state.control(ip=moved)
            self.changed({"ip": moved})
            try:
                with self.link(moved) as call:
                    out = body(call)
            except Exception as exc2:
                self.state.set(lamp_error=lamp_problem(exc2, moved))
                raise
        except Exception as exc:
            self.state.set(lamp_error=lamp_problem(exc, ip))
            raise
        self.searched = False
        return out

    def read(self):
        if self.state.snapshot()["running"]:
            raise LampBusy("a session is running")
        return self._exchange(None, None, None)

    def act(self, op=None, ct=None, bright=None):
        if self.state.snapshot()["running"]:
            if op != "toggle":
                raise LampBusy("a session is running")
            # one key press = dark: end the session (it puts the lamp back)
            # and then switch the lamp off
            self.state.control(running=False, status="stopping", notice=None)
            time.sleep(0.05)
            op = "off"
        return self._exchange(op, ct, bright)

    def _exchange(self, op, ct, bright):
        with self._turn() as snap:
            if not snap["ip"]:
                raise LampBusy("no bulb address - set one in Setup")
            now = self.clock()
            follow = snap["follow_day"] and not holding(snap, now)
            sending = {}

            def body(call):
                todo = op
                if todo == "toggle" and follow:
                    # turning on applies the curve, so find out which way first
                    (power,) = call([("get_prop", ["power"])])[0]
                    todo = "off" if power == "on" else "on"
                cmds = lamp_commands(todo, ct, bright)
                # a lamp coming on (up from off does too) gets the curve's white
                if todo in ("on", "up") and follow and ct is None:
                    sending["ct"] = day_target(snap, now)
                    cmds.append(("set_ct_abx", [sending["ct"], "smooth", LAMP_FADE_MS]))
                return call(cmds + [("get_prop", list(LAMP_PROPS))])

            results = self._talk(snap, body)
        lamp = lamp_state(results[-1])
        if "ct" in sending:                     # only once it has gone out
            self.sent = sending["ct"]
            self.fading_until = now + LAMP_FADE_MS / 1000 + 1
        self.state.set(lamp=lamp, lamp_error=None)
        # an off lamp ignores a temperature, so there is nothing to hold for
        if ct is not None and snap["follow_day"] and lamp and lamp["power"] == "on":
            self.state.control(day_held=next_resume(now))
        return lamp

    # -- Follow the day -------------------------------------------------------
    def run(self):
        while True:
            if not self.state.snapshot()["follow_day"]:
                self.wake.wait()                # off: no wakes at all
                self.wake.clear()
                continue
            try:
                what = self.tick()
            except Exception:
                what = None                     # the lamp may be away; next tick
            # a pulse or a cancelled fade is over in seconds, so the first
            # flow seen gets one early look; an On stop look runs on, and is
            # then only met on the usual tick
            soon = what == "flow" and not self.flow_seen
            self.flow_seen = what == "flow"
            self.wake.wait(FLOW_RETRY if soon else DAY_TICK)
            self.wake.clear()

    def tick(self):
        """One Follow the day check. -> what it did, for the tests."""
        snap = self.state.snapshot()
        if not snap["follow_day"] or snap["running"] or not snap["ip"]:
            return "idle"
        if self.yield_to():
            self.sent = None                    # reassert once it lets go
            return "idle"
        now = self.clock()
        if snap["day_held"]:
            if now < snap["day_held"]:
                return "held"
            self.state.control(day_held=None)
            self.sent = None
        if day_target(snap, now) == self.sent and not self.found_off:
            return "same"
        try:
            with self._turn() as snap:
                if not snap["follow_day"] or not snap["ip"]:
                    return "idle"
                target = day_target(snap, now)
                judge = self.sent is not None and now >= self.fading_until

                def body(call):
                    (power, mode, ct, *more) = call(
                        [("get_prop", ["power", "color_mode", "ct", "flowing"])])[0]
                    if power != "on":
                        return "off"
                    if more and more[0] == "1":
                        # a flow runs on the bulb (an On stop look, a sleep
                        # fade, a pulse): a set would end it
                        return "flow"
                    try:
                        mode, ct = int(mode), int(ct)
                    except (TypeError, ValueError):
                        mode, ct = None, None
                    if judge and not self.found_off and (
                            mode != 2 or ct is None or abs(ct - self.sent) > DAY_STEP):
                        return "hand"
                    call([("set_ct_abx", [target, "smooth", DAY_FADE_MS])])
                    return "sent"

                what = self._talk(snap, body)
        except LampBusy:
            return "idle"
        if what == "off":
            self.found_off = True
            return what
        self.found_off = False
        if what == "flow":
            self.sent = None                    # reassert once it has ended
            return what
        if what == "hand":
            self.state.control(day_held=next_resume(now))
            return what
        self.sent, self.fading_until = target, now + DAY_FADE_MS / 1000 + 1
        lamp = dict(snap["lamp"] or {}, power="on", color_mode=2, ct=target)
        self.state.set(lamp=lamp, lamp_error=None)
        return "sent"


class Engine(threading.Thread):
    daemon = True

    def __init__(self, state, restore=True, capture="auto"):
        super().__init__()
        self.state = state
        self.restore = restore
        self.capture = capture                  # auto / sck / mss
        self.idle = threading.Event()           # set while parked, for a clean exit
        self.idle.set()
        self.picker = None                      # an AutoPicker while Auto is on
        # the lamp lock: held while a session opens and closes the bulb and
        # by every plain exchange with the lamp while no session runs (lamp
        # mode, the signals, the sleep timer, the power-on look), so none
        # lands in the middle of another or of a session's open or restore.
        # Re-entrant: a session's Start claims it and ends Focus inside it.
        self.lamp_lock = threading.RLock()
        self.sleep = SleepTimer(state, gate=self.lamp_lock)
        self.on_session_end = None              # told when a session lets go
        self.link = None                        # the session's Link, for alerts

    def _mode(self, cfg):
        """The mode to run: the picker's choice under Auto, else the desk's.
        None right after Auto is switched, which ends the running loop so the
        session can start the next one with (or without) its probe."""
        if bool(cfg["auto"]) != (self.picker is not None):
            return None
        if self.picker is None:
            return cfg["mode"]
        mode = self.picker.pick(time.monotonic())
        if mode != cfg["auto_mode"]:
            self.state.set(auto_mode=mode)
        return mode

    def _watch(self, cfg, start=None):
        """Keep the picker in step with the Auto switch."""
        if not cfg["auto"]:
            self.picker = None
        elif self.picker is None:
            self.picker = AutoPicker(screen_permission() is not False,
                                     now=time.monotonic(), start=start)

    def run(self):
        while True:
            self.idle.set()
            self.state.started.wait()           # parked while the desk is stopped
            self.idle.clear()
            try:
                self._session()
            except BulbUnreachable as exc:
                self.state.set(running=False, status="error", notice={
                    "kind": "err", "text": str(exc), "action": exc.action})
                time.sleep(0.5)
            except Exception as exc:
                self.state.set(running=False, status=f"error: {exc}")
                time.sleep(0.5)

    def _session(self):
        """Own the bulb from Start to Stop.

        A mode, display, input or region change only swaps the loop: each
        loop returns when its own configuration moves, and the next one picks
        up on the same connection. So the lamp is restored once, when the
        desk really stops (or the process exits), and a change of mode does
        not cost the bulb a reconnect.
        """
        st = self.state
        cfg = st.snapshot()
        if not cfg["ip"]:
            st.set(running=False,
                   status="no bulb address - find one on the cloud deck")
            time.sleep(0.3)
            return
        # fail on a missing input or permission before the bulb is touched:
        # opening it turns the lamp on, and a restore right after would flash
        if (cfg["auto"] or cfg["mode"] != "screen") and cfg["mic"]:
            open_recorder(cfg["mic"])
        if not cfg["auto"] and cfg["mode"] != "music" and screen_permission() is False:
            raise RuntimeError("no Screen Recording permission - grant it in "
                               "System Settings > Privacy & Security")
        # Auto starts in fusion, which hears and sees everything, and makes
        # its first real pick after AutoPicker.SETTLE seconds
        self.picker = None
        self._watch(cfg)

        # Link owns the connection: it finds a bulb that moved, explains one
        # it cannot reach, and reconnects after a drop until Stop is pressed
        def report(word, notice=None):
            st.set(status=word, notice=notice and {
                "kind": notice[0], "text": notice[1], "action": None})

        def moved(ip, did):
            # a new address or a newly learned id is saved like any setting
            st.control(ip=ip, did=did)

        link = Link(cfg["ip"], did=cfg["did"], remember=self.restore,
                    status=report, stopped=st.stopped, moved=moved)
        with self.lamp_lock:
            opened = link.open()
        if not opened:
            st.set(status="stopped")
            return
        loops = {"music": self._audio_loop, "screen": self._screen_loop,
                 "fusion": self._fusion_loop}
        # a sleep timer dims the last minutes of a session, on a still
        # picture too: the fader re-sends the last frame as it dims
        dimmed = alone.Dimmed(link, lambda: alone.dim_factor(self.sleep.left()))
        self.sleep.session_started()
        done = threading.Event()
        fader = threading.Thread(target=self._fade, args=(dimmed, done), daemon=True)
        fader.start()
        self.link = link
        try:
            while True:
                cfg = st.snapshot()
                if not cfg["running"] or cfg["ip"] != link.ip:
                    break
                if link.lost:
                    if not link.recover():
                        break                   # Stop pressed while reconnecting
                    continue
                # Auto turned on mid-session starts from the mode running now
                self._watch(cfg, start=cfg["mode"])
                loops.get(self._mode(cfg), self._audio_loop)(st, dimmed)
        finally:
            self.picker = None
            self.link = None
            done.set()
            self.sleep.kick.set()
            fader.join(2)
            try:
                restore, then = self._leave(st.snapshot(), link)
            except Exception as exc:
                # whatever went wrong, the sender must stop and the socket close
                print(f"could not work out what to leave on the lamp: {exc}", file=sys.stderr)
                restore, then = self.restore, []
            self.sleep.session_ended()
            if restore and link.saved and not link.lost:
                st.set(status="restoring the lamp")
            with self.lamp_lock:
                link.close(restore=restore, then=then)
            st.set(status="stopped", level=0.0, hit=0.0, bands=[0.0] * 10,
                   input_db=None, signal=True)
            if self.on_session_end:
                self.on_session_end()

    def _fade(self, dimmed, done):
        """While a session runs: re-send the last frame as the sleep fade
        dims it, every Dimmed.REFRESH s inside the fade and not at all
        outside it (a set or cancelled timer wakes it through `kick`)."""
        timer = self.sleep
        while not done.is_set():
            left = timer.left()
            if left is None:
                wait = None
            elif left > alone.SLEEP_FADE_S:
                wait = min(left - alone.SLEEP_FADE_S, 300.0)
            else:
                wait = alone.Dimmed.REFRESH
            timer.kick.wait(wait)
            timer.kick.clear()
            if done.is_set():
                return
            with contextlib.suppress(Exception):
                dimmed.refresh()

    def _leave(self, cfg, link):
        """What Stop sends: -> (restore, then) for Link.close.

        The sleep timer ran out: no restore (it would switch the lamp back
        on), just off. An idle look: that flow instead of the restore; a
        pending timer still switches it off. The restore with a timer
        pending: the restored look, then the bulb fades it for the time
        left. A lamp restored to off keeps the bulb's timer and gets no fade."""
        if self.sleep.fired.is_set():
            self.sleep.fired.clear()
            return False, [("set_power", ["off", "smooth", 500])]
        anchors = None
        if cfg["idle"] == "drift":
            pals = all_palettes(cfg)
            name = cfg["palette"] if cfg["palette"] in pals else cfg["active_palette"]
            anchors = pals.get(name) or PALETTES["ember"]
        then = alone.idle_commands(cfg["idle"], anchors)
        left = self.sleep.left()
        step = None
        if not then and left is not None and left >= 1 and self.restore and not link.lost:
            # a lamp restored to off has nothing to fade; the bulb's timer
            # stays (switching an off lamp off is harmless), so nothing
            # follows the restore's set_power off, where it could be refused
            step = alone.look_step(link.saved)
        self.sleep.handed_over(step)            # Cancel never pushes a stale look
        if then:
            return False, then
        if step:
            return True, [alone.fade_flow(step, left)]
        return self.restore, []

    # -- screen mode --------------------------------------------------------
    def _screen_loop(self, st, sender):
        cfg = st.snapshot()
        if screen_permission() is False:
            raise RuntimeError("no Screen Recording permission - grant it in "
                               "System Settings > Privacy & Security")

        monitor, region = cfg["monitor"], cfg["region"]
        grab = Capture(monitor, region_box(cfg), cfg["screen_width"],
                       max(cfg["screen_fps"], 4), self.capture)
        color = ScreenColorEngine(min_sat=cfg["min_sat"],
                                  sat_boost=cfg["sat_boost"],
                                  gamma=cfg["screen_gamma"], calm=cfg["calm"])
        st.set(status="running", active_palette="screen", bpm=0.0,
               capture=grab.backend,
               device=f"display {monitor} "
                      f"{grab.box['width']}x{grab.box['height']}")

        pacer = CapturePacer()
        pause = PauseLight(cfg["pause"])
        last = time.monotonic()
        last_hsv = None
        cut_env = 0.0
        # under Auto, listen for sound - the one input this loop does not use
        picker, meter, ears = self.picker, MotionMeter(), None
        if picker is not None:
            with contextlib.suppress(Exception):
                ears = SoundProbe(open_recorder(cfg["mic"]), picker)
                ears.start()

        try:
            while True:
                cfg = st.snapshot()
                if not cfg["running"] or self._mode(cfg) != "screen" or sender.lost:
                    break
                if cfg["monitor"] != monitor or cfg["region"] != region:
                    break                       # recapture the new display or box

                color.calm = cfg["calm"]
                color.min_sat = float(cfg["min_sat"])
                color.sat_boost = float(cfg["sat_boost"])
                color.gamma = float(cfg["screen_gamma"])
                pause.delay = float(cfg["pause"])

                now = time.monotonic()
                dt, last = now - last, now
                frame = grab.frame()
                hue, sat, val, cut = color.tick(
                    dt, *color.measure(crop_letterbox(frame)))
                base = 1.0 / max(cfg["screen_fps"], 4)
                period = pacer.period(frame, max(cfg["screen_fps"], 4), now)
                if picker is not None:
                    picker.feed_motion(meter.feed(frame, now), now)

                v = int(round(float(np.clip(3 + 97 * val, 1, 100))))
                light = screen_frame(color, hue, sat, v)
                # a picture held still for `pause` s: soft warm white instead
                if pause.feed(pacer.moved_at, now):
                    light, v = pause.frame(), PauseLight.LEVEL

                # hue histogram, folded to 12 bars for the UI
                cut_env = max(cut_env * 0.72, 1.0 if cut else 0.0)
                hist = color.hist.reshape(12, 3).sum(axis=1)
                peak = float(hist.max())
                st.set(**light_telemetry(light, hue, v), capture=grab.backend,
                       paused=pause.on,
                       level=round(float(val), 3), hit=round(cut_env, 3),
                       bands=[round(float(b) / peak, 3) for b in hist]
                             if peak > 0 else [0.0] * 12)

                if cut or pause.entered or pause.left or not same_light(light, last_hsv):
                    # ramp for one frame gap so consecutive colours join up;
                    # into a pause slowly, out of it quickly
                    sender.put(light, sudden=cut and not pause.on,
                               duration=pause.ramp(light, last_hsv, period))
                    last_hsv = light

                # still picture: wait out the longer gap, but a moved control
                # (or, on a stream, a new picture) ends it so the desk never
                # feels slow
                grab.set_fps(max(cfg["screen_fps"], 4))
                if grab.pace(now, base, period, st.changed):
                    pacer.reset()
        finally:
            grab.close()
            st.set(capture=None, paused=False)
            if ears is not None:
                ears.close()

    # -- music mode ---------------------------------------------------------
    def _audio_loop(self, st, sender):
        cfg = st.snapshot()
        analyser = Analyser()
        mine = cfg["palettes"]
        color = ColorEngine(None, calm=cfg["calm"], fade=cfg["fade"],
                            palettes=all_palettes(cfg))
        lvl = LevelEngine(calm=cfg["calm"], punch=cfg["punch"])
        sync = BeatSync(timed=cfg["bulb_clock"])
        fx = DropFx()
        hold_until = 0.0                    # a bulb-timed beat's flow is running
        lamp = None                         # the frame the lamp was last left on
        mic = cfg["mic"]
        recorder = open_recorder(mic)
        st.set(device=str(getattr(recorder, "name", recorder))[:48])
        dt = analyser.dt
        watch = SignalWatch()

        last_sent = 0.0
        last_hsv = None
        st.set(status="running")
        # under Auto, glance at the screen - the one input this loop does not use
        picker, eyes = self.picker, None
        if picker is not None and picker.screen_ok:
            with contextlib.suppress(Exception):
                eyes = MotionProbe(Capture(cfg["monitor"], region_box(cfg),
                                           cfg["screen_width"], 1.0 / MotionMeter.PROBE,
                                           self.capture), picker)
                eyes.start()
                st.set(capture=eyes.grab.backend)

        try:
            with recorder.recorder(samplerate=SAMPLERATE, channels=2) as rec:
                while True:
                    cfg = st.snapshot()
                    if not cfg["running"] or self._mode(cfg) != "music" \
                            or cfg["mic"] != mic or sender.lost:
                        break

                    analyser.sensitivity = cfg["sensitivity"]
                    color.calm = lvl.calm = cfg["calm"]
                    color.fade = float(cfg["fade"])
                    lvl.punch = cfg["punch"]
                    if cfg["palettes"] is not mine:      # one added, edited or deleted
                        mine = cfg["palettes"]
                        apply_palette(color, cfg["palette"], all_palettes(cfg))
                    else:
                        apply_palette(color, cfg["palette"])

                    f = analyser.feed(rec.record(numframes=BLOCK))
                    sync.timed = cfg["bulb_clock"]
                    fired = sync.feed(f, lvl, dt, cfg["lead"] / 1000.0, color)
                    fx.feed(f)

                    color.maybe_switch(f["t"], f["bands"])
                    hue, sat = color.tick(dt, f["treble"], f["t"])

                    now = time.monotonic()
                    watch.feed(f["rms"], now)
                    if picker is not None:
                        picker.feed_sound(f["rms"], now)
                    # a predicted beat goes out on its block, not the next free
                    # one; a bulb-timed one's wait changes nothing on the LEDs,
                    # so its gap counts to the flash
                    if now - last_sent < (LED_GAP - sync.hold if fired
                                          else 1.0 / max(cfg["fps"], 5)):
                        continue

                    val = int(round(np.clip(
                        4 + 96 * (lvl.brightness ** cfg["gamma"]), 1, 100)))
                    hsv = (int(round(hue)) % 360, int(round(sat)), val)

                    # a build-up or a drop: one flow, then the loop again
                    effect = fx.take(now, hsv, cfg["drops"])
                    held = effect is not None or fx.holding(now)
                    # the page shows what the lamp shows: an effect's light
                    # while it runs, else this tick's
                    shown = fx.light if held else hsv
                    input_db, signal = watch.read(now)
                    st.set(hue=shown[0], sat=shown[1], val=shown[2], white=False, ct=0,
                           bpm=round(60.0 / max(f["beat"], 1e-3), 1),
                           level=round(lvl.brightness, 3),
                           hit=round(lvl.hit, 3),
                           bands=[round(float(b), 3) for b in f["bands"]],
                           input_db=input_db, signal=signal,
                           active_palette=color.pal)

                    if effect is not None:
                        sender.put(effect, True, 0)
                        last_sent, last_hsv, lamp = now, None, fx.end
                        continue
                    if held or now < hold_until:
                        continue
                    if fired and sync.hold and lamp is not None:
                        # hold what the lamp shows: this tick's frame may
                        # already carry the colour step meant for the flash
                        peak = (hsv[0], hsv[1], int(round(np.clip(
                            4 + 96 * sync.flash ** cfg["gamma"], 1, 100))))
                        sender.put(beat_flow(sync.hold, lamp, peak), True, 0)
                        hold_until = now + sync.hold + 0.05
                        fx.handback = False          # the flow was the hand-back
                        last_sent, last_hsv, lamp = now, peak, peak
                        continue

                    if not fired and same_light(hsv, last_hsv, hue_tol=2):
                        continue

                    fading = color.fade_t is not None
                    sender.put(hsv, sudden=lvl.hit > 0.35 and not fading and not fx.handback,
                               duration=fx.ramp(np.clip(1000.0 / max(cfg["fps"], 5), 30, 150)))
                    last_sent, last_hsv, lamp = now, hsv, hsv
        finally:
            if eyes is not None:
                eyes.close()
            st.set(bands=[0.0] * 10, input_db=None, signal=True, capture=None)

    # -- fusion mode: colour from the display, brightness from the audio ----
    def _fusion_loop(self, st, sender):
        cfg = st.snapshot()
        if screen_permission() is False:
            raise RuntimeError("no Screen Recording permission - grant it in "
                               "System Settings > Privacy & Security")

        monitor, region, mic = cfg["monitor"], cfg["region"], cfg["mic"]
        analyser = Analyser()
        lvl = LevelEngine(calm=cfg["calm"], punch=cfg["punch"])
        sync = BeatSync(timed=cfg["bulb_clock"])
        fx = DropFx()
        hold_until = 0.0
        lamp = None                         # the frame the lamp was last left on
        held_cut = False                    # a cut read while a flow held the loop
        color = ScreenColorEngine(min_sat=cfg["min_sat"],
                                  sat_boost=cfg["sat_boost"],
                                  gamma=cfg["screen_gamma"], calm=cfg["calm"])
        # the grab runs on its own thread: a ~26 ms capture inline would
        # starve rec.record() and drag the onsets behind the music
        recorder = open_recorder(mic)
        picker, meter = self.picker, MotionMeter()
        grab = Capture(monitor, region_box(cfg), cfg["screen_width"],
                       cfg["screen_fps"], self.capture)
        screen = ScreenColorThread(grab, color, fps=cfg["screen_fps"],
                                   on_frame=None if picker is None else
                                   lambda frame, now: picker.feed_motion(
                                       meter.feed(frame, now), now))
        watch = SignalWatch()
        pause = PauseLight(cfg["pause"])
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
                    if not cfg["running"] or self._mode(cfg) != "fusion" or sender.lost:
                        break
                    if cfg["monitor"] != monitor or cfg["region"] != region \
                            or cfg["mic"] != mic:
                        break                   # recapture on the new source
                    if screen.error:
                        raise RuntimeError(f"screen capture stopped: "
                                           f"{screen.error}")

                    analyser.sensitivity = cfg["sensitivity"]
                    lvl.calm = color.calm = cfg["calm"]
                    lvl.punch = cfg["punch"]
                    color.min_sat = float(cfg["min_sat"])
                    color.sat_boost = float(cfg["sat_boost"])
                    screen.fps = cfg["screen_fps"]
                    pause.delay = float(cfg["pause"])
                    if st.changed.is_set():     # a control moved: full rate
                        st.changed.clear()
                        screen.poke()

                    f = analyser.feed(rec.record(numframes=BLOCK))
                    # accents move brightness only
                    sync.timed = cfg["bulb_clock"]
                    fired = sync.feed(f, lvl, dt, cfg["lead"] / 1000.0)
                    fx.feed(f)

                    now = time.monotonic()
                    watch.feed(f["rms"], now)
                    if picker is not None:
                        picker.feed_sound(f["rms"], now)
                    if now - last_sent < (LED_GAP - sync.hold if fired
                                          else 1.0 / max(cfg["fps"], 5)):
                        continue

                    hue, sat, _val, cut = screen.read()
                    # reading takes the thread's sticky cut: one read while a
                    # flow holds the loop lands with the next send instead
                    cut, held_cut = cut or held_cut, False
                    val = int(round(np.clip(
                        4 + 96 * (lvl.brightness ** cfg["gamma"]), 1, 100)))
                    light = screen_frame(screen, hue, sat, val)
                    # still picture and no sound: a paused film, not an album cover
                    quiet = (lvl.silent_since is not None
                             and f["t"] - lvl.silent_since > PauseLight.QUIET)
                    if pause.feed(screen.moved_at(), now, quiet):
                        light, val = pause.frame(), PauseLight.LEVEL

                    effect = fx.take(now, light, cfg["drops"] and not pause.on)
                    held = effect is not None or fx.holding(now)
                    # the page shows what the lamp shows: an effect's light
                    # while it runs, else this tick's
                    cut_env = max(cut_env * 0.72, 1.0 if cut else 0.0)
                    input_db, signal = watch.read(now)
                    st.set(**(light_telemetry(fx.light, hue, fx.level) if held
                              else light_telemetry(light, hue, val)), paused=pause.on,
                           bpm=round(60.0 / max(f["beat"], 1e-3), 1),
                           capture=grab.backend,
                           level=round(lvl.brightness, 3),
                           hit=round(max(lvl.hit, cut_env), 3),
                           bands=[round(float(b), 3) for b in f["bands"]],
                           input_db=input_db, signal=signal)

                    if effect is not None:
                        sender.put(effect, True, 0)
                        last_sent, last_hsv, lamp, held_cut = now, None, fx.end, cut
                        continue
                    if held or now < hold_until:
                        held_cut = cut              # pause edges wait in PauseLight
                        continue
                    if (fired and sync.hold and lamp is not None and not pause.on
                            and not (pause.entered or pause.left)):
                        # hold what the lamp shows; the flash lands the new
                        # scene (a cut with it) 48-72 ms later, in 50 ms
                        peak = screen_frame(screen, hue, sat, int(round(np.clip(
                            4 + 96 * sync.flash ** cfg["gamma"], 1, 100))))
                        sender.put(beat_flow(sync.hold, lamp, peak), True, 0)
                        hold_until = now + sync.hold + 0.05
                        fx.handback = False          # the flow was the hand-back
                        last_sent, last_hsv, lamp = now, peak, peak
                        continue

                    if not (cut or fired or pause.entered or pause.left) \
                            and same_light(light, last_hsv):
                        continue

                    # a cut lands sudden even on the hand-back from an effect
                    sender.put(light, sudden=(cut or (lvl.hit > 0.35 and not fx.handback))
                               and not pause.on,
                               duration=fx.ramp(pause.ramp(light, last_hsv,
                                                           1.0 / max(cfg["fps"], 5))))
                    last_sent, last_hsv, lamp = now, light, light
        finally:
            screen.close()
            st.set(bands=[0.0] * 10, input_db=None, signal=True, capture=None, paused=False)


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
.st.quiet{color:var(--ink)} .st.quiet i{background:var(--warn)}
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
.seg button.off{opacity:.4}
.go{border:0;padding:8px 20px;border-radius:9px;background:var(--ink);color:var(--bg);font-weight:650;cursor:pointer}
.go[data-on="true"]{background:transparent;color:var(--ink);box-shadow:inset 0 0 0 1.5px var(--ink)}
.chk{display:flex;align-items:center;gap:8px;cursor:pointer}
.chk input{width:16px;height:16px;margin:0;accent-color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
/* presets: one click applies a look; Edit turns your own into rename fields */
.pchip{display:inline-flex;align-items:center;gap:2px;padding:6px 12px;border-radius:8px;border:0;cursor:pointer;
  background:var(--sf);box-shadow:inset 0 0 0 1px var(--line);font-size:13.5px}
.pchip:hover{box-shadow:inset 0 0 0 1px var(--dim)}
.pchip[aria-pressed="true"]{box-shadow:inset 0 0 0 1.5px var(--ink);font-weight:600}
.pchip.mine{padding:0 2px 0 0;cursor:default}
.pchip.mine button{border:0;background:none;padding:6px 4px 6px 12px;cursor:pointer;font-size:13.5px}
.pchip.mine button.x{color:var(--dim);padding:6px 8px 6px 4px;font-size:12px}
.pchip.edit{padding:3px;cursor:default}
.pchip.edit input{border:1px solid var(--ink);border-radius:6px;padding:3px 8px;width:130px;background:var(--bg);font-size:13.5px}
.pchip.edit input[aria-invalid="true"]{border-color:var(--err)}
.mini.ghost{box-shadow:none;color:var(--dim);padding-inline:6px}
.mini.ghost:hover{color:var(--ink);box-shadow:none}
.mini.ghost[aria-pressed="true"]{color:var(--ink);font-weight:600}
.chip{display:flex;align-items:center;gap:8px;padding:6px 11px 6px 7px;border-radius:8px;border:0;cursor:pointer;
  background:var(--sf);box-shadow:inset 0 0 0 1px var(--line);font-size:13.5px;text-transform:capitalize}
.chip s{display:block;width:30px;height:12px;border-radius:3px}
.chip:hover{box-shadow:inset 0 0 0 1px var(--dim)}
.chip[aria-pressed="true"]{box-shadow:inset 0 0 0 1.5px var(--ink);font-weight:600}
.chip.own,.pchip.mine.own{text-transform:none}
.pchip.mine s{display:inline-block;width:30px;height:12px;border-radius:3px;margin-right:8px;vertical-align:-1px}
.pchip.mine button.nm{padding-left:7px}

/* palette editor: the user's own palettes, typed or taken from a cover */
.pal{display:grid;gap:14px}
.ed{background:var(--sf);border-radius:10px;box-shadow:inset 0 0 0 1px var(--line);padding:14px;display:grid;gap:12px}
.ed .nm{display:flex;gap:10px;align-items:center}
.ed .nm img,.cno img{flex:none;width:44px;height:44px;border-radius:5px;object-fit:cover}
.sw{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.sw button{width:44px;height:44px;border-radius:8px;border:0;padding:0;cursor:pointer}
.sw button[aria-pressed="true"]{box-shadow:0 0 0 2px var(--sf),0 0 0 4px var(--ink)}
.sw .add{background:transparent;box-shadow:inset 0 0 0 1.5px var(--line);color:var(--dim);font-size:20px}
.sw .add:disabled{opacity:.4;cursor:default}
input[type=range].hue::-webkit-slider-runnable-track{height:8px;border-radius:4px;
  background:linear-gradient(90deg,#eb0000,#ebeb00,#00eb00,#00ebeb,#0000eb,#eb00eb,#eb0000)}
input[type=range].hue::-moz-range-track{height:8px;border-radius:4px;
  background:linear-gradient(90deg,#eb0000,#ebeb00,#00eb00,#00ebeb,#0000eb,#eb00eb,#eb0000)}
input[type=range].sat::-webkit-slider-runnable-track{height:8px;border-radius:4px;background:var(--st)}
input[type=range].sat::-moz-range-track{height:8px;border-radius:4px;background:var(--st)}
input[type=range].hue::-moz-range-progress,input[type=range].sat::-moz-range-progress{background:transparent}
input[type=range].hue::-webkit-slider-thumb,input[type=range].sat::-webkit-slider-thumb{margin-top:-4px;box-shadow:0 0 0 2px #fff}
.acts{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.link{border:0;background:none;color:var(--err);padding:8px 4px;cursor:pointer}
.cno{display:flex;align-items:center;gap:12px}
.cno p{margin:0;flex:1}
.side{position:relative}
.drop{position:absolute;inset:10px 14px 18px;border-radius:12px;border:2px dashed var(--ink);
  background:color-mix(in srgb,var(--bg) 94%,transparent);display:flex;align-items:center;justify-content:center;
  font-size:15px;font-weight:600;pointer-events:none;z-index:2}
.rows{display:grid;gap:4px}
.r{display:grid;grid-template-columns:150px 1fr 54px;gap:16px;align-items:center;min-height:34px}
.r output{text-align:right;color:var(--dim);font-size:13.5px}
.r input[type=checkbox]{width:16px;height:16px;margin:0;justify-self:start;accent-color:var(--ink);cursor:pointer}
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
#errnote{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
#errnote span{flex:1 1 260px}
.warn{color:var(--warn)} .err{color:var(--err)} .ok{color:var(--ok)}
.line{display:flex;gap:8px}
/* what the lamp does on its own: On stop looks, Sleep, the power-on look */
.pchip>s{display:block;width:30px;height:12px;border-radius:3px;margin-right:6px}
.chips>.lead{color:var(--dim);font-size:13.5px;align-self:center;margin-right:2px}
.lbl{display:grid;grid-template-columns:72px 1fr;gap:12px;align-items:center}
.lbl>span:first-child{color:var(--dim);font-size:13.5px}
#sleep{justify-self:start;flex-wrap:wrap}
#sleep button[aria-pressed="true"]{min-width:58px}
.confirm{background:var(--sf);border-radius:10px;box-shadow:inset 0 0 0 1px var(--line);padding:12px;display:grid;gap:10px}
.confirm p{margin:0}
.line>input{flex:1;min-width:0}
.btn{border:0;padding:8px 16px;border-radius:8px;background:var(--sf2);font-weight:550;white-space:nowrap;cursor:pointer}
.btn.pri{background:var(--ink);color:var(--bg)}
.btn:disabled{opacity:.4;cursor:default}
.mini{border:0;padding:4px 10px;border-radius:7px;background:transparent;box-shadow:inset 0 0 0 1px var(--line);
  font-size:13px;white-space:nowrap;cursor:pointer}
.mini:hover{box-shadow:inset 0 0 0 1px var(--dim)}
.mini:disabled{opacity:.5}

/* input level: raw RMS of the capture, -60..0 dBFS across the bar */
.meter{display:grid;grid-template-columns:1fr 64px;gap:16px;align-items:center}
.meter .bar{height:4px;border-radius:2px;background:var(--line);overflow:hidden}
.meter .bar i{display:block;height:100%;width:0;background:var(--ink)}
.meter output{text-align:right;color:var(--dim)}
.meter.none output{color:var(--warn);font-weight:600}

/* capture area: a still of the display, the box drawn over it */
.shot{position:relative;width:100%;aspect-ratio:16/10;border-radius:6px;overflow:hidden;background:var(--sf2);
  box-shadow:inset 0 0 0 1px var(--line);touch-action:none;cursor:crosshair;user-select:none;-webkit-user-select:none}
.shot img{position:absolute;inset:0;width:100%;height:100%;object-fit:fill;pointer-events:none}
.shot .shade{position:absolute;inset:0;background:rgba(0,0,0,.45);pointer-events:none}
.shot .box{position:absolute;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.4);cursor:move}
.shot .box i{position:absolute;width:11px;height:11px;background:#fff;border-radius:2px;box-shadow:0 0 0 1px rgba(0,0,0,.4)}
.shot .box i[data-h="nw"]{left:-6px;top:-6px;cursor:nwse-resize} .shot .box i[data-h="ne"]{right:-6px;top:-6px;cursor:nesw-resize}
.shot .box i[data-h="sw"]{left:-6px;bottom:-6px;cursor:nesw-resize} .shot .box i[data-h="se"]{right:-6px;bottom:-6px;cursor:nwse-resize}
.rline{display:flex;align-items:center;gap:10px;color:var(--dim);min-height:26px}

/* one desk, three modes - show only the controls that apply. Fusion draws
   from both sides, so it hides .no-fusion instead: the palette, the colour
   fade and the dark lift have nothing to act on when the display owns the
   hue and the audio owns the brightness. */
body[data-mode="music"] .only-screen,
body[data-mode="screen"] .only-music,
body[data-mode="fusion"] .no-fusion{display:none !important}
body[data-deck="runtime"] .deck-cloud,
body[data-deck="cloud"] .deck-runtime{display:none !important}

/* lamp mode: the bulb as a plain lamp while no session runs */
.lampctl{display:grid;gap:10px}
.tog{display:inline-flex;align-items:center;gap:10px;border:0;background:none;padding:2px 0;cursor:pointer;font-weight:600}
.tog i{width:34px;height:20px;border-radius:10px;background:var(--line);position:relative;flex:none}
.tog i::after{content:"";position:absolute;top:3px;left:3px;width:14px;height:14px;border-radius:50%;background:var(--sf)}
.tog[aria-checked="true"] i{background:var(--ink)}
.tog[aria-checked="true"] i::after{left:17px;background:var(--bg)}
.tog:disabled{opacity:.4;cursor:default}
.lampctl input[type=range]:disabled{opacity:.4;cursor:default}
input[type=range].kel::-webkit-slider-runnable-track{height:6px;border-radius:3px;background:linear-gradient(90deg,#ffa757,#ffc794,#ffdfc2,#fff1e7,#fef9ff)}
input[type=range].kel::-moz-range-track{height:6px;border-radius:3px;background:linear-gradient(90deg,#ffa757,#ffc794,#ffdfc2,#fff1e7,#fef9ff)}
input[type=range].kel::-moz-range-progress{background:transparent}
input[type=range].kel::-webkit-slider-thumb{margin-top:-5px;box-shadow:0 0 0 2px var(--sf)}
.strip{position:relative;height:34px;border-radius:6px;overflow:hidden;box-shadow:inset 0 0 0 1px var(--line);
  touch-action:none;cursor:ew-resize;user-select:none;-webkit-user-select:none}
.strip canvas{position:absolute;inset:0;width:100%;height:100%}
/* the strip is always pale, so its marks are dark in both themes */
.strip .now{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:#17181a;pointer-events:none}
.strip button{position:absolute;top:50%;width:12px;height:22px;margin:-11px 0 0 -6px;padding:0;border:0;border-radius:3px;
  background:#fff;box-shadow:0 0 0 1.5px #17181a;cursor:ew-resize}
.ticks{display:flex;justify-content:space-between;color:var(--dim);font-size:12px;margin-top:-4px}
.lro{color:#111}

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

/* ---------- signals: alert, Focus, sunrise ---------- */
.sig{display:grid;gap:10px;padding-top:16px;border-top:1px solid var(--line)}
.sig .sh{display:flex;align-items:center;gap:10px;font-weight:620}
.sig .ln{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.sig .sw button{width:28px;height:28px;border-radius:7px}
.sig .sw button[aria-pressed="true"]{box-shadow:0 0 0 2px var(--bg),0 0 0 3.5px var(--ink)}
.sig .sw button.pale{box-shadow:inset 0 0 0 1px var(--line)}
.sig .sw button.pale[aria-pressed="true"]{box-shadow:inset 0 0 0 1px var(--line),0 0 0 2px var(--bg),0 0 0 3.5px var(--ink)}
.sig .seg button{padding:5px 12px}
.clock{font-size:26px;font-weight:620;letter-spacing:-.01em}
.phase{color:var(--dim);font-weight:400;font-size:13.5px}
input.tm{width:auto;min-width:110px}
.tog{position:relative;flex:none;width:36px;height:21px;border-radius:11px;border:0;padding:0;background:var(--line);cursor:pointer}
.tog::after{content:"";position:absolute;left:2.5px;top:2.5px;width:16px;height:16px;border-radius:50%;background:var(--sf);transition:left .15s}
.tog[aria-checked="true"]{background:var(--ink)}
.tog[aria-checked="true"]::after{left:17.5px}
@media (prefers-reduced-motion:reduce){.tog::after{transition:none}}
.ramp{flex:1;min-width:60px;height:8px;border-radius:4px;background:linear-gradient(90deg,#3a1d08,#ff8a2a 30%,#ffd9a8 65%,#f2f6ff)}
.alm{flex:none;display:inline-flex;align-items:center;gap:6px;font-size:13.5px;font-weight:600;border:0;border-radius:7px;
  padding:5px 10px;background:transparent;box-shadow:inset 0 0 0 1.5px var(--ink);cursor:pointer}
.alm svg{width:15px;height:15px}
body[data-deck="cloud"] #alm{display:none}
</style></head><body data-mode="music" data-deck="runtime">

<div class="app">
  <div class="top">
    <span class="name">bulbd</span>
    <span class="st" id="status"><i></i><span id="stxt">idle</span></span>
    <button class="alm" id="alm" type="button" hidden aria-label="Sunrise alarm"><svg viewBox="0 0 16 16" fill="none"
      stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M2 12.5h12M4.5 12.5a3.5 3.5 0 0 1 7 0M8 3.5v2M3.3 6.6l1.3 1.2M12.7 6.6l-1.3 1.2"/></svg><span id="almt"></span></button>
    <span class="sp"></span>
    <button class="btn" id="sstop" type="button" hidden>Stop</button>
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
      <div class="ro lro" id="lro" hidden></div>
    </div>
    <canvas id="tape" hidden></canvas>
    <div class="spec" id="spec"></div>
  </div>

  <div class="side">
    <!-- ============================ RUNTIME ============================ -->
    <p class="note deck-runtime" id="errnote" hidden><span id="errtxt"></span>
      <button class="btn" id="errfix" type="button" hidden>Open Setup</button></p>
    <div class="lampctl deck-runtime" id="lampctl" hidden>
      <div class="transport">
        <button class="tog" id="lpower" type="button" role="switch" aria-checked="false"><i></i>Lamp</button>
        <span class="sp"></span>
        <label class="chk"><input type="checkbox" id="follow">Follow the day</label>
      </div>
      <div id="day" hidden>
        <div class="strip" id="strip"><canvas id="stripc"></canvas><span class="now" id="snow"></span>
          <button type="button" id="hwake" role="slider" aria-label="Morning" aria-valuemin="0" aria-valuemax="1425"></button>
          <button type="button" id="hdusk" role="slider" aria-label="Evening" aria-valuemin="0" aria-valuemax="1425"></button></div>
        <div class="ticks"><span>00</span><span>06</span><span>12</span><span>18</span><span>24</span></div>
      </div>
      <div class="rows">
        <div class="r" id="ctrow"><label for="lct">Temperature</label>
          <input type="range" class="kel" id="lct" min="2700" max="6500" step="50" value="4000"><output id="v-lct"></output></div>
        <div class="r"><label for="lbr">Brightness</label>
          <input type="range" id="lbr" min="1" max="100" step="1" value="50"><output id="v-lbr"></output></div>
      </div>
    </div>
    <div class="transport deck-runtime">
      <span class="seg" id="modes">
        <button type="button" data-mode="music" aria-pressed="true">Music</button>
        <button type="button" data-mode="screen" aria-pressed="false">Screen</button>
        <button type="button" data-mode="fusion" aria-pressed="false">Fusion</button>
      </span>
      <span class="sp"></span>
      <label class="chk"><input type="checkbox" id="auto">Auto</label>
      <label class="chk"><input type="checkbox" id="calm">Calm</label>
      <button class="go" id="go" type="button" data-on="false">Start</button>
    </div>

    <div class="chips deck-runtime" id="presets"></div>
    <div class="chips deck-runtime" id="idle"></div>
    <div class="lbl deck-runtime"><span id="sleeplabel">Sleep</span>
      <span class="seg" id="sleep" role="group" aria-labelledby="sleeplabel"></span></div>
    <p class="note err deck-runtime" id="sleeperr"></p>

    <div class="pal deck-runtime only-music no-fusion">
      <div class="chips" id="chips"></div>
      <div class="ed" id="ped" hidden>
        <div class="nm"><img id="pthumb" alt="" hidden>
          <input class="tx" id="pname" autocomplete="off" spellcheck="false" placeholder="Name" aria-label="Palette name"></div>
        <div class="sw" id="pswatches"></div>
        <p class="note err" id="perr"></p>
        <div class="rows">
          <div class="r"><label for="phue">Hue</label>
            <input type="range" class="hue" id="phue" min="0" max="359" step="1"><output id="v-phue"></output></div>
          <div class="r"><label for="psat">Saturation</label>
            <input type="range" class="sat" id="psat" min="45" max="100" step="1"><output id="v-psat"></output></div>
        </div>
        <div class="acts">
          <button class="mini" id="pleft" type="button" aria-label="Move left">&larr;</button>
          <button class="mini" id="pright" type="button" aria-label="Move right">&rarr;</button>
          <button class="mini" id="prm" type="button">Remove</button>
          <span class="sp"></span>
          <button class="link" id="pdel" type="button">Delete</button>
          <button class="btn" id="pcancel" type="button">Cancel</button>
          <button class="btn pri" id="psave" type="button">Save</button>
        </div>
      </div>
      <div class="in" id="cpick" hidden>
        <select id="cmon" aria-label="Display"></select>
        <div class="shot" id="cshot" tabindex="0" role="group" aria-label="Box around the cover">
          <img id="cimg" alt="" hidden>
          <div class="shade" id="cshade" hidden></div>
          <div class="box" id="cbox" hidden><i data-h="nw"></i><i data-h="ne"></i><i data-h="sw"></i><i data-h="se"></i></div>
        </div>
        <div class="rline"><span class="sp"></span>
          <button class="btn" id="cstop" type="button">Cancel</button>
          <button class="btn pri" id="ctake" type="button" disabled>Take colours</button></div>
      </div>
      <div class="cno" id="pno" hidden><img id="pnothumb" alt="">
        <p class="note warn">No colour in this cover to make a palette from.</p>
        <button class="btn" id="pnook" type="button">OK</button></div>
      <input type="file" id="pfile" accept="image/*" hidden>
    </div>
    <div class="drop" id="drop" hidden><span>Drop the cover</span></div>

    <div class="in deck-runtime only-screen">
      <label for="monitor">Display</label>
      <select id="monitor"></select>
      <p class="note warn" id="permnote"></p>
    </div>

    <div class="in deck-runtime only-screen">
      <span id="arealabel">Area</span>
      <div class="shot" id="shot" tabindex="0" role="group" aria-labelledby="arealabel" aria-describedby="areatxt">
        <img id="shotimg" alt="" hidden>
        <div class="shade" id="shade" hidden></div>
        <div class="box" id="box" hidden><i data-h="nw"></i><i data-h="ne"></i><i data-h="sw"></i><i data-h="se"></i></div>
      </div>
      <div class="rline"><span id="areatxt">Whole display</span><span class="sp"></span>
        <button class="mini" id="areaall" type="button" hidden>Whole display</button></div>
    </div>

    <div class="in deck-runtime only-music">
      <label for="mic">Input</label>
      <select id="mic"><option value="">Automatic</option></select>
      <div class="meter" id="meter" hidden><span class="bar"><i id="meterbar"></i></span><output id="meterdb"></output></div>
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
      <div class="r only-music"><label for="lead">Beat sync</label>
        <input type="range" id="lead" min="0" max="250" step="5" value="80"><output id="v-lead"></output></div>
      <div class="r only-music"><label for="bclock">Bulb clock</label>
        <input type="checkbox" id="bclock"><output></output></div>
      <div class="r only-music"><label for="drops">Drops</label>
        <input type="checkbox" id="drops"><output></output></div>
      <div class="r only-screen no-fusion"><label for="sgamma">Dark lift</label>
        <input type="range" id="sgamma" min="0.4" max="1.6" step="0.05" value="0.75"><output id="v-sgamma"></output></div>
      <div class="r only-screen"><label for="minsat">Colour floor</label>
        <input type="range" id="minsat" min="0" max="90" step="1" value="55"><output id="v-minsat"></output></div>
      <div class="r only-screen"><label for="sboost">Saturation boost</label>
        <input type="range" id="sboost" min="0.8" max="2" step="0.05" value="1.35"><output id="v-sboost"></output></div>
      <div class="r only-screen"><label for="sfps">Capture rate</label>
        <input type="range" id="sfps" min="4" max="30" step="1" value="10"><output id="v-sfps"></output></div>
      <div class="r only-screen"><label for="pause">Pause light</label>
        <input type="range" id="pause" min="0" max="120" step="1" value="0"><output id="v-pause"></output></div>
    </div>

    <!-- ============================= SIGNALS ============================= -->
    <div class="sig deck-runtime" id="alert">
      <div class="sh">Alert<span class="sp"></span><button class="btn" id="apulse" type="button">Pulse</button></div>
      <div class="sw" id="aswatch" role="group" aria-label="Alert colour"></div>
      <div class="ln">
        <span class="seg" id="apulses" role="group" aria-label="Pulses"></span>
        <span class="sp"></span>
        <span class="seg" id="aspeed" role="group" aria-label="Speed"></span>
      </div>
      <p class="note err" id="aerr"></p>
    </div>

    <div class="sig deck-runtime" id="focus">
      <div class="sh">Focus<span class="sp"></span><span class="phase" id="fphase"></span></div>
      <div class="ln"><span class="clock" id="fclock">25:00</span><span class="sp"></span>
        <button class="btn" id="fgo" type="button">Start</button>
        <button class="btn" id="freset" type="button" disabled>Reset</button></div>
      <div class="rows">
        <div class="r"><label for="fwork">Work</label>
          <input type="range" id="fwork" min="5" max="90" step="5" value="25"><output id="v-fwork"></output></div>
        <div class="r"><label for="fbreak">Break</label>
          <input type="range" id="fbreak" min="1" max="30" step="1" value="5"><output id="v-fbreak"></output></div>
      </div>
      <p class="note err" id="ferr"></p>
    </div>

    <div class="sig deck-runtime" id="sunrise">
      <div class="sh">Sunrise<span class="sp"></span>
        <button class="tog" id="son" type="button" role="switch" aria-checked="false" aria-label="Sunrise"></button></div>
      <div class="ln"><input class="tx tm" type="time" id="sat" value="07:00" aria-label="Sunrise time"><span class="ramp"></span></div>
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
        <div class="line"><span class="sp"></span><span class="st on" id="defok" hidden><i></i>Saved</span>
          <button class="btn" id="defbtn" type="button">Use as power-on look</button></div>
        <div class="confirm" id="defask" hidden>
          <p>The lamp will come on like this from the wall switch. The current power-on look cannot be brought back.</p>
          <p class="note err" id="deferr"></p>
          <div class="line"><span class="sp"></span>
            <button class="btn" id="defno" type="button">Cancel</button>
            <button class="btn pri" id="defyes" type="button">Save</button></div>
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
const PR = __PAL_RULES__;   // what palettes.py accepts: name length, anchors, saturation
const BUILTIN = __PRESETS__;
const $ = id => document.getElementById(id);
let holdUntil = 0;   // don't let polling stomp a fader you're dragging
let draft=null, drag=null, shotFor=null, shotStale=true, keyTimer=0;   // capture area

// a white frame's colour: Tanner Helland's fit of the black-body curve,
// scaled by the level like hsv2rgb's value
function kelvin2rgb(k,v){
  k/=100; let r,g,b;
  if(k<=66){ r=255; g=99.47*Math.log(k)-161.12; b=k<=19?0:138.52*Math.log(k-10)-305.04; }
  else{ r=329.7*Math.pow(k-60,-0.1332); g=288.12*Math.pow(k-60,-0.0755); b=255; }
  return [r,g,b].map(c=>Math.round(Math.max(0,Math.min(255,c))*v/100));
}
function hsv2rgb(h,s,v){
  h=((h%360)+360)%360; s/=100; v/=100;
  const c=v*s, x=c*(1-Math.abs((h/60)%2-1)), m=v-c;
  let r,g,b;
  if(h<60)[r,g,b]=[c,x,0]; else if(h<120)[r,g,b]=[x,c,0];
  else if(h<180)[r,g,b]=[0,c,x]; else if(h<240)[r,g,b]=[0,x,c];
  else if(h<300)[r,g,b]=[x,0,c]; else [r,g,b]=[c,0,x];
  return [(r+m)*255,(g+m)*255,(b+m)*255].map(Math.round);
}

const chips = $('chips');

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
              lead:'lead', fade:'fade',
              sgamma:'screen_gamma', minsat:'min_sat', sboost:'sat_boost',
              sfps:'screen_fps', pause:'pause', fwork:'focus_work', fbreak:'focus_break'};
function fmt(id, v){
  v=+v;
  if(id==='fps' || id==='sfps') return v+'/s';
  if(id==='fade') return v ? v.toFixed(1)+' s' : 'Off';
  if(id==='lead') return v ? v+' ms' : 'Off';
  if(id==='pause') return v ? v+' s' : 'Off';
  if(id==='minsat') return String(v);
  if(id==='fwork' || id==='fbreak') return v+' min';
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

function setStatus(text, notice){
  const st=$('status');
  // a failure arrives with a paragraph of advice: the bar says one word and
  // the advice goes in a note, where it has room to wrap. Link's reasons
  // come as a notice; anything else that failed as "error: <detail>".
  const m = /^error:\s*([\s\S]*)$/.exec(text);
  $('stxt').textContent = m ? 'error' : text.replace('on the cloud deck','in Setup');
  const n = notice || (m ? {kind:'err', text:m[1].charAt(0).toUpperCase()+m[1].slice(1)} : null);
  $('errnote').hidden = !n;
  $('errnote').className = 'note deck-runtime ' + (n ? n.kind : '');
  $('errtxt').textContent = n ? n.text : '';
  $('errfix').hidden = !(n && n.action==='setup');
  st.className = 'st' + (text==='running' || text==='connected' || text==='sunrise' ? ' on'
    : text==='no signal' ? ' quiet'
    : /^(connecting|starting|stopping|restoring|reconnecting|searching)/.test(text) ? ' busy'
    : /^(error|no bulb|server|unreachable)/.test(text) ? ' bad' : '');
}
$('errfix').addEventListener('click',()=>{ location.hash='cloud'; });

// the mode that is running: Auto's pick while Auto is on
function modeOf(s){ return s.auto ? (s.auto_mode || (s.screen_ok ? 'fusion' : 'music')) : s.mode; }
[...$('modes').children].forEach(b=>{
  b.onclick=()=>{
    const m=b.dataset.mode;
    [...$('modes').children].forEach(c=>c.setAttribute('aria-pressed', c===b));
    if(m!=='music' && document.body.dataset.mode==='music') shotStale = true;
    document.body.dataset.mode = m;
    $('auto').checked = false;           // a mode picked by hand turns Auto off
    send({mode:m, auto:false});
  };
});
function showDeck(name){
  const deck = name==='cloud' ? 'cloud' : 'runtime';
  document.body.dataset.deck = deck;
  if(deck==='cloud' && window.cloudUp) cloudPoll();   // not before its state exists
  if(deck==='runtime') shotStale = true;
}
$('addr').onclick=()=>{ location.hash='cloud'; };
$('done').onclick=()=>{ location.hash='runtime'; };
// #cloud in the URL opens straight onto Setup, so a link can point at it
showDeck(location.hash.slice(1));
addEventListener('hashchange', ()=>showDeck(location.hash.slice(1)));

$('monitor').addEventListener('change',e=>send({monitor:+e.target.value}));
$('mic').addEventListener('change',e=>send({mic:e.target.value || null}));
$('calm').addEventListener('change',e=>send({calm:e.target.checked}));
$('bclock').addEventListener('change',e=>send({bulb_clock:e.target.checked}));
$('drops').addEventListener('change',e=>send({drops:e.target.checked}));
$('auto').addEventListener('change',e=>send({auto:e.target.checked}));
$('go').addEventListener('click',()=>{
  const on = $('go').dataset.on === 'true';
  send({running: !on});
});
$('ipset').addEventListener('click',()=>send({ip:$('ip').value.trim()}));
$('ip').addEventListener('keydown',e=>{ if(e.key==='Enter') send({ip:$('ip').value.trim()}); });

/* -------------------------------- presets -------------------------------- */
// Built-ins and the user's own, as chips. The pressed chip is the preset the
// desk's look matches now; move a fader and none is. Edit turns the user's
// chips into a name to rename and a mark to delete.
let pEdit=false, pRename=null, pNaming=false, pKey='';
function same(look, s){
  return Object.keys(look).every(k => typeof look[k]==='number'
    ? Math.abs(look[k]-s[k]) < 1e-6 : look[k]===s[k]);
}
async function presetOp(body, input){
  const r = await post('/api/presets', body);
  if(!r.ok && input){ input.setAttribute('aria-invalid','true'); input.title = r.error||''; input.focus(); return false; }
  pKey=''; poll();
  return r.ok;
}
function nameField(value, commit){
  const wrap=document.createElement('span'); wrap.className='pchip edit';
  const i=document.createElement('input'); i.className='tx'; i.value=value; i.maxLength=32;
  i.setAttribute('aria-label', value ? 'New name for '+value : 'Name for this look');
  i.placeholder='Name';
  let done=false;
  const finish=async ok=>{
    if(done) return;
    const v=i.value.trim();
    if(ok && v && v!==value){ done = await commit(v, i); if(!done) return; }
    done=true; pRename=null; pNaming=false; pKey=''; i.blur(); renderPresets(live);
  };
  i.addEventListener('keydown',e=>{
    i.removeAttribute('aria-invalid');
    if(e.key==='Enter'){ e.preventDefault(); finish(true); }
    if(e.key==='Escape'){ e.preventDefault(); finish(false); }
  });
  i.addEventListener('blur',()=>{ if(!i.value.trim() || i.value.trim()===value) finish(false); });
  wrap.append(i);
  setTimeout(()=>{ i.focus(); i.select(); });
  return wrap;
}
function renderPresets(s){
  if(!s) return;
  const row=$('presets');
  // never rebuild under a name being typed
  if(row.contains(document.activeElement) && document.activeElement.tagName==='INPUT') return;
  const mine=(s.presets||[]);
  const all=BUILTIN.map(p=>[p,false]).concat(mine.map(p=>[p,true]));
  const on=(all.find(([p])=>same(p.look, s))||[{}])[0].name;
  const key=JSON.stringify([mine, on, pEdit, pRename, pNaming]);
  if(key===pKey) return;
  pKey=key; row.innerHTML='';
  if(pEdit && !mine.length) pEdit=false;
  all.forEach(([p, isMine])=>{
    if(pEdit && isMine){
      if(p.name===pRename){ row.append(nameField(p.name,(v,i)=>presetOp({op:'rename',name:p.name,to:v},i))); return; }
      const chip=document.createElement('span'); chip.className='pchip mine';
      const nm=document.createElement('button'); nm.type='button'; nm.textContent=p.name;
      nm.setAttribute('aria-label','Rename '+p.name);
      nm.onclick=()=>{ pRename=p.name; pKey=''; renderPresets(live); };
      const x=document.createElement('button'); x.type='button'; x.className='x'; x.textContent='\u2715';
      x.setAttribute('aria-label','Delete '+p.name);
      x.onclick=()=>presetOp({op:'delete',name:p.name});
      chip.append(nm,x); row.append(chip); return;
    }
    const b=document.createElement('button'); b.type='button'; b.className='pchip';
    b.textContent=p.name; b.setAttribute('aria-pressed', p.name===on);
    b.onclick=()=>{ [...row.querySelectorAll('.pchip')].forEach(c=>c.setAttribute('aria-pressed', c===b));
                    send({preset:p.name}); };
    row.append(b);
  });
  if(pNaming) row.append(nameField('',(v,i)=>presetOp({op:'save',name:v},i)));
  const mini=(text,fn)=>{ const b=document.createElement('button'); b.type='button';
    b.className='mini ghost'; b.textContent=text; b.onclick=fn; row.append(b); };
  if(pEdit) mini('Done',()=>{ pEdit=false; pRename=null; pKey=''; renderPresets(live); });
  else{
    if(!pNaming) mini('Save as preset',()=>{ pNaming=true; pKey=''; renderPresets(live); });
    if(mine.length) mini('Edit',()=>{ pEdit=true; pNaming=false; pKey=''; renderPresets(live); });
  }
}

/* -------------------------------- palettes -------------------------------- */
// The built-ins and the user's own, as chips. Edit turns the user's into a
// name to open in the editor and a mark to delete. The editor works on a
// copy (ped) and writes it with Save; a cover - dropped, pasted, picked from
// a file or boxed on a still of the screen - opens it with the colours
// already taken, and nothing changes on the lamp until Save.
let pEditMode=false, ped=null, palKey='';
const own = s => (s && s.palettes) || [];
const swatch = a => `rgb(${hsv2rgb(a[0],a[1],92)})`;
const strip = anchors => 'linear-gradient(90deg,'+anchors.map(swatch).join(',')+')';
function palNames(s){ return Object.keys(P).concat(own(s).map(p=>p.name)); }
function renderPalettes(s){
  if(!s) return;
  const row=chips, mine=own(s);
  const key=JSON.stringify([mine, s.palette, pEditMode, !!ped, !$('cpick').hidden, s.snapshot_ok]);
  if(key===palKey) return;
  palKey=key; row.innerHTML='';
  if(pEditMode && !mine.length) pEditMode=false;
  const list=[['auto',null,false]].concat(Object.keys(P).map(k=>[k,P[k],false]), mine.map(p=>[p.name,p.anchors,true]));
  list.forEach(([name,anchors,isMine])=>{
    const sw=document.createElement('s');
    sw.style.background = anchors ? strip(anchors)
      : 'repeating-linear-gradient(90deg,var(--dim) 0 2px,transparent 2px 5px)';
    if(pEditMode && isMine){
      const chip=document.createElement('span'); chip.className='pchip mine own';
      const nm=document.createElement('button'); nm.type='button'; nm.className='nm';
      nm.append(sw, document.createTextNode(name)); nm.setAttribute('aria-label','Edit '+name);
      nm.onclick=()=>openEditor({name, anchors});
      const x=document.createElement('button'); x.type='button'; x.className='x'; x.textContent='✕';
      x.setAttribute('aria-label','Delete '+name);
      x.onclick=()=>palOp({op:'delete', name});
      chip.append(nm,x); row.append(chip); return;
    }
    const b=document.createElement('button');
    b.className='chip'+(isMine?' own':''); b.type='button'; b.dataset.name=name;
    b.setAttribute('aria-pressed', name===s.palette);
    b.append(sw, document.createTextNode(name));
    b.onclick=()=>{ [...row.querySelectorAll('.chip')].forEach(c=>c.setAttribute('aria-pressed', c===b));
                    send({palette:name}); };
    row.append(b);
  });
  const mini=(text,fn,pressed)=>{ const b=document.createElement('button'); b.type='button';
    b.className='mini ghost'; b.textContent=text; b.onclick=fn;
    if(pressed!==undefined) b.setAttribute('aria-pressed', pressed); row.append(b); };
  if(pEditMode){ mini('Done',()=>{ pEditMode=false; palKey=''; renderPalettes(live); }); return; }
  if(!ped){
    mini('New palette',()=>{
      const cur=P[s.palette] || (mine.find(p=>p.name===s.palette)||{}).anchors || P[Object.keys(P)[0]];
      openEditor(null, {anchors:cur});
    });
    mini('From a cover',()=>$('pfile').click());
    if(s.snapshot_ok) mini('From the screen',()=>$('cpick').hidden ? openPick() : closePick(), !$('cpick').hidden);
  }
  if(mine.length && !ped && $('cpick').hidden) mini('Edit',()=>{ pEditMode=true; closePick(); palKey=''; renderPalettes(live); });
}
// -- the lamp on its own: On stop, Sleep ---------------------------------------
const IDLE = [['restore','Restore',null],
  ['candle','Candle','linear-gradient(90deg,#ff8a2a,#ffc56b,#ff9a3c)'],
  ['drift','Drift',''],
  ['breathe','Breathe','linear-gradient(90deg,#fff1dc,#6d685f,#fff1dc)']];
const SLEEP_STEPS = __SLEEP_STEPS__;
let idleKey='', sleepKey='';
function driftAnchors(s){
  return P[s.palette] || (own(s).find(p=>p.name===s.palette)||{}).anchors
    || P[s.active_palette] || (own(s).find(p=>p.name===s.active_palette)||{}).anchors || P.ember;
}
function renderIdle(s){
  const row=$('idle'), anchors=driftAnchors(s);
  const key=JSON.stringify([s.idle, anchors]);
  if(key===idleKey) return;
  idleKey=key; row.innerHTML='';
  const lead=document.createElement('span'); lead.className='lead'; lead.textContent='On stop'; row.append(lead);
  IDLE.forEach(([id,label,bg])=>{
    const b=document.createElement('button'); b.type='button'; b.className='pchip';
    b.setAttribute('aria-pressed', s.idle===id);
    if(bg!==null){ const sw=document.createElement('s'); sw.style.background = bg || strip(anchors); b.append(sw); }
    b.append(document.createTextNode(label));
    b.onclick=()=>{ [...row.querySelectorAll('.pchip')].forEach(c=>c.setAttribute('aria-pressed', c===b));
                    send({idle:id}); };
    row.append(b);
  });
}
function clock(sec){
  sec=Math.max(0, Math.ceil(sec));
  const h=Math.floor(sec/3600), m=Math.floor(sec%3600/60), x=String(sec%60).padStart(2,'0');
  return h ? h+':'+String(m).padStart(2,'0')+':'+x : m+':'+x;
}
function renderSleep(s){
  // the bulb counts; the page only shows what is left of `at`
  const left = s.sleep ? s.sleep.at - Date.now()/1000 : 0;
  const on = left > 0 ? (SLEEP_STEPS.find(m=>m*60 >= left-1) || SLEEP_STEPS[SLEEP_STEPS.length-1]) : 0;
  const key=JSON.stringify([on, on ? clock(left) : '']);
  if(key===sleepKey) return;
  sleepKey=key;
  const row=$('sleep');
  if(!row.children.length){
    [0].concat(SLEEP_STEPS).forEach(m=>{
      const b=document.createElement('button'); b.type='button'; b.dataset.min=m;
      b.onclick=async ()=>{
        $('sleeperr').textContent='';
        const r=await post('/api/sleep', {minutes:m});
        if(!r.ok) $('sleeperr').textContent=r.error||'The bulb did not answer.';
        sleepKey=''; poll();
      };
      row.append(b);
    });
  }
  [...row.children].forEach(b=>{
    const m=+b.dataset.min;
    b.setAttribute('aria-pressed', m===on);
    b.textContent = m===0 ? 'Off' : (m===on ? clock(left) : String(m));
    b.setAttribute('aria-label', m===0 ? 'Sleep off' : (m===on ? clock(left)+' left' : m+' minutes'));
  });
}

async function palOp(body){
  const r = await post('/api/palettes', body);
  palKey=''; poll();
  return r;
}

// -- the editor --------------------------------------------------------------
function openEditor(p, from){
  closePick(); $('pno').hidden=true; pEditMode=false;
  ped = {orig: p ? p.name : null, anchors: (p ? p.anchors : from.anchors).map(a=>[a[0],a[1]]), sel:0};
  $('pname').value = p ? p.name : (from.name || '');
  $('pname').maxLength = PR.name;
  $('pthumb').hidden = !(from && from.thumb);
  if(from && from.thumb) $('pthumb').src = from.thumb;
  $('perr').textContent=''; $('pdel').hidden = !ped.orig;
  $('ped').hidden=false;
  paintEditor(); palKey=''; renderPalettes(live);
  if(!p) $('pname').focus();
}
function closeEditor(){ ped=null; $('ped').hidden=true; palKey=''; renderPalettes(live); }
function paintEditor(){
  const e=ped, a=e.anchors[e.sel], wrap=$('pswatches');
  wrap.innerHTML='';
  e.anchors.forEach((c,i)=>{
    const b=document.createElement('button'); b.type='button';
    b.style.background=swatch(c); b.setAttribute('aria-pressed', i===e.sel);
    b.setAttribute('aria-label', `Colour ${i+1}, hue ${c[0]}, saturation ${c[1]}`);
    b.onclick=()=>{ e.sel=i; paintEditor(); };
    wrap.append(b);
  });
  const add=document.createElement('button'); add.type='button'; add.className='add'; add.textContent='+';
  add.setAttribute('aria-label','Add a colour'); add.disabled = e.anchors.length>=PR.max;
  add.onclick=()=>{ const c=e.anchors[e.sel]; e.anchors.splice(e.sel+1,0,[(c[0]+40)%360,c[1]]); e.sel++; paintEditor(); };
  wrap.append(add);
  $('phue').value=a[0]; $('psat').value=a[1];
  $('v-phue').textContent=a[0]+'°'; $('v-psat').textContent=a[1];
  $('psat').style.setProperty('--st', `linear-gradient(90deg,${swatch([a[0],PR.sat[0]])},${swatch([a[0],PR.sat[1]])})`);
  $('prm').disabled = e.anchors.length<=PR.min;
  $('pleft').disabled = e.sel===0; $('pright').disabled = e.sel===e.anchors.length-1;
}
function nudge(){
  const a=ped.anchors[ped.sel];
  a[0]=+$('phue').value; a[1]=+$('psat').value;
  $('perr').textContent='';
  paintEditor();
}
$('phue').addEventListener('input', nudge);
$('psat').addEventListener('input', nudge);
$('pleft').onclick=()=>{ const e=ped, i=e.sel; [e.anchors[i-1],e.anchors[i]]=[e.anchors[i],e.anchors[i-1]]; e.sel--; paintEditor(); };
$('pright').onclick=()=>{ const e=ped, i=e.sel; [e.anchors[i+1],e.anchors[i]]=[e.anchors[i],e.anchors[i+1]]; e.sel++; paintEditor(); };
$('prm').onclick=()=>{ const e=ped; e.anchors.splice(e.sel,1); e.sel=Math.min(e.sel,e.anchors.length-1); paintEditor(); };
$('pcancel').onclick=closeEditor;
$('pdel').onclick=async ()=>{ const r=await palOp({op:'delete', name:ped.orig}); if(r.ok) closeEditor(); else $('perr').textContent=r.error; };
$('pname').addEventListener('keydown', e=>{ if(e.key==='Enter'){ e.preventDefault(); $('psave').click(); } });
$('psave').onclick=async ()=>{
  const e=ped, name=$('pname').value.trim(), fresh=!e.orig;
  $('perr').textContent='';
  if(e.orig && name!==e.orig){
    const r=await post('/api/palettes', {op:'rename', name:e.orig, to:name});
    if(!r.ok){ $('perr').textContent=r.error; return; }
    e.orig=name;
  }
  const r=await post('/api/palettes', {op:'save', name, anchors:e.anchors});
  if(!r.ok){ $('perr').textContent=r.error; palKey=''; poll(); return; }
  closeEditor();
  if(fresh) send({palette:name}); else poll();
};

// -- a cover -----------------------------------------------------------------
function coverName(file){
  const base=(file && file.name || '').replace(/\.[^.]*$/,'').replace(/[-_]+/g,' ').replace(/\s+/g,' ').trim();
  let name = /^(image|img|pasted|screenshot|untitled)\b/i.test(base) || !base ? 'Cover'
    : base.charAt(0).toUpperCase()+base.slice(1);
  name=name.slice(0, PR.name);
  const taken=new Set(palNames(live).map(n=>n.toLowerCase()).concat(['auto','screen']));
  for(let i=2, b=name.slice(0, PR.name-3); taken.has(name.toLowerCase()); i++) name=b+' '+i;
  return name;
}
async function fromCover(src, name){
  // shrink to PR.side px in the browser, which also decodes it: the desk
  // needs no image library and the picture never leaves this machine
  const w0=src.width, h0=src.height, k=Math.min(1, PR.side/Math.max(w0,h0));
  const w=Math.max(1,Math.round(w0*k)), h=Math.max(1,Math.round(h0*k));
  const cv=document.createElement('canvas'); cv.width=w; cv.height=h;
  const cx=cv.getContext('2d'); cx.drawImage(src,0,0,w,h);
  const px=cx.getImageData(0,0,w,h).data, rgb=new Uint8Array(w*h*3);
  for(let i=0,j=0;i<px.length;i+=4){ rgb[j++]=px[i]; rgb[j++]=px[i+1]; rgb[j++]=px[i+2]; }
  let bin=''; for(let i=0;i<rgb.length;i+=0x8000) bin+=String.fromCharCode.apply(null, rgb.subarray(i,i+0x8000));
  const thumb=cv.toDataURL('image/png');
  const r=await post('/api/palettes/extract', {w, h, rgb:btoa(bin)});
  closePick();
  if(!r.ok) return;
  if(r.anchors) openEditor(null, {name, anchors:r.anchors, thumb});
  else{ closeEditor(); $('pnothumb').src=thumb; $('pno').hidden=false; }
}
async function fromFile(file){
  if(!file || !/^image\//.test(file.type) || modeOf(live||{})!=='music') return;
  try{ const bmp=await createImageBitmap(file); await fromCover(bmp, coverName(file)); bmp.close && bmp.close(); }
  catch(e){ /* not a picture the browser can read */ }
}
$('pfile').addEventListener('change', ()=>{ fromFile($('pfile').files[0]); $('pfile').value=''; });
$('pnook').onclick=()=>{ $('pno').hidden=true; };
document.addEventListener('paste', e=>{
  if(/^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement||{}).tagName||'')) return;
  const f=[...(e.clipboardData ? e.clipboardData.files : [])].find(f=>/^image\//.test(f.type));
  if(f){ e.preventDefault(); fromFile(f); }
});
// a file dragged over the desk: the side column becomes the place to drop it
let dragDepth=0;
const droppable = e => e.dataTransfer && [...e.dataTransfer.types].includes('Files')
  && document.body.dataset.deck==='runtime' && document.body.dataset.mode==='music';
document.addEventListener('dragenter', e=>{ if(!droppable(e)) return; dragDepth++; $('drop').hidden=false; });
document.addEventListener('dragleave', e=>{ if(!droppable(e)) return; if(--dragDepth<=0){ dragDepth=0; $('drop').hidden=true; } });
document.addEventListener('dragover', e=>{ if(droppable(e)) e.preventDefault(); });
document.addEventListener('drop', e=>{
  if(!droppable(e)) return;
  e.preventDefault(); dragDepth=0; $('drop').hidden=true;
  fromFile([...e.dataTransfer.files].find(f=>/^image\//.test(f.type)));
});

// -- a box on a still of the screen --------------------------------------------
// The box is kept in 0..1 of the still, which is all a crop needs.
let cbox=null, cdrag=null;
function openPick(){
  closeEditor(); $('pno').hidden=true;
  const sel=$('cmon');
  sel.innerHTML = live.monitors.map(m=>`<option value="${m.i}">Display ${m.i}, ${m.w} × ${m.h}</option>`).join('');
  sel.value = live.monitor; sel.hidden = live.monitors.length<2;
  $('cpick').hidden=false; cbox=null; paintPick(); loadPick();
  palKey=''; renderPalettes(live);
}
function closePick(){ if($('cpick').hidden) return; $('cpick').hidden=true; cbox=null; cdrag=null; palKey=''; renderPalettes(live); }
function loadPick(){
  const m=live.monitors.find(m=>m.i===+$('cmon').value); if(!m) return;
  $('cshot').style.aspectRatio = m.w+'/'+m.h;
  const img=$('cimg');
  img.onload=()=>{ img.hidden=false; }; img.onerror=()=>{ img.hidden=true; };
  img.src='/api/snapshot?monitor='+m.i+'&t='+Date.now();
}
$('cmon').addEventListener('change', ()=>{ cbox=null; paintPick(); loadPick(); });
$('cstop').onclick=closePick;
function paintPick(){
  const has=!!cbox;
  $('cbox').hidden=$('cshade').hidden=!has; $('ctake').disabled=!has;
  if(!has) return;
  const [L,T,W,H]=cbox.map(v=>v*100);
  Object.assign($('cbox').style,{left:L+'%',top:T+'%',width:W+'%',height:H+'%'});
  $('cshade').style.clipPath=`polygon(0 0,100% 0,100% 100%,0 100%,0 ${T}%,${L}% ${T}%,${L}% ${T+H}%,${L+W}% ${T+H}%,${L+W}% ${T}%,0 ${T}%)`;
}
function cpt(e){ const b=$('cshot').getBoundingClientRect();
  return [Math.max(0,Math.min(1,(e.clientX-b.left)/b.width)), Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))]; }
$('cshot').addEventListener('pointerdown', e=>{
  if(e.button>0) return;
  const p=cpt(e), h=e.target.dataset && e.target.dataset.h, r=cbox;
  if(h && r) cdrag={kind:'size', ax:h.includes('w')?r[0]+r[2]:r[0], ay:h.includes('n')?r[1]+r[3]:r[1]};
  else if(r && e.target===$('cbox')) cdrag={kind:'move', dx:p[0]-r[0], dy:p[1]-r[1]};
  else cdrag={kind:'size', ax:p[0], ay:p[1]};
  $('cshot').setPointerCapture(e.pointerId); e.preventDefault();
});
$('cshot').addEventListener('pointermove', e=>{
  if(!cdrag) return;
  const p=cpt(e);
  if(cdrag.kind==='move'){ const [,,w,h]=cbox;
    cbox=[Math.max(0,Math.min(1-w,p[0]-cdrag.dx)), Math.max(0,Math.min(1-h,p[1]-cdrag.dy)), w, h]; }
  else cbox=[Math.min(cdrag.ax,p[0]), Math.min(cdrag.ay,p[1]), Math.abs(p[0]-cdrag.ax), Math.abs(p[1]-cdrag.ay)];
  paintPick();
});
const cend=()=>{ if(!cdrag) return; cdrag=null; if(cbox && (cbox[2]<0.01 || cbox[3]<0.01)){ cbox=null; } paintPick(); };
$('cshot').addEventListener('pointerup', cend);
$('cshot').addEventListener('pointercancel', cend);
$('cshot').addEventListener('keydown', e=>{
  const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
  if(!d || !cbox) return;
  e.preventDefault();
  let [x,y,w,h]=cbox; const st=0.01;
  if(e.shiftKey){ w=Math.max(0.02,Math.min(1-x,w+d[0]*st)); h=Math.max(0.02,Math.min(1-y,h+d[1]*st)); }
  else{ x=Math.max(0,Math.min(1-w,x+d[0]*st)); y=Math.max(0,Math.min(1-h,y+d[1]*st)); }
  cbox=[x,y,w,h]; paintPick();
});
$('ctake').onclick=()=>{
  const img=$('cimg'); if(!cbox || img.hidden || !img.naturalWidth) return;
  const W=img.naturalWidth, H=img.naturalHeight;
  const sx=Math.round(cbox[0]*W), sy=Math.round(cbox[1]*H);
  const sw=Math.max(1,Math.round(cbox[2]*W)), sh=Math.max(1,Math.round(cbox[3]*H));
  const cv=document.createElement('canvas'); cv.width=sw; cv.height=sh;
  cv.getContext('2d').drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  fromCover(cv, coverName(null));
};

/* ------------------------------ audio input ------------------------------ */
// The device list costs a CoreAudio walk, so it is fetched on load and when
// the picker gets focus (a device may have been plugged in), never per poll.
let inputs={inputs:[],auto:''}, micKey='';
async function loadInputs(){
  try{ inputs = await (await fetch('/api/inputs')).json(); }catch(e){ return; }
  buildMics();
}
function buildMics(){
  const want = live ? (live.mic || '') : $('mic').value;
  const names = inputs.inputs.slice();
  const opts = [['', inputs.auto ? `Automatic (${inputs.auto})` : 'Automatic']]
    .concat(names.map(n=>[n,n]));
  if(want && !names.includes(want)) opts.push([want, want+' (not connected)']);
  const key = JSON.stringify(opts);
  if(key !== micKey){
    micKey = key;
    const sel=$('mic'); sel.innerHTML='';
    opts.forEach(([v,t])=>{ const o=document.createElement('option'); o.value=v; o.textContent=t; sel.append(o); });
  }
  if(document.activeElement !== $('mic')) $('mic').value = want;
}
$('mic').addEventListener('focus', loadInputs);
loadInputs();

function paintMeter(s){
  const on = s.running && modeOf(s)!=='screen' && s.input_db !== null;
  $('meter').hidden = !on;
  if(!on) return;
  $('meter').classList.toggle('none', !s.signal);
  $('meterbar').style.width = s.signal ? Math.max(0, Math.min(1, (s.input_db+60)/60))*100+'%' : '0';
  $('meterdb').textContent = s.signal ? (s.input_db >= -0.5 ? '0 dB' : '\u2212'+Math.round(-s.input_db)+' dB') : 'No signal';
}

/* ------------------------------ capture area ------------------------------ */
// The box lives in the display's own points; the frame is the display at
// its aspect ratio, with a still of it when the desk may serve one.
function mon(){ return live && live.monitors.find(m=>m.i===live.monitor); }
let regionKey='';
function paintRegion(r, live_=false){
  const m=mon(); if(!m) return;
  // the poll calls this every frame; touch the DOM only when the box moved
  const key=JSON.stringify([r,m.w,m.h]);
  if(live_ && key===regionKey) return;
  regionKey=key;
  $('shot').style.aspectRatio = m.w+'/'+m.h;
  const has=!!r;
  $('box').hidden = $('shade').hidden = !has;
  $('areaall').hidden = !has;
  $('areatxt').textContent = has ? `${r[2]} \u00d7 ${r[3]} at ${r[0]}, ${r[1]}` : 'Whole display';
  if(!has) return;
  const L=r[0]/m.w*100, T=r[1]/m.h*100, W=r[2]/m.w*100, H=r[3]/m.h*100;
  Object.assign($('box').style,{left:L+'%',top:T+'%',width:W+'%',height:H+'%'});
  $('shade').style.clipPath = `polygon(0 0,100% 0,100% 100%,0 100%,0 ${T}%,${L}% ${T}%,${L}% ${T+H}%,${L+W}% ${T+H}%,${L+W}% ${T}%,0 ${T}%)`;
}
function maybeShot(s){
  // one still when the area first shows, on a display change and on coming
  // back to the page - never on a timer
  if(!s.snapshot_ok || modeOf(s)==='music' || document.body.dataset.deck!=='runtime' || document.hidden) return;
  if(shotFor===s.monitor && !shotStale) return;
  shotFor=s.monitor; shotStale=false;
  const img=$('shotimg');
  img.onload=()=>{ img.hidden=false; };
  img.onerror=()=>{ img.hidden=true; };
  img.src='/api/snapshot?monitor='+s.monitor+'&t='+Date.now();
}
function pt(e){
  const m=mon(), b=$('shot').getBoundingClientRect();
  return [Math.max(0,Math.min(m.w,(e.clientX-b.left)/b.width*m.w)),
          Math.max(0,Math.min(m.h,(e.clientY-b.top)/b.height*m.h))];
}
function fit(x0,y0,x1,y1){
  const m=mon();
  const x=Math.round(Math.min(x0,x1)), y=Math.round(Math.min(y0,y1));
  return [x, y, Math.round(Math.min(Math.abs(x1-x0), m.w-x)), Math.round(Math.min(Math.abs(y1-y0), m.h-y))];
}
const MIN_AREA=32;
$('shot').addEventListener('pointerdown',e=>{
  const m=mon(); if(!m || e.button>0) return;
  const p=pt(e), r=draft || (live && live.region), h=e.target.dataset && e.target.dataset.h;
  if(h && r){            // resize from the corner opposite the handle
    drag={kind:'size', ax: h.includes('w') ? r[0]+r[2] : r[0], ay: h.includes('n') ? r[1]+r[3] : r[1]};
  }else if(r && e.target===$('box')){
    drag={kind:'move', dx:p[0]-r[0], dy:p[1]-r[1], w:r[2], h:r[3]};
  }else{
    drag={kind:'size', ax:p[0], ay:p[1], fresh:true};
  }
  $('shot').setPointerCapture(e.pointerId);
  e.preventDefault();
});
$('shot').addEventListener('pointermove',e=>{
  if(!drag) return;
  const m=mon(), p=pt(e);
  if(drag.kind==='move'){
    const x=Math.round(Math.max(0,Math.min(m.w-drag.w,p[0]-drag.dx)));
    const y=Math.round(Math.max(0,Math.min(m.h-drag.h,p[1]-drag.dy)));
    draft=[x,y,drag.w,drag.h];
  }else draft=fit(drag.ax,drag.ay,p[0],p[1]);
  paintRegion(draft);
});
function endDrag(){
  if(!drag) return;
  const r=draft; drag=null; draft=null;
  if(r && r[2]>=MIN_AREA && r[3]>=MIN_AREA) send({region:r});
  else paintRegion(live && live.region);       // a click, not a box
}
$('shot').addEventListener('pointerup',endDrag);
$('shot').addEventListener('pointercancel',endDrag);
$('shot').addEventListener('keydown',e=>{
  const m=mon(), r=(draft || (live && live.region));
  const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
  if(!m || !r || !d) return;
  e.preventDefault();
  const sx=Math.round(m.w/100), sy=Math.round(m.h/100);
  let [x,y,w,h]=r;
  if(e.shiftKey){ w=Math.max(MIN_AREA,Math.min(m.w-x,w+d[0]*sx)); h=Math.max(MIN_AREA,Math.min(m.h-y,h+d[1]*sy)); }
  else{ x=Math.max(0,Math.min(m.w-w,x+d[0]*sx)); y=Math.max(0,Math.min(m.h-h,y+d[1]*sy)); }
  draft=[x,y,w,h]; paintRegion(draft);
  holdUntil=Date.now()+1500;
  clearTimeout(keyTimer);
  keyTimer=setTimeout(()=>{ const r2=draft; draft=null; send({region:r2}); }, 400);
});
$('areaall').addEventListener('click',()=>{ draft=null; paintRegion(null); send({region:null}); });

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
  return modeOf(live)==='screen' ? Math.max(60, 1000/live.screen_fps) : 60;
}
async function poll(){
  if(busy) return; busy=true;
  clearTimeout(pollTimer);
  try{
    const s = await (await fetch('/api/state')).json();
    live = s;
    const [r,g,b] = s.white ? kelvin2rgb(s.ct, s.val) : hsv2rgb(s.hue, s.sat, s.val);
    const col = `rgb(${r},${g},${b})`;
    $('field').style.background = s.running ? col : '';
    $('ro').hidden = $('tape').hidden = !s.running;
    if(s.running){
      if(!wasRunning) tbuf=[];
      $('ro').style.color = (0.299*r+0.587*g+0.114*b) > 150 ? '#111' : '#fff';
      $('rh').textContent = s.white ? s.ct+' K' : s.hue+'°';
      $('rs').textContent = s.white ? '' : s.sat+'%'; $('rs').hidden = s.white;
      $('rv').textContent=s.val+'%';
      $('rb').textContent=s.bpm.toFixed(0)+' bpm'; $('rp').textContent=s.active_palette;
      $('rd').textContent=s.device;
      drawTape(col);
    }
    wasRunning = s.running;
    // a session that runs on a silent input looks like a dead lamp, so the
    // bar says so outright (screen mode has no input)
    const mode = modeOf(s);
    // Auto without Screen Recording can only ever be Music; say so where
    // the reasons go, unless something more urgent is there
    const autoNote = s.auto && !s.screen_ok
      ? {kind:'warn', text:'No Screen Recording permission, so Auto stays on Music.'} : null;
    const lampErr = !s.running && !s.notice && s.lamp_error ? {kind:'err', text:s.lamp_error} : null;
    setStatus(s.running && s.status==='running' && !s.signal && mode!=='screen' ? 'no signal'
              : !s.running && s.sunrise_ramping ? 'sunrise'
              : lampErr ? 'unreachable' : s.status,
              s.notice || autoNote || lampErr);
    paintLamp(s, r, g, b);
    paintSignals(s);
    paintMeter(s);
    maybeShot(s);
    const addr=$('addr');
    addr.classList.toggle('need', !s.ip);
    if(s.ip){ addr.innerHTML='Bulb <b></b>'; addr.lastChild.textContent=s.ip; }
    else addr.textContent='Set up bulb';
    $('go').dataset.on = s.running; $('go').textContent = s.running ? 'Stop' : 'Start';
    if(!ipTouched && document.activeElement !== $('ip')) $('ip').value = s.ip || '';

    const screenMode = mode === 'screen';
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
      $('auto').checked = s.auto;
      $('bclock').checked = s.bulb_clock;
      $('drops').checked = s.drops;
      renderPalettes(s);
      [...$('modes').children].forEach(c=>{
        c.setAttribute('aria-pressed', c.dataset.mode===mode);
        c.classList.toggle('off', s.auto && !s.screen_ok && c.dataset.mode!=='music');
      });
      document.body.dataset.mode = mode;
      $('monitor').value = s.monitor;
      if(!drag && !draft) paintRegion(s.region, true);
      buildMics();
      renderPresets(s);
      renderIdle(s);
      // faders follow the desk too, or a reload leaves them showing whatever
      // the browser restored rather than what is running
      Object.keys(KEYS).forEach(id=>{
        const el=$(id), v=s[KEYS[id]];
        if(document.activeElement===el || +el.value===v) return;
        el.value = v;
        paint(id);
      });
    }
    renderSleep(s);
  }catch(e){ live=null; setStatus('server unreachable', null); }
  busy=false;
  pollTimer = setTimeout(poll, pollDelay());
}
/* -------------------------------- signals ------------------------------- */
const SWATCH = ['#e5352b','#f29a1d','#2fb457','#2f7de1','#a24be0','#ffffff'];
const sig = {color:SWATCH[0], pulses:2, speed:'normal'};
function segOf(id, items, key){
  const box=$(id);
  items.forEach(([v,label])=>{
    const b=document.createElement('button'); b.type='button'; b.textContent=label;
    b.onclick=()=>{ sig[key]=v; [...box.children].forEach(c=>c.setAttribute('aria-pressed', c===b)); };
    b.setAttribute('aria-pressed', sig[key]===v);
    box.append(b);
  });
}
SWATCH.forEach(c=>{
  const b=document.createElement('button'); b.type='button'; b.style.background=c;
  b.setAttribute('aria-label', c); b.setAttribute('aria-pressed', c===sig.color);
  if(c==='#ffffff') b.className='pale';
  b.onclick=()=>{ sig.color=c; [...$('aswatch').children].forEach(x=>x.setAttribute('aria-pressed', x===b)); };
  $('aswatch').append(b);
});
segOf('apulses', [[1,'1'],[2,'2'],[3,'3'],[5,'5']], 'pulses');
segOf('aspeed', [['slow','Slow'],['normal','Normal'],['fast','Fast']], 'speed');
$('apulse').addEventListener('click', async ()=>{
  $('aerr').textContent='';
  const r = await post('/api/alert', sig);
  if(!r.ok) $('aerr').textContent = r.error;
});
async function focusOp(op){
  $('ferr').textContent='';
  const r = await post('/api/focus', {op});
  if(!r.ok) $('ferr').textContent = r.error;
  poll();
}
$('fgo').addEventListener('click', ()=>{
  const f = live && live.focus;
  focusOp(f && f.phase!=='idle' && !f.paused ? 'pause' : 'start');
});
$('freset').addEventListener('click', ()=>focusOp('reset'));
$('son').addEventListener('click', ()=>{
  const on = $('son').getAttribute('aria-checked')!=='true';
  $('son').setAttribute('aria-checked', on);
  send({sunrise_on:on});
});
$('sat').addEventListener('change', ()=>{
  const [h,m] = $('sat').value.split(':').map(Number);
  if(Number.isFinite(h) && Number.isFinite(m)) send({sunrise_at:h*60+m});
});
$('alm').addEventListener('click', ()=>$('sunrise').scrollIntoView({behavior:'smooth', block:'center'}));
$('sstop').addEventListener('click', async ()=>{ await post('/api/sunrise', {op:'stop'}); poll(); });
const mmss = n => String(Math.floor(n/60)).padStart(2,'0')+':'+String(n%60).padStart(2,'0');
function paintSignals(s){
  const f = s.focus || {phase:'idle'};
  const active = f.phase!=='idle';
  // Focus whites would fight a session's frames: it is a stopped-desk tool
  $('focus').hidden = s.running;
  $('fclock').textContent = mmss(active ? f.left : s.focus_work*60);
  $('fphase').textContent = active ? (f.phase==='work' ? 'Work' : 'Break')+' · '+f.round : '';
  $('fgo').textContent = !active ? 'Start' : f.paused ? 'Resume' : 'Pause';
  $('freset').disabled = !active;
  const at = mmss(s.sunrise_at);
  $('alm').hidden = !s.sunrise_on;
  $('almt').textContent = at;
  $('sstop').hidden = !(s.sunrise_ramping && !s.running);
  if(Date.now() > holdUntil){
    $('son').setAttribute('aria-checked', s.sunrise_on);
    if(document.activeElement !== $('sat')) $('sat').value = at;
  }
}

poll();
document.addEventListener('visibilitychange', ()=>{ if(!document.hidden){ shotStale=true; lampStale=true; poll(); cloudPoll(); } });
addEventListener('focus', ()=>{ lampStale=true; poll(); });


/* ------------------------------- lamp mode ------------------------------- */
// While no session runs the desk is a plain lamp. The lamp is read when this
// section comes into view (load, Stop, returning to the page), never on a
// timer; every change is one short connection on the desk's side.
const LM = __LAMP__;
let lampStale=true, lampReadAt=0, lampReadTimer=0, lampIp=null, lampHold=0, lampTimer=0, lampWant={}, dayDrag=null, dayDraft=null;
function smooth(x){ x=Math.min(Math.max(x,0),1); return x*x*(3-2*x); }
function dayKelvin(min, s){   // music_bulb_dash.day_kelvin
  const span=((s.dusk+LM.down-s.wake)%1440+1440)%1440, into=((min-s.wake)%1440+1440)%1440;
  const f = into>=span ? 0 : Math.min(smooth(into/LM.up), smooth((span-into)/LM.down));
  return Math.round((s.night_k+(s.day_k-s.night_k)*f)/LM.step)*LM.step;
}
function hhmm(m){ return String(Math.floor(m/60)).padStart(2,'0')+':'+String(m%60).padStart(2,'0'); }
function following(s){ return s.follow_day && !(s.day_held && s.day_held*1000 > Date.now()); }
async function lampRead(){
  // at most one read per 1.5 s; one asked for sooner is put off, not dropped
  const wait = 1500-(Date.now()-lampReadAt);
  if(wait > 0){
    if(!lampReadTimer) lampReadTimer=setTimeout(()=>{ lampReadTimer=0; lampRead(); }, wait);
    return;
  }
  lampStale=false;
  lampReadAt=Date.now();
  try{ await fetch('/api/lamp'); }catch(e){}
  poll();
}
async function lampDo(body){
  lampHold=Date.now()+1200;
  try{ await post('/api/lamp', body); }catch(e){}
  poll();
}
function lampSoon(key, v){   // a fader being dragged: at most ~3 sends a second
  lampWant[key]=v; lampHold=Date.now()+1200;
  if(lampTimer) return;
  lampTimer=setTimeout(()=>{ const b=lampWant; lampWant={}; lampTimer=0; lampDo(b); }, 330);
}
function paintFader(id, text){
  const el=$(id);
  el.style.setProperty('--p', (el.value-el.min)/(el.max-el.min)*100+'%');
  $('v-'+id).textContent=text;
}
function drawStrip(s){
  const c=$('stripc'), dpr=devicePixelRatio||1, W=Math.round(c.clientWidth*dpr)||1, H=Math.round(c.clientHeight*dpr)||1;
  if(c.width!==W || c.height!==H){ c.width=W; c.height=H; }
  const x=c.getContext('2d');
  for(let px=0; px<W; px++){
    const [r,g,b]=kelvin2rgb(dayKelvin(px/W*1440, s), 100);
    x.fillStyle=`rgb(${r},${g},${b})`; x.fillRect(px,0,1,H);
  }
}
let stripKey='';
function paintStrip(s){
  const d=dayDraft ? Object.assign({}, s, dayDraft) : s;
  const key=[d.wake,d.dusk,d.day_k,d.night_k,$('strip').clientWidth].join();
  if(key!==stripKey){ stripKey=key; drawStrip(d); }
  [['hwake',d.wake],['hdusk',d.dusk]].forEach(([id,m])=>{
    const h=$(id); h.style.left=(m/1440*100)+'%';
    h.setAttribute('aria-valuenow', m); h.setAttribute('aria-valuetext', hhmm(m)); h.title=hhmm(m);
  });
  const now=new Date();
  $('snow').style.left=((now.getHours()*60+now.getMinutes())/1440*100)+'%';
}
function paintLamp(s, r, g, b){
  const show = !s.running && !!s.ip;
  // read when the section comes into view or the address changes - never
  // because a read came back empty, which would make it a timer
  if(show && ($('lampctl').hidden || s.ip !== lampIp)) lampStale=true;
  lampIp = s.ip;
  $('lampctl').hidden = !show;
  const L=s.lamp, on=!!(L && L.power==='on'), held=Date.now() < lampHold;
  const lro=$('lro');
  lro.hidden = !(show && on);
  if(show){
    if(on){
      const [lr,lg,lb] = L.color_mode===2 ? kelvin2rgb(L.ct, 40+0.6*L.bright)
        : [(L.rgb>>16)&255,(L.rgb>>8)&255,L.rgb&255].map(c=>Math.round(c*(40+0.6*L.bright)/100));
      $('field').style.background=`rgb(${lr},${lg},${lb})`;
      lro.style.color=(0.299*lr+0.587*lg+0.114*lb) > 150 ? '#111' : '#fff';
      lro.textContent=(L.color_mode===2 ? L.ct+' K · ' : '')+L.bright+'%';
    } else $('field').style.background='';
  }
  if(!show){ return; }
  if(lampStale && !lampReadTimer && document.body.dataset.deck==='runtime' && !document.hidden) lampRead();
  const follow=following(s);
  $('lpower').setAttribute('aria-checked', on);
  $('lpower').disabled = !L;
  $('lct').disabled = $('lbr').disabled = !on;
  if(!held){
    $('follow').checked=follow;
    if(L){ $('lct').value=L.ct || 4000; $('lbr').value=L.bright || 1; }
  }
  $('day').hidden=!follow; $('ctrow').hidden=follow;
  paintFader('lct', L ? $('lct').value+' K' : '–');
  paintFader('lbr', L ? $('lbr').value+'%' : '–');
  if(follow) paintStrip(s);
}
$('lpower').addEventListener('click',()=>{
  const on=$('lpower').getAttribute('aria-checked')==='true';
  $('lpower').setAttribute('aria-checked', !on);
  lampDo({op:'toggle'});
});
$('lct').addEventListener('input',e=>{ paintFader('lct', e.target.value+' K'); lampSoon('ct', +e.target.value); });
$('lbr').addEventListener('input',e=>{ paintFader('lbr', e.target.value+'%'); lampSoon('bright', +e.target.value); });
$('follow').addEventListener('change',e=>{ lampHold=Date.now()+1200; send({follow_day:e.target.checked}); });
// the strip: drag the nearer handle, or focus one and use the arrow keys
function stripMin(e){
  const b=$('strip').getBoundingClientRect();
  return Math.round(Math.min(Math.max((e.clientX-b.left)/b.width,0),1)*1440/LM.grid)*LM.grid;
}
// the handles never cross: evening stays LM.minDay after morning (as clean_lamp)
function dayClamp(k, m){
  const top=1440-LM.grid;
  return k==='wake' ? Math.max(0, Math.min(m, live.dusk-LM.minDay))
                    : Math.min(top, Math.max(m, live.wake+LM.minDay));
}
$('strip').addEventListener('pointerdown',e=>{
  if(!live) return;
  const m=stripMin(e), dist=k=>Math.min(Math.abs(m-live[k]), 1440-Math.abs(m-live[k]));
  dayDrag = e.target.id==='hwake' ? 'wake' : e.target.id==='hdusk' ? 'dusk' : dist('wake')<=dist('dusk') ? 'wake' : 'dusk';
  $('strip').setPointerCapture(e.pointerId);
  dayDraft={[dayDrag]:dayClamp(dayDrag, m)}; paintStrip(live);
});
$('strip').addEventListener('pointermove',e=>{ if(dayDrag){ dayDraft={[dayDrag]:dayClamp(dayDrag, stripMin(e))}; paintStrip(live); } });
function dayCommit(){
  if(!dayDraft) return;
  const d=dayDraft; dayDrag=null;
  Object.assign(live, d); send(d); dayDraft=null;
}
$('strip').addEventListener('pointerup', dayCommit);
$('strip').addEventListener('pointercancel', ()=>{ dayDrag=null; dayDraft=null; if(live) paintStrip(live); });
let dayKeyTimer=0;
[['hwake','wake'],['hdusk','dusk']].forEach(([id,k])=>{
  $(id).addEventListener('keydown',e=>{
    const d={ArrowLeft:-1,ArrowDown:-1,ArrowRight:1,ArrowUp:1}[e.key];
    if(!d || !live) return;
    e.preventDefault();
    const m=dayClamp(k, ((dayDraft && dayDraft[k]) ?? live[k]) + d*LM.grid*(e.shiftKey?4:1));
    dayDraft={[k]:m}; paintStrip(live);
    clearTimeout(dayKeyTimer); dayKeyTimer=setTimeout(dayCommit, 400);
  });
});

/* ------------------------------ setup deck ------------------------------ */
$('defbtn').addEventListener('click', ()=>{ $('defask').hidden=false; $('defok').hidden=true; $('deferr').textContent=''; });
$('defno').addEventListener('click', ()=>{ $('defask').hidden=true; });
$('defyes').addEventListener('click', async ()=>{
  $('deferr').textContent='';
  const r=await post('/api/default', {});
  if(!r.ok){ $('deferr').textContent=r.error||'The bulb did not answer.'; return; }
  $('defask').hidden=true; $('defok').hidden=false;
});
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
    signals = None                  # lamp_signals.Signals: alerts, Focus, sunrise
    # Cloud login means a Xiaomi password crosses this socket. The desk has
    # no authentication, so on anything but loopback the endpoints stay shut
    # unless the operator asked for them with --allow-remote-login.
    cloud_enabled = True
    cloud_reason = ""
    local = True                    # bound to loopback; see _host_ok
    lamp = None                     # LampMode
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

    def _host_ok(self):
        """On a loopback bind, serve only a page that was itself opened on a
        loopback name. Without this a website can point its own hostname at
        127.0.0.1 (DNS rebinding) and read the desk as if it were same-origin
        - and the desk now serves a still of the screen."""
        if not self.local:
            return True
        host = self.headers.get("Host", "")
        name = host[1:host.find("]")] if host.startswith("[") else host.split(":")[0]
        name = name.lower()
        if name in ("localhost", "127.0.0.1", "::1") or name.endswith(".localhost"):
            return True
        self.close_connection = True
        self._send(403, "this desk answers on localhost only", "text/plain")
        return False

    def _own_page(self):
        """Refuse a request that another website made the browser send. The
        desk has no login, so without this any page open in the user's
        browser could POST to it: flash the lamp, stop a session, overwrite
        the bulb's power-on look. A browser names the asking page in Origin
        on every cross-site POST and marks the request Sec-Fetch-Site; the
        CLI and scripts send neither and pass."""
        site, origin = self.headers.get("Sec-Fetch-Site"), self.headers.get("Origin")
        ok = site in (None, "same-origin", "none")
        if ok and origin is not None:
            ok = origin.split("://", 1)[-1].lower() == self.headers.get("Host", "").lower()
        if not ok:
            self.close_connection = True
            self._send(403, "this desk takes requests from its own page only", "text/plain")
        return ok

    def do_GET(self):
        if not self._host_ok():
            return
        if self.path.startswith("/api/") and not self._own_page():
            return
        if self.path.startswith("/api/state"):
            snap = self.state.snapshot()
            engine = getattr(Handler, "engine", None)
            if snap["running"] and engine is not None:
                # the page paints what the lamp shows: dimmed by a sleep fade
                snap["val"] = alone.dim_level(snap["val"],
                                              alone.dim_factor(engine.sleep.left()))
            if self.signals is not None:
                snap.update(self.signals.view())
            self._send(200, json.dumps(snap), "application/json")
        elif self.path.startswith("/api/cloud/state"):
            snap = self.cloud.snapshot()
            snap["available"] = not self._cloud_ready()
            snap["reason"] = self._cloud_ready()
            snap["servers"] = list(cloud.SERVERS) if cloud else []
            self._json(200, snap)
        elif self.path.startswith("/api/inputs"):
            # enumerating devices costs a CoreAudio walk, so the page asks
            # for it when it loads and when the picker gets focus, not per poll
            self._json(200, audio_inputs())
        elif self.path.startswith("/api/snapshot"):
            self._snapshot()
        elif self.path.startswith("/api/lamp"):
            self._do_lamp(None)
        elif self.path in ("/", "/index.html"):
            page = (PAGE.replace("__PALETTES__", json.dumps(PALETTES))
                        .replace("__PAL_RULES__", json.dumps({
                            "name": pal_rules.NAME_MAX, "min": pal_rules.MIN_ANCHORS,
                            "max": pal_rules.MAX_ANCHORS, "sat": pal_rules.SAT_RANGE,
                            "side": ART_SIDE}))
                        .replace("__PRESETS__", json.dumps(BUILTIN_PRESETS))
                        .replace("__SLEEP_STEPS__", json.dumps(list(alone.SLEEP_STEPS)))
                        .replace("__LAMP__", json.dumps({
                            "up": DAY_UP, "down": DAY_DOWN, "step": DAY_STEP, "grid": GRID,
                            "minDay": MIN_DAY})))
            self._send(200, page, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if not self._host_ok() or not self._own_page():
            return
        if self.path.startswith("/api/cloud/"):
            self._do_cloud()
            return
        if self.path.startswith("/api/presets"):
            self._do_presets()
            return
        if self.path.startswith("/api/palettes/extract"):
            body = self._body()
            anchors, error = extract_palette(body)
            self._json(400 if error else 200,
                       {"ok": not error, "error": error,
                        "anchors": [list(a) for a in anchors] if anchors else None})
            return
        if self.path.startswith("/api/palettes"):
            self._do_palettes()
            return
        if self.path.startswith("/api/sleep"):
            # {"minutes": n} sets the bulb's off timer, 0 or null cancels it
            body = self._body()
            mins = body.get("minutes") if isinstance(body, dict) else None
            timer = Handler.engine.sleep
            if mins in (0, None):
                error = timer.cancel()
            else:
                mins = alone.clean_minutes(mins)
                error = timer.start(mins) if mins else "bad minutes"
            self._json(400 if error else 200, {"ok": not error, "error": error,
                                               "sleep": self.state.snapshot()["sleep"]})
            return
        if self.path.startswith("/api/default"):
            # the lamp's power-on look (set_default). Its old one cannot be
            # read back, so the page confirms first.
            self._body()
            snap = self.state.snapshot()
            ip = snap["ip"]
            if snap["running"]:
                # a session's frames are not a look to keep, and the session
                # holds the bulb's slots: stop first, as check_default.py says
                self._json(409, {"ok": False, "error": "Stop the session first."})
                return
            error = "" if ip else "Set up the bulb first."
            engine = getattr(Handler, "engine", None)
            if ip:
                # one plain exchange at a time, and none while a session
                # that is stopping still restores the lamp
                with (engine.sleep.plain() if engine is not None else contextlib.nullcontext()):
                    try:
                        music_bulb.lamp_call(ip, [("set_default", [])])
                    except Exception as exc:
                        error = lamp_error(exc)
            self._json(400 if error else 200, {"ok": not error, "error": error})
            return
        if self.path.startswith("/api/lamp"):
            body = self._body()
            if not isinstance(body, dict):
                self._send(400, "bad json", "text/plain")
                return
            self._do_lamp(body)
            return
        if self.path.split("?")[0] in ("/api/alert", "/api/focus", "/api/sunrise"):
            self._do_signals()
            return
        if not self.path.startswith("/api/control"):
            self.close_connection = True        # its body was never read
            self._send(404, "not found", "text/plain")
            return
        patch = self._body()
        if patch is None:
            self._send(400, "bad json", "text/plain")
            return
        snap = self.state.snapshot()
        # a preset is its look, applied like any other change; the list itself
        # changes only through /api/presets
        if isinstance(patch, dict):
            patch.pop("presets", None)
            patch.pop("palettes", None)
            patch.pop("sleep", None)            # only /api/sleep talks to the bulb
        if isinstance(patch, dict) and "preset" in patch:
            found = find_preset(patch.pop("preset"), snap)
            if found:
                patch = dict(found["look"], **patch)
        # the LEDs cannot resolve past ~30/s, so clamp rather than trust the UI
        clean = clean_patch(patch, snap)
        # a saved input may be unplugged (the desk then says so), but the
        # page can only pick one that is there now
        if clean.get("mic") and clean["mic"] != snap["mic"] \
                and clean["mic"] not in audio_inputs()["inputs"]:
            clean.pop("mic")
        # the bulb's id is learned on the wire, never taken from the page,
        # and a new address may be a different bulb
        clean.pop("did", None)
        # the hold is set by a hand-set temperature; ticking Follow the day
        # is what lifts it
        clean.pop("day_held", None)
        if clean.get("follow_day") is True:
            clean["day_held"] = None
        # Auto off keeps the mode it was running rather than jumping back to
        # the last one picked by hand
        if clean.get("auto") is False and snap["auto"] and "mode" not in clean \
                and snap["auto_mode"] in MODES:
            clean["mode"] = snap["auto_mode"]
        if "ip" in clean and clean["ip"] != snap["ip"]:
            clean["did"] = None
        if "running" in clean:
            clean["status"] = "starting" if clean["running"] else "stopping"
            clean["notice"] = None
        if "ip" in clean and snap["running"] and clean["ip"] != snap["ip"]:
            # the session reads the address once, at connect time, so a
            # live run has to end before the new bulb can be opened
            clean["running"] = False
            clean["status"] = f"bulb address is now {clean['ip']} - press Start"
        if clean.get("running") and not snap["running"] and self.signals is not None:
            # Focus puts the lamp back before the session reads it, and no
            # plain exchange with the lamp can slip in before `running`
            self.signals.claim_lamp(lambda: self.state.control(**clean))
        else:
            self.state.control(**clean)
        if self.lamp:
            self.lamp.changed(clean)
        if self.signals is not None and ("sunrise_on" in clean or "sunrise_at" in clean):
            self.signals.changed()
        self._send(200, json.dumps({"ok": True}), "application/json")

    def _do_lamp(self, body):
        """GET reads the lamp; POST {op?, ct?, bright?} changes it. Each is
        one short connection. While a session runs only a toggle is taken
        (it stops the session, then turns the lamp off)."""
        if self.lamp is None:
            self._json(503, {"ok": False, "error": "lamp mode is not running"})
            return
        try:
            if body is None:
                lamp = self.lamp.read()
            else:
                op = body.get("op")
                ct, bright = body.get("ct"), body.get("bright")
                if (op is not None and op not in LAMP_OPS) \
                        or (ct is not None and not _number(ct)) \
                        or (bright is not None and not _number(bright)):
                    self._json(400, {"ok": False, "error": "bad lamp action"})
                    return
                lamp = self.lamp.act(op, ct, bright)
        except LampBusy as exc:
            self._json(409, {"ok": False, "error": str(exc),
                             "lamp": self.state.snapshot()["lamp"]})
            return
        except Exception:
            self._json(502, {"ok": False, "error": self.state.snapshot()["lamp_error"]})
            return
        self._json(200, {"ok": True, "lamp": lamp})

    def _do_signals(self):
        body = self._body()
        if body is None or not isinstance(body, dict):
            self._send(400, "bad json", "text/plain")
            return
        if self.signals is None:
            self._json(503, {"ok": False, "error": "signals are off"})
            return
        path = self.path.split("?")[0]
        if path == "/api/alert":
            picked, error = clean_alert(body)
            if error:
                self._json(400, {"ok": False, "error": error})
                return
            code, payload = self.signals.alert(*picked)
        elif path == "/api/focus":
            code, payload = self.signals.focus_op(body.get("op"))
        else:
            if body.get("op") != "stop":
                self._json(400, {"ok": False, "error": "op must be stop"})
                return
            code, payload = self.signals.sunrise_stop()
        self._json(code, payload)

    def _do_presets(self):
        body = self._body()
        if body is None:
            self._send(400, "bad json", "text/plain")
            return
        presets, error = preset_op(body, self.state.snapshot())
        if not error:
            self.state.control(presets=presets)
        self._json(200 if not error else 400, {"ok": not error, "error": error})

    def _do_palettes(self):
        body = self._body()
        if body is None:
            self._send(400, "bad json", "text/plain")
            return
        patch, error = palette_op(body, self.state.snapshot())
        if not error:
            self.state.control(**patch)
        self._json(200 if not error else 400, {"ok": not error, "error": error})

    def _snapshot(self):
        snap = self.state.snapshot()
        if not snap["snapshot_ok"]:
            self._send(403, "snapshots are served only on loopback, with "
                            "Screen Recording permission", "text/plain")
            return
        m = re.search(r"[?&]monitor=(\d+)", self.path)
        monitor = int(m.group(1)) if m else snap["monitor"]
        try:
            png = display_snapshot(monitor)
        except Exception as exc:
            self._send(404, f"no snapshot: {exc}", "text/plain")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(png)

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
    p.add_argument("--mic", help="exact audio device name to capture; "
                   "can also be chosen on the desk")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--settings", default=settings_path(), metavar="PATH",
                   help="where the desk remembers its settings "
                        "(default: %(default)s); pass '' to keep nothing")
    p.add_argument("--no-restore", dest="restore", action="store_false",
                   help="leave the lamp on the last colour sent when the desk "
                        "stops, instead of putting back what it showed before")
    p.add_argument("--capture", choices=CAPTURE_BACKENDS, default="auto",
                   help="screen capture: a ScreenCaptureKit stream (sck), one "
                        f"mss grab per frame (mss), or auto (= {AUTO_CAPTURE}); "
                        "a stream that fails falls back to mss")
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

    loopback = args.host in ("127.0.0.1", "localhost", "::1", "")
    store = SettingsStore(args.settings or None)
    saved = store.load()
    # the command line wins over the file for this run
    if args.ip:
        saved["ip"] = args.ip
    if args.mic:
        saved["mic"] = args.mic
    state = State(saved, store=store, local=loopback)
    store.attach(state)
    engine = Engine(state, restore=args.restore, capture=args.capture)
    engine.start()
    lamp = LampMode(state, idle=engine.idle, lock=engine.lamp_lock)
    engine.on_session_end = lamp.session_ended
    lamp.start()
    Handler.lamp = lamp
    signals = Signals(state, link=lambda: engine.link,
                      parked=lambda timeout=0.0: engine.idle.wait(timeout),
                      lamp_lock=engine.lamp_lock)
    lamp.yield_to = signals.holding
    signals.start()
    Handler.signals = signals
    Handler.state = state
    Handler.engine = engine
    Handler.cloud = CloudSession()
    Handler.local = loopback
    Handler.cloud_enabled = loopback or args.allow_remote_login
    Handler.cloud_reason = ("" if Handler.cloud_enabled else
                            "cloud login is off because the desk is bound to "
                            f"{args.host} with no authentication - restart with "
                            "--allow-remote-login if that is what you want")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "localhost" if args.host == "127.0.0.1" else args.host
    ip = state.snapshot()["ip"]
    print(f"Desk ready at http://{shown}:{args.port}  (bulb {ip or 'not set'})")

    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # end a running session the normal way, so the lamp is put back and
        # its socket closed, before the daemon threads die with the process
        if state.snapshot()["running"]:
            print("\nstopping...")
            state.set(running=False, status="stopping")
            state.changed.set()
            engine.idle.wait(10)
        store.close()
        srv.server_close()
        print("bye")


if __name__ == "__main__":
    main()
