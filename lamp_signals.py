"""The lamp as a notifier and a timer: alert pulses, the Focus timer
(pomodoro) and the sunrise alarm.

Every effect is one colour flow the bulb runs itself (`start_cf`, or
`set_scene cf` for a lamp that is off), so nothing here streams frames.
Measured on the color5 (2026-11-14, firmware as shipped):

- `start_cf` with end action 0 puts back exactly the colour (rgb + bright)
  or white (ct + bright) the lamp showed before the flow.
- `start_cf` on a lamp that is off answers "ok" and does nothing;
  `set_scene ["cf", ...]` turns it on for the flow, and with action 0 it is
  off again afterwards with its remembered look untouched.
- A flow takes at most 16 steps: 16 was accepted, 24 refused with -5001
  "invalid params" (30 steps / 416 characters refused too).
- A step may last far longer than 65 535 ms: 65 536, 450 000 and 1 800 000
  were accepted, and a 70 000 ms step was still running after 5.5 s (no
  16-bit wrap). While a flow runs, get_prop reports the state from before
  it, so progress cannot be read back.

The desk side (`Signals`) knows nothing about HTTP: the handler calls it.
"""
import datetime as _dt
import re
import socket
import threading
import time

from music_bulb import LAMP_PORT, lamp_call, read_lamp, restore_commands

# ---------------------------------------------------------------------------
# alert pulses
# ---------------------------------------------------------------------------
SPEEDS = {"slow": 700, "normal": 350, "fast": 175}   # ms per half pulse
MAX_PULSES = 5                    # 10 steps; the bulb takes 16 at most
MAX_STEPS = 16
NAMED = {"red": 0xE5352B, "orange": 0xF29A1D, "green": 0x2FB457,
         "blue": 0x2F7DE1, "purple": 0xA24BE0, "white": 0xFFFFFF}
ALERT_DEFAULTS = {"color": "#e5352b", "pulses": 2, "speed": "normal"}
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_color(text):
    """'#rrggbb', 'rrggbb' or one of NAMED -> 0xRRGGBB, or None."""
    if not isinstance(text, str):
        return None
    text = text.strip().lower()
    if text in NAMED:
        return NAMED[text]
    m = _HEX.match(text)
    return int(m.group(1), 16) if m else None


def clean_alert(body):
    """(rgb, pulses, speed) from an /api/alert body, or (None, error).
    Only what a pulse needs is read; every other key is ignored."""
    if not isinstance(body, dict):
        return None, "expected a JSON object"
    rgb = parse_color(body.get("color", ALERT_DEFAULTS["color"]))
    if rgb is None or rgb == 0:
        return None, "color must be #rrggbb (not black) or one of " + ", ".join(NAMED)
    pulses = body.get("pulses", ALERT_DEFAULTS["pulses"])
    if isinstance(pulses, bool) or not isinstance(pulses, int) \
            or not 1 <= pulses <= MAX_PULSES:
        return None, f"pulses must be a whole number from 1 to {MAX_PULSES}"
    speed = body.get("speed", ALERT_DEFAULTS["speed"])
    if speed not in SPEEDS:
        return None, "speed must be " + ", ".join(SPEEDS)
    return (rgb, pulses, speed), ""


def alert_flow(rgb, pulses=2, speed="normal"):
    """-> (count, expression, ms): full brightness and down to 1% in the
    colour, `pulses` times. Sent with end action 0 the lamp then returns to
    what it showed."""
    half = SPEEDS[speed]
    steps = [f"{half},1,{rgb},100", f"{half},1,{rgb},1"] * pulses
    return len(steps), ",".join(steps), 2 * half * pulses


def flow_then_white(flow, ct, bright, ms=600):
    """An alert flow followed by a white it stays on (end action 1)."""
    count, expr, length = flow
    return count + 1, f"{expr},{ms},2,{ct},{bright}", length + ms


def alert_lamp(ip, flow, connect=socket.create_connection, timeout=2.0):
    """One short plain connection: ask whether the lamp is on, then pulse.
    A lamp that is off ignores start_cf, so it gets `set_scene cf`, which
    lights it for the pulse and leaves it off again with its look intact."""
    count, expr, _ = flow
    sock = connect((ip, LAMP_PORT), timeout)
    try:
        (power,) = lamp_call(ip, [("get_prop", ["power"])], sock=sock, timeout=timeout)
        if power and power[0] == "on":
            cmd = ("start_cf", [count, 0, expr])
        else:
            cmd = ("set_scene", ["cf", count, 0, expr])
        lamp_call(ip, [cmd], sock=sock, timeout=timeout)
    finally:
        try:
            sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# sunrise
# ---------------------------------------------------------------------------
SUNRISE_MS = 30 * 60 * 1000
# (share of the ramp, kelvin, brightness): long at the dim warm end, where
# each step is most visible, quicker once the room is lit
SUNRISE_STEPS = ((0.22, 2000, 6), (0.22, 2700, 22), (0.2, 3500, 48),
                 (0.18, 4600, 76), (0.18, 6500, 100))
LATE_OK = 20 * 60          # a Mac that wakes this late after the time still starts it
CHECK_S = 60               # longest wait: a sleeping Mac stops the monotonic clock


def sunrise_flow(total_ms=SUNRISE_MS):
    """-> (count, expression, ms): 1% at 1700 K, then up to 100% at 6500 K
    over `total_ms`, in a handful of steps the bulb interpolates. Sent with
    end action 1 the lamp stays at full."""
    steps = ["50,2,1700,1"]
    for share, ct, bright in SUNRISE_STEPS:
        steps.append(f"{max(50, int(round(total_ms * share)))},2,{ct},{bright}")
    return len(steps), ",".join(steps), total_ms + 50


def next_fire(now, at_minutes):
    """The first moment strictly after `now` (a naive local datetime) whose
    clock reads `at_minutes` past midnight."""
    h, m = divmod(int(at_minutes) % 1440, 60)
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t = (now + _dt.timedelta(days=1)).replace(hour=h, minute=m, second=0,
                                                   microsecond=0)
    return t


class AlarmPlan:
    """When the alarm fires and how long to sleep, from the wall clock.

    `step(now, on, at)` -> (fire, wait): fire when the set time has come (or
    passed by at most LATE_OK, for a Mac that just woke), wait = seconds
    until the next look (None = nothing armed, sleep until a change). Waits
    are capped at CHECK_S because macOS stops the monotonic clock while the
    machine sleeps, so a long wait would overshoot a wake-up."""

    def __init__(self):
        self.key = None
        self.target = None

    def step(self, now, on, at):
        if not on:
            self.key = self.target = None
            return False, None
        if self.key != at or self.target is None:
            self.key, self.target = at, next_fire(now, at)
        left = (self.target - now).total_seconds()
        fire = False
        if left <= 0:
            fire = -left <= LATE_OK
            self.target = next_fire(now, at)
            left = (self.target - now).total_seconds()
        return fire, min(left, CHECK_S)


# ---------------------------------------------------------------------------
# Focus (pomodoro)
# ---------------------------------------------------------------------------
WORK_WHITE = (5000, 90)          # kelvin, brightness
BREAK_WHITE = (2700, 45)
TO_BREAK = NAMED["green"]
TO_WORK = NAMED["red"]


class Focus:
    """The pomodoro state machine, on a clock passed in (monotonic seconds).

    phase: idle -> work -> break -> work ... ; `round` counts work blocks.
    A pause keeps what is left; new lengths apply from the next phase."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase, self.paused, self.end, self.left, self.round = "idle", False, None, None, 0

    @property
    def active(self):
        return self.phase != "idle"

    def start(self, now, work_s):
        """Start a first work block, or resume a paused one. -> True if it
        was idle (the lamp has to be taken)."""
        if self.phase == "idle":
            self.phase, self.round, self.end, self.paused = "work", 1, now + work_s, False
            return True
        if self.paused:
            self.end, self.left, self.paused = now + self.left, None, False
        return False

    def pause(self, now):
        if self.active and not self.paused:
            self.left, self.end, self.paused = max(0.0, self.end - now), None, True

    def due(self, now):
        return self.active and not self.paused and now >= self.end

    def advance(self, now, work_s, break_s):
        """Move to the next phase. -> the phase entered."""
        if self.phase == "work":
            self.phase, self.end = "break", now + break_s
        else:
            self.phase, self.end, self.round = "work", now + work_s, self.round + 1
        return self.phase

    def wait(self, now):
        """Seconds until the next change, or None while idle or paused."""
        if not self.active or self.paused:
            return None
        return max(0.0, self.end - now)

    def view(self, now):
        left = self.left if self.paused else (self.end - now if self.active else None)
        return {"phase": self.phase, "paused": self.paused, "round": self.round,
                "left": None if left is None else max(0, int(round(left)))}


# ---------------------------------------------------------------------------
# the desk's side
# ---------------------------------------------------------------------------
class Signals:
    """Alerts, Focus and the sunrise for one desk.

    `state` is the desk's State, `link()` returns the running session's Link
    or None. The lamp is reached through `lamp(ip, commands)` and
    `pulse(ip, flow)` (plain short connections), swappable for tests.

    Focus and the sunrise run only on a stopped desk: their whites would be
    painted over by a session's frames. Each has one thread that sleeps on
    an Event until its next change - nothing wakes while neither is armed.
    """

    def __init__(self, state, link=lambda: None, lamp=None, pulse=None, read=None,
                 clock=time.monotonic, wall=_dt.datetime.now, epoch=time.time):
        self.state = state
        self.link = link
        self.lamp = lamp or (lambda ip, cmds: lamp_call(ip, cmds))
        self.pulse = pulse or alert_lamp
        self.read = read or read_lamp
        self.clock, self.wall, self.epoch = clock, wall, epoch
        self.focus = Focus()
        self.saved = None                       # the lamp's look before Focus took it
        self.plan = AlarmPlan()
        self.ramp_until = 0.0                   # epoch seconds; the sunrise is running
        self.busy_until = 0.0                   # one alert at a time
        self._lock = threading.RLock()
        self._focus_wake = threading.Event()
        self._alarm_wake = threading.Event()
        self._threads = []

    def start(self):
        for target in (self._focus_loop, self._alarm_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def changed(self):
        """A setting moved (the alarm time or switch): look again."""
        self._alarm_wake.set()

    # -- what the page reads ---------------------------------------------
    def view(self):
        with self._lock:
            ramping = self.epoch() < self.ramp_until
            return {"focus": self.focus.view(self.clock()), "sunrise_ramping": ramping}

    def _ip(self):
        snap = self.state.snapshot()
        return snap["ip"], snap["running"]

    # -- alerts ------------------------------------------------------------
    def alert(self, rgb, pulses=2, speed="normal"):
        """Pulse once. -> (code, payload). With a session running the flow
        goes out on the session's own connection and its frames wait until
        the pulse has ended; otherwise one short plain connection."""
        flow = alert_flow(rgb, pulses, speed)
        with self._lock:
            now = self.clock()
            if now < self.busy_until:
                return 409, {"ok": False, "error": "a pulse is still running"}
            link = self.link()
            if link is not None and link.flow([flow[0], 0, flow[1]], flow[2]):
                via = "session"
            else:
                ip, running = self._ip()
                if not ip:
                    return 409, {"ok": False, "error": "no bulb address"}
                if running:
                    return 409, {"ok": False, "error": "the session is connecting"}
                try:
                    self.pulse(ip, flow)
                except Exception as exc:
                    return 502, {"ok": False, "error": f"the bulb did not take it: {exc}"}
                via = "lamp"
            self.busy_until = now + flow[2] / 1000.0
        return 200, {"ok": True, "via": via, "ms": flow[2]}

    # -- Focus -------------------------------------------------------------
    def _lengths(self):
        snap = self.state.snapshot()
        return snap["focus_work"] * 60, snap["focus_break"] * 60

    def focus_op(self, op):
        """start / pause / reset. -> (code, payload)."""
        with self._lock:
            ip, running = self._ip()
            if op == "start":
                if running:
                    return 409, {"ok": False, "error": "Focus runs on a stopped desk"}
                if not ip:
                    return 409, {"ok": False, "error": "no bulb address"}
                work_s, _ = self._lengths()
                fresh = not self.focus.active
                if fresh:
                    try:
                        self.saved = self.read(ip)
                        count, expr, _ = flow_then_white(
                            alert_flow(TO_WORK, 1, "normal"), *WORK_WHITE)
                        # set_scene: a lamp that is off comes on for the block
                        self.lamp(ip, [("set_scene", ["cf", count, 1, expr])])
                    except Exception as exc:
                        self.saved = None
                        return 502, {"ok": False, "error": f"could not reach the bulb: {exc}"}
                self.focus.start(self.clock(), work_s)
            elif op == "pause":
                self.focus.pause(self.clock())
            elif op == "reset":
                self.end_focus()
            else:
                return 400, {"ok": False, "error": "op must be start, pause or reset"}
            self._focus_wake.set()
        return 200, {"ok": True, "focus": self.focus.view(self.clock())}

    def end_focus(self):
        """Stop Focus and put the lamp back as it was before. Called by
        Reset and by a session's Start, before the session reads the lamp."""
        with self._lock:
            was = self.focus.active
            self.focus.reset()
            saved, self.saved = self.saved, None
            self._focus_wake.set()
            ip, _ = self._ip()
        if was and saved and ip:
            try:
                self.lamp(ip, restore_commands(saved))
            except Exception:
                pass

    def _focus_loop(self):
        while True:
            with self._lock:
                wait = self.focus.wait(self.clock())
            self._focus_wake.wait(wait)
            self._focus_wake.clear()
            with self._lock:
                now = self.clock()
                if not self.focus.due(now):
                    continue
                work_s, break_s = self._lengths()
                phase = self.focus.advance(now, work_s, break_s)
                ip, running = self._ip()
            if not ip or running:
                continue
            ct, bright = WORK_WHITE if phase == "work" else BREAK_WHITE
            colour = TO_WORK if phase == "work" else TO_BREAK
            count, expr, _ = flow_then_white(alert_flow(colour, 2, "normal"), ct, bright)
            # start_cf, not set_scene: a lamp switched off mid-block stays off
            try:
                self.lamp(ip, [("start_cf", [count, 1, expr])])
            except Exception:
                pass

    # -- sunrise -----------------------------------------------------------
    def sunrise_stop(self):
        with self._lock:
            ramping = self.epoch() < self.ramp_until
            self.ramp_until = 0.0
            ip, running = self._ip()
        if ramping and ip and not running:
            try:
                self.lamp(ip, [("stop_cf", [])])
            except Exception as exc:
                return 502, {"ok": False, "error": f"could not reach the bulb: {exc}"}
        return 200, {"ok": True}

    def session_started(self):
        """A session takes the lamp: Focus ends (lamp put back first) and a
        running sunrise is simply overwritten."""
        self.end_focus()
        with self._lock:
            self.ramp_until = 0.0

    def fire_sunrise(self, total_ms=SUNRISE_MS):
        """Start the ramp now, unless a session or Focus has the lamp.
        -> True if it was sent."""
        with self._lock:
            ip, running = self._ip()
            if not ip or running or self.focus.active:
                return False
        count, expr, ms = sunrise_flow(total_ms)
        try:
            self.lamp(ip, [("set_scene", ["cf", count, 1, expr])])
        except Exception:
            return False
        with self._lock:
            self.ramp_until = self.epoch() + ms / 1000.0
        return True

    def _alarm_loop(self):
        while True:
            snap = self.state.snapshot()
            fire, wait = self.plan.step(self.wall(), snap["sunrise_on"], snap["sunrise_at"])
            if fire:
                self.fire_sunrise()
            self._alarm_wake.wait(wait)
            self._alarm_wake.clear()
