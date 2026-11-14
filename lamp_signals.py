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
from collections import deque
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
ALERTS_PER_MIN = 6           # the desk is unauthenticated: no strobe on demand
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


def with_tail(flow, tail):
    """The pulse, then `tail` [(ms, mode, value, bright)] it carries on with
    (end action 1): what the lamp was in the middle of, so a pulse cannot end
    a sunrise or leave a Focus phase on the wrong white. The bulb runs one
    flow at a time, so recovering with action 0 would drop it."""
    count, expr, ms = flow
    steps = [f"{d},{m},{v},{b}" for d, m, v, b in tail]
    assert count + len(steps) <= MAX_STEPS
    return count + len(steps), ",".join([expr] + steps), ms + sum(t[0] for t in tail)


def alert_lamp(ip, flow, tail=None, connect=socket.create_connection, timeout=2.0):
    """One short plain connection: ask whether the lamp is on, then pulse.
    A lamp that is off ignores start_cf, so it gets `set_scene cf`, which
    lights it for the pulse and leaves it off again with its look intact.
    A lit lamp with a `tail` pulses and then carries on with it."""
    count, expr, _ = flow
    sock = connect((ip, LAMP_PORT), timeout)
    try:
        (power,) = lamp_call(ip, [("get_prop", ["power"])], sock=sock, timeout=timeout)
        if power and power[0] == "on" and tail:
            n, e, _ = with_tail(flow, tail)
            cmd = ("start_cf", [n, 1, e])
        elif power and power[0] == "on":
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
RETRY_S = 15               # a sunrise that could not be sent is tried again


def sunrise_flow(total_ms=SUNRISE_MS):
    """-> (count, expression, ms): 1% at 1700 K, then up to 100% at 6500 K
    over `total_ms`, in a handful of steps the bulb interpolates. Sent with
    end action 1 the lamp stays at full."""
    steps = ["50,2,1700,1"]
    for share, ct, bright in SUNRISE_STEPS:
        steps.append(f"{max(50, int(round(total_ms * share)))},2,{ct},{bright}")
    return len(steps), ",".join(steps), total_ms + 50


BRIDGE_MS = 300             # after a pulse mid-ramp: back to where the ramp is


def sunrise_rest(at_ms, total_ms=SUNRISE_MS):
    """The rest of a sunrise from `at_ms` after it was sent: a short step to
    where the ramp would be by then, then what is left of its schedule, so
    it still ends when it would have. At most 1 + len(SUNRISE_STEPS) steps."""
    t, frm, segs = 50, (1700, 1), []
    for share, ct, bright in SUNRISE_STEPS:
        d = max(50, int(round(total_ms * share)))
        segs.append((t, t + d, frm, (ct, bright)))
        t, frm = t + d, (ct, bright)
    for start, end, a, b in segs:
        if at_ms < end:
            k = max(0.0, (at_ms - start) / (end - start))
            here = (int(round(a[0] + k * (b[0] - a[0]))), max(1, int(round(a[1] + k * (b[1] - a[1])))))
            tail = [(BRIDGE_MS, 2, *here)]
            tail.append((max(50, end - at_ms - BRIDGE_MS), 2, *b))
            tail += [(e - s_, 2, *bb) for s_, e, _, bb in segs[segs.index((start, end, a, b)) + 1:]]
            return tail
    return [(BRIDGE_MS, 2, *segs[-1][3])]


def _exists(t):
    """False for a local time the clock skips (the spring-forward hour)."""
    return _dt.datetime.fromtimestamp(t.timestamp()) == t


def next_fire(now, at_minutes):
    """The first moment strictly after `now` (a naive local datetime) whose
    clock reads `at_minutes` past midnight. A time the clock skips at the
    spring change fires on the first minute that exists after it (02:30 ->
    03:00); in the autumn the repeated hour fires once, on its first pass."""
    h, m = divmod(int(at_minutes) % 1440, 60)
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t = (now + _dt.timedelta(days=1)).replace(hour=h, minute=m, second=0,
                                                   microsecond=0)
    for _ in range(180):
        if _exists(t):
            break
        t += _dt.timedelta(minutes=1)
    return t


class AlarmPlan:
    """When the alarm fires and how long to sleep, from the wall clock.

    `step(now, on, at)` -> (fire, wait). fire is True from the set time
    until it has been sent (`done(now)`) or LATE_OK has passed, every
    RETRY_S: a Mac that just woke may not have its Wi-Fi back, and a session
    or Focus holding the lamp wins until it lets go. wait = seconds until the
    next look (None = nothing armed, sleep until a change), capped at CHECK_S
    because macOS stops the monotonic clock while the machine sleeps.

    The first arming in a process (`catch_up`) also takes a time that passed
    less than LATE_OK ago, so a desk restarted just after the time still
    starts it; a time the user sets is always the next one ahead."""

    def __init__(self, catch_up=True):
        self.key = None
        self.target = None
        self.catch_up = catch_up

    def step(self, now, on, at):
        if not on:
            self.key = self.target = None
            self.catch_up = False
            return False, None
        if self.key != at or self.target is None:
            first, self.catch_up = self.catch_up and self.key is None, False
            self.key = at
            self.target = next_fire(now - _dt.timedelta(seconds=LATE_OK) if first else now, at)
        left = (self.target - now).total_seconds()
        if left > 0:
            return False, min(left, CHECK_S)
        if -left <= LATE_OK:
            return True, RETRY_S
        self.target = next_fire(now, at)                  # missed for good
        return False, min((self.target - now).total_seconds(), CHECK_S)

    def done(self, now):
        """The sunrise went out: the next one is tomorrow's."""
        if self.target is not None:
            self.target = next_fire(max(now, self.target), self.key)


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
    or None, `parked(timeout)` waits up to `timeout` s for the engine to be
    parked (no session, its restore done) and says whether it is. The lamp is
    reached through `lamp(ip, commands)` and `pulse(ip, flow, tail)` (plain
    short connections), swappable for tests.

    Two locks: `_lock` guards the state and is never held across the network
    (the page's poll takes it); `_lamp` serialises every plain exchange with
    the lamp, and a session's Start takes it too (`claim_lamp`), so nothing
    plain goes out while a session starts, runs or restores.

    Focus and the sunrise run only on a stopped desk: their whites would be
    painted over by a session's frames. Each has one thread that sleeps on
    an Event until its next change - nothing wakes while neither is armed.
    """

    def __init__(self, state, link=lambda: None, parked=None, lamp=None, pulse=None,
                 read=None, clock=time.monotonic, wall=_dt.datetime.now, epoch=time.time,
                 lamp_lock=None):
        self.state = state
        self.link = link
        self.parked = parked or (lambda timeout=0.0: True)
        self.lamp = lamp or (lambda ip, cmds: lamp_call(ip, cmds))
        self.pulse = pulse or alert_lamp
        self.read = read or read_lamp
        self.clock, self.wall, self.epoch = clock, wall, epoch
        self.focus = Focus()
        self.saved = None                       # the lamp's look before Focus took it
        self.plan = AlarmPlan()
        self.ramp_start = self.ramp_until = 0.0  # epoch seconds; the sunrise is running
        self.ramp_ms = SUNRISE_MS
        self.busy_until = 0.0                   # one alert at a time
        self.recent = deque()                   # alert times, for ALERTS_PER_MIN
        self._lock = threading.Lock()
        # the desk's lamp lock (re-entrant), shared with lamp mode, the sleep
        # timer and a session's open and close
        self._lamp = lamp_lock or threading.RLock()
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

    def _free(self, wait=0.0):
        """Called holding `_lamp`: no session running, starting or restoring."""
        return not self.state.snapshot()["running"] and self.parked(wait)

    def _settle(self, wait):
        """Before taking `_lamp`: give a session that is stopping up to
        `wait` s to finish its restore. Not under the lock - the session's
        close needs it."""
        if not self.state.snapshot()["running"]:
            self.parked(wait)

    def holding(self):
        """Focus or the sunrise has the lamp: Follow the day stands back."""
        with self._lock:
            return self.focus.active or self.epoch() < self.ramp_until

    # -- alerts ------------------------------------------------------------
    def _tail(self, pulse_ms):
        """What the lamp must carry on with after a pulse, or None."""
        with self._lock:
            now = self.epoch()
            if now < self.ramp_until:
                at = (now - self.ramp_start) * 1000.0 + pulse_ms
                return sunrise_rest(int(at), self.ramp_ms)
            if self.focus.active:
                ct, bright = WORK_WHITE if self.focus.phase == "work" else BREAK_WHITE
                return [(600, 2, ct, bright)]
        return None

    def alert(self, rgb, pulses=2, speed="normal"):
        """Pulse once. -> (code, payload). With a session running the flow
        goes out on the session's own connection and its frames wait until
        the pulse has ended; otherwise one short plain connection, once no
        session is starting or restoring."""
        flow = alert_flow(rgb, pulses, speed)
        with self._lock:
            now = self.clock()
            if now < self.busy_until:
                return 409, {"ok": False, "error": "a pulse is still running"}
            while self.recent and self.recent[0] <= now - 60:
                self.recent.popleft()
            if len(self.recent) >= ALERTS_PER_MIN:
                return 429, {"ok": False, "error": f"at most {ALERTS_PER_MIN} alerts a minute"}
            self.recent.append(now)
            self.busy_until = now + flow[2] / 1000.0

        def refuse(code, error):
            with self._lock:
                self.busy_until = 0.0
            return code, {"ok": False, "error": error}

        link = self.link()
        if link is not None and link.flow([flow[0], 0, flow[1]], flow[2]):
            return 200, {"ok": True, "via": "session", "ms": flow[2]}
        self._settle(3.0)
        with self._lamp:
            ip, running = self._ip()
            if not ip:
                return refuse(409, "no bulb address")
            if not self._free():
                return refuse(409, "the session has the lamp")
            try:
                self.pulse(ip, flow, self._tail(flow[2]))
            except Exception as exc:
                return refuse(502, f"the bulb did not take it: {exc}")
        return 200, {"ok": True, "via": "lamp", "ms": flow[2]}

    # -- Focus -------------------------------------------------------------
    def _lengths(self):
        snap = self.state.snapshot()
        return snap["focus_work"] * 60, snap["focus_break"] * 60

    def focus_op(self, op):
        """start / pause / reset. -> (code, payload)."""
        if op == "start":
            self._settle(3.0)
            with self._lamp:
                ip, running = self._ip()
                if not ip:
                    return 409, {"ok": False, "error": "no bulb address"}
                if not self._free():
                    return 409, {"ok": False, "error": "Focus runs on a stopped desk"}
                with self._lock:
                    fresh = not self.focus.active
                if fresh:
                    try:
                        saved = self.read(ip)
                        count, expr, _ = flow_then_white(
                            alert_flow(TO_WORK, 1, "normal"), *WORK_WHITE)
                        # set_scene: a lamp that is off comes on for the block
                        self.lamp(ip, [("set_scene", ["cf", count, 1, expr])])
                    except Exception as exc:
                        return 502, {"ok": False, "error": f"could not reach the bulb: {exc}"}
                work_s, _ = self._lengths()
                with self._lock:
                    if fresh:
                        self.saved = saved
                    self.focus.start(self.clock(), work_s)
        elif op == "pause":
            with self._lock:
                self.focus.pause(self.clock())
        elif op == "reset":
            self.end_focus()
        else:
            return 400, {"ok": False, "error": "op must be start, pause or reset"}
        self._focus_wake.set()
        with self._lock:
            return 200, {"ok": True, "focus": self.focus.view(self.clock())}

    def end_focus(self):
        """Stop Focus and put the lamp back as it was before."""
        with self._lamp:
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
            with self._lock:
                phase = self.focus.advance(now, work_s, break_s)
            ct, bright = WORK_WHITE if phase == "work" else BREAK_WHITE
            colour = TO_WORK if phase == "work" else TO_BREAK
            count, expr, _ = flow_then_white(alert_flow(colour, 2, "normal"), ct, bright)
            with self._lamp:
                ip, _ = self._ip()
                if not ip or not self._free(0.0):
                    continue
                # start_cf, not set_scene: a lamp switched off mid-block stays off
                try:
                    self.lamp(ip, [("start_cf", [count, 1, expr])])
                except Exception:
                    pass

    # -- the session takes the lamp ---------------------------------------
    def claim_lamp(self, start=lambda: None):
        """A session's Start: Focus ends (its restore lands first), a running
        sunrise is dropped, and `start()` (which sets `running`) is called
        holding the lamp lock, so no plain exchange can slip in between."""
        with self._lamp:
            self.end_focus()
            with self._lock:
                self.ramp_until = 0.0
            start()

    # -- sunrise -----------------------------------------------------------
    def sunrise_stop(self):
        with self._lamp:
            with self._lock:
                ramping = self.epoch() < self.ramp_until
                self.ramp_until = 0.0
            ip, running = self._ip()
            if ramping and ip and self._free(0.0):
                try:
                    self.lamp(ip, [("stop_cf", [])])
                except Exception as exc:
                    return 502, {"ok": False, "error": f"could not reach the bulb: {exc}"}
        return 200, {"ok": True}

    def fire_sunrise(self, total_ms=SUNRISE_MS):
        """Start the ramp now, unless a session or Focus has the lamp.
        -> True if it was sent."""
        count, expr, ms = sunrise_flow(total_ms)
        with self._lamp:
            ip, _ = self._ip()
            with self._lock:
                focus = self.focus.active
            if not ip or focus or not self._free(0.0):
                return False
            try:
                self.lamp(ip, [("set_scene", ["cf", count, 1, expr])])
            except Exception:
                return False
            with self._lock:
                self.ramp_start, self.ramp_ms = self.epoch(), total_ms
                self.ramp_until = self.ramp_start + ms / 1000.0
        return True

    def _alarm_loop(self):
        while True:
            snap = self.state.snapshot()
            fire, wait = self.plan.step(self.wall(), snap["sunrise_on"], snap["sunrise_at"])
            if fire and self.fire_sunrise():
                self.plan.done(self.wall())
                continue
            self._alarm_wake.wait(wait)
            self._alarm_wake.clear()
