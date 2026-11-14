"""What the lamp does on its own: after Stop, after the desk quits, at the switch.

Everything here is a list of LAN API commands for lamp_call() or for
release_bulb(then=...); nothing opens a connection. Measured on the color5
(2026-11-14):

- With the lamp off, start_cf answers "ok" and does nothing, and cron_add is
  refused (-5006 "illegal request"). So a timer needs the lamp on.
- cron_add [0, minutes] is accepted up to at least 1441; cron_get [0] answers
  [{"type": 0, "delay": minutes, "mix": 0}] and get_prop delayoff the same
  minutes; cron_del [0] clears it. It survives set_bright and start_cf, so a
  music session's frames do not cancel it.
- A flow step may be at least 2**31 ms. End action 2 switches the lamp off.
- stop_cf RUNS the end action: on a sleep flow it switches the lamp off. A
  one-step start_cf (action 1) replaces a flow without its end action, so
  that is how a fade is cancelled.
"""
import colorsys
import math
import threading
import time

IDLE_LOOKS = ("restore", "candle", "drift", "breathe")
SLEEP_STEPS = (15, 30, 60, 90, 120)            # minutes, the desk's Sleep row
SLEEP_MAX = 1440                               # cron_add took 1441; a day is plenty
SLEEP_FADE_S = 600                             # the last ten minutes dim to 1%

CANDLE_CT = 2000
CANDLE = ((800, 50), (900, 32), (1200, 62), (700, 44), (1300, 58), (2100, 38),
          (1000, 55), (600, 46), (900, 60))    # (ms, brightness), a slow uneven flicker
DRIFT_STEP_MS = 12000                          # one anchor to the next
DRIFT_BRIGHT = 60
BREATHE_CT = 3200
BREATHE = ((4000, 55), (4000, 8))              # in and out, 8 s a breath

MODE_RGB, MODE_CT = 1, 2


def rgb_int(hue, sat, val=100):
    """Hue 0-359, saturation and value 0-100 -> the API's 0xRRGGBB integer."""
    r, g, b = colorsys.hsv_to_rgb((hue % 360) / 360, max(0, min(sat, 100)) / 100,
                                  max(0, min(val, 100)) / 100)
    return (round(r * 255) << 16) | (round(g * 255) << 8) | round(b * 255)


def flow(steps, count=0, action=0):
    """("start_cf", params) for [(ms, mode, value, bright), ...]. count 0 runs
    for ever; action 0 = back to before, 1 = stay, 2 = off. The bulb wants
    each step at least 50 ms and brightness 1-100."""
    expr = ", ".join(f"{max(50, int(ms))}, {mode}, {int(value)}, {max(1, min(int(b), 100))}"
                     for ms, mode, value, b in steps)
    return ("start_cf", [count, action, expr])


def candle_flow(scale=1.0):
    return flow([(ms, MODE_CT, CANDLE_CT, b * scale) for ms, b in CANDLE])


def drift_flow(anchors, bright=DRIFT_BRIGHT):
    """Ramp from one palette anchor to the next and round again. Each step
    is a ramp to the anchor, so the hue never stops moving."""
    anchors = [tuple(a) for a in anchors][:8] or [(30, 80)]
    if len(anchors) == 1:
        anchors.append(anchors[0])
    return flow([(DRIFT_STEP_MS, MODE_RGB, rgb_int(h, s), bright) for h, s in anchors])


def breathe_flow():
    return flow([(ms, MODE_CT, BREATHE_CT, b) for ms, b in BREATHE])


def idle_commands(name, anchors=None):
    """What Stop sends instead of the restore: [] for "restore" (or anything
    unknown), else one forever-flow."""
    if name == "candle":
        return [candle_flow()]
    if name == "drift":
        return [drift_flow(anchors or [])]
    if name == "breathe":
        return [breathe_flow()]
    return []


def look_step(saved):
    """[mode, value, bright] for a still look read with read_lamp(), or None.
    HSV becomes RGB: a flow has no HSV step."""
    if not saved or saved.get("power") != "on" or not saved.get("bright"):
        return None
    mode, bright = saved.get("color_mode"), saved["bright"]
    if mode == 1 and saved.get("rgb") is not None:
        return [MODE_RGB, int(saved["rgb"]), int(bright)]
    if mode == 2 and saved.get("ct"):
        return [MODE_CT, int(saved["ct"]), int(bright)]
    if mode == 3 and saved.get("hue") is not None:
        return [MODE_RGB, rgb_int(saved["hue"], saved.get("sat") or 0), int(bright)]
    return None


def fade_flow(step, seconds, fade_s=SLEEP_FADE_S):
    """Hold `step` (the lamp's look), dim it to 1% over the last `fade_s`,
    then switch off (action 2). The bulb runs it with nothing connected."""
    mode, value, bright = step
    total = max(1, int(seconds * 1000))
    fade = min(total, int(fade_s * 1000))
    steps = []
    if total - fade >= 50:
        steps.append((total - fade, mode, value, bright))
    steps.append((fade, mode, value, 1))
    return flow(steps, count=len(steps), action=2)


def sleep_commands(minutes, step=None):
    """Set the bulb's off timer, plus the fade when the lamp shows a still
    look. Without `step` (a session or an idle look is running) the timer
    alone switches the lamp off; the desk dims a session itself."""
    cmds = [("cron_add", [0, int(minutes)])]
    if step:
        cmds.append(fade_flow(step, minutes * 60))
    return cmds


def cancel_commands(step=None):
    """Clear the timer and, if a fade was sent, put its look back with a
    one-step flow (stop_cf would run the fade's end action: off)."""
    cmds = [("cron_del", [0])]
    if step:
        mode, value, bright = step
        cmds.append(flow([(300, mode, value, bright)], count=1, action=1))
    return cmds


def clean_minutes(v):
    """Minutes 1..SLEEP_MAX as an int, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    v = int(round(v))
    return v if 1 <= v <= SLEEP_MAX else None


def clean_sleep(v, now=None):
    """The saved timer {at, step} if it is still ahead, else None."""
    now = time.time() if now is None else now
    if not isinstance(v, dict):
        return None
    at = v.get("at")
    if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at) \
            or not now < at <= now + SLEEP_MAX * 60 + 60:
        return None
    step = v.get("step")
    if not (isinstance(step, (list, tuple)) and len(step) == 3
            and all(isinstance(x, int) and not isinstance(x, bool) for x in step)
            and step[0] in (MODE_RGB, MODE_CT) and 1 <= step[2] <= 100):
        step = None
    return {"at": float(at), "step": list(step) if step else None}


def dim_factor(left_s, fade_s=SLEEP_FADE_S):
    """1.0 until the last `fade_s` seconds, then down to 0.01 at zero."""
    if left_s is None or left_s >= fade_s:
        return 1.0
    return max(0.01, left_s / fade_s)


def dim_level(v, factor):
    """One brightness number scaled, never below 1 (0 stays 0)."""
    if factor >= 1.0 or not v:
        return v
    return max(1, round(v * factor))


def dim_frame(frame, factor):
    """A frame with its brightness scaled: an (h, s, v) tuple or a White."""
    if factor >= 1.0:
        return frame
    if hasattr(frame, "ct") and hasattr(frame, "bright"):
        return type(frame)(frame.ct, dim_level(frame.bright, factor))
    h, s, v = frame
    return (h, s, dim_level(v, factor))


class Dimmed:
    """A Link whose frames are dimmed by `factor()`; everything else passes
    through. The desk hands this to its loops while a session runs.

    The loops send only when the light changes (same_light), so on a still
    picture or in silence nothing would carry the fade: refresh() re-sends
    the last frame, dimmed, when the loops have been quiet for REFRESH s and
    the dimmed level moved. Its ramp is REFRESH long, so the steps join up."""

    REFRESH = 2.0

    def __init__(self, link, factor, clock=time.monotonic):
        object.__setattr__(self, "_link", link)
        object.__setattr__(self, "_factor", factor)
        object.__setattr__(self, "_clock", clock)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_last", None)     # (frame, args, kwargs) as the loop sent it
        object.__setattr__(self, "_sent", None)     # the dimmed frame last on the wire
        object.__setattr__(self, "_at", 0.0)

    def put(self, frame, *a, **kw):
        with self._lock:
            out = dim_frame(frame, self._factor())
            object.__setattr__(self, "_last", (frame, a, kw))
            object.__setattr__(self, "_sent", out)
            object.__setattr__(self, "_at", self._clock())
            self._link.put(out, *a, **kw)

    def refresh(self):
        """-> True if the last frame was re-sent at the current dim."""
        with self._lock:
            if self._last is None or self._clock() - self._at < self.REFRESH:
                return False
            out = dim_frame(self._last[0], self._factor())
            if out == self._sent:
                return False
            object.__setattr__(self, "_sent", out)
            object.__setattr__(self, "_at", self._clock())
            self._link.put(out, False, int(self.REFRESH * 1000))
            return True

    def __getattr__(self, name):
        return getattr(self._link, name)

    def __setattr__(self, name, value):
        setattr(self._link, name, value)
