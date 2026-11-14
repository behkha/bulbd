"""The desk's music loop with Drops and Bulb clock on: the real
Engine._audio_loop on a recorded signal, its clock run by the input the way
record() paces it, and every send kept."""
import time

import numpy as np
import pytest

import music_bulb_dash as dash
from music_bulb import BLOCK, SAMPLERATE, Flow
from test_drops import AGAIN, GROOVE, pad, roll, song

DT = BLOCK / SAMPLERATE


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Mic:
    """A capture device that plays `pcm` once, a block per record(), moving
    the clock on by a block; at the end it stops the desk."""
    name = "test input"

    def __init__(self, pcm, clock, st):
        self.pcm, self.clock, self.st, self.i = pcm, clock, st, 0

    def recorder(self, samplerate, channels):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def record(self, numframes):
        block = self.pcm[self.i:self.i + numframes]
        self.i += numframes
        self.clock.t += DT
        if len(block) < numframes:
            self.st.set(running=False)
            return np.zeros((numframes, 2), np.float32)
        return block


class Sent:
    lost = False

    def __init__(self, clock):
        self.clock, self.rows = clock, []

    def put(self, frame, sudden, duration=40):
        self.rows.append((self.clock.t, frame, sudden, duration))


def play(monkeypatch, pcm, **cfg):
    clock = Clock()
    monkeypatch.setattr(dash.time, "monotonic", clock)
    st = dash.State({"ip": "192.168.1.2", "mode": "music", **cfg})
    st.set(running=True)
    mic = Mic(pcm, clock, st)
    monkeypatch.setattr(dash, "open_recorder", lambda name: mic)
    sent = Sent(clock)
    dash.Engine(st)._audio_loop(st, sent)
    t0 = 1000.0
    return [(t - t0, f, s, d) for t, f, s, d in sent.rows]


def flows(rows, count=None):
    return [(t, f) for t, f, *_ in rows if isinstance(f, Flow)
            and (count is None or f.count == count)]


def test_drops_on_swells_on_the_build_up_and_strobes_on_the_drop(monkeypatch):
    rows = play(monkeypatch, song(GROOVE, pad(4), roll(8), AGAIN), drops=True)
    swells = [t for t, f in flows(rows) if f.count == 1 and f.steps[0][0] == 2500]
    strobes = [t for t, f in flows(rows, 8)]
    assert len(swells) == 1 and 20 < swells[0] < 28
    assert len(strobes) == 1 and 28.0 <= strobes[0] < 28.2
    # nothing of the loop's own while the strobe runs; a soft frame after it
    end = strobes[0] + 4 / 3
    assert not [t for t, f, *_ in rows if strobes[0] < t < end - 1e-6]
    after = next(r for r in rows if r[0] >= end - 1e-6)
    assert after[3] >= dash.DropFx.HANDBACK_MS and after[2] is False


def test_drops_off_sends_no_effect(monkeypatch):
    rows = play(monkeypatch, song(GROOVE, pad(4), roll(8), AGAIN))
    assert not [f for _, f in flows(rows) if f.count != 2]


def test_bulb_clock_on_sends_each_predicted_beat_as_one_flow(monkeypatch):
    rows = play(monkeypatch, GROOVE, bulb_clock=True)
    beats = flows(rows, 2)
    assert len(beats) >= 10                          # 128 BPM, locked after a few s
    for t, f in beats:
        hold = f.steps[0][0] / 1000
        assert 0.048 <= hold <= 0.073
        inside = [r for r in rows if t < r[0] < t + hold + 0.05 - 1e-6]
        assert inside == []                          # the flow is the bulb's until it lands


def test_bulb_clock_off_sends_no_flow(monkeypatch):
    rows = play(monkeypatch, GROOVE)
    assert flows(rows) == []


# ------------------------------------------------------------ review round
class Sync(dash.BeatSync):
    made = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        Sync.made.append(self)


class LongStrobe(dash.DropFx):
    """A 12-flash strobe that lets go on the first tick a bulb-timed beat
    is handed over, half a second in: the hand-back tick sends a beat flow."""

    def __init__(self, *a, **k):
        super().__init__(flashes=12)
        self.started = None
        self.released = None

    def take(self, now, frame, on=True):
        flow = super().take(now, frame, on)
        if flow is not None:
            self.started, self.flow = now, flow
        return flow

    def holding(self, now):
        if (self.kind == "drop" and self.released is None and Sync.made[-1].hold
                and now > self.started + 0.5):
            self.until, self.released = now, now
        return super().holding(now)


def test_after_an_effect_a_beat_flow_holds_the_lamp_and_takes_the_hand_back(monkeypatch):
    Sync.made = []
    monkeypatch.setattr(dash, "BeatSync", Sync)
    monkeypatch.setattr(dash, "DropFx", LongStrobe)
    telemetry = []
    real_set = dash.State.set

    def spy(self, **kw):
        if "val" in kw:
            telemetry.append((dash.time.monotonic() - 1000.0, kw["hue"], kw["sat"], kw["val"]))
        real_set(self, **kw)
    monkeypatch.setattr(dash.State, "set", spy)
    # a groove that keeps going into the drop: the grid is trusted throughout
    from test_drops import track
    rows = play(monkeypatch, song(GROOVE, pad(6), track(128, secs=14)[0]),
                drops=True, bulb_clock=True)
    strobe = next((t, f) for t, f in flows(rows, 24))
    after = [r for r in rows if r[0] > strobe[0]]
    first = after[0]
    # 5: the first send after the effect is a beat flow holding the strobe's
    # own light, not the loop's newer colour
    assert isinstance(first[1], Flow) and first[1].count == 2
    assert first[1].steps[0][1:] == strobe[1].steps[1][1:]
    # 6: the hand-back went with that flow; no later frame gets the soft ramp
    assert not [r for r in after[1:] if not isinstance(r[1], Flow)
                and r[3] >= dash.DropFx.HANDBACK_MS]
    # 4: while the strobe ran the page showed the strobe's light
    t0, t1 = strobe[0], first[0]
    shown = {(h, s, v) for t, h, s, v in telemetry if t0 <= t < t1 - 1e-6}
    h, s, v = shown.pop()
    assert not shown
    assert (h, s, v)[2] == strobe[1].steps[1][3]


class Grab:
    backend, pushes, box = "sck", True, {"width": 200, "height": 130}

    def __init__(self, *a, **k):
        pass

    def close(self):
        pass


class Screen:
    """ScreenColorThread's contract, with a scene cut every 7th read."""
    reads = []

    def __init__(self, grab, engine, fps=10, on_frame=None):
        self.error, self.fps, self.n, self.hue = None, fps, 0, 30.0
        Screen.reads = []

    def start(self):
        pass

    def poke(self):
        pass

    def close(self):
        pass

    def read(self):
        self.n += 1
        cut = self.n % 7 == 0
        if cut:
            self.hue = (self.hue + 120) % 360
        Screen.reads.append((dash.time.monotonic() - 1000.0, cut))
        return self.hue, 90.0, 0.5, cut

    def white(self):
        return False, 6500

    def moved_at(self):
        return dash.time.monotonic()


@pytest.mark.parametrize("cfg", [dict(bulb_clock=True), dict(drops=True)],
                         ids=["bulb clock", "drops"])
def test_fusion_lands_every_scene_cut_even_when_a_flow_held_the_loop(monkeypatch, cfg):
    monkeypatch.setattr(dash, "Capture", Grab)
    monkeypatch.setattr(dash, "ScreenColorThread", Screen)
    monkeypatch.setattr(dash, "screen_permission", lambda: True)
    clock = Clock()
    monkeypatch.setattr(dash.time, "monotonic", clock)
    pcm = song(GROOVE, pad(4), roll(8), AGAIN) if cfg.get("drops") else GROOVE
    st = dash.State({"ip": "192.168.1.2", "mode": "fusion", **cfg})
    st.set(running=True)
    mic = Mic(pcm, clock, st)
    monkeypatch.setattr(dash, "open_recorder", lambda name: mic)
    sent = Sent(clock)
    dash.Engine(st)._fusion_loop(st, sent)
    rows = [(t - 1000.0, f, s, d) for t, f, s, d in sent.rows]
    cuts = [t for t, c in Screen.reads if c]
    held = 0
    for tc in cuts:
        nxt = next((r for r in rows if r[0] >= tc - 1e-9), None)
        if nxt is None:
            continue
        if nxt[0] > tc + 1e-9:
            held += 1                                   # the cut's own tick sent nothing
        f = nxt[1]
        landed = (isinstance(f, Flow) and f.count in (2, 8)) or nxt[2] is True
        assert landed, f"cut at {tc:.2f} went out smooth at {nxt[0]:.2f}: {nxt}"
    assert held > 0                                     # some cuts did fall in a held tick
