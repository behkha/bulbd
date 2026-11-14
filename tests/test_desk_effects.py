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
