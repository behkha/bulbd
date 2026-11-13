"""Sender with a fake bulb, and the exact commands yeelight puts on the wire."""
import json
import threading
import time

import pytest

from music_bulb import Sender

yeelight = pytest.importorskip("yeelight")


# ------------------------------------------------------------------- Sender
class FakeBulb:
    def __init__(self, fail_first=0):
        self.calls, self.music, self.fail = [], [], fail_first
        self.gate = None                               # an Event the first call waits on
        self.entered = threading.Event()
        self.seen = threading.Event()

    def set_hsv(self, h, s, v, effect, duration):
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(2)
            self.gate = None
        if self.fail:
            self.fail -= 1
            raise OSError("closed")
        self.calls.append(((h, s, v), effect, duration))
        self.seen.set()

    def stop_music(self):
        self.music.append("stop")

    def start_music(self):
        self.music.append("start")


@pytest.fixture
def sender():
    made = []

    def make(bulb):
        s = Sender(bulb); s.start(); made.append(s)
        return s

    yield make
    for s in made:
        s.stop(); s.join(2)


def test_sender_passes_smooth_and_sudden_frames(sender):
    bulb = FakeBulb(); s = sender(bulb)
    s.put((26, 92, 80), False, 40)
    assert bulb.seen.wait(2); bulb.seen.clear()
    s.put((0, 100, 100), True, 40)
    assert bulb.seen.wait(2)
    assert bulb.calls == [((26, 92, 80), "smooth", 40), ((0, 100, 100), "sudden", 0)]


def test_sender_delivers_only_the_latest_frame(sender):
    bulb = FakeBulb(); bulb.gate = threading.Event(); s = sender(bulb)
    s.put((1, 1, 1), False)
    assert bulb.entered.wait(2)                        # busy inside the first send
    for i in range(2, 8):
        s.put((i, i, i), False)
    bulb.seen.clear(); bulb.gate.set()
    assert bulb.seen.wait(2)
    time.sleep(0.05)
    assert [c[0] for c in bulb.calls] == [(1, 1, 1), (7, 7, 7)]


def test_sender_stops_promptly_when_idle(sender):
    s = sender(FakeBulb())
    time.sleep(0.05)
    t0 = time.monotonic(); s.stop(); s.join(2)
    assert not s.is_alive() and time.monotonic() - t0 < 0.5


def test_sender_stops_with_a_frame_pending(sender):
    bulb = FakeBulb(); s = sender(bulb)
    s.stop(); s.put((5, 5, 5), False)
    s.join(2)
    assert not s.is_alive()


def test_sender_reopens_music_mode_after_a_dropped_link_and_carries_on(sender, capsys):
    bulb = FakeBulb(fail_first=1); s = sender(bulb)
    s.put((1, 1, 1), False)
    deadline = time.monotonic() + 3
    while bulb.music != ["stop", "start"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert bulb.music == ["stop", "start"]
    s.put((2, 2, 2), False)
    assert bulb.seen.wait(2)
    assert bulb.calls[-1][0] == (2, 2, 2)
    assert "link dropped" in capsys.readouterr().err


# --------------------------------------------------------------------- wire
class FakeSock:
    def __init__(self):
        self.sent = []

    def send(self, data):
        if data.strip():
            self.sent.append(json.loads(data.decode().strip()))
        return len(data)

    def close(self):
        pass


@pytest.fixture
def wire():
    b = yeelight.Bulb("192.0.2.1", effect="sudden", duration=0, auto_on=True)
    sock = FakeSock()
    b._music_mode = True
    b._Bulb__socket = sock
    return b, sock


@pytest.mark.parametrize("hsv, kw, rgb, bright, dur", [
    ((26, 92, 80), dict(effect="smooth", duration=40), 0xff7a14, 80, 40),      # music at 25 fps
    ((26, 92, 80), dict(effect="smooth", duration=33), 0xff7a14, 80, 33),      # music at 30 fps
    ((216, 68, 28), dict(effect="smooth", duration=100), 0x5295ff, 28, 100),   # screen at 10 fps
    ((0, 100, 100), dict(effect="sudden", duration=0), 0xff0000, 100, 50),     # hit or scene cut
    ((359, 100, 50), dict(effect="smooth", duration=100), 0xff0000, 50, 100),
    ((180, 100, 50), dict(effect="smooth", duration=100), 0x00fdff, 50, 100),
])
def test_set_hsv_goes_out_as_one_start_cf(wire, hsv, kw, rgb, bright, dur):
    b, sock = wire
    b.set_hsv(*hsv, **kw)
    assert len(sock.sent) == 1
    msg = sock.sent[0]
    assert msg["method"] == "start_cf"
    assert msg["params"] == [1, 1, f"{dur}, 1, {rgb}, {bright}"]


def test_sudden_is_a_minimum_length_ramp_not_a_set(wire):
    """yeelight turns 'sudden' into a 50 ms flow; the colour is still sent as one command."""
    b, sock = wire
    b.set_hsv(10, 50, 50, effect="sudden", duration=0)
    assert [m["method"] for m in sock.sent] == ["start_cf"]
