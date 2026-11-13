"""Lamp restore: what is sent, and on which connection.

The bulb has only a handful of connection slots and frees a closed one
slowly, so the read before a session and the restore after it ride on the two
connections the session uses anyway. And this firmware ignores a command
until one more byte follows it, so every command is sent with a lone space.
"""
import json
import socket

import pytest

import music_bulb
from music_bulb import lamp_call, open_bulb, read_lamp, release_bulb, restore_commands

WARM = {"power": "on", "bright": 30, "color_mode": 2, "rgb": 16711680, "ct": 2700,
        "hue": 0, "sat": 100}


class ScriptedSock:
    """A socket that answers each command the way the bulb would."""

    def __init__(self, props=None, silent=False, pushes=()):
        self.props = props or ["on", "30", "2", "16711680", "2700", "0", "100"]
        self.silent = silent
        self.pushes = list(pushes)
        self.sent = []
        self.nudges = 0
        self.inbox = b""
        self.closed = False
        self.timeout = 5.0

    def sendall(self, data):
        if not data.strip():                           # the lone space
            self.nudges += 1
            return
        msg = json.loads(data.decode())
        self.sent.append((msg["method"], msg["params"]))
        if self.silent:
            return
        for push in self.pushes:                       # state pushes arrive first
            self.inbox += (json.dumps({"method": "props", "params": push}) + "\r\n").encode()
        self.pushes = []
        result = self.props if msg["method"] == "get_prop" else ["ok"]
        self.inbox += (json.dumps({"id": msg["id"], "result": result}) + "\r\n").encode()

    send = sendall

    def recv(self, n):
        if not self.inbox:
            raise socket.timeout("timed out")
        out, self.inbox = self.inbox[:n], self.inbox[n:]
        return out

    def settimeout(self, t):
        self.timeout = t

    def gettimeout(self):
        return self.timeout

    def close(self):
        self.closed = True


def no_new_connection(*a, **k):
    raise AssertionError("opened a connection of its own")


# ---------------------------------------------------------------- commands
def test_white_lamp_is_restored_as_colour_temperature():
    assert restore_commands(WARM) == [("set_ct_abx", [2700, "smooth", 300]),
                                      ("set_bright", [30, "smooth", 300])]


def test_colour_lamp_is_restored_as_rgb():
    saved = dict(WARM, color_mode=1, rgb=0x3366FF, bright=80)
    assert restore_commands(saved) == [("set_rgb", [0x3366FF, "smooth", 300]),
                                       ("set_bright", [80, "smooth", 300])]


def test_hsv_lamp_is_restored_as_hsv():
    saved = dict(WARM, color_mode=3, hue=200, sat=70)
    assert restore_commands(saved)[0] == ("set_hsv", [200, 70, "smooth", 300])


def test_a_lamp_that_was_off_gets_its_look_back_and_then_goes_off():
    cmds = restore_commands(dict(WARM, power="off"))
    assert cmds[-1] == ("set_power", ["off", "smooth", 300])
    assert cmds[0][0] == "set_ct_abx"


def test_out_of_range_values_are_clamped():
    cmds = dict(restore_commands(dict(WARM, ct=9000, bright=250)))
    assert cmds["set_ct_abx"][0] == 6500 and cmds["set_bright"][0] == 100


# ---------------------------------------------------------------- lamp_call
def test_read_lamp_parses_the_bulb_answer():
    assert read_lamp("x", sock=ScriptedSock()) == WARM


def test_a_state_push_is_not_taken_for_the_answer():
    sock = ScriptedSock(pushes=[{"bright": 99}])
    assert read_lamp("x", sock=sock)["bright"] == 30


def test_every_command_is_followed_by_the_byte_the_firmware_waits_for():
    sock = ScriptedSock()
    lamp_call("x", [("set_ct_abx", [2700]), ("set_bright", [30])], sock=sock)
    assert len(sock.sent) == 2 and sock.nudges == 2


def test_a_borrowed_socket_is_left_open_with_its_timeout():
    sock = ScriptedSock()
    lamp_call("x", [("set_bright", [30])], sock=sock, connect=no_new_connection)
    assert not sock.closed and sock.timeout == 5.0


def test_its_own_socket_is_always_closed():
    sock = ScriptedSock(silent=True)
    with pytest.raises(Exception):
        lamp_call("x", [("set_bright", [30])], timeout=0.05, connect=lambda *a: sock)
    assert sock.closed


def test_a_silent_bulb_times_out_instead_of_hanging():
    with pytest.raises((TimeoutError, socket.timeout)):
        lamp_call("x", [("get_prop", ["power"])], timeout=0.05, sock=ScriptedSock(silent=True))


# ---------------------------------------------------------------- release
class StoppedBulb:
    """A yeelight.Bulb as release_bulb() sees it."""

    def __init__(self, after_stop, saved=WARM):
        self._ip = "192.0.2.1"
        self._after_stop = after_stop
        self._Bulb__socket = ScriptedSock()          # the music connection
        self.music_sock = self._Bulb__socket
        self.bulbd_saved = saved

    def stop_music(self):
        self.music_sock.close()
        self._Bulb__socket = self._after_stop        # the fresh control connection
        if self._after_stop is None:
            raise RuntimeError("could not reach the bulb")


def test_restore_goes_out_on_the_connection_stop_opened(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", no_new_connection)
    control = ScriptedSock()
    bulb = StoppedBulb(control)
    assert release_bulb(bulb, restore=True) is True
    assert [m for m, _ in control.sent] == ["set_ct_abx", "set_bright"]
    assert control.closed and bulb._Bulb__socket is None


def test_no_restore_sends_nothing_but_still_closes(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", no_new_connection)
    control = ScriptedSock()
    assert release_bulb(StoppedBulb(control), restore=False) is False
    assert control.sent == [] and control.closed


def test_a_failed_restore_never_raises_and_still_closes(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", no_new_connection)
    monkeypatch.setattr(music_bulb, "lamp_call", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    control = ScriptedSock()
    assert release_bulb(StoppedBulb(control), restore=True) is False
    assert control.closed


def test_when_stop_never_reached_the_bulb_one_fresh_try_also_ends_music_mode(monkeypatch):
    fresh = ScriptedSock()
    opened = []
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: opened.append(a) or fresh)
    monkeypatch.setattr(music_bulb.lamp_call, "__defaults__",
                        (2.0, socket.create_connection, None))
    assert release_bulb(StoppedBulb(None), restore=True) is True
    assert len(opened) == 1
    assert [m for m, _ in fresh.sent] == ["set_music", "set_ct_abx", "set_bright"]
    assert fresh.closed


# ---------------------------------------------------------------- open
def fake_bulb_class(log, fail_first=False, props=None):
    class Fake:
        made = 0

        def __init__(self, ip, **kw):
            Fake.made += 1
            self.n = Fake.made
            self._ip = ip
            self._Bulb__socket = None

        @property
        def _socket(self):
            if self._Bulb__socket is None:
                self._Bulb__socket = ScriptedSock(props=props)
                log.append(("connect", self.n))
            return self._Bulb__socket

        def turn_on(self):
            log.append(("turn_on", self.n))

        def start_music(self):
            if fail_first and self.n == 1:
                raise RuntimeError("Bulb closed the connection.")
            log.append(("start_music", self.n))

        def stop_music(self):
            log.append(("stop_music", self.n))

    return Fake


def test_the_state_is_read_on_the_sessions_own_connection(monkeypatch):
    import yeelight
    log = []
    monkeypatch.setattr(yeelight, "Bulb", fake_bulb_class(log))
    monkeypatch.setattr(socket, "create_connection", no_new_connection)
    bulb = open_bulb("192.0.2.1")
    assert bulb.bulbd_saved == WARM
    assert log == [("connect", 1), ("turn_on", 1), ("start_music", 1)]
    assert bulb._Bulb__socket.sent[0][0] == "get_prop"


def test_no_remember_reads_nothing(monkeypatch):
    import yeelight
    log = []
    monkeypatch.setattr(yeelight, "Bulb", fake_bulb_class(log))
    bulb = open_bulb("192.0.2.1", remember=False)
    assert bulb.bulbd_saved is None
    assert log == [("turn_on", 1), ("start_music", 1)]


def test_a_retry_keeps_the_first_reading(monkeypatch):
    """The failed attempt turned the lamp on; reading again would save that."""
    import yeelight
    log = []
    monkeypatch.setattr(yeelight, "Bulb", fake_bulb_class(log, fail_first=True))
    monkeypatch.setattr(music_bulb.time, "sleep", lambda s: None)
    bulb = open_bulb("192.0.2.1")
    assert bulb.n == 2 and bulb.bulbd_saved == WARM
    assert bulb._Bulb__socket is None or bulb._Bulb__socket.sent == []
