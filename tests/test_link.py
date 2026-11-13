"""Self-healing connection: the miio hello, finding a bulb that moved,
explaining a failure, and reconnecting after a drop - all against fakes."""
import socket
import threading
import time

import pytest

import music_bulb
from music_bulb import (HELLO, BulbUnreachable, Link, classify, find_bulb, hello_id,
                        miio_hello)

BULB = 0x12AB34CD
OTHER = 0x0BADF00D


def answer(did):
    """A hello answer as a color5 sends it: header, length, 4 zero bytes, id."""
    return bytes.fromhex("21310020") + b"\0" * 4 + did.to_bytes(4, "big") + b"\xab" * 20


class FakeLAN:
    """UDP sockets on a pretend /24. `hosts` maps address -> device id;
    `broadcast` says whether this router passes the broadcast hello."""

    def __init__(self, hosts, broadcast=True):
        self.hosts, self.broadcast = dict(hosts), broadcast
        self.sent = []

    def __call__(self):
        lan = self

        class Sock:
            def __init__(self):
                self.inbox = []

            def setsockopt(self, *a):
                pass

            def settimeout(self, t):
                pass

            def sendto(self, data, dest):
                assert data == HELLO and dest[1] == 54321
                lan.sent.append(dest[0])
                if dest[0].endswith(".255"):
                    if lan.broadcast:
                        self.inbox += [(answer(d), (a, 54321)) for a, d in lan.hosts.items()]
                elif dest[0] in lan.hosts:
                    self.inbox.append((answer(lan.hosts[dest[0]]), (dest[0], 54321)))

            def recvfrom(self, n):
                if not self.inbox:
                    raise socket.timeout("timed out")
                return self.inbox.pop(0)

            def close(self):
                pass

        return Sock()


def net(near, udp):
    return "192.168.1.11", "192.168.1.255"


# ------------------------------------------------------------------ hello
def test_hello_id_reads_the_device_id():
    assert hello_id(answer(BULB)) == BULB
    assert hello_id(b"\x21\x31" + b"\0" * 10) is None
    assert hello_id(b"x" * 32) is None


def test_hello_gets_the_id_from_the_address_asked():
    lan = FakeLAN({"192.168.1.2": BULB})
    assert miio_hello("192.168.1.2", udp=lan) == BULB
    assert miio_hello("192.168.1.3", udp=lan, tries=2) is None
    assert lan.sent.count("192.168.1.3") == 2                  # bounded: two tries


def test_find_uses_the_broadcast_first():
    lan = FakeLAN({"192.168.1.7": BULB, "192.168.1.9": OTHER})
    assert find_bulb(BULB, "192.168.1.2", udp=lan, net=net, timeout=0.05) == "192.168.1.7"
    assert lan.sent == ["192.168.1.255"]


def test_find_sweeps_the_subnet_when_the_router_drops_broadcasts():
    lan = FakeLAN({"192.168.1.7": BULB}, broadcast=False)
    assert find_bulb(BULB, "192.168.1.2", udp=lan, net=net, timeout=0.05) == "192.168.1.7"
    swept = lan.sent[1:]
    assert len(swept) == 253 and "192.168.1.11" not in swept   # never itself


def test_find_returns_none_when_the_bulb_is_nowhere():
    lan = FakeLAN({"192.168.1.9": OTHER})
    t0 = time.monotonic()
    assert find_bulb(BULB, "192.168.1.2", udp=lan, net=net, timeout=0.05) is None
    assert time.monotonic() - t0 < 2.0


# ---------------------------------------------------------------- explain
def failure(cls, stage):
    exc = cls("boom")
    exc.bulbd_stage = stage
    return exc


@pytest.mark.parametrize("exc, answered, known, kind", [
    (failure(socket.timeout, "connect"), False, False, "no_device"),
    (failure(OSError, "connect"), False, False, "no_device"),          # host unreachable
    (failure(ConnectionRefusedError, "connect"), False, False, "not_bulb"),
    (failure(ConnectionRefusedError, "connect"), True, False, "lan_off"),
    (failure(ConnectionRefusedError, "connect"), True, True, "busy"),   # it worked before
    (failure(socket.timeout, "connect"), True, True, "busy"),
    (failure(TimeoutError, "query"), True, True, "busy"),
    (failure(ConnectionError, "query"), True, False, "busy"),
    (failure(socket.timeout, "music"), True, True, "blocked"),
    (failure(RuntimeError, "music"), True, True, "busy"),
    (RuntimeError("odd"), True, True, "other"),
])
def test_classify(exc, answered, known, kind):
    assert classify(exc, answered, known) == kind


@pytest.mark.parametrize("kind, words, action", [
    ("no_device", "Nothing answers at 192.168.1.2", None),
    ("not_bulb", "is another device", "setup"),
    ("lan_off", "Enable LAN in Setup", "setup"),
    ("busy", "switch it off and on at the wall", None),
    ("blocked", "firewall", None),
])
def test_reasons_are_plain_and_say_what_to_do(kind, words, action):
    exc = BulbUnreachable(kind, "192.168.1.2")
    assert words in str(exc) and exc.action == action
    assert "Errno" not in str(exc) and "Traceback" not in str(exc)


# ------------------------------------------------------------------- Link
class FakeBulb:
    def __init__(self, ip):
        self._ip = ip
        self._Bulb__socket = object()
        self.sent, self.fail = [], False
        self.bulbd_saved = None

    def set_hsv(self, h, s, v, effect, duration):
        if self.fail:
            raise OSError("reset by peer")
        self.sent.append((h, s, v))


class FakeLink(Link):
    """Link with the network replaced: `lan` is address -> device id,
    `opens` a list of what each open does (a bulb, or an exception)."""

    START_WAITS = (0.0, 0.0)
    RETRY = (0, 0, 0)

    def __init__(self, lan, opens=None, **kw):
        self.log = []
        super().__init__(status=lambda w, n=None: self.log.append((w, n)), **kw)
        self.lan, self.opens = lan, list(opens or [])
        self.hellos, self.finds, self.opened, self.moves = [], [], [], []
        self.moved = lambda ip, did: self.moves.append((ip, did))

    def _hello(self, ip):
        self.hellos.append(ip)
        return self.lan.get(ip)

    def _find(self, did, ip):
        self.finds.append(did)
        return next((a for a, d in self.lan.items() if d == did), None)

    def _open(self, ip):
        self.opened.append(ip)
        what = self.opens.pop(0) if self.opens else "ok"
        if isinstance(what, BaseException):
            raise what
        self.memo.setdefault("saved", {"power": "on", "bright": 30})
        return FakeBulb(ip)

    def _wait(self, secs, tick=None):
        if tick:
            tick(3)
        return self.stopped.is_set()


@pytest.fixture
def links():
    made = []

    def make(*a, **kw):
        link = FakeLink(*a, **kw)
        made.append(link)
        return link

    yield make
    for link in made:
        link.close()


def test_first_open_learns_the_id_and_reports_it(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2")
    assert link.open() is True
    assert link.did == BULB and link.moves == [("192.168.1.2", BULB)]
    assert link.hellos == ["192.168.1.2"] and link.finds == []


def test_a_bulb_that_moved_is_found_by_its_id_and_the_address_saved(links):
    link = links({"192.168.1.7": BULB}, ip="192.168.1.2", did=BULB)
    assert link.open() is True
    assert link.ip == "192.168.1.7" and link.opened == ["192.168.1.7"]
    assert link.moves == [("192.168.1.7", BULB)]
    assert ("ok", "The bulb moved from 192.168.1.2 to 192.168.1.7.") in [n for _, n in link.log]


def test_a_lost_hello_is_not_a_missing_bulb(links):
    """The first hello went missing; the search finds the bulb where it
    always was. That is an open, not "nothing answers", and not a move."""
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2", did=BULB)
    link._hello = lambda ip: None                       # dropped on the Wi-Fi
    assert link.open() is True
    assert link.ip == "192.168.1.2" and link.opened == ["192.168.1.2"]
    assert link.finds == [BULB] and link.moves == []
    assert not any(n and "moved" in n[1] for _, n in link.log)


def test_another_device_on_the_old_address_starts_a_search(links):
    link = links({"192.168.1.2": OTHER, "192.168.1.8": BULB}, ip="192.168.1.2", did=BULB)
    assert link.open() and link.ip == "192.168.1.8"


@pytest.mark.parametrize("lan, kind", [({}, "no_device"), ({"192.168.1.2": OTHER}, "not_bulb")])
def test_a_bulb_that_is_nowhere_is_explained_without_opening_anything(links, lan, kind):
    link = links(lan, ip="192.168.1.2", did=BULB)
    with pytest.raises(BulbUnreachable) as err:
        link.open()
    assert err.value.kind == kind and link.opened == []
    if kind == "no_device":
        assert "anywhere else on this network" in str(err.value)


def test_lan_off_fails_at_once_and_points_at_setup(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2",
                 opens=[failure(ConnectionRefusedError, "connect")])
    with pytest.raises(BulbUnreachable) as err:
        link.open()
    assert err.value.kind == "lan_off" and err.value.action == "setup"
    assert len(link.opened) == 1                                # no point retrying


def test_busy_slots_are_retried_then_explained(links):
    busy = [failure(TimeoutError, "query") for _ in range(3)]
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2", did=BULB, opens=busy)
    with pytest.raises(BulbUnreachable) as err:
        link.open()
    assert err.value.kind == "busy" and len(link.opened) == 3
    assert [w for w, _ in link.log] == ["connecting", "connecting 2/3", "connecting 3/3"]


def test_busy_then_free_opens(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2", opens=[failure(TimeoutError, "query")])
    assert link.open() is True and len(link.opened) == 2


def test_stop_while_waiting_to_retry_ends_the_open(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2", opens=[failure(TimeoutError, "query")])
    link.stopped.set()
    assert link.open() is False


def drop(link):
    link.bulb.fail = True
    link.put((1, 2, 3), False)
    assert link.sender.lost.wait(2)


def test_a_drop_is_noticed_and_recovered_on_the_same_address(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2")
    link.open()
    first = link.bulb
    drop(link)
    assert link.lost
    assert link.recover() is True
    assert link.bulb is not first and not link.lost
    assert first._Bulb__socket is None                          # closed, nothing sent
    link.put((4, 5, 6), False)
    time.sleep(0.05)
    assert link.bulb.sent == [(4, 5, 6)]
    words = [w for w, _ in link.log]
    assert "reconnecting 1/3" in words and words[-1] == "connected"
    assert ("warn", "Lost the bulb. Trying again in 3 s.") in [n for _, n in link.log]


def test_a_reconnect_keeps_the_look_saved_at_start(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2")
    link.open()
    saved = link.saved
    drop(link)
    link.memo["saved"] = saved
    link.recover()
    assert link.bulb.bulbd_saved == saved


def test_reconnect_follows_a_bulb_that_rebooted_onto_a_new_address(links):
    lan = {"192.168.1.2": BULB}
    link = links(lan, ip="192.168.1.2")
    link.open()
    drop(link)
    lan.clear(); lan["192.168.1.40"] = BULB
    assert link.recover() is True and link.ip == "192.168.1.40"
    assert link.log[-1] == ("connected", ("ok", "The bulb moved from 192.168.1.2 to 192.168.1.40."))


def test_reconnect_gives_up_after_its_tries_with_the_reason(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2")
    link.open()
    drop(link)
    link.opens = [failure(TimeoutError, "query")] * 3
    with pytest.raises(BulbUnreachable) as err:
        link.recover()
    assert len(link.opened) == 4                                # 1 open + RETRY tries
    assert str(err.value).startswith("Lost the bulb and could not get it back.")
    assert err.value.kind == "busy"


def test_stop_ends_a_reconnect_at_once(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2")
    link.open()
    drop(link)
    link.stopped.set()
    assert link.recover() is False and len(link.opened) == 1


def test_real_waits_are_interrupted_by_stop():
    link = Link("192.0.2.1")
    t = threading.Timer(0.1, link.stopped.set)
    t.start()
    t0 = time.monotonic()
    assert link._wait(30) is True
    assert time.monotonic() - t0 < 1.0


def test_closing_a_lost_link_sends_nothing(links, monkeypatch):
    monkeypatch.setattr(music_bulb, "release_bulb",
                        lambda *a, **k: pytest.fail("talked to a dropped bulb"))
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2")
    link.open()
    drop(link)
    assert link.close(restore=True) is False


def test_closing_a_healthy_link_restores_what_start_saw(links, monkeypatch):
    seen = []
    monkeypatch.setattr(music_bulb, "release_bulb", lambda bulb, restore: seen.append(restore) or True)
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2")
    link.open()
    assert link.close(restore=True) is True
    assert seen == [{"power": "on", "bright": 30}]


def test_a_healthy_session_makes_no_extra_connection(links):
    link = links({"192.168.1.2": BULB}, ip="192.168.1.2")
    link.open()
    for i in range(50):
        link.put((i, 50, 50), False)
    time.sleep(0.1)
    assert link.opened == ["192.168.1.2"] and link.hellos == ["192.168.1.2"]
