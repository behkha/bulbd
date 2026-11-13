"""The desk's half of the features: settings, the control endpoint and the
engine's session loop, with the bulb and the network replaced by fakes."""
import io
import json
import threading
import time

import pytest

import music_bulb_dash as dash
from music_bulb import BulbUnreachable


def make_state(**saved):
    return dash.State(saved or None, store=None)


def post(state, patch, path="/api/control"):
    """Run Handler.do_POST without a socket. -> (status code, body)."""
    h = dash.Handler.__new__(dash.Handler)
    h.state = state
    body = json.dumps(patch).encode()
    h.headers = {"Content-Length": str(len(body)), "Host": "localhost:8765"}
    h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
    h.path, h.command, h.request_version = path, "POST", "HTTP/1.1"
    h.requestline, h.client_address, h.close_connection = "", ("127.0.0.1", 0), False
    h.do_POST()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split()[1]), payload


# --------------------------------------------------------------- settings
@pytest.mark.parametrize("did, kept", [(0x12AB34CD, 0x12AB34CD), (None, None), (-1, "gone"),
                                       (2 ** 32, "gone"), ("1e5a", "gone"), (1.5, "gone"), (True, "gone")])
def test_device_id_is_validated(did, kept):
    out = dash.clean_patch({"did": did}, {"monitors": []})
    assert out.get("did", "gone") == kept


def test_device_id_is_saved_but_never_taken_from_the_page():
    st = make_state(ip="192.168.1.2", did=0x12AB34CD)
    assert "did" in dash.SAVED_KEYS and st.snapshot()["did"] == 0x12AB34CD
    post(st, {"did": 7})
    assert st.snapshot()["did"] == 0x12AB34CD


def test_a_new_address_forgets_the_old_bulbs_id():
    st = make_state(ip="192.168.1.2", did=0x12AB34CD)
    post(st, {"ip": "192.168.1.30"})
    snap = st.snapshot()
    assert snap["ip"] == "192.168.1.30" and snap["did"] is None
    post(st, {"ip": "192.168.1.30"})                     # same address: nothing to forget
    assert st.snapshot()["did"] is None


def test_start_and_stop_clear_the_note():
    st = make_state(ip="192.168.1.2")
    st.set(notice={"kind": "err", "text": "x", "action": None})
    post(st, {"running": True})
    assert st.snapshot()["notice"] is None
    assert not st.stopped.is_set() and st.started.is_set()
    post(st, {"running": False})
    assert st.stopped.is_set() and not st.started.is_set()


# ----------------------------------------------------------------- engine
class FakeLink:
    """Stands in for music_bulb.Link inside Engine._session."""
    made = []
    plan = {}

    def __init__(self, ip, did=None, remember=True, status=None, stopped=None, moved=None):
        self.ip, self.did, self.status, self.stopped, self.moved = ip, did, status, stopped, moved
        self.lost, self.closed, self.recovers, self.saved = False, None, 0, {"power": "on"}
        FakeLink.made.append(self)

    def open(self):
        if "open" in self.plan:
            raise self.plan["open"]
        if self.plan.get("move"):
            self.ip = self.plan["move"]
            self.moved(self.ip, 0x12AB34CD)
        return True

    def recover(self):
        self.recovers += 1
        self.lost = False
        return not self.stopped.is_set()

    def put(self, *a, **k):
        pass

    def close(self, restore=False):
        self.closed = restore


@pytest.fixture
def engine(monkeypatch):
    FakeLink.made, FakeLink.plan = [], {}
    monkeypatch.setattr(dash, "Link", FakeLink)
    monkeypatch.setattr(dash, "screen_permission", lambda: True)
    st = make_state(ip="192.168.1.2")
    eng = dash.Engine(st)
    return st, eng


def run_session(st, eng, loop):
    eng._audio_loop = loop
    st.set(running=True)
    t = threading.Thread(target=eng._session, daemon=True)
    t.start()
    t.join(3)
    assert not t.is_alive()


def test_a_dropped_link_is_recovered_and_the_loop_resumes(engine):
    st, eng = engine
    calls = []

    def loop(st_, link):
        calls.append(link)
        if len(calls) == 1:
            link.lost = True                     # the sender saw a send fail
        else:
            st_.set(running=False)

    run_session(st, eng, loop)
    link = FakeLink.made[0]
    assert link.recovers == 1 and len(calls) == 2
    assert link.closed is True                   # restored once, at the end


def test_stop_during_a_reconnect_ends_the_session(engine):
    st, eng = engine

    def loop(st_, link):
        link.lost = True
        st_.set(running=False)                   # Stop pressed while it was down

    run_session(st, eng, loop)
    assert FakeLink.made[0].recovers <= 1 and FakeLink.made[0].closed is not None


def test_a_bulb_found_elsewhere_updates_the_saved_address(engine):
    st, eng = engine
    FakeLink.plan = {"move": "192.168.1.40"}
    run_session(st, eng, lambda st_, link: st_.set(running=False))
    snap = st.snapshot()
    assert snap["ip"] == "192.168.1.40" and snap["did"] == 0x12AB34CD


def test_engine_run_turns_bulb_unreachable_into_a_notice(engine, monkeypatch):
    st, eng = engine
    FakeLink.plan = {"open": BulbUnreachable("busy", "192.168.1.2")}
    done = threading.Event()
    real_set = st.set

    def watch(**kw):
        real_set(**kw)
        if kw.get("status") == "error":
            done.set()

    st.set = watch
    eng.start()
    real_set(running=True)
    assert done.wait(3)
    snap = st.snapshot()
    assert snap["running"] is False and snap["notice"]["kind"] == "err"
    assert "switch it off and on at the wall" in snap["notice"]["text"]
