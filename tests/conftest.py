"""Pytest setup: the suite is offline - no bulb, no screen, no audio device."""
import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if a test ever tries to reach a bulb (or anything else)."""
    import socket

    def refuse(*a, **k):
        raise AssertionError("tests must not open sockets")

    monkeypatch.setattr(socket.socket, "connect", refuse)
