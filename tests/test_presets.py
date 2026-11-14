"""Presets: what they hold, the built-ins, applying one through clean_patch,
managing your own, and keeping them in the settings file."""
import io
import json

import pytest

import music_bulb_dash as dash
from test_desk import post

SNAP = {"monitors": [{"i": 1, "w": 1800, "h": 1169, "x": 0, "y": 0}]}


def state(**saved):
    return dash.State(saved or None)


def ops(st, body):
    code, payload = post(st, body, path="/api/presets")
    return code, json.loads(payload)


# ------------------------------------------------------------- what it holds
def test_a_preset_holds_the_look_and_nothing_about_the_room():
    look = set(dash.LOOK_KEYS)
    assert look == set(dash.LOOK_DEFAULTS)
    assert look <= set(dash.SAVED_KEYS)
    assert not look & {"ip", "did", "mic", "monitor", "region", "presets", "lead", "pause"}
    # every fader but Beat sync and the pause light, and the switches added with Auto
    assert set(dash.NUMERIC) - {"lead", "pause"} <= look and {"auto", "mode"} <= look


@pytest.mark.parametrize("preset", dash.BUILTIN_PRESETS, ids=lambda p: p["name"])
def test_builtins_are_complete_and_already_valid(preset):
    look = preset["look"]
    assert set(look) == set(dash.LOOK_KEYS)
    assert dash.clean_look(look, SNAP) == look


def test_builtins_are_different_looks():
    looks = [json.dumps(p["look"], sort_keys=True) for p in dash.BUILTIN_PRESETS]
    assert len(set(looks)) == len(looks)
    assert {p["look"]["mode"] for p in dash.BUILTIN_PRESETS} == {"music", "screen", "fusion"}


# ------------------------------------------------------------------- apply
def test_applying_a_preset_sets_its_look_and_leaves_the_rest():
    st = state(ip="192.168.1.2", mic="BlackHole 2ch", lead=120, punch=0.3, mode="screen", auto=True)
    post(st, {"preset": "Lounge"})
    snap = st.snapshot()
    lounge = dash.BUILTIN_PRESETS[1]["look"]
    assert all(snap[k] == v for k, v in lounge.items())
    assert (snap["ip"], snap["mic"], snap["lead"], snap["running"]) == ("192.168.1.2", "BlackHole 2ch", 120, False)


def test_a_preset_goes_through_clean_patch(monkeypatch):
    seen = []
    real = dash.clean_patch
    monkeypatch.setattr(dash, "clean_patch", lambda p, s: seen.append(dict(p)) or real(p, s))
    st = state()
    post(st, {"preset": "Cinema"})
    assert seen and seen[-1]["mode"] == "screen"


def test_an_unknown_preset_changes_nothing():
    st = state(punch=0.3)
    post(st, {"preset": "Nope"})
    assert st.snapshot()["punch"] == 0.3


def test_the_page_cannot_write_the_list_through_control():
    st = state()
    post(st, {"presets": [{"name": "Sneaky", "look": {"punch": 0.5}}]})
    assert st.snapshot()["presets"] == []


# ---------------------------------------------------------------- your own
def test_save_rename_apply_delete():
    st = state(punch=0.6, palette="nocturne")
    assert ops(st, {"op": "save", "name": "  Late   night "}) == (200, {"ok": True, "error": ""})
    [mine] = st.snapshot()["presets"]
    assert mine["name"] == "Late night" and mine["look"]["punch"] == 0.6
    assert ops(st, {"op": "rename", "name": "Late night", "to": "After hours"})[0] == 200
    post(st, {"punch": 0.9})
    post(st, {"preset": "After hours"})
    assert st.snapshot()["punch"] == 0.6 and st.snapshot()["palette"] == "nocturne"
    assert ops(st, {"op": "delete", "name": "After hours"})[0] == 200
    assert st.snapshot()["presets"] == []


def test_saving_again_under_the_same_name_updates_it():
    st = state(punch=0.6)
    ops(st, {"op": "save", "name": "Mine"})
    post(st, {"punch": 0.4})
    ops(st, {"op": "save", "name": "Mine"})
    assert [p["look"]["punch"] for p in st.snapshot()["presets"]] == [0.4]


@pytest.mark.parametrize("body, words", [
    ({"op": "save", "name": "club"}, "Club is a built-in preset"),
    ({"op": "save", "name": ""}, "Give the preset a name"),
    ({"op": "save", "name": "x" * 33}, "Give the preset a name"),
    ({"op": "rename", "name": "Mine", "to": "Lounge"}, "already a preset called Lounge"),
    ({"op": "rename", "name": "Mine", "to": "other"}, "already a preset called other"),
    ({"op": "delete", "name": "Club"}, "No such preset of yours"),
    ({"op": "explode", "name": "Mine"}, "bad request"),
])
def test_bad_requests_are_refused_with_a_reason(body, words):
    st = state()
    ops(st, {"op": "save", "name": "Mine"})
    ops(st, {"op": "save", "name": "Other"})
    code, payload = ops(st, body)
    assert code == 400 and words in payload["error"]
    assert [p["name"] for p in st.snapshot()["presets"]] == ["Mine", "Other"]


def test_the_list_has_a_ceiling():
    st = state()
    for i in range(dash.MAX_PRESETS):
        assert ops(st, {"op": "save", "name": f"P{i}"})[0] == 200
    code, payload = ops(st, {"op": "save", "name": "One more"})
    assert code == 400 and len(st.snapshot()["presets"]) == dash.MAX_PRESETS


# ------------------------------------------------------------ settings file
def test_presets_survive_a_restart(tmp_path):
    path = tmp_path / "settings.json"
    store = dash.SettingsStore(str(path))
    st = dash.State(store.load(), store=store)
    store.attach(st)
    ops(st, {"op": "save", "name": "Late night"})
    store.close()
    again = dash.State(dash.SettingsStore(str(path)).load())
    assert [p["name"] for p in again.snapshot()["presets"]] == ["Late night"]


def test_a_file_from_before_presets_still_loads(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"version": 1, "settings": {
        "ip": "192.168.1.2", "mode": "screen", "punch": 0.5, "palette": "auto"}}))
    snap = dash.State(dash.SettingsStore(str(path)).load()).snapshot()
    assert snap["presets"] == [] and snap["mode"] == "screen" and snap["punch"] == 0.5
    assert snap["auto"] is False and snap["lead"] == 80 and snap["did"] is None


def test_a_broken_entry_costs_only_itself(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"version": 1, "settings": {"presets": [
        {"name": "Good", "look": {"punch": 0.5, "mode": "music"}},
        {"name": "Club", "look": {"punch": 0.5}},                 # a built-in's name
        {"name": "good", "look": {"punch": 0.4}},                 # a duplicate
        {"name": "No look"},
        "nonsense",
        {"name": "Clamped", "look": {"punch": 9, "lead": 200, "ip": "10.0.0.1"}},
    ]}}))
    presets = dash.State(dash.SettingsStore(str(path)).load()).snapshot()["presets"]
    assert presets == [{"name": "Good", "look": {"punch": 0.5, "mode": "music"}},
                       {"name": "Clamped", "look": {"punch": 1.0}}]


def test_the_page_gets_the_builtins():
    h = dash.Handler.__new__(dash.Handler)
    h.state = state()
    h.headers = {"Host": "localhost:8765"}
    h.wfile = io.BytesIO()
    h.path, h.command, h.request_version = "/index.html", "GET", "HTTP/1.1"
    h.requestline, h.client_address, h.close_connection = "", ("127.0.0.1", 0), False
    h.do_GET()
    page = h.wfile.getvalue().decode()
    assert "__PRESETS__" not in page and '"name": "Concert"' in page
