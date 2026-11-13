"""The user's own palettes: what a valid one is, taking one from album art,
keeping them in the settings file, and what deleting or renaming one that
is in use leaves behind."""
import colorsys
import json

import numpy as np
import pytest

import music_bulb_dash as dash
from music_bulb import PALETTES, ColorEngine
from palettes import (MAX_ANCHORS, MIN_STEP, PaletteError, clean_anchors,
                      palette_from_image, step_distance)

SNAP = {"monitors": [], "palette": "auto", "presets": [], "palettes": []}
SUNSET = [[12, 90], [36, 84], [330, 80], [280, 70]]


def hue_of(rgb):
    return colorsys.rgb_to_hsv(*rgb)[0] * 360


def circ(a, b):
    return abs(((a - b + 180) % 360) - 180)


# ------------------------------------------------------------------- rules
def test_every_built_in_is_a_valid_palette():
    for name, anchors in PALETTES.items():
        assert clean_anchors(anchors) == list(anchors), name


def test_hues_wrap_and_saturation_is_clamped_to_what_the_engine_shows():
    assert clean_anchors([[370.4, 20], [-30, 130.6]]) == [(10, 45), (330, 100)]


@pytest.mark.parametrize("anchors", [
    [[10, 90]],                                         # one is not a palette
    [[i * 40, 90] for i in range(MAX_ANCHORS + 1)],     # too many
    [[10, 90], [12, 90], [200, 90]],                    # neighbours alike
    [[10, 90], [200, 90], [8, 90]],                     # ...last to first too
    [[10, "red"], [200, 90]], [[10, True], [200, 90]], [[10], [20, 90]],
    [[float("nan"), 90], [200, 90]], "ember", None,
])
def test_what_is_not_a_palette(anchors):
    with pytest.raises(PaletteError):
        clean_anchors(anchors)


def test_the_step_distance_is_on_the_hue_saturation_disc():
    assert step_distance((0, 100), (180, 100)) == pytest.approx(200)
    assert step_distance((90, 50), (90, 80)) == pytest.approx(30)
    assert step_distance((5, 50), (5, 50)) == 0
    assert MIN_STEP <= step_distance((8, 84), (14, 78))   # peach's own wrap


# ------------------------------------------------------------- album art
def cover(blocks, size=120, noise=0.0, seed=0):
    """A picture of vertical bands: [(rgb, share)]."""
    img = np.zeros((size, size, 3), np.float32)
    x = 0
    for rgb, share in blocks:
        w = int(round(share * size))
        img[:, x:x + w] = rgb
        x += w
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 1)


def test_the_dominant_colours_come_out_strongest_first():
    img = cover([((0.9, 0.1, 0.1), .45), ((0.1, 0.3, 0.9), .30),
                 ((0.95, 0.8, 0.1), .15), ((0.05, 0.05, 0.05), .10)], noise=0.02)
    pal = palette_from_image(img)
    hues = [h for h, _ in pal]
    assert circ(hues[0], 0) < 6                          # red, the biggest
    for want in (hue_of((0.1, 0.3, 0.9)), hue_of((0.95, 0.8, 0.1))):
        assert min(circ(h, want) for h in hues) < 8
    assert all(45 <= s <= 100 for _, s in pal)
    assert clean_anchors(pal) == pal


def test_uint8_art_of_any_size_works():
    img = (cover([((0.2, 0.8, 0.3), .5), ((0.8, 0.2, 0.7), .5)], size=700) * 255).astype(np.uint8)
    pal = palette_from_image(img)
    assert pal and len(pal) >= 3


def test_a_one_colour_cover_still_steps():
    pal = palette_from_image(cover([((0.1, 0.5, 0.9), 1.0)], noise=0.03))
    assert len(pal) >= 3
    base = hue_of((0.1, 0.5, 0.9))
    assert all(circ(h, base) <= 50 for h, _ in pal)      # neighbours, not strangers
    assert min(step_distance(pal[i - 1], pal[i]) for i in range(len(pal))) >= MIN_STEP


@pytest.mark.parametrize("img", [
    cover([((0.5, 0.5, 0.5), .5), ((0.95, 0.95, 0.95), .3), ((0, 0, 0), .2)]),   # grey
    cover([((0.55, 0.5, 0.45), 1.0)], noise=0.01),                               # sepia
    cover([((0.04, 0.0, 0.08), 1.0)]),                                           # nearly black
    cover([((0.9, 0.9, 0.9), .99), ((1, 0, 0), .01)]),                           # a speck
])
def test_art_without_colour_gives_no_palette(img):
    assert palette_from_image(img) is None


def test_steps_are_ordered_so_each_one_shows():
    img = cover([((0.9, 0.1, 0.1), .4), ((0.9, 0.3, 0.1), .2),
                 ((0.1, 0.2, 0.9), .2), ((0.1, 0.9, 0.3), .2)])
    pal = palette_from_image(img)
    steps = [step_distance(pal[i - 1], pal[i]) for i in range(len(pal))]
    assert min(steps) >= 40                              # no red-to-orange next to each other


# ------------------------------------------------------------ the desk
def test_saving_validates_and_clamps():
    patch, err = dash.palette_op({"op": "save", "name": "  Sunset  ", "anchors": SUNSET}, SNAP)
    assert not err and patch["palettes"] == [{"name": "Sunset", "anchors": SUNSET}]
    patch, err = dash.palette_op({"op": "save", "name": "Pale", "anchors": [[0, 10], [180, 20]]}, SNAP)
    assert patch["palettes"][0]["anchors"] == [[0, 45], [180, 45]]


@pytest.mark.parametrize("name", ["ember", "Ember", "auto", "screen", "", "x" * 25, None])
def test_names_that_cannot_be_taken(name):
    _, err = dash.palette_op({"op": "save", "name": name, "anchors": SUNSET}, SNAP)
    assert err


def test_saving_under_an_existing_name_of_yours_edits_it():
    snap = dict(SNAP, palettes=[{"name": "Sunset", "anchors": SUNSET}])
    patch, err = dash.palette_op({"op": "save", "name": "Sunset",
                                  "anchors": [[100, 90], [200, 90]]}, snap)
    assert not err and patch["palettes"] == [{"name": "Sunset", "anchors": [[100, 90], [200, 90]]}]


def test_a_bad_palette_is_refused_with_the_reason():
    _, err = dash.palette_op({"op": "save", "name": "Flat", "anchors": [[10, 90], [11, 90]]}, SNAP)
    assert "too alike" in err


def test_a_custom_palette_can_be_chosen_and_used_in_a_preset():
    snap = dict(SNAP, palettes=[{"name": "Sunset", "anchors": SUNSET}])
    assert dash.clean_patch({"palette": "Sunset"}, snap) == {"palette": "Sunset"}
    assert dash.clean_patch({"palette": "Sunset"}, SNAP) == {}
    presets = dash.clean_presets([{"name": "Dusk", "look": {"palette": "Sunset"}}], snap)
    assert presets[0]["look"]["palette"] == "Sunset"


def test_control_never_takes_palettes():
    from test_desk import post
    st = dash.State()
    post(st, {"palettes": [{"name": "Sneaky", "anchors": SUNSET}]})
    assert st.snapshot()["palettes"] == []


def test_the_palettes_endpoint_saves_and_refuses():
    from test_desk import post
    st = dash.State()
    code, body = post(st, {"op": "save", "name": "Sunset", "anchors": SUNSET}, "/api/palettes")
    assert code == 200 and st.snapshot()["palettes"][0]["name"] == "Sunset"
    code, body = post(st, {"op": "save", "name": "ember", "anchors": SUNSET}, "/api/palettes")
    assert code == 400 and "already" in json.loads(body)["error"]


def in_use():
    return dict(SNAP, palette="Sunset",
                palettes=[{"name": "Sunset", "anchors": SUNSET}],
                presets=[{"name": "Dusk", "look": {"palette": "Sunset", "calm": True}},
                         {"name": "Day", "look": {"palette": "citrus"}}])


def test_deleting_a_palette_in_use_leaves_auto_behind():
    patch, err = dash.palette_op({"op": "delete", "name": "Sunset"}, in_use())
    assert not err and patch["palettes"] == [] and patch["palette"] == "auto"
    assert patch["presets"][0]["look"] == {"palette": "auto", "calm": True}
    assert patch["presets"][1]["look"] == {"palette": "citrus"}


def test_renaming_a_palette_in_use_carries_its_users_along():
    patch, err = dash.palette_op({"op": "rename", "name": "Sunset", "to": "Dusk sky"}, in_use())
    assert not err and patch["palette"] == "Dusk sky"
    assert patch["palettes"] == [{"name": "Dusk sky", "anchors": SUNSET}]
    assert patch["presets"][0]["look"]["palette"] == "Dusk sky"


def test_a_built_in_cannot_be_deleted_or_renamed():
    for op in ({"op": "delete", "name": "ember"}, {"op": "rename", "name": "ember", "to": "x"}):
        _, err = dash.palette_op(op, SNAP)
        assert err


# ------------------------------------------------------------ settings file
def test_palettes_persist_and_a_bad_one_costs_only_itself(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"version": 1, "settings": {
        "palette": "Sunset",
        "palettes": [{"name": "Sunset", "anchors": SUNSET},
                     {"name": "Broken", "anchors": [[1, 90]]},
                     {"name": "ember", "anchors": SUNSET},
                     {"name": "Sunset", "anchors": [[100, 90], [200, 90]]}],
        "presets": [{"name": "Dusk", "look": {"palette": "Sunset"}}]}}))
    store = dash.SettingsStore(str(path))
    st = dash.State(store.load(), store=store)
    snap = st.snapshot()
    assert [p["name"] for p in snap["palettes"]] == ["Sunset"]
    assert snap["palette"] == "Sunset"
    assert snap["presets"][0]["look"]["palette"] == "Sunset"
    store.attach(st)
    st.control(palettes=snap["palettes"] + [{"name": "Moss", "anchors": [[90, 80], [130, 70]]}])
    store.close()
    saved = json.loads(path.read_text())["settings"]
    assert [p["name"] for p in saved["palettes"]] == ["Sunset", "Moss"]


def test_an_older_settings_file_without_palettes_loads(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"version": 1, "settings": {"palette": "lagoon", "calm": True}}))
    st = dash.State(dash.SettingsStore(str(path)).load())
    assert st.snapshot()["palettes"] == [] and st.snapshot()["palette"] == "lagoon"


def test_a_saved_choice_of_a_palette_that_is_gone_falls_back():
    st = dash.State({"palette": "Gone", "palettes": []})
    assert st.snapshot()["palette"] == "auto"


# ------------------------------------------------------------- the engine
def test_the_engine_steps_through_a_custom_palette():
    pals = dict(PALETTES, Sunset=[tuple(a) for a in SUNSET])
    c = ColorEngine("Sunset", palettes=pals)
    seen = []
    for k in range(8):
        c.step(10.0 * (k + 1), 0.5)
        seen.append((c.target_hue, c.target_sat))
    assert seen[:4] == [(36, 84), (330, 80), (280, 70), (12, 90)]   # discrete anchors, in order


def test_automatic_switching_includes_custom_palettes():
    pals = dict(PALETTES, Sunset=[tuple(a) for a in SUNSET])
    c = ColorEngine(None, palettes=pals)
    landed = set()
    rng = np.random.default_rng(3)
    t = 0.0
    for _ in range(400):
        t += 25.0
        c.slow = np.zeros(10)
        c.maybe_switch(t, rng.random(10))
        landed.add(c.pal)
    assert "Sunset" in landed


def test_a_palette_deleted_under_the_engine_falls_back():
    pals = dict(PALETTES, Sunset=[tuple(a) for a in SUNSET])
    c = ColorEngine("Sunset", palettes=pals)
    dash.apply_palette(c, "auto", dict(PALETTES))
    c.step(100.0, 0.5)                                   # no KeyError
    assert c.pal in PALETTES and not c.locked


# ------------------------------------------------------------ from a cover
def rgb_body(img):
    import base64
    a = (np.asarray(img) * 255).astype(np.uint8)
    return {"w": a.shape[1], "h": a.shape[0], "rgb": base64.b64encode(a.tobytes()).decode()}


def test_the_page_sends_a_shrunk_cover_and_gets_anchors_back():
    from test_desk import post
    img = cover([((0.9, 0.1, 0.1), .5), ((0.1, 0.3, 0.9), .5)], size=96)
    code, body = post(dash.State(), rgb_body(img), "/api/palettes/extract")
    out = json.loads(body)
    assert code == 200 and out["ok"] and len(out["anchors"]) >= 3
    assert clean_anchors(out["anchors"]) == [tuple(a) for a in out["anchors"]]


def test_a_colourless_cover_comes_back_as_none():
    from test_desk import post
    code, body = post(dash.State(), rgb_body(cover([((0.5, 0.5, 0.5), 1.0)], size=32)),
                      "/api/palettes/extract")
    assert code == 200 and json.loads(body)["anchors"] is None


@pytest.mark.parametrize("body", [
    {"w": 4, "h": 4, "rgb": "AAAA"},                         # too short
    {"w": 500, "h": 1, "rgb": ""},                           # too big
    {"w": "4", "h": 4, "rgb": ""}, {"w": 2, "h": 2, "rgb": "not base64!"}, [],
])
def test_a_bad_picture_is_refused(body):
    anchors, err = dash.extract_palette(body)
    assert anchors is None and err


def test_extracting_changes_nothing_on_the_desk():
    from test_desk import post
    st = dash.State()
    before = st.snapshot()
    post(st, rgb_body(cover([((0.9, 0.1, 0.1), 1.0)], size=16)), "/api/palettes/extract")
    assert st.snapshot()["palettes"] == before["palettes"] and st.snapshot()["palette"] == "auto"


def test_the_page_gets_the_palette_rules_from_the_one_place_they_live():
    import palettes
    page = dash.PAGE.replace("__PAL_RULES__", "{}")
    assert "__PAL_RULES__" in dash.PAGE and "PR.name" in page and "PR.side" in page
    assert dash.ART_SIDE >= 96 and palettes.NAME_MAX == 24
