"""
palettes.py - what a valid palette is, and a palette taken from a picture.

A palette is what music mode's ColorEngine walks: a short list of
(hue, saturation) anchors, one step per accent, in order, wrapping round.
The built-ins in music_bulb.PALETTES are four anchors each; the user's own
follow the rules here, whether typed on the desk, loaded from the settings
file or derived from album art.

    2-8 anchors          one is not a palette; more than 8 and a step is
                         rarely seen twice in a song section
    hue 0-359            whole degrees
    saturation 45-100    ColorEngine clips its output to this range, so
                         anything outside it could never be shown
    neighbours apart     each anchor at least MIN_STEP from the one before
                         it, the last from the first included: on the hue /
                         saturation disc (saturation as the radius, 0-100),
                         a step shorter than that is not seen as a step
"""

import math

import numpy as np

MIN_ANCHORS, MAX_ANCHORS = 2, 8
SAT_RANGE = (45, 100)
MIN_STEP = 10.0
MAX_PALETTES = 24
NAME_MAX = 24
RESERVED = ("auto", "screen")       # palette setting / telemetry words


class PaletteError(ValueError):
    """A palette the desk cannot use; the message is shown as it is."""


def step_distance(a, b):
    """Distance between two anchors on the disc where hue is the angle and
    saturation the radius (0-100)."""
    (h1, s1), (h2, s2) = a, b
    d = math.radians(h1 - h2)
    return math.sqrt(max(s1 * s1 + s2 * s2 - 2 * s1 * s2 * math.cos(d), 0.0))


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def clean_anchors(anchors):
    """-> [(hue, sat)] as the engine will use them: hues wrapped to 0-359,
    saturations clamped into SAT_RANGE, all whole numbers. Raises
    PaletteError for anything that cannot be made a palette."""
    if not isinstance(anchors, (list, tuple)):
        raise PaletteError("A palette is a list of colours.")
    if not MIN_ANCHORS <= len(anchors) <= MAX_ANCHORS:
        raise PaletteError(f"A palette has {MIN_ANCHORS} to {MAX_ANCHORS} colours.")
    out = []
    for a in anchors:
        if not isinstance(a, (list, tuple)) or len(a) != 2 or not all(_number(v) for v in a):
            raise PaletteError("Each colour is a hue and a saturation.")
        h = int(round(a[0])) % 360
        s = int(round(min(max(a[1], SAT_RANGE[0]), SAT_RANGE[1])))
        out.append((h, s))
    for i, a in enumerate(out):
        if step_distance(out[i - 1], a) < MIN_STEP:
            raise PaletteError("Two colours next to each other are too alike to "
                               "see the step between them.")
    return out


# ---------------------------------------------------------------------------
# from a picture
# ---------------------------------------------------------------------------
VIVID_SAT, VIVID_VAL = 0.22, 0.18     # a pixel with colour in it
MIN_VIVID = 0.02                      # share of the picture that must have colour
PEAK_SHARE = 0.08                     # a second colour must weigh this much of the first
SPREAD = 22                           # degrees between a lone colour and its neighbours


def _hsv(rgb):
    mx = rgb.max(axis=-1)
    d = mx - rgb.min(axis=-1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(mx)
    nz = d > 1e-6
    ri = nz & (mx == r)
    gi = nz & (mx == g) & ~ri
    bi = nz & ~ri & ~gi
    h[ri] = ((g[ri] - b[ri]) / d[ri]) % 6.0
    h[gi] = (b[gi] - r[gi]) / d[gi] + 2.0
    h[bi] = (r[bi] - g[bi]) / d[bi] + 4.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h * 60.0, s, mx


def _small(rgb, side=96):
    """Float RGB 0..1, at most `side` px on its long edge (box-averaged)."""
    a = np.asarray(rgb)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    a = a[..., :3].astype(np.float32)
    k = max(1, int(math.ceil(max(a.shape[0], a.shape[1]) / side)))
    if k > 1:
        h, w = a.shape[0] // k * k, a.shape[1] // k * k
        a = a[:h, :w].reshape(h // k, k, w // k, k, 3).mean(axis=(1, 3))
    return a


def _order(anchors):
    """The strongest colour first, the rest in the order whose smallest step
    (last to first included) is largest - every step should be seen."""
    import itertools
    first, rest = anchors[0], anchors[1:]
    best, key = anchors, None
    for perm in itertools.permutations(rest):
        seq = [first, *perm]
        steps = [step_distance(seq[i - 1], seq[i]) for i in range(len(seq))]
        k = (min(steps), sum(steps))
        if key is None or k > key:
            best, key = seq, k
    return list(best)


def palette_from_image(rgb, count=4):
    """The dominant vivid colours of a picture as palette anchors, strongest
    first, or None if the picture has no colour to take (black and white,
    grey, sepia, nearly black).

    Vivid, lit pixels vote into 5-degree hue bins, weighted like the screen
    engine's (saturation^1.5 x value). The peaks at least 30 degrees apart
    become anchors, each the weighted circular mean of the pixels within 20
    degrees of it. Art with one or two colours gets neighbours SPREAD
    degrees either side of the main one, so there are at least three steps.
    """
    a = _small(rgb)
    h, s, v = _hsv(a)
    vivid = (s > VIVID_SAT) & (v > VIVID_VAL)
    if vivid.mean() < MIN_VIVID or int(vivid.sum()) < 12:
        return None
    hv, sv, vv = h[vivid], s[vivid], v[vivid]
    w = sv ** 1.5 * vv

    bins = 72
    idx = (hv / 5.0).astype(int) % bins
    hist = np.bincount(idx, weights=w, minlength=bins)
    kern = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    hist = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), kern, mode="valid")

    found = []
    left = hist.copy()
    while len(found) < count:
        top = int(np.argmax(left))
        if left[top] <= 0 or (found and left[top] < PEAK_SHARE * found[0][2]):
            break
        centre = top * 5.0 + 2.5
        near = np.abs(((hv - centre + 180.0) % 360.0) - 180.0) <= 20.0
        ww = w[near]
        if ww.sum() > 0:
            ang = np.radians(hv[near])
            hue = math.degrees(math.atan2(float((np.sin(ang) * ww).sum()),
                                          float((np.cos(ang) * ww).sum()))) % 360.0
            sat = float((sv[near] * ww).sum() / ww.sum()) * 100.0 * 1.25
            found.append((hue, sat, float(left[top])))
        for k in range(-6, 7):                     # nothing within 30 degrees
            left[(top + k) % bins] = 0.0

    if not found:
        return None
    anchors = [(h_, min(max(s_, SAT_RANGE[0]), SAT_RANGE[1])) for h_, s_, _ in found]
    h0, s0 = anchors[0]
    for k in (1, -1, 2, -2):                       # a lone colour gets neighbours
        if len(anchors) >= 3:
            break
        cand = ((h0 + k * SPREAD) % 360.0, max(SAT_RANGE[0], s0 * (0.88 if k % 2 else 0.95)))
        if all(step_distance(cand, x) >= MIN_STEP * 1.5 for x in anchors):
            anchors.append(cand)
    anchors = [(int(round(h_)) % 360, int(round(s_))) for h_, s_ in anchors]
    anchors = _order(anchors)
    # drop any neighbour still too close (two peaks that merged in rounding)
    while len(anchors) > MIN_ANCHORS:
        try:
            return clean_anchors(anchors)
        except PaletteError:
            anchors.pop()
    try:
        return clean_anchors(anchors)
    except PaletteError:
        return None
