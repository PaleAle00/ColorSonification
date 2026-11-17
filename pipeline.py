"""
pipeline.py - the ORIGINAL sonification plus optional colour features.

The original presonify.py and synth.py (same folder) are imported and called unmodified, so with the colour features
off `extract` + `render` give sample-for-sample the original sound.  The additions:

    colour timbre   the dominant colour of every pitch strip (dominance.py) changes the TIMBRE of that strip's pad and bell
                    notes (timbres.py).  Notes, level curves, panning, leveler and reverb are the original ones, and the
                    new stems follow the loudness of the original stems (loudness.py): colour never changes loudness.
    grey mute       strips that are grey AND dark are faded out of the pad and the bells.
    motion_scan     an extra control track: movement of the video inside a band of columns around the scan line.
                    Like every other track it can drive any layer (render(..., mapping={"air": "motion_scan"})).
"""
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

import presonify  # noqa: E402
import synth  # noqa: E402

import dominance as dom  # noqa: E402
import timbres as tb  # noqa: E402
from loudness import follow  # noqa: E402

ORIGINAL_GUI = dict(balance="loudness", rhythm="on", beat_split=True)     # how sonify_gui.py calls synth.render_layer
SMALL = (192, 108)                                                        # frame size used to measure movement


@dataclass
class Colour:
    """the colour settings; the defaults switch everything off (= the original sound)."""
    timbre: bool = False                                             # dominant colour changes the timbre
    assignment: dict = field(default_factory=lambda: dict(tb.PRESETS["Elements"]))   # colour class -> timbre name
    depth: float = 1.0                                               # 0 = original timbre everywhere ... 1 = full class timbres
    weighting: str = "K"                                             # how "equally loud" is measured: "K", "A" or "K+A"
    mute_grey: bool = False                                          # fade out grey-and-dark strips
    grey_limit: float = 0.25                                         # ... darker than this (0..1)
    grey_tolerance: float = 0.15                                     # ... and less saturated than this (0..1)


def extract(path, motion_width=0.1, **kw):
    """original presonify.presonify(path, **kw) -> (controls, raw_loudness, meta), with two extra control tracks:
        rgb          (N, bands, 3)  colour of every band of the scanned column
        motion_scan  (N,) 0..1      frame-to-frame change inside `motion_width` (share of the picture width) around the
                                    scanned column; 0 for stills
    Both are read from the frames the original already decodes (a tap on its column_features) and smoothed like its tracks."""
    rgbs, moves, seen = [], [], dict(frame=None, now=None, before=None)
    original = presonify.column_features

    def tapped(rgb, hsv, x, n_bands, fine=None):
        rgbs.append(cv2.resize(rgb[:, x:x + 1], (1, n_bands), interpolation=cv2.INTER_AREA)[:, 0])
        if rgb is not seen["frame"]:                                 # a new frame: keep a small grey copy of it and of the last one
            grey = cv2.cvtColor(cv2.resize(rgb, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            seen.update(frame=rgb, before=seen["now"], now=grey)
        if seen["before"] is None:                                   # still picture, or the first frame
            moves.append(0.0)
        else:
            mid, half = x * SMALL[0] // rgb.shape[1], max(1, round(motion_width * SMALL[0] / 2))
            cols = slice(max(mid - half, 0), mid + half + 1)
            moves.append(float(cv2.absdiff(seen["now"][:, cols], seen["before"][:, cols]).mean()))
        return original(rgb, hsv, x, n_bands, fine)

    presonify.column_features = tapped
    try:
        out, raw_loud, meta = presonify.presonify(Path(path), **kw)
    finally:
        presonify.column_features = original
    smooth = lambda a: gaussian_filter1d(a, meta["smooth_s"] * meta["control_rate_hz"], axis=0, mode="nearest").astype(np.float32)
    moves = np.array(moves)
    out["rgb"] = smooth(np.stack(rgbs))
    out["motion_scan"] = smooth(np.clip(moves / max(np.percentile(moves, 99), 1e-9), 0, 1))     # scaled like the original motion
    return out, raw_loud, meta


def strip_weights(c, strips):
    """class weights of every pitch strip: (N, strips, classes)."""
    t = c["t"].astype(float)
    w = dom.class_weights(c["rgb"], (len(t) - 1) / t[-1])
    return np.stack([synth.pool(w[:, :, k], strips) for k in range(w.shape[2])], axis=2)


def tone(f, w, voices, reference, colour, bpm, t_c, t):
    """one strip's note at f Hz.  Its harmonics follow the class weights w (N, classes): classes are crossfaded on the POWER
    of every harmonic and each is matched to the weighted power of `reference`, so the loudness stays put."""
    power = np.stack([tb.matched(v, f, reference, colour.weighting) for v in voices]) ** 2    # (classes, harmonics)
    shape = np.sqrt((1 - colour.depth) * tb.partials(reference, f) ** 2 + colour.depth * (w @ power))   # (N, harmonics)
    out = np.zeros(len(t))
    for h in np.nonzero(shape.max(axis=0) > 1e-6)[0]:
        out += np.interp(t, t_c, shape[:, h]) * np.sin(2 * np.pi * (h + 1) * f * t)
    pulse = sum(np.interp(t, t_c, w[:, k]) * v.depth * np.sin(2 * np.pi * v.pulses * bpm / 60 * t)
                for k, v in enumerate(voices) if v.depth and w[:, k].max() > 1e-3)             # pulses of the classes present
    return out * (1 + colour.depth * pulse)


def recolour(stems, c, colour, kw, progress):
    """replace the pad and bell stems by colour-timbred ones of the same loudness.  Everything except the harmonic
    recipe repeats the original render_layer (same drivers, notes, level curves, panning, leveler, reverb, fades)."""
    p = {k: v.default for k, v in inspect.signature(synth.render_layer).parameters.items()} | kw
    mp = synth.DEFAULT_MAPPING | (p["mapping"] or {})
    c = synth.with_derived(c)
    t_c = c["t"].astype(float)
    t = np.arange(int(t_c[-1] * synth.SR)) / synth.SR
    bg = synth.pool(synth.resolve(c, mp["pad"], "matrix"), p["voices"])
    ev = synth.sparsify(synth.pool(synth.resolve(c, mp["events"], "matrix"), p["voices"]), p["event_voices"],
                        p["event_floor"], p["event_power"], p["event_smooth_frames"])
    V = bg.shape[1]
    f = synth.mtof(synth.ladder(V, p["root"], p["mode"], p["base_midi"] + p["shift"]))
    pans = np.linspace(p["pad_pan_width"], -p["pad_pan_width"], V)
    w = strip_weights(c, V)
    voices = [tb.TIMBRES[colour.assignment[k]] for k in dom.CLASSES]
    pad_ref, bell_ref = (1.0, p["pad_harmonic_base"] + 0.5 * p["pad_harmonic_hue"]), (1.0, p["event_partial_gain"])
    new = {"pad": np.zeros((len(t), 2)), "events": np.zeros((len(t), 2))}
    for b in range(V):
        progress(b / V, "colour timbres")
        lr = np.array(synth.pan_gains(pans[b]))
        a = (np.clip(synth.up(bg[:, b], t_c, t), 0, None) ** p["pad_amp_exponent"]
             * (p["pad_rolloff_ref_hz"] / f[b]) ** p["pad_rolloff_exponent"])
        new["pad"] += np.outer(a * tone(f[b], w[:, b], voices, pad_ref, colour, p["rhythm_bpm"], t_c, t), lr)
        e = np.clip(synth.up(ev[:, b], t_c, t), 0, None)
        if e.max() > 0:
            fe = f[b] * 2.0 ** (p["event_shift_semitones"] / 12.0)
            new["events"] += np.outer(e * tone(fe, w[:, b], voices, bell_ref, colour, p["rhythm_bpm"], t_c, t), lr)
    rng = np.random.default_rng(p["seed"] + 2000)                    # own generator: the original voices stay unchanged
    for name, x in new.items():
        if min(np.abs(x).max(), np.abs(stems[name]).max()) > 1e-9:   # (a silent stem stays as it is)
            x = synth.reverb(synth.leveler(x, p["leveler_strength"], p["leveler_tau"], p["leveler_percentile"]), rng,
                             p["reverb_tau"], p["reverb_length"], p["reverb_wet"])
            stems[name] = follow(synth.fades(x, p["fade_in"], p["fade_out"]), stems[name], colour.weighting)


def render(c, colour=None, **kw):
    """original synth.render_layer(c, **kw) -> stems, plus the colour features chosen in `colour` (None = all off)."""
    colour = colour or Colour()
    report = kw.get("progress") or (lambda f, m="": None)
    if colour.timbre:                                                # the progress bar is shared by the two passes
        kw = dict(kw, progress=lambda f, m="": report(0.5 * f, m))
    if colour.mute_grey:
        gate = dom.grey_gate(c["rgb"], colour.grey_limit, colour.grey_tolerance).astype(np.float32)
        c = dict(c, background=c["background"] * gate, events=c["events"] * gate)
    stems = synth.render_layer(c, **kw)
    if colour.timbre:
        recolour(stems, c, colour, kw, lambda f, m="": report(0.5 + 0.5 * f, m))
    return stems
