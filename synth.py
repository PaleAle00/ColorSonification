#!/usr/bin/env python3
"""
synth.py - render the control files made by presonify.py (*_ctrl.npz) to audio.

Every input file becomes one LAYER with four (optionally six) voices.  Each voice is written as
its own stereo WAV "stem" (same length, same sample rate, already at the level
used in the mix) so you can remix them in Audacity / Reaper / LMMS / any DAW,
plus a ready-made mix.  Mixing tracks from different sources = passing several
files:

    python synth.py out/aerosol_ctrl.npz out/co2_ctrl.npz -o audio/mix
    python synth.py out/aerosol_ctrl.npz -o audio/aerosol --root D --mode pentatonic
    python synth.py a_ctrl.npz b_ctrl.npz -o audio/mix --gain 1.0 0.6 --air waves off

Voices (per layer)
    pad     one soft tone per group of image bands (--voices, default 12); top rows =
            higher notes.  Notes come from a scale; the default PENTATONIC scale
            has no semitone steps, so any combination of notes is consonant.
            Level <- `background`.  Timbre (2nd harmonic) <- hue, gated by chroma.
            Stereo: low notes left, high notes right.
    events  bell-like tones, level <- `events`.  Only the strongest few bright
            spots sound at the same moment (--event-voices), so busy pictures stay
            clear instead of turning into a cluster.
    drone   root + fifth, level <- loudness, brightness <- warmth
    air     per-layer choice (--air):
              waves  (default) noise shaped into slow ocean waves
              hiss   raw white noise (the old behaviour)
              off    nothing
            silent for still images in every mode
    beat    (optional, rhythm="on")  kick + snare + hi-hat on a 16-step grid.  Nothing is random: at every grid
            step the picture decides whether a pattern note plays.  kick <- bottom third of the picture,
            snare <- middle third, hats <- texture (fine detail), all raised a little by video motion.
            Busy/bright picture parts -> denser beat; dark or empty parts -> silence.
    sparkle (optional, rhythm="on")  short high plucks, one per grid step that has a strong onset ("flux": an edge or
            hot spot passing the scan line).  Pitch = scale note of the band where the onset happened.
            Works identically for stills (the scan line crosses edges) and videos (plus real change in time).

Library use (what the reprocess_*.py scripts do):
    stems = render_layer(control_dict, root=..., mode=..., ...)   # every tunable is a keyword
    write_mix(prefix, [("", stems, 1.0)], ...)                    # mix + stems + spectrogram

All numeric tunables are keyword parameters whose defaults reproduce the standard sound.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

SR = 44100
MODES = {  # semitone steps
    "pentatonic": [0, 2, 4, 7, 9],            # major pentatonic: no semitones -> never clashes
    "major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10], "lydian": [0, 2, 4, 6, 7, 9, 11],
}
ROOTS = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
VOICE_GAIN = {"pad": 0.60, "events": 0.42, "drone": 0.40, "air": 0.22}   # relative levels (default)
LAYER_SHIFT = [0, 12, -12]                                               # per-layer register when mixing several files
BASE_MIDI = 40                                                           # pad range starts at D2 for root D
VOICE_GAIN.update({"beat": 0.50, "sparkle": 0.35, "kick": 0.50, "snare": 0.35, "hat": 0.25})                       # only used when rhythm="on"
# loudness balance ("balance='loudness'"): level of every voice in dB (K-weighted) relative to the pad
BALANCE_OFFSETS_DB = {"pad": 0.0, "events": -2.0, "drone": -1.0, "air": -17.0, "beat": -5.0, "sparkle": -8.0,
                      "kick": -5.0, "snare": -13.0, "hat": -21.0}          # kick/snare/hat only when beat_split=True

# ---- which picture feature drives which voice (the "mapping") --------------------------------------------------
# Every driver is a list of (feature, weight) pairs; several pairs are averaged with those weights.
MATRIX_FEATURES = ["background", "events", "onsets"]                      # (time x bands): one value per picture strip
SCALAR_FEATURES = ["loudness", "centroid", "spread", "saturation", "hue", "chroma", "colour", "warmth", "motion",
                   "flux", "texture", "zone_low", "zone_mid", "zone_high"]    # one value per moment
DEFAULT_MAPPING = {
    "pad": [("background", 1.0)], "events": [("events", 1.0)], "drone": [("loudness", 1.0)],
    "air": [("motion", 1.0)], "kick": [("zone_low", 1.0)], "snare": [("zone_mid", 1.0)],
    "hat": [("texture", 1.0)], "sparkle": [("flux", 1.0)],
    "pad_timbre": [("colour", 1.0)], "drone_timbre": [("warmth", 1.0)],
}


def with_derived(c):
    """add the derived feature 'colour' (how red/magenta AND colourful) to a control dict."""
    if "colour" not in c:
        c = dict(c)
        c["colour"] = (c["chroma"] * (0.5 + 0.5 * np.cos(2 * np.pi * c["hue"]))).astype(np.float32)
    return c


def resolve(c, spec, kind):
    """combine one or more features into the single driver a voice listens to.
    kind: 'matrix' -> (N, B);  'scalar' -> (N,) in 0..1;  'signed' -> (N,) in -1..1 (the timbre of the drone).
    A single feature with weight 1 in its native shape is returned unchanged."""
    if isinstance(spec, str):
        spec = [(spec, 1.0)]
    spec = [(n, float(w)) for n, w in spec if w > 0]
    if not spec:
        N = len(c["t"])
        return np.zeros((N, c["background"].shape[1])) if kind == "matrix" else np.zeros(N)
    parts = []
    for name, w in spec:
        a = np.asarray(c[name])
        if kind == "matrix":
            if a.ndim == 1:                                           # a scalar drives every strip equally
                a = np.repeat(a[:, None], c["background"].shape[1], axis=1)
        else:
            if a.ndim == 2:                                           # a matrix drives a scalar through its mean
                a = a.mean(axis=1)
                a = np.clip(a / max(np.percentile(a, 99), 1e-9), 0, 1)
            elif kind == "signed" and name != "warmth":
                a = 2 * a - 1
            elif kind == "scalar" and name == "warmth":
                a = 0.5 + 0.5 * np.clip(a, -1, 1)
        parts.append((a, w))
    if len(parts) == 1 and (parts[0][0].ndim == 2 or kind != "matrix"):
        return parts[0][0]
    tot = sum(w for _, w in parts)
    return sum(a * w for a, w in parts) / tot


def resample_controls(c, duration, rate=None):
    """stretch / squeeze every control track to a new length (seconds).  Hue is interpolated around the circle."""
    t_old = np.asarray(c["t"], float)
    r = rate or (len(t_old) - 1) / t_old[-1]
    N = int(round(duration * r))
    u_old = np.linspace(0, 1, len(t_old)); u_new = np.linspace(0, 1, N)
    out = {}
    for k, v in c.items():
        v = np.asarray(v)
        if k == "t":
            out[k] = (np.arange(N) / r).astype(np.float32)
        elif v.ndim >= 1 and v.shape[0] == len(t_old) and v.dtype.kind == "f":
            if k == "hue":
                ang = np.unwrap(2 * np.pi * v.astype(float))
                out[k] = ((np.interp(u_new, u_old, ang) / (2 * np.pi)) % 1.0).astype(np.float32)
            elif v.ndim == 1:
                out[k] = np.interp(u_new, u_old, v).astype(np.float32)
            else:
                out[k] = np.stack([np.interp(u_new, u_old, v[:, j]) for j in range(v.shape[1])], 1).astype(np.float32)
        else:
            out[k] = v
    return out
# 16-step patterns (one bar of 16th notes; index 0 = beat 1).  A note plays when  weight x picture-activity >= threshold,
# so a brighter / busier picture lets the lower-weight (off-beat) steps play too.
KICK_PATTERN = [1.0, 0, 0, 0.25, 0, 0, 0.5, 0, 0.9, 0, 0.3, 0, 0, 0, 0.45, 0]
SNARE_PATTERN = [0, 0, 0, 0, 1.0, 0, 0, 0.25, 0, 0, 0, 0.3, 1.0, 0, 0, 0.5]
HAT_PATTERN = [0.9, 0.35, 0.7, 0.35, 0.9, 0.35, 0.7, 0.35, 0.9, 0.35, 0.7, 0.35, 0.9, 0.35, 0.7, 0.45]


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def ladder(n, root: str, mode: str, base_midi: int):
    """n scale notes upward from `root` at/below base_midi; returned TOP FIRST (voice 0 = top rows)."""
    steps = MODES[mode]
    start = base_midi - (base_midi - ROOTS[root]) % 12
    notes = [start + 12 * (i // len(steps)) + steps[i % len(steps)] for i in range(n)]
    return np.array(notes[::-1], float)


def pool(M, V):
    """average neighbouring bands: (N, B) -> (N, V)."""
    if M.shape[1] == V:
        return M
    return np.stack([M[:, g].mean(axis=1) for g in np.array_split(np.arange(M.shape[1]), V)], axis=1)


def sparsify(E, k=3, floor=0.2, power=2.0, smooth_frames=3.0):
    """keep only the k strongest events per moment.  Subtracting the (k+1)-th largest value
    (instead of switching voices on/off) keeps the result continuous in time."""
    E = np.clip((E - floor) / (1 - floor), 0, 1) ** power
    if k < E.shape[1]:
        kth = -np.partition(-E, k, axis=1)[:, k:k + 1]
        E = np.maximum(E - kth, 0)
    E = gaussian_filter1d(E, smooth_frames, axis=0, mode="nearest")
    return np.clip(E / max(np.percentile(E, 99), 1e-9), 0, 1)


def up(x, t_ctrl, t_aud):
    """smooth (cubic) upsample of a control track to audio rate."""
    return interp1d(t_ctrl, x, kind="cubic", assume_sorted=True)(t_aud)


def pan_gains(p):  # p in -1..1, equal power
    a = (np.asarray(p) + 1) * np.pi / 4
    return np.cos(a), np.sin(a)


def leveler(x, strength=0.7, tau=0.25, percentile=60.0):
    """gentle two-sided compressor: passages louder than the typical level are pulled down."""
    a = np.exp(-1.0 / (tau * SR))
    p = lfilter([1 - a], [1, -a], (x ** 2).mean(axis=1))
    p = lfilter([1 - a], [1, -a], p[::-1])[::-1]                     # forward+backward = no lag
    env = np.sqrt(p) + 1e-9
    g = np.minimum(1.0, np.percentile(env, percentile) / env) ** strength
    return x * g[:, None]


def reverb(x, rng, tau=0.9, length=2.6, wet=0.30):
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.empty_like(x)
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / tau)
        ir /= np.sqrt((ir ** 2).sum())
        out[:, ch] = (1 - wet) * x[:, ch] + wet * fftconvolve(x[:, ch], ir)[: len(x)]
    return out


def fades(x, fin=0.6, fout=1.5):
    n = len(x)
    a, b = int(fin * SR), int(fout * SR)
    w = np.ones(n)
    w[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    w[-b:] = np.minimum(w[-b:], 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b)))
    return x * w[:, None]


def ocean_waves(n, rng, motion, period=(6.0, 10.0), rise=(1.8, 2.6), decay=(3.0, 4.5), amp=(0.6, 1.0),
                floor=0.12, body_cutoff=450.0, foam_band=(600.0, 3000.0), foam_power=2.5,
                body_gain=0.7, foam_gain=0.45, motion_influence=0.4):
    """white noise shaped into slow ocean swells: dark rumble in the trough, bright foam on the crest."""
    t = np.arange(n) / SR
    env = np.full(n, floor)                                        # never fully silent between waves
    s = rng.uniform(0.0, 2.0)
    while s < t[-1]:
        tau = t - s
        r_, d_, a_ = rng.uniform(*rise), rng.uniform(*decay), rng.uniform(*amp)
        r = np.clip(tau / r_, 0, 1)
        env += np.where(tau > 0, a_ * r * r * (3 - 2 * r) * np.where(tau > r_, np.exp(-(tau - r_) / d_), 1.0), 0)
        s += rng.uniform(*period)                                  # a new wave every `period` seconds
    env /= env.max()
    body_f = butter(2, body_cutoff, "lp", fs=SR, output="sos")
    foam_f = butter(2, list(foam_band), "bp", fs=SR, output="sos")
    gain = (1.0 - motion_influence) + motion_influence * motion / max(motion.max(), 1e-9)
    out = np.empty((n, 2))
    for ch in range(2):
        body = sosfilt(body_f, rng.standard_normal(n))
        foam = sosfilt(foam_f, rng.standard_normal(n))
        out[:, ch] = gain * (body_gain * env * body + foam_gain * env ** foam_power * foam)
    return out


def _k_sos(fs=SR):
    """ITU-R BS.1770 K-weighting (shelf + high-pass) as second-order sections."""
    out = []
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / fs); Vh = 10 ** (G / 20); Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    out.append([(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0,
                1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / fs); a0 = 1 + K / Q + K * K
    out.append([1, -2, 1, 1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])
    return np.array(out)


def k_loudness(x, fs=SR):
    """gated K-weighted loudness in dB (LUFS-like): how loud a stem SOUNDS while it is playing."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    y = sosfilt(_k_sos(fs), m)
    k = int(0.4 * fs); n = len(y) // k
    if n == 0:
        return -120.0
    ms = (y[: n * k].reshape(n, k) ** 2).mean(axis=1)
    ms = ms[ms > 1e-10]
    if len(ms) == 0:
        return -120.0
    g = ms[ms > 0.1 * ms.mean()]                                   # relative gate: ignore blocks 10 dB below average
    return float(-0.691 + 10 * np.log10(g.mean()))


def _env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def _bp(rng, n, band, order=2):
    return sosfilt(butter(order, list(band), "bp", fs=SR, output="sos"), rng.standard_normal(n))


def hit_kick(f0, f1, sweep, decay):
    n = int(min(6 * decay, 1.0) * SR); t = np.arange(n) / SR
    ph = 2 * np.pi * (f1 * t + (f0 - f1) * sweep * (1 - np.exp(-t / sweep)))
    return np.sin(ph) * _env(n, 0.002, decay)


def hit_snare(rng, tone_hz, band, decay, tone_gain):
    n = int(min(6 * decay, 0.8) * SR); t = np.arange(n) / SR
    noise = _bp(rng, n, band); noise /= max(np.abs(noise).max(), 1e-9)
    return noise * _env(n, 0.001, decay) + tone_gain * np.sin(2 * np.pi * tone_hz * t) * _env(n, 0.001, 0.6 * decay)


def hit_hat(rng, band, decay):
    n = int(min(6 * decay, 0.4) * SR)
    noise = _bp(rng, n, band); noise /= max(np.abs(noise).max(), 1e-9)
    return noise * _env(n, 0.0008, decay)


def hit_pluck(freq, decay, partial_gain, partial_ratio, attack):
    n = int(min(6 * decay, 1.5) * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * t) + partial_gain * np.sin(2 * np.pi * partial_ratio * freq * t)) * _env(n, attack, decay)


def rhythm_schedule(c, bpm, swing, threshold, motion_boost, kick_pattern, snare_pattern, hat_pattern,
                    sparkle_percentile, voices, act=None, flux=None):
    """decide WHICH hits play at WHICH grid time.  Returns (times, kinds, velocities, bands) as python lists."""
    step = 60.0 / bpm / 4.0                                         # one 16th note
    t_end = float(c["t"][-1]) - 1.5
    tc = c["t"].astype(float)
    mot = c["motion"].astype(float)
    on = pool(c["onsets"], voices)
    flux = (c["flux"] if flux is None else flux).astype(float)
    flux_thr = max(np.percentile(flux, sparkle_percentile), 0.05)
    if act is None:
        act = {"kick": c["zone_low"], "snare": c["zone_mid"], "hat": c["texture"]}
    pat = {"kick": kick_pattern, "snare": snare_pattern, "hat": hat_pattern}
    res = []
    j = 0
    while True:
        t0 = j * step + (swing * step if j % 2 else 0.0)            # swing delays every 2nd 16th
        if t0 > t_end:
            break
        i = int(np.searchsorted(tc, t0)); i = min(i, len(tc) - 1)
        lo, hi = max(i - 1, 0), min(i + 2, len(tc))
        for kind in ("kick", "snare", "hat"):
            w = pat[kind][j % 16]
            if w <= 0:
                continue
            a = min(float(act[kind][i]) + motion_boost * float(mot[i]), 1.0)
            v = a * w
            if v >= threshold:
                res.append((t0, kind, 0.3 + 0.7 * min((v - threshold) / (1 - threshold), 1.0), j))
        w_ = max(i - int(round(step * 50 / 2)), 0), min(i + int(round(step * 50 / 2)) + 1, len(tc))
        fl = float(flux[w_[0]:w_[1]].max())
        need = flux_thr if j % 2 == 0 else flux_thr + 0.5 * (1 - flux_thr)   # in-between 16ths: only very strong onsets
        if fl >= need:
            b = int(np.argmax(on[w_[0]:w_[1]].max(axis=0)))
            res.append((t0, "sparkle", 0.3 + 0.7 * min((fl - need) / max(1 - need, 1e-6) * 0.5 + 0.5 * (fl >= flux_thr), 1.0), b))
        j += 1
    return res


def render_rhythm(c, n, f_notes, rng, bpm, swing, threshold, motion_boost, kick_pattern, snare_pattern, hat_pattern,
                  kick_freq_start, kick_freq_end, kick_sweep, kick_decay, snare_tone_hz, snare_noise_band,
                  snare_decay, snare_tone_gain, hat_band, hat_decay, beat_gain, beat_pan_width, beat_reverb_wet,
                  sparkle_percentile, sparkle_shift_semitones, sparkle_decay, sparkle_partial_gain,
                  sparkle_partial_ratio, sparkle_attack, pans, reverb_tau, reverb_length, act=None, flux=None,
                  split=False):
    """returns (beat_stem, sparkle_stem, counts)  -  or ({kick, snare, hat}, sparkle_stem, counts) when split."""
    V = len(f_notes)
    sched = rhythm_schedule(c, bpm, swing, threshold, motion_boost, kick_pattern, snare_pattern, hat_pattern,
                            sparkle_percentile, V, act, flux)
    kick = np.zeros((n, 2)); sh = np.zeros((n, 2)); sp = np.zeros((n, 2)); hh = np.zeros((n, 2))
    counts = {"kick": 0, "snare": 0, "hat": 0, "sparkle": 0}
    hat_side = 1.0

    def put(buf, t0, sig, gain, pan):
        i0 = int(t0 * SR)
        if i0 >= n:
            return
        m = min(len(sig), n - i0)
        L, R = pan_gains(pan)
        buf[i0:i0 + m, 0] += float(L) * gain * sig[:m]
        buf[i0:i0 + m, 1] += float(R) * gain * sig[:m]

    for t0, kind, vel, extra in sched:
        counts[kind] += 1
        if kind == "kick":
            put(kick, t0, hit_kick(kick_freq_start, kick_freq_end, kick_sweep, kick_decay), vel * beat_gain["kick"], 0.0)
        elif kind == "snare":
            put(sh, t0, hit_snare(rng, snare_tone_hz, snare_noise_band, snare_decay, snare_tone_gain),
                vel * beat_gain["snare"], 0.0)
        elif kind == "hat":
            put(hh if split else sh, t0, hit_hat(rng, hat_band, hat_decay), vel * beat_gain["hat"], hat_side * beat_pan_width)
            hat_side = -hat_side
        else:
            fz = f_notes[extra] * 2.0 ** (sparkle_shift_semitones / 12.0)
            put(sp, t0, hit_pluck(fz, sparkle_decay, sparkle_partial_gain, sparkle_partial_ratio, sparkle_attack),
                vel, float(pans[extra]))
    if split:
        beat = {"kick": kick, "snare": reverb(sh, rng, reverb_tau, reverb_length, beat_reverb_wet),
                "hat": reverb(hh, rng, reverb_tau, reverb_length, beat_reverb_wet)}
    else:
        beat = kick + reverb(sh, rng, reverb_tau, reverb_length, beat_reverb_wet)
    sparkle = reverb(sp, rng, reverb_tau, reverb_length, 0.30)
    return beat, sparkle, counts


class Stems(dict):
    """dict {voice: (n,2) array} that also remembers how many drum / sparkle hits were played (.counts)."""
    counts = None


def render_layer(c: dict, root: str = "D", mode: str = "pentatonic", shift: int = 0, seed: int = 0,
                 voices: int = 12, event_voices: int = 3, air_mode: str = "waves",
                 base_midi: int = BASE_MIDI, voice_gain: dict | None = None,
                 # --- pad voice ---
                 pad_pan_width: float = 0.7, pad_amp_exponent: float = 1.5,
                 pad_rolloff_exponent: float = 0.75, pad_rolloff_ref_hz: float = 220.0,
                 pad_harmonic_base: float = 0.10, pad_harmonic_hue: float = 0.45,
                 # --- events voice ---
                 event_shift_semitones: float = 12.0, event_floor: float = 0.2, event_power: float = 2.0,
                 event_smooth_frames: float = 3.0, event_partial_gain: float = 0.25,
                 event_partial_ratio: float = 2.0,
                 # --- drone voice ---
                 drone_fifth_ratio: float = 1.5, drone_loudness_power: float = 2.0,
                 drone_harmonic_base: float = 0.15, drone_harmonic_warmth: float = 0.45,
                 # --- air voice: waves ---
                 wave_period: tuple = (6.0, 10.0), wave_rise: tuple = (1.8, 2.6), wave_decay: tuple = (3.0, 4.5),
                 wave_amp: tuple = (0.6, 1.0), wave_floor: float = 0.12, wave_body_cutoff: float = 450.0,
                 wave_foam_band: tuple = (600.0, 3000.0), wave_foam_power: float = 2.5,
                 wave_body_gain: float = 0.7, wave_foam_gain: float = 0.45, wave_motion_influence: float = 0.4,
                 # --- air voice: hiss ---
                 hiss_cutoff: float = 3000.0, hiss_motion_power: float = 1.5,
                 # --- leveler / reverb / fades / stem level ---
                 leveler_strength: float = 0.7, leveler_tau: float = 0.25, leveler_percentile: float = 60.0,
                 reverb_wet: float = 0.30, reverb_tau: float = 0.9, reverb_length: float = 2.6,
                 fade_in: float = 0.6, fade_out: float = 1.5, stem_ref_percentile: float = 99.9,
                 # --- rhythm: beat + sparkle voices (off = identical to the earlier four-voice sound) ---
                 rhythm: str = "off", rhythm_bpm: float = 100.0, rhythm_swing: float = 0.0,
                 rhythm_threshold: float = 0.25, rhythm_motion_boost: float = 0.25,
                 kick_pattern: list = None, snare_pattern: list = None, hat_pattern: list = None,
                 kick_freq_start: float = 130.0, kick_freq_end: float = 50.0, kick_sweep: float = 0.06,
                 kick_decay: float = 0.28, snare_tone_hz: float = 190.0, snare_noise_band: tuple = (1200.0, 5000.0),
                 snare_decay: float = 0.14, snare_tone_gain: float = 0.5, hat_band: tuple = (5000.0, 9000.0),
                 hat_decay: float = 0.035, beat_gain: dict | None = None, beat_pan_width: float = 0.3,
                 beat_reverb_wet: float = 0.12, sparkle_percentile: float = 65.0,
                 sparkle_shift_semitones: float = 24.0, sparkle_decay: float = 0.30,
                 sparkle_partial_gain: float = 0.20, sparkle_partial_ratio: float = 3.0, sparkle_attack: float = 0.003,
                 # --- loudness balance of the voices ---
                 balance: str = "peak", balance_offsets_db: dict | None = None,
                 # --- which picture feature drives which voice / separate drum stems / progress reporting ---
                 mapping: dict | None = None, beat_split: bool = False, progress=None):
    """Render one control file (dict of arrays from presonify) into stereo stems.  Returns {voice: (n,2) array}.
    mapping    {voice: [(feature, weight), ...]} - see DEFAULT_MAPPING; missing voices keep their default driver.
    beat_split True = kick / snare / hat become three stems instead of one "beat" stem.
    progress   optional function(fraction 0..1, text); it may raise to cancel."""
    P = progress or (lambda f, m="": None)
    mp = dict(DEFAULT_MAPPING)
    if mapping:
        mp.update(mapping)
    c = with_derived(c)
    P(0.02, "preparing")
    vg = dict(VOICE_GAIN)
    if voice_gain:
        vg.update(voice_gain)
    rng = np.random.default_rng(seed)
    t_c = c["t"].astype(float)
    n = int(t_c[-1] * SR)
    t = np.arange(n) / SR
    bg = pool(resolve(c, mp["pad"], "matrix"), voices)
    ev = sparsify(pool(resolve(c, mp["events"], "matrix"), voices), event_voices, event_floor, event_power, event_smooth_frames)
    V = bg.shape[1]
    f = mtof(ladder(V, root, mode, base_midi + shift))
    pans = np.linspace(pad_pan_width, -pad_pan_width, V)           # voice 0 (high) right, last (low) left
    pad = np.zeros((n, 2)); evs = np.zeros((n, 2))

    colour = np.clip(up(resolve(c, mp["pad_timbre"], "scalar"), t_c, t), 0, 1)

    for b in range(V):
        P(0.05 + 0.40 * b / V, "pad and bells")
        L, R = pan_gains(pans[b])
        a = np.clip(up(bg[:, b], t_c, t), 0, None) ** pad_amp_exponent * (pad_rolloff_ref_hz / f[b]) ** pad_rolloff_exponent
        ph = 2 * np.pi * f[b] * t
        s = a * (np.sin(ph) + (pad_harmonic_base + pad_harmonic_hue * colour) * np.sin(2 * ph))
        pad[:, 0] += L * s; pad[:, 1] += R * s
        e = np.clip(up(ev[:, b], t_c, t), 0, None)
        fe = f[b] * 2.0 ** (event_shift_semitones / 12.0)          # bell pitch = pad note shifted up
        be = e * (np.sin(2 * np.pi * fe * t) + event_partial_gain * np.sin(2 * np.pi * event_partial_ratio * fe * t))
        evs[:, 0] += L * be; evs[:, 1] += R * be

    P(0.46, "drone")
    loud = np.clip(up(resolve(c, mp["drone"], "scalar"), t_c, t), 0, 1) ** drone_loudness_power
    warm = np.clip(up(resolve(c, mp["drone_timbre"], "signed"), t_c, t), -1, 1)
    root_f = mtof(ladder(1, root, mode, base_midi + shift)[0])
    dr = np.zeros(n)
    for ff in (root_f, root_f * drone_fifth_ratio):
        ph = 2 * np.pi * ff * t
        dr += np.sin(ph) + (drone_harmonic_base + drone_harmonic_warmth * (0.5 + 0.5 * warm)) * np.sin(2 * ph)
    drone = (loud * dr)[:, None] * np.ones((1, 2))

    P(0.52, "waves")
    motion = np.clip(up(resolve(c, mp["air"], "scalar"), t_c, t), 0, 1)
    if motion.max() <= 1e-6 or air_mode == "off":
        air = np.zeros((n, 2))                                     # stills / switched off: nothing to play
    elif air_mode == "hiss":
        sos = butter(4, hiss_cutoff, "hp", fs=SR, output="sos")
        air = np.stack([sosfilt(sos, rng.standard_normal(n)) for _ in range(2)], 1) * motion[:, None] ** hiss_motion_power
    else:
        air = ocean_waves(n, rng, motion, wave_period, wave_rise, wave_decay, wave_amp, wave_floor,
                          wave_body_cutoff, wave_foam_band, wave_foam_power, wave_body_gain, wave_foam_gain,
                          wave_motion_influence)

    P(0.60, "reverb")
    stems = {"pad": reverb(leveler(pad, leveler_strength, leveler_tau, leveler_percentile), rng,
                           reverb_tau, reverb_length, reverb_wet),
             "events": reverb(leveler(evs, leveler_strength, leveler_tau, leveler_percentile), rng,
                              reverb_tau, reverb_length, reverb_wet),
             "drone": leveler(drone, leveler_strength, leveler_tau, leveler_percentile),
             "air": air}
    counts = None
    if rhythm == "on":
        if "onsets" not in c:
            raise ValueError("this control file has no rhythm tracks - re-run presonify.py (new version) first")
        rr = np.random.default_rng(seed + 1000)                    # separate generator: old voices stay unchanged
        bg_ = dict(kick=1.0, snare=0.7, hat=0.35)
        if beat_gain:
            bg_.update(beat_gain)
        P(0.72, "beat and plucks")
        beat, sparkle, counts = render_rhythm(
            c, n, f, rr, rhythm_bpm, rhythm_swing, rhythm_threshold, rhythm_motion_boost,
            kick_pattern or KICK_PATTERN, snare_pattern or SNARE_PATTERN, hat_pattern or HAT_PATTERN,
            kick_freq_start, kick_freq_end, kick_sweep, kick_decay, snare_tone_hz, snare_noise_band, snare_decay,
            snare_tone_gain, hat_band, hat_decay, bg_, beat_pan_width, beat_reverb_wet, sparkle_percentile,
            sparkle_shift_semitones, sparkle_decay, sparkle_partial_gain, sparkle_partial_ratio, sparkle_attack,
            pans, reverb_tau, reverb_length,
            {"kick": resolve(c, mp["kick"], "scalar"), "snare": resolve(c, mp["snare"], "scalar"),
             "hat": resolve(c, mp["hat"], "scalar")}, resolve(c, mp["sparkle"], "scalar"), beat_split)
        if beat_split:
            stems.update(beat)
        else:
            stems["beat"] = beat
        stems["sparkle"] = sparkle
    if balance not in ("peak", "loudness"):
        raise ValueError("balance must be 'peak' or 'loudness'")
    offs = dict(BALANCE_OFFSETS_DB)
    if balance_offsets_db:
        offs.update(balance_offsets_db)
    P(0.88, "balancing levels")
    pad_lk = -23.0                                                 # fallback if the pad is silent
    for k in ["pad"] + [k for k in stems if k != "pad"]:           # pad first: it is the loudness reference
        v = stems[k]
        ref = np.percentile(np.abs(v), stem_ref_percentile)
        if ref <= 1e-9:
            continue
        if balance == "loudness" and k != "pad":
            v = v / ref                                            # any scale; then set its K-weighted loudness
            stems[k] = fades(v * 10 ** ((pad_lk + offs[k] - k_loudness(v)) / 20), fade_in, fade_out)
        else:
            stems[k] = fades(v / ref * vg[k], fade_in, fade_out)
            if k == "pad":
                pad_lk = k_loudness(stems["pad"])
    P(1.0, "done")
    out = Stems(stems)
    out.counts = counts                                            # number of hits per kind (None when rhythm is off)
    return out


def rms_step_db(x, win=0.100):
    """largest level change between neighbouring 100 ms frames (dB).  The window spans several periods of the
    lowest notes (a shorter one measures oscillation phase, not loudness).  Fade-in/out is skipped."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    k = int(win * SR)
    m = m[int(1.0 * SR): len(m) - int(2.0 * SR)]
    fr = m[: len(m) // k * k].reshape(-1, k)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    ok = (db[1:] > -50) & (db[:-1] > -50)
    return float(np.abs(np.diff(db))[ok].max()) if ok.any() else 0.0


def to_wav(path: Path, x, fmt: str = "int16"):
    """fmt: 'int16' (16-bit PCM, plays everywhere) or 'float32' (32-bit float: no rounding, best for further mixing)."""
    if fmt == "float32":
        wavfile.write(str(path), SR, np.clip(x, -1, 1).astype(np.float32))
    else:
        wavfile.write(str(path), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def write_mix(out_prefix: Path, layers: list, peak_target: float = 0.89, write_stems: bool = True,
              stems_dir: Path | None = None, write_spectrogram: bool = True,
              spectrogram_max_hz: float = 6000.0, spectrogram_nfft: int = 4096,
              write_mix_file: bool = True, wav_format: str = "int16") -> str:
    """Sum layers into <out_prefix>_mix.wav (+ stems + spectrogram).  `layers` = [(name, stems, gain), ...].
    Mix and stems share ONE scale factor so re-mixing the stems in a DAW reproduces the mix.  Returns a summary line."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    N = max(len(v) for _, st, _ in layers for v in st.values())
    mix = np.zeros((N, 2))
    for _, st, g in layers:
        for v in st.values():
            mix[: len(v)] += g * v
    peak = np.abs(mix).max()
    k = peak_target / peak if peak > 0 else 1.0
    if write_mix_file:
        to_wav(Path(str(out_prefix) + "_mix.wav"), mix * k, wav_format)
    if write_stems:
        sd = Path(stems_dir) if stems_dir else out_prefix.parent
        sd.mkdir(parents=True, exist_ok=True)
        for name, st, g in layers:
            for vn, v in st.items():
                if np.abs(v).max() > 1e-6:                          # silent voices (e.g. air on a still) are skipped
                    pad = np.zeros((N, 2)); pad[: len(v)] = g * v * k
                    stem_name = f"{out_prefix.name}_{name}_{vn}.wav" if name else f"{out_prefix.name}_{vn}.wav"
                    to_wav(sd / stem_name, pad, wav_format)
    if write_spectrogram and write_mix_file:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
        ax.specgram((mix * k).mean(axis=1), NFFT=spectrogram_nfft, Fs=SR, noverlap=spectrogram_nfft * 3 // 4,
                    cmap="magma", vmin=-130)
        ax.set_ylim(0, spectrogram_max_hz); ax.set_xlabel("s"); ax.set_ylabel("Hz")
        ax.set_title(f"{out_prefix.name}_mix spectrogram")
        fig.savefig(str(out_prefix) + "_mix_spectrogram.png", dpi=100); plt.close(fig)
    return (f"{out_prefix.name}: {N / SR:.1f} s, {len(layers)} layer(s), peak {20*np.log10(peak_target):.1f} dBFS, "
            f"mean level {20*np.log10(np.sqrt(((mix*k)**2).mean())+1e-12):.1f} dBFS, "
            f"largest 100 ms level change {rms_step_db(mix*k):.1f} dB, NaN: {bool(np.isnan(mix).any())}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="*_ctrl.npz files from presonify.py")
    p.add_argument("-o", "--out", type=Path, required=True, help="output prefix, e.g. audio/mix")
    p.add_argument("--root", default="D", choices=list(ROOTS))
    p.add_argument("--mode", default="pentatonic", choices=list(MODES))
    p.add_argument("--voices", type=int, default=12, help="pad tones per layer (fewer = lower and simpler)")
    p.add_argument("--event-voices", type=int, default=3, help="bright spots allowed to ring at the same moment")
    p.add_argument("--gain", type=float, nargs="*", help="one gain per input file (default 1)")
    p.add_argument("--air", nargs="*", default=["waves"], choices=["waves", "hiss", "off"],
                   help="per input file: waves (default) | hiss | off")
    p.add_argument("--rhythm", choices=["on", "off"], default="off", help="add beat + sparkle voices")
    p.add_argument("--bpm", type=float, default=100.0, help="tempo of the beat grid")
    p.add_argument("--balance", choices=["peak", "loudness"], default="peak",
                   help="loudness = set every voice to a perceptual (K-weighted) level relative to the pad")
    p.add_argument("--no-stems", action="store_true")
    a = p.parse_args(argv)
    gains = (a.gain or []) + [1.0] * len(a.inputs)
    airs = a.air + [a.air[-1]] * len(a.inputs)

    layers = []
    for i, path in enumerate(a.inputs):
        c = {k: v for k, v in np.load(path, allow_pickle=False).items()}
        name = path.stem.replace("_ctrl", "")
        layers.append((name, render_layer(c, a.root, a.mode, LAYER_SHIFT[i % 3], i, a.voices, a.event_voices, airs[i], rhythm=a.rhythm, rhythm_bpm=a.bpm, balance=a.balance),
                       gains[i]))
    print(write_mix(a.out, layers, write_stems=not a.no_stems))


if __name__ == "__main__":
    main()
