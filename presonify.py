#!/usr/bin/env python3
"""
presonify.py - turn ANY image or video into one smooth, source-agnostic set of
control tracks ("pre-sonification output") that a synthesizer can read directly.

Design rules
------------
1. ONE representation for everything.  A still image is a video with one frame.
   Audio time (tau) drives two things at once: a scan position x(tau) across the
   picture, and (videos only) the playback position t(tau).
       still : x sweeps 0 -> 1 over the duration            (a longitude sweep)
       video : x sweeps AND the video plays at the same time (diagonal cut through
               the space-time cube), or x stays fixed ('--scan fixed')
   Scan modes:  sweep     constant speed left -> right (default)
                variable  same path, but the speed changes with position: --scan-ratio 4 means 4x faster at the
                          edges than in the middle (below 1 = slower at the edges); --scan-curve sets the shape
                fixed     one column, at --scan-position (0 = left edge, 0.5 = middle, 1 = right edge)
2. Stills must sound good ALONE.  All primary tracks depend on spatial structure
   only.  Time-dependent tracks (motion) are an optional extra layer: for a still
   they are simply zero, nothing breaks.
3. No abrupt changes by construction, not by luck:
     - spatial pre-filter (Gaussian blur) before any column is read,
     - rows pooled into B bands with area averaging (anti-aliased),
     - robust percentile normalisation + log compression (hot spots cannot
       dominate or clip),
     - every track is low-passed (zero-phase Gaussian) at the control rate,
     - hue is smoothed as sin/cos, never as an angle (no 359 -> 0 jumps),
     - the scan ping-pongs instead of jumping back to the start.
4. Sharp features are not removed, they are separated.  Each band track is split
   into a slow BACKGROUND layer and a positive EVENTS layer (difference of
   Gaussians) - bright spots become soft swells that can drive their own voice.

Output  <prefix>_ctrl.npz  (all float32, N = duration * rate control frames)
    t            (N,)    seconds
    x_pos        (N,)    scan position 0..1 (left..right)
    frame_pos    (N,)    video position 0..1 (0 for stills)
    background   (N, B)  0..1  smooth per-band energy;  band 0 = TOP rows
    events       (N, B)  0..1  salient bright spots (DoG, positive part)
    loudness     (N,)    0..1  mean brightness of the scanned column
    centroid     (N,)    0..1  brightness-weighted vertical position (0 = top)
    spread       (N,)    0..1  vertical spread of brightness
    saturation   (N,)    0..1
    hue          (N,)    0..1  (circular, brightness-weighted; gaps bridged where undefined)
    chroma       (N,)    0..1  how colourful the column is = how much to trust `hue`
    warmth       (N,)    -1..1 mean (R - B)
    motion       (N,)    0..1  frame-to-frame change (0 for stills)
  -- rhythm / detail tracks (work identically for stills and videos) --
    onsets       (N, B)  0..1  "something just got brighter" per band (positive time-derivative of lightly smoothed bands)
    flux         (N,)    0..1  overall onset strength = how fast the scanned column is changing (edges, hot spots, cuts)
    texture      (N,)    0..1  fine detail / edge density inside the scanned column (smooth sky = 0, busy coastline = 1)
    zone_low     (N,)    0..1  brightness of the BOTTOM third of the picture   (drives kick)
    zone_mid     (N,)    0..1  brightness of the MIDDLE third                  (drives snare)
    zone_high    (N,)    0..1  brightness of the TOP third                     (drives hats)
    meta_json    string  all parameters + normalisation limits
Also writes <prefix>_ctrl.png (preview).

Examples
    python presonify.py Aerosols_..._72dpi.png -o output/aerosol
    python presonify.py ecco2_sst_flow.webmhd.webm -o output/ocean --duration 40
    python presonify.py dyamond_co2.mp4 -o output/co2 --scan fixed
    python presonify.py clip.mp4 -o output/clip --passes 3 --smooth 0.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# --------------------------------------------------------------------------- #
# source access (streaming: never holds a whole video in memory)
# --------------------------------------------------------------------------- #
def open_source(path: Path):
    """Return (is_video, n_frames, fps, frame_iterator_factory)."""
    if path.suffix.lower() in IMG_EXT:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {path}")
        return False, 1, 0.0, lambda: iter([(0, img)])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    def frames():
        c = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, f = c.read()
            if not ok:
                break
            yield i, f
            i += 1
        c.release()
    return True, n, fps, frames


def prepare_frame(bgr: np.ndarray, work_w: int, blur: float):
    """BGR uint8 -> blurred float32 RGB 0..1 at working resolution + HSV."""
    h, w = bgr.shape[:2]
    if w > work_w:
        bgr = cv2.resize(bgr, (work_w, int(round(h * work_w / w))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if blur > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=blur * rgb.shape[1], sigmaY=blur * rgb.shape[0])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)            # H in degrees 0..360
    return rgb, hsv


# --------------------------------------------------------------------------- #
# raw (unsmoothed) features of one column
# --------------------------------------------------------------------------- #
def column_features(rgb, hsv, x: int, n_bands: int, fine=None):
    col = rgb[:, x, :]                                    # (H,3)
    lum = col @ LUMA                                      # (H,)
    bands = cv2.resize(lum.reshape(-1, 1), (1, n_bands), interpolation=cv2.INTER_AREA).ravel()
    w = lum + 1e-6
    y = np.linspace(0.0, 1.0, len(lum), dtype=np.float32)
    centroid = float((w * y).sum() / w.sum())
    spread = float(np.sqrt((w * (y - centroid) ** 2).sum() / w.sum()))
    sat = hsv[:, x, 1]
    ang = np.deg2rad(hsv[:, x, 0])
    ws = w * sat
    hs, hc = float((ws * np.sin(ang)).mean()), float((ws * np.cos(ang)).mean())   # zero vector if black
    return dict(bands=bands, loud=float(lum.mean()), centroid=centroid, spread=spread,
                sat=float((w * sat).sum() / w.sum()), hue_s=hs, hue_c=hc,
                warmth=float((col[:, 0] - col[:, 2]).mean()),
                texture=0.0 if fine is None else float(np.abs(np.diff(fine[:, x])).mean()))


def robust01(a: np.ndarray, lo_p=1.0, hi_p=99.5, k=8.0):
    """percentile-normalise to 0..1 then log-compress (heavy-tailed hot spots)."""
    lo, hi = np.percentile(a, lo_p), np.percentile(a, hi_p)
    z = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return (np.log1p(k * z) / np.log1p(k)).astype(np.float32), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
SCAN_CURVES = {"cosine": lambda d: (1 - np.cos(np.pi * d)) / 2, "linear": lambda d: d,
               "quadratic": lambda d: d ** 2, "quartic": lambda d: d ** 4}


def scan_path(u, scan="sweep", passes=1, position=0.5, curve="cosine", ratio=4.0):
    """scan position x (0 = left edge ... 1 = right edge) for audio-time fractions u (0..1).
    sweep: constant speed;  variable: speed profile v(x) = 1 + (ratio - 1) * shape(|2x - 1|), i.e. `ratio` times the
    centre speed at the edges;  fixed: always `position`."""
    u = np.asarray(u, float)
    if scan == "fixed":
        return np.full(len(u), float(np.clip(position, 0.0, 1.0)))
    s = u * passes
    w = 1 - np.abs(1 - (s % 2.0)) if passes > 1 else u                 # ping-pong triangle, never a jump
    if scan != "variable":
        return w
    xs = np.linspace(0.0, 1.0, 4001)
    v = 1.0 + (float(ratio) - 1.0) * SCAN_CURVES[curve](np.abs(2 * xs - 1))
    tt = np.concatenate([[0.0], np.cumsum(0.5 * (1 / v[1:] + 1 / v[:-1]))])   # time needed to reach x
    tt /= tt[-1]
    return np.interp(w, tt, xs)                                          # invert: position reached at time w


def presonify(path: Path, duration=30.0, rate=50.0, n_bands=24, blur=0.006, scan="sweep",
              passes=1, smooth=0.25, work_w=960, event_scale=4.0,
              norm_low_pct=1.0, norm_high_pct=99.5, compress_k=8.0, motion_pct=99.0,
              event_pct=99.0, band_blur=0.8, background_band_blur=1.5, hue_trust=0.03,
              texture_blur=0.0015, flux_smooth=0.06, onset_pct=99.0, zone_smooth=0.12, texture_k=4.0,
              progress=None, preview_frames=0, preview_width=360,
              scan_position=0.5, scan_curve="cosine", scan_ratio=4.0):
    is_video, n_frames, fps, frames = open_source(path)
    N = int(round(duration * rate))
    u = np.linspace(0.0, 1.0, N, dtype=np.float64)
    x_pos = scan_path(u, scan, passes, scan_position, scan_curve, scan_ratio)
    f_idx = np.rint(u * (n_frames - 1)).astype(int) if is_video else np.zeros(N, int)
    frame_pos = f_idx / max(n_frames - 1, 1)

    raw = {k: [None] * N for k in ("bands", "loud", "centroid", "spread", "sat", "hue_s", "hue_c", "warmth", "motion", "texture")}
    steps_of = {}
    for i, f in enumerate(f_idx):
        steps_of.setdefault(int(f), []).append(i)

    prev_small, motion_of = None, {}
    last = int(f_idx.max())
    # --- optional: small copies of the video frames that will actually be read, for the GUI preview ----------- #
    want, pv, pv_size, src_size = {}, [], None, None
    if is_video and preview_frames > 0:
        steps = np.rint(np.linspace(0, N - 1, min(int(preview_frames), N))).astype(int)
        pv = [None] * len(steps)
        for k, st in enumerate(steps):
            want.setdefault(int(f_idx[st]), []).append(k)
    for fi, bgr in frames():
        if fi > last:
            break
        if src_size is None:
            src_size = (int(bgr.shape[1]), int(bgr.shape[0]))
        if fi in want:
            h_, w_ = bgr.shape[:2]
            small_rgb = cv2.cvtColor(cv2.resize(bgr, (preview_width, max(1, int(round(h_ * preview_width / w_)))),
                                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            for k in want[fi]:
                pv[k] = small_rgb
        if progress is not None and (fi % 6 == 0 or fi == last):
            progress(fi / max(last, 1), "reading frames")             # may raise to cancel
        rgb, hsv = prepare_frame(bgr, work_w, blur)
        small = cv2.resize(rgb @ LUMA, (96, 48), interpolation=cv2.INTER_AREA)
        motion_of[fi] = 0.0 if prev_small is None else float(np.abs(small - prev_small).mean())
        prev_small = small
        fine = None
        for i in steps_of.get(fi, []):
            x = int(round(x_pos[i] * (rgb.shape[1] - 1)))
            if fine is None:                                  # lightly blurred luminance: keeps the fine detail
                fine = prepare_frame(bgr, work_w, texture_blur)[0] @ LUMA
            r = column_features(rgb, hsv, x, n_bands, fine)
            for k, v in r.items():
                key = {"loud": "loud"}.get(k, k)
                raw[key][i] = v
            raw["motion"][i] = motion_of[fi]
    A = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    # --- normalise (robust) ------------------------------------------------- #
    bands, lo, hi = robust01(A["bands"], norm_low_pct, norm_high_pct, compress_k)
    loud, _, _ = robust01(A["loud"], norm_low_pct, norm_high_pct, compress_k)
    motion = A["motion"] / max(np.percentile(A["motion"], motion_pct), 1e-9) if is_video else np.zeros(N, np.float32)
    motion = np.clip(motion, 0, 1).astype(np.float32)
    raw_loud = loud.copy()                                # kept for the preview comparison
    raw_bands = bands.copy()

    # --- smooth: zero-phase Gaussian along the control axis ------------------ #
    sg = smooth * rate                                    # sigma in control frames
    sm = lambda a, s=sg: gaussian_filter1d(a, s, axis=0, mode="nearest")
    S = gaussian_filter(bands, sigma=(sg, band_blur), mode="nearest")              # time x band
    background = gaussian_filter(S, sigma=(sg * event_scale, background_band_blur), mode="nearest")
    events = np.maximum(S - background, 0.0)
    events = np.clip(events / max(np.percentile(events, event_pct), 1e-9), 0, 1)
    # --- rhythm / detail tracks ------------------------------------------------- #
    fg = flux_smooth * rate                                                      # much shorter smoothing than `smooth`
    light = gaussian_filter(bands, sigma=(fg, band_blur), mode="nearest")       # fast enough to keep onsets sharp
    d = np.maximum(np.diff(light, axis=0, prepend=light[:1]), 0.0)              # only "getting brighter"
    onsets = np.clip(d / max(np.percentile(d, onset_pct), 1e-9), 0, 1) ** 0.5
    flux = onsets.mean(axis=1)
    flux = np.clip(flux / max(np.percentile(flux, onset_pct), 1e-9), 0, 1)
    tex, _, _ = robust01(A["texture"], norm_low_pct, norm_high_pct, texture_k)
    texture = gaussian_filter1d(tex, fg, mode="nearest")
    texture = np.clip((texture - texture.min()) / max(np.percentile(texture, 99) - texture.min(), 1e-9), 0, 1)
    zs = zone_smooth * rate
    thirds = np.array_split(np.arange(n_bands), 3)                              # band 0 = top rows
    zone = {}
    for name, ix in zip(("zone_high", "zone_mid", "zone_low"), thirds):
        z = gaussian_filter1d(light[:, ix].mean(axis=1), zs, mode="nearest")
        lo_, hi_ = np.percentile(z, 5), np.percentile(z, 99)
        zone[name] = np.clip((z - lo_) / max(hi_ - lo_, 1e-9), 0, 1)
    hs_, hc_ = sm(A["hue_s"]), sm(A["hue_c"])                                  # smooth circular mean as vector
    mag = np.hypot(hs_, hc_)
    ok = mag > hue_trust * max(np.percentile(mag, 99), 1e-9)                       # hue is meaningless on black/grey
    chroma = np.clip(mag / max(np.percentile(mag, 99), 1e-9), 0, 1)
    if ok.sum() >= 2:                                                         # bridge gaps on the unit circle
        idx = np.arange(N)
        ux = np.interp(idx, idx[ok], (hs_ / np.maximum(mag, 1e-12))[ok])
        uy = np.interp(idx, idx[ok], (hc_ / np.maximum(mag, 1e-12))[ok])
    else:
        ux, uy = np.zeros(N), np.ones(N)
    hue = (np.arctan2(ux, uy) / (2 * np.pi)) % 1.0

    out = dict(t=(np.arange(N) / rate).astype(np.float32), x_pos=x_pos.astype(np.float32),
               frame_pos=frame_pos.astype(np.float32), background=background.astype(np.float32),
               events=events.astype(np.float32), loudness=sm(loud),
               centroid=sm(A["centroid"]), spread=sm(A["spread"]), saturation=sm(A["sat"]),
               hue=hue.astype(np.float32), chroma=chroma.astype(np.float32), warmth=sm(A["warmth"]), motion=sm(motion),
               onsets=onsets.astype(np.float32), flux=flux.astype(np.float32), texture=texture.astype(np.float32),
               **{k: v.astype(np.float32) for k, v in zone.items()})
    meta = dict(source=path.name, is_video=is_video, n_frames=n_frames, fps=fps, duration_s=duration,
                control_rate_hz=rate, n_bands=n_bands, blur_frac=blur, scan=scan, passes=passes, scan_position=scan_position, scan_curve=scan_curve,
                scan_ratio=scan_ratio,
                smooth_s=smooth, event_scale=event_scale, band_row0="top", work_width_px=work_w,
                norm_low_pct=norm_low_pct, norm_high_pct=norm_high_pct, compress_k=compress_k, motion_pct=motion_pct,
                event_pct=event_pct, band_blur=band_blur, background_band_blur=background_band_blur, hue_trust=hue_trust,
                texture_blur=texture_blur, flux_smooth=flux_smooth, onset_pct=onset_pct, zone_smooth=zone_smooth, texture_k=texture_k,
                band_norm_p1_p99_5=[lo, hi], note="raw_loudness kept in npz for comparison")
    meta["source_w"], meta["source_h"] = (src_size or (0, 0))
    if pv:                                                      # fill frames that could not be decoded with the previous one
        prevf = next((f for f in pv if f is not None), None)
        for k in range(len(pv)):
            if pv[k] is None:
                pv[k] = prevf
            prevf = pv[k]
        out['_preview'] = np.stack(pv)                          # (K, h, w, 3) uint8, K evenly spaced along the audio time
    out['_raw_bands'] = raw_bands
    out['_S'] = S.astype(np.float32)
    return out, raw_loud, meta


def roughness(a: np.ndarray, rate: float, f_cut: float = 4.0) -> float:
    """share of signal power above f_cut Hz (0 = perfectly smooth, 1 = pure noise)."""
    a = a - a.mean()
    p = np.abs(np.fft.rfft(a)) ** 2
    f = np.fft.rfftfreq(len(a), 1 / rate)
    return float(p[f > f_cut].sum() / max(p.sum(), 1e-12))


def band_roughness(M: np.ndarray, rate: float, f_cut: float = 4.0) -> float:
    """power above f_cut Hz / total power, pooled over all bands (time axis = 0)."""
    M = M - M.mean(axis=0, keepdims=True)
    p = np.abs(np.fft.rfft(M, axis=0)) ** 2
    f = np.fft.rfftfreq(M.shape[0], 1 / rate)
    return float(p[f > f_cut].sum() / max(p.sum(), 1e-12))


def max_step(M: np.ndarray) -> float:
    """largest jump between two consecutive control frames (0..1 scale)."""
    return float(np.abs(np.diff(M, axis=0)).max())


def preview(out: dict, raw_loud: np.ndarray, rate: float, title: str, png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = out["t"]
    ext = (t[0], t[-1], out["background"].shape[1], 0)
    fig, ax = plt.subplots(6, 1, figsize=(13, 15), constrained_layout=True, sharex=True)
    ax[0].imshow(out["background"].T, aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=1)
    ax[0].set_title("background layer (band 0 = top rows)")
    ax[1].imshow(out["events"].T, aspect="auto", extent=ext, cmap="viridis", vmin=0, vmax=1)
    ax[1].set_title("events layer (salient bright spots)")
    for k in ("loudness", "centroid", "spread", "saturation", "hue", "chroma", "motion"):
        ax[2].plot(t, out[k], lw=1.1, label=k)
    ax[2].legend(ncol=7, fontsize=8, loc="upper right"); ax[2].set_title("scalar tracks"); ax[2].set_ylim(-0.02, 1.02)
    rb, S = out["_raw_bands"], out["_S"]
    j = int(np.argmax(np.abs(np.diff(rb, axis=0)).max(axis=0)))      # most abrupt band
    ax[3].plot(t, rb[:, j], lw=0.7, color="0.55", label=f"band {j} raw (max step {np.abs(np.diff(rb[:, j])).max():.2f})")
    ax[3].plot(t, S[:, j], lw=1.6, label=f"band {j} smoothed (max step {np.abs(np.diff(S[:, j])).max():.3f})")
    ax[3].legend(fontsize=8); ax[3].set_title("most abrupt band: raw vs smoothed")
    for k in ("flux", "texture", "zone_low", "zone_mid", "zone_high"):
        if k in out:
            ax[4].plot(t, out[k], lw=0.9, label=k)
    ax[4].legend(ncol=5, fontsize=8, loc="upper right"); ax[4].set_title("rhythm / detail tracks (drive the beat and sparkles)")
    ax[4].set_ylim(-0.02, 1.02)
    ax[5].plot(t, out["x_pos"], label="scan x position"); ax[5].plot(t, out["frame_pos"], "--", label="video position")
    ax[5].legend(fontsize=8); ax[5].set_xlabel("audio time (s)")
    fig.suptitle(title)
    fig.savefig(png, dpi=100); plt.close(fig)


def write_outputs(prefix: Path, out: dict, raw_loud: np.ndarray, meta: dict, make_preview: bool = True,
                  preview_title: str | None = None) -> str:
    """Save <prefix>_ctrl.npz (+ <prefix>_ctrl.png preview) and return a one-line smoothness report."""
    rate = meta["control_rate_hz"]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rb, S, pvw = out.pop("_raw_bands"), out.pop("_S"), out.pop("_preview", None)
    np.savez_compressed(str(prefix) + "_ctrl.npz", **out, raw_loudness=raw_loud, meta_json=json.dumps(meta))
    out["_raw_bands"], out["_S"] = rb, S
    if pvw is not None:
        out["_preview"] = pvw
    if make_preview:
        title = preview_title or f"{meta['source']}  ({'video' if meta['is_video'] else 'still'}, {meta['scan']})"
        preview(out, raw_loud, rate, title, Path(str(prefix) + "_ctrl.png"))
    nan = any(np.isnan(v).any() for k, v in out.items() if not k.startswith("_"))
    return (f"{meta['source'][:34]:34s} bands: HF-power raw {band_roughness(rb, rate):.4f} -> {band_roughness(S, rate):.5f}"
            f" | max step raw {max_step(rb):.2f} -> {max_step(S):.3f} | events max step {max_step(out['events']):.3f} | NaN: {nan}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True, help="output prefix")
    p.add_argument("--duration", type=float, default=30.0, help="audio length in seconds")
    p.add_argument("--rate", type=float, default=50.0, help="control rate (Hz)")
    p.add_argument("--bands", type=int, default=24, help="vertical bands (pitch resolution)")
    p.add_argument("--blur", type=float, default=0.006, help="pre-filter sigma as fraction of width/height")
    p.add_argument("--scan", choices=["sweep", "variable", "fixed"], default="sweep")
    p.add_argument("--scan-position", type=float, default=0.5, help="fixed: where the column sits (0 left .. 1 right)")
    p.add_argument("--scan-curve", choices=list(SCAN_CURVES), default="cosine", help="variable: shape of the speed curve")
    p.add_argument("--scan-ratio", type=float, default=4.0, help="variable: edge speed / centre speed")
    p.add_argument("--passes", type=int, default=1, help="left-right-left... passes over the duration")
    p.add_argument("--smooth", type=float, default=0.25, help="low-pass sigma in seconds (bigger = smoother)")
    p.add_argument("--event-scale", type=float, default=4.0, help="background/event time-scale ratio")
    a = p.parse_args(argv)

    out, raw_loud, meta = presonify(a.input, a.duration, a.rate, a.bands, a.blur, a.scan,
                                    a.passes, a.smooth, event_scale=a.event_scale, scan_position=a.scan_position,
                                    scan_curve=a.scan_curve, scan_ratio=a.scan_ratio)
    print(write_outputs(a.out, out, raw_loud, meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
