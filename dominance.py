"""
dominance.py - which colour dominates each pitch strip, as soft class weights, plus the black / grey gate.

The colour of a strip is the mean R, G, B of its rows in the scanned column (smoothed in time like every other control
track).  It becomes five weights that always add up to 1:

    warm (red / orange) | green (yellow / green) | blue (cyan / blue) | magenta (violet / pink) | neutral (grey, black, white)

    1. hue decides between the four colour classes; around each border (HUE_EDGES) the two neighbours crossfade over
       +-HUE_FADE degrees, so a colour sitting on a border is half one class, half the other - never a coin toss;
    2. saturation and brightness decide how much goes to "neutral" instead (hue means nothing on grey or black);
    3. the weights are low-passed in time (STICKY_S), so a class cannot flicker;
    4. the weights are sharpened (power SHARPEN, then rescaled to sum 1): the leading class takes most of the sound,
       so a dominant colour is heard as ITS timbre and not as a blend, while an even tie stays an even blend.
The largest weight names the dominant class; "confidence" is its lead over the runner-up (0 = tie, 1 = pure).
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

CLASSES = ("warm", "green", "blue", "magenta", "neutral")
DISPLAY_RGB = np.array([(240, 110, 60), (150, 210, 70), (70, 150, 240), (220, 90, 210), (150, 156, 168)], float)
HUE_EDGES = (345.0, 45.0, 165.0, 265.0)       # degrees where warm, green, blue, magenta START (each ends where the next starts)
HUE_FADE = 15.0                               # half-width of the crossfade around every border (degrees)
SATURATION = (0.08, 0.22)                     # HSV saturation: below = neutral, above = fully a colour
SHARPEN = 2.0                                 # > 1 favours the leading class
DARK = (0.04, 0.10)                           # brightness (max of R, G, B): below = black, hue not trusted
STICKY_S = 0.4                                # time smoothing of the weights (seconds)


def smoothstep(x, lo, hi):
    """0 below lo, 1 above hi, smooth S-curve in between."""
    u = np.clip((np.asarray(x, float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def hsv(rgb):
    """(..., 3) R, G, B in 0..1 -> hue in degrees, saturation, brightness (max channel)."""
    h = cv2.cvtColor(np.ascontiguousarray(rgb, np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(rgb.shape)
    return h[..., 0], h[..., 1], h[..., 2]


def class_weights(rgb, rate):
    """rgb (N, strips, 3) at `rate` control frames per second -> weights (N, strips, 5), each row summing to 1."""
    hue, sat, val = hsv(rgb)
    after = [smoothstep((hue - e + 180.0) % 360.0 - 180.0, -HUE_FADE, HUE_FADE) for e in HUE_EDGES]   # 1 once past a border
    colourful = smoothstep(sat, *SATURATION) * smoothstep(val, *DARK)
    w = [colourful * after[k] * (1 - after[(k + 1) % 4]) for k in range(4)] + [1 - colourful]
    w = gaussian_filter1d(np.stack(w, axis=-1), STICKY_S * rate, axis=0, mode="nearest") ** SHARPEN
    return w / w.sum(axis=-1, keepdims=True)


def dominant(weights):
    """-> (index of the dominant class, confidence 0..1 = its lead over the runner-up), both shaped like one weight."""
    top2 = np.sort(weights, axis=-1)[..., -2:]
    return weights.argmax(axis=-1), top2[..., 1] - top2[..., 0]


def grey_gate(rgb, limit=0.25, tolerance=0.15):
    """1 = the strip sounds, 0 = muted.  Muted when it is grey (saturation under `tolerance`) AND darker than `limit`;
    both borders are soft, so a strip fades out instead of switching off."""
    _, sat, val = hsv(rgb)
    grey = 1 - smoothstep(sat, 0.67 * tolerance, 1.33 * tolerance)
    dark = 1 - smoothstep(val, 0.8 * limit, 1.2 * limit)
    return 1 - grey * dark
