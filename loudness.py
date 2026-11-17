"""
loudness.py - how "equally loud" is measured when a colour changes the timbre.

Three weightings can be chosen (Colour.weighting):
    "K"     the ITU-R BS.1770 curve already used by synth.py for the voice balance; nearly flat, right at high volume
    "A"     the IEC 61672 curve; cuts the bass strongly, closer to the ear at low volume and on small loudspeakers
    "K+A"   halfway between the two (their dB readings are averaged)
A timbre with more overtones moves power from the bass to the middle, where K and A disagree - so the same pair of sounds
can be equal under one curve and several dB apart under the other.  Only listening can say which fits your playback.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import bilinear_zpk, sosfilt, sosfreqz, zpk2sos

from synth import SR, _k_sos


def _a_sos():
    """A-weighting as second-order sections (analog prototype, bilinear transform; 0 dB at 1 kHz)."""
    w = 2 * np.pi * np.array([20.598997, 107.65265, 737.86223, 12194.217])
    z, p, k = bilinear_zpk([0, 0, 0, 0], [-w[0], -w[0], -w[1], -w[2], -w[3], -w[3]], w[3] ** 2 * 10 ** (1.9997 / 20), SR)
    return zpk2sos(z, p, k)


WEIGHTINGS = {"K": (_k_sos(),), "A": (_a_sos(),), "K+A": (_k_sos(), _a_sos())}


def note_power(amps, f, weighting, pulse_depth=0.0):
    """weighted power of a note at f Hz whose harmonics 1, 2, 3 ... have the amplitudes `amps`."""
    freqs = f * np.arange(1, len(amps) + 1)
    powers = [((amps * np.abs(sosfreqz(sos, worN=freqs, fs=SR)[1])) ** 2).sum() for sos in WEIGHTINGS[weighting]]
    return float(np.exp(np.mean(np.log(powers))) * (1 + pulse_depth ** 2 / 2))


def follow(x, target, weighting, block=0.01, window=0.25):
    """scale x over time so that its short-term weighted loudness equals that of `target`: whatever the timbre does
    (harmonics of different notes falling together, the reverb favouring one partial), the level curve stays the one of
    the original sound.  The gain follows the power of `block`-second pieces smoothed over `window` seconds (slower than
    any timbre pulse) and stays within 12 dB of its average; a second pass removes what the first left over."""
    n = int(block * SR)

    def level(y, sos):                                           # log of the smoothed weighted power, one value per block
        k = sosfilt(sos, y.mean(axis=1)) ** 2
        return np.log(gaussian_filter1d(k[:len(k) // n * n].reshape(-1, n).mean(axis=1), window / block, mode="nearest") + 1e-12)

    for _ in range(2):
        gain = np.mean([level(target, sos) - level(x, sos) for sos in WEIGHTINGS[weighting]], axis=0) / 2    # log amplitude gain
        gain = np.exp(gain.mean() + np.clip(gain - gain.mean(), -np.log(4), np.log(4)))
        x = x * np.interp(np.arange(len(x)), (np.arange(len(gain)) + 0.5) * n, gain)[:, None]
    return x
