"""
timbres.py - the sound characters a colour class can have, and the presets that assign one to each class.

A timbre is a recipe of harmonics (whole multiples of the note, so every class stays on the scale ladder and nothing
beats against anything) plus an optional pulse locked to the tempo of the drums.  The five default timbres differ in
KIND, not only in brightness: buzzing (steady), hollow (swelling on the beat), glassy (shimmering in sixteenths),
organ-like (throbbing in eighths), plain.
A timbre never sets a level: `matched()` scales every recipe to the same weighted power (loudness.py).
"""
from dataclasses import dataclass

import numpy as np

from loudness import note_power
from synth import SR

HARMONICS = 10                                # harmonics a recipe may use
MAX_HZ = 0.42 * SR                            # partials above this are left out (no aliasing)


@dataclass(frozen=True)
class Timbre:
    about: str
    harmonics: tuple                          # amplitude of harmonic 1, 2, 3, ...
    pulses: float = 0.0                       # volume pulses per beat (2 = eighth notes, 4 = sixteenths); 0 = steady
    depth: float = 0.0                        # how deep the pulse goes (0 = none, 1 = down to silence)


TIMBRES = {
    "Soft": Timbre("the original pad sound: a sine with a little octave", (1, 0.3)),
    "Pure": Timbre("a bare sine, the plainest possible", (1,)),
    "Brass": Timbre("buzzing and full: every harmonic, falling off slowly", tuple(1 / h for h in range(1, 11))),
    "Reed": Timbre("hollow, clarinet-like: odd harmonics only", (1, 0, 1 / 3, 0, 1 / 5, 0, 1 / 7, 0, 1 / 9)),
    "Breath": Timbre("hollow and mellow (a few odd harmonics), swelling once per beat", (1, 0, 0.5, 0, 0.15), 1, 0.6),
    "Glass": Timbre("glassy: two and three octaves above the note, shimmering in sixteenths", (0.5, 0, 0, 1, 0, 0, 0, 0.7), 4, 0.5),
    "Organ": Timbre("organ-like: the note with its octave and double octave, throbbing in eighths", (1, 1, 0, 0.6), 2, 0.7),
    "Pulse": Timbre("the soft sound, throbbing in eighths", (1, 0.3), 2, 0.7),
}
PRESETS = {                                   # colour class -> timbre name
    "Elements": dict(warm="Brass", green="Breath", blue="Glass", magenta="Organ", neutral="Soft"),
    "Steady": dict(warm="Brass", green="Reed", blue="Pure", magenta="Soft", neutral="Pure"),
    "Pulsing": dict(warm="Brass", green="Pulse", blue="Glass", magenta="Organ", neutral="Pure"),
}


def partials(harmonics, f):
    """amplitudes of the harmonics of a note at f Hz (zero where the recipe is silent or the partial too high)."""
    a = np.zeros(HARMONICS)
    a[:len(harmonics)] = harmonics
    return a * (f * np.arange(1, HARMONICS + 1) < MAX_HZ)


def matched(timbre, f, reference, weighting):
    """harmonic amplitudes of `timbre` at f Hz, scaled to the weighted power of the `reference` recipe at the same note."""
    a = partials(timbre.harmonics, f)
    return a * np.sqrt(note_power(partials(reference, f), f, weighting) / note_power(a, f, weighting, timbre.depth))
