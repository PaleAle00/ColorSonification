#!/usr/bin/env python3
"""
dominance_gui.py - window of the ColorDominance sonification  (Earth Information Jukebox, NASA Space Apps 2026).

    pip install PySide6 numpy scipy opencv-python
    python dominance_gui.py                    (or:  python dominance_gui.py "C:\\path\\to\\working folder")
presonify.py and synth.py (the original sonification) sit in this same folder.

Workflow
    1. Pick a working folder; every picture / video in it is listed on the left.
    2. Select a file, check the PICTURE tab, press [1 Extract]: the picture is scanned silently.  The centre shows what
       will be heard: every pitch strip over time, bright = loud, coloured by its colour class.
    3. COLOUR tab: switch "dominant colour changes the timbre" on, choose which timbre each colour class gets (a preset
       or your own), press [Listen] to hear the five timbres.  Optionally mute black / grey strips.
       LAYERS: the voices of the original sonification, and which track of the picture each one follows.
       SOUND: key, scale, tempo ...  Nothing is extracted again for these tabs.
    4. Press [2 Generate audio]: mix + stems are written, the display switches to the real audio spectrum.
With the colour switch and the grey mute off, the result is sample for sample the original sonification.
Height of a pixel = note, brightness = volume, dominant colour = timbre (never volume).
"""
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)
from scipy.io import wavfile
from scipy.signal import stft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dominance as dom  # noqa: E402
import pipeline as P  # noqa: E402
import timbres as tb  # noqa: E402
from pipeline import presonify, synth  # noqa: E402
from widgets import STYLE, ComboNW, DSpinNW, HeatView, SourceView, SpinNW, StripMeters  # noqa: E402

MEDIA_EXT = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".mpg", ".mpeg"} | presonify.IMG_EXT
CUSTOM = "Custom"
CLASS_LABELS = dict(warm="Red / orange", green="Yellow / green", blue="Cyan / blue", magenta="Violet / pink", neutral="Grey / black / white")
LAYERS = {"pad": ("Pad chord", ("pad",)), "bells": ("Bells", ("events",)), "drone": ("Drone", ("drone",)), "waves": ("Waves", ("air",)),
          "drums": ("Drums", ("kick", "snare", "hat")), "plucks": ("Plucks", ("sparkle",))}       # layer -> (name, original stems)
TRACKS = [                                                             # the control tracks a layer can follow
    ("Brightness per strip", "background"), ("Bright spots per strip", "events"), ("Onsets per strip", "onsets"),
    ("Overall brightness", "loudness"), ("Vertical centre of brightness", "centroid"), ("Vertical spread of brightness", "spread"),
    ("Colour saturation", "saturation"), ("Colourfulness", "chroma"), ("Warm colour intensity", "colour"), ("Warmth (red vs blue)", "warmth"),
    ("Movement of the whole frame", "motion"), ("Movement near the scan line", "motion_scan"), ("Edge / onset strength", "flux"),
    ("Texture / fine detail", "texture"), ("Bottom third brightness", "zone_low"), ("Middle third brightness", "zone_mid"),
    ("Top third brightness", "zone_high")]
THIRDS = "thirds"                                                      # drums as in the original: kick, snare, hat on three tracks
TABS = ("Picture", "Colour", "Layers", "Sound", "Output")
HINTS = {
    "Picture": "How the picture is read (the original scan).  Changing anything here means it must be extracted again.",
    "Colour": "The dominant colour of every pitch strip can change the TIMBRE of that strip's pad and bell notes - never their "
              "loudness.  With the two switches on this tab off, the sound is exactly the original.",
    "Layers": "The voices of the original sonification.  Untick one to leave it out; the level is relative to the original "
              "balance; 'follows' is the track of the picture that drives the voice (each one starts on its original track).",
    "Sound": "Settings of the original sound that change it audibly (see the sweep table in the README); the rest are fixed.",
    "Output": "Files are written with the original writer: one mix, one stem per voice in a 'stems' folder, the control tracks.",
}


# ----------------------------------------------------------------------------------------------- the controls -------
def choice(items, default):
    return ("choice", items, default)


def number(lo, hi, default, step=1, suffix="", decimals=0):
    return ("number", lo, hi, default, step, suffix, decimals)


def check(text, default):
    return ("check", text, default)


DRIVER = {"pad": choice(TRACKS, "background"), "bells": choice(TRACKS, "events"), "drone": choice(TRACKS, "loudness"),
          "waves": choice(TRACKS, "motion"), "plucks": choice(TRACKS, "flux"),
          "drums": choice([("Bottom / middle third + texture", THIRDS)] + TRACKS, THIRDS)}
DRIVER_TIP = ("The track of the picture that drives this voice.\n'Movement near the scan line' follows what moves in the part of "
              "a video being scanned right now (band width: Picture tab).")

# (key, tab, label, kind, tooltip).   label "@key" = the check box `key` is the label, and it enables this control.
CONTROLS = [
    # ---- Picture: how the original presonify reads the picture ----
    ("scan", "Picture", "Scan", choice([("Sweep · constant speed", "sweep"), ("Sweep · variable speed", "variable"), ("Fixed column", "fixed")], "sweep"),
     "Sweep: a scan line travels across the picture while the sound plays (a video plays along).\n"
     "Variable speed: different speed near the edges than in the middle.\nFixed column: always the same column - for videos."),
    ("scan_curve", "Picture", "Speed curve", choice([("Smooth ease (cosine)", "cosine"), ("Linear ramp", "linear"), ("Flat middle (quadratic)", "quadratic"),
                                                     ("Very flat middle (quartic)", "quartic")], "cosine"), "How the speed changes with position (variable speed only)."),
    ("scan_ratio", "Picture", "Edge ÷ centre speed", number(0.1, 20, 4.0, 0.5, " ×", 1), "4 = four times faster at the edges than in the middle; below 1 = slower at the edges."),
    ("scan_position", "Picture", "Column position", number(0, 100, 50, 5, " %"), "Where the fixed column sits: 0 % = left edge, 100 % = right edge."),
    ("passes", "Picture", "Passes", number(1, 8, 1), "How often the scan crosses the picture (left-right-left ...) during the sound."),
    ("length", "Picture", "Length", number(5, 600, 30, 5, " s"), "Duration of the sound; the whole picture / video is stretched to it."),
    ("bands", "Picture", "Pitch bands", number(8, 48, 24, 2), "Horizontal bands the picture is cut into (top = high). They are grouped into the pad voices of the Sound tab."),
    ("smooth", "Picture", "Smoothing", number(0.05, 1.5, 0.25, 0.05, " s", 2), "How slowly the sound may change. Small = busy, large = calm."),
    ("blur_pct", "Picture", "Pre-blur", number(0, 5, 0.6, 0.1, " %", 1), "Blur applied before reading, as a share of the picture size. More = softer, fewer bells."),
    ("flux_smooth", "Picture", "Hit sharpness", number(0.01, 0.5, 0.06, 0.01, " s", 2), "Time scale of the onsets that fire drums and plucks. Small = many sharp hits."),
    ("compress_k", "Picture", "Highlight squash", number(0.5, 40, 8.0, 0.5, "", 1), "How strongly the brightest spots are compressed. High = dim areas sound too."),
    ("norm_high_pct", "Picture", "Full-scale level", number(90, 100, 99.5, 0.1, " %", 1), "The brightness percentile that counts as full volume."),
    ("motion_width", "Picture", "Motion band", number(2, 100, 10, 2, " %"),
     "Width of the band of columns around the scan line whose movement is measured (track 'Movement near the scan line').\n100 % = the whole frame."),
    # ---- Colour: the new features ----
    ("colour_on", "Colour", "", check("dominant colour changes the timbre", False),
     "Off: the original sound.  On: every strip's pad and bell notes take the timbre of the strip's colour class."),
    ("preset", "Colour", "Timbres", choice([(n, n) for n in tb.PRESETS] + [("Custom - chosen below", CUSTOM)], "Elements"),
     "Ready-made assignments of a timbre to every colour class.  Changing a class below switches this to Custom."),
] + [("timbre_" + k, "Colour", CLASS_LABELS[k], choice([(n, n) for n in tb.TIMBRES], tb.PRESETS["Elements"][k]),
      f"The timbre of strips whose dominant colour is {CLASS_LABELS[k].lower()}:\n" + "\n".join(f"{n} - {t.about}" for n, t in tb.TIMBRES.items()))
     for k in dom.CLASSES] + [
    ("depth", "Colour", "Colour depth", number(0, 100, 100, 10, " %"), "How far the timbres move away from the original pad sound. 0 % = original timbre everywhere."),
    ("weighting", "Colour", "Equal loudness by", choice([("K-weighting", "K"), ("A-weighting", "A"), ("Halfway K / A", "K+A")], "K"),
     "The curve that decides when two timbres count as equally loud.\nK (used for the voice balance of the original) is nearly flat; "
     "A cuts the bass like the ear does at low volume.\n"
     "If the richer timbres seem louder than the plain ones, choose Halfway or A (see README)."),
    ("mute", "Colour", None, check("mute dark grey below", False),
     "A strip that is grey (red, green, blue nearly equal) AND darker than the limit fades out of the pad and the bells.  White and bright grey still sound."),
    ("mute_below", "Colour", "@mute", number(0, 100, 25, 5, " %"), "Brightness under which a grey strip is muted (0 % = nothing, 100 % = every grey, even white)."),
    ("mute_tolerance", "Colour", "Grey tolerance", number(0, 50, 15, 5, " %"), "How much colour a strip may have and still count as grey (saturation limit)."),
    # ---- Layers: the voices of the original synth ----
    ("on_pad", "Layers", None, check(LAYERS["pad"][0], True), "Soft sustained tones, one per pitch strip."),
    ("level_pad", "Layers", "@on_pad", number(-30, 12, 0.0, 1.0, " dB", 1), "Level relative to the original balance."),
    ("drive_pad", "Layers", "   follows", DRIVER["pad"], DRIVER_TIP),
    ("on_bells", "Layers", None, check(LAYERS["bells"][0], True), "Bell-like tones; only the strongest few ring at once."),
    ("level_bells", "Layers", "@on_bells", number(-30, 12, 0.0, 1.0, " dB", 1), "Level relative to the original balance."),
    ("drive_bells", "Layers", "   follows", DRIVER["bells"], DRIVER_TIP),
    ("on_drone", "Layers", "", check(LAYERS["drone"][0], True), "A steady low root + fifth."),
    ("drive_drone", "Layers", "   follows", DRIVER["drone"], DRIVER_TIP),
    ("on_waves", "Layers", None, check(LAYERS["waves"][0], True), "Noise that swells with its driver (movement by default, so silent for still pictures)."),
    ("level_waves", "Layers", "@on_waves", number(-30, 12, 0.0, 1.0, " dB", 1), "Level relative to the original balance."),
    ("drive_waves", "Layers", "   follows", DRIVER["waves"], DRIVER_TIP),
    ("wave_follow", "Layers", "   how closely", number(0, 100, 40, 10, " %"),
     "How much of the waves' volume comes from their track.  40 % (original): the waves only swell a little with it.\n"
     "100 %: the volume is the track - silent where nothing moves."),
    ("air_mode", "Layers", "Waves sound", choice([("Ocean waves", "waves"), ("Soft hiss", "hiss")], "waves"), "Character of the Waves voice."),
    ("on_drums", "Layers", None, check(LAYERS["drums"][0], True), "Kick, snare and hi-hat on a 16-step grid; the picture decides which steps play."),
    ("level_drums", "Layers", "@on_drums", number(-30, 12, 0.0, 1.0, " dB", 1), "Level relative to the original balance."),
    ("drive_drums", "Layers", "   follows", DRIVER["drums"], DRIVER_TIP),
    ("on_plucks", "Layers", None, check(LAYERS["plucks"][0], True), "Short high plucks fired by the strongest moments of their driver."),
    ("level_plucks", "Layers", "@on_plucks", number(-30, 12, 0.0, 1.0, " dB", 1), "Level relative to the original balance."),
    ("drive_plucks", "Layers", "   follows", DRIVER["plucks"], DRIVER_TIP),
    # ---- Sound: the settings of the original synth that passed the sweep ----
    ("root", "Sound", "Key", choice([(r, r) for r in synth.ROOTS], "D"), "Home note of the scale."),
    ("mode", "Sound", "Scale", choice([("Pentatonic (calm, never clashes)", "pentatonic"), ("Major (bright)", "major"), ("Minor (dark)", "minor"),
                                       ("Dorian (open minor)", "dorian"), ("Lydian (dreamy)", "lydian")], "pentatonic"), "Pentatonic has no semitone steps, so any notes together sound consonant."),
    ("shift", "Sound", "Pitch shift", number(-24, 24, 0, 1, " st"), "Moves everything up or down in semitones. ±12 = one octave."),
    ("voices", "Sound", "Pad voices", number(4, 24, 12), "Number of pitch strips = pad notes (top strip = highest)."),
    ("event_voices", "Sound", "Bells at once", number(1, 8, 3), "Most bells allowed to ring at the same moment. 1 = very sparse."),
    ("event_shift", "Sound", "Bell pitch", number(0, 36, 12, 1, " st"), "Bells sound this many semitones above their pad note. 12 = one octave."),
    ("reverb", "Sound", "Reverb", number(0, 80, 30, 5, " %"), "Room echo on pad, bells and percussion. 0 = dry."),
    ("bpm", "Sound", "Tempo", number(40, 200, 100, 5, " bpm"), "Speed of the beat grid (16th notes)."),
    ("swing", "Sound", "Swing", number(0, 50, 0, 5, " %"), "Delays every second 16th note. 0 = straight, 33 = shuffle."),
    ("density", "Sound", "Beat density", number(0, 100, 60, 5, " %"), "Higher = more drum hits."),
    ("pluck_amount", "Sound", "Pluck amount", number(0, 90, 35, 5, " %"), "Share of grid steps that may fire a pluck (the strongest onsets first)."),
    ("motion_boost", "Sound", "Motion → beat", number(0, 1, 0.25, 0.05, "", 2), "How much motion in a video adds drum hits. No effect on still pictures."),
    ("pluck_shift", "Sound", "Pluck pitch", choice([("1 octave above the pad (+12)", 12), ("2 octaves (+24)", 24), ("3 octaves (+36)", 36)], 24), "Register of the plucks."),
    ("seed", "Sound", "Variation", number(0, 9999, 0), "Another number = another (equally valid) reverb texture and wave timing."),
    ("fade_in", "Sound", "Fade in", number(0.05, 10, 0.6, 0.1, " s", 2), "Keep at least 0.05 s to avoid a click."),
    ("fade_out", "Sound", "Fade out", number(0.05, 10, 1.5, 0.1, " s", 2), "Length of the fade at the end."),
    # ---- Output ----
    ("out_base", "Output", "Folder", ("text", "automatic: <working folder>/ColorDominance_output"), "Where the files go (one sub-folder per picture / video)."),
    ("stems", "Output", "", check("also write one stem per voice", True), "The stems add up to the mix (one scale factor for all)."),
    ("wav_format", "Output", "Format", choice([("WAV 16-bit", "int16"), ("WAV 32-bit float (no rounding)", "float32")], "int16"), "16-bit plays everywhere; 32-bit float keeps full precision."),
]


def make(kind):
    """the widget of one control"""
    if kind[0] == "choice":
        w = ComboNW(); w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon); w.setMinimumContentsLength(12)
        for text, value in kind[1]:
            w.addItem(text, value)
        w.setCurrentIndex(w.findData(kind[2]))
    elif kind[0] == "number":
        _, lo, hi, default, step, suffix, decimals = kind
        w = DSpinNW() if decimals else SpinNW()
        if decimals:
            w.setDecimals(decimals)
        w.setRange(lo, hi); w.setSingleStep(step); w.setValue(default); w.setSuffix(suffix); w.setMaximumWidth(150)
    elif kind[0] == "check":
        w = QCheckBox(kind[1]); w.setChecked(kind[2])
    else:
        w = QLineEdit(); w.setPlaceholderText(kind[1])
    return w


def get(w):
    return (w.isChecked() if isinstance(w, QCheckBox) else w.value() if isinstance(w, (QSpinBox, QDoubleSpinBox))
            else w.currentData() if isinstance(w, QComboBox) else w.text())


def put(w, v):
    """set a control without firing its change signal"""
    w.blockSignals(True)
    if isinstance(w, QCheckBox):
        w.setChecked(v)
    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
        w.setValue(v)
    elif isinstance(w, QComboBox):
        w.setCurrentIndex(w.findData(v))
    else:
        w.setText(v)
    w.blockSignals(False)


def changed(w):
    return next(getattr(w, n) for n in ("toggled", "valueChanged", "currentIndexChanged", "textChanged") if hasattr(w, n))


# ----------------------------------------------------------------------------------- settings -> pipeline arguments --
def pre_args(g):
    """Picture tab -> keyword arguments of pipeline.extract (= original presonify.presonify + motion_width)"""
    return dict(duration=float(g["length"]), n_bands=g["bands"], blur=g["blur_pct"] / 100, scan=g["scan"],
                scan_position=g["scan_position"] / 100, scan_curve=g["scan_curve"], scan_ratio=g["scan_ratio"], passes=g["passes"],
                smooth=g["smooth"], compress_k=g["compress_k"], norm_high_pct=g["norm_high_pct"], flux_smooth=g["flux_smooth"],
                motion_width=g["motion_width"] / 100)


def syn_args(g, bands):
    """Layers + Sound tabs -> keyword arguments of the original synth.render_layer (the call the original window makes)"""
    mapping = dict(pad=g["drive_pad"], events=g["drive_bells"], drone=g["drive_drone"], air=g["drive_waves"], sparkle=g["drive_plucks"])
    if g["drive_drums"] != THIRDS:
        mapping |= dict.fromkeys(("kick", "snare", "hat"), g["drive_drums"])
    return dict(P.ORIGINAL_GUI, mapping=mapping, root=g["root"], mode=g["mode"], shift=g["shift"], seed=g["seed"], voices=min(g["voices"], bands),
                event_voices=g["event_voices"], event_shift_semitones=float(g["event_shift"]),
                air_mode=g["air_mode"] if g["on_waves"] else "off", wave_motion_influence=g["wave_follow"] / 100, reverb_wet=g["reverb"] / 100, fade_in=g["fade_in"], fade_out=g["fade_out"],
                rhythm="on" if g["on_drums"] or g["on_plucks"] else "off", rhythm_bpm=float(g["bpm"]), rhythm_swing=g["swing"] / 100,
                rhythm_threshold=0.55 - 0.5 * g["density"] / 100, rhythm_motion_boost=g["motion_boost"],
                sparkle_percentile=100.0 - g["pluck_amount"], sparkle_shift_semitones=float(g["pluck_shift"]))


def colour_of(g):
    return P.Colour(timbre=g["colour_on"], assignment={k: g["timbre_" + k] for k in dom.CLASSES}, depth=g["depth"] / 100,
                    weighting=g["weighting"], mute_grey=g["mute"], grey_limit=g["mute_below"] / 100, grey_tolerance=g["mute_tolerance"] / 100)


def finish(stems, g):
    """drop the layers that are switched off and apply the layer levels (0 dB = untouched)."""
    return {k: stems[k] * 10 ** (g.get("level_" + layer, 0.0) / 20)
            for layer, (_, names) in LAYERS.items() if g["on_" + layer] for k in names if k in stems}


def heard_map(c, g):
    """what the pad will play, as a picture: level (time, strips) 0..1, its image (time, strips, 3) uint8 and the strip
    colours.  Brightness = level of the strip, colour = its colour class (teal while colour timbres are off)."""
    strips, col = min(g["voices"], c["background"].shape[1]), colour_of(g)
    if col.mute_grey:
        c = dict(c, background=c["background"] * dom.grey_gate(c["rgb"], col.grey_limit, col.grey_tolerance))
    level = synth.pool(synth.resolve(synth.with_derived(c), g["drive_pad"], "matrix"), strips) ** 1.5
    level = level / max(level.max(), 1e-9)
    tint = P.strip_weights(c, strips) @ dom.DISPLAY_RGB if col.timbre else np.broadcast_to([79.0, 209.0, 197.0], level.shape + (3,))
    return level, (level[..., None] ** 0.7 * tint).astype(np.uint8), tint


def audition(g):
    """the five class timbres one after the other on the same chord, then all together (for the Listen button)."""
    notes = synth.mtof(synth.ladder(8, g["root"], g["mode"], synth.BASE_MIDI + 12))[[7, 5, 3]]
    t = np.arange(int(2.4 * synth.SR)) / synth.SR
    t_c, col = np.array([0.0, 3.0]), colour_of(g)
    env = np.minimum(t / 0.05, 1) * np.minimum((t[-1] - t) / 0.3, 1)
    voices = [tb.TIMBRES[col.assignment[k]] for k in dom.CLASSES]
    one = lambda w: env * sum(P.tone(f, np.tile(w, (2, 1)), voices, (1.0, 0.325), col, g["bpm"], t_c, t) for f in notes)
    x = np.concatenate([one(w) for w in np.eye(len(voices))] + [one(np.full(len(voices), 1 / len(voices)))])
    return np.repeat(0.5 * x[:, None] / np.abs(x).max(), 2, axis=1)


def analyse_audio(path):
    """what the spectrum display shows: (frames, 96) log-frequency level at 60 frames/s, 0..1."""
    sr, x = wavfile.read(path)
    f, _, Z = stft(x.mean(axis=1), sr, nperseg=4096, noverlap=4096 - sr // 60)
    edges = np.geomspace(40, 12000, 97)
    first = np.searchsorted(f, edges)                                 # first spectrum line of every band
    bands = (np.add.reduceat(np.abs(Z) ** 2, first[:-1], axis=0) / np.maximum(np.diff(first), 1)[:, None]).T    # mean power per band
    db = 10 * np.log10(bands + 1e-12) + 3.0 * np.log2(np.sqrt(edges[:-1] * edges[1:]) / 250.0)      # +3 dB / octave tilt
    return np.clip((db - (np.percentile(db, 99.7) - 62)) / 62, 0, 1).astype(np.float32)


def to_image(rgb):
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()


def thumbnail(path):
    """-> (small QImage of a picture, or of a frame one third into a video;  one line describing the file),
    or None for a file that cannot be read (an unfinished download, say)"""
    if path.suffix.lower() in presonify.IMG_EXT:
        bgr, info = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2), "image"     # decoded at half size: faster on big pictures
    else:
        cap = cv2.VideoCapture(str(path))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 3)
        bgr, info = cap.read()[1], f"video · {n / max(fps, 1):.1f} s"
    if bgr is None:
        return None
    k = 640 / bgr.shape[1]
    return to_image(cv2.cvtColor(cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)), info


class Task(QThread):
    """runs fn(progress) in the background; `done` carries its result, `failed` the error text."""
    progress, done, failed = Signal(float, str), Signal(object), Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------------------------- main window ---------
class MainWindow(QMainWindow):
    def __init__(self, folder=None):
        super().__init__()
        self.setWindowTitle("Earth Information Jukebox · colour dominance")
        self.resize(1560, 900)
        self.folder, self.cur, self.task, self.last_out = None, None, None, None
        self.infos, self.thumbs, self.feat = {}, {}, {}                 # per file: info line, thumbnail, extraction
        self.mode, self.audio, self.map, self.f, self.shown = "heard", None, None, 0.0, None
        self.tmp = Path(tempfile.mkdtemp(prefix="colour_dominance_"))
        self.P = {}                                                     # setting -> its control
        self.build()
        for menu, items in (("&Settings", (("Save settings…", self.save_settings), ("Load settings…", self.load_settings),
                                           ("Reset everything", lambda: self.apply(self.defaults)))),
                            ("&Help", (("How it works", lambda: QMessageBox.information(self, "How it works", __doc__.split("Workflow")[1].strip())),))):
            m = self.menuBar().addMenu(menu)
            for text, fn in items:
                a = QAction(text, self); a.triggered.connect(fn); m.addAction(a)
        self.player, self.speaker = QMediaPlayer(), QAudioOutput()
        self.player.setAudioOutput(self.speaker)
        self.clock = QTimer(self, interval=16, timeout=self.tick)       # moves the play head
        self.map_timer = QTimer(self, interval=120, singleShot=True, timeout=self.refresh_map)
        self.wire()
        self.defaults = self.gather()
        if folder:
            self.set_folder(folder)
        self.on_param()

    # ------------------------------------------------------------------ construction
    def build(self):
        """the whole window in code (execute_ui.py replaces this by loading dominance_main.ui)"""
        root = QWidget(); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(12, 10, 12, 10); outer.setSpacing(10)
        top = QHBoxLayout(); top.setSpacing(8); outer.addLayout(top)
        lab = QLabel("Working folder"); lab.setObjectName("dim"); top.addWidget(lab)
        self.folder_edit = QLineEdit(readOnly=True, placeholderText="choose a folder with pictures and videos…")
        self.btn_browse = QPushButton("Browse…")
        top.addWidget(self.folder_edit, 1); top.addWidget(self.btn_browse)

        self.split = QSplitter(Qt.Horizontal, childrenCollapsible=False); outer.addWidget(self.split, 1)
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(6)
        self.count_lbl = QLabel("Files"); self.count_lbl.setObjectName("title"); ll.addWidget(self.count_lbl)
        self.list = QListWidget(spacing=2, horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff, textElideMode=Qt.ElideMiddle)
        self.list.setIconSize(QPixmap(76, 48).size())
        ll.addWidget(self.list, 1); left.setMinimumWidth(270); self.split.addWidget(left)

        centre = QWidget(); cl = QVBoxLayout(centre); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(8)
        seg = QHBoxLayout(); seg.setSpacing(0); cl.addLayout(seg)
        self.btn_heard, self.btn_aud = QPushButton("Strips"), QPushButton("Spectrum")
        for b in (self.btn_heard, self.btn_aud):
            b.setObjectName("seg"); b.setCheckable(True); b.setAutoExclusive(True); seg.addWidget(b)
        self.btn_heard.setChecked(True)
        seg.addStretch(1)
        self.vis_title = QLabel(""); self.vis_title.setObjectName("dim"); seg.addWidget(self.vis_title)
        row = QHBoxLayout(); row.setSpacing(8); cl.addLayout(row, 5)
        self.source, self.meters, self.heat = SourceView(), StripMeters(), HeatView()
        row.addWidget(self.source, 4); row.addWidget(self.meters, 5); cl.addWidget(self.heat, 5)
        tr = QHBoxLayout(); tr.setSpacing(10); cl.addLayout(tr)
        self.play_btn = QPushButton("▶"); self.play_btn.setFixedWidth(46)
        self.slider = QSlider(Qt.Horizontal, maximum=1000)
        self.time_lbl = QLabel("0:00.0 / 0:00.0"); self.time_lbl.setObjectName("dim"); self.time_lbl.setFixedWidth(110)
        tr.addWidget(self.play_btn); tr.addWidget(self.slider, 1); tr.addWidget(self.time_lbl)
        self.split.addWidget(centre)

        self.tabs = QTabWidget(); self.tabs.setMinimumWidth(430); self.tabs.setMaximumWidth(520)
        for title in TABS:
            s = QScrollArea(widgetResizable=True, frameShape=QFrame.NoFrame, horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff)
            s.setWidget(self.tab(title)); self.tabs.addTab(s, title)
        self.split.addWidget(self.tabs); self.split.setStretchFactor(1, 1)

        bot = QHBoxLayout(); bot.setSpacing(12); outer.addLayout(bot)
        self.btn_extract = QPushButton("1   Extract"); self.btn_extract.setObjectName("primary")
        self.btn_sonify = QPushButton("2   Generate audio"); self.btn_sonify.setObjectName("secondary")
        self.btn_open = QPushButton("Open output folder", enabled=False)
        self.progress = QProgressBar(maximum=1000); self.progress.setFixedWidth(220)
        self.status = QLabel("Choose a folder to begin."); self.status.setObjectName("dim")
        for w, stretch in ((self.btn_extract, 0), (self.btn_sonify, 0), (self.btn_open, 0), (self.progress, 0), (self.status, 1)):
            bot.addWidget(w, stretch)

    def tab(self, title):
        """one tab = a hint, then one form row per control of CONTROLS (plus the Listen button on the Colour tab)."""
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 14); lay.setSpacing(10)
        hint = QLabel(HINTS[title], wordWrap=True); hint.setObjectName("hint"); lay.addWidget(hint)
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignLeft); form.setVerticalSpacing(9); lay.addLayout(form)
        for key, tab, label, kind, tip in CONTROLS:
            if tab != title:
                continue
            w = self.P[key] = make(kind); w.setToolTip(tip)
            if label == "":                                                # a check box on its own, across both columns
                form.addRow(w)
            elif label is not None:                                        # (None: a check box that labels the next row)
                form.addRow(self.P[label[1:]] if label.startswith("@") else QLabel(label, toolTip=tip), w)
        if title == "Colour":
            self.btn_listen = QPushButton("♪  Listen to the five timbres", toolTip="Plays one chord in each class timbre (red/orange, "
                                          "yellow/green, cyan/blue, violet/pink, grey), then a blend of all.")
            lay.addWidget(self.btn_listen)
        lay.addStretch(1)
        return page

    def wire(self):
        """connect every button and control"""
        self.btn_browse.clicked.connect(self.choose_folder)
        self.list.currentItemChanged.connect(self.on_select)
        self.btn_heard.clicked.connect(lambda: self.set_mode("heard")); self.btn_aud.clicked.connect(lambda: self.set_mode("audio"))
        self.heat.seek.connect(self.seek); self.slider.sliderMoved.connect(lambda v: self.seek(v / 1000))
        self.play_btn.clicked.connect(lambda: self.pause() if self.clock.isActive() else self.play())
        self.btn_listen.clicked.connect(self.listen)
        self.btn_extract.clicked.connect(self.run_extract); self.btn_sonify.clicked.connect(self.run_sonify)
        self.btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_out))))
        for w in self.P.values():
            changed(w).connect(self.on_param)
        self.P["preset"].currentIndexChanged.connect(self.preset_chosen)
        for k in dom.CLASSES:
            self.P["timbre_" + k].currentIndexChanged.connect(self.timbre_chosen)
        self.progress.hide()

    # ------------------------------------------------------------------ settings
    def gather(self):
        return {k: get(w) for k, w in self.P.items()}

    def apply(self, settings):
        for k, v in settings.items():
            put(self.P[k], v)
        self.on_param()

    def save_settings(self):
        p = QFileDialog.getSaveFileName(self, "Save settings", str(self.folder or Path.home()), "JSON (*.json)")[0]
        if p:
            Path(p).write_text(json.dumps(self.gather(), indent=1), encoding="utf-8")

    def load_settings(self):
        p = QFileDialog.getOpenFileName(self, "Load settings", str(self.folder or Path.home()), "JSON (*.json)")[0]
        if p:
            self.apply(json.loads(Path(p).read_text(encoding="utf-8")))

    def picture_key(self):
        """the Picture settings: an extraction made with other values is out of date"""
        g = self.gather()
        return tuple(g[key] for key, tab, *_ in CONTROLS if tab == "Picture")

    def preset_chosen(self):
        if self.P["preset"].currentData() != CUSTOM:
            for k, timbre in tb.PRESETS[self.P["preset"].currentData()].items():
                put(self.P["timbre_" + k], timbre)
        self.on_param()

    def timbre_chosen(self):
        """a class timbre was changed by hand: show the preset it happens to equal, otherwise Custom."""
        now = {k: self.P["timbre_" + k].currentData() for k in dom.CLASSES}
        put(self.P["preset"], next((n for n, a in tb.PRESETS.items() if a == now), CUSTOM))

    def on_param(self):
        """keep the window consistent with the settings: which rows are shown / enabled, button states, the strip map."""
        g = self.gather()
        for key, _tab, label, _kind, _tip in CONTROLS:                       # "@box" rows follow their check box
            if label and label.startswith("@"):
                self.P[key].setEnabled(g[label[1:]])
        self.P["mute_tolerance"].setEnabled(g["mute"]); self.P["air_mode"].setEnabled(g["on_waves"]); self.P["wave_follow"].setEnabled(g["on_waves"])
        for key in ["preset", "depth", "weighting"] + ["timbre_" + k for k in dom.CLASSES]:
            self.P[key].setEnabled(g["colour_on"])
        for key, mode in (("scan_curve", "variable"), ("scan_ratio", "variable"), ("scan_position", "fixed")):
            form = next(f for f in self.findChildren(QFormLayout) if f.getWidgetPosition(self.P[key])[0] >= 0)
            form.setRowVisible(self.P[key], g["scan"] == mode)               # only the options of the chosen scan mode
        self.update_buttons(); self.map_timer.start()

    def update_buttons(self):
        fe = self.feat.get(self.cur)
        stale = fe is not None and fe["key"] != self.picture_key()
        busy = self.task is not None
        self.btn_extract.setEnabled(self.cur is not None and not busy)
        self.btn_extract.setText("1   Re-extract" if stale else "1   Extract")
        self.btn_sonify.setEnabled(fe is not None and not stale and not busy)
        self.btn_sonify.setToolTip("Picture settings changed since extraction - extract again first." if stale else "")
        self.list.setEnabled(not busy)

    def listen(self):
        self.pause(); self.player.setSource(QUrl())
        synth.to_wav(self.tmp / "audition.wav", audition(self.gather()))
        self.player.setSource(QUrl.fromLocalFile(str(self.tmp / "audition.wav"))); self.player.play()
        self.status.setText("listening: red/orange, yellow/green, cyan/blue, violet/pink, grey - then a blend of all five.")

    # ------------------------------------------------------------------ folder / files
    def choose_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Working folder", str(self.folder or Path.home()))
        if p:
            self.set_folder(p)

    def set_folder(self, folder):
        self.folder = Path(folder); self.folder_edit.setText(str(self.folder))
        self.list.clear()
        for p in sorted((p for p in self.folder.iterdir() if p.suffix.lower() in MEDIA_EXT), key=lambda p: p.name.lower()):
            thumb = thumbnail(p)
            if thumb:                                                        # unreadable files are left out
                self.thumbs[p], self.infos[p] = thumb
                it = QListWidgetItem(QIcon(QPixmap.fromImage(thumb[0])), f"{p.name}\n{thumb[1]}")
                it.setData(Qt.UserRole, p); self.list.addItem(it)
        self.count_lbl.setText(f"Files  ·  {self.list.count()}")
        self.status.setText(f"{self.list.count()} picture / video file(s) found.")
        self.list.setCurrentRow(0)

    def on_select(self, item):
        if item is None:                                                     # the list was just cleared
            return
        self.cur = item.data(Qt.UserRole)
        self.pause(); self.audio, self.map, self.f, self.shown = None, None, 0.0, None
        self.btn_aud.setEnabled(False); self.set_mode("heard")
        self.source.set_source(QPixmap.fromImage(self.thumbs[self.cur]), self.infos[self.cur]); self.source.set_scan(None)
        self.meters.reset(); self.heat.set_rgb(None); self.vis_title.setText("")
        self.refresh_map(); self.update_buttons()

    # ------------------------------------------------------------------ the two jobs
    def start(self, job, done, text):
        """run job(progress) in the background, then done(result) here.  (The slots are methods, not lambdas, so that
        Qt runs them in the window's thread.)"""
        self.task, self.after = Task(job), done
        self.task.progress.connect(self.on_progress); self.task.done.connect(self.on_done); self.task.failed.connect(self.on_failed)
        self.progress.show(); self.status.setText(text); self.update_buttons()
        self.task.start()

    def on_progress(self, f, text):
        self.progress.setValue(int(f * 1000)); self.status.setText(text + " …")

    def on_done(self, result):
        self.task = None
        self.progress.hide(); self.after(result); self.update_buttons()

    def on_failed(self, error):
        print(error)
        self.status.setText("failed: " + error.strip().splitlines()[-1])
        self.task = None
        self.progress.hide(); self.update_buttons()

    def run_extract(self):
        g, path, key = self.gather(), self.cur, self.picture_key()

        def job(progress):
            out, raw, meta = P.extract(path, preview_frames=min(g["length"] * 12, 720), progress=progress, **pre_args(g))
            return dict(out=out, raw=raw, meta=meta, key=key, c={k: v for k, v in out.items() if not k.startswith("_")})

        def done(r):
            self.feat[path] = r
            share = P.strip_weights(r["c"], 12).mean(axis=(0, 1))
            self.status.setText("extracted.  Colour classes found: "
                                + ", ".join(f"{CLASS_LABELS[k].lower()} {100 * s:.0f} %" for k, s in zip(dom.CLASSES, share) if s >= 0.005))
            self.audio = None; self.btn_aud.setEnabled(False)
            self.vis_title.setText(f"{path.name}  ·  {len(r['c']['t'])} control steps")
            self.set_mode("heard"); self.refresh_map(); self.update_buttons(); self.play()

        self.start(job, done, f"reading {path.name} …")

    def run_sonify(self):
        g, path, fe = self.gather(), self.cur, self.feat[self.cur]
        self.pause(); self.player.setSource(QUrl())                          # release the file: Windows locks files being played
        folder = Path(g["out_base"] or self.folder / "ColorDominance_output") / path.stem

        def job(progress):
            stems = P.render(fe["c"], colour_of(g), progress=lambda f, m: progress(0.85 * f, m), **syn_args(g, fe["c"]["background"].shape[1]))
            progress(0.9, "writing files")
            synth.write_mix(folder / path.stem, [("", finish(stems, g), 1.0)], peak_target=10 ** (-1 / 20), write_stems=g["stems"],
                            stems_dir=folder / "stems", write_spectrogram=False, wav_format=g["wav_format"])
            presonify.write_outputs(folder / path.stem, dict(fe["out"]), fe["raw"], fe["meta"], make_preview=False)
            mix = folder / f"{path.stem}_mix.wav"
            return dict(play=QUrl.fromLocalFile(str(mix)), heat=analyse_audio(mix), dur=len(stems["pad"]) / synth.SR)

        def done(audio):
            self.audio, self.last_out = audio, folder
            self.btn_open.setEnabled(True); self.btn_aud.setEnabled(True)
            self.status.setText(f"done - files written to {folder}")
            self.set_mode("audio"); self.play()

        self.start(job, done, f"generating audio for {path.name} …")

    # ------------------------------------------------------------------ display and transport
    def refresh_map(self):
        if self.cur in self.feat:
            self.map = heard_map(self.feat[self.cur]["c"], self.gather())
            if self.mode == "heard":
                self.heat.set_rgb(self.map[1], "what will be heard:  time →, high notes at the top, bright = loud, colour = colour class")
            self.frame()

    def set_mode(self, mode):
        """what the big display shows: "heard" = the strip map, "audio" = the spectrum of the generated mix"""
        self.mode = mode
        self.btn_heard.setChecked(mode == "heard"); self.btn_aud.setChecked(mode == "audio")
        if mode == "audio":
            self.heat.set_matrix(self.audio["heat"], "audio spectrum   (time →, high frequencies at the top)")
        self.pause(); self.seek(0.0); self.refresh_map()

    def duration(self):
        return self.audio["dur"] if self.mode == "audio" else float(self.P["length"].value())

    def play(self):
        self.play_btn.setText("⏸"); self.anchor = (time.perf_counter(), self.f)
        if self.mode == "audio":
            if self.player.source() != self.audio["play"]:
                self.player.setSource(self.audio["play"])
            self.player.setPosition(int(self.f * self.duration() * 1000)); self.player.play()
        self.clock.start()

    def pause(self):
        self.play_btn.setText("▶"); self.clock.stop(); self.player.pause()

    def seek(self, f):
        self.f, self.anchor = f, (time.perf_counter(), f)
        if self.mode == "audio":
            self.player.setPosition(int(f * self.duration() * 1000))
        self.frame()

    def tick(self):
        self.f = self.anchor[1] + (time.perf_counter() - self.anchor[0]) / self.duration()
        if self.f >= 1.0:                                                    # loop
            self.seek(0.0)
            if self.mode == "audio":
                self.player.play()
        self.frame()

    def frame(self):
        """draw everything that depends on the play position self.f (0..1)"""
        dur, clock = self.duration(), lambda s: f"{int(s // 60)}:{s % 60:04.1f}"
        self.slider.setValue(int(self.f * 1000))
        self.time_lbl.setText(f"{clock(self.f * dur)} / {clock(dur)}")
        self.heat.set_pos(self.f)
        fe = self.feat.get(self.cur)
        if fe is None or self.map is None:
            return
        c, pv = fe["c"], fe["out"].get("_preview")
        i = round(self.f * (len(c["t"]) - 1))
        self.meters.push(self.map[0][i], self.map[2][i])
        if pv is not None:                                                   # video: show the frame being read right now
            k = round(self.f * (len(pv) - 1))
            if self.shown != k:
                self.shown = k; self.source.set_pixmap(QPixmap.fromImage(to_image(pv[k])))
        self.source.set_scan(float(c["x_pos"][i]), f"{self.infos[self.cur]}\nscan {c['x_pos'][i] * 100:.0f} % across")

    def closeEvent(self, e):
        self.player.stop()
        if self.task is not None:
            self.task.wait()
        super().closeEvent(e)


def main(window=MainWindow):
    app = QApplication(sys.argv)
    app.setStyle("Fusion"); app.setStyleSheet(STYLE)
    w = window(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
