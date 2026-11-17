#!/usr/bin/env python3
"""
execute_ui.py - runs the ColorDominance window drawn in Qt Designer (dominance_main.ui).

    python execute_ui.py                                  (or:  python execute_ui.py "C:\\path\\to\\working folder")

The .ui file only describes how the window LOOKS; everything it DOES is the code of dominance_gui.py, so both windows
behave the same.  The .ui is read at every start: edit it in Designer (pyside6-designer dominance_main.ui), save, run this.

Rules for editing in Designer (widgets are found by objectName)
    * Move, resize, re-word, re-order, put things in other tabs or layouts: free.
    * Do not rename or delete:  p_<setting> (every control of the five tabs) and the widgets listed in NAMED below.
    * Combo boxes keep the ORDER and NUMBER of their items (the real values are attached by position; the texts are yours).
    * Keep the promoted classes (SpinNW / DSpinNW / ComboNW / SourceView / StripMeters / HeatView, header widgets.h).
"""
import sys
from pathlib import Path

from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QWidget

FROZEN = getattr(sys, "frozen", False)                              # True inside the PyInstaller executable
HERE = Path(__file__).resolve().parent                              # code + dominance_main.ui (inside the bundle when frozen)
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else HERE # where the exe / the scripts live: the examples sit next to it
sys.path.insert(0, str(HERE))
import dominance_gui as G  # noqa: E402
from widgets import ComboNW, DSpinNW, HeatView, SourceView, SpinNW, StripMeters  # noqa: E402

NAMED = ("folder_edit", "btn_browse", "count_lbl", "list", "btn_heard", "btn_aud", "vis_title", "source", "meters", "heat", "play_btn",
         "slider", "time_lbl", "tabs", "btn_extract", "btn_sonify", "btn_open", "progress", "status", "btn_listen")


class UiWindow(G.MainWindow):
    """the ColorDominance window, drawn by the .ui file.  All behaviour is inherited from dominance_gui.MainWindow."""

    def build(self):
        loader = QUiLoader()
        for cls in (SourceView, StripMeters, HeatView, SpinNW, DSpinNW, ComboNW):
            loader.registerCustomWidget(cls)
        loaded = loader.load(str(HERE / "dominance_main.ui"))
        central = loaded.centralWidget()
        self.setCentralWidget(central)                                # (moves it out of `loaded` into this window)
        for name in NAMED:
            setattr(self, name, central.findChild(QWidget, name))
        for key, _tab, _label, kind, _tip in G.CONTROLS:
            w = self.P[key] = central.findChild(QWidget, "p_" + key)
            if kind[0] == "choice":                                   # attach the real values to the items, by position
                for i, (_text, value) in enumerate(kind[1]):
                    w.setItemData(i, value)
        central.findChild(QWidget, "split").setSizes([280, 860, 460])


def default_folder():
    """the working folder opened when none is given on the command line: the bundled examples ("WORKING FOLDER" next to the app)."""
    for name in ("WORKING FOLDER", "working folder"):
        if (APP_DIR / name).is_dir():
            return str(APP_DIR / name)
    return None


def selftest():
    """python execute_ui.py --selftest   (no window): extract + render + write one example, to check an installation / build."""
    import tempfile
    from scipy.io import wavfile
    import pipeline as P
    from synth import write_mix
    assert (HERE / "dominance_main.ui").is_file(), "dominance_main.ui is missing"
    folder = Path(default_folder() or ".")
    pictures = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert pictures, f"no example picture in {folder}"
    pic = min(pictures, key=lambda p: p.stat().st_size)            # the smallest one is enough
    out, _raw, _meta = P.extract(pic, duration=4.0)
    c = {k: v for k, v in out.items() if not k.startswith("_")}
    stems = P.render(c, P.Colour(timbre=True, mute_grey=True), **P.ORIGINAL_GUI)
    with tempfile.TemporaryDirectory() as tmp:
        write_mix(Path(tmp) / "test", [("", stems, 1.0)], write_stems=False, write_spectrogram=False)
        rate, data = wavfile.read(Path(tmp) / "test_mix.wav")
    assert rate == 44100 and len(data) > 4 * rate * 0.9 and abs(data).max() > 0, "the rendered audio is empty"
    print(f"SELFTEST OK  ({pic.name}: {len(data) / rate:.1f} s of audio)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:                                     # (a windowed exe has no console: the result goes to selftest.log)
        import contextlib, io, traceback
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                selftest()
            code = 0
        except BaseException:
            log.write(traceback.format_exc()); code = 1
        try:
            (APP_DIR / "selftest.log").write_text(log.getvalue(), encoding="utf-8")
        except OSError:
            pass
        sys.stdout = sys.stdout or sys.__stdout__
        if sys.stdout:
            print(log.getvalue())
        raise SystemExit(code)
    if len(sys.argv) < 2:
        sys.argv.append(default_folder() or "")                      # (an empty path = no folder)
    raise SystemExit(G.main(UiWindow))
