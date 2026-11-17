# PyInstaller spec - Earth Information Jukebox (colour dominance sonification)
#   pyinstaller dominance.spec --noconfirm        ->  dist/EarthInformationJukebox/EarthInformationJukebox(.exe)
# One-folder build on purpose: starts instantly (a one-file exe would unpack ~300 MB on every launch) and the example
# folders can sit next to the exe, where the app finds them and where it is allowed to write its output.
a = Analysis(
    ["execute_ui.py"],
    pathex=["."],
    datas=[("dominance_main.ui", ".")],                 # the Qt Designer window, read at every start
    hiddenimports=["presonify", "synth", "dominance", "pipeline", "timbres", "loudness", "widgets", "dominance_gui"],
    excludes=["matplotlib", "tkinter", "pytest", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="EarthInformationJukebox", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="EarthInformationJukebox")
