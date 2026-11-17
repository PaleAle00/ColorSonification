# Earth Information Jukebox - colour-dominance sonification studio

NASA Space Apps 2026. Turns a satellite picture or video into sound: height of a pixel = note, brightness = volume,
dominant colour = timbre (never volume).

## Download (Windows)
Get `EarthInformationJukebox-windows.zip` from the **Releases** page (or from the latest *Actions* run, artifact
`EarthInformationJukebox-windows`), unzip it anywhere you can write to, run `EarthInformationJukebox.exe`.

The zip contains the app plus two example folders next to the exe:
* `WORKING FOLDER/` - example pictures; **opened automatically at start** (the default input).
* `OUTPUTS_EXAMPLE/` - what the app writes for `SAR1.jpg` (mix, stems, control data).

## Use
1. *Browse...* to another working folder if you like - every picture / video in it is listed on the left.
2. Select a file, press **1 Extract features**. The centre shows the extracted features moving in time (silent).
3. *Picture*, *Colour*, *Layers*, *Sound* and *Output* tabs: how the picture is scanned, which colour gets which timbre,
   which feature drives each voice (switch voices off, set levels), key / scale / tempo, file format.
4. Press **2 Generate audio**. Files are written (default: `ColorDominance_output/` inside the working folder) and
   the display switches to the real audio spectrum and plays it.

*Settings > Save / Load settings* keeps named presets (JSON).

## Run from source (Python 3.12+)
```
pip install -r requirements.txt
python execute_ui.py                       # opens ./WORKING FOLDER
python execute_ui.py "C:\path\to\folder"   # or any folder
python dominance_gui.py                    # same window built in code instead of dominance_main.ui
python execute_ui.py --selftest            # headless check: extract + render + write one example
```
`dominance_main.ui` is the Qt Designer layout (`pyside6-designer dominance_main.ui`); `build_ui.py` regenerates it.

## Build the executable yourself
```
pip install -r requirements.txt pyinstaller
pyinstaller dominance.spec --noconfirm
```
Result: `dist/EarthInformationJukebox/`. Copy `WORKING FOLDER` next to the exe to make it the default input.
The GitHub Action in `.github/workflows/build.yml` does this on every push and attaches the zip to releases
(push a tag such as `v1.0`).

## Files
`presonify.py`, `synth.py` - the original sonification (picture -> control tracks -> audio) ·
`pipeline.py`, `dominance.py`, `timbres.py`, `loudness.py` - colour-dominance additions ·
`dominance_gui.py`, `widgets.py`, `execute_ui.py`, `dominance_main.ui`, `build_ui.py` - the window.
