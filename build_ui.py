#!/usr/bin/env python3
"""
build_ui.py - writes dominance_main.ui, the Qt Designer version of the ColorDominance window.

    python build_ui.py                  -> overwrites dominance_main.ui next to this script

Everything is read from the live window of dominance_gui.py (its CONTROLS table: labels, ranges, tooltips, choices,
defaults), so the .ui and the code never drift apart.  Run it again after changing CONTROLS; it overwrites Designer edits.
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox  # noqa: E402

import dominance_gui as G  # noqa: E402

ENUMS = ("orientation", "frameShape", "horizontalScrollBarPolicy", "textElideMode", "sizeAdjustPolicy")
BIG = 16777215


def prop(parent, name, value, kind=None):
    """one <property>; the XML type is guessed from the Python value unless `kind` says otherwise."""
    p = ET.SubElement(parent, "property", name=name)
    kind = kind or ("enum" if name in ENUMS else "bool" if isinstance(value, bool) else "number" if isinstance(value, int)
                    else "double" if isinstance(value, float) else "size" if isinstance(value, tuple) else "string")
    if kind == "size":
        s = ET.SubElement(p, "size"); ET.SubElement(s, "width").text, ET.SubElement(s, "height").text = str(value[0]), str(value[1])
    else:
        ET.SubElement(p, kind).text = {True: "true", False: "false"}.get(value, str(value)) if kind == "bool" else str(value)
    if name == "role":
        p.set("stdset", "0")                                               # dynamic property: the styling hook of widgets.STYLE


def widget(parent, cls, name, **props):
    w = ET.SubElement(parent, "widget", {"class": cls, "name": name})
    for k, v in props.items():
        prop(w, k, v)
    return w


def layout(parent, cls, name, margin=None, spacing=None, stretch=None):
    lay = ET.SubElement(parent, "layout", {"class": cls, "name": name})
    if stretch:
        lay.set("stretch", ",".join(map(str, stretch)))
    for k in ("leftMargin", "topMargin", "rightMargin", "bottomMargin") if margin is not None else ():
        prop(lay, k, margin)
    if spacing is not None:
        prop(lay, "verticalSpacing" if cls == "QFormLayout" else "spacing", spacing)
    return lay


def add(lay, cls, name, row=None, column=0, colspan=None, **props):
    """a widget as the next item of a layout (with row / column for form layouts)."""
    at = {} if row is None else {"row": str(row), "column": str(column)} | ({"colspan": str(colspan)} if colspan else {})
    return widget(ET.SubElement(lay, "item", at), cls, name, **props)


def sub(lay, cls, name, **kw):
    return layout(ET.SubElement(lay, "item"), cls, name, **kw)


def spacer(lay, name, vertical):
    sp = ET.SubElement(ET.SubElement(lay, "item"), "spacer", name=name)
    prop(sp, "orientation", "Qt::Vertical" if vertical else "Qt::Horizontal")


def control(form, key, w, row, column, colspan=None):
    """one setting, as the same kind of control (same range, default, tooltip, choices) as in the live window."""
    common = dict(row=row, column=column, colspan=colspan, toolTip=w.toolTip())
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        number = float if isinstance(w, QDoubleSpinBox) else int
        extra = dict(decimals=w.decimals()) if isinstance(w, QDoubleSpinBox) else {}
        add(form, "DSpinNW" if extra else "SpinNW", "p_" + key, **common, **extra, minimum=number(w.minimum()), maximum=number(w.maximum()),
            singleStep=number(w.singleStep()), value=number(w.value()), suffix=w.suffix(), maximumSize=(150, BIG))
    elif isinstance(w, QComboBox):
        e = add(form, "ComboNW", "p_" + key, **common, sizeAdjustPolicy="QComboBox::AdjustToMinimumContentsLengthWithIcon", minimumContentsLength=12)
        for i in range(w.count()):
            prop(ET.SubElement(e, "item"), "text", w.itemText(i))
        prop(e, "currentIndex", w.currentIndex())
    elif isinstance(w, QCheckBox):
        add(form, "QCheckBox", "p_" + key, **common, text=w.text(), checked=w.isChecked())
    else:
        add(form, "QLineEdit", "p_" + key, **common, placeholderText=w.placeholderText())


def tab_page(tabs, win, title):
    name = title.lower()
    tab = widget(tabs, "QWidget", "tab_" + name)
    ET.SubElement(ET.SubElement(tab, "attribute", name="title"), "string").text = title
    sc = add(layout(tab, "QVBoxLayout", "lay_tab_" + name, margin=0, spacing=0), "QScrollArea", "scroll_" + name, widgetResizable=True,
             frameShape="QFrame::NoFrame", horizontalScrollBarPolicy="Qt::ScrollBarAlwaysOff")
    lay = layout(widget(sc, "QWidget", "page_" + name), "QVBoxLayout", "lay_" + name, margin=14, spacing=10)
    add(lay, "QLabel", "hint_" + name, text=G.HINTS[title], wordWrap=True, role="hint")
    form, row = sub(lay, "QFormLayout", "form_" + name, spacing=9), 0
    for key, tab_name, label, _kind, tip in G.CONTROLS:
        if tab_name != title or label is None:                             # (a label-less check box is placed by its partner)
            continue
        if label == "":
            control(form, key, win.P[key], row, 0, colspan=2)
        else:
            if label.startswith("@"):
                control(form, label[1:], win.P[label[1:]], row, 0)
            else:
                add(form, "QLabel", "lbl_" + key, row, 0, text=label, toolTip=tip)
            control(form, key, win.P[key], row, 1)
        row += 1
    if title == "Colour":
        add(lay, "QPushButton", "btn_listen", text=win.btn_listen.text(), toolTip=win.btn_listen.toolTip())
    spacer(lay, "spacer_" + name, vertical=True)


def build(win):
    ui = ET.Element("ui", version="4.0")
    ET.SubElement(ui, "class").text = "MainWindow"
    mw = widget(ui, "QMainWindow", "MainWindow", windowTitle=win.windowTitle())
    geo = ET.SubElement(ET.SubElement(mw, "property", name="geometry"), "rect")
    for k, v in zip(("x", "y", "width", "height"), (0, 0, 1560, 900)):
        ET.SubElement(geo, k).text = str(v)
    outer = layout(widget(mw, "QWidget", "centralwidget"), "QVBoxLayout", "lay_outer", margin=10, spacing=10, stretch=(0, 1, 0))

    top = sub(outer, "QHBoxLayout", "lay_top", spacing=8, stretch=(0, 1, 0))
    add(top, "QLabel", "lbl_folder", text="Working folder", role="dim")
    add(top, "QLineEdit", "folder_edit", readOnly=True, placeholderText=win.folder_edit.placeholderText())
    add(top, "QPushButton", "btn_browse", text="Browse…")

    split = add(outer, "QSplitter", "split", orientation="Qt::Horizontal", childrenCollapsible=False)
    left = layout(widget(split, "QWidget", "panel_left", minimumSize=(270, 0)), "QVBoxLayout", "lay_left", margin=0, spacing=6)
    add(left, "QLabel", "count_lbl", text="Files", role="title")
    add(left, "QListWidget", "list", iconSize=(76, 48), spacing=2, horizontalScrollBarPolicy="Qt::ScrollBarAlwaysOff", textElideMode="Qt::ElideMiddle")

    centre = layout(widget(split, "QWidget", "panel_centre"), "QVBoxLayout", "lay_centre", margin=0, spacing=8, stretch=(0, 5, 5, 0))
    seg = sub(centre, "QHBoxLayout", "lay_seg", spacing=0)
    add(seg, "QPushButton", "btn_heard", text="Strips", checkable=True, checked=True, autoExclusive=True, role="seg")
    add(seg, "QPushButton", "btn_aud", text="Spectrum", checkable=True, autoExclusive=True, role="seg")
    spacer(seg, "spacer_seg", vertical=False)
    add(seg, "QLabel", "vis_title", text="", role="dim")
    views = sub(centre, "QHBoxLayout", "lay_views", spacing=8, stretch=(4, 5))
    add(views, "SourceView", "source", minimumSize=(220, 150))
    add(views, "StripMeters", "meters", minimumSize=(160, 150))
    add(centre, "HeatView", "heat", minimumSize=(0, 110))
    tr = sub(centre, "QHBoxLayout", "lay_transport", spacing=10, stretch=(0, 1, 0))
    add(tr, "QPushButton", "play_btn", text="▶", minimumSize=(46, 0), maximumSize=(46, BIG))
    add(tr, "QSlider", "slider", maximum=1000, orientation="Qt::Horizontal")
    add(tr, "QLabel", "time_lbl", text="0:00.0 / 0:00.0", role="dim", minimumSize=(110, 0), maximumSize=(110, BIG))

    right = layout(widget(split, "QWidget", "panel_right", minimumSize=(430, 0), maximumSize=(520, BIG)), "QVBoxLayout", "lay_right", margin=0)
    tabs = add(right, "QTabWidget", "tabs", currentIndex=0)
    for title in G.TABS:
        tab_page(tabs, win, title)

    bot = sub(outer, "QHBoxLayout", "lay_bottom", spacing=12, stretch=(0, 0, 0, 0, 1))
    add(bot, "QPushButton", "btn_extract", text="1   Extract", role="primary")
    add(bot, "QPushButton", "btn_sonify", text="2   Generate audio", role="secondary")
    add(bot, "QPushButton", "btn_open", text="Open output folder", enabled=False)
    add(bot, "QProgressBar", "progress", maximum=1000, value=0, minimumSize=(220, 0), maximumSize=(220, BIG))
    add(bot, "QLabel", "status", text="Choose a folder to begin.", role="dim")

    custom = ET.SubElement(ui, "customwidgets")
    for cls, base in (("SourceView", "QWidget"), ("StripMeters", "QWidget"), ("HeatView", "QWidget"), ("SpinNW", "QSpinBox"),
                      ("DSpinNW", "QDoubleSpinBox"), ("ComboNW", "QComboBox")):
        c = ET.SubElement(custom, "customwidget")
        ET.SubElement(c, "class").text, ET.SubElement(c, "extends").text, ET.SubElement(c, "header").text = cls, base, "widgets.h"
    return ui


def main():
    app = QApplication(sys.argv)                                           # noqa: F841  (Qt needs it alive)
    ui = build(G.MainWindow())
    ET.indent(ui, space=" ")
    out = HERE / "dominance_main.ui"
    ET.ElementTree(ui).write(out, encoding="utf-8", xml_declaration=True)
    print("written", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
