"""
widgets.py - theme and custom-painted widgets of the ColorDominance window (taken from ColorSonification/gui_widgets.py).
Nothing here knows about audio or the pipeline: the widgets only draw arrays they are given.
"""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSizePolicy, QSpinBox, QWidget


PAL = dict(bg="#11151c", panel="#192029", panel2="#212a36", panel3="#2a3544", text="#dce3ee", dim="#8793a6",
           accent="#4fd1c5", accent2="#2c9c92", amber="#f6ad55")

STYLE = f"""
* {{ font-size: 10pt; }}
QWidget {{ background: {PAL['bg']}; color: {PAL['text']}; }}
QMainWindow, QDialog {{ background: {PAL['bg']}; }}
QFrame#card, QFrame[role="card"] {{ background: {PAL['panel']}; border-radius: 8px; }}
QLabel#title, QLabel[role="title"] {{ font-size: 11pt; font-weight: 600; color: {PAL['text']}; }}
QLabel#dim, QLabel[role="dim"] {{ color: {PAL['dim']}; }}
QLabel#hint, QLabel[role="hint"] {{ color: {PAL['dim']}; font-size: 9pt; }}
QTabWidget::pane {{ border: 0; background: {PAL['panel']}; border-radius: 8px; }}
QTabBar::tab {{ background: transparent; color: {PAL['dim']}; padding: 7px 14px; margin-right: 2px;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {PAL['text']}; border-bottom: 2px solid {PAL['accent']}; }}
QTabBar::tab:hover {{ color: {PAL['text']}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: 0; }}
QListWidget {{ background: {PAL['panel']}; border: 0; border-radius: 8px; padding: 4px; outline: 0; }}
QListWidget::item {{ padding: 5px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {PAL['panel3']}; }}
QListWidget::item:hover {{ background: {PAL['panel2']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {PAL['panel2']}; border: 1px solid {PAL['panel3']};
        border-radius: 5px; padding: 3px 6px; min-height: 20px; selection-background-color: {PAL['accent2']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {PAL['accent2']}; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0px; border: 0; }}
QComboBox QAbstractItemView {{ background: {PAL['panel2']}; selection-background-color: {PAL['accent2']};
        border: 1px solid {PAL['panel3']}; outline: 0; }}
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {{ color: {PAL['dim']}; background: {PAL['panel']}; }}
QPushButton {{ background: {PAL['panel3']}; border: 0; border-radius: 6px; padding: 7px 14px; }}
QPushButton:hover {{ background: #364357; }}
QPushButton:disabled {{ color: #5b6677; background: {PAL['panel2']}; }}
QPushButton#primary, QPushButton[role="primary"] {{ background: {PAL['accent']}; color: #07211f; font-weight: 700; padding: 9px 20px; }}
QPushButton#primary:hover, QPushButton[role="primary"]:hover {{ background: #6fe0d5; }}
QPushButton#primary:disabled, QPushButton[role="primary"]:disabled {{ background: {PAL['panel3']}; color: #5b6677; }}
QPushButton#secondary, QPushButton[role="secondary"] {{ background: {PAL['amber']}; color: #2b1a05; font-weight: 700; padding: 9px 20px; }}
QPushButton#secondary:hover, QPushButton[role="secondary"]:hover {{ background: #fbc27f; }}
QPushButton#secondary:disabled, QPushButton[role="secondary"]:disabled {{ background: {PAL['panel3']}; color: #5b6677; }}
QPushButton#seg, QPushButton[role="seg"] {{ background: {PAL['panel2']}; border-radius: 0; padding: 5px 16px; }}
QPushButton#seg:checked, QPushButton[role="seg"]:checked {{ background: {PAL['panel3']}; color: {PAL['accent']}; font-weight: 600; }}
QPushButton#flat, QPushButton[role="flat"] {{ background: transparent; color: {PAL['dim']}; padding: 3px 8px; }}
QPushButton#flat:hover, QPushButton[role="flat"]:hover {{ color: {PAL['text']}; }}
QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; border-radius: 8px; border: 1px solid #9aa6b8;
        background: #e9eef6; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border: 1px solid {PAL['accent']}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ border: 1px solid #9aa6b8;
        background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 #000000, stop:0.40 #000000, stop:0.46 #e9eef6, stop:1 #e9eef6); }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ background: #5b6677; border: 1px solid #4a5568; }}
QSlider::groove:horizontal {{ height: 4px; background: {PAL['panel3']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {PAL['accent2']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {PAL['accent']}; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }}
QProgressBar {{ background: {PAL['panel2']}; border: 0; border-radius: 4px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {PAL['accent']}; border-radius: 4px; }}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {PAL['panel3']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {PAL['panel2']}; color: {PAL['text']}; border: 1px solid {PAL['panel3']}; padding: 4px; }}
QPlainTextEdit {{ background: {PAL['panel']}; border: 0; border-radius: 6px; color: {PAL['dim']}; font-family: Consolas, monospace; font-size: 9pt; }}
QMenuBar {{ background: {PAL['bg']}; }}
QMenuBar::item:selected {{ background: {PAL['panel3']}; }}
QMenu {{ background: {PAL['panel2']}; border: 1px solid {PAL['panel3']}; }}
QMenu::item:selected {{ background: {PAL['accent2']}; }}
QSplitter::handle {{ background: {PAL['bg']}; }}
"""

# --------------------------------------------------------------------------- scroll-safe inputs --------------------

class _NoWheel:
    """spin boxes / combos only react to the mouse wheel once clicked, so scrolling a panel never changes a value."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class SpinNW(_NoWheel, QSpinBox):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.setKeyboardTracking(False)


class DSpinNW(_NoWheel, QDoubleSpinBox):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.setKeyboardTracking(False)


class ComboNW(_NoWheel, QComboBox):
    pass


# --------------------------------------------------------------------------- arrays -> pictures ------------------
ANCHORS = np.array([(9, 11, 24), (44, 26, 92), (132, 38, 112), (236, 104, 70), (252, 238, 176)], float)    # magma-like
LUT = np.stack([np.interp(np.linspace(0, 1, 256), (0, 0.22, 0.48, 0.74, 1), ANCHORS[:, k]) for k in range(3)], 1).astype(np.uint8)


def pixmap(rgb):
    """(rows, columns, 3) uint8 -> QPixmap, row 0 at the top"""
    rgb = np.ascontiguousarray(rgb)
    return QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy())


# --------------------------------------------------------------------------- source preview --------------------------
class SourceView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pix, self.x, self.caption = None, None, ""
        self.setMinimumSize(220, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_source(self, pix, caption=""):
        self.pix, self.caption = pix, caption
        self.update()

    def set_pixmap(self, pix):
        """swap the picture (the current video frame) without touching the caption."""
        self.pix = pix
        self.update()

    def set_scan(self, x, caption=None):
        self.x = x
        if caption is not None:
            self.caption = caption
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(PAL["panel"]))
        r = QRectF(self.rect()).adjusted(8, 8, -8, -40)
        if self.pix is None:
            p.setPen(QColor(PAL["dim"])); p.drawText(r, Qt.AlignCenter, "choose a file on the left")
            return
        s = self.pix.size(); k = min(r.width() / s.width(), r.height() / s.height())
        w, h = s.width() * k, s.height() * k
        img = QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)
        p.drawPixmap(img, self.pix, QRectF(self.pix.rect()))
        if self.x is not None:
            xx = img.left() + float(self.x) * img.width()
            p.fillRect(QRectF(img.left(), img.top(), xx - img.left(), img.height()), QColor(0, 0, 0, 90))
            p.setPen(QPen(QColor(PAL["accent"]), 2)); p.drawLine(QPointF(xx, img.top()), QPointF(xx, img.bottom()))
        p.setPen(QColor(PAL["dim"])); p.setFont(QFont(self.font().family(), 8))
        p.drawText(QRectF(8, self.height() - 36, self.width() - 16, 32), Qt.AlignLeft | Qt.AlignVCenter, self.caption)


class HeatView(QWidget):
    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pix = None; self.pos = 0.0; self.title = ""
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_matrix(self, M, title=""):
        """M: (time, rows) in 0..1; the LAST column is drawn at the top."""
        self.pix, self.title = pixmap(LUT[(np.clip(M, 0, 1) * 255).astype(np.uint8)[:, ::-1].T]), title
        self.update()

    def set_rgb(self, A, title=""):
        """A: (time, strips, 3) uint8 colours; strip 0 at the top."""
        self.pix, self.title = (None if A is None else pixmap(A.transpose(1, 0, 2))), title
        self.update()

    def set_pos(self, f):
        self.pos = f; self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(PAL["panel"]))
        a = QRectF(self.rect())
        if self.pix is not None:
            p.drawPixmap(a, self.pix, QRectF(self.pix.rect()))
        x = a.left() + self.pos * a.width()
        p.fillRect(QRectF(a.left(), a.top(), x - a.left(), a.height()), QColor(0, 0, 0, 70))
        p.setPen(QPen(QColor("#ffffff"), 1.5)); p.drawLine(QPointF(x, a.top()), QPointF(x, a.bottom()))
        p.setPen(QColor(255, 255, 255, 190)); p.setFont(QFont(self.font().family(), 8))
        p.drawText(QRectF(8, 4, 400, 16), Qt.AlignLeft | Qt.AlignVCenter, self.title)

    def mousePressEvent(self, e):
        self._seek(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._seek(e)

    def _seek(self, e):
        self.seek.emit(float(np.clip(e.position().x() / self.width(), 0, 1)))


class StripMeters(QWidget):
    """live level of every pitch strip: one bar per strip (top = high note); the bar's colour is the strip's colour class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.level, self.colour = None, None
        self.setMinimumSize(160, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def reset(self):
        self.level = None; self.update()

    def push(self, level, colour):
        """level: (strips,) in 0..1;  colour: (strips, 3) in 0..255"""
        self.level, self.colour = level, colour
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PAL["panel"]))
        r = QRectF(self.rect()).adjusted(76, 8, -8, -8)
        p.setFont(QFont(self.font().family(), 8)); p.setPen(QColor(PAL["dim"]))
        if self.level is None:
            p.drawText(QRectF(self.rect()), Qt.AlignCenter, "level and colour class of every strip")
            return
        p.drawText(QRectF(8, r.top(), 64, 14), Qt.AlignLeft, "high notes")
        p.drawText(QRectF(8, r.bottom() - 14, 64, 14), Qt.AlignLeft, "low notes")
        h = r.height() / len(self.level)
        p.setPen(Qt.NoPen)
        for s, (v, c) in enumerate(zip(self.level, self.colour)):
            y = r.top() + s * h + 0.12 * h
            p.setBrush(QColor(PAL["panel2"])); p.drawRect(QRectF(r.left(), y, r.width(), 0.76 * h))
            p.setBrush(QColor(*(int(x) for x in c))); p.drawRect(QRectF(r.left(), y, r.width() * float(np.clip(v, 0, 1)), 0.76 * h))
