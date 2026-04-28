import sys
import numpy as np
import sounddevice as sd
from collections import deque, Counter

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from audio import (
    SAMPLERATE, CHANNELS, BLOCKSIZE, BUFFERSIZE, MINFREQ, MAXFREQ,
    audiobuf, audiocb,
    detectnotes, smoothpc, guesschord, pcname,
)

UPDATEMS = 30
LABELUPDATEMS = 300
HISTORYLENGTH = 12

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#f7f7f8"
SURFACE  = "#ffffff"
BORDER   = "#e4e4e8"
TEXT     = "#1a1a1a"
TEXT_DIM = "#9090a0"
ACCENT   = "#5b6af0"   # soft indigo — the one pop of colour

pg.setConfigOptions(antialias=True, foreground=TEXT_DIM, background=SURFACE)


def styled_plot(title, x_label, x_units=None):
    pw = pg.PlotWidget()
    pw.setBackground(SURFACE)
    pw.getPlotItem().hideButtons()
    pw.setTitle(
        f'<span style="color:{TEXT_DIM};font-size:9px;letter-spacing:2px;">'
        f'{title.upper()}</span>',
        size="1pt",
    )
    for side in ("left", "bottom", "top", "right"):
        ax = pw.getPlotItem().getAxis(side)
        ax.setPen(pg.mkPen(BORDER, width=1))
        ax.setTextPen(pg.mkPen(TEXT_DIM))
    kw = {"units": x_units} if x_units else {}
    pw.setLabel("bottom", x_label, color=TEXT_DIM, **{"font-size": "9px"}, **kw)
    pw.setLabel("left", "", color=BG)
    pw.showGrid(x=True, y=True, alpha=0.5)
    return pw


class LiveVisualizer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer")
        self.resize(1000, 680)

        self.setStyleSheet(f"QWidget {{ background:{BG}; color:{TEXT}; }}")

        self.recentchords = deque(maxlen=HISTORYLENGTH)
        self.recentnotes  = deque(maxlen=HISTORYLENGTH)
        self.recentconf   = deque(maxlen=HISTORYLENGTH)

        self.lastlabelupdate = 0
        self.shownnotes = "—"
        self.shownchord = "No chord"
        self.shownconf  = 0.0

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplots)
        self.timer.start(UPDATEMS)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(16)

        # ── Info row ──────────────────────────────────────────────────────────
        info_frame = QtWidgets.QFrame()
        info_frame.setStyleSheet(
            f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;"
        )
        info_layout = QtWidgets.QHBoxLayout(info_frame)
        info_layout.setContentsMargins(20, 14, 20, 14)
        info_layout.setSpacing(0)

        def make_stat(label):
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(3)
            lbl = QtWidgets.QLabel(label.upper())
            lbl.setStyleSheet(
                f"color:{TEXT_DIM}; font-size:9px; letter-spacing:1px; border:none; background:transparent;"
            )
            val = QtWidgets.QLabel("—")
            val.setStyleSheet(
                f"color:{TEXT}; font-size:18px; font-weight:500; border:none; background:transparent;"
            )
            col.addWidget(lbl)
            col.addWidget(val)
            return col, val

        col_chord, self.lbl_chord = make_stat("Chord")
        col_notes, self.lbl_notes = make_stat("Notes")
        col_conf,  self.lbl_conf  = make_stat("Confidence")

        def vdiv():
            d = QtWidgets.QFrame()
            d.setFrameShape(QtWidgets.QFrame.VLine)
            d.setStyleSheet(f"color:{BORDER};")
            return d

        info_layout.addLayout(col_chord)
        info_layout.addSpacing(24)
        info_layout.addWidget(vdiv())
        info_layout.addSpacing(24)
        info_layout.addLayout(col_notes, 1)
        info_layout.addSpacing(24)
        info_layout.addWidget(vdiv())
        info_layout.addSpacing(24)
        info_layout.addLayout(col_conf)

        root.addWidget(info_frame)

        # ── Plots ─────────────────────────────────────────────────────────────
        self.waveplot = styled_plot("Waveform", "Sample")
        self.waveplot.setYRange(-1, 1)
        self.wavecurve = self.waveplot.plot(
            pen=pg.mkPen(color=TEXT, width=1.0)
        )

        self.specplot = styled_plot("Spectrum", "Frequency", "Hz")
        self.specplot.setXRange(MINFREQ, MAXFREQ)
        self.speccurve = self.specplot.plot(
            pen=pg.mkPen(color=ACCENT, width=1.5)
        )
        self.specfill = pg.FillBetweenItem(
            self.speccurve,
            pg.PlotDataItem([MINFREQ, MAXFREQ], [0, 0]),
            brush=pg.mkBrush(QtGui.QColor(ACCENT + "22")),
        )
        self.specplot.addItem(self.specfill)

        for plot in (self.waveplot, self.specplot):
            plot.setStyleSheet(
                f"border:1px solid {BORDER}; border-radius:10px;"
            )

        root.addWidget(self.waveplot, 3)
        root.addWidget(self.specplot, 3)

    def updateplots(self):
        if len(audiobuf) < BUFFERSIZE // 2:
            return

        x = np.array(audiobuf, dtype=np.float32)

        step = max(1, len(x) // 2048)
        self.wavecurve.setData(x[::step])

        window   = np.hanning(len(x))
        xw       = x * window
        spectrum = np.fft.rfft(xw)
        freqs    = np.fft.rfftfreq(len(xw), d=1.0 / SAMPLERATE)
        mags     = np.abs(spectrum)

        valid = (freqs >= MINFREQ) & (freqs <= MAXFREQ)
        fplot = freqs[valid]
        mplot = mags[valid]

        if len(mplot) > 0 and np.max(mplot) > 0:
            mplot = mplot / np.max(mplot)

        self.speccurve.setData(fplot, mplot)

        notenames, pitchclasses = detectnotes(freqs, mags)
        stablepc    = smoothpc(pitchclasses)
        stablenames = [pcname(pc) for pc in stablepc]
        chordname, confidence = guesschord(stablepc)

        self.recentchords.append(chordname)
        self.recentnotes.append(tuple(sorted(stablenames)))
        self.recentconf.append((chordname, confidence))

        nowms = QtCore.QTime.currentTime().msecsSinceStartOfDay()

        if nowms - self.lastlabelupdate >= LABELUPDATEMS:
            self.lastlabelupdate = nowms

            if self.recentchords:
                self.shownchord = Counter(self.recentchords).most_common(1)[0][0]

            if self.recentnotes:
                bestnotes = Counter(self.recentnotes).most_common(1)[0][0]
                self.shownnotes = "  ".join(bestnotes) if bestnotes else "—"

            matchconf = [c for ch, c in self.recentconf if ch == self.shownchord]
            self.shownconf = sum(matchconf) / len(matchconf) if matchconf else 0.0

        chord_color = ACCENT if self.shownchord not in ("No chord", "Unknown") else TEXT_DIM
        self.lbl_chord.setStyleSheet(
            f"color:{chord_color}; font-size:18px; font-weight:500; border:none; background:transparent;"
        )
        self.lbl_chord.setText(self.shownchord)
        self.lbl_notes.setText(self.shownnotes)
        self.lbl_conf.setText(f"{self.shownconf:.0%}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QtGui.QPalette()
    for role, color in [
        (QtGui.QPalette.Window,          BG),
        (QtGui.QPalette.WindowText,      TEXT),
        (QtGui.QPalette.Base,            SURFACE),
        (QtGui.QPalette.AlternateBase,   BG),
        (QtGui.QPalette.Text,            TEXT),
        (QtGui.QPalette.Button,          SURFACE),
        (QtGui.QPalette.ButtonText,      TEXT),
        (QtGui.QPalette.Highlight,       ACCENT),
        (QtGui.QPalette.HighlightedText, "#ffffff"),
    ]:
        pal.setColor(role, QtGui.QColor(color))
    app.setPalette(pal)

    win = LiveVisualizer()
    win.show()

    print(sd.query_devices())
    print("Default input device:", sd.default.device)

    stream = sd.InputStream(
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        dtype='float32',
        blocksize=BLOCKSIZE,
        callback=audiocb,
    )
    stream.start()

    exitcode = app.exec()
    stream.stop()
    stream.close()
    sys.exit(exitcode)


if __name__ == "__main__":
    main()
