import sys
import numpy as np
import sounddevice as sd
from collections import deque, Counter

from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

from audio import (
    SAMPLERATE, CHANNELS, BLOCKSIZE, BUFFERSIZE, MINFREQ, MAXFREQ,
    audiobuf, audiocb,
    detectnotes, smoothpc, guesschord, pcname,
)

UPDATEMS = 30
LABELUPDATEMS = 300
HISTORYLENGTH = 12


class LiveVisualizer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Audio Note / Chord Visualizer")
        self.resize(1000, 700)

        self.recentchords = deque(maxlen=HISTORYLENGTH)
        self.recentnotes = deque(maxlen=HISTORYLENGTH)
        self.recentconf = deque(maxlen=HISTORYLENGTH)

        self.lastlabelupdate = 0
        self.shownnotes = "None"
        self.shownchord = "No chord"
        self.shownconf = 0.0

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.infolabel = QtWidgets.QLabel("Listening...")
        self.infolabel.setStyleSheet("font-size: 16px; padding: 8px;")
        layout.addWidget(self.infolabel)

        self.waveplot = pg.PlotWidget(title="Live Waveform")
        self.waveplot.setLabel("left", "Amplitude")
        self.waveplot.setLabel("bottom", "Sample")
        self.waveplot.setYRange(-1, 1)
        self.wavecurve = self.waveplot.plot(pen='c')
        layout.addWidget(self.waveplot)

        self.specplot = pg.PlotWidget(title="Live Frequency Spectrum")
        self.specplot.setLabel("left", "Magnitude")
        self.specplot.setLabel("bottom", "Frequency", units="Hz")
        self.specplot.setXRange(MINFREQ, MAXFREQ)
        self.speccurve = self.specplot.plot(pen='y')
        layout.addWidget(self.specplot)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplots)
        self.timer.start(UPDATEMS)

    def updateplots(self):
        if len(audiobuf) < BUFFERSIZE // 2:
            return

        x = np.array(audiobuf, dtype=np.float32)

        self.wavecurve.setData(x)

        window = np.hanning(len(x))
        xw = x * window
        spectrum = np.fft.rfft(xw)
        freqs = np.fft.rfftfreq(len(xw), d=1.0 / SAMPLERATE)
        mags = np.abs(spectrum)

        valid = (freqs >= MINFREQ) & (freqs <= MAXFREQ)
        fplot = freqs[valid]
        mplot = mags[valid]

        if len(mplot) > 0 and np.max(mplot) > 0:
            mplot = mplot / np.max(mplot)

        self.speccurve.setData(fplot, mplot)

        notenames, pitchclasses = detectnotes(freqs, mags)
        stablepc = smoothpc(pitchclasses)
        stablenames = [pcname(pc) for pc in stablepc]
        chordname, confidence = guesschord(stablepc)

        self.recentchords.append(chordname)
        self.recentnotes.append(tuple(sorted(stablenames)))
        self.recentconf.append((chordname, confidence))

        nowms = QtCore.QTime.currentTime().msecsSinceStartOfDay()

        if nowms - self.lastlabelupdate >= LABELUPDATEMS:
            self.lastlabelupdate = nowms

            if self.recentchords:
                chordcounts = Counter(self.recentchords)
                self.shownchord = chordcounts.most_common(1)[0][0]

            if self.recentnotes:
                notecounts = Counter(self.recentnotes)
                bestnotes = notecounts.most_common(1)[0][0]
                self.shownnotes = ", ".join(bestnotes) if bestnotes else "None"

            matchconf = [
                conf for chord, conf in self.recentconf
                if chord == self.shownchord
            ]

            if matchconf:
                self.shownconf = sum(matchconf) / len(matchconf)
            else:
                self.shownconf = 0.0

        self.infolabel.setText(
            f"Detected notes: {self.shownnotes}    |    "
            f"Chord guess: {self.shownchord}    |    "
            f"Confidence: {self.shownconf:.2f}"
        )


def main():
    app = QtWidgets.QApplication(sys.argv)

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