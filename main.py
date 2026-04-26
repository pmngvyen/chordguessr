import sys
import numpy as np
import sounddevice as sd
from collections import deque, Counter

from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

SAMPLERATE = 44100
CHANNELS = 1
BLOCKSIZE = 1024
BUFFERSIZE = 8192
MINFREQ = 50
MAXFREQ = 2000
TOPPEAKS = 8
UPDATEMS = 30

LABELUPDATEMS = 300
HISTORYLENGTH = 12

NOTENAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORDLIB = {
    "C major":  {0, 4, 7},
    "C minor":  {0, 3, 7},
    "C# major": {1, 5, 8},
    "C# minor": {1, 4, 8},
    "D major":  {2, 6, 9},
    "D minor":  {2, 5, 9},
    "D# major": {3, 7, 10},
    "D# minor": {3, 6, 10},
    "E major":  {4, 8, 11},
    "E minor":  {4, 7, 11},
    "F major":  {5, 9, 0},
    "F minor":  {5, 8, 0},
    "F# major": {6, 10, 1},
    "F# minor": {6, 9, 1},
    "G major":  {7, 11, 2},
    "G minor":  {7, 10, 2},
    "G# major": {8, 0, 3},
    "G# minor": {8, 11, 3},
    "A major":  {9, 1, 4},
    "A minor":  {9, 0, 4},
    "A# major": {10, 2, 5},
    "A# minor": {10, 1, 5},
    "B major":  {11, 3, 6},
    "B minor":  {11, 2, 6},
}

audiobuf = deque(maxlen=BUFFERSIZE)
recentpitches = deque(maxlen=6)

def freqmidi(freq):
    if freq <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midinote(midi):
    octave = (midi // 12) - 1
    return f"{NOTENAMES[midi % 12]}{octave}"


def freqpc(freq):
    midi = freqmidi(freq)
    if midi is None:
        return None
    return midi % 12


def pcname(pc):
    return NOTENAMES[pc % 12]


def findpeaks(mags):
    if len(mags) < 3:
        return np.array([], dtype=int)

    peaks = []
    for i in range(1, len(mags) - 1):
        if mags[i] > mags[i - 1] and mags[i] > mags[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def detectnotes(freqs, mags):
    valid = (freqs >= MINFREQ) & (freqs <= MAXFREQ)
    freqs = freqs[valid]
    mags = mags[valid]

    if len(freqs) == 0 or np.max(mags) <= 0:
        return [], []

    peakidx = findpeaks(mags)
    if len(peakidx) == 0:
        return [], []

    peakfreqs = freqs[peakidx]
    peakmags = mags[peakidx]

    threshold = 0.25 * np.max(peakmags)
    keep = peakmags >= threshold
    peakfreqs = peakfreqs[keep]
    peakmags = peakmags[keep]

    if len(peakfreqs) == 0:
        return [], []

    order = np.argsort(peakmags)[::-1]
    peakfreqs = peakfreqs[order][:TOPPEAKS]

    notenames = []
    pitchclasses = []
    seen = set()

    for f in peakfreqs:
        midi = freqmidi(f)
        if midi is None:
            continue
        pc = midi % 12
        if pc not in seen:
            seen.add(pc)
            pitchclasses.append(pc)
            notenames.append(midinote(midi))

    return notenames, pitchclasses


def smoothpc(pitchclasses):
    recentpitches.append(set(pitchclasses))
    counts = {}
    for s in recentpitches:
        for pc in s:
            counts[pc] = counts.get(pc, 0) + 1

    stable = [pc for pc, count in counts.items() if count >= 2]
    stable.sort()
    return stable


def guesschord(pitchclasses):
    if not pitchclasses:
        return "No chord", 0.0

    detected = set(pitchclasses)
    bestname = "Unknown"
    bestscore = -999
    bestoverlap = 0
    bestextra = 0
    bestmissing = 0

    for chordname, chordtones in CHORDLIB.items():
        overlap = len(detected & chordtones)
        extra = len(detected - chordtones)
        missing = len(chordtones - detected)

        score = 3.0 * overlap - 1.2 * extra - 0.8 * missing

        if score > bestscore:
            bestscore = score
            bestname = chordname
            bestoverlap = overlap
            bestextra = extra
            bestmissing = missing

    if bestscore < 3:
        return "Unknown", 0.0

    confidence = bestoverlap / (bestoverlap + bestextra + bestmissing + 1e-6)
    confidence = max(0.0, min(1.0, confidence))

    return bestname, confidence


def audiocb(indata, frames, time, status):
    if status:
        print(status)
    mono = np.squeeze(indata).astype(np.float32)
    audiobuf.extend(mono)


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