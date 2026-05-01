import numpy as np
from collections import deque

SAMPLERATE = 44100
CHANNELS = 1
BLOCKSIZE = 1024
BUFFERSIZE = 8192
MINFREQ = 50
MAXFREQ = 2000
TOPPEAKS = 8

NOTENAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORDLIB = {
    "C major": {0, 4, 7},
    "C minor": {0, 3, 7},
    "C# major": {1, 5, 8},
    "C# minor": {1, 4, 8},
    "D major": {2, 6, 9},
    "D minor": {2, 5, 9},
    "D# major": {3, 7, 10},
    "D# minor": {3, 6, 10},
    "E major": {4, 8, 11},
    "E minor": {4, 7, 11},
    "F major": {5, 9, 0},
    "F minor": {5, 8, 0},
    "F# major": {6, 10, 1},
    "F# minor": {6, 9, 1},
    "G major": {7, 11, 2},
    "G minor": {7, 10, 2},
    "G# major": {8, 0, 3},
    "G# minor": {8, 11, 3},
    "A major": {9, 1, 4},
    "A minor": {9, 0, 4},
    "A# major": {10, 2, 5},
    "A# minor": {10, 1, 5},
    "B major": {11, 3, 6},
    "B minor": {11, 2, 6},
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
