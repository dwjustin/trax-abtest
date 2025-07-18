import librosa
from librosa.sequence import dtw
from scipy.spatial.distance import cdist

def extract_chroma(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)  # shape: (12, time)
    return chroma.T, sr  # shape: (time, 12)


def compute_dtw(X, Y):
    D = cdist(X, Y, metric='cosine')  # cosine works well for chroma
    cost, wp = dtw(C=D)
    return cost, wp


def detect_changed_regions(wp, threshold=10):
    changed_segments = []
    for i, (a, b) in enumerate(reversed(wp)):  # DTW path is usually from end to start
        if abs(a - b) > threshold:
            changed_segments.append((a, b))
    return changed_segments

def group_changes(changes, min_gap=5):
    if not changes:
        return []

    grouped = []
    start = changes[0]
    for i in range(1, len(changes)):
        if abs(changes[i][0] - changes[i-1][0]) > min_gap:
            grouped.append((start, changes[i-1]))
            start = changes[i]
    grouped.append((start, changes[-1]))
    return grouped

def index_to_time(index, sr, hop_length):
    return index * hop_length / sr

chroma1, sr1 = extract_chroma('original.wav')
chroma2, sr2 = extract_chroma('reverse_included_2.wav')
cost, wp = compute_dtw(chroma1, chroma2)
changes = detect_changed_regions(wp, threshold=10)
grouped = group_changes(changes)

time_changes = [(index_to_time(a[0], sr1, 512), index_to_time(b[0], sr2, 512)) for a, b in grouped]
print(time_changes)
