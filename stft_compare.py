# input two audio files
# convert wav to stft
# DTW alignment analysis

import librosa
from librosa.sequence import dtw
from scipy.spatial.distance import cdist
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# Add argument parser
parser = argparse.ArgumentParser(description='Compare two audio files using STFT and DTW analysis')
parser.add_argument('--input', nargs=2, required=True, metavar=('file1.wav', 'file2.wav'),
                    help='Two audio files to compare')
args = parser.parse_args()

# Extract input file paths
input_file1, input_file2 = args.input

# Create output directory based on first input file name
base_name = os.path.splitext(os.path.basename(input_file1))[0]
output_dir = f"{base_name}_comparison"
os.makedirs(output_dir, exist_ok=True)

print(f"Input files: {input_file1} vs {input_file2}")
print(f"Output directory: {output_dir}")

y1, sr1 = librosa.load(input_file1, sr=None)
y2, sr2 = librosa.load(input_file2, sr=None)

stft1 = librosa.stft(y1, n_fft=128, hop_length=64, win_length=128)
stft2 = librosa.stft(y2, n_fft=128, hop_length=64, win_length=128)

mag1 = np.abs(stft1)
mag2 = np.abs(stft2)

# Convert to dB scale
db1 = librosa.amplitude_to_db(mag1, ref=np.max)
db2 = librosa.amplitude_to_db(mag2, ref=np.max)

# DTW functions (same as abtest.py)
def compute_dtw(X, Y):
    """Compute DTW using cosine distance"""
    D = cdist(X, Y, metric='cosine')  # cosine works well for spectral features
    cost, wp = dtw(C=D)
    return cost, wp

def detect_changed_regions(wp, threshold=0):
    changed_segments = []
    for _, (a, b) in enumerate(reversed(wp)):  # DTW path is usually from end to start
        if abs(a - b) > threshold:
            changed_segments.append((a, b))
    return changed_segments

def group_changes(changes, min_gap=10):
    if not changes:
        return []

    grouped = []
    start = changes[0]
    for i in range(1, len(changes)):
        if abs(changes[i][0] - changes[i-1][0]) > min_gap: # 음원 a를 기준으로 달라진 인덱스를 그룹으로 묶음
            grouped.append((start, changes[i-1]))
            start = changes[i]
    grouped.append((start, changes[-1]))
    return grouped

def index_to_time(index, sr, hop_length):
    return index * hop_length / sr

# Apply DTW to STFT spectrograms (transpose for time x freq format)
print("Computing DTW alignment...")
stft1_for_dtw = db1.T  # Convert to (time, freq) format
stft2_for_dtw = db2.T

dtw_cost, warping_path = compute_dtw(stft1_for_dtw, stft2_for_dtw)
print(f"DTW cost: {dtw_cost}")

# Detect changed regions (same logic as abtest.py)
changes = detect_changed_regions(warping_path, threshold=10)
grouped_changes = group_changes(changes, min_gap=10)

# Debug: Let's see what deviations we're actually getting
wp_array = np.array(warping_path)
deviations = np.abs(wp_array[:, 0] - wp_array[:, 1])
print(f"\nDTW Path Analysis:")
print(f"Total path points: {len(warping_path)}")
print(f"Deviation stats: min={deviations.min():.1f}, max={deviations.max():.1f}, mean={deviations.mean():.1f}")
print(f"Points with deviation > 10: {np.sum(deviations > 10)}")
print(f"Points with deviation > 50: {np.sum(deviations > 50)}")
print(f"Points with deviation > 100: {np.sum(deviations > 100)}")

print(f"\nDetected {len(changes)} individual changes")
print(f"Grouped into {len(grouped_changes)} regions")

# Convert to time - fix the logic to handle grouped changes properly
hop_length = 64
time_changes = []
for start_change, end_change in grouped_changes:
    # start_change and end_change are tuples (a, b) representing frame indices
    start_time = index_to_time(start_change[0], sr1, hop_length)
    end_time = index_to_time(end_change[0], sr1, hop_length)
    time_changes.append((start_time, end_time))

# Output format same as abtest.py
for i, (start_time, end_time) in enumerate(time_changes):
    print(f"{start_time:.2f} ~ {end_time:.2f}")

# Debug: Let's also print the actual frame changes to understand what's happening
print("\nDebug - Frame changes:")
for i, (start_change, end_change) in enumerate(grouped_changes):
    print(f"Region {i+1}: frames {start_change} to {end_change}")

# Warping path visualization
plt.figure(figsize=(10, 8))

# Plot DTW alignment path
wp_array = np.array(warping_path)
plt.plot(wp_array[:, 1], wp_array[:, 0], 'b-', linewidth=1, alpha=0.7)
plt.scatter(wp_array[:, 1], wp_array[:, 0], c='red', s=1, alpha=0.5)
plt.xlabel(f'Time frames ({os.path.basename(input_file2)})')
plt.ylabel(f'Time frames ({os.path.basename(input_file1)})')
plt.title('DTW Alignment Path')
plt.grid(True, alpha=0.3)

# Highlight grouped change regions using actual frame indices
for i, (start_change, end_change) in enumerate(grouped_changes[:5]):  # Show first 5 regions
    # Use the actual frame indices from the changes
    start_frame_1 = start_change[0]  # Frame index in file 1
    end_frame_1 = end_change[0]      # Frame index in file 1
    start_frame_2 = start_change[1]  # Frame index in file 2
    end_frame_2 = end_change[1]      # Frame index in file 2
    
    # Highlight regions on both axes
    plt.axhspan(start_frame_1, end_frame_1, alpha=0.3, color='orange', 
                label=f'Region {i+1}' if i < 3 else '')
    plt.axvspan(start_frame_2, end_frame_2, alpha=0.3, color='orange')

if grouped_changes:
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dtw_warping_path.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Warping path visualization saved to {output_dir}/dtw_warping_path.png")

# # Save numerical results
# np.savez(os.path.join(output_dir, 'dtw_analysis_data.npz'),
#          dtw_cost=dtw_cost,
#          warping_path=np.array(warping_path),
#          grouped_changes=np.array(grouped_changes) if grouped_changes else np.array([]),
#          time_changes=np.array(time_changes) if time_changes else np.array([]),
#          hop_length=hop_length,
#          sr=sr1)

# print(f"Analysis data saved to {output_dir}/dtw_analysis_data.npz")