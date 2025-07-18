import librosa
import numpy as np
from scipy.spatial.distance import cdist
from librosa.sequence import dtw

def compute_dtw(X, Y, metric='euclidean'):
    """
    Compute DTW between two sequences
    
    Args:
        X, Y: Input sequences (can be PCM data or features)
        metric: Distance metric ('euclidean', 'cosine', 'manhattan', etc.)
    """
    if metric == 'cosine':
        D = cdist(X, Y, metric='cosine')
    elif metric == 'manhattan':
        D = cdist(X, Y, metric='cityblock')  # cityblock is the same as manhattan
    else:
        D = cdist(X, Y, metric=metric)
    cost, wp = dtw(C=D)
    return cost, wp

def compute_dtw_pcm(pcm1, pcm2, window_size=1024, hop_size=512, metric='euclidean'):
    """
    Compute DTW on PCM data using sliding windows
    
    Args:
        pcm1, pcm2: PCM data arrays
        window_size: Size of sliding window in samples
        hop_size: Hop size between windows in samples
        metric: Distance metric for DTW
    """
    # Create sliding windows
    def extract_windows(pcm, window_size, hop_size):
        windows = []
        for i in range(0, len(pcm) - window_size + 1, hop_size):
            window = pcm[i:i + window_size]
            windows.append(window)
        return np.array(windows)
    
    # Extract windows from both PCM streams
    windows1 = extract_windows(pcm1, window_size, hop_size)
    windows2 = extract_windows(pcm2, window_size, hop_size)
    
    print(f"Extracted {len(windows1)} windows from PCM1, {len(windows2)} windows from PCM2")
    
    # Compute DTW
    cost, wp = compute_dtw(windows1, windows2, metric)
    
    return cost, wp, windows1, windows2

def compute_dtw_pcm_direct(pcm1, pcm2, downsample_factor=1, metric='euclidean'):
    """
    Compute DTW directly on PCM data without window extraction
    
    Args:
        pcm1, pcm2: PCM data arrays
        downsample_factor: Factor to downsample (1=no downsampling, 10=every 10th sample)
        metric: Distance metric for DTW
    """
    # Downsample if needed
    if downsample_factor > 1:
        pcm1_ds = pcm1[::downsample_factor]
        pcm2_ds = pcm2[::downsample_factor]
        print(f"Downsampled from {len(pcm1)} to {len(pcm1_ds)} samples (factor: {downsample_factor})")
    else:
        pcm1_ds = pcm1
        pcm2_ds = pcm2
    
    # Reshape for DTW (each sample becomes a 1D feature)
    X = pcm1_ds.reshape(-1, 1)
    Y = pcm2_ds.reshape(-1, 1)
    
    print(f"Comparing {len(X)} vs {len(Y)} samples directly")
    
    # Compute DTW
    cost, wp = compute_dtw(X, Y, metric)
    
    return cost, wp, pcm1_ds, pcm2_ds

def analyze_dtw_path(wp, windows1, windows2, sample_rate, hop_size):
    """
    Analyze the DTW warping path to find regions of differences
    
    Args:
        wp: DTW warping path
        windows1, windows2: Window arrays
        sample_rate: Sample rate in Hz
        hop_size: Hop size in samples
    """
    # Convert window indices to time
    def window_to_time(window_idx, hop_size, sample_rate):
        return window_idx * hop_size / sample_rate
    
    differences = []
    
    # Analyze the warping path
    for i, j in wp:
                    # Calculate distance between corresponding windows
            if i < len(windows1) and j < len(windows2):
                dist = float(np.linalg.norm(windows1[i] - windows2[j]))
                time1 = window_to_time(i, hop_size, sample_rate)
                time2 = window_to_time(j, hop_size, sample_rate)
                
                if dist > 0.1:  # Threshold for significant difference
                    differences.append((time1, time2, dist))
    
    return differences

def analyze_dtw_path_direct(wp, pcm1_ds, pcm2_ds, sample_rate, downsample_factor=1):
    """
    Analyze the DTW warping path for direct PCM comparison
    
    Args:
        wp: DTW warping path
        pcm1_ds, pcm2_ds: Downsampled PCM data
        sample_rate: Sample rate in Hz
        downsample_factor: Downsampling factor used
    """
    # Convert sample indices to time
    def sample_to_time(sample_idx, sample_rate, downsample_factor):
        return (sample_idx * downsample_factor) / sample_rate
    
    differences = []
    
    # Analyze the warping path
    for i, j in wp:
        # Calculate distance between corresponding samples
        if i < len(pcm1_ds) and j < len(pcm2_ds):
            dist = abs(pcm1_ds[i] - pcm2_ds[j])
            time1 = sample_to_time(i, sample_rate, downsample_factor)
            time2 = sample_to_time(j, sample_rate, downsample_factor)
            
            if dist > 0.01:  # Threshold for significant difference
                differences.append((time1, time2, dist))
    
    return differences

def compute_dtw_hybrid(pcm1, pcm2, window_size=1024, hop_size=512, metric='euclidean'):
    """
    Hybrid approach: Use windowed DTW to find regions of interest, then detailed analysis
    
    Args:
        pcm1, pcm2: PCM data arrays
        window_size: Size of sliding window in samples
        hop_size: Hop size between windows in samples
        metric: Distance metric for DTW
    """
    # Step 1: Use windowed DTW to find regions of interest
    print("Step 1: Windowed DTW for region detection...")
    cost, wp, windows1, windows2 = compute_dtw_pcm(pcm1, pcm2, window_size, hop_size, metric)
    
    # Step 2: Analyze warping path to find regions with differences
    differences = analyze_dtw_path(wp, windows1, windows2, sr1, hop_size)
    
    # Step 3: For each region, do detailed direct comparison
    detailed_regions = []
    
    if differences:
        print(f"\nStep 2: Detailed analysis of {len(differences)} regions...")
        
        for time1, time2, dist in differences[:5]:  # Limit to first 5 regions
            # Convert time to sample indices
            start_sample1 = int(time1 * sr1)
            end_sample1 = int((time1 + window_size/sr1) * sr1)
            start_sample2 = int(time2 * sr1)
            end_sample2 = int((time2 + window_size/sr1) * sr1)
            
            # Extract region for detailed analysis
            region1 = pcm1[start_sample1:end_sample1]
            region2 = pcm2[start_sample2:end_sample2]
            
            print(f"  Analyzing region: {time1:.3f}s - {time1 + window_size/sr1:.3f}s")
            print(f"    Samples: {len(region1)} vs {len(region2)}")
            
            # Direct comparison within this region
            region_changes = compare_pcm_bytes(region1, region2, threshold=0.001)
            
            if region_changes:
                # Convert back to global time
                global_changes = [start_sample1 + change for change in region_changes]
                detailed_regions.append({
                    'time_range': (time1, time1 + window_size/sr1),
                    'changes': global_changes,
                    'change_count': len(region_changes)
                })
    
    return detailed_regions

def compare_pcm_bytes(pcm1, pcm2, threshold=0.01):
    """
    Compare PCM data byte by byte and detect changes
    
    Args:
        pcm1, pcm2: PCM data arrays (normalized float values)
        threshold: Threshold for detecting significant differences
    """
    changes = []
    
    # Ensure both arrays are the same length
    min_length = min(len(pcm1), len(pcm2))
    pcm1 = pcm1[:min_length]
    pcm2 = pcm2[:min_length]

    print(f"pcm1: {pcm1}")
    print(f"pcm2: {pcm2}")
    
    print(f"Comparing {min_length} samples...")
    
    # Compare each sample
    for i in range(min_length):
        diff = abs(pcm1[i] - pcm2[i])
        if diff > threshold:
            changes.append(i)
    
    return changes

def group_consecutive_changes(changes, sample_rate, min_gap_seconds=0.1):
    """
    Group consecutive changes into time segments
    
    Args:
        changes: List of sample indices where changes were detected
        sample_rate: Sample rate in Hz
        min_gap_seconds: Minimum gap in seconds to separate regions
    """
    if not changes:
        return []
    
    min_gap_samples = int(min_gap_seconds * sample_rate)
    grouped = []
    
    start = changes[0]
    for i in range(1, len(changes)):
        gap = changes[i] - changes[i-1]
        if gap > min_gap_samples:
            # End current group and start new one
            end = changes[i-1]
            start_time = start / sample_rate
            end_time = end / sample_rate
            grouped.append((start_time, end_time))
            start = changes[i]
    
    # Add the last group
    end = changes[-1]
    start_time = start / sample_rate
    end_time = end / sample_rate
    grouped.append((start_time, end_time))
    
    return grouped

def analyze_change_statistics(changes, sample_rate):
    """Analyze the statistics of detected changes"""
    if not changes:
        print("No changes detected")
        return
    
    print(f"\nChange Statistics:")
    print(f"Total changes: {len(changes)}")
    print(f"Change percentage: {(len(changes) / (changes[-1] - changes[0] + 1)) * 100:.2f}%")
    
    # Calculate gaps between changes
    if len(changes) > 1:
        gaps = [changes[i] - changes[i-1] for i in range(1, len(changes))]
        gap_times = [gap / sample_rate for gap in gaps]
        
        print(f"Gap statistics:")
        print(f"  Min gap: {min(gap_times):.6f}s")
        print(f"  Max gap: {max(gap_times):.6f}s")
        print(f"  Mean gap: {np.mean(gap_times):.6f}s")
        print(f"  Median gap: {np.median(gap_times):.6f}s")

def compare_pcm_bytes_optimized(pcm1, pcm2, threshold=0.01, chunk_size=10000):
    """
    Optimized PCM comparison that processes data in chunks to save memory
    
    Args:
        pcm1, pcm2: PCM data arrays
        threshold: Threshold for detecting significant differences
        chunk_size: Size of chunks to process at once
    """
    changes = []
    
    # Ensure both arrays are the same length
    min_length = min(len(pcm1), len(pcm2))
    pcm1 = pcm1[:min_length]
    pcm2 = pcm2[:min_length]
    
    print(f"Comparing {min_length} samples in chunks of {chunk_size}...")
    
    # Process in chunks to save memory
    for start_idx in range(0, min_length, chunk_size):
        end_idx = min(start_idx + chunk_size, min_length)
        
        chunk1 = pcm1[start_idx:end_idx]
        chunk2 = pcm2[start_idx:end_idx]
        
        # Compare each sample in this chunk
        for i in range(len(chunk1)):
            diff = abs(chunk1[i] - chunk2[i])
            if diff > threshold:
                changes.append(start_idx + i)
        
        # Progress indicator
        if start_idx % (chunk_size * 10) == 0:
            progress = (start_idx / min_length) * 100
            print(f"  Progress: {progress:.1f}%")
    
    return changes

# Load audio files
y1, sr1 = librosa.load('s4_pad_sametime/s4_without_pad.wav', sr=None)
y2, sr2 = librosa.load('s4_pad_sametime/s4_with_pad.wav', sr=None)

print(f"Audio 1: {len(y1)} samples at {sr1} Hz ({len(y1)/sr1:.2f} seconds)")
print(f"Audio 2: {len(y2)} samples at {sr2} Hz ({len(y2)/sr2:.2f} seconds)")

# Ensure same sample rate
if sr1 != sr2:
    print(f"Warning: Different sample rates! Resampling to {sr1} Hz")
    y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    sr2 = sr1

# Test DTW approach with PCM data (windowed)
print("\n=== Testing DTW with PCM data (windowed) ===")

# Try different window sizes and metrics
window_sizes = [512, 1024, 2048]
metrics = ['euclidean', 'cosine', 'manhattan']

for window_size in window_sizes:
    for metric in metrics:
        print(f"\n--- DTW with window_size={window_size}, metric={metric} ---")
        
        try:
            # Compute DTW on PCM data
            cost, wp, windows1, windows2 = compute_dtw_pcm(y1, y2, window_size, window_size//2, metric)
            
            # Handle cost as array or scalar
            if hasattr(cost, 'shape') and cost.shape:
                cost_value = float(cost[-1, -1])  # Get the final cost value
            else:
                cost_value = float(cost)
            
            print(f"DTW cost: {cost_value:.6f}")
            print(f"Warping path length: {len(wp)}")
            
            # Analyze the warping path
            differences = analyze_dtw_path(wp, windows1, windows2, sr1, window_size//2)
            
            if differences:
                print(f"Found {len(differences)} significant differences")
                
                # Group differences by time regions
                time_regions = []
                for time1, time2, dist in differences[:10]:  # Show first 10
                    print(f"  Time1: {time1:.3f}s, Time2: {time2:.3f}s, Distance: {dist:.6f}")
                    
                    # Create time region
                    start_time = min(time1, time2)
                    end_time = max(time1, time2) + (window_size / sr1)
                    time_regions.append((start_time, end_time))
                
                # Merge overlapping regions
                if time_regions:
                    merged_regions = []
                    time_regions.sort()
                    
                    current_start, current_end = time_regions[0]
                    for start, end in time_regions[1:]:
                        if start <= current_end:
                            current_end = max(current_end, end)
                        else:
                            merged_regions.append((current_start, current_end))
                            current_start, current_end = start, end
                    
                    merged_regions.append((current_start, current_end))
                    
                    print(f"\nMerged time regions:")
                    for i, (start, end) in enumerate(merged_regions):
                        duration = end - start
                        print(f"  Region {i+1}: {start:.3f}s - {end:.3f}s (duration: {duration:.3f}s)")
            else:
                print("No significant differences detected")
                
        except Exception as e:
            print(f"Error with window_size={window_size}, metric={metric}: {e}")

# Test DTW approach with PCM data (direct, no windows)
print("\n=== Testing DTW with PCM data (direct, no windows) ===")

# Try different downsampling factors and metrics
downsample_factors = [1, 10, 100, 1000]  # 1=no downsampling, 10=every 10th sample, etc.
metrics = ['euclidean', 'cosine']

for downsample_factor in downsample_factors:
    for metric in metrics:
        print(f"\n--- DTW with downsample_factor={downsample_factor}, metric={metric} ---")
        
        try:
            # Compute DTW directly on PCM data
            cost, wp, pcm1_ds, pcm2_ds = compute_dtw_pcm_direct(y1, y2, downsample_factor, metric)
            
            # Handle cost as array or scalar
            if hasattr(cost, 'shape') and cost.shape:
                cost_value = float(cost[-1, -1])  # Get the final cost value
            else:
                cost_value = float(cost)
            
            print(f"DTW cost: {cost_value:.6f}")
            print(f"Warping path length: {len(wp)}")
            
            # Analyze the warping path
            differences = analyze_dtw_path_direct(wp, pcm1_ds, pcm2_ds, sr1, downsample_factor)
            
            if differences:
                print(f"Found {len(differences)} significant differences")
                
                # Group differences by time regions
                time_regions = []
                for time1, time2, dist in differences[:10]:  # Show first 10
                    print(f"  Time1: {time1:.3f}s, Time2: {time2:.3f}s, Distance: {dist:.6f}")
                    
                    # Create time region
                    start_time = min(time1, time2)
                    end_time = max(time1, time2) + (downsample_factor / sr1)
                    time_regions.append((start_time, end_time))
                
                # Merge overlapping regions
                if time_regions:
                    merged_regions = []
                    time_regions.sort()
                    
                    current_start, current_end = time_regions[0]
                    for start, end in time_regions[1:]:
                        if start <= current_end:
                            current_end = max(current_end, end)
                        else:
                            merged_regions.append((current_start, current_end))
                            current_start, current_end = start, end
                    
                    merged_regions.append((current_start, current_end))
                    
                    print(f"\nMerged time regions:")
                    for i, (start, end) in enumerate(merged_regions):
                        duration = end - start
                        print(f"  Region {i+1}: {start:.3f}s - {end:.3f}s (duration: {duration:.3f}s)")
            else:
                print("No significant differences detected")
                
        except Exception as e:
            print(f"Error with downsample_factor={downsample_factor}, metric={metric}: {e}")

# Test hybrid approach
print("\n=== Testing Hybrid PCM Comparison ===")

# Try different window sizes and metrics for hybrid
window_sizes = [512, 1024, 2048]
metrics = ['euclidean', 'cosine', 'manhattan']

for window_size in window_sizes:
    for metric in metrics:
        print(f"\n--- Hybrid DTW with window_size={window_size}, metric={metric} ---")
        
        try:
            # Compute DTW on PCM data
            detailed_regions = compute_dtw_hybrid(y1, y2, window_size, window_size//2, metric)
            
            if detailed_regions:
                print(f"Found {len(detailed_regions)} regions with differences")
                
                for i, region_info in enumerate(detailed_regions[:5]): # Show first 5
                    print(f"  Region {i+1}: {region_info['time_range'][0]:.3f}s - {region_info['time_range'][1]:.3f}s")
                    print(f"    Changes: {region_info['change_count']} samples")
                    
                    # Group changes within this region
                    time_segments = group_consecutive_changes(region_info['changes'], sr1)
                    print(f"    Regions within this region: {len(time_segments)}")
                    
                    if len(time_segments) <= 10:  # Show details if not too many
                        for j, (start_time, end_time) in enumerate(time_segments):
                            duration = end_time - start_time
                            print(f"      Region {j+1}: {start_time:.3f}s - {end_time:.3f}s (duration: {duration:.3f}s)")
            else:
                print("No significant differences detected in any region")
                
        except Exception as e:
            print(f"Error with window_size={window_size}, metric={metric}: {e}")

# Test optimized byte-by-byte comparison (memory efficient)
print("\n=== Optimized byte-by-byte comparison (memory efficient) ===")
thresholds = [0.001, 0.01, 0.05, 0.1]

for threshold in thresholds:
    print(f"\n--- Testing threshold: {threshold} ---")
    
    # Compare PCM bytes with memory optimization
    changes = compare_pcm_bytes_optimized(y1, y2, threshold, chunk_size=50000)
    
    if changes:
        print(f"Detected {len(changes)} samples with differences")
        
        # Analyze statistics
        analyze_change_statistics(changes, sr1)
        
        # Group changes into regions
        for min_gap in [0.1, 0.2, 0.5]:
            time_segments = group_consecutive_changes(changes, sr1, min_gap)
            print(f"\nWith {min_gap}s minimum gap: {len(time_segments)} regions")
            
            if len(time_segments) <= 10:  # Show details if not too many
                for i, (start_time, end_time) in enumerate(time_segments):
                    duration = end_time - start_time
                    print(f"  Region {i+1}: {start_time:.3f}s - {end_time:.3f}s (duration: {duration:.3f}s)")
            
            # Stop if we found a reasonable number of regions
            if 2 <= len(time_segments) <= 10:
                break
    else:
        print("No differences detected with this threshold")

# Original approach for comparison (if memory allows)
print("\n=== Original byte-by-byte comparison (if memory allows) ===")
try:
    thresholds = [0.001, 0.01, 0.05, 0.1]
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold} ---")
        
        # Compare PCM bytes
        changes = compare_pcm_bytes(y1, y2, threshold)
        
        if changes:
            print(f"Detected {len(changes)} samples with differences")
            
            # Analyze statistics
            analyze_change_statistics(changes, sr1)
            
            # Group changes into regions
            for min_gap in [0.1, 0.2, 0.5]:
                time_segments = group_consecutive_changes(changes, sr1, min_gap)
                print(f"\nWith {min_gap}s minimum gap: {len(time_segments)} regions")
                
                if len(time_segments) <= 10:  # Show details if not too many
                    for i, (start_time, end_time) in enumerate(time_segments):
                        duration = end_time - start_time
                        print(f"  Region {i+1}: {start_time:.3f}s - {end_time:.3f}s (duration: {duration:.3f}s)")
                
                # Stop if we found a reasonable number of regions
                if 2 <= len(time_segments) <= 10:
                    break
        else:
            print("No differences detected with this threshold")
except MemoryError:
    print("Memory error - skipping original approach")
except Exception as e:
    print(f"Error in original approach: {e}")

# Convert to PCM bytes for display
pcm1_bytes = y1.tobytes()
pcm2_bytes = y2.tobytes()

print(f"\nPCM byte sizes:")
print(f"PCM1: {len(pcm1_bytes)} bytes")
print(f"PCM2: {len(pcm2_bytes)} bytes")

# Write PCM bytes to text file
with open('pcm_bytes.txt', 'w') as f:
    f.write(f"PCM1: {pcm1_bytes}\n")
    f.write(f"PCM2: {pcm2_bytes}\n")

print("PCM bytes written to 'pcm_bytes.txt'")