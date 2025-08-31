#!/usr/bin/env python3
"""
Fast Multi-Channel Correlation Analysis Framework

Optimized for speed and efficiency with parallel processing and smart caching.
"""

import os
import numpy as np
import json
import datetime as _dt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp
from functools import partial
import pandas as pd

# Import our custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import prove_transform as pt
except ImportError:
    print("Warning: Could not import prove_transform module")
    pt = None

def fast_load_multichannel_data(file_path: str, fs_hz: float = 1.0) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Fast loading of multi-channel data with optimized parsing.

    Args:
        file_path: Path to data file
        fs_hz: Sampling frequency

    Returns:
        Tuple of (time_array, channel_dict, metadata)
    """
    print(f"⚡ Fast-loading multi-channel data...")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Fast parsing - skip header and parse numeric data
        lines = f.readlines()

    # Find first numeric line
    start_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line[0].isalpha():
            try:
                float(line.split()[0])
                start_idx = i
                break
            except (ValueError, IndexError):
                continue

    # Parse numeric data efficiently
    numeric_lines = lines[start_idx:]
    data_matrix = []

    for line in numeric_lines:
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.split()]
            data_matrix.append(values)
        except (ValueError, IndexError):
            continue

    if not data_matrix:
        raise ValueError(f"No numeric data found in {file_path}")

    # Convert to numpy array efficiently
    arr = np.array(data_matrix, dtype=np.float32)
    n_samples = arr.shape[0]

    # Create time array
    t = np.arange(n_samples, dtype=np.float32) / fs_hz

    # Create channel dictionary efficiently
    channels = {}
    n_channels = arr.shape[1]

    for j in range(n_channels):
        channel_data = arr[:, j]
        # Fast NaN handling
        nan_mask = np.isnan(channel_data)
        if np.any(nan_mask):
            # Interpolate NaN values
            valid_mask = ~nan_mask
            if np.any(valid_mask):
                channel_data = np.interp(np.arange(n_samples), np.arange(n_samples)[valid_mask], channel_data[valid_mask])
            else:
                channel_data = np.zeros(n_samples)  # All NaN, set to zero

        channels[f"diff_{j+1}"] = channel_data

    # Filter channels (remove those with too many zeros or constant values)
    valid_channels = {}
    for name, data in channels.items():
        # Check for constant values or too many zeros
        if np.std(data) > 1e-6 and np.count_nonzero(data) > n_samples * 0.1:
            valid_channels[name] = data

    metadata = {
        'file_path': file_path,
        'sampling_rate_hz': fs_hz,
        'n_samples': n_samples,
        'duration_s': n_samples / fs_hz,
        'n_channels': len(valid_channels),
        'channel_names': list(valid_channels.keys()),
        'total_channels_original': n_channels
    }

    print(f"✅ Fast-loaded {len(valid_channels)} valid channels ({n_samples} samples each)")
    return t, valid_channels, metadata

def parallel_channel_analysis(channel_data: Tuple[str, np.ndarray], args, tau_values) -> Dict:
    """
    Analyze a single channel (designed for parallel execution).

    Args:
        channel_data: Tuple of (channel_name, voltage_data)
        args: Parsed arguments
        tau_values: List of tau values

    Returns:
        Analysis results for this channel
    """
    channel_name, V = channel_data

    # Fast spike detection
    spikes = fast_detect_spikes(V, args.fs, args.min_amp_mV, args.max_amp_mV,
                               args.min_isi_s, args.baseline_win_s)

    spike_times = np.array([s['t_s'] for s in spikes], dtype=np.float32)
    spike_amps = np.array([s['amplitude_mV'] for s in spikes], dtype=np.float32)
    spike_durs = np.array([s['duration_s'] for s in spikes], dtype=np.float32)

    # Fast statistical analysis
    amp_stats = fast_stats_and_entropy(spike_amps)
    dur_stats = fast_stats_and_entropy(spike_durs)

    isi = np.diff(spike_times) if spike_times.size >= 2 else np.array([], dtype=np.float32)
    isi_stats = fast_stats_and_entropy(isi)

    # Fast tau analysis
    band_fracs = fast_compute_tau_fractions(V, args.fs, tau_values, args.nu0)

    # Fast advanced metrics
    spike_train_metrics = fast_spike_train_metrics(spikes, len(V), args.fs)
    multiscale_entropy = fast_multiscale_entropy(spikes, args.fs)

    # Compile results
    timestamp = _dt.datetime.now().isoformat(timespec='seconds')
    results = {
        'file': args.file,
        'channel': channel_name,
        'fs_hz': args.fs,
        'spike_count': int(len(spikes)),
        'created_by': 'joe knowles',
        'timestamp': timestamp,
        'intended_for': 'peer_review',
        'amplitude_stats': amp_stats,
        'duration_stats': dur_stats,
        'isi_stats': isi_stats,
        'band_fractions': band_fracs,
        'spike_train_metrics': spike_train_metrics,
        'multiscale_entropy': multiscale_entropy
    }

    return {
        'channel_name': channel_name,
        'results': results,
        'spike_times': spike_times,
        'spike_count': len(spikes),
        'complexity': multiscale_entropy.get('interpretation', 'unknown'),
        'dominant_tau': max(band_fracs, key=band_fracs.get) if band_fracs else None
    }

def fast_detect_spikes(v_mV: np.ndarray, fs_hz: float, min_amp_mV: float,
                      max_amp_mV: float, min_isi_s: float, baseline_win_s: float) -> List[Dict]:
    """
    Fast spike detection optimized for performance.

    Args:
        v_mV: Voltage data
        fs_hz: Sampling frequency
        min_amp_mV: Minimum amplitude threshold
        max_amp_mV: Maximum amplitude threshold
        min_isi_s: Minimum inter-spike interval
        baseline_win_s: Baseline window size

    Returns:
        List of spike dictionaries
    """
    # Fast moving average for baseline
    w = max(1, int(baseline_win_s * fs_hz))
    if w >= len(v_mV):
        baseline = np.full_like(v_mV, np.mean(v_mV))
    else:
        # Use convolution for fast moving average
        kernel = np.ones(w, dtype=np.float32) / w
        baseline = np.convolve(v_mV, kernel, mode='same')
        # Handle edges
        baseline[:w//2] = baseline[w//2]
        baseline[-(w//2):] = baseline[-(w//2)]

    # Detrend
    x = v_mV - baseline

    # Fast peak detection
    thr = min_amp_mV
    abs_x = np.abs(x)

    # Find potential peaks
    above_threshold = abs_x >= thr
    if not np.any(above_threshold):
        return []

    peak_indices = []
    i = 0
    n = len(x)

    while i < n:
        if above_threshold[i]:
            # Find local maximum in this region
            start = i
            while i < n and above_threshold[i]:
                i += 1
            end = i

            # Find peak in this window
            window = abs_x[start:end]
            if len(window) > 0:
                peak_idx = start + np.argmax(window)
                if abs_x[peak_idx] >= thr:
                    peak_indices.append(peak_idx)
        else:
            i += 1

    # Apply refractory period
    if not peak_indices:
        return []

    min_gap = int(min_isi_s * fs_hz)
    filtered_peaks = [peak_indices[0]]

    for peak in peak_indices[1:]:
        if peak - filtered_peaks[-1] >= min_gap:
            filtered_peaks.append(peak)

    # Extract spike properties
    spikes = []
    for peak_idx in filtered_peaks:
        amp = float(v_mV[peak_idx])

        if abs(amp) > max_amp_mV:
            continue

        # Fast half-width calculation
        half_amp = abs(amp) / 2
        left_idx = peak_idx
        while left_idx > 0 and abs(x[left_idx]) >= half_amp:
            left_idx -= 1

        right_idx = peak_idx
        while right_idx < len(x) - 1 and abs(x[right_idx]) >= half_amp:
            right_idx += 1

        duration_s = float((right_idx - left_idx) / fs_hz)

        spikes.append({
            't_idx': int(peak_idx),
            't_s': float(peak_idx / fs_hz),
            'amplitude_mV': amp,
            'duration_s': duration_s,
        })

    return spikes

def fast_stats_and_entropy(values: np.ndarray, nbins: int = 20) -> Dict:
    """
    Fast statistical analysis with entropy calculation.

    Args:
        values: Array of values
        nbins: Number of histogram bins

    Returns:
        Dictionary of statistics
    """
    if values.size == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'skewness': 0.0,
            'kurtosis_excess': 0.0,
            'shannon_entropy_bits': 0.0,
        }

    # Fast statistics using numpy
    res = {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'skewness': 0.0,
        'kurtosis_excess': 0.0,
        'shannon_entropy_bits': 0.0,
    }

    if values.size >= 3:
        # Fast higher-order statistics
        mu = res['mean']
        sig = res['std']

        if sig > 0:
            z = (values - mu) / sig
            res['skewness'] = float(np.mean(z ** 3))
            res['kurtosis_excess'] = float(np.mean(z ** 4) - 3.0)

        # Fast entropy calculation
        if nbins > 0:
            hist, _ = np.histogram(values, bins=nbins, density=False)
            p = hist.astype(np.float32) / (np.sum(hist) + 1e-12)
            nz = p[p > 0]
            if len(nz) > 0:
                res['shannon_entropy_bits'] = float(-np.sum(nz * np.log2(nz)))

    return res

def fast_compute_tau_fractions(V: np.ndarray, fs_hz: float, tau_values: List[float], nu0: int) -> Dict:
    """
    Fast computation of tau band fractions.

    Args:
        V: Voltage data
        fs_hz: Sampling frequency
        tau_values: List of tau values
        nu0: Number of u0 points

    Returns:
        Dictionary of band fractions
    """
    if pt is None or len(V) < 10:
        return {}

    try:
        t = np.arange(len(V), dtype=np.float32) / fs_hz

        def V_func(t_vals):
            return np.interp(t_vals, t, V, left=0, right=0)

        U_max = np.sqrt(t[-1] if len(t) > 1 else 1.0)
        u0_grid = np.linspace(0.0, U_max, nu0, endpoint=False, dtype=np.float32)

        # Fast computation
        powers = pt.compute_tau_band_powers(V_func, u0_grid, np.array(tau_values, dtype=np.float32))

        # Fast dominant band calculation
        dom_idx = np.argmax(powers, axis=1)
        unique, counts = np.unique(dom_idx, return_counts=True)
        fracs = np.zeros(len(tau_values), dtype=np.float32)
        for i, c in zip(unique, counts):
            fracs[i] = c / len(dom_idx)

        return {str(tau_values[i]): float(fracs[i]) for i in range(len(tau_values))}

    except Exception:
        return {}

def fast_spike_train_metrics(spikes: List[Dict], signal_length: int, fs_hz: float) -> Dict:
    """
    Fast computation of spike train metrics.

    Args:
        spikes: List of spike dictionaries
        signal_length: Length of original signal
        fs_hz: Sampling frequency

    Returns:
        Dictionary of spike train metrics
    """
    if len(spikes) < 2:
        return {
            'victor_distance': None,
            'local_variation': None,
            'cv_squared': None,
            'fano_factor': None,
            'burst_index': None,
        }

    # Extract spike times efficiently
    spike_times = np.array([s['t_s'] for s in spikes], dtype=np.float32)
    isis = np.diff(spike_times)

    if len(isis) < 2:
        return {
            'victor_distance': None,
            'local_variation': None,
            'cv_squared': None,
            'fano_factor': None,
            'burst_index': None,
        }

    # Fast Victor distance
    victor_distance = float(np.sqrt(np.sum((isis[1:] - isis[:-1])**2)) / len(isis))

    # Fast local variation
    lv_numerator = np.sum(3 * (isis[1:-1] - isis[:-2])**2)
    lv_denominator = np.sum((isis[1:-1] + isis[:-2])**2)
    local_variation = float(lv_numerator / lv_denominator) if lv_denominator > 0 else 0.0

    # Fast CV²
    cv_squared_values = []
    for i in range(len(isis) - 1):
        if isis[i] > 0 and isis[i+1] > 0:
            cv2 = (isis[i+1] - isis[i])**2 / ((isis[i] + isis[i+1])/2)**2
            cv_squared_values.append(cv2)
    cv_squared = float(np.mean(cv_squared_values)) if cv_squared_values else 0.0

    # Fast Fano factor
    bin_size = 60.0  # 1 minute bins
    n_bins = max(1, int(signal_length / fs_hz / bin_size))
    spike_counts_per_bin = np.zeros(n_bins, dtype=np.int32)

    for spike_time in spike_times:
        bin_idx = min(int(spike_time / bin_size), n_bins - 1)
        spike_counts_per_bin[bin_idx] += 1

    mean_count = np.mean(spike_counts_per_bin)
    var_count = np.var(spike_counts_per_bin, dtype=np.float32)
    fano_factor = float(var_count / mean_count) if mean_count > 0 else 0.0

    # Fast burst index
    if len(isis) > 0:
        mean_isi = np.mean(isis)
        median_isi = np.median(isis)
        burst_index = float(median_isi / mean_isi) if mean_isi > 0 else 0.0
    else:
        burst_index = 0.0

    return {
        'victor_distance': victor_distance,
        'local_variation': local_variation,
        'cv_squared': cv_squared,
        'fano_factor': fano_factor,
        'burst_index': burst_index,
    }

def fast_multiscale_entropy(spikes: List[Dict], fs_hz: float, max_scale: int = 6) -> Dict:
    """
    Fast multiscale entropy calculation.

    Args:
        spikes: List of spike dictionaries
        fs_hz: Sampling frequency
        max_scale: Maximum scale

    Returns:
        Dictionary with MSE results
    """
    if len(spikes) < 10:
        return {
            'mse_values': [],
            'mean_mse': None,
            'complexity_index': None,
            'scale_range': [1, max_scale],
            'interpretation': 'insufficient_data'
        }

    # Create binary spike train efficiently
    spike_times = np.array([s['t_s'] for s in spikes], dtype=np.float32)
    duration = spike_times[-1] - spike_times[0] if len(spike_times) > 1 else 1.0

    dt = 1.0 / fs_hz
    n_samples = max(100, int(duration / dt))  # Minimum 100 samples
    spike_train = np.zeros(n_samples, dtype=np.int8)

    for spike_time in spike_times:
        idx = int((spike_time - spike_times[0]) / dt)
        if 0 <= idx < n_samples:
            spike_train[idx] = 1

    mse_values = []
    m = 2  # Embedding dimension
    r = 0.2  # Tolerance

    for scale in range(1, min(max_scale + 1, n_samples // 10)):
        if scale == 1:
            coarse_signal = spike_train.astype(np.float32)
        else:
            # Fast downsampling
            coarse_signal = spike_train[::scale].astype(np.float32)
            if len(coarse_signal) < m + 1:
                break

        if len(coarse_signal) < 10:
            break

        # Fast sample entropy
        entropy = fast_sample_entropy(coarse_signal, m, r)
        mse_values.append(entropy)

    if mse_values:
        mse_array = np.array(mse_values, dtype=np.float32)
        mean_mse = float(np.mean(mse_array))

        if len(mse_array) >= 3:
            complexity_index = float(mse_array[0] / mse_array[-1]) if mse_array[-1] > 0 else 0
        else:
            complexity_index = 0

        if complexity_index > 1.5:
            interpretation = 'high_complexity'
        elif complexity_index > 1.0:
            interpretation = 'moderate_complexity'
        elif complexity_index > 0.5:
            interpretation = 'low_complexity'
        else:
            interpretation = 'very_low_complexity'
    else:
        mean_mse = None
        complexity_index = None
        interpretation = 'no_data'

    return {
        'mse_values': [float(x) for x in mse_values],
        'mean_mse': mean_mse,
        'complexity_index': complexity_index,
        'scale_range': [1, len(mse_values)],
        'interpretation': interpretation,
        'description': 'Fast multiscale entropy for spike train complexity'
    }

def fast_sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Fast sample entropy calculation.

    Args:
        signal: Input signal
        m: Embedding dimension
        r: Tolerance

    Returns:
        Sample entropy value
    """
    if len(signal) < m + 1:
        return 0.0

    # Normalize signal
    signal = signal.astype(np.float32)
    if np.std(signal) > 0:
        signal = (signal - np.mean(signal)) / np.std(signal)

    # Fast pattern matching
    n = len(signal)

    # Count m-patterns
    count_m = 0
    for i in range(n - m):
        for j in range(i + 1, n - m + 1):
            dist = np.max(np.abs(signal[i:i+m] - signal[j:j+m]))
            if dist <= r:
                count_m += 1

    # Count m+1-patterns
    count_mp1 = 0
    for i in range(n - m - 1):
        for j in range(i + 1, n - m):
            dist = np.max(np.abs(signal[i:i+m+1] - signal[j:j+m+1]))
            if dist <= r:
                count_mp1 += 1

    if count_m == 0 or count_mp1 == 0:
        return 0.0

    return -np.log(count_mp1 / count_m)

def run_fast_multichannel_analysis(file_path: str, output_dir: str, fs_hz: float = 1.0,
                                  max_workers: Optional[int] = None, quiet: bool = False) -> Dict:
    """
    Run fast multi-channel analysis with parallel processing.

    Args:
        file_path: Path to data file
        output_dir: Output directory
        fs_hz: Sampling frequency
        max_workers: Maximum number of parallel workers
        quiet: Suppress detailed progress output

    Returns:
        Complete analysis results
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers by default

    print("🚀 Starting Fast Multi-Channel Analysis"    print(f"⚡ Using {max_workers} parallel workers")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Fast data loading
    with tqdm(total=1, desc="📊 Fast Loading Data", ncols=80, disable=quiet) as pbar:
        t, channels, metadata = fast_load_multichannel_data(file_path, fs_hz)
        pbar.update(1)

    # Prepare analysis arguments
    class Args:
        def __init__(self):
            self.file = file_path
            self.fs = fs_hz
            self.min_amp_mV = 0.2
            self.max_amp_mV = 50.0
            self.min_isi_s = 30.0
            self.baseline_win_s = 600.0
            self.nu0 = 64  # Reduced for speed

    args = Args()
    tau_values = [5.5, 24.5, 104.0]

    # Parallel analysis
    channel_items = list(channels.items())
    print(f"🔍 Analyzing {len(channel_items)} channels in parallel...")

    all_results = {}
    channel_summaries = []

    # Process channels in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create partial function for parallel execution
        analyze_func = partial(parallel_channel_analysis, args=args, tau_values=tau_values)

        # Submit all tasks
        futures = [executor.submit(analyze_func, channel_item) for channel_item in channel_items]

        # Collect results with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures),
                          desc="🔬 Parallel Analysis",
                          ncols=80,
                          disable=quiet):
            try:
                result = future.result()
                channel_name = result['channel_name']

                all_results[channel_name] = {
                    'results': result['results'],
                    'spike_times': result['spike_times'],
                }

                summary = {
                    'channel': channel_name,
                    'spike_count': result['spike_count'],
                    'victor_distance': result['results'].get('spike_train_metrics', {}).get('victor_distance', None),
                    'complexity': result['complexity'],
                    'dominant_tau': result['dominant_tau']
                }
                channel_summaries.append(summary)

                if not quiet:
                    print(f"✅ {channel_name}: {result['spike_count']} spikes, complexity: {result['complexity']}")

            except Exception as e:
                print(f"❌ Error processing channel: {e}")

    # Multi-channel summary
    if len(channel_summaries) > 1 and not quiet:
        print(f"\n📊 Multi-Channel Summary:")
        print(f"   • Total channels analyzed: {len(channel_summaries)}")
        total_spikes = sum(s['spike_count'] for s in channel_summaries)
        print(f"   • Total spikes across all channels: {total_spikes}")
        active_channels = sum(1 for s in channel_summaries if s['spike_count'] > 0)
        print(f"   • Channels with spiking activity: {active_channels}")

        # Complexity distribution
        complexities = [s['complexity'] for s in channel_summaries]
        for comp_type in ['high_complexity', 'moderate_complexity', 'low_complexity', 'very_low_complexity']:
            count = complexities.count(comp_type)
            if count > 0:
                print(f"   • {comp_type}: {count} channels")

    # Save results
    multichannel_results = {
        'metadata': {
            'file': file_path,
            'channels_analyzed': list(all_results.keys()),
            'n_channels': len(all_results),
            'fs_hz': fs_hz,
            'timestamp': _dt.datetime.now().isoformat(),
            'created_by': 'joe knowles',
            'intended_for': 'peer_review',
            'analysis_type': 'fast_multichannel',
            'parallel_workers': max_workers
        },
        'channel_summaries': channel_summaries,
        'individual_results': {ch: data['results'] for ch, data in all_results.items()}
    }

    json_path = os.path.join(output_dir, 'fast_multichannel_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(multichannel_results, f, indent=2, default=str)

    # Write bibliography
    bib_path = os.path.join(target_dir, 'references.md')
    with open(bib_path, 'w') as f:
        f.write("- On spiking behaviour of Pleurotus djamor (Sci Rep 2018): https://www.nature.com/articles/s41598-018-26007-1?utm_source=chatgpt.com\n")
        f.write("- Multiscalar electrical spiking in Schizophyllum commune (Sci Rep 2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/?utm_source=chatgpt.com#Sec2\n")
        f.write("- Language of fungi derived from electrical spiking activity (R. Soc. Open Sci. 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/?utm_source=chatgpt.com\n")
        f.write("- Electrical response of fungi to changing moisture content (Fungal Biol Biotech 2023): https://fungalbiolbiotech.biomedcentral.com/articles/10.1186/s40694-023-00155-0?utm_source=chatgpt.com\n")
        f.write("- Electrical activity of fungi: Spikes detection and complexity analysis (Biosystems 2021): https://www.sciencedirect.com/science/article/pii/S0303264721000307\n")

    print(f"\n🎉 Fast Multi-Channel Analysis Complete!")
    print(f"📊 Analyzed {len(all_results)} channels using {max_workers} parallel workers")
    print(f"📈 Total spikes: {sum(s['spike_count'] for s in channel_summaries)}")
    print(f"📁 Results saved to {json_path}")

    return multichannel_results

def main():
    """Main function for command-line usage."""
    import argparse

    ap = argparse.ArgumentParser(description='Fast Multi-Channel Correlation Analysis')
    ap.add_argument('--file', required=True, help='Zenodo data file path')
    ap.add_argument('--output_dir', default='', help='Output directory')
    ap.add_argument('--fs', type=float, default=1.0, help='Sampling frequency (Hz)')
    ap.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    ap.add_argument('--quiet', action='store_true', help='Suppress detailed progress output')

    args = ap.parse_args()

    # Set default output directory
    if not args.output_dir:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        timestamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        args.output_dir = f'/home/kronos/mushroooom/results/zenodo/_composites/{timestamp}_fast_multichannel_{base_name}'

    # Run analysis
    results = run_fast_multichannel_analysis(args.file, args.output_dir, args.fs, args.workers, args.quiet)

if __name__ == '__main__':
    main()
