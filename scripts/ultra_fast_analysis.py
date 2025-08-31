#!/usr/bin/env python3
"""
Ultra-Fast Multi-Channel Analysis

Streamlined version focusing on essential metrics with maximum speed optimization.
"""

import os
import numpy as np
import json
import datetime as _dt
from typing import Dict, List, Tuple
import pandas as pd
import multiprocessing as mp
from functools import partial

# Fast imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def ultra_fast_load(file_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Ultra-fast data loading with minimal processing.

    Args:
        file_path: Path to data file

    Returns:
        Tuple of (channel_dict, metadata)
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find first numeric line
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not any(c.isalpha() for c in line):
            try:
                float(line.split()[0])
                start_idx = i
                break
            except:
                continue

    # Fast parsing
    data = []
    for line in lines[start_idx:]:
        line = line.strip()
        if line:
            try:
                values = [float(x) for x in line.split()]
                data.append(values)
            except:
                continue

    if not data:
        raise ValueError("No data found")

    arr = np.array(data, dtype=np.float32)
    channels = {f"ch_{i+1}": arr[:, i] for i in range(arr.shape[1])}

    return channels

def ultra_fast_spike_detection(signal: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Ultra-fast spike detection using vectorized operations.

    Args:
        signal: Input signal
        threshold: Detection threshold (MAD-based)

    Returns:
        Array of spike times (indices)
    """
    # Fast baseline removal
    baseline = pd.Series(signal).rolling(window=1000, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    detrended = signal - baseline

    # MAD-based threshold
    mad = np.median(np.abs(detrended - np.median(detrended)))
    dynamic_threshold = threshold * mad

    # Find peaks above threshold
    abs_signal = np.abs(detrended)
    peaks = np.where(abs_signal > dynamic_threshold)[0]

    # Remove adjacent peaks (refractory period)
    if len(peaks) > 1:
        diffs = np.diff(peaks)
        keep = np.concatenate([[True], diffs > 50])  # 50 sample refractory
        peaks = peaks[keep]

    return peaks

def ultra_fast_complexity(signal: np.ndarray, spike_times: np.ndarray) -> Dict:
    """
    Ultra-fast complexity estimation.

    Args:
        signal: Original signal
        spike_times: Spike time indices

    Returns:
        Complexity metrics
    """
    if len(spike_times) < 3:
        return {'complexity': 0.0, 'type': 'low'}

    # ISI complexity
    isis = np.diff(spike_times)
    if len(isis) < 2:
        return {'complexity': 0.0, 'type': 'low'}

    cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
    complexity = min(cv, 2.0)  # Cap at 2.0

    if complexity > 1.0:
        comp_type = 'high'
    elif complexity > 0.5:
        comp_type = 'medium'
    else:
        comp_type = 'low'

    return {'complexity': float(complexity), 'type': comp_type}

def ultra_fast_tau_analysis(signal: np.ndarray, fs: float = 1.0) -> Dict:
    """
    Ultra-fast dominant frequency analysis.

    Args:
        signal: Input signal
        fs: Sampling frequency

    Returns:
        Dominant frequency information
    """
    # Fast FFT
    n_fft = min(4096, len(signal))
    freqs = np.fft.rfftfreq(n_fft, d=1/fs)
    fft = np.abs(np.fft.rfft(signal[:n_fft]))

    # Find dominant frequency
    peak_idx = np.argmax(fft[1:]) + 1  # Skip DC
    dominant_freq = freqs[peak_idx]

    # Convert to tau (time scale)
    if dominant_freq > 0:
        dominant_tau = 1.0 / dominant_freq
    else:
        dominant_tau = 100.0  # Default

    return {'dominant_tau': float(dominant_tau), 'dominant_freq': float(dominant_freq)}

def process_channel_ultra_fast(channel_data: Tuple[str, np.ndarray], fs: float) -> Dict:
    """
    Process a single channel with ultra-fast algorithms.

    Args:
        channel_data: Tuple of (channel_name, signal)
        fs: Sampling frequency

    Returns:
        Channel analysis results
    """
    name, signal = channel_data

    # Ultra-fast spike detection
    spike_times = ultra_fast_spike_detection(signal)

    # Ultra-fast complexity
    complexity = ultra_fast_complexity(signal, spike_times)

    # Ultra-fast tau analysis
    tau_info = ultra_fast_tau_analysis(signal, fs)

    return {
        'channel': name,
        'spike_count': len(spike_times),
        'spike_rate_hz': len(spike_times) / (len(signal) / fs),
        'complexity_score': complexity['complexity'],
        'complexity_type': complexity['type'],
        'dominant_tau_s': tau_info['dominant_tau'],
        'dominant_freq_hz': tau_info['dominant_freq'],
        'signal_std': float(np.std(signal)),
        'signal_mean': float(np.mean(signal))
    }

def run_ultra_fast_analysis(file_path: str, output_file: str = None, workers: int = None) -> Dict:
    """
    Run ultra-fast multi-channel analysis.

    Args:
        file_path: Path to data file
        output_file: Output JSON file path
        workers: Number of parallel workers

    Returns:
        Analysis results
    """
    print("⚡ ULTRA-FAST Multi-Channel Analysis")
    print("=" * 50)

    if workers is None:
        workers = min(mp.cpu_count(), 8)

    # Ultra-fast loading
    print("📊 Loading data...")
    channels = ultra_fast_load(file_path)

    # Parallel processing
    print(f"🔬 Processing {len(channels)} channels with {workers} workers...")

    fs = 1.0  # Assume 1 Hz

    with mp.Pool(workers) as pool:
        results = pool.map(partial(process_channel_ultra_fast, fs=fs), channels.items())

    # Compile results
    summary = {
        'metadata': {
            'file': file_path,
            'n_channels': len(channels),
            'sampling_rate_hz': fs,
            'total_samples': len(next(iter(channels.values()))),
            'analysis_type': 'ultra_fast',
            'timestamp': _dt.datetime.now().isoformat(),
            'parallel_workers': workers
        },
        'channel_results': results,
        'summary_stats': {
            'total_spikes': sum(r['spike_count'] for r in results),
            'active_channels': sum(1 for r in results if r['spike_count'] > 0),
            'avg_spike_rate': np.mean([r['spike_rate_hz'] for r in results]),
            'complexity_distribution': {
                'high': sum(1 for r in results if r['complexity_type'] == 'high'),
                'medium': sum(1 for r in results if r['complexity_type'] == 'medium'),
                'low': sum(1 for r in results if r['complexity_type'] == 'low')
            }
        }
    }

    # Save results
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_file = f'/home/kronos/mushroooom/results/zenodo/_composites/{timestamp}_ultra_fast_{base_name}.json'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n🎉 Ultra-Fast Analysis Complete!")
    print(f"📊 Processed {len(channels)} channels in parallel")
    print(f"📈 Total spikes detected: {summary['summary_stats']['total_spikes']}")
    print(f"⚡ Active channels: {summary['summary_stats']['active_channels']}")
    print(f"📁 Results saved to: {output_file}")

    return summary

def main():
    """Command-line interface."""
    import argparse

    ap = argparse.ArgumentParser(description='Ultra-Fast Multi-Channel Analysis')
    ap.add_argument('--file', required=True, help='Data file path')
    ap.add_argument('--output', help='Output JSON file')
    ap.add_argument('--workers', type=int, help='Number of parallel workers')

    args = ap.parse_args()

    results = run_ultra_fast_analysis(args.file, args.output, args.workers)

if __name__ == '__main__':
    main()
