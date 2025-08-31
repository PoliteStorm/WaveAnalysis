#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import datetime as _dt
from typing import Dict, Tuple, List
from tqdm import tqdm
import time
import pandas as pd

import prove_transform as pt
from viz import plot_heatmap, plot_surface3d, plot_time_series_with_spikes
from viz.plotting import plot_linepair, assemble_summary_panel, plot_histogram, plot_tau_trends_ci


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad-((w+1)%2)), mode='edge')
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(xp, kernel, mode='valid')
    # Ensure same length as x
    if y.shape[0] != x.shape[0]:
        y = y[:x.shape[0]]
    return y


def detect_spikes(
    v_mV: np.ndarray,
    fs_hz: float,
    min_amp_mV: float,
    max_amp_mV: float,
    min_isi_s: float,
    baseline_win_s: float,
) -> List[Dict]:
    # Detrend with moving average
    w = max(1, int(round(baseline_win_s * fs_hz)))
    baseline = moving_average(v_mV, w)
    x = v_mV - baseline
    # Use absolute peaks above min_amp
    thr = min_amp_mV
    above = np.where(np.abs(x) >= thr)[0]
    events: List[Dict] = []
    if len(above) == 0:
        return events
    # Group contiguous indices into candidate events
    groups: List[Tuple[int, int]] = []
    start = above[0]
    prev = above[0]
    for idx in above[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = idx
        prev = idx
    groups.append((start, prev))
    # Enforce refractory (min ISI)
    min_gap = int(round(min_isi_s * fs_hz))
    filtered: List[Tuple[int, int]] = []
    last_end = -1e9
    for (a, b) in groups:
        if a - last_end < min_gap:
            # merge
            if filtered:
                filtered[-1] = (filtered[-1][0], b)
            else:
                filtered.append((a, b))
        else:
            filtered.append((a, b))
        last_end = filtered[-1][1]
    # Extract peak, amplitude, duration (half-height width)
    for a, b in filtered:
        seg = x[a:b+1]
        if seg.size == 0:
            continue
        # True peak on raw v_mV within [a,b]
        seg_raw = v_mV[a:b+1]
        peak_idx_local = int(np.argmax(np.abs(seg)))
        peak_idx = a + peak_idx_local
        amp = float(v_mV[peak_idx])
        if np.abs(amp) < min_amp_mV or np.abs(amp) > max_amp_mV:
            continue
        # Half-height width on absolute detrended
        half = 0.5 * np.abs(x[peak_idx])
        # search left
        li = peak_idx
        while li > a and np.abs(x[li]) >= half:
            li -= 1
        # search right
        ri = peak_idx
        while ri < b and np.abs(x[ri]) >= half:
            ri += 1
        dur_s = float((ri - li) / fs_hz)
        events.append({
            't_idx': int(peak_idx),
            't_s': float(peak_idx / fs_hz),
            'amplitude_mV': amp,
            'duration_s': dur_s,
        })
    return events


def stats_and_entropy(values: np.ndarray, nbins: int = 20) -> Dict:
    res = {
        'count': int(values.size),
        'mean': float(np.mean(values)) if values.size else 0.0,
        'std': float(np.std(values)) if values.size else 0.0,
        'min': float(np.min(values)) if values.size else 0.0,
        'max': float(np.max(values)) if values.size else 0.0,
        'skewness': 0.0,
        'kurtosis_excess': 0.0,
        'shannon_entropy_bits': 0.0,
    }
    if values.size >= 3:
        x = values.astype(float)
        mu = np.mean(x)
        sig = np.std(x)
        if sig > 0:
            z = (x - mu) / sig
            m3 = float(np.mean(z ** 3))
            m4 = float(np.mean(z ** 4))
            res['skewness'] = m3
            res['kurtosis_excess'] = m4 - 3.0
        # Entropy on histogram
        hist, _ = np.histogram(x, bins=nbins, density=False)
        p = hist.astype(float) / (np.sum(hist) + 1e-12)
        nz = p[p > 0]
        H = -np.sum(nz * np.log2(nz))
        res['shannon_entropy_bits'] = float(H)
    return res


def compute_tau_band_fractions(V: np.ndarray, fs_hz: float, tau_values: List[float], nu0: int) -> Dict:
    t = np.arange(len(V)) / fs_hz
    def V_func(t_vals):
        return np.interp(t_vals, t, V)
    U_max = np.sqrt(t[-1] if len(t) > 1 else 1.0)
    u0_grid = np.linspace(0.0, U_max, nu0, endpoint=False)
    powers = pt.compute_tau_band_powers(V_func, u0_grid, np.array(tau_values))
    dom_idx = np.argmax(powers, axis=1)
    unique, counts = np.unique(dom_idx, return_counts=True)
    fracs = np.zeros(len(tau_values), dtype=float)
    for i, c in zip(unique, counts):
        fracs[i] = c / len(dom_idx)
    return {str(tau_values[i]): float(fracs[i]) for i in range(len(tau_values))}


def analyze_single_channel(V, channel_name, t, args, tau_values):
    """
    Analyze a single channel and return results dictionary.

    Args:
        V: Voltage data array
        channel_name: Name of the channel
        t: Time array
        args: Parsed arguments
        tau_values: List of tau values

    Returns:
        Dictionary containing analysis results
    """
    results = {}

    # Stage 2: Spike Detection
    with tqdm(total=1, desc=f"🔍 Detecting Spikes ({channel_name})", ncols=80) as pbar:
        spikes = detect_spikes(V, args.fs, args.min_amp_mV, args.max_amp_mV, args.min_isi_s, args.baseline_win_s)
        spike_times = np.array([s['t_s'] for s in spikes], dtype=float)
        spike_amps = np.array([s['amplitude_mV'] for s in spikes], dtype=float)
        spike_durs = np.array([s['duration_s'] for s in spikes], dtype=float)
        isi = np.diff(spike_times) if spike_times.size >= 2 else np.array([], dtype=float)
        pbar.update(1)

    # Stage 3: Statistical Analysis
    with tqdm(total=3, desc=f"📈 Computing Statistics ({channel_name})", ncols=80) as pbar:
        amp_stats = stats_and_entropy(spike_amps)
        pbar.update(1)
        dur_stats = stats_and_entropy(spike_durs)
        pbar.update(1)
        isi_stats = stats_and_entropy(isi)
        pbar.update(1)

    # Stage 4: √t Transform Analysis
    with tqdm(total=1, desc=f"🌀 √t Transform ({channel_name})", ncols=80) as pbar:
        band_fracs = compute_tau_band_fractions(V, args.fs, tau_values, args.nu0)
        pbar.update(1)

    # Stage 5: Advanced Spike Train Metrics
    with tqdm(total=2, desc=f"🧠 Advanced Metrics ({channel_name})", ncols=80) as pbar:
        spike_train_metrics = compute_spike_train_metrics(spikes, len(V), args.fs)
        results['spike_train_metrics'] = spike_train_metrics
        pbar.update(1)

        multiscale_entropy = compute_multiscale_entropy(spikes, args.fs)
        results['multiscale_entropy'] = multiscale_entropy
        pbar.update(1)

    # Compile results
    timestamp = _dt.datetime.now().isoformat(timespec='seconds')
    out = {
        'file': args.file,
        'channel': channel_name,
        'fs_hz': args.fs,
        'spike_count': int(len(spikes)),
        'created_by': 'joe knowles',
        'timestamp': timestamp,
        'intended_for': 'peer_review'
    }
    out['amplitude_stats'] = amp_stats
    out['duration_stats'] = dur_stats
    out['isi_stats'] = isi_stats
    out['band_fractions'] = band_fracs

    # Add spike train metrics
    out.update(results)

    return out, spikes, spike_times, spike_amps, spike_durs, t, V

def main():
    ap = argparse.ArgumentParser(description='Spike + stats + √t-band analysis (Zenodo)')
    ap.add_argument('--file', required=True, help='Zenodo TXT file')
    ap.add_argument('--channel', default='', help='Channel name to analyze')
    ap.add_argument('--fs', type=float, default=1.0, help='Sampling rate Hz')
    ap.add_argument('--min_amp_mV', type=float, default=0.2)
    ap.add_argument('--max_amp_mV', type=float, default=50.0)
    ap.add_argument('--min_isi_s', type=float, default=30.0)
    ap.add_argument('--baseline_win_s', type=float, default=600.0)
    ap.add_argument('--taus', default='5.5,24.5,104')
    ap.add_argument('--nu0', type=int, default=128)
    ap.add_argument('--json_out', default='')
    ap.add_argument('--out_dir', default='/home/kronos/mushroooom/results')
    ap.add_argument('--plot', action='store_true', help='Save visuals (heatmaps, 3D, spikes)')
    ap.add_argument('--export_csv', action='store_true', help='Export tau-band time series and spike times as CSV')
    ap.add_argument('--quicklook', action='store_true', help='Faster plotting: fewer windows, skip heavy plots')
    ap.add_argument('--config', type=str, default='', help='Path to species config JSON (overrides args if present)')
    ap.add_argument('--window', type=str, default='gaussian', choices=['gaussian','morlet'], help='Window for √t transform')
    ap.add_argument('--detrend_u', action='store_true', help='Apply linear detrend in u-domain before FFT')
    ap.add_argument('--stimulus_csv', type=str, default='', help='CSV file with stimulus timing data (columns: time_s, stimulus_type)')
    ap.add_argument('--stimulus_window', type=float, default=300.0, help='Analysis window around stimuli (seconds)')
    ap.add_argument('--scan_channels', action='store_true', help='Scan and analyze all available channels (overrides --channel)')
    args = ap.parse_args()
    # optional stimulus CSV
    ap_stim = args

    # Optional species config overrides
    base_name = os.path.splitext(os.path.basename(args.file))[0].replace(' ', '_')
    cfg_path = args.config
    auto_cfg = os.path.join(os.path.dirname(__file__), 'configs', f'{base_name}.json')
    if (not cfg_path) and os.path.isfile(auto_cfg):
        cfg_path = auto_cfg
    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        args.fs = float(cfg.get('fs_hz', args.fs))
        args.min_amp_mV = float(cfg.get('min_amp_mV', args.min_amp_mV))
        args.min_isi_s = float(cfg.get('min_isi_s', args.min_isi_s))
        args.baseline_win_s = float(cfg.get('baseline_win_s', args.baseline_win_s))
        taus_cfg = cfg.get('taus')
        if taus_cfg:
            args.taus = ','.join(str(x) for x in taus_cfg)
        if args.quicklook:
            args.nu0 = int(cfg.get('nu0_quicklook', max(8, args.nu0 // 4)))
        else:
            args.nu0 = int(cfg.get('nu0_plot', args.nu0))

    print("🚀 Starting Fungal Electrophysiology Analysis")
    print(f"📁 Processing: {os.path.basename(args.file)}")
    print(f"⚙️  Sampling Rate: {args.fs} Hz, Window: {args.window}, Detrend: {args.detrend_u}")
    print()

    # Stage 1: Data Loading
    with tqdm(total=2, desc="📊 Loading Data", ncols=80) as pbar:
        tau_values = [float(x) for x in args.taus.split(',') if x.strip()]
        pbar.update(1)

        t, channels = pt.load_zenodo_timeseries(args.file)
        pbar.update(1)

    # Filter channels (remove those with too many NaN)
    valid_channels = {}
    for name, data in channels.items():
        nan_fraction = np.sum(np.isnan(data)) / len(data)
        if nan_fraction < 0.5:  # Keep channels with <50% NaN
            valid_data = pd.Series(data).interpolate(method='linear', limit_direction='both').values
            valid_channels[name] = valid_data

    print(f"✅ Loaded {len(valid_channels)} valid channels from {len(channels)} total")

    # Determine which channels to analyze
    if args.scan_channels:
        channels_to_analyze = list(valid_channels.keys())
        print(f"🔍 Scanning all {len(channels_to_analyze)} channels...")
    else:
        # Single channel mode (original behavior)
        pick = args.channel if args.channel and args.channel in valid_channels else None
        if pick is None:
            for name, vec in valid_channels.items():
                if np.isfinite(vec).any():
                    pick = name
                    break
        channels_to_analyze = [pick] if pick else []
        print(f"✅ Analyzing single channel '{pick}'")

    # Analyze channels
    all_results = {}
    channel_summaries = []

    for i, channel_name in enumerate(channels_to_analyze):
        print(f"\n🔍 Analyzing Channel {i+1}/{len(channels_to_analyze)}: {channel_name}")

        V = valid_channels[channel_name]

        # Analyze this channel
        out, spikes, spike_times, spike_amps, spike_durs, t_chan, V_chan = analyze_single_channel(
            V, channel_name, t, args, tau_values
        )

        all_results[channel_name] = {
            'results': out,
            'spikes': spikes,
            'spike_times': spike_times,
            'spike_amps': spike_amps,
            'spike_durs': spike_durs,
            'time': t_chan,
            'voltage': V_chan
        }

        # Summary for this channel
        summary = {
            'channel': channel_name,
            'spike_count': out['spike_count'],
            'victor_distance': out.get('spike_train_metrics', {}).get('victor_distance', None),
            'complexity': out.get('multiscale_entropy', {}).get('interpretation', 'unknown'),
            'dominant_tau': max(out.get('band_fractions', {}), key=out.get('band_fractions', {}).get) if out.get('band_fractions') else None
        }
        channel_summaries.append(summary)

        print(f"✅ {channel_name}: {out['spike_count']} spikes, complexity: {summary['complexity']}")

    # Multi-channel summary
    if len(channels_to_analyze) > 1:
        print(f"\n📊 Multi-Channel Summary:")
        print(f"   • Total channels analyzed: {len(channels_to_analyze)}")
        print(f"   • Total spikes across all channels: {sum(s['spike_count'] for s in channel_summaries)}")
        print(f"   • Channels with spiking activity: {sum(1 for s in channel_summaries if s['spike_count'] > 0)}")

        # Complexity distribution
        complexities = [s['complexity'] for s in channel_summaries]
        for comp_type in ['high_complexity', 'moderate_complexity', 'low_complexity', 'very_low_complexity']:
            count = complexities.count(comp_type)
            if count > 0:
                print(f"   • {comp_type}: {count} channels")

    # Stage 6: Stimulus-Response Validation (if applicable and single channel mode)
    stimulus_validation = None
    if args.stimulus_csv and os.path.isfile(args.stimulus_csv) and not args.scan_channels:
        with tqdm(total=3, desc="🔬 Stimulus Validation", ncols=80) as pbar:
            try:
                import csv
                stimulus_times = []
                with open(args.stimulus_csv) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'time_s' in row and row['time_s']:
                            stimulus_times.append(float(row['time_s']))
                        elif 't_s' in row and row['t_s']:
                            stimulus_times.append(float(row['t_s']))
                pbar.update(1)

                if stimulus_times:
                    # Use the first channel's data for stimulus validation
                    first_channel = channels_to_analyze[0]
                    V_first = valid_channels[first_channel]
                    stimulus_validation = validate_stimulus_response(
                        V_first, stimulus_times,
                        pre_window=args.stimulus_window,
                        post_window=args.stimulus_window,
                        fs_hz=args.fs
                    )
                    pbar.update(2)
                    print(f"✅ Stimulus-response validation completed: {len(stimulus_times)} stimuli analyzed")
                else:
                    pbar.update(2)
                    print("⚠️  No valid stimulus times found in CSV")
            except Exception as e:
                pbar.update(2)
                print(f"⚠️  Stimulus validation failed: {str(e)}")
    else:
        if args.stimulus_csv and args.scan_channels:
            print("⚠️  Stimulus validation skipped in multi-channel mode")
        elif args.stimulus_csv:
            print(f"⚠️  Stimulus CSV not found: {args.stimulus_csv}")

    # Organize results directory
    base = os.path.splitext(os.path.basename(args.file))[0].replace(' ', '_')
    timestamp = _dt.datetime.now().isoformat(timespec='seconds')
    target_dir = os.path.join(args.out_dir, 'zenodo', base, timestamp)
    os.makedirs(target_dir, exist_ok=True)

    if args.scan_channels:
        # Save multi-channel results
        multichannel_results = {
            'metadata': {
                'file': args.file,
                'channels_analyzed': channels_to_analyze,
                'n_channels': len(channels_to_analyze),
                'fs_hz': args.fs,
                'timestamp': timestamp,
                'created_by': 'joe knowles',
                'intended_for': 'peer_review',
                'analysis_type': 'multichannel_scan'
            },
            'channel_summaries': channel_summaries,
            'individual_results': {ch: data['results'] for ch, data in all_results.items()}
        }

        if stimulus_validation:
            multichannel_results['stimulus_validation'] = stimulus_validation

        json_path = args.json_out if args.json_out else os.path.join(target_dir, 'multichannel_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(multichannel_results, f, default=str)

        print(f"✅ Multi-channel results saved to {json_path}")
    else:
        # Single channel mode - save individual results
        channel_name = channels_to_analyze[0]
        out = all_results[channel_name]['results']
        if stimulus_validation:
            out['stimulus_validation'] = stimulus_validation

        json_path = args.json_out if args.json_out else os.path.join(target_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(out, f, default=str)

    # Write bibliography
    bib_path = os.path.join(target_dir, 'references.md')
    with open(bib_path, 'w') as f:
        f.write("- On spiking behaviour of Pleurotus djamor (Sci Rep 2018): https://www.nature.com/articles/s41598-018-26007-1?utm_source=chatgpt.com\n")
        f.write("- Multiscalar electrical spiking in Schizophyllum commune (Sci Rep 2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/?utm_source=chatgpt.com#Sec2\n")
        f.write("- Language of fungi derived from electrical spiking activity (R. Soc. Open Sci. 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/?utm_source=chatgpt.com\n")
        f.write("- Electrical response of fungi to changing moisture content (Fungal Biol Biotech 2023): https://fungalbiolbiotech.biomedcentral.com/articles/10.1186/s40694-023-00155-0?utm_source=chatgpt.com\n")
        f.write("- Electrical activity of fungi: Spikes detection and complexity analysis (Biosystems 2021): https://www.sciencedirect.com/science/article/pii/S0303264721000307\n")

    print(json.dumps({'json': json_path, 'dir': target_dir}))

    # Stage 7: Visualizations and Exports (if requested)
    if args.plot or args.export_csv:
        print()
        print("🎨 Generating Visualizations and Exports...")

        if args.scan_channels:
            # Multi-channel visualization
            print("📊 Generating multi-channel visualizations...")

            # Create overview plots for all channels
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Multi-Channel Analysis Overview - {base}', fontsize=14)

            # Channel activity summary
            channel_names = list(all_results.keys())
            spike_counts = [all_results[ch]['results']['spike_count'] for ch in channel_names]

            ax = axes[0, 0]
            bars = ax.bar(range(len(channel_names)), spike_counts, alpha=0.7)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Spike Count')
            ax.set_title('Spiking Activity by Channel')
            ax.set_xticks(range(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=45)

            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, spike_counts)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       str(count), ha='center', va='bottom')

            # Complexity distribution
            complexities = [all_results[ch]['results'].get('multiscale_entropy', {}).get('interpretation', 'unknown')
                           for ch in channel_names]

            ax = axes[0, 1]
            comp_counts = {}
            for comp in complexities:
                comp_counts[comp] = comp_counts.get(comp, 0) + 1

            comp_labels = list(comp_counts.keys())
            comp_values = list(comp_counts.values())

            ax.bar(comp_labels, comp_values, alpha=0.7, color='green')
            ax.set_xlabel('Complexity Level')
            ax.set_ylabel('Channel Count')
            ax.set_title('Complexity Distribution')
            ax.tick_params(axis='x', rotation=45)

            # Tau band preferences
            ax = axes[1, 0]
            tau_prefs = []
            for ch in channel_names:
                band_fracs = all_results[ch]['results'].get('band_fractions', {})
                if band_fracs:
                    dominant_tau = max(band_fracs, key=band_fracs.get)
                    tau_prefs.append(float(dominant_tau))
                else:
                    tau_prefs.append(0)

            ax.hist(tau_prefs, bins=10, alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Dominant τ (seconds)')
            ax.set_ylabel('Channel Count')
            ax.set_title('Tau Band Preferences')
            ax.axvline(np.mean([t for t in tau_prefs if t > 0]), color='red',
                      linestyle='--', label=f'Mean: {np.mean([t for t in tau_prefs if t > 0]):.1f}')
            ax.legend()

            # Correlation between channels (simplified)
            ax = axes[1, 1]
            if len(channel_names) >= 2:
                # Simple correlation matrix of spike counts
                spike_matrix = np.array([all_results[ch]['spike_times'] for ch in channel_names])

                # Create correlation matrix if we have enough data
                if all(len(times) > 1 for times in spike_matrix):
                    # This is a simplified correlation - in practice you'd use cross-correlation
                    corr_matrix = np.eye(len(channel_names))
                    ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set_title('Channel Correlation Matrix (Simplified)')
                    ax.set_xticks(range(len(channel_names)))
                    ax.set_yticks(range(len(channel_names)))
                    ax.set_xticklabels(channel_names, rotation=45)
                    ax.set_yticklabels(channel_names)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Channel Correlations')
            else:
                ax.text(0.5, 0.5, 'Need ≥2 channels\nfor correlation',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Channel Correlations')

            plt.tight_layout()
            plt.savefig(os.path.join(target_dir, 'multichannel_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Multi-channel overview saved to {target_dir}")

        else:
            # Single channel visualization (original behavior)
            channel_data = all_results[channels_to_analyze[0]]
            out = channel_data['results']
            spikes = channel_data['spikes']
            spike_times = channel_data['spike_times']
            spike_amps = channel_data['spike_amps']
            V = channel_data['voltage']

            # Count total steps for progress bar
            total_steps = 0
            if args.plot:
                total_steps += 4  # spikes, heatmap, surface, summary
            if args.export_csv:
                total_steps += 2  # tau csv, spike csv

            with tqdm(total=total_steps, desc="📊 Generating Outputs", ncols=80) as pbar:
                # Spike overlay
                t_plot = np.arange(len(V)) / args.fs
                spikes_t = spike_times
                p_spikes = os.path.join(target_dir, 'spikes_overlay.png')
                if args.plot:
                    p_spikes = plot_time_series_with_spikes(
                        t_plot,
                        V,
                        spikes_t,
                        title=f"{base} | {channels_to_analyze[0]} | spikes",
                        out_path=p_spikes,
                        stim_times_s=None,  # Could add stimulus overlay here
                    )
                    pbar.update(1)

                # τ-band powers over windows (u0)
                U_max = np.sqrt(len(V) / args.fs) if len(V) > 1 else 1.0
                u0_grid = np.linspace(0.0, U_max, args.nu0, endpoint=False)
                def V_func(t_vals):
                    return np.interp(t_vals, np.arange(len(V)) / args.fs, V)
                powers = pt.compute_tau_band_powers(V_func, u0_grid, np.array(tau_values), window=args.window, detrend_u=args.detrend_u)

                # Heatmap
                hm_path = os.path.join(target_dir, 'tau_band_power_heatmap.png')
                if args.plot:
                    plot_heatmap(
                        Z=powers,
                        x=np.array(tau_values),
                        y=u0_grid ** 2,
                        title=f"{base} | {channels_to_analyze[0]} | τ-band power vs time",
                        xlabel='τ',
                        ylabel='time (s)',
                        out_path=hm_path,
                        dpi=160,
                    )
                    pbar.update(1)

                # 3D surface
                surf_path = os.path.join(target_dir, 'tau_band_power_surface.png')
                if args.plot:
                    plot_surface3d(
                        Z=powers,
                        x=np.array(tau_values),
                        y=u0_grid ** 2,
                        title=f"{base} | {channels_to_analyze[0]} | τ-band power (3D)",
                        xlabel='τ',
                        ylabel='time (s)',
                        zlabel='power',
                        out_path=surf_path,
                        dpi=160,
                    )
                    pbar.update(1)

                # Summary panel
                if args.plot:
                    comp_path = os.path.join(target_dir, 'stft_vs_sqrt_line.png')
                    panel_path = os.path.join(target_dir, 'summary_panel.png')
                    assemble_summary_panel(
                        [p_spikes, hm_path, surf_path, comp_path],
                        ["Spikes", "τ-band heatmap", "τ-band 3D", "Comparison"],
                        out_path=panel_path,
                    )
                    pbar.update(1)

                # CSV exports
                if args.export_csv:
                    import csv
                    times = (u0_grid ** 2).astype(float)
                    tau_arr = np.array(tau_values, dtype=float)
                    tau_path = os.path.join(target_dir, 'tau_band_timeseries.csv')
                    with open(tau_path, 'w', newline='') as f:
                        meta = [
                            f"# file: {args.file}",
                            f"# species: {base}",
                            f"# channel: {channels_to_analyze[0]}",
                            f"# timestamp: {timestamp}",
                            f"# fs_hz: {args.fs}",
                            f"# taus: {','.join([str(t) for t in tau_arr])}",
                        ]
                        f.write("\n".join(meta) + "\n")
                        w = csv.writer(f)
                        cols = ['time_s'] + [f'tau_{t:g}' for t in tau_arr] + [f'tau_{t:g}_norm' for t in tau_arr]
                        w.writerow(cols)
                        for i, tm in enumerate(times):
                            row_p = powers[i, :].astype(float)
                            norm = float(np.sum(row_p) + 1e-12)
                            row_n = (row_p / norm).tolist()
                            w.writerow([float(tm)] + [float(x) for x in row_p.tolist()] + [float(x) for x in row_n])
                    pbar.update(1)

                    if spike_times.size > 0:
                        spike_csv = os.path.join(target_dir, 'spike_times_s.csv')
                        with open(spike_csv, 'w', newline='') as f:
                            f.write(f"# file: {args.file}\n# species: {base}\n# channel: {channels_to_analyze[0]}\n# timestamp: {timestamp}\n")
                            w = csv.writer(f)
                            w.writerow(['t_s'])
                            for s in spike_times.tolist():
                                w.writerow([float(s)])
                        pbar.update(1)

    print()
    print("🎉 Analysis Complete!")
    if args.scan_channels:
        print(f"📊 Multi-channel analysis: {len(channels_to_analyze)} channels processed")
        print(f"📈 Total spikes across all channels: {sum(s['spike_count'] for s in channel_summaries)}")
        print(f"📁 Results saved to: {target_dir}")
        print(f"📋 Files generated: multichannel_metrics.json, multichannel_overview.png, references.md")
    else:
        channel_name = channels_to_analyze[0]
        out = all_results[channel_name]['results']
        multiscale_entropy = out.get('multiscale_entropy', {})
        print(f"📊 Results saved to: {target_dir}")
        print(f"📈 Found {out['spike_count']} spikes with advanced metrics computed")
        print(f"🧠 Spike train complexity: {multiscale_entropy.get('interpretation', 'unknown')}")
        print(f"📁 Files generated: metrics.json, references.md")
        if args.plot:
            print(f"📊 Visualizations: heatmap, 3D surface, summary panel")
        if args.export_csv:
            print(f"📋 CSV exports: tau_band_timeseries.csv, spike_times_s.csv")


def validate_stimulus_response(v_signal, stimulus_times, pre_window=300, post_window=600, fs_hz=1.0):
    """
    Validate stimulus-response detection capability with comprehensive statistical analysis.

    Args:
        v_signal: Voltage signal array
        stimulus_times: Array of stimulus timing points (in seconds)
        pre_window: Pre-stimulus analysis window (seconds)
        post_window: Post-stimulus analysis window (seconds)
        fs_hz: Sampling frequency

    Returns:
        Dict containing statistical validation results
    """
    import numpy as np
    from scipy import stats

    # Convert stimulus times to sample indices
    stimulus_indices = np.round(np.array(stimulus_times) * fs_hz).astype(int)

    pre_samples = int(pre_window * fs_hz)
    post_samples = int(post_window * fs_hz)

    responses = []

    for stim_idx in stimulus_indices:
        # Define analysis windows
        pre_start = max(0, stim_idx - pre_samples)
        pre_end = stim_idx
        post_start = stim_idx
        post_end = min(len(v_signal), stim_idx + post_samples)

        # Extract signal segments
        pre_signal = v_signal[pre_start:pre_end]
        post_signal = v_signal[post_start:post_end]

        if len(pre_signal) == 0 or len(post_signal) == 0:
            continue

        # Basic statistics
        pre_stats = {
            'mean': float(np.mean(pre_signal)),
            'std': float(np.std(pre_signal)),
            'median': float(np.median(pre_signal)),
            'min': float(np.min(pre_signal)),
            'max': float(np.max(pre_signal))
        }

        post_stats = {
            'mean': float(np.mean(post_signal)),
            'std': float(np.std(post_signal)),
            'median': float(np.median(post_signal)),
            'min': float(np.min(post_signal)),
            'max': float(np.max(post_signal))
        }

        # Statistical tests
        try:
            # T-test for means
            t_stat, p_value = stats.ttest_ind(pre_signal, post_signal, equal_var=False)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(pre_signal) + np.var(post_signal)) / 2)
            if pooled_std > 0:
                cohens_d = abs(np.mean(post_signal) - np.mean(pre_signal)) / pooled_std
            else:
                cohens_d = 0.0

            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(pre_signal, post_signal, alternative='two-sided')

            # Signal change metrics
            mean_change = post_stats['mean'] - pre_stats['mean']
            median_change = post_stats['median'] - pre_stats['median']
            std_change = post_stats['std'] - pre_stats['std']

            response = {
                'stimulus_time_s': float(stim_idx / fs_hz),
                'pre_window_samples': len(pre_signal),
                'post_window_samples': len(post_signal),
                'pre_stats': pre_stats,
                'post_stats': post_stats,
                'statistical_tests': {
                    't_test': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05)
                    },
                    'mann_whitney': {
                        'u_statistic': float(u_stat),
                        'p_value': float(u_p_value),
                        'significant': bool(u_p_value < 0.05)
                    },
                    'effect_size': {
                        'cohens_d': float(cohens_d),
                        'interpretation': _interpret_cohens_d(cohens_d)
                    }
                },
                'signal_changes': {
                    'mean_change': float(mean_change),
                    'median_change': float(median_change),
                    'std_change': float(std_change),
                    'mean_change_percent': float((mean_change / abs(pre_stats['mean'])) * 100) if pre_stats['mean'] != 0 else 0.0
                }
            }

            responses.append(response)

        except Exception as e:
            print(f"Warning: Could not analyze stimulus at {stim_idx / fs_hz:.1f}s: {str(e)}")
            continue

    # Aggregate results
    if responses:
        significant_responses = sum(1 for r in responses if r['statistical_tests']['t_test']['significant'])
        strong_effects = sum(1 for r in responses if r['statistical_tests']['effect_size']['cohens_d'] >= 0.5)

        summary = {
            'total_stimuli': len(stimulus_times),
            'analyzed_responses': len(responses),
            'significant_responses': significant_responses,
            'significant_percentage': float(significant_responses / len(responses) * 100) if responses else 0.0,
            'strong_effect_responses': strong_effects,
            'strong_effect_percentage': float(strong_effects / len(responses) * 100) if responses else 0.0,
            'average_cohens_d': float(np.mean([r['statistical_tests']['effect_size']['cohens_d'] for r in responses])),
            'median_p_value': float(np.median([r['statistical_tests']['t_test']['p_value'] for r in responses]))
        }
    else:
        summary = {
            'total_stimuli': len(stimulus_times),
            'analyzed_responses': 0,
            'significant_responses': 0,
            'significant_percentage': 0.0,
            'strong_effect_responses': 0,
            'strong_effect_percentage': 0.0,
            'average_cohens_d': 0.0,
            'median_p_value': 1.0
        }

    return {
        'summary': summary,
        'individual_responses': responses,
        'analysis_parameters': {
            'pre_window_s': pre_window,
            'post_window_s': post_window,
            'fs_hz': fs_hz,
            'pre_samples': pre_samples,
            'post_samples': post_samples
        }
    }


def compute_spike_train_metrics(spikes, signal_length, fs_hz):
    """
    Compute advanced spike train metrics including Victor distance and other
    sophisticated measures for quantifying spike train structure and complexity.
    """
    if len(spikes) < 2:
        return {
            'victor_distance': None,
            'local_variation': None,
            'cv_squared': None,
            'fano_factor': None,
            'burst_index': None,
            'regularity_metrics': None,
            'complexity_measures': None
        }

    # Extract spike times
    spike_times = np.array([s['t_s'] for s in spikes])
    spike_times = np.sort(spike_times)  # Ensure sorted

    # Calculate ISIs
    isis = np.diff(spike_times)

    if len(isis) < 2:
        return {
            'victor_distance': None,
            'local_variation': None,
            'cv_squared': None,
            'fano_factor': None,
            'burst_index': None,
            'regularity_metrics': None,
            'complexity_measures': None
        }

    # Victor Distance (spike train distance metric)
    # This measures the dissimilarity between spike trains
    victor_distance = np.sqrt(np.sum((isis[1:] - isis[:-1])**2)) / len(isis)

    # Local Variation (LV) - measures irregularity
    lv_numerator = np.sum(3 * (isis[1:-1] - isis[:-2])**2)
    lv_denominator = np.sum((isis[1:-1] + isis[:-2])**2)
    local_variation = lv_numerator / lv_denominator if lv_denominator > 0 else 0

    # CV² (coefficient of variation squared)
    cv_squared_values = []
    for i in range(len(isis) - 1):
        if isis[i] > 0 and isis[i+1] > 0:
            cv2 = (isis[i+1] - isis[i])**2 / ((isis[i] + isis[i+1])/2)**2
            cv_squared_values.append(cv2)
    cv_squared = np.mean(cv_squared_values) if cv_squared_values else 0

    # Fano Factor (variance-to-mean ratio for spike counts)
    # Divide signal into time bins
    bin_size = 60.0  # 1 minute bins
    n_bins = int((signal_length / fs_hz) / bin_size)
    if n_bins > 0:
        spike_counts_per_bin = np.zeros(n_bins)
        for spike_time in spike_times:
            bin_idx = min(int(spike_time / bin_size), n_bins - 1)
            spike_counts_per_bin[bin_idx] += 1

        mean_count = np.mean(spike_counts_per_bin)
        var_count = np.var(spike_counts_per_bin)
        fano_factor = var_count / mean_count if mean_count > 0 else 0
    else:
        fano_factor = 0

    # Burst Index - measures burstiness
    if len(isis) > 0:
        mean_isi = np.mean(isis)
        median_isi = np.median(isis)
        burst_index = median_isi / mean_isi if mean_isi > 0 else 0
    else:
        burst_index = 0

    # Regularity metrics
    regularity_metrics = {
        'cv': np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0,
        'cv2': cv_squared,
        'lv': local_variation,
        'burst_index': burst_index,
        'fano_factor': fano_factor
    }

    # Complexity measures
    complexity_measures = {
        'entropy_rate': -np.sum([p * np.log2(p) for p in isis / np.sum(isis) if p > 0]),
        'fractal_dimension': estimate_fractal_dimension(spike_times),
        'lyapunov_exponent': estimate_lyapunov_exponent(isis),
        'victor_distance': victor_distance
    }

    return {
        'victor_distance': float(victor_distance),
        'local_variation': float(local_variation),
        'cv_squared': float(cv_squared),
        'fano_factor': float(fano_factor),
        'burst_index': float(burst_index),
        'regularity_metrics': regularity_metrics,
        'complexity_measures': complexity_measures
    }


def compute_multiscale_entropy(spikes, fs_hz, max_scale=10):
    """
    Compute multiscale entropy (MSE) of spike train to quantify complexity
    across different time scales.
    """
    if len(spikes) < 10:
        return {
            'mse_values': [],
            'mean_mse': None,
            'complexity_index': None,
            'scale_range': [1, max_scale],
            'interpretation': 'insufficient_data'
        }

    # Extract spike times and convert to binary spike train
    spike_times = np.array([s['t_s'] for s in spikes])
    duration = spike_times[-1] - spike_times[0] if len(spike_times) > 1 else 1.0

    # Create binary spike train (1 at spike times, 0 elsewhere)
    dt = 1.0 / fs_hz  # Time step
    n_samples = int(duration / dt)
    spike_train = np.zeros(n_samples)

    for spike_time in spike_times:
        idx = int((spike_time - spike_times[0]) / dt)
        if 0 <= idx < n_samples:
            spike_train[idx] = 1

    mse_values = []

    for scale in range(1, max_scale + 1):
        # Coarse-graining: average adjacent samples
        if scale == 1:
            coarse_signal = spike_train
        else:
            # Resample by taking every 'scale' samples and averaging
            coarse_signal = []
            for i in range(0, len(spike_train) - scale + 1, scale):
                chunk = spike_train[i:i+scale]
                coarse_signal.append(np.mean(chunk))
            coarse_signal = np.array(coarse_signal)

        if len(coarse_signal) < 3:
            mse_values.append(0)
            continue

        # Compute sample entropy for this scale
        entropy = sample_entropy(coarse_signal, m=2, r=0.2)
        mse_values.append(entropy)

    # Calculate complexity metrics
    mse_array = np.array(mse_values)
    mean_mse = float(np.mean(mse_array))

    # Complexity index: ratio of fine to coarse scale entropy
    if len(mse_array) >= 3:
        complexity_index = float(mse_array[0] / mse_array[-1]) if mse_array[-1] > 0 else 0
    else:
        complexity_index = 0

    # Interpret complexity
    if complexity_index > 1.5:
        interpretation = 'high_complexity'
    elif complexity_index > 1.0:
        interpretation = 'moderate_complexity'
    elif complexity_index > 0.5:
        interpretation = 'low_complexity'
    else:
        interpretation = 'very_low_complexity'

    return {
        'mse_values': [float(x) for x in mse_values],
        'mean_mse': mean_mse,
        'complexity_index': complexity_index,
        'scale_range': [1, max_scale],
        'interpretation': interpretation,
        'description': 'Multiscale entropy quantifies spike train complexity across temporal scales'
    }


def sample_entropy(signal, m=2, r=0.2):
    """
    Compute sample entropy of a signal.
    Sample entropy measures the regularity/complexity of a time series.
    """
    if len(signal) < m + 1:
        return 0

    # Normalize signal
    signal = np.array(signal)
    if np.std(signal) > 0:
        signal = (signal - np.mean(signal)) / np.std(signal)

    # Compute sample entropy
    def _phi(m_val):
        patterns = []
        for i in range(len(signal) - m_val + 1):
            pattern = signal[i:i+m_val]
            patterns.append(pattern)

        patterns = np.array(patterns)
        count = 0

        for i in range(len(patterns)):
            distances = np.max(np.abs(patterns - patterns[i]), axis=1)
            matches = np.sum(distances <= r)
            count += matches

        if count == 0:
            return 0

        return count / (len(patterns) * len(patterns))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return 0

    return -np.log(phi_m1 / phi_m)


def estimate_fractal_dimension(spike_times):
    """Estimate fractal dimension of spike train using sandbox method."""
    if len(spike_times) < 10:
        return 0

    # Simple box-counting approach for spike trains
    spike_times = np.sort(spike_times)
    duration = spike_times[-1] - spike_times[0]

    # Use different box sizes
    box_sizes = np.logspace(-1, 1, 10)  # 0.1 to 10 seconds
    box_counts = []

    for box_size in box_sizes:
        n_boxes = int(duration / box_size) + 1
        boxes = np.zeros(n_boxes)

        for spike_time in spike_times:
            box_idx = int((spike_time - spike_times[0]) / box_size)
            if 0 <= box_idx < n_boxes:
                boxes[box_idx] = 1

        box_counts.append(np.sum(boxes))

    # Estimate fractal dimension from slope
    if len(box_counts) > 1:
        log_boxes = np.log(box_counts)
        log_sizes = np.log(box_sizes[:len(box_counts)])

        # Remove any infinite or NaN values
        valid_idx = np.isfinite(log_boxes) & np.isfinite(log_sizes)
        if np.sum(valid_idx) > 1:
            slope, _ = np.polyfit(log_sizes[valid_idx], log_boxes[valid_idx], 1)
            return -slope  # Negative because N ∝ r^(-D)

    return 0


def estimate_lyapunov_exponent(isis):
    """Estimate largest Lyapunov exponent for spike train dynamics."""
    if len(isis) < 5:
        return 0

    # Simple approach: look at ISI trajectory divergence
    isis = np.array(isis)

    # Compute trajectory differences
    differences = []
    for lag in range(1, min(5, len(isis)//2)):
        diff = np.abs(isis[lag:] - isis[:-lag])
        differences.append(np.mean(diff))

    if len(differences) > 1:
        # Fit exponential: diff ∝ exp(λ * lag)
        lags = np.arange(1, len(differences) + 1)
        log_diffs = np.log(differences)

        # Remove any infinite or NaN values
        valid_idx = np.isfinite(log_diffs)
        if np.sum(valid_idx) > 1:
            slope, _ = np.polyfit(lags[valid_idx], log_diffs[valid_idx], 1)
            return slope  # This is the Lyapunov exponent

    return 0


def _interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


if __name__ == '__main__':
    main()


