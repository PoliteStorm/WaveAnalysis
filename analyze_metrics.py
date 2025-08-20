#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import datetime as _dt
from typing import Dict, Tuple, List

import prove_transform as pt
from viz import plot_heatmap, plot_surface3d, plot_time_series_with_spikes


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


def main():
    ap = argparse.ArgumentParser(description='Spike + stats + √t-band analysis (Zenodo)')
    ap.add_argument('--file', required=True, help='Zenodo TXT file')
    ap.add_argument('--channel', default='', help='Channel name to analyze')
    ap.add_argument('--fs', type=float, default=1.0, help='Sampling rate Hz')
    ap.add_argument('--min_amp_mV', type=float, default=0.2)
    ap.add_argument('--max_amp_mV', type=float, default=50.0)
    ap.add_argument('--min_isi_s', type=float, default=30.0)
    ap.add_argument('--baseline_win_s', type=float, default=300.0)
    ap.add_argument('--taus', default='5.5,24.5,104')
    ap.add_argument('--nu0', type=int, default=128)
    ap.add_argument('--json_out', default='')
    ap.add_argument('--out_dir', default='/home/kronos/mushroooom/results')
    ap.add_argument('--plot', action='store_true', help='Save visuals (heatmaps, 3D, spikes)')
    args = ap.parse_args()

    tau_values = [float(x) for x in args.taus.split(',') if x.strip()]
    t, channels = pt.load_zenodo_timeseries(args.file)
    # pick channel
    pick = args.channel if args.channel and args.channel in channels else None
    if pick is None:
        for name, vec in channels.items():
            if np.isfinite(vec).any():
                pick = name
                break
    V = np.nan_to_num(channels[pick], nan=np.nanmean(channels[pick]))

    # Spike detection
    spikes = detect_spikes(V, args.fs, args.min_amp_mV, args.max_amp_mV, args.min_isi_s, args.baseline_win_s)
    spike_times = np.array([s['t_s'] for s in spikes], dtype=float)
    spike_amps = np.array([s['amplitude_mV'] for s in spikes], dtype=float)
    spike_durs = np.array([s['duration_s'] for s in spikes], dtype=float)
    isi = np.diff(spike_times) if spike_times.size >= 2 else np.array([], dtype=float)

    # Stats
    amp_stats = stats_and_entropy(spike_amps)
    dur_stats = stats_and_entropy(spike_durs)
    isi_stats = stats_and_entropy(isi)

    # √t band fractions
    band_fracs = compute_tau_band_fractions(V, args.fs, tau_values, args.nu0)

    timestamp = _dt.datetime.now().isoformat(timespec='seconds')
    out = {
        'file': args.file,
        'channel': pick,
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

    # Organize results directory
    base = os.path.splitext(os.path.basename(args.file))[0].replace(' ', '_')
    target_dir = os.path.join(args.out_dir, 'zenodo', base, timestamp)
    os.makedirs(target_dir, exist_ok=True)
    json_path = args.json_out if args.json_out else os.path.join(target_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(out, f)
    # Write bibliography
    bib_path = os.path.join(target_dir, 'references.md')
    with open(bib_path, 'w') as f:
        f.write("- On spiking behaviour of Pleurotus djamor (Sci Rep 2018): https://www.nature.com/articles/s41598-018-26007-1?utm_source=chatgpt.com\n")
        f.write("- Multiscalar electrical spiking in Schizophyllum commune (Sci Rep 2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/?utm_source=chatgpt.com#Sec2\n")
        f.write("- Language of fungi derived from electrical spiking activity (R. Soc. Open Sci. 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/?utm_source=chatgpt.com\n")
        f.write("- Electrical response of fungi to changing moisture content (Fungal Biol Biotech 2023): https://fungalbiolbiotech.biomedcentral.com/articles/10.1186/s40694-023-00155-0?utm_source=chatgpt.com\n")
        f.write("- Electrical activity of fungi: Spikes detection and complexity analysis (Biosystems 2021): https://www.sciencedirect.com/science/article/pii/S0303264721000307\n")
    print(json.dumps({'json': json_path, 'dir': target_dir}))

    # Optional visuals
    if args.plot:
        # Spike overlay
        t = np.arange(len(V)) / args.fs
        spikes_t = np.array([s['t_s'] for s in spikes], dtype=float) if spikes else np.array([], dtype=float)
        plot_time_series_with_spikes(
            t,
            V,
            spikes_t,
            title=f"{base} | {pick} | spikes",
            out_path=os.path.join(target_dir, 'spikes_overlay.png'),
        )
        # Heatmap of τ-band powers over windows (u0)
        # Recompute powers with the same settings to ensure alignment
        U_max = np.sqrt(len(V) / args.fs) if len(V) > 1 else 1.0
        u0_grid = np.linspace(0.0, U_max, args.nu0, endpoint=False)
        def V_func(t_vals):
            return np.interp(t_vals, np.arange(len(V)) / args.fs, V)
        powers = pt.compute_tau_band_powers(V_func, u0_grid, np.array(tau_values))
        # Heatmap: rows=u0 (time), cols=tau
        hm_path = os.path.join(target_dir, 'tau_band_power_heatmap.png')
        plot_heatmap(
            Z=powers,
            x=np.array(tau_values),
            y=u0_grid ** 2,
            title=f"{base} | {pick} | τ-band power vs time",
            xlabel='τ',
            ylabel='time (s)',
            out_path=hm_path,
        )
        # 3D surface of the same
        surf_path = os.path.join(target_dir, 'tau_band_power_surface.png')
        plot_surface3d(
            Z=powers,
            x=np.array(tau_values),
            y=u0_grid ** 2,
            title=f"{base} | {pick} | τ-band power (3D)",
            xlabel='τ',
            ylabel='time (s)',
            zlabel='power',
            out_path=surf_path,
        )


if __name__ == '__main__':
    main()


