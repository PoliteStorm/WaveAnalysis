#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import datetime as _dt
from typing import Dict, Tuple, List

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

    # Optional stimulus-response validation
    stimulus_validation = None
    if args.stimulus_csv and os.path.isfile(args.stimulus_csv):
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

            if stimulus_times:
                stimulus_validation = validate_stimulus_response(
                    V, stimulus_times,
                    pre_window=args.stimulus_window,
                    post_window=args.stimulus_window,
                    fs_hz=args.fs
                )
                out['stimulus_validation'] = stimulus_validation
                print(f"✅ Stimulus-response validation completed: {len(stimulus_times)} stimuli analyzed")
            else:
                print("⚠️  No valid stimulus times found in CSV")
        except Exception as e:
            print(f"⚠️  Stimulus validation failed: {str(e)}")
    else:
        if args.stimulus_csv:
            print(f"⚠️  Stimulus CSV not found: {args.stimulus_csv}")

    # Organize results directory
    base = os.path.splitext(os.path.basename(args.file))[0].replace(' ', '_')
    target_dir = os.path.join(args.out_dir, 'zenodo', base, timestamp)
    os.makedirs(target_dir, exist_ok=True)
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

    # Optional visuals and CSV exports
    if args.plot or args.export_csv:
        # Spike overlay
        t = np.arange(len(V)) / args.fs
        spikes_t = np.array([s['t_s'] for s in spikes], dtype=float) if spikes else np.array([], dtype=float)
        p_spikes = os.path.join(target_dir, 'spikes_overlay.png')
        if args.plot:
            # optional stimulus CSV overlay if present
            stim_times = None
            if hasattr(args, 'stim_csv') and args.stim_csv and os.path.isfile(args.stim_csv):
                try:
                    import csv
                    tt = []
                    with open(args.stim_csv) as f:
                        r = csv.DictReader(f)
                        for row in r:
                            if 't_s' in row and row['t_s']:
                                tt.append(float(row['t_s']))
                    if tt:
                        stim_times = np.array(tt, dtype=float)
                except Exception:
                    stim_times = None
            p_spikes = plot_time_series_with_spikes(
                t,
                V,
                spikes_t,
                title=f"{base} | {pick} | spikes",
                out_path=p_spikes,
                stim_times_s=stim_times,
            )
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
                title=f"{base} | {pick} | τ-band power vs time",
                xlabel='τ',
                ylabel='time (s)',
                out_path=hm_path,
                dpi=160,
            )
        # Bootstrap CIs for normalized τ-power trends
        if args.plot:
            # Normalize per-row
            P = powers.astype(float)
            row_sum = np.sum(P, axis=1, keepdims=True) + 1e-12
            Pn = P / row_sum
            # Simple bootstrap over windows (resample rows with replacement)
            rng = np.random.default_rng(0)
            B = 256
            acc = []
            for _ in range(B):
                idx = rng.integers(0, Pn.shape[0], size=Pn.shape[0])
                acc.append(Pn[idx].mean(axis=0))
            acc = np.asarray(acc)  # (B, n_tau)
            mean = Pn  # per-window series
            # For plotting CI as bands across time, compute a rolling mean per τ to stabilize
            means_ts = Pn  # shape (n_time, n_tau)
            lo = np.tile(np.percentile(acc, 2.5, axis=0), (Pn.shape[0], 1))
            hi = np.tile(np.percentile(acc, 97.5, axis=0), (Pn.shape[0], 1))
            plot_tau_trends_ci(
                time_s=u0_grid ** 2,
                taus=np.array(tau_values),
                means=means_ts,
                lo=lo,
                hi=hi,
                title=f"{base} | {pick} | τ-power trends with 95% CI",
                out_path=os.path.join(target_dir, 'tau_trends_ci.png'),
            )
        # 3D surface
        surf_path = os.path.join(target_dir, 'tau_band_power_surface.png')
        if args.plot:
            plot_surface3d(
                Z=powers,
                x=np.array(tau_values),
                y=u0_grid ** 2,
                title=f"{base} | {pick} | τ-band power (3D)",
                xlabel='τ',
                ylabel='time (s)',
                zlabel='power',
                out_path=surf_path,
                dpi=160,
            )
        # STFT vs √t comparison for a mid window + numeric SNR/concentration
        comp_path = os.path.join(target_dir, 'stft_vs_sqrt_line.png')
        if args.plot:
            mid = len(u0_grid) // 2
            u0_mid = float(u0_grid[mid])
            def V_func2(t_vals):
                return np.interp(t_vals, np.arange(len(V)) / args.fs, V)
            u_grid = np.linspace(0, u0_grid[-1] if len(u0_grid) else 1.0, 512, endpoint=False)
            k_fft, W = pt.sqrt_time_transform_fft(V_func2, float(tau_values[0]), u_grid, u0=u0_mid,
                                                  window=args.window, detrend_u=args.detrend_u)
            Pk = np.abs(W) ** 2
            t_grid = np.arange(len(V)) / args.fs
            t0 = u0_mid ** 2
            sigma_t = max(1e-6, 2.0 * u0_mid * float(tau_values[0]))
            omega_fft, G = pt.stft_fft(V_func2, t0, sigma_t, t_grid)
            Pw = np.abs(G) ** 2
            plot_linepair(
                x1=k_fft, y1=Pk, label1='√t | P(k)',
                x2=omega_fft, y2=Pw, label2='STFT | P(ω)',
                title=f"{base} | {pick} | mid-window spectral comparison",
                xlabel1='k', xlabel2='ω', ylabel='power',
                out_path=comp_path,
            )
            # Numeric SNR and concentration
            # target index: max peak
            ti_sqrt = int(np.argmax(Pk))
            ti_stft = int(np.argmax(Pw))
            def conc(arr):
                s = float(np.sum(arr) + 1e-12)
                return float(np.max(arr) / s)
            def snr(arr, idx, excl=3):
                n = len(arr)
                m = np.ones(n, dtype=bool)
                lo = max(0, idx - excl)
                hi = min(n, idx + excl + 1)
                m[lo:hi] = False
                bg = float(np.median(arr[m]) + 1e-12) if np.any(m) else 1e-12
                return float(arr[idx] / bg)
            snr_sqrt = snr(Pk, ti_sqrt)
            snr_stft = snr(Pw, ti_stft)
            conc_sqrt = conc(Pk)
            conc_stft = conc(Pw)
            with open(os.path.join(target_dir, 'snr_concentration.json'), 'w') as f:
                json.dump({
                    'snr': {'sqrt': snr_sqrt, 'stft': snr_stft},
                    'concentration': {'sqrt': conc_sqrt, 'stft': conc_stft}
                }, f)

            # Ablation: Gaussian/Morlet × detrend on/off, plus STFT baseline
            configs = [
                ('gaussian', False),
                ('gaussian', True),
                ('morlet', False),
                ('morlet', True),
            ]
            ab = []
            for win, detr in configs:
                kf, Wv = pt.sqrt_time_transform_fft(V_func2, float(tau_values[0]), u_grid, u0=u0_mid,
                                                    window=win, detrend_u=detr)
                Pv = np.abs(Wv) ** 2
                ti = int(np.argmax(Pv))
                ab.append({
                    'window': win,
                    'detrend_u': bool(detr),
                    'snr': snr(Pv, ti),
                    'concentration': conc(Pv)
                })
            ab_out = {
                'u0': float(u0_mid),
                'tau': float(tau_values[0]),
                'sqrt_ablation': ab,
                'stft': {'snr': snr_stft, 'concentration': conc_stft}
            }
            with open(os.path.join(target_dir, 'snr_ablation.json'), 'w') as f:
                json.dump(ab_out, f, indent=2)
            # Lightweight markdown table
            md = [
                '| Setting | SNR | Concentration |',
                '|---|---:|---:|'
            ]
            for row in ab:
                md.append(f"| √t {row['window']} detrend={row['detrend_u']} | {row['snr']:.2f} | {row['concentration']:.4f} |")
            md.append(f"| STFT | {snr_stft:.2f} | {conc_stft:.4f} |")
            with open(os.path.join(target_dir, 'snr_ablation.md'), 'w') as f:
                f.write('\n'.join(md))
            # Summary panel
            panel_path = os.path.join(target_dir, 'summary_panel.png')
            assemble_summary_panel(
                [p_spikes, hm_path, surf_path, comp_path],
                ["Spikes", "τ-band heatmap", "τ-band 3D", "STFT vs √t"],
                out_path=panel_path,
            )
            # Spike histograms
            if spike_times.size > 1:
                isi = np.diff(spike_times)
                plot_histogram(isi, bins=30, title=f"{base} | {pick} | ISI", xlabel='seconds', out_path=os.path.join(target_dir,'hist_isi.png'))
            if spike_amps.size > 0:
                plot_histogram(spike_amps, bins=30, title=f"{base} | {pick} | amplitude (mV)", xlabel='mV', out_path=os.path.join(target_dir,'hist_amp.png'))
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
                    f"# channel: {pick}",
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
            if spikes_t.size > 0:
                spike_csv = os.path.join(target_dir, 'spike_times_s.csv')
                with open(spike_csv, 'w', newline='') as f:
                    f.write(f"# file: {args.file}\n# species: {base}\n# channel: {pick}\n# timestamp: {timestamp}\n")
                    w = csv.writer(f)
                    w.writerow(['t_s'])
                    for s in spikes_t.tolist():
                        w.writerow([float(s)])


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


