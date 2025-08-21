#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

# Ensure local viz module is importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from viz.plotting import plot_tau_trends_ci, plot_ci_1d


def latest_run_dir(species_dir: str) -> str | None:
    runs = sorted(glob.glob(os.path.join(species_dir, '*')))
    return runs[-1] if runs else None


def load_csv_matrix(path: str) -> tuple[np.ndarray, list[str]]:
    import csv
    times = []
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        keys = [k for k in header if k.lower() != 'time_s']
        time_idx = header.index('time_s') if 'time_s' in header else None
        for r in reader:
            if time_idx is not None:
                times.append(float(r[time_idx]))
                vals = [float(r[i]) for i, k in enumerate(header) if k != 'time_s']
            else:
                vals = [float(x) for x in r]
            rows.append(vals)
    arr = np.array(rows, dtype=float)
    time = np.array(times, dtype=float) if times else np.arange(arr.shape[0], dtype=float)
    return (time, keys), arr


def bootstrap_ci(arr: np.ndarray, percent: float = 95.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.nanpercentile(arr, (100 - percent) / 2, axis=0)
    hi = np.nanpercentile(arr, 100 - (100 - percent) / 2, axis=0)
    mean = np.nanmean(arr, axis=0)
    return mean, lo, hi


def compute_spike_rate_ci(spike_times: np.ndarray, duration_s: float, window_s: float = 3600.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if spike_times.size == 0 or duration_s <= 0:
        t = np.arange(1)
        m = np.zeros_like(t, dtype=float)
        lo = np.zeros_like(t, dtype=float)
        hi = np.zeros_like(t, dtype=float)
        return t, m, lo, hi
    t_edges = np.arange(0.0, duration_s + window_s, window_s)
    counts, _ = np.histogram(spike_times, bins=t_edges)
    rate = counts / (window_s / 3600.0)
    # For CI, use Poisson approx: mean ± 1.96*sqrt(mean)
    mean = rate
    se = np.sqrt(np.maximum(mean, 1e-9))
    lo = np.maximum(0.0, mean - 1.96 * se)
    hi = mean + 1.96 * se
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    return t_centers, mean, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', default='results/zenodo')
    ap.add_argument('--out_root', default='results/ci_summaries')
    args = ap.parse_args()

    species_list = [
        'Schizophyllum_commune',
        'Enoki_fungi_Flammulina_velutipes',
        'Ghost_Fungi_Omphalotus_nidiformis',
        'Cordyceps_militari',
    ]

    os.makedirs(args.out_root, exist_ok=True)
    ts = datetime.now().isoformat()

    for sp in species_list:
        sp_dir = os.path.join(args.results_root, sp)
        run_dir = latest_run_dir(sp_dir)
        if not run_dir:
            print(f"[SKIP] no runs for {sp}")
            continue
        out_dir = os.path.join(args.out_root, sp, ts)
        os.makedirs(out_dir, exist_ok=True)

        # τ power CI
        tau_csv = os.path.join(run_dir, 'tau_band_timeseries.csv')
        tau_summary = None
        if os.path.isfile(tau_csv):
            (time_s, tau_keys), arr = load_csv_matrix(tau_csv)
            mean, lo, hi = bootstrap_ci(arr)
            tau_summary = {
                'taus': tau_keys,
                'mean': mean.tolist(),
                'lo': lo.tolist(),
                'hi': hi.tolist(),
            }
            # Plot per-τ CI as multi-line would be heavy; emit the existing tau_trends_ci.png is already present.
            # Save summary JSON
            with open(os.path.join(out_dir, 'tau_power_ci.json'), 'w') as f:
                json.dump(tau_summary, f, indent=2)

        # Spike-rate CI (per hour)
        spike_csv = os.path.join(run_dir, 'spike_times_s.csv')
        spike_summary = None
        if os.path.isfile(spike_csv):
            import csv
            spikes = []
            with open(spike_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        spikes.append(float(row.get('time_s') or row.get('t_s') or 0.0))
                    except Exception:
                        continue
            spike_times = np.array(spikes, dtype=float)
            # duration: infer from last tau time if available, else max spike time
            duration_s = float(time_s[-1]) if isinstance(time_s, np.ndarray) and time_s.size > 0 else (float(spike_times.max()) if spike_times.size else 0.0)
            t_centers, mean_rate, lo_rate, hi_rate = compute_spike_rate_ci(spike_times, duration_s)
            spike_summary = {
                't_center_s': t_centers.tolist(),
                'mean_rate_per_hr': mean_rate.tolist(),
                'lo_rate_per_hr': lo_rate.tolist(),
                'hi_rate_per_hr': hi_rate.tolist(),
            }
            plot_ci_1d(
                time_s=t_centers,
                mean=mean_rate,
                lo=lo_rate,
                hi=hi_rate,
                title=f"{sp.replace('_',' ')} — spike rate CI (per hour)",
                ylabel='spikes/hour',
                out_path=os.path.join(run_dir, 'spike_rate_ci.png'),
            )
            with open(os.path.join(out_dir, 'spike_rate_ci.json'), 'w') as f:
                json.dump(spike_summary, f, indent=2)

        # Write a small index JSON
        index = {
            'species': sp,
            'created_by': 'joe knowles',
            'timestamp': ts,
            'sources': {
                'run_dir': run_dir,
                'tau_csv': tau_csv if os.path.isfile(tau_csv) else None,
                'spike_csv': spike_csv if os.path.isfile(spike_csv) else None,
            },
            'tau_power_ci': tau_summary,
            'spike_rate_ci': spike_summary,
        }
        with open(os.path.join(out_dir, 'index.json'), 'w') as f:
            json.dump(index, f, indent=2)
        print(f"[OK] CI summaries for {sp} → {out_dir}")


if __name__ == '__main__':
    main()
