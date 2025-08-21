#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from typing import Tuple, List

import numpy as np

# Ensure local viz import works when executed as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from viz.plotting import plot_heatmap, plot_surface3d


def latest_run_dir(species_dir: str) -> str | None:
    runs = sorted(glob.glob(os.path.join(species_dir, '*')))
    return runs[-1] if runs else None


def read_tau_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read tau_band_timeseries.csv robustly.
    Returns: time_s (N,), taus (M,), Z (M, N) with rows=y=taus, cols=x=time.
    """
    import csv
    with open(path, 'r') as f:
        # skip comment lines
        header_line = None
        while True:
            pos = f.tell()
            line = f.readline()
            if line == '':
                raise ValueError(f"Empty or comment-only CSV: {path}")
            if not line.lstrip().startswith('#'):
                header_line = line
                break
        f.seek(pos)
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        if 'time_s' in header:
            t_idx = header.index('time_s')
            tau_cols: List[int] = [i for i, k in enumerate(header) if k != 'time_s']
            tau_vals: List[float] = [float(header[i]) for i in tau_cols]
        else:
            # assume all columns are taus, generate time index
            t_idx = None
            tau_cols = list(range(len(header)))
            tau_vals = [float(h) for h in header]
        times: List[float] = []
        rows: List[List[float]] = []
        for r in reader:
            if len(r) == 0 or r[0].lstrip().startswith('#'):
                continue
            r = [c.strip() for c in r]
            if t_idx is not None and t_idx < len(r):
                try:
                    times.append(float(r[t_idx]))
                except Exception:
                    continue
            vals: List[float] = []
            for i in tau_cols:
                try:
                    vals.append(float(r[i]))
                except Exception:
                    vals.append(np.nan)
            rows.append(vals)
    arr = np.array(rows, dtype=float)  # shape (N_time, M_tau)
    time_s = np.array(times, dtype=float) if times else np.arange(arr.shape[0], dtype=float)
    taus = np.array(tau_vals, dtype=float)
    Z = np.transpose(arr)  # (M_tau, N_time)
    return time_s, taus, Z


def main():
    ap = argparse.ArgumentParser(description='Replot τ heatmap/surface from existing CSV')
    ap.add_argument('--results_root', default='results/zenodo')
    ap.add_argument('--species', nargs='*', default=[
        'Schizophyllum_commune',
        'Enoki_fungi_Flammulina_velutipes',
        'Ghost_Fungi_Omphalotus_nidiformis',
        'Cordyceps_militari',
    ])
    ap.add_argument('--out_suffix', default='replot')
    args = ap.parse_args()

    for sp in args.species:
        sp_dir = os.path.join(args.results_root, sp)
        run_dir = latest_run_dir(sp_dir)
        if not run_dir:
            print(f"[SKIP] No runs: {sp}")
            continue
        csv_path = os.path.join(run_dir, 'tau_band_timeseries.csv')
        if not os.path.isfile(csv_path):
            print(f"[SKIP] Missing CSV: {csv_path}")
            continue
        try:
            time_s, taus, Z = read_tau_csv(csv_path)
        except Exception as e:
            print(f"[ERR] {sp}: {e}")
            continue
        # Heatmap
        heat_path = os.path.join(run_dir, f'tau_band_power_heatmap_{args.out_suffix}.png')
        plot_heatmap(
            Z=Z,
            x=time_s,
            y=taus,
            title=f"{sp.replace('_',' ')} — τ-power heatmap (replot)",
            xlabel='time (s)',
            ylabel='τ (√s)',
            out_path=heat_path,
            cmap='magma',
            aspect='auto',
            dpi=140,
        )
        # Surface
        sur_path = os.path.join(run_dir, f'tau_band_power_surface_{args.out_suffix}.png')
        plot_surface3d(
            Z=Z,
            x=time_s,
            y=taus,
            title=f"{sp.replace('_',' ')} — τ-power surface (replot)",
            xlabel='time (s)',
            ylabel='τ (√s)',
            zlabel='power (arb)',
            out_path=sur_path,
            stride=4,
            dpi=140,
        )
        print(f"[OK] {sp}: {heat_path} | {sur_path}")


if __name__ == '__main__':
    main()


