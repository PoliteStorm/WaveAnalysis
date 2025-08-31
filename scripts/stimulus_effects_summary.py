#!/usr/bin/env python3
"""
Summarize stimulus-response effect sizes per stimulus type using existing
validation in analyze_metrics.py.

Inputs:
- --file: Zenodo TXT electrophysiology file
- --stimulus_csv: CSV with columns time_s, stimulus_type
- --pre_s, --post_s: window sizes in seconds

Output:
- JSON summary at results/stimulus_effects/<timestamp>/summary.json
"""

import os
import json
import argparse
from datetime import datetime as _dt
from typing import Dict, List
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyze_metrics import validate_stimulus_response  # type: ignore
from prove_transform import load_zenodo_timeseries  # type: ignore


def load_stimulus_csv(path: str) -> List[Dict]:
    import csv
    rows: List[Dict] = []
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = float(row.get('time_s', ''))
            except Exception:
                continue
            stim = (row.get('stimulus_type') or 'unknown').strip()
            rows.append({'time_s': t, 'stimulus_type': stim})
    return rows


def summarize_effects(file_path: str, stim_csv: str, pre_s: float, post_s: float,
                      channel: str = '') -> Dict:
    t, channels = load_zenodo_timeseries(file_path)
    pick = channel if channel in channels else None
    if pick is None:
        for name, vec in channels.items():
            if np.isfinite(vec).any():
                pick = name
                break
    if pick is None:
        raise RuntimeError('No valid channel found')
    V = channels[pick]
    stims = load_stimulus_csv(stim_csv)
    stim_times = [s['time_s'] for s in stims]

    res = validate_stimulus_response(V, stim_times, pre_window=pre_s, post_window=post_s, fs_hz=1.0)

    # Aggregate by stimulus_type by nearest match time
    by_type: Dict[str, List[Dict]] = {}
    for s in stims:
        # find closest response entry
        best = None
        best_dt = 1e12
        for r in res:
            dt = abs((r.get('stimulus_time_s', 0.0)) - s['time_s'])
            if dt < best_dt:
                best_dt = dt
                best = r
        if best is None:
            continue
        by_type.setdefault(s['stimulus_type'], []).append(best)

    summary: Dict[str, Dict] = {}
    for k, arr in by_type.items():
        d_vals = [x.get('effect_size', {}).get('cohens_d', np.nan) for x in arr]
        p_vals = [x.get('t_test', {}).get('p_value', np.nan) for x in arr]
        u_p_vals = [x.get('mann_whitney', {}).get('p_value', np.nan) for x in arr]
        summary[k] = {
            'n': int(len(arr)),
            'mean_d': float(np.nanmean(d_vals)) if len(d_vals) else float('nan'),
            'median_d': float(np.nanmedian(d_vals)) if len(d_vals) else float('nan'),
            'median_p_ttest': float(np.nanmedian(p_vals)) if len(p_vals) else float('nan'),
            'median_p_mw': float(np.nanmedian(u_p_vals)) if len(u_p_vals) else float('nan')
        }

    return {
        'file': file_path,
        'channel': pick,
        'pre_s': float(pre_s),
        'post_s': float(post_s),
        'summary_by_stimulus': summary,
        'responses': res
    }


def main():
    ap = argparse.ArgumentParser(description='Stimulus effect-size summarizer')
    ap.add_argument('--file', required=True, help='Zenodo TXT file')
    ap.add_argument('--stimulus_csv', required=True, help='CSV with time_s, stimulus_type')
    ap.add_argument('--pre_s', type=float, default=300.0)
    ap.add_argument('--post_s', type=float, default=600.0)
    ap.add_argument('--channel', default='')
    ap.add_argument('--out_root', default='results/stimulus_effects')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    ts = _dt.now().isoformat(timespec='seconds').replace(':', '-')
    out_dir = os.path.join(args.out_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    try:
        out = summarize_effects(args.file, args.stimulus_csv, args.pre_s, args.post_s, args.channel)
    except Exception as e:
        out = {'error': str(e)}

    out_path = os.path.join(out_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(out_path)


if __name__ == '__main__':
    main()


