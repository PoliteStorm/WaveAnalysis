#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from viz.plotting import plot_tau_trends_ci


SPECIES = [
    'Schizophyllum_commune',
    'Enoki_fungi_Flammulina_velutipes',
    'Ghost_Fungi_Omphalotus_nidiformis',
    'Cordyceps_militari',
]


def latest_dir(pattern: str) -> str | None:
    items = sorted(glob.glob(pattern))
    return items[-1] if items else None


def load_tau_ci(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def unify_taus(ci_dicts):
    # Collect all taus (focus on raw values, not _norm)
    all_taus = set()
    for d in ci_dicts:
        for t in d['taus']:
            t_str = str(t)
            if '_norm' not in t_str:
                try:
                    all_taus.add(float(t_str.replace('tau_', '')))
                except Exception:
                    pass
    taus_sorted = sorted(all_taus)

    # Reindex each dict to unified tau order (raw values only)
    out = []
    for d in ci_dicts:
        tau_to_idx = {}
        for i, t in enumerate(d['taus']):
            t_str = str(t)
            if '_norm' not in t_str:
                try:
                    tau_val = float(t_str.replace('tau_', ''))
                    tau_to_idx[tau_val] = i
                except Exception:
                    pass

        mean = np.zeros(len(taus_sorted), dtype=float)
        lo = np.zeros(len(taus_sorted), dtype=float)
        hi = np.zeros(len(taus_sorted), dtype=float)
        for j, t in enumerate(taus_sorted):
            if t in tau_to_idx:
                i = tau_to_idx[t]
                mean[j] = float(d['mean'][i])
                lo[j] = float(d['lo'][i])
                hi[j] = float(d['hi'][i])
            else:
                mean[j] = np.nan
                lo[j] = np.nan
                hi[j] = np.nan
        out.append({'mean': mean, 'lo': lo, 'hi': hi})
    return np.array(taus_sorted, dtype=float), out


def main():
    ap = argparse.ArgumentParser(description='Make multi-species τ-trend figure with CI shading')
    ap.add_argument('--ci_root', default='results/ci_summaries')
    ap.add_argument('--out_root', default='results/summaries')
    args = ap.parse_args()

    ci_dicts = []
    have = []
    for sp in SPECIES:
        d = latest_dir(os.path.join(args.ci_root, sp, '*'))
        if not d:
            continue
        p = os.path.join(d, 'tau_power_ci.json')
        if os.path.isfile(p):
            ci = load_tau_ci(p)
            ci_dicts.append(ci)
            have.append(sp)

    if not ci_dicts:
        print('[SKIP] no CI summaries found')
        return

    taus, aligned = unify_taus(ci_dicts)
    ts = datetime.now().isoformat(timespec='seconds')
    out_dir = os.path.join(args.out_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    # Create one panel per species stacked vertically
    for sp, d in zip(have, aligned):
        n_time = len(d['mean'])  # here mean is per-τ; we plot a simple bar-like line
        # For display, treat taus as x and values as curves of mean with bands
        time_s = taus  # abuse API: x-axis holds τ
        means = d['mean'][:, None]
        lo = d['lo'][:, None]
        hi = d['hi'][:, None]
        path = os.path.join(out_dir, f'{sp}_tau_trend_multispecies.png')
        plot_tau_trends_ci(
            time_s=time_s,
            taus=np.array([1.0]),  # single curve; labels suppressed
            means=means,
            lo=lo,
            hi=hi,
            title=f'{sp.replace("_"," ")} — τ power (mean±95% CI) vs τ',
            out_path=path,
        )

    index = {
        'created_by': 'joe knowles',
        'timestamp': ts,
        'species': have,
        'taus': taus.tolist(),
        'figs': [f'{sp}_tau_trend_multispecies.png' for sp in have],
    }
    with open(os.path.join(out_dir, 'multi_species_tau_trends.json'), 'w') as f:
        json.dump(index, f, indent=2)
    print(f'[OK] Wrote multi-species τ trend figures to {out_dir}')


if __name__ == '__main__':
    main()



