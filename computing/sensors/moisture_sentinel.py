#!/usr/bin/env python3
import os
import json
import csv
import argparse
import numpy as np
from datetime import datetime

DEFAULT_FAST_TAU = 5.5


def find_latest_run(species_root: str) -> str | None:
    runs = sorted([os.path.join(species_root, d) for d in os.listdir(species_root)], key=str)
    return runs[-1] if runs else None


def load_tau_timeseries(run_dir: str, tau: float) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(run_dir, 'tau_band_timeseries.csv')
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    times = []
    fast_vals = []
    with open(path, 'r') as f:
        rdr = csv.DictReader(f)
        key = f'tau_{tau:g}'
        for row in rdr:
            try:
                t = float(row.get('time_s', 'nan'))
                v = float(row.get(key, 'nan'))
                if np.isfinite(t) and np.isfinite(v):
                    times.append(t)
                    fast_vals.append(v)
            except Exception:
                continue
    return np.asarray(times, dtype=float), np.asarray(fast_vals, dtype=float)


def window_indices(times: np.ndarray, center: float, pre_s: float, post_s: float) -> tuple[np.ndarray, np.ndarray]:
    pre_mask = (times >= (center - pre_s)) & (times < center)
    post_mask = (times >= center) & (times <= (center + post_s))
    return pre_mask, post_mask


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a), np.var(b)
    pooled = np.sqrt((s1 + s2) / 2.0) if (s1 + s2) > 0 else 0.0
    return float((m2 - m1) / pooled) if pooled > 0 else 0.0


def mwu_p(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(a, b, alternative='two-sided')
        return float(p)
    except Exception:
        return float('nan')


def perm_p_value(a: np.ndarray, b: np.ndarray, iters: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(cohens_d(a, b))
    concat = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(iters):
        perm = rng.permutation(concat)
        ap = perm[:n_a]
        bp = perm[n_a:]
        if abs(cohens_d(ap, bp)) >= observed:
            count += 1
    return (count + 1) / (iters + 1)


def detect_auto_events(times: np.ndarray, vals: np.ndarray, pre_s: float, post_s: float, min_gap_s: float, min_peak_z: float) -> list[float]:
    # Simple global z-score and local maxima detection with coverage checks
    if times.size < 5:
        return []
    v = vals.astype(float)
    mu = float(np.mean(v))
    sd = float(np.std(v)) if np.std(v) > 0 else 1.0
    z = (v - mu) / sd
    events = []
    last_t = -1e12
    for i in range(1, len(v) - 1):
        if z[i] >= min_peak_z and v[i] > v[i - 1] and v[i] >= v[i + 1]:
            t0 = float(times[i])
            if (t0 - last_t) < min_gap_s:
                continue
            # ensure coverage exists for pre/post windows
            if (times[0] <= (t0 - pre_s)) and (times[-1] >= (t0 + post_s)):
                events.append(t0)
                last_t = t0
    return events


def run_sentinel(species: str, pre_s: float, post_s: float, tau: float, events_csv: str | None,
                 auto_events: bool, min_gap_s: float, min_peak_z: float,
                 d_thresh: float = 0.5, p_thresh: float = 0.05) -> dict:
    base = os.path.join('results', 'zenodo', species)
    run_dir = find_latest_run(base)
    if not run_dir:
        return {'error': 'no_run', 'species': species}
    times, vals = load_tau_timeseries(run_dir, tau)
    if times.size == 0:
        return {'error': 'no_tau_csv', 'run_dir': run_dir}
    # Load events
    events = []
    if events_csv and os.path.isfile(events_csv):
        with open(events_csv, 'r') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    if (row.get('stimulus_type','') or '').lower() in ('moisture','water'):
                        events.append(float(row.get('time_s')))
                except Exception:
                    continue
    if (not events) and auto_events:
        events = detect_auto_events(times, vals, pre_s, post_s, min_gap_s=min_gap_s, min_peak_z=min_peak_z)
    # Evaluate
    reports = []
    n_trigger = 0
    for t0 in events:
        pre_mask, post_mask = window_indices(times, t0, pre_s, post_s)
        a = vals[pre_mask]
        b = vals[post_mask]
        if a.size < 5 or b.size < 5:
            continue
        d = cohens_d(a, b)
        p_mwu = mwu_p(a, b)
        p_perm = perm_p_value(a, b, iters=200)
        triggered = (abs(d) >= d_thresh) and (p_mwu < p_thresh) and (p_perm < p_thresh)
        if triggered:
            n_trigger += 1
        reports.append({
            'event_time_s': float(t0),
            'n_pre': int(a.size),
            'n_post': int(b.size),
            'cohens_d': float(d),
            'mwu_p': float(p_mwu),
            'perm_p': float(p_perm),
            'pre_mean': float(np.mean(a)),
            'post_mean': float(np.mean(b)),
            'triggered': bool(triggered),
        })
    return {
        'created_by': 'joe knowles',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'species': species,
        'run_dir': run_dir,
        'tau_fast': float(tau),
        'pre_s': float(pre_s),
        'post_s': float(post_s),
        'auto_events': bool(auto_events),
        'min_gap_s': float(min_gap_s),
        'min_peak_z': float(min_peak_z),
        'gate': {'d_thresh': float(d_thresh), 'p_thresh': float(p_thresh)},
        'events_evaluated': int(len(reports)),
        'events_triggered': int(n_trigger),
        'reports': reports,
    }


def main():
    ap = argparse.ArgumentParser(description='Moisture Sentinel — fast τ-band gate with statistical confirmation')
    ap.add_argument('--species', default='Schizophyllum_commune')
    ap.add_argument('--pre_s', type=float, default=300.0)
    ap.add_argument('--post_s', type=float, default=600.0)
    ap.add_argument('--tau', type=float, default=DEFAULT_FAST_TAU)
    ap.add_argument('--events_csv', default='')
    ap.add_argument('--auto_events', action='store_true', help='Auto-detect candidate moisture events from fast-τ series')
    ap.add_argument('--min_gap_s', type=float, default=1800.0)
    ap.add_argument('--min_peak_z', type=float, default=1.5)
    ap.add_argument('--d_thresh', type=float, default=0.5)
    ap.add_argument('--p_thresh', type=float, default=0.05)
    ap.add_argument('--json_out', default='')
    args = ap.parse_args()

    out = run_sentinel(
        species=args.species,
        pre_s=args.pre_s,
        post_s=args.post_s,
        tau=args.tau,
        events_csv=(args.events_csv or None),
        auto_events=bool(args.auto_events),
        min_gap_s=args.min_gap_s,
        min_peak_z=args.min_peak_z,
        d_thresh=args.d_thresh,
        p_thresh=args.p_thresh,
    )
    js = args.json_out or os.path.join('results', 'sentinels', args.species, datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.json')
    os.makedirs(os.path.dirname(js), exist_ok=True)
    with open(js, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'json': js, 'events_evaluated': out.get('events_evaluated', 0)}))


if __name__ == '__main__':
    main()
