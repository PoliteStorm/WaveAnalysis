#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime
import numpy as np

# Reuse core routines from the sentinel to ensure identical gating
from computing.sensors.moisture_sentinel import (
    find_latest_run,
    load_tau_timeseries,
    window_indices,
    cohens_d,
    mwu_p,
    perm_p_value,
    detect_auto_events,
)


def sample_null_centers(times: np.ndarray, pre_s: float, post_s: float, n_null: int, 
                        exclude: list[float], min_gap_s: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    t_min = float(times[0] + pre_s)
    t_max = float(times[-1] - post_s)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max <= t_min):
        return []
    excl = np.asarray(exclude, dtype=float) if exclude else np.array([], dtype=float)
    centers = []
    for _ in range(n_null * 5):  # oversample with rejection
        t0 = float(rng.uniform(t_min, t_max))
        if excl.size > 0 and np.min(np.abs(excl - t0)) < min_gap_s:
            continue
        centers.append(t0)
        if len(centers) >= n_null:
            break
    return centers


def evaluate_centers(times: np.ndarray, vals: np.ndarray, centers: list[float], pre_s: float, post_s: float,
                     d_thresh: float, p_thresh: float, perm_iters: int) -> tuple[int, int, list[dict]]:
    triggered = 0
    reports: list[dict] = []
    evaluated = 0
    for t0 in centers:
        pre_mask, post_mask = window_indices(times, t0, pre_s, post_s)
        a = vals[pre_mask]
        b = vals[post_mask]
        if a.size < 5 or b.size < 5:
            continue
        evaluated += 1
        d = cohens_d(a, b)
        p_mwu = mwu_p(a, b)
        p_perm = perm_p_value(a, b, iters=perm_iters)
        ok = (abs(d) >= d_thresh) and (p_mwu < p_thresh) and (p_perm < p_thresh)
        if ok:
            triggered += 1
        reports.append({
            't_center_s': float(t0),
            'n_pre': int(a.size),
            'n_post': int(b.size),
            'cohens_d': float(d),
            'mwu_p': float(p_mwu),
            'perm_p': float(p_perm),
            'triggered': bool(ok),
        })
    return triggered, evaluated, reports


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> dict:
    # Simple two-proportion z-test (approximate)
    import math
    if min(n1, n2) == 0:
        return {'z': float('nan'), 'p': float('nan')}
    p1 = x1 / n1
    p2 = x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if p > 0 and p < 1 else float('inf')
    if not np.isfinite(se) or se == 0:
        return {'z': float('nan'), 'p': float('nan')}
    z = (p1 - p2) / se
    try:
        from math import erf, sqrt
        pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    except Exception:
        pval = float('nan')
    return {'z': float(z), 'p': float(pval), 'p1': float(p1), 'p2': float(p2)}


def main():
    ap = argparse.ArgumentParser(description='Sensor viability evaluator: TPR vs null FPR for τ-band gate')
    ap.add_argument('--species', default='Schizophyllum_commune')
    ap.add_argument('--tau', type=float, default=5.5)
    ap.add_argument('--pre_s', type=float, default=300.0)
    ap.add_argument('--post_s', type=float, default=600.0)
    ap.add_argument('--d_thresh', type=float, default=0.5)
    ap.add_argument('--p_thresh', type=float, default=0.05)
    ap.add_argument('--perm_iters', type=int, default=200)
    ap.add_argument('--auto_events', action='store_true')
    ap.add_argument('--min_gap_s', type=float, default=1800.0)
    ap.add_argument('--min_peak_z', type=float, default=1.5)
    ap.add_argument('--n_null', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resample_dt_s', type=float, default=60.0)
    ap.add_argument('--json_out', default='')
    args = ap.parse_args()

    base = os.path.join('results', 'zenodo', args.species)
    # Choose the run with the longest usable τ series to ensure coverage
    run_dir = None
    best_span = -1.0
    times = np.array([])
    vals = np.array([])
    if os.path.isdir(base):
        runs = sorted([os.path.join(base, d) for d in os.listdir(base)], key=str)
        for cand in runs:
            t_c, v_c = load_tau_timeseries(cand, args.tau)
            if t_c.size > 1:
                span = float(t_c[-1] - t_c[0])
                if span > best_span:
                    best_span = span
                    run_dir = cand
                    times, vals = t_c, v_c
    if run_dir is None:
        raise SystemExit('no tau series')

    # Optional uniform resampling to ensure adequate samples per window
    try:
        if args.resample_dt_s and args.resample_dt_s > 0 and times.size > 1:
            t0 = float(times[0])
            t1 = float(times[-1])
            grid = np.arange(t0, t1 + args.resample_dt_s, args.resample_dt_s, dtype=float)
            vals = np.interp(grid, times.astype(float), vals.astype(float))
            times = grid
    except Exception:
        pass

    # True events via auto detection (or fallback to simple regular grid if none)
    events = []
    if args.auto_events:
        events = detect_auto_events(times, vals, args.pre_s, args.post_s, args.min_gap_s, args.min_peak_z)
    if not events:
        # Regular grid every min_gap_s, ensuring coverage
        t = float(times[0] + args.pre_s)
        t_end = float(times[-1] - args.post_s)
        while t <= t_end:
            events.append(t)
            t += args.min_gap_s

    t_trig, t_eval, t_reports = evaluate_centers(
        times, vals, events, args.pre_s, args.post_s, args.d_thresh, args.p_thresh, args.perm_iters
    )

    # Null centers sampled away from events
    null_centers = sample_null_centers(times, args.pre_s, args.post_s, args.n_null, events, args.min_gap_s, args.seed)
    n_trig, n_eval, n_reports = evaluate_centers(
        times, vals, null_centers, args.pre_s, args.post_s, args.d_thresh, args.p_thresh, args.perm_iters
    )

    test = two_proportion_z(t_trig, t_eval, n_trig, n_eval)

    out = {
        'created_by': 'joe knowles',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'species': args.species,
        'run_dir': run_dir,
        'tau_fast': float(args.tau),
        'pre_s': float(args.pre_s),
        'post_s': float(args.post_s),
        'gate': {'d_thresh': float(args.d_thresh), 'p_thresh': float(args.p_thresh), 'perm_iters': int(args.perm_iters)},
        'events_eval': {'count': int(t_eval), 'triggered': int(t_trig), 'rate': (t_trig / t_eval) if t_eval else float('nan')},
        'null_eval': {'count': int(n_eval), 'triggered': int(n_trig), 'rate': (n_trig / n_eval) if n_eval else float('nan')},
        'two_prop_test': test,
        'samples': {
            'events': t_reports[:50],  # cap to keep JSON small
            'null': n_reports[:50],
        }
    }

    js = args.json_out or os.path.join('results', 'sentinels', args.species, 'viability', datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.json')
    os.makedirs(os.path.dirname(js), exist_ok=True)
    with open(js, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'json': js, 'events_rate': out['events_eval']['rate'], 'null_rate': out['null_eval']['rate'], 'p': out['two_prop_test'].get('p')}))


if __name__ == '__main__':
    main()


