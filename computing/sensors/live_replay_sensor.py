#!/usr/bin/env python3
import os, time, csv, json, argparse, urllib.request
from datetime import datetime, timezone
import numpy as np

FAST_TAU = 5.5


def load_tau_csv(path: str, tau: float):
    times, vals = [], []
    # Skip commented metadata lines that start with '#'
    with open(path, 'r') as f:
        lines = [ln for ln in f if not ln.lstrip().startswith('#')]
    if not lines:
        return np.array([]), np.array([])
    rdr = csv.DictReader(lines)
    key = f'tau_{tau:g}'
    for row in rdr:
        try:
            t = float(row.get('time_s', 'nan'))
            v = float(row.get(key, 'nan'))
            if np.isfinite(t) and np.isfinite(v):
                times.append(t)
                vals.append(v)
        except Exception:
            continue
    return np.asarray(times, dtype=float), np.asarray(vals, dtype=float)


def poll_open_meteo_rh(lat: float, lon: float):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=relative_humidity_2m&timezone=UTC"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode('utf-8'))
        rh = float(data.get('current', {}).get('relative_humidity_2m'))
        return rh, url
    except Exception:
        return None, url


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
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


def main():
    ap = argparse.ArgumentParser(description='Live replay sensor: τ CSV at 1× + Open-Meteo RH')
    ap.add_argument('--species', default='Schizophyllum_commune')
    ap.add_argument('--tau', type=float, default=FAST_TAU)
    ap.add_argument('--lat', type=float, default=51.5074)
    ap.add_argument('--lon', type=float, default=-0.1278)
    ap.add_argument('--pre_s', type=float, default=600.0)
    ap.add_argument('--post_s', type=float, default=1200.0)
    ap.add_argument('--d_thresh', type=float, default=0.5)
    ap.add_argument('--p_thresh', type=float, default=0.05)
    ap.add_argument('--out_dir', default='results/live_replay')
    args = ap.parse_args()

    # Pick latest τ CSV
    sp_root = os.path.join('results', 'zenodo', args.species)
    runs = sorted([os.path.join(sp_root, d) for d in os.listdir(sp_root)])
    if not runs:
        raise SystemExit('No runs for species')
    # pick the most recent run that actually has the τ CSV
    run_dir = None
    tau_csv = None
    for cand in reversed(runs):
        p = os.path.join(cand, 'tau_band_timeseries.csv')
        if os.path.isfile(p):
            run_dir = cand
            tau_csv = p
            break
    if run_dir is None or tau_csv is None:
        raise SystemExit('No tau_band_timeseries.csv found for species')
    times, vals = load_tau_csv(tau_csv, args.tau)
    if times.size == 0:
        raise SystemExit('Empty tau series')

    os.makedirs(args.out_dir, exist_ok=True)
    events = []
    t0_wall = time.time()
    t_start = float(times[0])

    # Poll RH initially
    rh_last, rh_url = poll_open_meteo_rh(args.lat, args.lon)
    rh_hist = []

    i = 0
    while i < len(times):
        # Wall-time pacing at 1×
        t_target = t0_wall + (times[i] - t_start)
        time.sleep(max(0.0, t_target - time.time()))
        # Update RH every 5 minutes
        if (not rh_hist) or ((len(rh_hist) > 0) and (datetime.now(timezone.utc).minute % 5 == 0)):
            rh_now, rh_url = poll_open_meteo_rh(args.lat, args.lon)
            if rh_now is not None:
                rh_hist.append((time.time(), rh_now))
                rh_last = rh_now
        # Sliding window gate when enough history exists
        t_now = times[i]
        pre_mask = (times >= (t_now - args.pre_s)) & (times < t_now)
        post_mask = (times >= t_now) & (times <= (t_now + args.post_s))
        a = vals[pre_mask]
        b = vals[post_mask]
        if a.size >= 30 and b.size >= 30:
            d = cohens_d(a, b)
            p_mwu = mwu_p(a, b)
            p_perm = perm_p_value(a, b, iters=200)
            # RH stability guard
            rh_vals = [v for (_, v) in rh_hist[-6:]]  # last ~30 min if 5-min polls
            d_rh = (rh_vals[-1] - rh_vals[0]) if len(rh_vals) >= 2 else 0.0
            stable = abs(d_rh) <= 5.0  # within 5% RH
            triggered = (abs(d) >= args.d_thresh) and (p_mwu < args.p_thresh) and (p_perm < args.p_thresh) and stable
            if triggered:
                evt = {
                    'created_by': 'joe knowles',
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'species': args.species,
                    'run_dir': run_dir,
                    't_center_s': float(t_now),
                    'gate': {'d': float(d), 'mwu_p': float(p_mwu), 'perm_p': float(p_perm)},
                    'rh': {'current': rh_last, 'delta_approx_30m': d_rh, 'source_url': rh_url},
                    'params': {'pre_s': args.pre_s, 'post_s': args.post_s, 'tau': args.tau},
                }
                events.append(evt)
                print(json.dumps(evt), flush=True)
        i += 1

    # Write summary
    out_path = os.path.join(args.out_dir, f"{args.species}_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json")
    with open(out_path, 'w') as f:
        json.dump({'events': events}, f, indent=2)
    print(json.dumps({'summary': out_path, 'n_events': len(events)}))


if __name__ == '__main__':
    main()
