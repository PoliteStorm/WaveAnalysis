#!/usr/bin/env python3
"""
Stimulus MI and Decoder

Builds simple pre/post features around stimulus times and evaluates:
- Mutual information (MI) between features and stimulus labels with permutation p-value
- Decoder accuracy (LogisticRegression, fallback RandomForest) with stratified CV
- Permutation baseline for decoder accuracy

Outputs JSON summary under results/stimulus_mi_decoder/<timestamp>/summary.json
"""

import os
import json
import argparse
from datetime import datetime as _dt
from typing import Dict, List, Tuple
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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


def segment_indices(center_s: float, pre_s: float, post_s: float, fs_hz: float, n: int) -> Tuple[slice, slice]:
    c = int(round(center_s * fs_hz))
    pre_len = int(round(pre_s * fs_hz))
    post_len = int(round(post_s * fs_hz))
    pre_lo = max(0, c - pre_len)
    pre_hi = c
    post_lo = c
    post_hi = min(n, c + post_len)
    return slice(pre_lo, pre_hi), slice(post_lo, post_hi)


def basic_stats(x: np.ndarray) -> Dict:
    if x.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'rms': 0.0, 'abs_mean': 0.0, 'slope': 0.0}
    t = np.arange(x.size, dtype=float)
    # slope via least squares
    denom = (np.sum((t - t.mean()) ** 2) + 1e-12)
    slope = float(np.sum((t - t.mean()) * (x - x.mean())) / denom)
    return {
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        'median': float(np.median(x)),
        'rms': float(np.sqrt(np.mean(x * x))),
        'abs_mean': float(np.mean(np.abs(x))),
        'slope': slope,
    }


def build_features(V: np.ndarray, stim_times: List[float], pre_s: float, post_s: float, fs_hz: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    n = V.size
    X = []
    y: List[str] = []
    for s in stim_times:
        pre_idx, post_idx = segment_indices(s, pre_s, post_s, fs_hz, n)
        pre = V[pre_idx]
        post = V[post_idx]
        a = basic_stats(pre)
        b = basic_stats(post)
        feat = [
            b['mean'] - a['mean'],
            b['std'] - a['std'],
            b['median'] - a['median'],
            b['rms'] - a['rms'],
            b['abs_mean'] - a['abs_mean'],
            b['slope'] - a['slope'],
            # context features
            a['mean'], a['std'], b['mean'], b['std']
        ]
        X.append(feat)
        y.append('')  # placeholder; caller assigns labels
    return np.array(X, dtype=float), y


def sample_nonstim_times(n_total: int, stim_times: List[float], pre_s: float, post_s: float,
                         num_needed: int, fs_hz: float = 1.0, seed: int = 42) -> List[float]:
    rng = np.random.default_rng(seed)
    T = n_total / max(1e-9, fs_hz)
    protected = []
    for s in stim_times:
        protected.append((max(0.0, s - pre_s), min(T, s + post_s)))
    times = []
    tries = 0
    while len(times) < num_needed and tries < 10000:
        t = float(rng.uniform(pre_s, max(pre_s + 1.0, T - post_s)))
        if all(not (a <= t <= b) for (a, b) in protected):
            times.append(t)
        tries += 1
    return times


def mutual_information(X: np.ndarray, y: List[str], n_perm: int = 200, seed: int = 42) -> Dict:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import mutual_info_classif
    le = LabelEncoder()
    if X.shape[0] < 2 or len(set(y)) < 2:
        return {'mi_per_feature': [], 'mi_mean': 0.0, 'perm_mean_null': 0.0, 'perm_p': 1.0, 'classes': []}
    y_enc = le.fit_transform(y)
    mi = mutual_info_classif(X, y_enc, discrete_features=False, random_state=seed)
    mi_mean = float(np.mean(mi))
    # permutation baseline
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_enc)
        mi_p = mutual_info_classif(X, y_perm, discrete_features=False)
        null.append(np.mean(mi_p))
    null = np.array(null, dtype=float)
    p = float((np.sum(null >= mi_mean) + 1) / (len(null) + 1))
    return {
        'mi_per_feature': mi.tolist(),
        'mi_mean': mi_mean,
        'perm_mean_null': float(np.mean(null)),
        'perm_p': p,
        'classes': le.classes_.tolist()
    }


def decoder_accuracy(X: np.ndarray, y: List[str], n_splits: int = 5, n_perm: int = 200, seed: int = 42) -> Dict:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    le = LabelEncoder()
    if X.shape[0] < 2 or len(set(y)) < 2:
        return {'cv_accuracy': 0.0, 'cv_std': 0.0, 'perm_mean_null': 0.0, 'perm_p': 1.0, 'classes': []}
    y_enc = le.fit_transform(y)

    # Try logistic regression first; fallback to RF if class separability is odd
    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                         LogisticRegression(max_iter=200, n_jobs=None))
    min_class = int(min(np.bincount(y_enc)))
    splits = max(2, min(n_splits, len(np.unique(y_enc)), min_class))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    try:
        scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring='accuracy')
        acc = float(np.mean(scores))
    except Exception:
        rf = RandomForestClassifier(n_estimators=200, random_state=seed)
        scores = cross_val_score(rf, X, y_enc, cv=cv, scoring='accuracy')
        acc = float(np.mean(scores))

    # permutation baseline
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_enc)
        try:
            s = cross_val_score(pipe, X, y_perm, cv=cv, scoring='accuracy')
            null.append(np.mean(s))
        except Exception:
            null.append(1.0 / max(1, len(np.unique(y_enc))))
    null = np.array(null, dtype=float)
    p = float((np.sum(null >= acc) + 1) / (len(null) + 1))
    return {
        'cv_accuracy': acc,
        'cv_std': float(np.std(scores)),
        'perm_mean_null': float(np.mean(null)),
        'perm_p': p,
        'classes': le.classes_.tolist()
    }


def main():
    ap = argparse.ArgumentParser(description='Stimulus MI and Decoder')
    ap.add_argument('--file', required=True)
    ap.add_argument('--stimulus_csv', required=True)
    ap.add_argument('--pre_s', type=float, default=300.0)
    ap.add_argument('--post_s', type=float, default=600.0)
    ap.add_argument('--channel', default='')
    ap.add_argument('--perm', type=int, default=200)
    ap.add_argument('--out_root', default='results/stimulus_mi_decoder')
    args = ap.parse_args()

    t, channels = load_zenodo_timeseries(args.file)
    pick = args.channel if args.channel in channels else None
    if pick is None:
        for name, vec in channels.items():
            if np.isfinite(vec).any():
                pick = name
                break
    if pick is None:
        raise SystemExit('No valid channel found')
    V = np.asarray(channels[pick], dtype=float)
    stims = load_stimulus_csv(args.stimulus_csv)
    # Filter to valid in-range stimuli
    T = len(V) / 1.0
    stims = [s for s in stims if (args.pre_s <= s['time_s'] <= max(args.pre_s + 1.0, T - args.post_s))]
    synthetic = False
    if not stims:
        # Auto-generate synthetic stimuli evenly spaced to illustrate method
        synthetic = True
        K = 6
        if T <= (args.pre_s + args.post_s + 1.0):
            raise SystemExit('Signal too short for any pre/post windows')
        times = np.linspace(args.pre_s + 1.0, T - args.post_s - 1.0, K)
        labels = ['moisture', 'mechanical', 'temperature', 'light', 'chemical', 'touch']
        stims = [{'time_s': float(t), 'stimulus_type': labels[i % len(labels)]} for i, t in enumerate(times)]
    stim_times = [s['time_s'] for s in stims]
    # Build positive (stimulus) samples
    X_pos, _ = build_features(V, stim_times, args.pre_s, args.post_s, fs_hz=1.0)
    y_pos = [s['stimulus_type'] for s in stims]
    # Build negative (non-stimulus) samples for binary sanity checks
    non_times = sample_nonstim_times(len(V), stim_times, args.pre_s, args.post_s, num_needed=len(stim_times), fs_hz=1.0)
    X_neg, _ = build_features(V, non_times, args.pre_s, args.post_s, fs_hz=1.0)
    y_neg = ['none'] * X_neg.shape[0]

    # Combine multi-class (using only positives) and binary (pos vs none)
    X_multi = X_pos
    y_multi = y_pos
    X_bin = np.vstack([X_pos, X_neg]) if X_neg.size else X_pos
    y_bin = y_pos + y_neg

    mi_multi = mutual_information(X_multi, y_multi, n_perm=min(100, args.perm))
    dec_multi = decoder_accuracy(X_multi, y_multi, n_splits=3, n_perm=min(100, args.perm))
    mi_bin = mutual_information(X_bin, y_bin, n_perm=min(100, args.perm))
    dec_bin = decoder_accuracy(X_bin, y_bin, n_splits=3, n_perm=min(100, args.perm))

    ts = _dt.now().isoformat(timespec='seconds').replace(':', '-')
    out_dir = os.path.join(args.out_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump({
            'file': args.file,
            'channel': pick,
            'stimulus_csv': args.stimulus_csv,
            'pre_s': args.pre_s,
            'post_s': args.post_s,
            'synthetic': synthetic,
            'n_samples_multi': int(X_multi.shape[0]),
            'n_samples_bin': int(X_bin.shape[0]),
            'features_per_sample': int(X_multi.shape[1] if X_multi.size else 0),
            'mi_multi': mi_multi,
            'decoder_multi': dec_multi,
            'mi_bin': mi_bin,
            'decoder_bin': dec_bin
        }, f, indent=2)
    print(out_path)


if __name__ == '__main__':
    main()


