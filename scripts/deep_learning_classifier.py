#!/usr/bin/env python3
"""
Deep Learning Classifier (Quick) for Species/Pattern Classification

Scans results/zenodo/*/*/metrics.json, builds a feature table, trains a small MLP,
and writes a concise report and confusion matrix.
"""

import os
import json
import glob
import datetime as _dt
from typing import Dict, List, Tuple

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RESULTS_ROOT = 'results/zenodo'


def find_metrics_files(root: str) -> List[Tuple[str, str]]:
    paths: List[Tuple[str, str]] = []
    for sp_dir in glob.glob(os.path.join(root, '*')):
        if not os.path.isdir(sp_dir) or os.path.basename(sp_dir).startswith('_'):
            continue
        species = os.path.basename(sp_dir)
        runs = glob.glob(os.path.join(sp_dir, '*'))
        for run_dir in runs:
            if not os.path.isdir(run_dir):
                continue
            metrics_path = os.path.join(run_dir, 'metrics.json')
            if os.path.isfile(metrics_path):
                paths.append((species, metrics_path))
    return paths


def safe_get(d: Dict, path: List[str], default: float = 0.0) -> float:
    cur = d
    try:
        for p in path:
            cur = cur[p]
        v = float(cur)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def feature_vector(metrics: Dict) -> Dict[str, float]:
    f: Dict[str, float] = {}
    # Amplitude stats
    f['amp_mean'] = safe_get(metrics, ['amplitude_stats', 'mean'])
    f['amp_std'] = safe_get(metrics, ['amplitude_stats', 'std'])
    f['amp_skew'] = safe_get(metrics, ['amplitude_stats', 'skewness'])
    f['amp_kurt'] = safe_get(metrics, ['amplitude_stats', 'kurtosis_excess'])
    f['amp_H'] = safe_get(metrics, ['amplitude_stats', 'shannon_entropy_bits'])

    # Duration stats
    f['dur_mean'] = safe_get(metrics, ['duration_stats', 'mean'])
    f['dur_std'] = safe_get(metrics, ['duration_stats', 'std'])

    # ISI stats
    f['isi_mean'] = safe_get(metrics, ['isi_stats', 'mean'])
    f['isi_std'] = safe_get(metrics, ['isi_stats', 'std'])

    # Band fractions (robustly collect present taus)
    band_fracs = metrics.get('band_fractions', {}) or {}
    for tau_key in band_fracs.keys():
        try:
            tau_float = float(tau_key)
        except Exception:
            continue
        f[f'tau_frac_{tau_float:g}'] = float(band_fracs[tau_key])

    # Spike train metrics
    f['victor_distance'] = safe_get(metrics, ['spike_train_metrics', 'victor_distance'])
    f['lv'] = safe_get(metrics, ['spike_train_metrics', 'local_variation'])
    f['cv2'] = safe_get(metrics, ['spike_train_metrics', 'cv_squared'])
    f['fano'] = safe_get(metrics, ['spike_train_metrics', 'fano_factor'])
    f['burst_index'] = safe_get(metrics, ['spike_train_metrics', 'burst_index'])

    # Multiscale entropy
    f['mse_mean'] = safe_get(metrics, ['multiscale_entropy', 'mean_mse'])
    f['complexity_index'] = safe_get(metrics, ['multiscale_entropy', 'complexity_index'])

    # Spike count
    f['spike_count'] = float(metrics.get('spike_count', 0.0))

    return f


def build_dataset(paths: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_rows: List[List[float]] = []
    y: List[str] = []
    feature_names_set: set = set()

    # First pass: collect all feature names
    cache: List[Tuple[str, Dict[str, float]]] = []
    for species, mpath in paths:
        try:
            with open(mpath, 'r') as f:
                metrics = json.load(f)
            fv = feature_vector(metrics)
            cache.append((species, fv))
            feature_names_set.update(fv.keys())
        except Exception:
            continue

    feature_names = sorted(list(feature_names_set))

    # Second pass: assemble rows in consistent order
    for species, fv in cache:
        row = [float(fv.get(name, 0.0)) for name in feature_names]
        if not any(np.isfinite(row)):
            continue
        X_rows.append(row)
        y.append(species)

    X = np.array(X_rows, dtype=float)
    y_arr = np.array(y, dtype=str)
    return X, y_arr, feature_names


def train_quick_mlp(X: np.ndarray, y: np.ndarray) -> Dict:
    if len(np.unique(y)) < 2 or len(y) < 10:
        return {
            'ok': False,
            'reason': 'insufficient_classes_or_samples',
            'n_samples': int(len(y)),
            'n_classes': int(len(np.unique(y)))
        }

    # Base pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam',
                              alpha=1e-3, batch_size=32, learning_rate_init=1e-3,
                              max_iter=400, random_state=42))
    ])

    # Determine if we have enough samples per class for StratifiedKFold
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts)) if counts.size > 0 else 0

    if min_count < 2 or len(y) < 10:
        # Not enough data for CV: fit on all data and report training accuracy only
        pipeline.fit(X, y)
        yhat = pipeline.predict(X)
        acc_train = accuracy_score(y, yhat)
        labels = sorted(list(np.unique(y)))
        cm = confusion_matrix(y, yhat, labels=labels).tolist()
        cls_rep = classification_report(y, yhat, labels=labels, output_dict=True)
        return {
            'ok': True,
            'mode': 'fit_only',
            'reason': 'insufficient_class_counts_for_cv',
            'n_samples': int(len(y)),
            'n_classes': int(len(np.unique(y))),
            'train_accuracy': float(acc_train),
            'labels': labels,
            'confusion_matrix': cm,
            'classification_report': cls_rep,
        }

    n_splits = min(5, min_count)
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs: List[float] = []
    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        pipeline.fit(Xtr, ytr)
        yhat = pipeline.predict(Xte)
        accs.append(accuracy_score(yte, yhat))
        y_true_all.extend(yte.tolist())
        y_pred_all.extend(yhat.tolist())

    acc_mean = float(np.mean(accs)) if accs else 0.0
    labels = sorted(list(np.unique(y)))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels).tolist()
    cls_rep = classification_report(y_true_all, y_pred_all, labels=labels, output_dict=True)

    return {
        'ok': True,
        'mode': 'cv',
        'n_samples': int(len(y)),
        'n_classes': int(len(np.unique(y))),
        'cv_accuracy_mean': acc_mean,
        'labels': labels,
        'confusion_matrix': cm,
        'classification_report': cls_rep,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Quick deep learning classifier for species')
    ap.add_argument('--root', default=RESULTS_ROOT)
    ap.add_argument('--out_root', default='results/ml')
    args = ap.parse_args()

    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    out_dir = os.path.join(args.out_root, f'{ts}_quick_mlp')
    os.makedirs(out_dir, exist_ok=True)

    paths = find_metrics_files(args.root)
    X, y, feature_names = build_dataset(paths)

    report = {
        'created_by': 'joe knowles',
        'timestamp': ts,
        'n_samples': int(len(y)),
        'n_features': int(X.shape[1] if X.ndim == 2 else 0),
        'feature_names': feature_names,
    }

    results = train_quick_mlp(X, y)
    report['training'] = results

    out_json = os.path.join(out_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Minimal markdown summary
    out_md = os.path.join(out_dir, 'results.md')
    with open(out_md, 'w') as f:
        f.write('# Quick MLP Classification Report\n\n')
        f.write(f"Created by joe knowles — {ts}\n\n")
        f.write(f"Samples: {report['n_samples']}, Features: {report['n_features']}\n\n")
        if results.get('ok'):
            f.write(f"CV Accuracy (mean): {results['cv_accuracy_mean']:.3f}\n\n")
            f.write('Labels: ' + ', '.join(results['labels']) + '\n\n')
        else:
            f.write(f"Training skipped: {results.get('reason')} (samples={results.get('n_samples')}, classes={results.get('n_classes')})\n")

    print(out_json)


if __name__ == '__main__':
    main()


