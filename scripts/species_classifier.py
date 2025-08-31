#!/usr/bin/env python3
import os
import glob
import json
import datetime as _dt
from typing import Dict, List, Tuple

import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import prove_transform as pt


def clean_label_from_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace(' ', '_')


def basic_signal_features(x: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: 0.0 for k in [
            'mean', 'std', 'median', 'iqr', 'pos_frac', 'zcr', 'absdiff_mean', 'absdiff_std',
            'spec_centroid', 'spec_bw', 'band_lo', 'band_mid', 'band_hi'
        ]}
    x = x - np.mean(x)
    mean = float(np.mean(x))
    std = float(np.std(x))
    median = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(q75 - q25)
    pos_frac = float(np.mean(x > 0))
    # zero-cross rate
    if len(x) > 1:
        zcr = float(np.mean(np.diff(np.signbit(x)) != 0))
    else:
        zcr = 0.0
    # differences
    if len(x) > 1:
        dx = np.diff(x)
        absdiff_mean = float(np.mean(np.abs(dx)))
        absdiff_std = float(np.std(np.abs(dx)))
    else:
        absdiff_mean = 0.0
        absdiff_std = 0.0
    # spectral features
    n = len(x)
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / max(1, n - 1)))
    X = np.fft.rfft(x * win)
    psd = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd_sum = float(np.sum(psd) + 1e-12)
    spec_centroid = float(np.sum(freqs * psd) / psd_sum)
    var = np.sum(((freqs - spec_centroid) ** 2) * psd) / psd_sum
    spec_bw = float(np.sqrt(max(var, 0.0)))
    # band powers (relative)
    def band_power(f0: float, f1: float) -> float:
        mask = (freqs >= f0) & (freqs < f1)
        return float(np.sum(psd[mask]) / psd_sum)
    band_lo = band_power(0.0, 0.01)
    band_mid = band_power(0.01, 0.05)
    band_hi = band_power(0.05, 0.5)
    return {
        'mean': mean, 'std': std, 'median': median, 'iqr': iqr, 'pos_frac': pos_frac,
        'zcr': zcr, 'absdiff_mean': absdiff_mean, 'absdiff_std': absdiff_std,
        'spec_centroid': spec_centroid, 'spec_bw': spec_bw,
        'band_lo': band_lo, 'band_mid': band_mid, 'band_hi': band_hi
    }


def collect_dataset(data_dir: str, segments: int = 8, min_segment_points: int = 2048, random_sample: bool = True, rng_seed: int = 42) -> Tuple[np.ndarray, List[str], List[str]]:
    X = []
    y = []
    paths = []
    rng = np.random.default_rng(rng_seed)
    for path in sorted(glob.glob(os.path.join(data_dir, '*.txt'))):
        try:
            t, channels = pt.load_zenodo_timeseries(path)
            label = clean_label_from_filename(path)
            # pick first valid channel
            vec = None
            for _, v in channels.items():
                if np.isfinite(v).any():
                    vec = np.asarray(v, dtype=np.float32)
                    break
            if vec is None:
                continue
            n = len(vec)
            if n < min_segment_points:
                feats = basic_signal_features(vec, fs=1.0)
                X.append([feats[k] for k in sorted(feats.keys())])
                y.append(label)
                paths.append(path)
            else:
                seg_len = min_segment_points
                if random_sample:
                    max_start = max(1, n - seg_len)
                    starts = rng.integers(0, max_start, size=segments)
                else:
                    hop = max(1, (n - seg_len) // max(1, segments))
                    starts = np.arange(0, min(n - seg_len + 1, segments * hop), hop)
                for idx, start in enumerate(starts[:segments]):
                    s = int(start)
                    e = s + seg_len
                    if e > n:
                        s = max(0, n - seg_len)
                        e = s + seg_len
                    seg = vec[s:e]
                    feats = basic_signal_features(seg, fs=1.0)
                    X.append([feats[k] for k in sorted(feats.keys())])
                    y.append(label)
                    paths.append(f"{path}#seg{idx}")
        except Exception:
            continue
    if not X:
        return np.zeros((0, 1)), [], []
    return np.asarray(X, dtype=np.float32), y, paths


def train_and_eval(X: np.ndarray, y: List[str], seed: int = 42) -> Dict:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    unique, counts = np.unique(y, return_counts=True)
    if len(counts) == 0:
        return {'error': 'no_samples'}
    min_count = int(counts.min())
    if min_count < 2:
        return {'error': 'insufficient_per_class_samples', 'label_counts': {str(k): int(v) for k, v in zip(unique, counts)}}
    n_splits = int(max(2, min(5, min_count)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = sorted(list(set(y)))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    fold_metrics = []
    cms = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = [y[i] for i in train_idx]
        yte = [y[i] for i in test_idx]
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=seed)
        )
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        acc = float(accuracy_score(yte, yp))
        f1w = float(f1_score(yte, yp, average='weighted'))
        cm = confusion_matrix(yte, yp, labels=labels)
        cms.append(cm.tolist())
        fold_metrics.append({'accuracy': acc, 'f1_weighted': f1w})

    accs = [m['accuracy'] for m in fold_metrics]
    f1s = [m['f1_weighted'] for m in fold_metrics]
    return {
        'labels': labels,
        'folds': fold_metrics,
        'confusion_matrices': cms,
        'mean_accuracy': float(np.mean(accs)),
        'mean_f1_weighted': float(np.mean(f1s)),
        'std_accuracy': float(np.std(accs)),
        'std_f1_weighted': float(np.std(f1s)),
    }


def write_outputs(out_dir: str, dataset_paths: List[str], metrics: Dict, feature_names: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    meta = {
        'created_by': 'joe knowles',
        'timestamp': ts,
        'intended_for': 'species_classification',
        'n_samples': len(dataset_paths),
        'feature_names': feature_names,
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'meta': meta, 'metrics': metrics, 'paths': dataset_paths}, f, indent=2)

    # Simple HTML
    html = os.path.join(out_dir, 'summary.html')
    with open(html, 'w') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><title>Species Classifier</title></head><body>\n')
        f.write('<h1>Species Classification (MLP)</h1>\n')
        if 'error' in metrics:
            f.write(f"<p><strong>ERROR:</strong> {metrics['error']}</p>")
            if 'label_counts' in metrics:
                f.write(f"<pre>{json.dumps(metrics['label_counts'], indent=2)}</pre>")
        else:
            f.write(f"<p>Mean accuracy: {metrics['mean_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}<br/>")
            f.write(f"Mean F1 (weighted): {metrics['mean_f1_weighted']:.3f} ± {metrics['std_f1_weighted']:.3f}</p>")
            f.write('<h3>Labels</h3><pre>')
            f.write(json.dumps(metrics['labels'], indent=2))
            f.write('</pre>')
            f.write('<h3>Per-fold metrics</h3><pre>')
            f.write(json.dumps(metrics['folds'], indent=2))
            f.write('</pre>')
        f.write('</body></html>')
    return html


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Species classification using MLP on basic features')
    ap.add_argument('--data_dir', default=os.path.join('data', 'zenodo_5790768'))
    ap.add_argument('--out_root', default=os.path.join('results', 'classification'))
    ap.add_argument('--segments', type=int, default=6)
    ap.add_argument('--min_segment_points', type=int, default=2048)
    ap.add_argument('--deterministic', action='store_true')
    args = ap.parse_args()

    try:
        X, y, paths = collect_dataset(
            args.data_dir,
            segments=args.segments,
            min_segment_points=args.min_segment_points,
            random_sample=(not args.deterministic),
            rng_seed=42
        )
        ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        out_dir = os.path.join(args.out_root, ts)
        feature_names = sorted(basic_signal_features(np.array([0.0])).keys())
        if len(y) < 4:
            metrics = {'error': 'insufficient_samples', 'count': len(y)}
        else:
            metrics = train_and_eval(X, y)
        html = write_outputs(out_dir, paths, metrics, feature_names)
        print(json.dumps({'out_dir': out_dir, 'summary_html': html, 'n': len(y), 'metrics': metrics}))
    except ModuleNotFoundError as e:
        ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        out_dir = os.path.join(args.out_root, ts)
        os.makedirs(out_dir, exist_ok=True)
        metrics = {'error': 'missing_dependency', 'module': str(e)}
        feature_names = sorted(basic_signal_features(np.array([0.0])).keys())
        html = write_outputs(out_dir, [], metrics, feature_names)
        print(json.dumps({'out_dir': out_dir, 'summary_html': html, 'metrics': metrics}))
    except Exception as e:
        ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        out_dir = os.path.join(args.out_root, ts)
        os.makedirs(out_dir, exist_ok=True)
        metrics = {'error': 'exception', 'message': str(e)}
        feature_names = sorted(basic_signal_features(np.array([0.0])).keys())
        html = write_outputs(out_dir, [], metrics, feature_names)
        print(json.dumps({'out_dir': out_dir, 'summary_html': html, 'metrics': metrics}))


if __name__ == '__main__':
    main()


