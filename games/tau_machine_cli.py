#!/usr/bin/env python3
"""
Tau Machine CLI: a logic-style game on τ-band √t-transform features

Players compose simple rules (threshold gates and synchrony-like combos)
over τ-band power features and spike stats to solve biological tasks
like event detection or classification. The score is cross-validated
accuracy/F1 minus a small complexity penalty.

Data sources supported:
- CSV with time and one or more voltage channels
- Zenodo-format text files via existing loader in prove_transform.py
- Synthetic demo from prove_transform.synthesize_signal

Dependencies: numpy, argparse; optional scikit-learn for CV metrics
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Local modules
import prove_transform as pt
from ml_pipeline import extract_features_from_file, simple_train_test_split


def _safe_import_sklearn_metrics():
    try:
        from sklearn.metrics import f1_score, accuracy_score
        return f1_score, accuracy_score
    except Exception:
        return None, None


@dataclass
class Gate:
    """A simple threshold gate on a single feature index.

    kind: one of {">", "<", ">=", "<="}
    idx: feature column index in X
    thr: threshold value
    weight: optional contribution weight used by combiners
    """
    kind: str
    idx: int
    thr: float
    weight: float = 1.0

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.idx]
        if self.kind == ">":
            return (col > self.thr).astype(float)
        if self.kind == ">=":
            return (col >= self.thr).astype(float)
        if self.kind == "<":
            return (col < self.thr).astype(float)
        if self.kind == "<=":
            return (col <= self.thr).astype(float)
        raise ValueError(f"Unknown gate op: {self.kind}")


@dataclass
class Rule:
    """A rule combines multiple gates with a simple aggregator.

    mode: one of {"AND", "OR", "MEAN", "WEIGHTED"}
    threshold: decision threshold applied to the aggregated score
    """
    gates: List[Gate]
    mode: str = "AND"
    threshold: float = 0.5

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.gates:
            return np.zeros(X.shape[0], dtype=float)
        G = np.stack([g.evaluate(X) for g in self.gates], axis=1)
        if self.mode == "AND":
            return (np.all(G > 0.5, axis=1)).astype(float)
        if self.mode == "OR":
            return (np.any(G > 0.5, axis=1)).astype(float)
        if self.mode == "MEAN":
            return np.mean(G, axis=1)
        if self.mode == "WEIGHTED":
            w = np.array([g.weight for g in self.gates], dtype=float)
            w = w / (np.sum(w) + 1e-12)
            return (G @ w)
        raise ValueError(f"Unknown rule mode: {self.mode}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = self.score(X)
        return (s >= self.threshold).astype(int)


def complexity_cost(rule: Rule) -> float:
    # Simple token cost: more gates and non-binary modes are costlier
    base = 0.01 * len(rule.gates)
    if rule.mode in ("MEAN", "WEIGHTED"):
        base += 0.02
    return base


def _load_features_from_csv(csv_path: str, taus: List[float], nu0: int, channel: Optional[str]) -> Tuple[np.ndarray, np.ndarray, str]:
    # Expect CSV headers: time, V1[, V2..][, label]
    import csv
    times: List[float] = []
    series: Dict[str, List[float]] = {}
    label_series: Optional[List[int]] = None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        voltage_cols = [c for c in fieldnames if c.lower().startswith("v")]
        has_label = "label" in fieldnames
        for row in reader:
            times.append(float(row.get("time", "0") or 0))
            for c in voltage_cols:
                series.setdefault(c, []).append(float(row.get(c, "0") or 0))
            if has_label:
                label_series = label_series or []
                label_series.append(int(row["label"]))
    # Choose channel
    pick = channel if channel and channel in series else (voltage_cols[0] if voltage_cols else None)
    if pick is None:
        raise RuntimeError("CSV missing voltage columns starting with V*")
    t = np.array(times, dtype=float)
    V = np.array(series[pick], dtype=float)

    def V_func(t_vals: np.ndarray) -> np.ndarray:
        return np.interp(t_vals, t, np.nan_to_num(V, nan=float(np.nanmean(V))))

    U_max = float(np.sqrt(t[-1])) if t.size > 1 else 1.0
    u0_grid = np.linspace(0.0, U_max, nu0, endpoint=False)
    N_u = 1024
    u_grid = np.linspace(0.0, U_max, N_u, endpoint=False)

    feats_list: List[List[float]] = []
    for u0 in u0_grid:
        window_feats: List[float] = []
        power_per_tau: List[float] = []
        kstats: List[Tuple[float, float, float, float]] = []
        for tau in taus:
            k_fft, W = pt.sqrt_time_transform_fft(V_func, tau, u_grid, u0=u0)
            P = np.abs(W) ** 2
            power = float(np.sum(P))
            # centroid, bandwidth, peaks, entropy
            total = float(np.sum(P) + 1e-12)
            centroid = float(np.sum(k_fft * P) / total)
            var = float(np.sum(((k_fft - centroid) ** 2) * P) / total)
            bw = float(np.sqrt(max(var, 0.0)))
            # peaks
            thr = np.percentile(P, 90.0) if P.size >= 3 else 0.0
            peaks = 0
            for i in range(1, len(P) - 1):
                if P[i] > P[i - 1] and P[i] > P[i + 1] and P[i] >= thr:
                    peaks += 1
            # entropy in k
            nbins = 32
            edges = np.linspace(k_fft.min(), k_fft.max(), nbins + 1)
            idx = np.clip(np.digitize(k_fft, edges) - 1, 0, nbins - 1)
            bins = np.bincount(idx, weights=P, minlength=nbins).astype(float)
            p = bins / (np.sum(bins) + 1e-12)
            nz = p[p > 0]
            Hk = float(-np.sum(nz * np.log2(nz)))
            power_per_tau.append(power)
            kstats.append((centroid, bw, float(peaks), Hk))
        power_arr = np.array(power_per_tau, dtype=float)
        norm = float(np.sum(power_arr) + 1e-12)
        power_norm = (power_arr / norm).tolist()
        for i in range(len(taus)):
            c, bw, pks, Hk = kstats[i]
            window_feats += [power_norm[i], c, bw, pks, Hk]
        if len(power_norm) >= 3:
            f, s, vs = power_norm[0], power_norm[1], power_norm[2]
            eps = 1e-9
            window_feats += [float(f / (s + eps)), float(s / (vs + eps)), float(f / (vs + eps))]
        feats_list.append(window_feats)
    feats = np.array(feats_list, dtype=float)
    return feats, u0_grid ** 2, pick


def _load_dataset(data_dir: str, taus: List[float], nu0: int, channel: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a simple labeled dataset from files in data_dir.

    Expects filenames to encode labels, or a companion JSON with labels.
    Strategy:
      - If there is labels.json in data_dir mapping filename->label, use it
      - Else, derive label from filename prefix before first '_'
    Returns X (stacked windows), y (labels per window), groups (file index)
    """
    label_map_path = os.path.join(data_dir, "labels.json")
    explicit_map: Dict[str, Any] = {}
    if os.path.isfile(label_map_path):
        with open(label_map_path, "r") as f:
            explicit_map = json.load(f)

    X_all: List[np.ndarray] = []
    y_all: List[int] = []
    groups: List[int] = []
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".txt") or f.lower().endswith(".csv")]
    files.sort()
    for file_index, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        label = explicit_map.get(fname)
        if label is None:
            base = os.path.splitext(fname)[0]
            label = base.split("_")[0]
        label_str = str(label)
        # Extract features
        if fname.lower().endswith(".csv"):
            feats, times, channel_used = _load_features_from_csv(path, taus, nu0, channel)
        else:
            feats, times, channel_used = extract_features_from_file(path, taus, nu0, channel_hint=channel, n_u=1024)
        y_vec = np.array([label_str] * feats.shape[0])
        X_all.append(feats)
        y_all.append(y_vec)
        groups.extend([file_index] * feats.shape[0])
    if not X_all:
        raise RuntimeError(f"No data files found in {data_dir}")
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    groups = np.array(groups, dtype=int)
    # Map labels to ints for metrics
    classes, y_int = np.unique(y, return_inverse=True)
    return X, y_int, groups


def evaluate_rule(rule: Rule, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    f1_fn, acc_fn = _safe_import_sklearn_metrics()
    y_pred = rule.predict(X)
    if f1_fn is None:
        # Fallback metrics without sklearn
        acc = float(np.mean((y_pred == y).astype(float)))
        # Simple F1 for binary tasks when labels are 0/1
        tp = float(np.sum((y_pred == 1) & (y == 1)))
        fp = float(np.sum((y_pred == 1) & (y == 0)))
        fn = float(np.sum((y_pred == 0) & (y == 1)))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
    else:
        acc = float(acc_fn(y, y_pred))
        f1 = float(f1_fn(y, y_pred, average="binary" if len(np.unique(y)) == 2 else "macro"))
    return {"accuracy": acc, "f1": f1}


def cross_validate_rule(rule: Rule, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None, k: int = 5) -> Dict[str, float]:
    # Simple k-fold CV; if groups provided, use group-wise split by files
    n = X.shape[0]
    if groups is None:
        idx = np.arange(n)
        folds = np.array_split(idx, k)
    else:
        # Group-based: split unique groups
        uniq = np.unique(groups)
        folds = np.array_split(uniq, min(k, len(uniq)))
    acc_list: List[float] = []
    f1_list: List[float] = []
    for fold in folds:
        if groups is None:
            test_idx = fold
        else:
            mask = np.isin(groups, fold)
            test_idx = np.where(mask)[0]
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        metrics = evaluate_rule(rule, X[test_idx], y[test_idx])
        acc_list.append(metrics["accuracy"])
        f1_list.append(metrics["f1"])
    acc = float(np.mean(acc_list)) if acc_list else 0.0
    f1 = float(np.mean(f1_list)) if f1_list else 0.0
    score = f1 - complexity_cost(rule)
    return {"cv_accuracy": acc, "cv_f1": f1, "score": score}


def parse_gate(spec: str) -> Gate:
    # Format: idx op thr [weight]
    # Example: 0 >= 0.3 1.5
    parts = spec.strip().split()
    if len(parts) < 3:
        raise ValueError("Gate spec must be: <idx> <op> <thr> [weight]")
    idx = int(parts[0])
    op = parts[1]
    thr = float(parts[2])
    w = float(parts[3]) if len(parts) >= 4 else 1.0
    return Gate(kind=op, idx=idx, thr=thr, weight=w)


def build_default_rule(n_features: int) -> Rule:
    # A reasonable starting point: require first τ band power_norm > 0.33
    gates = [Gate(kind=">", idx=0, thr=0.33, weight=1.0)] if n_features > 0 else []
    return Rule(gates=gates, mode="AND", threshold=0.5)


def cmd_demo(args: argparse.Namespace) -> None:
    # Create synthetic data: two classes with different √t power distributions
    rng = np.random.default_rng(0)
    n_samples = 400
    # Features: [tau1_power_norm, c1, bw1, pks1, Hk1, tau2_..., tau3_..., ratios(3)] => 3*5 + 3 = 18
    d = 18
    X = np.zeros((n_samples, d), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    # Class 0: higher slow band
    for i in range(n_samples // 2):
        f, s, vs = rng.dirichlet([1.0, 1.5, 3.0])
        X[i, 0] = f; X[i, 5] = s; X[i, 10] = vs
        X[i, 15:18] = [f / (s + 1e-9), s / (vs + 1e-9), f / (vs + 1e-9)]
        # dummy k-stats
        X[i, 1] = 2.0; X[i, 2] = 0.5; X[i, 3] = 2.0; X[i, 4] = 1.0
        X[i, 6] = 5.0; X[i, 7] = 1.0; X[i, 8] = 1.0; X[i, 9] = 1.2
        X[i, 11] = 9.0; X[i, 12] = 2.0; X[i, 13] = 0.0; X[i, 14] = 2.1
        y[i] = 0
    # Class 1: higher fast band
    for i in range(n_samples // 2, n_samples):
        f, s, vs = rng.dirichlet([3.0, 1.5, 1.0])
        X[i, 0] = f; X[i, 5] = s; X[i, 10] = vs
        X[i, 15:18] = [f / (s + 1e-9), s / (vs + 1e-9), f / (vs + 1e-9)]
        X[i, 1] = 2.2; X[i, 2] = 0.6; X[i, 3] = 1.0; X[i, 4] = 1.1
        X[i, 6] = 5.1; X[i, 7] = 1.2; X[i, 8] = 1.1; X[i, 9] = 1.3
        X[i, 11] = 8.9; X[i, 12] = 2.3; X[i, 13] = 0.0; X[i, 14] = 2.2
        y[i] = 1
    rule = build_default_rule(d)
    res = cross_validate_rule(rule, X, y, groups=None, k=5)
    print(json.dumps({"demo_cv": res, "n_features": d, "rule": rule.__dict__}, indent=2))


def cmd_play(args: argparse.Namespace) -> None:
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    if args.data_dir:
        X, y, groups = _load_dataset(args.data_dir, taus, args.nu0, args.channel)
    else:
        # Single file mode
        if not args.file:
            raise SystemExit("--file or --data_dir required")
        if args.file.lower().endswith(".csv"):
            X, times, channel_used = _load_features_from_csv(args.file, taus, args.nu0, args.channel)
        else:
            X, times, channel_used = extract_features_from_file(args.file, taus, args.nu0, channel_hint=args.channel, n_u=1024)
        # Binary labels via time thresholds if provided
        if args.label_windows:
            # Format: start:end,start:end in seconds; label=1 within any window else 0
            windows = []
            for w in args.label_windows.split(","):
                a, b = w.split(":")
                windows.append((float(a), float(b)))
            t = np.linspace(0.0, 1.0, X.shape[0])
            # Map u0 windows (unknown exact times per row); use normalized index
            y = np.zeros(X.shape[0], dtype=int)
            for a, b in windows:
                ia = max(0, int(round(a * X.shape[0] / max(1.0, args.duration_hint))))
                ib = min(X.shape[0], int(round(b * X.shape[0] / max(1.0, args.duration_hint))))
                y[ia:ib] = 1
            groups = None
        else:
            # No labels => use zeros, metrics are not meaningful
            y = np.zeros(X.shape[0], dtype=int)
            groups = None

    # Build rule
    if args.gate:
        gates = [parse_gate(gs) for gs in args.gate]
        rule = Rule(gates=gates, mode=args.mode, threshold=args.threshold)
    else:
        rule = build_default_rule(X.shape[1])
    # Evaluate
    if args.cv:
        result = cross_validate_rule(rule, X, y, groups=groups, k=args.cv)
    else:
        metrics = evaluate_rule(rule, X, y)
        result = {**metrics, "score": metrics.get("f1", 0.0) - complexity_cost(rule)}
    out = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "rule": {
            "mode": rule.mode,
            "threshold": rule.threshold,
            "gates": [g.__dict__ for g in rule.gates],
            "complexity_cost": complexity_cost(rule),
        },
        "result": result,
    }
    print(json.dumps(out, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tau Machine: τ-band logic game over √t features")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Demo
    d = sub.add_parser("demo", help="Run synthetic demo and evaluate default rule")
    d.set_defaults(func=cmd_demo)

    # Play
    pl = sub.add_parser("play", help="Play on a dataset or single file")
    pl.add_argument("--data_dir", type=str, default="", help="Directory with .txt/.csv and optional labels.json")
    pl.add_argument("--file", type=str, default="", help="Single file path (.txt or .csv)")
    pl.add_argument("--channel", type=str, default="", help="Channel hint to select")
    pl.add_argument("--taus", type=str, default="5.5,24.5,104", help="Comma-separated τ values")
    pl.add_argument("--nu0", type=int, default=64, help="Number of u0 windows per file")
    pl.add_argument("--gate", action="append", help="Gate spec: '<idx> <op> <thr> [weight]'. Repeatable.")
    pl.add_argument("--mode", choices=["AND", "OR", "MEAN", "WEIGHTED"], default="AND")
    pl.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for rule output")
    pl.add_argument("--cv", type=int, default=5, help="k-folds; set 0 to disable CV")
    pl.add_argument("--label_windows", type=str, default="", help="For single file: 'start:end,start:end' seconds labeled 1")
    pl.add_argument("--duration_hint", type=float, default=3600.0, help="Total duration seconds for label window mapping")
    pl.set_defaults(func=cmd_play)
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

