#!/usr/bin/env python3
"""
Web frontend for Tau Machine game.

Start with:
  uvicorn web.app:app --host 0.0.0.0 --port 8000
"""
import os
import io
import json
import uuid
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Reuse game logic and feature extraction
from games.tau_machine_cli import (
    Gate, Rule, parse_gate, cross_validate_rule, evaluate_rule, complexity_cost,
)
from ml_pipeline import extract_features_from_file
import prove_transform as pt


app = FastAPI(title="Tau Machine")
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _make_synthetic_dataset(n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    d = 18
    X = np.zeros((n_samples, d), dtype=float)
    y = np.zeros(n_samples, dtype=int)
    # Class 0: higher slow band
    for i in range(n_samples // 2):
        f, s, vs = rng.dirichlet([1.0, 1.5, 3.0])
        X[i, 0] = f; X[i, 5] = s; X[i, 10] = vs
        X[i, 15:18] = [f / (s + 1e-9), s / (vs + 1e-9), f / (vs + 1e-9)]
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
    return X, y


def _load_features_from_csv(csv_path: str, taus: List[float], nu0: int, channel: Optional[str]) -> Tuple[np.ndarray, np.ndarray, str]:
    import csv
    times: List[float] = []
    series: Dict[str, List[float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        voltage_cols = [c for c in fieldnames if c.lower().startswith("v")]
        for row in reader:
            times.append(float(row.get("time", "0") or 0))
            for c in voltage_cols:
                series.setdefault(c, []).append(float(row.get(c, "0") or 0))
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
            total = float(np.sum(P) + 1e-12)
            centroid = float(np.sum(k_fft * P) / total)
            var = float(np.sum(((k_fft - centroid) ** 2) * P) / total)
            bw = float(np.sqrt(max(var, 0.0)))
            thr = np.percentile(P, 90.0) if P.size >= 3 else 0.0
            peaks = 0
            for i in range(1, len(P) - 1):
                if P[i] > P[i - 1] and P[i] > P[i + 1] and P[i] >= thr:
                    peaks += 1
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


def _save_upload(file: UploadFile) -> str:
    os.makedirs("/workspace/cache/uploads", exist_ok=True)
    ext = os.path.splitext(file.filename or "upload")[1]
    out_path = f"/workspace/cache/uploads/{uuid.uuid4().hex}{ext}"
    with open(out_path, "wb") as f:
        f.write(file.file.read())
    return out_path


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/api/play")
async def api_play(
    source: str = Form("demo"),
    taus: str = Form("5.5,24.5,104"),
    nu0: int = Form(64),
    mode: str = Form("AND"),
    threshold: float = Form(0.5),
    cv: int = Form(5),
    gates_text: str = Form(""),
    channel: str = Form(""),
    label_windows: str = Form(""),
    duration_hint: float = Form(3600.0),
    file: Optional[UploadFile] = File(None),
):
    try:
        tau_values = [float(x) for x in taus.split(",") if x.strip()]
        # Build rule
        gates: List[Gate] = []
        for line in (gates_text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            gates.append(parse_gate(line))
        rule = Rule(gates=gates, mode=mode, threshold=threshold)

        # Dataset
        if source == "demo":
            X, y = _make_synthetic_dataset()
            groups = None
        elif source == "file":
            if file is None:
                return JSONResponse({"error": "file required for source=file"}, status_code=400)
            path = _save_upload(file)
            try:
                if path.lower().endswith(".csv"):
                    X, times, ch = _load_features_from_csv(path, tau_values, nu0, channel or None)
                else:
                    X, times, ch = extract_features_from_file(path, tau_values, nu0, channel_hint=(channel or None), n_u=1024)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
            # Labels from windows if provided, else zeros
            if label_windows:
                y = np.zeros(X.shape[0], dtype=int)
                total = max(1.0, float(duration_hint))
                for spec in label_windows.split(","):
                    if not spec:
                        continue
                    a_str, b_str = spec.split(":")
                    a, b = float(a_str), float(b_str)
                    ia = max(0, int(round(a * X.shape[0] / total)))
                    ib = min(X.shape[0], int(round(b * X.shape[0] / total)))
                    y[ia:ib] = 1
            else:
                y = np.zeros(X.shape[0], dtype=int)
            groups = None
        else:
            return JSONResponse({"error": f"unknown source: {source}"}, status_code=400)

        # Evaluate
        if cv and cv > 0:
            res = cross_validate_rule(rule, X, y, groups=groups, k=cv)
        else:
            m = evaluate_rule(rule, X, y)
            res = {**m, "score": m.get("f1", 0.0) - complexity_cost(rule)}

        return JSONResponse({
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "result": res,
            "rule": {
                "mode": rule.mode,
                "threshold": rule.threshold,
                "gates": [g.__dict__ for g in rule.gates],
                "complexity_cost": complexity_cost(rule),
            },
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

