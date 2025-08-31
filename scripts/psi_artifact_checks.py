#!/usr/bin/env python3
"""
Artifact checks for psi windows in √t transform.

Focus: Gaussian vs Bump on a target file (default: Ghost Fungi),
varying detrend, bump support, FFT length, and testing against
phase-randomized surrogates for significance.

Outputs a JSON report under results/psi_artifacts/<timestamp>/report.json
with per-config real metrics and surrogate distributions + p-values.
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


def gaussian(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2)


def bump(x: np.ndarray, support: float = 3.0) -> np.ndarray:
    a = float(support)
    z = x / a
    out = np.zeros_like(x, dtype=float)
    m = np.abs(z) < 1.0
    out[m] = np.exp(-1.0 / (1.0 - z[m] * z[m]))
    if np.max(out) > 0:
        out = out / np.max(out)
    return out


def sqrt_time_transform_fft(V_func, tau, u_grid, u0=0.0, window: str = "gaussian",
                            detrend_u: bool = False, bump_support: float = 3.0,
                            fft_shrink_pow2: int = 0):
    x = (u_grid - u0) / tau
    if window == "bump":
        psi = bump(x, support=bump_support)
    else:
        psi = gaussian(x, 1.0)

    t_vals = u_grid ** 2
    V_vals = V_func(t_vals)
    f_u = 2.0 * u_grid * V_vals * psi

    du = u_grid[1] - u_grid[0]
    # Energy normalization
    win_energy = np.sqrt(np.sum(psi ** 2) * du)
    if win_energy > 0:
        f_u = f_u / win_energy

    if detrend_u:
        w = (np.abs(psi) > (0.05 * np.max(np.abs(psi))))
        if np.any(w):
            ug = u_grid[w]
            fg = f_u[w]
            try:
                a, b = np.polyfit(ug, fg, 1)
                f_u = f_u - (a * u_grid + b)
            except Exception:
                pass

    N = len(f_u)
    pow2 = (N - 1).bit_length()
    pow2 = max(8, pow2 - max(0, fft_shrink_pow2))
    N_fft = 1 << pow2
    f_pad = np.zeros(N_fft, dtype=float)
    f_pad[: min(N_fft, N)] = f_u[: min(N_fft, N)]
    F = np.fft.rfft(f_pad)
    k_fft = 2.0 * np.pi * np.fft.rfftfreq(N_fft, d=du)
    W = F * du
    return k_fft, W


def spectral_concentration(power_arr: np.ndarray) -> float:
    total = np.sum(power_arr)
    if total <= 0:
        return 0.0
    return float(np.max(power_arr) / total)


def snr_vs_background(power_arr: np.ndarray, target_idx: int, exclude_width: int = 3) -> float:
    n = len(power_arr)
    mask = np.ones(n, dtype=bool)
    lo = max(0, target_idx - exclude_width)
    hi = min(n, target_idx + exclude_width + 1)
    mask[lo:hi] = False
    bg = np.median(power_arr[mask]) if np.any(mask) else 1e-12
    return float(power_arr[target_idx] / (bg + 1e-12))


def compute_metrics(V: np.ndarray, t: np.ndarray, taus: List[float], u0_grid: np.ndarray,
                    window: str, detrend: bool, bump_support: float, fft_shrink_pow2: int) -> Dict:
    def V_func(t_vals):
        return np.interp(t_vals, t, np.nan_to_num(V, nan=np.nanmean(V)))
    U_max = u0_grid[-1] if len(u0_grid) > 0 else 0.0
    N_u = 1024
    u_grid = np.linspace(0.0, U_max if U_max > 0 else 1.0, N_u, endpoint=False)

    conc_vals, snr_vals = [], []
    for u0 in u0_grid:
        for tau in taus:
            k, W = sqrt_time_transform_fft(V_func, tau, u_grid, u0=u0, window=window,
                                           detrend_u=detrend, bump_support=bump_support,
                                           fft_shrink_pow2=fft_shrink_pow2)
            P = np.abs(W) ** 2
            peak_idx = int(np.argmax(P))
            conc_vals.append(spectral_concentration(P))
            snr_vals.append(snr_vs_background(P, peak_idx, exclude_width=5))
    return {
        'avg_concentration': float(np.mean(conc_vals)) if conc_vals else 0.0,
        'avg_snr': float(np.mean(snr_vals)) if snr_vals else 0.0,
        'n_points': int(len(conc_vals))
    }


def phase_randomize(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phase = np.angle(X)
    # Randomize non-DC, non-Nyquist phases
    n = X.shape[0]
    rand_phase = rng.uniform(-np.pi, np.pi, size=n)
    rand_phase[0] = 0.0
    X_surr = mag * np.exp(1j * rand_phase)
    x_surr = np.fft.irfft(X_surr, n=len(x))
    return x_surr.astype(float)


def surrogate_test(V: np.ndarray, t: np.ndarray, taus: List[float], u0_grid: np.ndarray,
                   window: str, detrend: bool, bump_support: float, fft_shrink_pow2: int,
                   n_surrogates: int = 50, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    real = compute_metrics(V, t, taus, u0_grid, window, detrend, bump_support, fft_shrink_pow2)
    snr_real = real['avg_snr']
    conc_real = real['avg_concentration']
    snr_surr = []
    conc_surr = []
    for _ in range(n_surrogates):
        V_s = phase_randomize(V, rng)
        m = compute_metrics(V_s, t, taus, u0_grid, window, detrend, bump_support, fft_shrink_pow2)
        snr_surr.append(m['avg_snr'])
        conc_surr.append(m['avg_concentration'])
    snr_surr = np.array(snr_surr, dtype=float)
    conc_surr = np.array(conc_surr, dtype=float)
    # One-sided p-values: P(surrogate >= real)
    p_snr = float((np.sum(snr_surr >= snr_real) + 1) / (len(snr_surr) + 1))
    p_conc = float((np.sum(conc_surr >= conc_real) + 1) / (len(conc_surr) + 1))
    return {
        'real': real,
        'surrogate': {
            'snr': snr_surr.tolist(),
            'concentration': conc_surr.tolist(),
            'p_snr': p_snr,
            'p_concentration': p_conc
        }
    }


def main():
    ap = argparse.ArgumentParser(description='Psi artifact checks (Gaussian vs Bump)')
    ap.add_argument('--file', default='data/zenodo_5790768/Ghost Fungi Omphalotus nidiformis.txt')
    ap.add_argument('--channel', default='')
    ap.add_argument('--taus', default='5.5,24.5,104')
    ap.add_argument('--nu0', type=int, default=8)
    ap.add_argument('--n_surrogates', type=int, default=50)
    ap.add_argument('--out_root', default='results/psi_artifacts')
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(',') if x.strip()]
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
    U_max = float(np.sqrt(t[-1]))
    u0_grid = np.linspace(0.0, U_max, max(4, args.nu0), endpoint=False)

    configs = [
        ('gaussian', False, 3.0, 0),
        ('gaussian', True, 3.0, 0),
        ('bump', False, 3.0, 0),
        ('bump', True, 3.0, 0),
        ('bump', True, 2.5, 0),
        ('bump', True, 3.5, 0),
        ('bump', True, 3.0, 2),  # smaller FFT size
    ]

    results: Dict[str, Dict] = {}
    for (w, detrend, support, shrink) in configs:
        key = f"{w}_detrend_{int(detrend)}_supp_{support}_shrink_{shrink}"
        results[key] = surrogate_test(V, t, taus, u0_grid, w, detrend, support, shrink,
                                      n_surrogates=args.n_surrogates)

    ts = _dt.now().isoformat(timespec='seconds').replace(':', '-')
    out_dir = os.path.join(args.out_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'report.json')
    with open(out_path, 'w') as f:
        json.dump({
            'file': args.file,
            'channel': pick,
            'taus': taus,
            'nu0': int(args.nu0),
            'configs': configs,
            'results': results
        }, f, indent=2)
    print(out_path)


if __name__ == '__main__':
    main()


