#!/usr/bin/env python3
"""
Psi-sweep utility for √t-warped transform

Evaluates multiple window functions (psi) at matched settings and reports
SNR and spectral concentration across u0, τ.

Windows supported:
- gaussian
- morlet (real)
- dog (Mexican hat / DoG-2)
- bump (compactly supported smooth bump)

Outputs a JSON summary under results/psi_sweep/<timestamp>/summary.json
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime as _dt
from typing import Dict, List, Tuple

# Reuse robust Zenodo loader from prove_transform
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prove_transform import load_zenodo_timeseries  # type: ignore


def gaussian(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2)


def morlet_real(x: np.ndarray, sigma: float = 1.0, omega0: float = 5.0) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2) * np.cos(omega0 * x)


def dog2_mexican_hat(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    # Mexican hat (Ricker): (1 - x^2/sigma^2) * exp(-x^2/(2 sigma^2))
    z = x / sigma
    return (1.0 - z * z) * np.exp(-0.5 * z * z)


def bump(x: np.ndarray, support: float = 3.0) -> np.ndarray:
    # Smooth compact bump: exp(-1/(1 - (x/a)^2)) for |x|<a else 0
    a = float(support)
    z = x / a
    out = np.zeros_like(x, dtype=float)
    m = np.abs(z) < 1.0
    # Avoid divide-by-zero; where m is True, denominator positive
    out[m] = np.exp(-1.0 / (1.0 - z[m] * z[m]))
    # Normalize peak to ~1 for fair comparison
    if np.max(out) > 0:
        out = out / np.max(out)
    return out


def sqrt_time_transform_fft(V_func, tau, u_grid, u0=0.0, window: str = "gaussian", detrend_u: bool = False):
    x = (u_grid - u0) / tau
    if window == "morlet":
        psi = morlet_real(x, sigma=1.0, omega0=5.0)
    elif window == "dog":
        psi = dog2_mexican_hat(x, sigma=1.0)
    elif window == "bump":
        psi = bump(x, support=3.0)
    else:
        psi = gaussian(x, 1.0)

    t_vals = u_grid ** 2
    V_vals = V_func(t_vals)
    f_u = 2.0 * u_grid * V_vals * psi

    du = u_grid[1] - u_grid[0]
    # Normalize window energy to remove τ bias
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
    # modest FFT size for speed
    N_fft = 1 << max(8, (N - 1).bit_length() - 2)
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


def compute_metrics_for_windows(V: np.ndarray, t: np.ndarray, tau_values: List[float],
                                u0_grid: np.ndarray, windows: List[str], detrend: bool) -> Dict:
    # Build callable for interpolation
    def V_func(t_vals):
        return np.interp(t_vals, t, np.nan_to_num(V, nan=np.nanmean(V)))

    U_max = u0_grid[-1] if len(u0_grid) > 0 else 0.0
    # modest grid for speed
    N_u = 1024
    u_grid = np.linspace(0.0, U_max if U_max > 0 else 1.0, N_u, endpoint=False)

    out: Dict[str, Dict] = {}
    for wname in windows:
        conc_vals = []
        snr_vals = []
        for u0 in u0_grid:
            for tau in tau_values:
                k, W = sqrt_time_transform_fft(V_func, tau, u_grid, u0=u0, window=wname, detrend_u=detrend)
                P = np.abs(W) ** 2
                peak_idx = int(np.argmax(P))
                conc_vals.append(spectral_concentration(P))
                snr_vals.append(snr_vs_background(P, peak_idx, exclude_width=5))
        out[wname] = {
            'avg_concentration': float(np.mean(conc_vals)) if conc_vals else 0.0,
            'avg_snr': float(np.mean(snr_vals)) if snr_vals else 0.0,
            'n_points': int(len(conc_vals))
        }
    return out


def analyze_file(path: str, channel_pick: str, taus: List[float], nu0: int,
                 windows: List[str], detrend: bool) -> Dict:
    t, channels = load_zenodo_timeseries(path)
    pick = channel_pick if channel_pick in channels else None
    if pick is None:
        for name, arr in channels.items():
            if np.isfinite(arr).any():
                pick = name
                break
    if pick is None:
        raise RuntimeError("No finite channels found")
    V = channels[pick]
    U_max = float(np.sqrt(t[-1]))
    u0_grid = np.linspace(0.0, U_max, max(4, nu0), endpoint=False)
    metrics = compute_metrics_for_windows(V, t, taus, u0_grid, windows, detrend)
    return {
        'file': path,
        'channel': pick,
        'taus': taus,
        'nu0': int(nu0),
        'windows': windows,
        'detrend_u': bool(detrend),
        'metrics': metrics
    }


def main():
    ap = argparse.ArgumentParser(description="Psi-sweep for √t transform")
    ap.add_argument('--file', help='Zenodo TXT file (analyze single file)')
    ap.add_argument('--channel', default='', help='Channel to use')
    ap.add_argument('--taus', default='5.5,24.5,104', help='Comma-separated τ values')
    ap.add_argument('--nu0', type=int, default=16, help='Number of u0 positions')
    ap.add_argument('--windows', default='gaussian,morlet,dog,bump', help='Comma-separated window names')
    ap.add_argument('--detrend_u', action='store_true', help='Apply u-domain detrend')
    ap.add_argument('--from_cross_modal_summary', default='', help='Path to cross-modal summary.json to select one file per species')
    ap.add_argument('--out_root', default='results/psi_sweep', help='Output root directory')
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(',') if x.strip()]
    windows = [w.strip() for w in args.windows.split(',') if w.strip()]

    files: List[str] = []
    if args.file:
        files = [args.file]
    elif args.from_cross_modal_summary and os.path.exists(args.from_cross_modal_summary):
        try:
            arr = json.load(open(args.from_cross_modal_summary, 'r'))
            seen_species = set()
            for e in arr:
                fp = e.get('file') or ''
                species = os.path.basename(fp).replace('.txt', '')
                if fp and species not in seen_species and os.path.exists(fp):
                    files.append(fp)
                    seen_species.add(species)
        except Exception:
            pass
    else:
        # Fallback: analyze all Zenodo txt files if present (lightweight params)
        for root, _, fnames in os.walk('data'):
            for fn in fnames:
                if fn.lower().endswith('.txt'):
                    files.append(os.path.join(root, fn))

    ts = _dt.now().isoformat(timespec='seconds').replace(':', '-')
    out_dir = os.path.join(args.out_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for fp in files:
        try:
            res = analyze_file(fp, args.channel, taus, args.nu0, windows, args.detrend_u)
            results.append(res)
        except Exception as e:
            results.append({'file': fp, 'error': str(e)})

    out_path = os.path.join(out_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(out_path)


if __name__ == '__main__':
    main()


