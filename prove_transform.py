#!/usr/bin/env python3
import numpy as np
import argparse
import re
from typing import Dict, Tuple, Optional, List


def gaussian(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)


def morlet_real(x, sigma=1.0, omega0=5.0):
    # Real Morlet (no correction term); sigma controls envelope width
    return np.exp(-0.5 * (x / sigma) ** 2) * np.cos(omega0 * x)


def synthesize_signal(
    t: np.ndarray,
    u_center: float,
    sigma_u: float,
    k0: float,
    A1: float,
    A2: float,
    omega2: float,
    spike_amp: float,
    noise_sigma: float,
    random_state: int = 0,
):
    rng = np.random.default_rng(random_state)

    # √t-locked component: s1(t) = A1 * cos(k0 * sqrt(t) + phi) * Gaussian in u around u_center
    u = np.sqrt(np.maximum(t, 0.0))
    phi1 = rng.uniform(0, 2 * np.pi)
    env_u = gaussian(u - u_center, sigma_u)
    s1 = A1 * np.cos(k0 * u + phi1) * env_u

    # Ordinary sinusoid in t with a broad Gaussian envelope (so it is detectable by STFT)
    T_total = t[-1]
    env_t = gaussian(t - 0.5 * T_total, 0.25 * T_total)
    phi2 = rng.uniform(0, 2 * np.pi)
    s2 = A2 * np.cos(omega2 * t + phi2) * env_t

    # Sparse spikes
    spikes = np.zeros_like(t)
    num_spikes = 20
    spike_positions = rng.uniform(0.05 * T_total, 0.95 * T_total, size=num_spikes)
    spike_width = 0.002 * T_total
    for tp in spike_positions:
        spikes += spike_amp * gaussian(t - tp, spike_width) * np.sign(rng.standard_normal())

    noise = rng.normal(0.0, noise_sigma, size=t.shape)
    V = s1 + s2 + spikes + noise
    return V, {
        "u": u,
        "s1": s1,
        "s2": s2,
        "spikes": spikes,
        "env_u": env_u,
        "env_t": env_t,
        "phi1": phi1,
        "phi2": phi2,
    }


def sqrt_time_transform_fft(V_func, tau, u_grid, u0=0.0, window: str = "gaussian", detrend_u: bool = False):
    # f_tau(u) = 2u V(u^2) psi((u - u0)/tau)
    x = (u_grid - u0) / tau
    if window == "morlet":
        psi = morlet_real(x, sigma=1.0, omega0=5.0)
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
    # Optional linear detrend in u-domain to reduce low-k leakage
    if detrend_u:
        # Fit a line a*u + b over support where psi has non-negligible weight
        w = (np.abs(psi) > (0.05 * np.max(np.abs(psi))))
        if np.any(w):
            ug = u_grid[w]
            fg = f_u[w]
            try:
                a, b = np.polyfit(ug, fg, 1)
                f_u = f_u - (a * u_grid + b)
            except Exception:
                pass
    # Zero-pad to next power of two for efficient FFT
    N = len(f_u)
    N_fft = 1 << (N - 1).bit_length()
    f_pad = np.zeros(N_fft, dtype=float)
    f_pad[:N] = f_u
    F = np.fft.rfft(f_pad)
    k_fft = 2.0 * np.pi * np.fft.rfftfreq(N_fft, d=du)
    # Continuous integral approximated by sum ⇒ multiply by du
    W = F * du
    return k_fft, W


def stft_fft(V_func, t0, sigma_t, t_grid):
    # Window in t is Gaussian centered at t0, then single FFT
    win = gaussian(t_grid - t0, sigma_t)
    V_vals = V_func(t_grid)
    g = V_vals * win
    dt = t_grid[1] - t_grid[0]
    N = len(g)
    N_fft = 1 << (N - 1).bit_length()
    g_pad = np.zeros(N_fft, dtype=float)
    g_pad[:N] = g
    G = np.fft.rfft(g_pad)
    omega_fft = 2.0 * np.pi * np.fft.rfftfreq(N_fft, d=dt)
    # Continuous integral approximation scaling
    G = G * dt
    return omega_fft, G


def spectral_concentration(power_arr: np.ndarray):
    # concentration = max(power) / sum(power)
    total = np.sum(power_arr)
    if total <= 0:
        return 0.0
    return float(np.max(power_arr) / total)


def snr_vs_background(power_arr: np.ndarray, target_idx: int, exclude_width: int = 2):
    # Compare power at target index vs median of the rest
    n = len(power_arr)
    mask = np.ones(n, dtype=bool)
    lo = max(0, target_idx - exclude_width)
    hi = min(n, target_idx + exclude_width + 1)
    mask[lo:hi] = False
    bg = np.median(power_arr[mask]) if np.any(mask) else 1e-12
    return float(power_arr[target_idx] / (bg + 1e-12))


def run_case(case_name: str, T_total: float, dt: float, u_center: float, sigma_u: float, k0: float,
             A1: float, A2: float, omega2: float, spike_amp: float, noise_sigma: float, seed: int):
    t = np.arange(0.0, T_total, dt)
    V, _ = synthesize_signal(t, u_center, sigma_u, k0, A1, A2, omega2, spike_amp, noise_sigma, random_state=seed)

    def V_func(t_vals):
        return np.interp(t_vals, t, V)

    U_max = np.sqrt(T_total)
    N_u = 4096
    u_grid = np.linspace(0.0, U_max, N_u, endpoint=False)
    tau = sigma_u
    # Center the √t-domain window at the event location u_center
    k_fft, W_vals = sqrt_time_transform_fft(V_func, tau, u_grid, u0=u_center)
    power_k = np.abs(W_vals) ** 2
    idx_k0 = int(np.argmin(np.abs(k_fft - k0)))
    conc_sqrt = spectral_concentration(power_k)
    snr_sqrt = snr_vs_background(power_k, idx_k0, exclude_width=3)

    t_center = u_center ** 2
    sigma_t = 2.0 * u_center * sigma_u
    omega_inst = k0 / (2.0 * u_center)
    omega_fft, G_vals = stft_fft(V_func, t_center, sigma_t, t)
    power_w = np.abs(G_vals) ** 2
    idx_w0 = int(np.argmin(np.abs(omega_fft - omega_inst)))
    conc_stft = spectral_concentration(power_w)
    snr_stft = snr_vs_background(power_w, idx_w0, exclude_width=3)

    print(f"\n=== {case_name} ===")
    print(f"u_center={u_center}, t_center={t_center:.2f}, sigma_u={sigma_u}, tau={tau}")
    print(f"k0={k0}, omega_inst={omega_inst:.4f} rad/s")
    print("√t-transform:")
    print(f"  conc={conc_sqrt:.6f}, SNR(k0)={snr_sqrt:.2f}")
    print("t-STFT:")
    print(f"  conc={conc_stft:.6f}, SNR(omega_inst)={snr_stft:.2f}")
    print("Ratios (√t / STFT):")
    print(f"  SNR ratio={((snr_sqrt+1e-12)/(snr_stft+1e-12)):.2f}x, conc ratio={((conc_sqrt+1e-12)/(conc_stft+1e-12)):.2f}x")


def main():
    parser = argparse.ArgumentParser(description="√t-transform demo and analysis")
    parser.add_argument("--mode", choices=["demo", "csv", "zenodo"], default="demo")
    parser.add_argument("--file", type=str, default="", help="CSV file with columns: time, V1[, V2..][, moisture]")
    parser.add_argument("--channel", type=str, default="V1", help="Channel name to analyze (for csv mode)")
    parser.add_argument("--nu0", type=int, default=64, help="Number of u0 positions (csv mode)")
    parser.add_argument("--taus", type=str, default="5.5,24.5,104", help="Comma-separated τ values to evaluate")
    args = parser.parse_args()

    if args.mode == "demo":
        # Common params
        T_total = 3600.0
        dt = 0.1
        k0 = 12.0
        A1 = 0.8
        A2 = 0.1  # keep ordinary sinusoid weak
        omega2 = 2 * np.pi * 0.01
        spike_amp = 0.2
        noise_sigma = 0.3

        # Case A: early-time √t-locked event (hard for t-STFT)
        run_case(
            case_name="Case A: early event",
            T_total=T_total,
            dt=dt,
            u_center=10.0,
            sigma_u=3.0,
            k0=k0,
            A1=A1,
            A2=A2,
            omega2=omega2,
            spike_amp=spike_amp,
            noise_sigma=noise_sigma,
            seed=1,
        )

        # Case B: late-time √t-locked event (easier for t-STFT)
        run_case(
            case_name="Case B: late event",
            T_total=T_total,
            dt=dt,
            u_center=60.0,
            sigma_u=6.0,
            k0=k0,
            A1=A1,
            A2=A2,
            omega2=omega2,
            spike_amp=spike_amp,
            noise_sigma=noise_sigma,
            seed=2,
        )
        return

    if args.mode == "zenodo":
        if not args.file:
            raise SystemExit("--file must point to a Zenodo .txt file for zenodo mode")
        tau_values = np.array([float(x) for x in args.taus.split(",") if x.strip()])
        t, channels = load_zenodo_timeseries(args.file)
        # pick first finite channel as default if not provided
        pick = None
        for name, vec in channels.items():
            if np.isfinite(vec).any():
                pick = name
                break
        if pick is None:
            raise SystemExit("No finite-valued channels found in Zenodo file")
        if args.channel in channels:
            pick = args.channel
        V = channels[pick]
        print(f"Analyzing channel: {pick}")

        def V_func(t_vals):
            return np.interp(t_vals, t, np.nan_to_num(V, nan=np.nanmean(V)))

        U_max = np.sqrt(t[-1])
        u0_grid = np.linspace(0.0, U_max, args.nu0, endpoint=False)
        band_powers = compute_tau_band_powers(V_func, u0_grid, tau_values)
        dom_idx = np.argmax(band_powers, axis=1)
        dom_tau = tau_values[dom_idx]
        dom_time = u0_grid ** 2
        # Report simple summary
        unique, counts = np.unique(dom_tau, return_counts=True)
        print("Zenodo analysis summary (time-in-τ):")
        for ut, c in zip(unique, counts):
            print(f"  τ={ut:g}: {(c/len(dom_tau))*100:.1f}%")
        return

    # CSV analysis mode
    if not args.file:
        raise SystemExit("--file is required for csv mode")

    tau_values = np.array([float(x) for x in args.taus.split(",") if x.strip()])
    t, channels, moisture = load_csv_timeseries(args.file)
    if args.channel not in channels:
        raise SystemExit(f"Channel {args.channel} not in CSV columns {list(channels.keys())}")
    V = channels[args.channel]

    # Build callable
    def V_func(t_vals):
        return np.interp(t_vals, t, V)

    U_max = np.sqrt(t[-1])
    u0_grid = np.linspace(0.0, U_max, args.nu0, endpoint=False)

    # Compute τ-band powers over u0
    band_powers = compute_tau_band_powers(V_func, u0_grid, tau_values)

    # Map τ to biological families
    families = ["very_fast", "slow", "very_slow"]
    family_map = {tau_values[i]: families[i] if i < len(families) else f"tau_{i}" for i in range(len(tau_values))}

    # Dominant family over time
    dom_idx = np.argmax(band_powers, axis=1)
    dom_tau = tau_values[dom_idx]
    dom_family = [family_map[tv] for tv in dom_tau]
    dom_time = u0_grid ** 2

    print("CSV analysis summary:")
    for fam in set(dom_family):
        frac = np.mean([1 if f == fam else 0 for f in dom_family])
        print(f"  time-in-{fam}: {frac*100:.1f}%")

    # If moisture present, correlate band power with moisture derivative
    if moisture is not None:
        m = moisture
        mdt = (m[1:] - m[:-1]) / np.maximum(1e-9, (t[1:] - t[:-1]))
        # Resample mdt to dom_time via interpolation
        mdt_time = 0.5 * (t[1:] + t[:-1])
        mdt_on_u = np.interp(dom_time, mdt_time, mdt)
        for i, tau in enumerate(tau_values):
            r = corrcoef_safe(band_powers[:, i], mdt_on_u)
            label = family_map[tau]
            print(f"  corr(band_power[{label}], d(moisture)/dt) = {r:.3f}")


def compute_tau_band_powers(V_func, u0_grid: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    # For each (u0, tau), compute ∑_k |W(k; u0, tau)|^2
    # Using FFT spectrum magnitudes
    # Return array shape (len(u0_grid), len(tau_values))
    U_max = u0_grid[-1] if len(u0_grid) > 0 else 0.0
    # Build a shared u_grid for FFT resolution
    # Use power-of-two length near 4096 for efficiency
    N_u = 4096
    u_grid = np.linspace(0.0, U_max if U_max > 0 else 1.0, N_u, endpoint=False)
    powers = np.zeros((len(u0_grid), len(tau_values)), dtype=float)
    for iu, u0 in enumerate(u0_grid):
        for it, tau in enumerate(tau_values):
            _, W = sqrt_time_transform_fft(V_func, tau, u_grid, u0=u0)
            powers[iu, it] = float(np.sum(np.abs(W) ** 2))
    return powers


def load_csv_timeseries(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
    # Try to load with headers; expected columns: time, V1[, V2..][, moisture]
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if "time" not in data.dtype.names:
        raise ValueError("CSV must contain a 'time' column header")
    t = np.asarray(data["time"], dtype=float)
    channels: Dict[str, np.ndarray] = {}
    moisture: Optional[np.ndarray] = None
    for name in data.dtype.names:
        if name == "time":
            continue
        arr = np.asarray(data[name], dtype=float)
        if name.lower().startswith("moist"):
            moisture = arr
        else:
            channels[name] = arr
    return t, channels, moisture


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if denom <= 0:
        return 0.0
    return float(np.sum(x0 * y0) / denom)


def load_zenodo_timeseries(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Robustly parse whitespace-delimited numeric table with optional header lines.
    # Many Zenodo txt files start with non-numeric headers; skip them.
    numeric_rows: List[List[float]] = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Quickly filter non-numeric lines
            if re.search(r"[A-Za-z]", s):
                continue
            parts = re.split(r"\s+", s)
            row: List[float] = []
            ok = True
            for p in parts:
                try:
                    if p.lower() == 'nan':
                        row.append(float('nan'))
                    else:
                        row.append(float(p))
                except ValueError:
                    ok = False
                    break
            if ok and row:
                numeric_rows.append(row)
    if not numeric_rows:
        raise ValueError(f"No numeric data found in {path}")
    # Pad ragged rows to max length with NaN
    max_len = max(len(r) for r in numeric_rows)
    arr = np.full((len(numeric_rows), max_len), np.nan, dtype=float)
    for i, r in enumerate(numeric_rows):
        arr[i, :len(r)] = r
    n, m = arr.shape
    # Build time vector assuming 1 Hz sampling
    t = np.arange(n, dtype=float)
    # Name channels diff_1_2, diff_3_4, ... or col_k
    channels: Dict[str, np.ndarray] = {}
    for j in range(m):
        name = f"diff_{j+1}"
        channels[name] = arr[:, j]
    return t, channels


if __name__ == "__main__":
    main()


