#!/usr/bin/env python3
import os
import numpy as np

# Use a non-interactive backend suitable for headless environments
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception as _e:  # pragma: no cover
    plt = None


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_heatmap(Z: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 title: str,
                 xlabel: str,
                 ylabel: str,
                 out_path: str,
                 cmap: str = "magma",
                 aspect: str = "auto",
                 dpi: int = 140,
                 figsize: tuple | None = None) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create heatmap")
    _ensure_parent_dir(out_path)
    if figsize is None:
        figsize = (10, 5.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # rows correspond to y, columns to x
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    im = ax.imshow(Z, origin='lower', aspect=aspect, cmap=cmap, extent=extent)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Power (arb)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_surface3d(Z: np.ndarray,
                   x: np.ndarray,
                   y: np.ndarray,
                   title: str,
                   xlabel: str,
                   ylabel: str,
                   zlabel: str,
                   out_path: str,
                   stride: int = 2,
                   dpi: int = 140,
                   figsize: tuple | None = None) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create 3D surface plot")
    _ensure_parent_dir(out_path)
    X, Y = np.meshgrid(x, y)
    if figsize is None:
        figsize = (10, 5.5)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_time_series_with_spikes(t: np.ndarray,
                                 v: np.ndarray,
                                 spike_times_s: np.ndarray,
                                 title: str,
                                 out_path: str,
                                 dpi: int = 140,
                                 figsize: tuple | None = None,
                                 stim_times_s: np.ndarray | None = None) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create time-series plot")
    _ensure_parent_dir(out_path)
    if figsize is None:
        figsize = (11, 3.8)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(t, v, lw=0.8, color='#444444')
    # optional stimuli markers
    if stim_times_s is not None and getattr(stim_times_s, 'size', 0) > 0:
        for s in stim_times_s:
            ax.axvline(float(s), color='gold', alpha=0.5, lw=1.0)
    if spike_times_s is not None and getattr(spike_times_s, 'size', 0) > 0:
        y0 = np.interp(spike_times_s, t, v)
        ax.scatter(spike_times_s, y0, s=16, color='crimson', label='spikes', zorder=3)
        ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (mV)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_histogram(data: np.ndarray,
                   bins: int,
                   title: str,
                   xlabel: str,
                   out_path: str,
                   dpi: int = 140,
                   figsize: tuple | None = None) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create histogram")
    _ensure_parent_dir(out_path)
    if figsize is None:
        figsize = (6.5, 3.6)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(data, bins=bins, color='steelblue', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_linepair(x1: np.ndarray, y1: np.ndarray, label1: str,
                  x2: np.ndarray, y2: np.ndarray, label2: str,
                  title: str, xlabel1: str, xlabel2: str, ylabel: str,
                  out_path: str, dpi: int = 160, figsize: tuple | None = None) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create comparison plot")
    _ensure_parent_dir(out_path)
    if figsize is None:
        figsize = (12, 4.0)
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharey=True)
    axs[0].plot(x1, y1, lw=1.2)
    axs[0].set_title(label1)
    axs[0].set_xlabel(xlabel1)
    axs[0].set_ylabel(ylabel)
    axs[1].plot(x2, y2, lw=1.2, color='darkorange')
    axs[1].set_title(label2)
    axs[1].set_xlabel(xlabel2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def assemble_summary_panel(image_paths: list[str], titles: list[str], out_path: str,
                           dpi: int = 140, ncols: int = 2) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create summary panel")
    _ensure_parent_dir(out_path)
    imgs = [plt.imread(p) for p in image_paths if os.path.isfile(p)]
    n = len(imgs)
    if n == 0:
        raise RuntimeError("no images to assemble")
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6.5), dpi=dpi)
    axes = np.array(axes).reshape(-1)
    for i, img in enumerate(imgs):
        ax = axes[i]
        ax.imshow(img)
        ax.set_axis_off()
        if i < len(titles):
            ax.set_title(titles[i], fontsize=10)
    for j in range(i + 1, nrows * ncols):
        axes[j].set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_tau_trends_ci(
    time_s: np.ndarray,
    taus: np.ndarray,
    means: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    title: str,
    out_path: str,
    dpi: int = 140,
    figsize: tuple | None = None,
) -> str:
    """
    Plot τ-normalized power trends with 95% CI shading.
    means/lo/hi are shape (n_time, n_tau).
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create tau trends plot")
    _ensure_parent_dir(out_path)
    if figsize is None:
        figsize = (10.5, 4.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = plt.cm.tab10(np.linspace(0, 1, len(taus)))
    for j, tau in enumerate(taus):
        ax.plot(time_s, means[:, j], lw=1.2, color=colors[j], label=f"τ={float(tau):g}")
        ax.fill_between(time_s, lo[:, j], hi[:, j], color=colors[j], alpha=0.2, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('normalized τ-power')
    ax.legend(loc='upper right', ncols=min(len(taus), 3), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path



def plot_spiral_fingerprint(
    band_fractions: dict,
    spike_count: int,
    snr_sqrt: float,
    snr_stft: float,
    concentration_sqrt: float,
    title: str,
    out_path: str,
    amplitude_entropy_bits: float | None = None,
    taus_for_labels: list[float] | None = None,
    ci_halfwidths: list[float] | None = None,
    dpi: int = 160,
    figsize: tuple | None = None,
) -> str:
    """
    Create a 3D spiral-based fingerprint visualization from summary metrics.

    Visual encoding (schematic, consistent across species):
    - Gold center: global origin of the fingerprint.
    - Red inner circle: fast τ band (small τ). Radius ∝ band fraction.
    - Blue middle/outer circles: slower τ bands (larger τ). Radii ∝ band fractions.
    - Triangles on circles: spike markers (count capped to avoid clutter).
    - Spiral (grey): contextual geometry; z-level encodes √t spectral concentration.
    - z-offset also depends weakly on log10(SNR√/SNR_STFT) to reflect contrast.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create spiral fingerprint")
    _ensure_parent_dir(out_path)
    if figsize is None:
        figsize = (8.5, 7.5)

    # Normalize inputs and set style parameters
    tau_keys = sorted([float(k) for k in band_fractions.keys()])
    if len(tau_keys) == 0:
        raise ValueError("band_fractions is empty")
    # Map smallest τ → fast (inner, red), largest τ → very slow (outer, blue)
    taus_sorted = tau_keys
    if taus_for_labels is not None and len(taus_for_labels) == len(taus_sorted):
        taus_sorted = list(taus_for_labels)
    fracs = [max(0.0, float(band_fractions[str(t)])) for t in taus_sorted]
    total = sum(fracs) if sum(fracs) > 0 else 1.0
    fracs = [f / total for f in fracs]

    # Radii scaling
    base_radius = 1.0
    ring_spacing = 0.7
    ring_radii = [base_radius + i * ring_spacing for i in range(len(taus_sorted))]
    # Effective radii proportional to fraction (but ensure visible minimum)
    min_visible = 0.25
    ring_effective = [r * (min_visible + (1 - min_visible) * frac) for r, frac in zip(ring_radii, fracs)]

    # z-height from concentration and SNR contrast
    conc = float(concentration_sqrt)
    conc = max(0.0, min(1.0, conc))
    snr_ratio = (float(snr_sqrt) + 1e-12) / (float(snr_stft) + 1e-12)
    snr_contrast = np.clip(0.5 + 0.5 * np.tanh(np.log10(max(snr_ratio, 1e-12))), 0.0, 1.0)
    z_height = 1.0 + 2.0 * conc + 0.5 * snr_contrast

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Draw spiral
    t = np.linspace(0, 6 * np.pi, 800)
    a, b = 0.2, 0.12
    r = a + b * t
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = (z_height / (t.max() + 1e-9)) * t
    ax.plot3D(x, y, z, color='#808080', alpha=0.7, lw=1.2)

    # Gold center
    ax.scatter([0], [0], [0], s=120, color='gold', edgecolors='black', linewidths=0.6, zorder=5)

    # Colors for rings: inner red, then two blues
    ring_colors = []
    if len(taus_sorted) >= 1:
        ring_colors.append('red')
    if len(taus_sorted) >= 2:
        ring_colors.append('#1f77b4')  # blue
    for _ in range(max(0, len(taus_sorted) - 2)):
        ring_colors.append('#4fa3ff')  # lighter blue for additional rings

    # Draw circles and triangles
    tri_cap = min(48, max(3, int(spike_count)))
    # triangle size encodes amplitude entropy (bounded for readability)
    if amplitude_entropy_bits is None:
        tri_size = 24
    else:
        # Map entropy in [0, ~6] → size [18, 60]
        e = float(np.clip(amplitude_entropy_bits, 0.0, 6.0))
        tri_size = 18 + (60 - 18) * (e / 6.0)
    for idx, (re, color) in enumerate(zip(ring_effective, ring_colors)):
        ang = np.linspace(0, 2 * np.pi, 512)
        cx = re * np.cos(ang)
        cy = re * np.sin(ang)
        cz = np.full_like(cx, 0.15 + 0.3 * idx)
        # ring thickness encodes uncertainty (CI half-width relative to mean frac)
        lw = 2.0
        if ci_halfwidths is not None and idx < len(ci_halfwidths):
            ci_hw = max(0.0, float(ci_halfwidths[idx]))
            base = max(1e-6, fracs[idx])
            rel = np.clip(ci_hw / base, 0.0, 1.0)
            lw = 2.0 + 6.0 * rel
        ax.plot3D(cx, cy, cz, color=color, lw=lw, alpha=0.9)
        # Triangles along circle
        tri_angles = np.linspace(0, 2 * np.pi, tri_cap, endpoint=False)
        tx = re * np.cos(tri_angles)
        ty = re * np.sin(tri_angles)
        tz = np.full_like(tx, 0.15 + 0.3 * idx)
        ax.scatter(tx, ty, tz, marker='^', s=tri_size, color=color, edgecolors='black', linewidths=0.3, alpha=0.9)
        # Annotate ring with τ and fraction
        try:
            tau_val = float(taus_sorted[idx])
            frac_val = float(fracs[idx])
            ax.text(re * 1.05, 0.0, 0.15 + 0.3 * idx,
                    f"τ={tau_val:g}, f={frac_val:.2f}",
                    color=color, fontsize=8)
        except Exception:
            pass

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z (schematic power/SNR)')
    # Legend/encoding key
    enc = (
        "Rings: τ bands; radius ∝ mean fraction; thickness ∝ 95% CI half-width\n"
        "Triangles: spike markers; size ∝ amplitude entropy\n"
        "Spiral z: √t concentration + SNR contrast"
    )
    ax.text2D(0.02, 0.02, enc, transform=ax.transAxes, fontsize=8, color='#333333')
    ax.view_init(elev=24, azim=45)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

