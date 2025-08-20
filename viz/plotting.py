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
                 aspect: str = "auto") -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create heatmap")
    _ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
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
                   stride: int = 2) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create 3D surface plot")
    _ensure_parent_dir(out_path)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(8, 5), dpi=110)
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
                                 out_path: str) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create time-series plot")
    _ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=110)
    ax.plot(t, v, lw=0.8, color='#444444')
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


