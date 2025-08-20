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
                                 figsize: tuple | None = None) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available; cannot create time-series plot")
    _ensure_parent_dir(out_path)
    if figsize is None:
        figsize = (11, 3.8)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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


