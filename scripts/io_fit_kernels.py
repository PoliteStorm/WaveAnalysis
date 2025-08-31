#!/usr/bin/env python3
"""
io_fit_kernels.py

Fit stimulus→response models:
- FIR kernels on √t band powers (fast/mid/slow)
- Poisson GLM on spike counts

Saves timestamped outputs under results/io_control/<stamp>/

Author: Joe Knowles
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from io_common import create_run_context, save_json, save_figure
from io_loaders import load_stimulus_df, load_bands_df


def load_stimulus_timeseries() -> pd.DataFrame:
	"""Return df with columns: time_s, stim_moisture, stim_light, stim_mech, stim_elec."""
	# TODO: replace with real protocol import
	t = np.arange(0, 6*3600, 1.0)  # 6 hours @ 1 Hz
	df = pd.DataFrame({
		"time_s": t,
		"stim_moisture": (np.sin(2*np.pi*t/1800)>0).astype(float),
		"stim_light": (np.sin(2*np.pi*t/2400 + 1.2)>0).astype(float),
		"stim_mech": (np.sin(2*np.pi*t/3600 + 0.6)>0).astype(float),
		"stim_elec": (np.sin(2*np.pi*t/900 + 0.3)>0).astype(float),
	})
	return df


def load_sqrt_bandpowers() -> pd.DataFrame:
	"""Return df with columns: time_s, band_fast, band_mid, band_slow."""
	# TODO: replace with actual √t transform outputs
	t = np.arange(0, 6*3600, 1.0)
	rng = np.random.default_rng(123)
	noise = lambda: rng.normal(0, 0.1, size=t.size)
	df = pd.DataFrame({
		"time_s": t,
		"band_fast": 0.5 + 0.1*np.sin(2*np.pi*t/900) + noise(),
		"band_mid": 0.6 + 0.1*np.sin(2*np.pi*t/2400 + 0.8) + noise(),
		"band_slow": 0.7 + 0.1*np.sin(2*np.pi*t/3600 + 1.1) + noise(),
	})
	return df


def load_spike_counts() -> pd.DataFrame:
	"""Return df with columns: time_s, spikes (counts per second)."""
	t = np.arange(0, 6*3600, 1.0)
	rng = np.random.default_rng(321)
	lam = 0.02 + 0.02*np.sin(2*np.pi*t/1800)
	spikes = rng.poisson(lam)
	return pd.DataFrame({"time_s": t, "spikes": spikes})


def build_design_matrix(stim_df: pd.DataFrame, lags_s: int = 3600) -> Tuple[pd.DataFrame, List[str]]:
	"""Build lagged stimulus design up to lags_s seconds back (FIR)."""
	cols = [c for c in stim_df.columns if c != "time_s"]
	design = pd.DataFrame({"time_s": stim_df["time_s"]})
	feat_names: List[str] = []
	for c in cols:
		for k in range(lags_s+1):
			name = f"{c}_lag{k}"
			design[name] = stim_df[c].shift(k, fill_value=0.0)
			feat_names.append(name)
	return design, feat_names


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> np.ndarray:
	"""Closed-form ridge: (X^T X + αI)^{-1} X^T y."""
	XtX = X.T @ X
	n = XtX.shape[0]
	coef = np.linalg.solve(XtX + alpha*np.eye(n), X.T @ y)
	return coef


def poisson_glm_newton(X: np.ndarray, y: np.ndarray, iters: int = 50, lam: float = 1e-3) -> np.ndarray:
	"""Simple Poisson GLM with Newton steps and L2 regularization."""
	w = np.zeros(X.shape[1])
	for _ in range(iters):
		eta = X @ w
		mu = np.exp(eta)
		W = mu  # diag
		XTW = X.T * W
		H = XTW @ X + lam*np.eye(X.shape[1])
		g = X.T @ (y - mu) - lam*w
		delta = np.linalg.solve(H, g)
		w += delta
	return w


def main() -> None:
	run = create_run_context()
	out = run.path
	# Load
	stim = load_stimulus_timeseries()
	bands = load_sqrt_bandpowers()
	spk = load_spike_counts()
	# Join on time
	df = stim.merge(bands, on="time_s").merge(spk, on="time_s")
	# Design matrix
	design, feat_names = build_design_matrix(stim)
	D = df.merge(design, on="time_s")
	# Targets
	Y = D[["band_fast", "band_mid", "band_slow"]].to_numpy()
	# Remove time_s, bands, spikes from features
	X = D.drop(columns=["time_s", "band_fast", "band_mid", "band_slow", "spikes"]).to_numpy()
	# Fit FIR kernels (ridge)
	coefs = {}
	for i, target in enumerate(["band_fast", "band_mid", "band_slow"]):
		w = fit_ridge(X, Y[:, i], alpha=10.0)
		coefs[target] = w.tolist()
	# Plot example kernel sums per stimulus
	stim_cols = [c for c in stim.columns if c != "time_s"]
	fig, ax = plt.subplots(figsize=(8, 4))
	for target, w in coefs.items():
		# Sum lags per stimulus for a quick view
		ws = np.array(w).reshape(len(stim_cols), -1)
		sums = ws.sum(axis=1)
		ax.plot(sums, marker='o', label=target)
	ax.set_xticks(range(len(stim_cols)))
	ax.set_xticklabels(stim_cols, rotation=20)
	ax.set_title("FIR kernel sums by stimulus")
	ax.legend()
	save_figure(fig, out / "fir_kernel_summaries.png")
	plt.close(fig)
	# Poisson GLM for spikes
	w_glm = poisson_glm_newton(X, D["spikes"].to_numpy(), iters=30, lam=1e-3)
	# Save
	save_json({
		"author": run.metadata["author"],
		"git": run.metadata["git_hash"],
		"features": feat_names,
		"fir_coefs": coefs,
		"poisson_glm": w_glm.tolist(),
	}, out / "io_models.json")
	print(f"Saved results to {out}")


if __name__ == "__main__":
	main()
