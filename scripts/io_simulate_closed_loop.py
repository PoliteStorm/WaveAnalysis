#!/usr/bin/env python3
"""
io_simulate_closed_loop.py

Closed-loop simulation using learned FIR kernels to control √t band powers.

Author: Joe Knowles
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from io_common import create_run_context, save_json, save_figure


def convolve_fir(stim_mat: np.ndarray, w: np.ndarray) -> np.ndarray:
	"""Convolve multi-input stimulus (T x M) with FIR kernels (M x L) -> predicted y (T,)."""
	T, M = stim_mat.shape
	L = w.shape[1]
	Y = np.zeros(T)
	for m in range(M):
		Y += np.convolve(stim_mat[:, m], w[m, :], mode="full")[:T]
	return Y


def synthesize_stimuli(T: int, M: int, rng: np.random.Generator) -> np.ndarray:
	"""Simple pulse trains per channel."""
	X = np.zeros((T, M))
	for m in range(M):
		period = rng.integers(600, 1800)
		X[::period, m] = 1.0
		X[:, m] = np.minimum(1.0, np.convolve(X[:, m], np.ones(60), mode="same"))
	return X


def main(models_path: str | Path = "results/io_control/latest/io_models.json") -> None:
	run = create_run_context("results/io_control/simulations")
	out = run.path
	# Load models
	models_file = Path(models_path)
	if models_file.name == "latest":
		# user can symlink/point to latest; here we assume direct path
		pass
	models = json.loads(Path(models_path).read_text())
	stim_names = [n for n in models["features"] if n.endswith("_lag0")]
	M = len(stim_names)
	# Build FIR kernels matrix (per target)
	targets = ["band_fast", "band_mid", "band_slow"]
	W: Dict[str, np.ndarray] = {}
	L = int(len(models["fir_coefs"][targets[0]])/M)
	for t in targets:
		w = np.array(models["fir_coefs"][t])
		W[t] = w.reshape(M, L)
	# Simulate
	T = 3*3600
	rng = np.random.default_rng(42)
	X = synthesize_stimuli(T, M, rng)
	pred = {t: convolve_fir(X, W[t]) for t in targets}
	# Simple controller: scale stimuli to push mid band to target
	target_level = np.percentile(pred["band_mid"], 75)
	for k in range(3):
		scale = 1.0 + 0.1*k
		pred_k = {t: convolve_fir(X*scale, W[t]) for t in targets}
		fig, ax = plt.subplots(figsize=(9, 3))
		ax.plot(pred["band_mid"], label="mid (baseline)", alpha=0.6)
		ax.plot(pred_k["band_mid"], label=f"mid (scaled x{scale:.2f})")
		ax.axhline(target_level, color="k", ls="--", lw=1, label="target")
		ax.set_title("Closed-loop proxy: target mid-band level vs stimulus scale")
		ax.legend(ncol=3, fontsize=8)
		save_figure(fig, out / f"closed_loop_scale_{k}.png")
		plt.close(fig)
	# Save summary
	save_json({
		"author": run.metadata["author"],
		"git": run.metadata["git_hash"],
		"stim_features": stim_names,
		"targets": targets,
		"kernel_length": L,
		"sim_duration_s": T,
	}, out / "simulation_summary.json")
	print(f"Saved simulation outputs to {out}")


if __name__ == "__main__":
	main()
