#!/usr/bin/env python3
"""
io_loaders.py

Data-driven loaders for IO-control experiments.
- Stimulus CSV auto-discovery
- √t band NPZ cache auto-discovery

Author: Joe Knowles
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _most_recent(paths: List[str]) -> Optional[str]:
	if not paths:
		return None
	paths_sorted = sorted(paths, key=lambda p: os.path.getmtime(p))
	return paths_sorted[-1]


def load_stimulus_df(root: str | Path = ".") -> pd.DataFrame:
	"""Load stimulus timeline from deliverables/stimulus_protocol/**.csv if available.

	Expected columns (best-effort): time_s plus any stim_* columns.
	If none found, returns a synthetic multi-channel protocol for 6h@1Hz.
	"""
	root = Path(root)
	candidates = glob.glob(str(root / "deliverables" / "stimulus_protocol" / "**" / "*.csv"), recursive=True)
	csv_path = _most_recent(candidates)
	if csv_path:
		try:
			df = pd.read_csv(csv_path)
			# Heuristic: ensure time_s exists
			if "time_s" not in df.columns:
				# try to infer from sample index
				df = df.copy()
				df.insert(0, "time_s", np.arange(len(df), dtype=float))
			# rename stimulus columns to stim_ prefix if needed
			cols = {}
			for c in df.columns:
				if c != "time_s" and not c.startswith("stim_"):
					cols[c] = f"stim_{c}"
			if cols:
				df = df.rename(columns=cols)
			return df
		except Exception:
			pass
	# Fallback synthetic
	t = np.arange(0, 6*3600, 1.0)
	return pd.DataFrame({
		"time_s": t,
		"stim_moisture": (np.sin(2*np.pi*t/1800)>0).astype(float),
		"stim_light": (np.sin(2*np.pi*t/2400 + 1.2)>0).astype(float),
		"stim_mech": (np.sin(2*np.pi*t/3600 + 0.6)>0).astype(float),
		"stim_elec": (np.sin(2*np.pi*t/900 + 0.3)>0).astype(float),
	})


def load_bands_df(root: str | Path = ".") -> pd.DataFrame:
	"""Load √t band powers from cache/features/*.npz if available.

	Tries keys: time_s, band_fast/mid/slow. If missing, tries generic arrays.
	Falls back to synthetic bands if no cache found.
	"""
	root = Path(root)
	candidates = glob.glob(str(root / "cache" / "features" / "*.npz"))
	npz_path = _most_recent(candidates)
	if npz_path:
		try:
			data = np.load(npz_path, allow_pickle=True)
			keys = set(data.files)
			if {"time_s", "band_fast", "band_mid", "band_slow"}.issubset(keys):
				return pd.DataFrame({
					"time_s": data["time_s"],
					"band_fast": data["band_fast"],
					"band_mid": data["band_mid"],
					"band_slow": data["band_slow"],
				})
			# Attempt to infer from generic arrays
			arrs = [data[k] for k in data.files if data[k].ndim == 1 and data[k].size > 100]
			if arrs:
				T = min(a.size for a in arrs)
				t = np.arange(T)
				A = np.vstack([a[:T] for a in arrs[:3]])
				return pd.DataFrame({
					"time_s": t.astype(float),
					"band_fast": A[0],
					"band_mid": A[1] if A.shape[0] > 1 else A[0],
					"band_slow": A[2] if A.shape[0] > 2 else A[0],
				})
		except Exception:
			pass
	# Fallback synthetic
	t = np.arange(0, 6*3600, 1.0)
	rng = np.random.default_rng(123)
	noise = lambda: rng.normal(0, 0.1, size=t.size)
	return pd.DataFrame({
		"time_s": t,
		"band_fast": 0.5 + 0.1*np.sin(2*np.pi*t/900) + noise(),
		"band_mid": 0.6 + 0.1*np.sin(2*np.pi*t/2400 + 0.8) + noise(),
		"band_slow": 0.7 + 0.1*np.sin(2*np.pi*t/3600 + 1.1) + noise(),
	})

