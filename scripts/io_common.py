#!/usr/bin/env python3
"""
io_common.py

Shared utilities for IO-control experiments.
- Timestamped run directories
- Author & git metadata capture
- Safe filesystem helpers

Author: Joe Knowles
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

AUTHOR_NAME = "Joe Knowles"


def iso_timestamp() -> str:
	"""Return an ISO-like timestamp safe for directory names."""
	return _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def get_git_hash(repo_root: str | Path = ".") -> str:
	"""Return short git hash if available, else 'unknown'."""
	try:
		out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root))
		return out.decode().strip()
	except Exception:
		return "unknown"


def ensure_dir(p: str | Path) -> Path:
	"""Create directory if missing (including parents)."""
	path = Path(p)
	path.mkdir(parents=True, exist_ok=True)
	return path


@dataclasses.dataclass
class RunContext:
	base_dir: Path
	stamp: str
	path: Path
	metadata: Dict[str, Any]


def create_run_context(base_results_dir: str | Path = "results/io_control") -> RunContext:
	stamp = iso_timestamp()
	base = ensure_dir(base_results_dir)
	path = ensure_dir(base / stamp)
	meta = {
		"author": AUTHOR_NAME,
		"timestamp": stamp,
		"git_hash": get_git_hash(Path.cwd()),
	}
	(meta_path := path / "run_metadata.json").write_text(json.dumps(meta, indent=2))
	return RunContext(base_dir=base, stamp=stamp, path=path, metadata=meta)


def save_json(obj: Dict[str, Any], out_path: str | Path) -> None:
	Path(out_path).write_text(json.dumps(obj, indent=2))


def save_text(text: str, out_path: str | Path) -> None:
	Path(out_path).write_text(text)


def save_figure(fig, out_path: str | Path) -> None:
	"""Save a matplotlib figure with tight layout as PNG."""
	out = Path(out_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out, dpi=200)
	fig.clf()
