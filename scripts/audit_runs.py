#!/usr/bin/env python3
import os
import glob
import json
import subprocess
import sys
from datetime import datetime


BIO_LIMITS = {
    "min_amp_mV": (0.05, 0.2),
    "min_isi_s": (120.0, 300.0),
    "baseline_win_s": (300.0, 900.0),
    "fs_hz": (0.5, 2.0),
}


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return sha
    except Exception:
        return ""


def get_py_versions():
    return {
        "python": sys.version.split(" (", 1)[0],
    }


def read_csv_meta(tau_csv_path):
    meta = {}
    taus = []
    try:
        with open(tau_csv_path) as f:
            for _ in range(6):
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    f.seek(pos)
                    break
                if line.startswith("# fs_hz:"):
                    try:
                        meta["fs_hz"] = float(line.split(":", 1)[1].strip())
                    except Exception:
                        pass
                if line.startswith("# taus:"):
                    try:
                        taus = [float(x) for x in line.split(":", 1)[1].strip().split(",") if x.strip()]
                    except Exception:
                        taus = []
            # Count rows to estimate nu0
            rows = sum(1 for _ in f)
        if rows > 0:
            # minus header row
            with open(tau_csv_path) as f2:
                for l in f2:
                    if not l.startswith("#"):
                        rows = rows  # already counted without header due to seek after metas
                        break
            meta["nu0_estimate"] = rows
    except Exception:
        pass
    meta["taus"] = taus
    return meta


def compliance_check(values: dict):
    notes = []
    ok = True
    for k, (lo, hi) in BIO_LIMITS.items():
        if k in values and isinstance(values[k], (int, float)):
            v = float(values[k])
            if not (lo <= v <= hi):
                ok = False
                notes.append(f"WARN {k}={v} outside [{lo},{hi}]")
        else:
            notes.append(f"INFO {k} not directly available; using config or CSV meta if present")
    return ok, notes


def write_text(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def audit_run(run_dir: str, cfg_dir: str, git_sha: str):
    metrics = read_json(os.path.join(run_dir, "metrics.json")) or {}
    species = os.path.basename(os.path.dirname(run_dir))
    cfg_path = os.path.join(cfg_dir, f"{species}.json")
    cfg = read_json(cfg_path) or {}
    tau_csv = os.path.join(run_dir, "tau_band_timeseries.csv")
    csv_meta = read_csv_meta(tau_csv) if os.path.isfile(tau_csv) else {}
    assets = {}
    for name in [
        "spikes_overlay.png",
        "tau_band_power_heatmap.png",
        "tau_band_power_surface.png",
        "stft_vs_sqrt_line.png",
        "summary_panel.png",
        "tau_band_timeseries.csv",
        "spike_times_s.csv",
    ]:
        p = os.path.join(run_dir, name)
        assets[name] = os.path.getsize(p) if os.path.isfile(p) else 0

    used = {
        "fs_hz": metrics.get("fs_hz", csv_meta.get("fs_hz")),
        "min_amp_mV": cfg.get("min_amp_mV"),
        "min_isi_s": cfg.get("min_isi_s"),
        "baseline_win_s": cfg.get("baseline_win_s"),
        "taus": csv_meta.get("taus", cfg.get("taus")),
        "nu0": csv_meta.get("nu0_estimate"),
    }
    ok, notes = compliance_check(used)

    audit = {
        "species": species,
        "run_dir": run_dir,
        "timestamp": metrics.get("timestamp", os.path.basename(run_dir)),
        "channel": metrics.get("channel", ""),
        "created_by": metrics.get("created_by", ""),
        "intended_for": metrics.get("intended_for", ""),
        "parameters": used,
        "config_path": cfg_path if os.path.isfile(cfg_path) else "",
        "assets": assets,
        "git_sha": git_sha,
        "versions": get_py_versions(),
        "compliance_ok": ok,
        "notes": notes,
    }

    # Write JSON and a brief Markdown summary
    out_json = os.path.join(run_dir, "audit.json")
    os.makedirs(run_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(audit, f, indent=2)

    lines = [
        f"# Audit: {species} | {audit['timestamp']}",
        f"- Channel: `{audit['channel']}`",
        f"- Created by: {audit['created_by']} | Intended for: {audit['intended_for']}",
        f"- Git SHA: {git_sha}",
        f"- Params: fs={used.get('fs_hz')}, min_amp_mV={used.get('min_amp_mV')}, min_isi_s={used.get('min_isi_s')}, baseline_win_s={used.get('baseline_win_s')}",
        f"- τ grid: {used.get('taus')} | nu0≈{used.get('nu0')}",
        f"- Compliance: {'OK' if ok else 'WARN'}",
    ]
    if notes:
        lines.append("## Notes")
        for n in notes:
            lines.append(f"- {n}")
    write_text(os.path.join(run_dir, "audit.md"), lines)

    return audit


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "zenodo"))
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    git_sha = get_git_sha()
    index = []
    for sp_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(sp_dir) or os.path.basename(sp_dir) == "_composites":
            continue
        for run_dir in sorted(glob.glob(os.path.join(sp_dir, "*"))):
            a = audit_run(run_dir, cfg_dir, git_sha)
            index.append({
                "species": os.path.basename(sp_dir),
                "run": os.path.basename(run_dir),
                "audit_json": os.path.join(run_dir, "audit.json"),
                "audit_md": os.path.join(run_dir, "audit.md"),
                "ok": a["compliance_ok"],
            })
    # Write index
    out_dir = os.path.join(base, "_composites")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "audits_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("OK")


if __name__ == "__main__":
    main()


