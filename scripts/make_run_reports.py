#!/usr/bin/env python3
import os
import glob
import json


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    s = float(n)
    i = 0
    while s >= 1024 and i < len(units) - 1:
        s /= 1024.0
        i += 1
    return f"{s:.1f} {units[i]}"


def infer_observations(spike_count: int, band_fracs: dict[str, float]) -> list[str]:
    obs = []
    # Band dominance
    if band_fracs:
        items = sorted(((float(k), float(v)) for k, v in band_fracs.items()), key=lambda x: x[1], reverse=True)
        top_tau, top_frac = items[0]
        obs.append(f"Dominant τ ≈ {top_tau:g} (time-in-band {top_frac*100:.1f}%).")
        if len(items) >= 2:
            second_tau, second_frac = items[1]
            if top_frac >= second_frac + 0.2:
                obs.append("Strong single-band dominance.")
            elif top_frac - second_frac < 0.1:
                obs.append("Mixed bands, two-band competition.")
    # Spiking level
    if spike_count is not None:
        if spike_count >= 50:
            obs.append("High spike activity.")
        elif spike_count >= 10:
            obs.append("Moderate spike activity.")
        else:
            obs.append("Sparse spikes.")
    return obs


def make_report(run_dir: str) -> str | None:
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path) as f:
        m = json.load(f)
    species = os.path.basename(os.path.dirname(run_dir))
    timestamp = m.get('timestamp', os.path.basename(run_dir))
    channel = m.get('channel', '')
    spike_count = m.get('spike_count', 0)
    band_fracs: dict[str, float] = m.get('band_fractions', {})

    # Collect assets
    assets = []
    for name in [
        'spikes_overlay.png',
        'tau_band_power_heatmap.png',
        'tau_band_power_surface.png',
        'stft_vs_sqrt_line.png',
        'summary_panel.png',
        'tau_band_timeseries.csv',
        'spike_times_s.csv',
    ]:
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            assets.append((name, human_size(os.path.getsize(p))))

    # Observations
    observations = infer_observations(spike_count, band_fracs)

    # Build markdown
    lines = []
    lines.append(f"# Report: {species} | {timestamp}")
    lines.append("")
    lines.append(f"- Channel: `{channel}`")
    lines.append(f"- Spike count: {spike_count}")
    if band_fracs:
        lines.append("- Band fractions (time-in-band):")
        for k, v in sorted(band_fracs.items(), key=lambda kv: float(kv[0])):
            try:
                kf = float(k)
            except Exception:
                kf = k
            lines.append(f"  - τ={kf}: {float(v)*100:.1f}%")
    lines.append("")
    if observations:
        lines.append("## Observations")
        for s in observations:
            lines.append(f"- {s}")
        lines.append("")
    if assets:
        lines.append("## Assets")
        for name, sz in assets:
            lines.append(f"- {name} ({sz})")
        lines.append("")
    out_path = os.path.join(run_dir, 'report.md')
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    return out_path


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'zenodo'))
    for species_dir in sorted(glob.glob(os.path.join(base, '*'))):
        if not os.path.isdir(species_dir) or os.path.basename(species_dir) == '_composites':
            continue
        for run_dir in sorted(glob.glob(os.path.join(species_dir, '*'))):
            make_report(run_dir)
    print("OK")


if __name__ == '__main__':
    main()


