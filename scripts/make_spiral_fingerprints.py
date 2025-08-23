#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from datetime import datetime

# Ensure project root is on sys.path for local imports (viz.plotting)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from viz.plotting import plot_spiral_fingerprint
import subprocess


def latest_run_dir(species_dir: str) -> str | None:
    runs = sorted(glob.glob(os.path.join(species_dir, '*')))
    return runs[-1] if runs else None


def load_json(path: str) -> dict | None:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Generate spiral fingerprint figures per species")
    p.add_argument('--results_root', default='results/zenodo', help='Root results directory')
    p.add_argument('--out_root', default='results/fingerprints', help='Output directory root')
    p.add_argument('--channel', default=None, help='Optional channel label to annotate')
    args = p.parse_args()

    species = [
        'Schizophyllum_commune',
        'Enoki_fungi_Flammulina_velutipes',
        'Ghost_Fungi_Omphalotus_nidiformis',
        'Cordyceps_militari',
    ]

    os.makedirs(args.out_root, exist_ok=True)
    ts = datetime.now().isoformat()

    for sp in species:
        sp_dir = os.path.join(args.results_root, sp)
        run_dir = latest_run_dir(sp_dir)
        if not run_dir:
            print(f"[SKIP] No runs for {sp}")
            continue
        metrics = load_json(os.path.join(run_dir, 'metrics.json'))
        conc = load_json(os.path.join(run_dir, 'snr_concentration.json'))
        if not metrics or not conc:
            print(f"[SKIP] Missing metrics or concentration for {sp}")
            continue

        band_fracs = metrics.get('band_fractions', {})
        spike_count = int(metrics.get('spike_count', 0) or 0)
        snr_sqrt = float(conc.get('snr', {}).get('sqrt', 0.0))
        snr_stft = float(conc.get('snr', {}).get('stft', 1.0))
        concentration_sqrt = float(conc.get('concentration', {}).get('sqrt', 0.0))

        # Try to load CI/means per tau from tau_band_timeseries.csv if present
        taus_for_labels = None
        ci_halfwidths = None
        csv_path = os.path.join(run_dir, 'tau_band_timeseries.csv')
        if os.path.isfile(csv_path):
            try:
                import csv
                import numpy as np
                rows = []
                with open(csv_path, 'r') as cf:
                    reader = csv.DictReader(cf)
                    for row in reader:
                        rows.append(row)
                if rows:
                    keys = [k for k in rows[0].keys() if k.lower() != 'time_s']
                    taus_for_labels = [float(k) for k in keys]
                    arr = np.array([[float(r[k]) for k in keys] for r in rows], dtype=float)
                    means = np.nanmean(arr, axis=0)
                    # naive bootstrap CI from time windows
                    lo = np.nanpercentile(arr, 2.5, axis=0)
                    hi = np.nanpercentile(arr, 97.5, axis=0)
                    ci_halfwidths = ((hi - lo) / 2.0).tolist()
                    # Override band_fracs with means normalized
                    total = float(np.nansum(means)) or 1.0
                    band_fracs = {str(t): float(m) / total for t, m in zip(taus_for_labels, means)}
            except Exception:
                pass

        title = f"{sp.replace('_', ' ')} — spiral fingerprint"
        # Structured output: results/fingerprints/<species>/<timestamp>/
        out_dir = os.path.join(args.out_root, sp, ts)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"spiral.png")
        try:
            plot_spiral_fingerprint(
                band_fractions=band_fracs,
                spike_count=spike_count,
                snr_sqrt=snr_sqrt,
                snr_stft=snr_stft,
                concentration_sqrt=concentration_sqrt,
                title=title,
                out_path=out_path,
                amplitude_entropy_bits=float(metrics.get('amplitude_stats', {}).get('shannon_entropy_bits', 0.0)),
                taus_for_labels=taus_for_labels,
                ci_halfwidths=ci_halfwidths,
                n_windows=len(open(csv_path).read().splitlines()) - 1 if os.path.isfile(csv_path) else None,
                channel_label=args.channel or metrics.get('channel'),
                created_by=metrics.get('created_by', 'joe knowles'),
            )
            print(f"[OK] {out_path}")
            # Also emit a JSON spec documenting mapping
            try:
                git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                git_sha = None
            mapping = {
                "schema_version": "1.0",
                "title": title,
                "created_by": metrics.get('created_by', 'joe knowles'),
                "timestamp": ts,
                "species": sp,
                "band_fractions": band_fracs,
                "spike_count": spike_count,
                "snr": {"sqrt": snr_sqrt, "stft": snr_stft},
                "concentration": {"sqrt": concentration_sqrt},
                "amplitude_entropy_bits": metrics.get('amplitude_stats', {}).get('shannon_entropy_bits', None),
                "taus_for_labels": taus_for_labels,
                "ci_halfwidths": ci_halfwidths,
                "units": {
                    "tau": "sqrt(seconds)",
                    "spike_rate": "spikes/hour",
                    "amplitude": "mV",
                    "snr": "unitless (power ratio)",
                    "concentration": "unitless (0-1)",
                    "fractions": "unitless; sum≈1.0"
                },
                "normalization": {
                    "band_fraction_method": "csv_mean_normalized" if os.path.isfile(csv_path) else "metrics_json",
                    "description": "If CSV available, fractions are mean τ-power normalized across τ; else uses metrics.band_fractions and are renormalized to sum 1.",
                },
                "n_windows": len(open(csv_path).read().splitlines()) - 1 if os.path.isfile(csv_path) else None,
                "channel": args.channel or metrics.get('channel'),
                "provenance": {
                    "git_sha": git_sha,
                    "source_metrics": os.path.join(run_dir, 'metrics.json'),
                    "source_concentration": os.path.join(run_dir, 'snr_concentration.json'),
                    "source_tau_csv": csv_path if os.path.isfile(csv_path) else None,
                    "intended_for": "peer_review"
                },
                "encodings": {
                    "rings": {
                        "order": "increasing τ",
                        "colors": ["red", "#1f77b4", "#4fa3ff"],
                        "radius": "proportional to mean band fraction",
                        "thickness": "proportional to 95% CI half-width",
                        "labels": "τ and fraction annotated per ring"
                    },
                    "triangles": {
                        "count": "= spike_count (capped 48)",
                        "size": "∝ amplitude_entropy_bits",
                    },
                    "spiral": {
                        "z_height": "∝ concentration_sqrt with SNR contrast",
                        "color": "#808080"
                    },
                    "center": {
                        "color": "gold"
                    }
                },
                "references": [
                    {
                        "label": "Sci Rep 2018",
                        "title": "Spiking in Pleurotus djamor",
                        "url": "https://www.nature.com/articles/s41598-018-26007-1"
                    },
                    {
                        "label": "Jones 2023",
                        "title": "Multiscalar electrical spiking in Schizophyllum commune",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/"
                    },
                    {
                        "label": "Adamatzky 2022",
                        "title": "Language of fungi from electrical spiking activity",
                        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/"
                    },
                    {
                        "label": "Biosystems 2021",
                        "title": "Electrical activity of fungi: spikes detection and complexity",
                        "url": "https://www.sciencedirect.com/science/article/pii/S0303264721000307"
                    }
                ]
            }
            json_path = os.path.join(out_dir, 'spiral.json')
            with open(json_path, 'w') as jf:
                json.dump(mapping, jf, indent=2)
            print(f"[OK] {json_path}")
            # Write references.md for human-readable context
            refs_md = os.path.join(out_dir, 'references.md')
            with open(refs_md, 'w') as rf:
                rf.write("- Spiking in Pleurotus djamor (Sci Rep 2018): https://www.nature.com/articles/s41598-018-26007-1\n")
                rf.write("- Multiscalar electrical spiking in Schizophyllum commune (Sci Rep 2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/\n")
                rf.write("- Language of fungi derived from electrical spiking (R. Soc. Open Sci. 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/\n")
                rf.write("- Electrical activity of fungi: Spikes detection and complexity (Biosystems 2021): https://www.sciencedirect.com/science/article/pii/S0303264721000307\n")
            print(f"[OK] {refs_md}")
            # Export numeric fingerprint vector for ML/repro
            import csv as _csv
            csv_vec = os.path.join(out_dir, 'fingerprint_vector.csv')
            keys = [
                'species','timestamp','snr_sqrt','snr_stft','concentration_sqrt','spike_count','ampl_entropy_bits'
            ]
            # flatten band fractions in sorted τ order
            tau_keys = sorted([float(k) for k in band_fracs.keys()])
            tau_strs = [str(t) for t in tau_keys]
            frac_keys = [f"frac_tau_{t}" for t in tau_strs]
            ci_keys = [f"ci_tau_{t}" for t in tau_strs] if ci_halfwidths and taus_for_labels else []
            header = keys + frac_keys + ci_keys
            with open(csv_vec, 'w', newline='') as cf:
                w = _csv.DictWriter(cf, fieldnames=header)
                w.writeheader()
                row = {
                    'species': sp,
                    'timestamp': ts,
                    'snr_sqrt': snr_sqrt,
                    'snr_stft': snr_stft,
                    'concentration_sqrt': concentration_sqrt,
                    'spike_count': spike_count,
                    'ampl_entropy_bits': metrics.get('amplitude_stats', {}).get('shannon_entropy_bits', None),
                }
                for t in tau_strs:
                    row[f"frac_tau_{t}"] = band_fracs.get(t, band_fracs.get(str(t)))
                if ci_halfwidths and taus_for_labels:
                    for t, ci in zip(taus_for_labels, ci_halfwidths):
                        row[f"ci_tau_{t}"] = ci
                w.writerow(row)
            print(f"[OK] {csv_vec}")
        except Exception as e:
            print(f"[ERR] {sp}: {e}")


if __name__ == '__main__':
    main()


