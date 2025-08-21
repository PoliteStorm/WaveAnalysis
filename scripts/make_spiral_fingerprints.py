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

        title = f"{sp.replace('_', ' ')} — spiral fingerprint"
        out_path = os.path.join(args.out_root, f"{sp}_{ts}.png")
        try:
            plot_spiral_fingerprint(
                band_fractions=band_fracs,
                spike_count=spike_count,
                snr_sqrt=snr_sqrt,
                snr_stft=snr_stft,
                concentration_sqrt=concentration_sqrt,
                amplitude_entropy_bits=float(metrics.get('amplitude_stats', {}).get('shannon_entropy_bits', 0.0)),
                title=title,
                out_path=out_path,
            )
            print(f"[OK] {out_path}")
            # Also emit a JSON spec documenting mapping
            mapping = {
                "title": title,
                "created_by": metrics.get('created_by', 'unknown'),
                "timestamp": ts,
                "species": sp,
                "band_fractions": band_fracs,
                "spike_count": spike_count,
                "snr": {"sqrt": snr_sqrt, "stft": snr_stft},
                "concentration": {"sqrt": concentration_sqrt},
                "amplitude_entropy_bits": metrics.get('amplitude_stats', {}).get('shannon_entropy_bits', None),
                "encodings": {
                    "rings": {
                        "order": "increasing τ",
                        "colors": ["red", "#1f77b4", "#4fa3ff"],
                        "radius": "proportional to band fraction",
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
                }
            }
            json_path = os.path.splitext(out_path)[0] + '.json'
            with open(json_path, 'w') as jf:
                json.dump(mapping, jf, indent=2)
            print(f"[OK] {json_path}")
        except Exception as e:
            print(f"[ERR] {sp}: {e}")


if __name__ == '__main__':
    main()


