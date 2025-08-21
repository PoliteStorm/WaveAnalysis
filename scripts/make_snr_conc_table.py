#!/usr/bin/env python3
import argparse
import glob
import json
import os
from datetime import datetime


SPECIES = [
    'Schizophyllum_commune',
    'Enoki_fungi_Flammulina_velutipes',
    'Ghost_Fungi_Omphalotus_nidiformis',
    'Cordyceps_militari',
]


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
    ap = argparse.ArgumentParser(description='Compile cross-species SNR/concentration table')
    ap.add_argument('--results_root', default='results/zenodo')
    ap.add_argument('--out_root', default='results/summaries')
    args = ap.parse_args()

    ts = datetime.now().isoformat(timespec='seconds')
    out_dir = os.path.join(args.out_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for sp in SPECIES:
        sp_dir = os.path.join(args.results_root, sp)
        run_dir = latest_run_dir(sp_dir)
        if not run_dir:
            continue
        conc_path = os.path.join(run_dir, 'snr_concentration.json')
        data = load_json(conc_path)
        if not data:
            continue
        row = {
            'species': sp,
            'run_dir': run_dir,
            'snr_sqrt': float(data.get('snr', {}).get('sqrt', 0.0)),
            'snr_stft': float(data.get('snr', {}).get('stft', 0.0)),
            'conc_sqrt': float(data.get('concentration', {}).get('sqrt', 0.0)),
            'conc_stft': float(data.get('concentration', {}).get('stft', 0.0)),
        }
        # derived ratios
        row['snr_ratio_sqrt_over_stft'] = (row['snr_sqrt'] + 1e-12) / (row['snr_stft'] + 1e-12)
        row['conc_ratio_sqrt_over_stft'] = (row['conc_sqrt'] + 1e-12) / (row['conc_stft'] + 1e-12)
        rows.append(row)

    # Write JSON
    out_json = os.path.join(out_dir, 'snr_concentration_table.json')
    payload = {
        'created_by': 'joe knowles',
        'timestamp': ts,
        'rows': rows,
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    # Write CSV
    out_csv = os.path.join(out_dir, 'snr_concentration_table.csv')
    import csv
    fields = [
        'species', 'run_dir', 'snr_sqrt', 'snr_stft', 'conc_sqrt', 'conc_stft',
        'snr_ratio_sqrt_over_stft', 'conc_ratio_sqrt_over_stft'
    ]
    with open(out_csv, 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # Write Markdown
    out_md = os.path.join(out_dir, 'snr_concentration_table.md')
    def fmt(x: float) -> str:
        return f"{x:.4g}"
    with open(out_md, 'w') as mf:
        mf.write(f"# SNR and Spectral Concentration (√t vs STFT)\n\n")
        mf.write(f"Created by joe knowles — {ts}\n\n")
        mf.write("| species | SNR(√t) | SNR(STFT) | SNR ratio √t/STFT | conc(√t) | conc(STFT) | conc ratio √t/STFT |\n")
        mf.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            mf.write(
                f"| {r['species']} | {fmt(r['snr_sqrt'])} | {fmt(r['snr_stft'])} | {fmt(r['snr_ratio_sqrt_over_stft'])} | "
                f"{fmt(r['conc_sqrt'])} | {fmt(r['conc_stft'])} | {fmt(r['conc_ratio_sqrt_over_stft'])} |\n"
            )

    print(f"[OK] Wrote {out_json}\n[OK] Wrote {out_csv}\n[OK] Wrote {out_md}")


if __name__ == '__main__':
    main()


