#!/usr/bin/env python3
import argparse
import glob
import os
from datetime import datetime


SPECIES = [
    'Schizophyllum_commune',
    'Enoki_fungi_Flammulina_velutipes',
    'Ghost_Fungi_Omphalotus_nidiformis',
    'Cordyceps_militari',
]


def latest_dir(path_glob: str) -> str | None:
    items = sorted(glob.glob(path_glob))
    return items[-1] if items else None


def main():
    ap = argparse.ArgumentParser(description='Create per-species index HTML linking latest visuals and results')
    ap.add_argument('--results_root', default='results/zenodo')
    ap.add_argument('--finger_root', default='results/fingerprints')
    args = ap.parse_args()

    for sp in SPECIES:
        run_dir = latest_dir(os.path.join(args.results_root, sp, '*'))
        fing_dir = latest_dir(os.path.join(args.finger_root, sp, '*'))
        if not fing_dir:
            print(f"[SKIP] No fingerprints for {sp}")
            continue
        # Collect assets
        spiral_png = os.path.join(fing_dir, 'spiral.png')
        sphere_html = os.path.join(fing_dir, 'sphere.html')
        spiral_json = os.path.join(fing_dir, 'spiral.json')
        sphere_json = os.path.join(fing_dir, 'sphere.json')
        fp_csv = os.path.join(fing_dir, 'fingerprint_vector.csv')
        refs_md = os.path.join(fing_dir, 'references.md')

        heatmap = surface = summary = stft = spikes = spike_ci = None
        if run_dir:
            heatmap = os.path.join(run_dir, 'tau_band_power_heatmap.png')
            surface = os.path.join(run_dir, 'tau_band_power_surface.png')
            summary = os.path.join(run_dir, 'summary_panel.png')
            stft = os.path.join(run_dir, 'stft_vs_sqrt_line.png')
            spikes = os.path.join(run_dir, 'spikes_overlay.png')
            spike_ci = os.path.join(run_dir, 'spike_rate_ci.png')

        # Build HTML
        ts = datetime.now().isoformat()
        html = [
            '<html><head><meta charset="utf-8"/>',
            f'<title>{sp} — Visual Index</title>',
            '<style>body{font-family:system-ui,Arial,sans-serif;margin:16px} .row{display:flex;gap:12px;flex-wrap:wrap} img{max-width:48%;height:auto;border:1px solid #ddd} .card{margin-bottom:16px}</style>',
            '</head><body>',
            f'<h2>{sp.replace("_"," ")} — Visual Index</h2>',
            f'<p>Created by joe knowles — {ts}</p>',
            '<div class="card"><h3>Fingerprints</h3>',
            f'<p><a href="{os.path.relpath(spiral_png, fing_dir)}">Spiral PNG</a> · <a href="{os.path.relpath(spiral_json, fing_dir)}">spiral.json</a> · <a href="{os.path.relpath(fp_csv, fing_dir)}">fingerprint_vector.csv</a></p>' if os.path.isfile(spiral_png) else '<p>Spiral: not found</p>',
            f'<p><a href="{os.path.relpath(sphere_html, fing_dir)}">Sphere HTML</a> · <a href="{os.path.relpath(sphere_json, fing_dir)}">sphere.json</a></p>' if os.path.isfile(sphere_html) else '<p>Sphere: not found</p>',
            f'<p><a href="{os.path.relpath(refs_md, fing_dir)}">references.md</a></p>' if os.path.isfile(refs_md) else '',
            '</div>',
        ]
        if run_dir:
            html += ['<div class="card"><h3>Latest run images</h3><div class="row">']
            for img in [summary, heatmap, surface, stft, spikes, spike_ci]:
                if img and os.path.isfile(img):
                    html.append(f'<a href="{os.path.relpath(img, fing_dir)}" target="_blank"><img src="{os.path.relpath(img, fing_dir)}"/></a>')
            html += ['</div></div>']
        html += ['</body></html>']
        out_path = os.path.join(fing_dir, 'index.html')
        with open(out_path, 'w') as f:
            f.write('\n'.join(html))
        print(f"[OK] {out_path}")


if __name__ == '__main__':
    main()


