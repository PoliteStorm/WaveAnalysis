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


def latest_dir(pattern: str) -> str | None:
    items = sorted(glob.glob(pattern))
    return items[-1] if items else None


def main():
    ap = argparse.ArgumentParser(description='Create a top-level results index')
    ap.add_argument('--out', default='results/index.html')
    args = ap.parse_args()

    ts = datetime.now().isoformat()
    html = [
        '<html><head><meta charset="utf-8"/>',
        '<title>WaveAnalysis — Results Index</title>',
        '<style>body{font-family:system-ui,Arial,sans-serif;margin:16px} ul{line-height:1.6} h2{margin-top:1.2em}</style>',
        '</head><body>',
        '<h1>WaveAnalysis — Results Index</h1>',
        f'<p>Created by joe knowles — {ts}</p>',
        '<h2>Per-species fingerprint indexes</h2>',
        '<ul>'
    ]
    for sp in SPECIES:
        latest_fp = latest_dir(os.path.join('results', 'fingerprints', sp, '*'))
        if latest_fp and os.path.isfile(os.path.join(latest_fp, 'index.html')):
            rel = os.path.relpath(os.path.join(latest_fp, 'index.html'), os.path.dirname(args.out))
            html.append(f'<li>{sp.replace("_"," ")}: <a href="{rel}">{rel}</a></li>')
        else:
            html.append(f'<li>{sp.replace("_"," ")}: (no fingerprint index found)</li>')
    html.append('</ul>')

    html.append('<h2>Cross-species summaries</h2>')
    latest_sum = latest_dir(os.path.join('results', 'summaries', '*'))
    if latest_sum:
        for ext in ('csv', 'json', 'md'):
            path = os.path.join(latest_sum, f'snr_concentration_table.{ext}')
            if os.path.isfile(path):
                rel = os.path.relpath(path, os.path.dirname(args.out))
                html.append(f'<div><a href="{rel}">{rel}</a></div>')
    else:
        html.append('<p>(no summaries found)</p>')

    html.append('<h2>Audits</h2>')
    audits = sorted(glob.glob(os.path.join('results', 'audits', '*')))
    if audits:
        html.append('<ul>')
        for a in audits[-10:]:
            j = os.path.join(a, 'audit.json')
            m = os.path.join(a, 'audit.md')
            if os.path.isfile(j) or os.path.isfile(m):
                if os.path.isfile(j):
                    relj = os.path.relpath(j, os.path.dirname(args.out))
                    html.append(f'<li><a href="{relj}">{relj}</a></li>')
                if os.path.isfile(m):
                    relm = os.path.relpath(m, os.path.dirname(args.out))
                    html.append(f'<li><a href="{relm}">{relm}</a></li>')
        html.append('</ul>')
    else:
        html.append('<p>(no audits found)</p>')

    html.append('</body></html>')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(html))
    print(f"[OK] {args.out}")


if __name__ == '__main__':
    main()


