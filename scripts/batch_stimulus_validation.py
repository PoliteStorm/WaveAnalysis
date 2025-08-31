#!/usr/bin/env python3
"""
Batch-run stimulus-response validation using analyze_metrics on multiple files.
"""

import os
import subprocess
import datetime as _dt


def run_one(data_file: str, stim_csv: str, out_root: str):
    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    out_dir = os.path.join(out_root, f"batch_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        'python3', 'analyze_metrics.py',
        '--file', data_file,
        '--stimulus_csv', stim_csv,
        '--stimulus_window', '600',
        '--quicklook'
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=False)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Batch stimulus validation')
    ap.add_argument('--data_dir', default='data/zenodo_5790768')
    ap.add_argument('--stim_csv', required=True)
    ap.add_argument('--out_root', default='results/zenodo/_composites')
    args = ap.parse_args()

    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
             if f.endswith('.txt') and not f.startswith('__MACOSX')]
    for fpath in files:
        run_one(fpath, args.stim_csv, args.out_root)


if __name__ == '__main__':
    main()


