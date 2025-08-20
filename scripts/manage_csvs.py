#!/usr/bin/env python3
import os
import glob
import csv


def index_csvs(root: str) -> list[dict]:
    rows = []
    for sp_dir in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(sp_dir) or os.path.basename(sp_dir) == '_composites':
            continue
        for run_dir in sorted(glob.glob(os.path.join(sp_dir, '*'))):
            tau_csv = os.path.join(run_dir, 'tau_band_timeseries.csv')
            spike_csv = os.path.join(run_dir, 'spike_times_s.csv')
            rows.append({
                'species': os.path.basename(sp_dir),
                'run': os.path.basename(run_dir),
                'tau_csv': tau_csv if os.path.isfile(tau_csv) else '',
                'spike_csv': spike_csv if os.path.isfile(spike_csv) else '',
            })
    return rows


def write_index(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['species', 'run', 'tau_csv', 'spike_csv'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'zenodo'))
    rows = index_csvs(root)
    write_index(rows, os.path.join(root, '_composites', 'csv_index.csv'))
    print('OK')


if __name__ == '__main__':
    main()


