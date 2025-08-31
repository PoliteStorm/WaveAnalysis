#!/usr/bin/env python3
import os, json, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser(description='Plot psi-sweep summary PNG from summary.json list(s)')
    ap.add_argument('--inputs', nargs='+', required=True, help='One or more results/psi_sweep/*/summary.json files')
    ap.add_argument('--out_png', required=True)
    args = ap.parse_args()

    # Collect metrics across inputs (average per window type)
    agg = {}
    for path in args.inputs:
        if not os.path.exists(path):
            continue
        arr = json.load(open(path, 'r'))
        for entry in arr:
            metrics = entry.get('metrics', {})
            for wname, m in metrics.items():
                s = agg.setdefault(wname, {'conc': [], 'snr': []})
                s['conc'].append(float(m.get('avg_concentration', 0.0)))
                s['snr'].append(float(m.get('avg_snr', 0.0)))

    if not agg:
        print('No data to plot')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    win_names = sorted(agg.keys())
    conc_means = [np.mean(agg[w]['conc']) for w in win_names]
    snr_means = [np.mean(agg[w]['snr']) for w in win_names]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), dpi=140)
    ax[0].bar(win_names, conc_means, color='slateblue')
    ax[0].set_title('Spectral concentration (avg)')
    ax[0].set_ylabel('max(power)/sum(power)')
    ax[0].set_ylim(0, max(1.0, max(conc_means)*1.2))
    ax[1].bar(win_names, snr_means, color='teal')
    ax[1].set_title('SNR vs background (avg)')
    ax[1].set_ylabel('ratio')
    ax[1].set_ylim(0, max(1.0, max(snr_means)*1.2))
    for a in ax:
        a.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png)
    plt.close(fig)
    print(args.out_png)

if __name__ == '__main__':
    main()


