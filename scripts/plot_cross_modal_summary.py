#!/usr/bin/env python3
import os, json, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser(description='Plot cross-modal summary PNG from batch summary.json')
    ap.add_argument('--summary_json', required=True)
    ap.add_argument('--out_png', required=True)
    args = ap.parse_args()

    if not os.path.exists(args.summary_json):
        print('Summary not found')
        return
    arr = json.load(open(args.summary_json, 'r'))
    # Extract per-run proxy correlations: use max of absolute Pearson across pairs
    labels = []
    c1 = []
    c2 = []
    for run in arr:
        name = os.path.basename(run.get('file','')).replace('.txt','')
        st = run.get('stats', {})
        vals = []
        for k, v in st.items():
            r = v.get('pearson_r')
            if r is not None and not (isinstance(r, float) and (np.isnan(r) or np.isinf(r))):
                vals.append(abs(float(r)))
        if vals:
            vals_sorted = sorted(vals, reverse=True)
            labels.append(name)
            c1.append(vals_sorted[0])
            c2.append(vals_sorted[1] if len(vals_sorted) > 1 else 0.0)

    if not labels:
        print('No stats to plot')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.8), dpi=140)
    ax.bar(x - 0.15, c1, width=0.3, label='max |r|')
    ax.bar(x + 0.15, c2, width=0.3, label='2nd max |r|')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('correlation')
    ax.set_title('Audio–signal alignment (proxy correlations)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png)
    plt.close(fig)
    print(args.out_png)

if __name__ == '__main__':
    main()


