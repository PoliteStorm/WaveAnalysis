# 01_Workflow (low‑RAM, Chromebook‑safe)

## Setup
```
python3 -m venv .venv
./.venv/bin/pip install -U pip scikit-learn scipy matplotlib pillow
```

## √t band analysis on Zenodo txt
```
python3 prove_transform.py --mode zenodo --file 'data/zenodo_5790768/Schizophyllum commune.txt' --taus 5.5,24.5,104 --nu0 128 --plot_sqrt --plot_stft_compare
```

## Spike/statistics + √t fractions (timestamped JSON)
```
python3 analyze_metrics.py --file 'data/zenodo_5790768/Schizophyllum commune.txt' \
  --fs 1 --min_amp_mV 0.1 --min_isi_s 120 --baseline_win_s 600 \
  --plot --export_csv --quicklook --config configs/Schizophyllum_commune.json
```

## ML per-file, progress on, timestamped outputs
Example (Schizophyllum):
```
./.venv/bin/python ml_pipeline.py \
  --data_dir data/zenodo_5790768 \
  --only_file "Schizophyllum commune.txt" \
  --nu0 5 --n_u 160 --taus 5.5,24.5,104 \
  --lofo --out_dir results/ml --json_out "" --progress --plot_ml
```
Repeat for:
- "Enoki fungi Flammulina velutipes.txt"
- "Ghost Fungi Omphalotus nidiformis.txt"
- "Cordyceps militari.txt"

## Combined cached pass (all files/channels)
```
./.venv/bin/python ml_pipeline.py \
  --data_dir data/zenodo_5790768 \
  --nu0 5 --n_u 160 --taus 5.5,24.5,104 \
  --lofo --out_dir results/ml --json_out "" --progress --plot_ml
```

## Notes
- All runs print [STATUS] lines for files, channels, folds, and writes.
- Outputs: timestamped results.json and references.md under results/.
