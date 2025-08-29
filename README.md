# √t-transformed fungal bioelectric fingerprints

Reproducible pipeline for analyzing long-duration fungal electrical recordings with a square-root time (√t) transform, producing τ-band fingerprints, spike statistics, and ML readouts.

## Quickstart

```bash
python3 -m pip install -r requirements.txt
```

Run analysis on a Zenodo TXT file (1 Hz sampled, columns are channels):

```bash
python3 analyze_metrics.py --file /path/to/Zenodo_file.txt --plot --export_csv --out_dir results --baselines --bootstrap_conc
```

Run ML pipeline with leave-one-file-out CV:

```bash
python3 ml_pipeline.py --data_dir /path/to/zenodo/files --lofo --out_dir results/ml --progress
```

## Outputs
- Figures and CSVs under `results/zenodo/<species>/<timestamp>/`
- Cross-species summaries under `results/summaries/`
- Confidence intervals under `results/ci_summaries/`

## Citation
See `CITATION.cff`.

## License
MIT (see `LICENSE`).
