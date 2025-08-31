## Findings so far (simple summary)
- √t bands highlight slow/very‑slow rhythms that standard STFT smears over long durations.
- Features from √t (band fractions, k‑centroid/width/entropy) combine with spike stats to differentiate species/states.
- On limited data, per‑channel CV (leave‑one‑channel‑out) is preferable to per‑file CV.

What’s new in this iteration (peer‑review oriented)
- CI bands: We compute and export τ‑power and spike‑rate confidence intervals per species. See `results/ci_summaries/<species>/<timestamp>/` and per‑run `spike_rate_ci.png`.
- √t vs STFT table: Cross‑species numeric SNR and spectral concentration in `results/summaries/<timestamp>/snr_concentration_table.*`.
- Replot‑from‑CSV: τ heatmaps/surfaces can be regenerated from `tau_band_timeseries.csv` without recomputation (scripts/replot_from_csv.py).
- Fingerprints:
  - Spiral (static): `results/fingerprints/<species>/<timestamp>/spiral.png` with explicit mapping in `spiral.json` and numeric `fingerprint_vector.csv`.
  - Spherical (interactive): `results/fingerprints/<species>/<timestamp>/sphere.html` + `sphere.json`; hover shows τ and fraction; CI shown as ring thickness; √t concentration + SNR contrast shown as surface bump.
- ML: LOCO CV (leave‑one‑channel‑out), calibration curves, Brier score, and permutation importance are produced under `results/ml/<timestamp>*/`.

How it recognizes biological patterns
- Spikes: baseline‑subtracted thresholding → rate, ISI, amplitude structure.
- Rhythms: √t‑warped windows stabilize long‑time oscillations, making peaks compact in k.
- Combining both gives a multi‑scale “signature.”

Planned improvements
- Parameter sweeps for τ grid; Morlet windows; u‑domain detrend; confidence intervals via bootstrapping; caching all intermediates; skipping low‑power windows to save CPU.

Code references (reproducibility)
- √t transform and STFT: `prove_transform.py` (sqrt_time_transform_fft, stft_fft)
- Metrics and plots: `analyze_metrics.py`, `viz/plotting.py`
- ML pipeline and diagnostics: `ml_pipeline.py`
- CI summaries: `scripts/make_ci_summaries.py`
- Fingerprints: `scripts/make_spiral_fingerprints.py`, `scripts/make_spherical_fingerprints.py`
- Replot utility: `scripts/replot_from_csv.py`

Result locations (latest)
- Per‑species runs: `results/zenodo/<species>/<timestamp>/`
- Fingerprints: `results/fingerprints/<species>/<timestamp>/`
- Cross‑species summaries: `results/summaries/<timestamp>/`

Bibliography (starter)
- Adamatzky, A. (2022). Fungal networks. `https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/`
- Jones, D. et al. (2023). Electrical spiking in fungi. `https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/`
- Olsson, H., Hansson, B. (2021). Signal processing in biological systems. `https://www.sciencedirect.com/science/article/pii/S0303264721000307`
- Nature (2018): Rhythms and complexity in living systems. `https://www.nature.com/articles/s41598-018-26007-1`
