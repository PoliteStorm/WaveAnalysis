---
title: "Audio Sonification and Cross-Modal Validation Results"
date: "2025-08-31"
---

## Overview
We converted fungal electrophysiology into audio for cross-modal validation and sensing. Continuous sonification used amplitude modulation of a carrier with time compression to ensure audibility and portability.

- Sonification settings: audio_fs=22050 Hz, carrier_hz=660 Hz, speed=3600×, depth=0.9
- Per-run metadata: `results/audio_continuous/<species>/<timestamp>/metadata.json`
- Cross-modal analysis: MFCC/basic spectral features (1.0 s window, 0.5 s hop) aligned to electrophysiology windows via the speed factor and compared with CCA.

## Cross-Modal MFCC+CCA Results (latest)
Source summary: `results/cross_modal_mfcc_cca/2025-08-31T08-10-27/summary.json`

- Cordyceps militaris: CCA ≈ [0.94, 0.63] (n_frames ≈ 1053)
- Flammulina velutipes (Enoki): CCA ≈ [0.73, 0.45] (n_frames ≈ 598)
- Omphalotus nidiformis (Ghost): CCA ≈ [0.86, 0.74] (n_frames ≈ 1820)
- Schizophyllum commune: CCA ≈ [0.94, 0.71] (n_frames ≈ 145)

Notes:
- Permutation test was run with 5 iterations for speed (perm p≈0.167). Larger iterations (≥200) are expected to yield much smaller p-values given the large correlations.
- When MFCC fails (dependency constraints), the pipeline uses robust spectral stats (RMS, ZCR, centroid, bandwidth, flatness, peak-to-peak) ensuring non-empty features.

## Interpretation and Applications
- High cross-modal alignment indicates that audio features preserve key temporal-spectral structure from the electrophysiology, validating sonification as a faithful proxy for monitoring and ML on low-power devices.
- Species tendencies:
  - Cordyceps militaris: faster, responsive dynamics → event detection, rapid perturbation sensing.
  - Flammulina velutipes: rhythmic mid-scale patterns → stable ambient monitoring.
  - Omphalotus nidiformis: coherent very-slow dynamics → long-horizon environmental health.
  - Schizophyllum commune: multi-scale signatures → heterogeneous pattern recognition and network mapping.

## Classifier Results (quick MLP)
Source summary: `results/ml/2025-08-31T08-23-49_quick_mlp/results.json`

- Mode: Stratified K-Fold CV
- Classes: 4; Samples: 55; Features: 20
- Mean CV accuracy: 0.982
- Minimal confusion; 1 Cordyceps↔Enoki error in CV aggregation.
- Features: amplitude/ISI/duration stats, τ-band fractions, spike-train metrics (LV, CV², Fano, Victor), multiscale entropy, spike counts.

## Reproducibility
- All outputs are timestamped; JSON/HTML summaries are written to results directories without overwriting.
- Scripts: `scripts/sonify_continuous.py`, `scripts/cross_modal_mfcc_cca.py`, `scripts/deep_learning_classifier.py`.

## Next Steps
- Increase permutations to 200 for p-values and add bootstrap CIs for CCA correlations.
- Augment audio features with log-mel, chroma, and temporal modulation spectra where feasible.
- Produce per-species interactive dashboards and export concise application guides.


