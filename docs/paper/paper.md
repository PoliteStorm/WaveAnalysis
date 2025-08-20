---
title: "A √t-Warped Wave Transform Reveals Multi-Scale Electrical Rhythms in Fungal Networks"
author:
  - "Joe Knowles"
date: "2025-08-20"
keywords: [fungal electrophysiology, sqrt-time transform, wave analysis, machine learning, biosensing]
geome	ry: margin=1in
fontsize: 11pt
---

# Abstract
We introduce a √t-warped wave transform that concentrates slow multi-scale rhythms observed in fungal electrical activity. Compared to standard STFT, the method yields sharper spectra and stable τ-band trajectories across hours. Coupled with spike statistics and a lightweight ML pipeline, we demonstrate species-level separability and reproducible metrics on open datasets. The pipeline is Chromebook-safe, cached, and fully audited.

# Introduction
Fungal electrical activity exhibits spikes and slow oscillatory modulations across seconds to hours. Standard linear-time analyses often smear long-time structure. We propose a √t-warped transform tailored to sublinear time evolution.

# Methods
## √t Transform
Definition with u=√t substitution, energy-normalized windows, FFT evaluation, and u0-centered windows for localization.

## Spike Detection and Metrics
Baseline subtraction, thresholding, minimum ISI; information-theoretic and distributional metrics.

## Datasets and Processing
Zenodo data (fs=1 Hz); τ={5.5, 24.5, 104}; ν0 tuned for quicklook vs full runs; caching and float32 to reduce RAM.

## Machine Learning
Features per τ (normalized power, k-centroid/width/entropy, peak counts) plus spike statistics; LOFO/LOCO CV; diagnostics.

# Results
## √t vs STFT
Sharper peaks and higher SNR in √t compared to STFT for matched windows.

![Schizophyllum commune summary](figs/Schizophyllum_commune_summary.png){ width=90% }

![τ-band heatmap](figs/Schizophyllum_commune_heatmap.png){ width=48% }
![τ-band surface](figs/Schizophyllum_commune_surface.png){ width=48% }

![ISI histogram](figs/Schizophyllum_commune_hist_isi.png){ width=48% }
![Amplitude histogram](figs/Schizophyllum_commune_hist_amp.png){ width=48% }

## ML diagnostics
![Feature importance](figs/ml_feature_importance.png){ width=60% }
![Confusion matrix](figs/ml_confusion_matrix.png){ width=48% }
![Calibration](figs/ml_calibration.png){ width=48% }

# Discussion
√t unlocks multi-scale structure; distinct species τ-band signatures; implications for biosensing and biocomputing.

# Reproducibility and Audits
All runs timestamped and audited; parameters comply with biologically grounded ranges. See results/zenodo/_composites for indexes.

# Conclusion
The √t transform provides a robust, efficient view of fungal dynamics across scales, enabling ML-based sensing. Future work: more data, site diversity, stimuli overlays, bootstrap CIs.

# References
- Adamatzky, A. (2022). Fungal networks.
- Jones, D. et al. (2023). Electrical spiking in fungi.
- Olsson, H., Hansson, B. (2021). Signal processing in biological systems.
- Sci Rep (2018). Spiking in Pleurotus djamor.
