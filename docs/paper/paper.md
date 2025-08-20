---
title: "A √t-Warped Wave Transform Reveals Multi-Scale Electrical Rhythms in Fungal Networks"
author:
  - "Joe Knowles"
date: "2025-08-20"
keywords: [fungal electrophysiology, sqrt-time transform, wave analysis, machine learning, biosensing]
geometry: margin=1in
fontsize: 11pt
---

# Abstract
We introduce a √t-warped wave transform that concentrates slow multi-scale rhythms observed in fungal electrical activity. Compared to standard STFT, the method yields sharper spectra and stable τ-band trajectories across hours. Coupled with spike statistics and a lightweight ML pipeline, we demonstrate species-level separability and reproducible metrics on open datasets. The pipeline is Chromebook-safe, cached, and fully audited.

# Introduction
Fungal electrical activity exhibits spikes and slow oscillatory modulations across seconds to hours. Prior studies (Adamatzky 2022; Jones et al. 2023; Sci Rep 2018; Biosystems 2021) report multi-scalar patterns with long-tail dynamics. Linear-time analyses can smear long-time structure. We propose a √t-warped transform tailored to sublinear time evolution, providing a unified view across scales.

# Methods
## √t Transform (derivation)
We define
\[ W(k,\tau; u_0) = \int_0^\infty V(t)\, \psi\!\left(\frac{\sqrt{t}-u_0}{\tau}\right) e^{-ik\sqrt{t}}\, dt. \]
Let \(u = \sqrt{t}\Rightarrow dt = 2u\,du\). Then
\[ W(k,\tau; u_0) = \int_0^\infty 2u\, V(u^2)\, \psi\!\left(\frac{u-u_0}{\tau}\right) e^{-iku}\, du. \]
We evaluate this on a uniform \(u\)-grid using rFFT. To avoid τ bias, we energy-normalize the window: \(\psi \leftarrow \psi / \|\psi\|_2\). The center \(u_0\) localizes late events.

## Baselines and STFT
For comparison, we use a Gaussian-windowed STFT in t with center \(t_0=u_0^2\) and \(\sigma_t=2u_0\tau\) (matching time spread). We compare spectra side-by-side for the same window.

## Spike Detection and Statistics
We subtract a moving-average baseline (300–900 s), detect peaks above 0.05–0.2 mV with minimum ISI 120–300 s, and compute rate, ISI/amplitude entropy, skewness, and kurtosis.

## Datasets and Processing
Zenodo data (fs=1 Hz). τ={5.5, 24.5, 104} (very fast/slow/very slow). ν0 set to 16–64 depending on quicklook vs full runs. Caching and float32 minimize RAM.

## Machine Learning
Features per τ (normalized power, k-centroid/width/entropy, peak counts) plus spike statistics; LOFO/LOCO CV; diagnostics (feature importance, confusion matrix, calibration).

# Results
## √t vs STFT
Sharper peaks and higher SNR in √t compared to STFT for matched windows (consistent with slow processes reported in the literature).

![Schizophyllum commune summary](figs/Schizophyllum_commune_summary.png){ width=90% }

![τ-band heatmap](figs/Schizophyllum_commune_heatmap.png){ width=48% } ![τ-band surface](figs/Schizophyllum_commune_surface.png){ width=48% }

![Spikes overlay](figs/Schizophyllum_commune_spikes.png){ width=90% }

![STFT vs √t](figs/Schizophyllum_commune_stft_vs_sqrt.png){ width=70% }

![ISI histogram](figs/Schizophyllum_commune_hist_isi.png){ width=48% } ![Amplitude histogram](figs/Schizophyllum_commune_hist_amp.png){ width=48% }

(Analogous figures for other species appear in the Supplement.)

## ML diagnostics
![Feature importance](figs/ml_feature_importance.png){ width=60% }
![Confusion matrix](figs/ml_confusion_matrix.png){ width=48% } ![Calibration](figs/ml_calibration.png){ width=48% }

# Discussion
√t unlocks multi-scale structure: species exhibit distinct τ-band signatures (Schizophyllum: slow/very-slow dominance; Enoki: balanced mid-τ; Omphalotus: very-slow; Cordyceps: intermittent fast/slow surges). This aligns with Adamatzky’s observations on fungal network rhythms and with spiking reports in Sci Rep 2018/2023 and Biosystems 2021. The transform provides a practical readout for sensing and potential biocomputing (logic via band thresholds; reservoir-like multi-band states).

# Reproducibility and Audits
Runs are timestamped and audited (`audit.md/json`), with parameters within biologically grounded ranges (fs≈1 Hz; min_amp 0.05–0.2 mV; ISI 120–300 s; baseline 300–900 s; τ bands spanning fast/slow/very slow). See `results/zenodo/_composites` for species gallery, CSV index, and audit index.

# Conclusion
The √t transform provides a robust, efficient view of fungal dynamics across scales, enabling ML-based sensing. Future work: more data and sites, stimuli/moisture overlays, LOCO across instruments, bootstrap CIs for band trends and accuracy.

# References
- Adamatzky, A. (2022). Fungal networks. https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/
- Jones, D. et al. (2023). Electrical spiking in fungi. https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/
- Olsson, H., Hansson, B. (2021). Signal processing in biological systems. https://www.sciencedirect.com/science/article/pii/S0303264721000307
- Sci Rep (2018). Spiking in Pleurotus djamor. https://www.nature.com/articles/s41598-018-26007-1
