---
title: "A √t‑Warped Wave Transform Reveals Multi‑Scale Electrical Rhythms in Fungal Networks"
author:
  - "Joe Knowles"
date: "2025-08-20"
keywords: [fungal electrophysiology, sqrt‑time transform, wave analysis, spike statistics, machine learning, biosensing, biocomputing]
geometry: margin=1in
fontsize: 11pt
---

# Abstract
Fungal electrical activity exhibits spikes and slow oscillatory modulations over seconds to hours. We introduce a √t‑warped wave transform that concentrates long‑time structure into compact spectral peaks, improving time‑frequency localization for sublinear temporal dynamics. On open fungal datasets (fs≈1 Hz) the method yields sharper spectra than STFT, stable τ‑band trajectories, and species‑specific multi‑scale “signatures.” Coupled with spike statistics and a lightweight ML pipeline, we obtain reproducible diagnostics under leave‑one‑file‑out validation. All analyses are timestamped, audited, and designed for low‑RAM devices.

# 1. Introduction
Electrophysiological studies of fungi (Adamatzky 2022; Jones et al. 2023; Sci Rep 2018; Biosystems 2021) report spiking and multi‑scale rhythms whose time scales span orders of magnitude. Linear‑time analyses often blur slowly evolving structure. We propose a √t‑warped transform tailored to sublinear temporal evolution, revealing stable band trajectories across hours and providing a practical readout for sensing and biocomputing.

# 2. Related work
- Adamatzky (2022) surveyed fungal network dynamics and biocomputing perspectives.
- Jones et al. (2023) and Sci Rep (2018) detail spiking and multi‑scalar rhythms across species; Adamatzky (2022, arXiv:2203.11198) extends cross‑species comparisons.
- Slow bioelectric methods in plants/fungi (Volkov) motivate robust baselining and drift handling.
- Advanced time–frequency methods—synchrosqueezing (Daubechies), reassignment (Auger & Flandrin), Hilbert–Huang (Huang)—improve concentration for non‑stationary signals. Multitaper (Thomson) provides robust spectra/SNR baselines; Mallat’s wavelet/scattering theory guides window choices.
- Spike train metrics and multiscale entropy complement Shannon entropy for slow rhythms.

# 3. Methods
## 3.1 √t‑Warped Wave Transform
We analyze voltage \(V(t)\) with a windowed transform in \(u = \sqrt{t}\):

\[
W(k,\tau; u_0) 
= \int_{0}^{\infty} V(t)\, \psi\!\left(\frac{\sqrt{t} - u_0}{\tau}\right) e^{-i k \sqrt{t}} \, dt.
\tag{1}
\]

Substituting \(u = \sqrt{t}\) (so \(dt = 2u\,du\)) gives:

\[
W(k,\tau; u_0)
= \int_{0}^{\infty} 2u\, V(u^2)\, \psi\!\left(\frac{u - u_0}{\tau}\right) e^{-i k u} \, du.
\tag{2}
\]

Implementation: energy‑normalized window; u‑grid rFFT; scan \(u_0\); optional Morlet/detrend (ablation).

## 3.2 STFT baseline
Gaussian STFT in t with \(t_0 = u_0^2\), \(\sigma_t = 2 u_0 \tau\).

## 3.3 Spike detection and statistics
Moving‑average baseline (300–900 s), thresholds 0.05–0.2 mV, min ISI 120–300 s; rate, ISI/amplitude entropy/skewness/kurtosis.

## 3.4 Data and processing
Zenodo (fs=1 Hz). τ={5.5, 24.5, 104}. Quicklook \(\nu_0\)≈16; full \(\nu_0\)≈64. float32 + caching.

## 3.5 Machine learning
√t bands + spike stats; LOFO/LOCO CV; feature importance, confusion, calibration.

## 3.6 Reproducibility
Timestamped, audited runs; composites README, CSV and audit indexes.

# 4. Results
## 4.1 √t vs STFT (Schizophyllum commune)
Figure 1 shows a multi‑panel summary for a representative run: the √t τ‑band heatmap and surface, spike overlay, and STFT‑vs‑√t spectral comparison for a matched window. √t spectra exhibit narrower peaks and higher SNR, and τ‑band trajectories remain stable across hours.

Figure 1A. Summary panel (√t transform, spikes, comparison)

![Schizophyllum commune summary](figs/Schizophyllum_commune_summary.png){ width=90% }

Figure 1B. τ‑band heatmap and surface (√t domain)

![τ‑band heatmap](figs/Schizophyllum_commune_heatmap.png){ width=49% } ![τ‑band surface](figs/Schizophyllum_commune_surface.png){ width=49% }

Figure 1C. Spikes overlay (baseline‑subtracted overlay) and STFT vs √t spectral line (matched window)

![Spikes overlay](figs/Schizophyllum_commune_spikes.png){ width=85% }

![STFT vs √t](figs/Schizophyllum_commune_stft_vs_sqrt.png){ width=70% }

Figure 1D. ISI and amplitude histograms

![ISI histogram](figs/Schizophyllum_commune_hist_isi.png){ width=49% } ![Amplitude histogram](figs/Schizophyllum_commune_hist_amp.png){ width=49% }

## 4.2 Species‑level profiles
Qualitatively, we observe distinct τ‑band “signatures”:
- Schizophyllum commune: slow/very‑slow dominance; sparse spikes.
- Flammulina velutipes (Enoki): balanced mid‑τ with moderate spikes.
- Omphalotus nidiformis (Ghost): pronounced very‑slow τ; few spikes.
- Cordyceps militaris: intermittent fast/slow surges with visible spikes.
These align with multi‑scalar rhythms described by Jones et al. (2023) and Sci Rep (2018) and become clearer under √t warping.

## 4.3 ML diagnostics
Feature importance highlights √t band fractions and k‑shape features; confusion matrices show strong separability on current data; calibration curves are near‑diagonal. (Figures in the ML folder accompany the peer‑review package.)

## 4.4 Cross‑species SNR and spectral concentration
We summarize √t versus STFT performance across species using a numeric table built from the latest runs. For each species we report SNR(√t), SNR(STFT), spectral concentration(√t), concentration(STFT), and the √t/STFT ratios. The table is exported in CSV/JSON/Markdown under `results/summaries/<timestamp>/snr_concentration_table.*` and is included in the peer‑review package. These values quantify the concentration and contrast improvements visible in Figure 1 and species‑level profiles.

## 4.5 Spiral fingerprint supplements (exploratory)
To aid fast between‑species comparison, we provide a supplementary “spiral fingerprint” per species that encodes: ring radius ∝ mean τ‑band fraction (fast→slow from inner→outer), ring thickness ∝ 95% CI half‑width, triangle size ∝ spike amplitude entropy, and spiral height ∝ √t concentration with SNR contrast. Each figure is accompanied by a JSON spec and a numeric feature CSV at `results/fingerprints/<species>/<timestamp>/`. This schematic is reproducible and documented, and is presented alongside the standard quantitative plots (τ‑heatmaps, CI bands, STFT vs √t lines) for scientific interpretation.

# 5. Discussion
### 5.1 How √t enhances prior findings
- Concentration and stability across hours complement Adamatzky’s network‑level observations and the multi‑scalar rhythms in Sci Rep 2018/Jones 2023.
- √t provides a compact, reproducible readout for sensing; band dominance is a candidate logic state (biocomputing framing).

### 5.2 Ablation and alternatives (future work)
- Windows: Gaussian vs Morlet; add reassignment/synchrosqueezing ablations for concentration.
- Detrend: u‑domain detrend on/off; quantify impact on low‑k leakage.
- Spectra: add multitaper SNR/concentration baselines.
- Adaptive lenses: EMD/HHT for comparison on slow drifts.

# 6. Conclusion
The √t‑warped wave transform provides a tidy, computationally efficient view of fungal dynamics across scales, enabling robust spectral and spike‑based features for ML. It corroborates and sharpens the multi‑scale phenomena reported in the literature and offers a practical basis for fungal sensing/computing.

# References
- Adamatzky, A. (2022). Fungal networks. https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/
- Jones, D. et al. (2023). Electrical spiking in fungi. https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/
- Olsson, H., Hansson, B. (2021). Signal processing in biological systems. https://www.sciencedirect.com/science/article/pii/S0303264721000307
- Sci Rep (2018). Spiking in Pleurotus djamor. https://www.nature.com/articles/s41598-018-26007-1
