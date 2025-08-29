Title: Concentrating multiscale fungal bioelectric rhythms with a square-root time transform enables species fingerprints and ML readouts

Authors: Joe Knowles (corresponding)
Affiliation: Independent Researcher
Contact: [email]

Abstract
We analyze long-duration fungal bioelectric recordings using a square-root time (√t) wave transform that concentrates slow, multiscale rhythms more cleanly than linear-time STFT. From √t spectra we derive τ-band fractions, k-shape statistics, and concentration/SNR diagnostics, and combine these with spike statistics to form species fingerprints and machine-learning features. On public Zenodo data (record 5790768), we observe consistent very-slow band dominance in Cordyceps and Omphalotus, a balanced slow/very-slow profile in Schizophyllum, and highly active, very-slow-dominated Enoki. √t concentration exceeds STFT by ~1.14–2.08× across species. Results align with literature reporting species-specific spiking and multihour oscillations, and provide a practical, reproducible pipeline for sensing and classification.

Keywords: fungal electrophysiology, time–frequency analysis, √t transform, species fingerprinting, machine learning

1. Introduction
Fungi exhibit electrical spikes and slow oscillations over seconds to hours, with species- and state-specific patterns. Prior studies report spiking structure and long-timescale rhythms and discuss the need for robust analysis of nonstationary bioelectric signals. We introduce a square-root time (√t) warping that compresses long windows and stabilizes slow rhythms, producing sharper spectral concentration than linear-time STFT on the same windows. We quantify τ-band activity and spikes and demonstrate cross-species differences.

2. Data
We use publicly available recordings from Zenodo (record 5790768), text files with ~1 Hz sampling of multiple differential channels per species. Example species analyzed: Schizophyllum commune, Enoki fungi Flammulina velutipes, Ghost Fungi Omphalotus nidiformis, Cordyceps militari.

3. Methods
3.1 √t transform
Define u = √t and compute W(k, τ; u0) on a uniform u-grid with window ψ(u/τ − u0) and rFFT over u. Energy normalization removes τ bias. We scan u0 over ν0 centers (typ. ν0≈5–24; examples use ν0≈17) and τ over {5.5, 24.5, 104} to capture fast/slow/very-slow rhythms. We report:
- τ-band fractions (normalized power per τ within window)
- k-shape statistics (centroid, bandwidth, peak count, entropy)
- concentration and SNR diagnostics

3.2 Spike statistics
We detrend with a long moving baseline (e.g., 600 s), detect thresholded events with refractory (e.g., min_amp 0.1 mV, min_isi 120 s), and compute amplitude/ISI distributions and entropies.

3.3 Machine learning (optional)
We treat channels as samples, aggregate per-window √t features and spike statistics, and evaluate simple classifiers with cross-validation (leave-one-file-out for small-N). Diagnostics include feature importance and calibration.

4. Results
Per-species representative runs (metrics.json):
- Schizophyllum commune (diff_1): τ-band fractions 5.5→0.1875, 24.5→0.375, 104→0.4375; spike_count=10 (sparse).
- Enoki fungi Flammulina velutipes (diff_1): 5.5→0.0625, 24.5→0.1875, 104→0.75; spike_count=1297 (high).
- Ghost Fungi Omphalotus nidiformis (diff_1): 5.5→0.0625, 24.5→0.125, 104→0.8125; spike_count=6 (very low).
- Cordyceps militari (diff_1): 5.5→0.0, 24.5→0.125, 104→0.875; spike_count=1114 (high).

Cross-species concentration and SNR (snr_concentration_table.json):
√t concentration exceeds STFT by ~1.14–2.08× across species; per-species figures show sharper √t peaks and reduced background leakage versus matched-window STFT.

5. Discussion
The observed species profiles are consistent with literature reporting species-specific spiking and slow, multihour dynamics. The √t transform provides a practical lens that stabilizes long-timescale rhythms and yields compact features for downstream ML and sensing. Differences between high-activity (Enoki, Cordyceps) and low-activity (Ghost, Schizophyllum) regimes are clear in both spikes and τ-band dominance.

6. Limitations and future work
- Benchmark against multitaper, reassignment, and synchrosqueezing under identical windows.
- Bootstrap confidence intervals for τ-fractions, spikes, and ML performance.
- Expand τ grid and include stimuli logs (moisture, temperature) for pre/post analyses.
- Increase sample size and metadata (substrate, geometry) for generalization.

7. Data and code availability
All outputs are timestamped under results/. Key entry points: results/index.html; per-run artifacts under results/zenodo/<species>/<timestamp>/; cross-species summaries under results/summaries/<timestamp>/. Data source: Zenodo record 5790768.

8. Acknowledgments
Author thanks open-source communities and AI assistants (Claude 4 for initial transform ideation; and other AI tools for drafting) for research support.

References (selection)
- Adamatzky (2022) Fungal networks. https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/
- Jones et al. (2023) Electrical spiking/info processing in fungi. https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/
- Scientific Reports (2018) Spiking rhythms in Pleurotus. https://www.nature.com/articles/s41598-018-26007-1
- Royal Society Open Science (2022) Language of fungi. https://royalsocietypublishing.org/doi/full/10.1098/rsos.211926
- Fungal Biology & Biotechnology (2023) Moisture response. https://fungalbiolbiotech.biomedcentral.com/articles/10.1186/s40694-023-00155-0
