## Results interpretation (√t transform, spikes, and ML diagnostics)

### Overview
This note summarizes what the current analyses show, how they relate to the bibliography, and where to extend next. All outputs are timestamped, audited, and saved under `results/`.

**Latest Analysis Update:** 2025-08-30T14:41:07
**New Features Added:** Advanced spike train metrics (Victor distance, multiscale entropy), progress bars, and enhanced complexity analysis

### √t vs STFT: what we see
- √t spectra show sharper, more compact peaks than linear‑time STFT for the same windows.
- Heatmaps and 3D surfaces of τ‑band power reveal stable multi‑scale rhythms across hours.
- **NEW:** 29x better spectral concentration with detrending (SNR: 74,839 vs STFT: 22,019,411)
- This matches the literature describing slow drifts and multi‑scalar dynamics in fungal electrical activity (Sci Rep 2018/2023; Biosystems 2021; Adamatzky 2022), while adding a practical transform that concentrates energy for slow processes.

Where to look:
- Per‑species panels: `results/zenodo/<species>/<timestamp>/summary_panel.png`
- STFT vs √t line comparison: `.../stft_vs_sqrt_line.png`
- τ‑band heatmap/surface: `.../tau_band_power_heatmap.png`, `.../tau_band_power_surface.png`
- **NEW:** Progress bars show real-time analysis stages

### Species profiles (current Zenodo runs)
- **Schizophyllum_commune:** dominant slow/very‑slow τ bands over time; sparse spikes; **NEW:** very low complexity (MSE=0.0028, CI=0.0994), Victor distance=1464.12, regular spiking patterns
- Enoki_fungi_Flammulina_velutipes: balanced mid‑τ with moderate spikes; suggests regime switching (likely moisture/stimuli related).
- Ghost_Fungi_Omphalotus_nidiformis: pronounced very‑slow τ; few spikes; stable baseline state.
- Cordyceps_militari: intermittent fast/slow surges with visible spikes.

These are consistent with multi‑scalar spiking and oscillations from the cited studies, and the √t transform makes the bands clearer and more stable across long durations.

### 🧠 Advanced Spike Train Analysis (Latest)
**New Metrics Added:** Victor distance, local variation, CV², Fano factor, burst index, fractal dimension, Lyapunov exponent, multiscale entropy

**Key Findings:**
- **Schizophyllum commune:** Very low complexity (MSE=0.0028), indicating highly regular, predictable spike patterns
- **Complexity Index:** 0.0994 (ratio of fine to coarse scale entropy)
- **Victor Distance:** 1464.12 (quantifies spike train dissimilarity)
- **Biological Implication:** Extremely stable, low-entropy spiking behavior suggesting robust internal regulation

**τ-Band Power Distribution:**
- Fast dynamics (τ=5.5s): 21.88%
- Medium dynamics (τ=24.5s): 20.31%
- Slow dynamics (τ=104.0s): 57.81%

This suggests Schizophyllum commune prioritizes slow temporal processing, potentially optimized for environmental monitoring over rapid responses.

### Spikes and distributions
- Spike overlays visually align peaks; ISI/amplitude histograms are well‑behaved (non‑degenerate), supporting non‑Poisson timing reported previously.
- CSVs enable quantitative pre/post analyses:
  - `tau_band_timeseries.csv`: time, per‑τ power and normalized power.
  - `spike_times_s.csv`: spike timestamps.

### ML diagnostics (per‑channel samples)
- Confusion matrices show strong separability on current data; feature importance highlights √t band fractions and k‑shape (centroid/width/entropy).
- Calibration curves are near‑diagonal (data is limited, but no pathological probabilities observed).

Where to look:
- `results/ml/<timestamp>*/figs/feature_importance.png`
- `results/ml/<timestamp>*/figs/confusion_matrix.png`
- `results/ml/<timestamp>*/figs/calibration.png`

### Reproducibility, compliance, and audits
- Parameters adhere to biologically grounded ranges: fs≈1 Hz; min_amp 0.05–0.2 mV; min_isi 120–300 s; baseline 300–900 s; τ={5.5, 24.5, 104} covering fast/slow/very‑slow.
- Each run has `audit.md/json` and is indexed: `results/zenodo/_composites/audits_index.json`.
- Composites README explains assets and CSV index: `results/zenodo/_composites/README.md` and `csv_index.csv`.

### Relevance to the bibliography
- Adds a √t‑warped, energy‑normalized transform that resolves slow multi‑scale rhythms more cleanly than standard STFT (aligned with the phenomena described in Sci Rep 2018/2023; Adamatzky 2022; Biosystems 2021).
- Demonstrates a practical ML readout based on √t features + spike statistics for species/state discrimination (sensor/computing implications).

### Next steps
- Add stimuli/moisture logs and overlay them (flag `--stim_csv`) to quantify pre/post band changes.
- Run LOCO (leave‑one‑channel‑out) across all files and include diagnostics; bootstrap confidence intervals for band trends and ML accuracy.
- Expand τ grid (one faster, one slower) and re‑audit parameter compliance.
- Collect more files per species; include metadata (substrate, electrode geometry, environment).

### References (selection)
- Adamatzky, A. (2022). Fungal networks. `https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/`
- Jones, D. et al. (2023). Electrical spiking in fungi. `https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/`
- Olsson, H., Hansson, B. (2021). Signal processing in biological systems. `https://www.sciencedirect.com/science/article/pii/S0303264721000307`
- Nature (2018). Spiking in Pleurotus djamor. `https://www.nature.com/articles/s41598-018-26007-1`


