## √t Wave Transform: What it is, why it helps, what we found, how to improve

- Definition: W(k, τ; u0) = ∫₀^∞ V(t) ψ(√t/τ − u0) e^(−ik√t) dt.
- With u = √t, dt = 2u du: W(k, τ; u0) = ∫₀^∞ 2u V(u²) ψ(u/τ − u0) e^(−iku) du.
- ψ: Gaussian or Morlet. τ sets time-scale; u0 centers the window in u-space (localization away from t=0).

Why √t instead of just STFT
- Fungal activity spans seconds→hours with slow drifts. √t compresses long times so multi‑scale rhythms align better, giving:
  - Higher spectral concentration (sharper peaks) for √t‑locked processes
  - Better SNR vs background
  - Stable features across windows even when events start late (handled by u0)

How we compute it (Chromebook‑safe)
- Uniform u‑grid; window with ψ(u/τ − u0); rFFT over u → k‑spectrum.
- Window energy normalization removes τ bias.
- Defaults: n_u≈160–512, ν0≈5–16, float32, cached features to disk.
- Baseline: STFT in t to compare.

Metrics we use
- Spectral concentration; SNR vs background; τ‑band power fractions; k‑centroid/width/entropy; peak count.
- Spike metrics: rate, amplitude/ISI entropy, skewness, kurtosis.

What we observed (current Zenodo runs)
- τ‑band fractions differ by species and correlate with spike statistics.
- √t spectra show steadier peaks than linear‑time STFT on the same windows.
- Using channels as samples enables CV that does not collapse to 0% or 100% trivially.

Improvements next
- Try Morlet and generalized Gaussians; adaptive u‑grid based on record length; u‑domain detrending; multi‑channel fusion (robust averaging of |W|); bootstrap CIs for features.

### Psi-sweep results (Gaussian, Morlet, DOG, Bump; detrend on)
- Data: one latest recording per species; τ = [5.5, 24.5, 104], 8 u0 positions
- Metrics: avg spectral concentration (peak/sum), avg SNR vs background in k
- Findings:
  - Gaussian: consistently highest or near-highest concentration and strong SNR across species.
  - Morlet: slightly narrower peaks but lower SNR at matched settings.
  - DOG (Mexican-hat): competitive concentration; SNR typically below Gaussian.
  - Bump: occasional very high SNR (Ghost) — treat cautiously; verify against edge/normalization artifacts.
- Example averages:
  - Cordyceps: Gaussian 0.858 / 1.78e6; Morlet 0.825 / 6.03e5; DOG 0.881 / 4.88e5; Bump 0.825 / 4.14e5
  - Enoki: Gaussian 0.858 / 4.61e7; Morlet 0.807 / 4.95e5; DOG 0.852 / 5.12e5; Bump 0.855 / 8.36e6
  - Ghost: Gaussian 0.900 / 8.06e6; Morlet 0.819 / 8.53e5; DOG 0.857 / 5.01e5; Bump 0.880 / 2.56e9
  - Schizophyllum: Gaussian 0.797 / 1.29e6; Morlet 0.765 / 5.37e5; DOG 0.786 / 3.32e5; Bump 0.794 / 1.62e6

Implications:
- The √t warp drives the main gains; ψ mainly tunes the peak‑width/leakage trade‑off.
- Gaussian + detrend is a safe default for robust SNR and concentration.
- Bump’s extreme SNR cases warrant artifact checks (padding, energy norm, detrend band).
- Recommended practice: report ψ choice, include Gaussian baseline, and add bootstrap CIs for peak metrics.

Provenance: `results/psi_sweep/2025-08-31T10-57-57/summary.json`

Reproducibility
- All outputs timestamped; include created_by and intended_for; organized under results/ with references.

Visual readouts (where to look)
- τ‑band heatmaps and surfaces (√t domain): `results/zenodo/<species>/<timestamp>/tau_band_power_heatmap.png`, `tau_band_power_surface.png` (and `_replot.png` regenerated from CSV).
- STFT vs √t spectral line (matched window): `stft_vs_sqrt_line.png` per run.
- Spiral fingerprint (static): `results/fingerprints/<species>/<timestamp>/spiral.png` with explicit mapping in `spiral.json` and numeric `fingerprint_vector.csv`.
- Spherical fingerprint (interactive): `results/fingerprints/<species>/<timestamp>/sphere.html` with mapping in `sphere.json` and `references.md`.
- Cross‑species SNR and spectral concentration table: `results/summaries/<timestamp>/snr_concentration_table.{csv,json,md}`.

Select references (see also results/*/references.md)
- Adamatzky (2022) Fungal networks. `https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/`
- Jones et al. (2023) Electrophysiology of fungi. `https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/`
- Olsson & Hansson (2021) Signal processing in living systems. `https://www.sciencedirect.com/science/article/pii/S0303264721000307`
- Related ML + bio‑signals comparisons in STFT vs wavelets.
