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

Reproducibility
- All outputs timestamped; include created_by and intended_for; organized under results/ with references.

Select references (see also results/*/references.md)
- Adamatzky (2022) Fungal networks. `https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/`
- Jones et al. (2023) Electrophysiology of fungi. `https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/`
- Olsson & Hansson (2021) Signal processing in living systems. `https://www.sciencedirect.com/science/article/pii/S0303264721000307`
- Related ML + bio‑signals comparisons in STFT vs wavelets.
