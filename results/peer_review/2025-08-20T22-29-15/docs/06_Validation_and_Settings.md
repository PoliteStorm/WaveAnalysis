## Biologically grounded defaults (rationale)
- Sampling rate (fs): 1 Hz is typical for slow fungal rhythms; increase if expecting fast spikes.
- Spike thresholds: 0.05–0.2 mV (species‑dependent); baseline window 300–900 s; minimum ISI 120–300 s.
  - Rationale: prevents counting baseline drift and avoids refractory‑like double counts.
- √t grid: τ ∈ {5.5, 24.5, 104}; ν0 ≈ 5–16; n_u ≈ 160–512; float32.
  - Rationale: spans fast/slow/very‑slow bands; safe on low‑RAM hardware.
- Window energy normalization: removes τ bias so bands are comparable.
- Skip low‑power windows: reduces CPU and variance from noise‑only segments.

## Validation plan
- Synthetic controls: inject √t‑locked sinusoids + noise; show spectral concentration and SNR gains vs STFT.
- Real data checks: correlate τ‑band fractions with spike rates and ISI stats; verify stability across u0.
- Cross‑validation: LOCO (leave‑one‑channel‑out) preferred; LOFO used with multiple files per class.
- Robustness: mild parameter sweeps around thresholds; effects should be monotone, not brittle.

## Anti‑overfitting/forced‑param safeguards
- Fix defaults but report all settings with outputs; timestamp and credit.
- Use caching only for speed; never reuse labels or leak folds.
- Track seeds and versions; re‑run with held‑out channels to confirm.

## Reporting
- Always emit JSON with: created_by, timestamp, intended_for, file/channel, parameters, metrics, references.
