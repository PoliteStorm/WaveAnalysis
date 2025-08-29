Title: √t-transformed fungal bioelectric fingerprints — one page

Author: Joe Knowles | Contact: [email]

What this is
- Long fungal recordings analyzed with a square-root time (√t) transform to concentrate slow rhythms.
- From √t spectra: τ-band fractions, k-shape, concentration/SNR; combined with spike statistics for species fingerprints and ML features.

Why it matters
- Fungi exhibit species/state-specific electrical spikes and multihour oscillations. √t clarifies these slow rhythms beyond plain STFT, enabling sensing and classification.

Key findings (representative runs)
- Schizophyllum: slow/very-slow balanced; spikes=10.
- Enoki: very-slow dominated; spikes=1297.
- Ghost Fungi: very-slow dominated; spikes=6.
- Cordyceps: very-slow dominated (no fast band); spikes=1114.
- √t concentration > STFT by ~1.14–2.08× across species.

Open the figures
- Cross-species: deliverables/figures/summaries/snr_concentration_table.md
- Species panels: deliverables/figures/species/<Species>/*.png
- Fingerprints (press images): deliverables/press_kit/*2025-08-22T00:26:43.428982.*

Links (workspace)
- Results index: results/index.html
- Per-run artifacts: results/zenodo/<species>/<timestamp>/

Next steps
- Strong baselines (multitaper/synchrosqueezing), bootstrap CIs, stimuli overlays.
- Lightweight classifiers for species/state detection; expand dataset and metadata.
