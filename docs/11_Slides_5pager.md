1) Problem & Idea
- Long fungal recordings: spikes + multihour rhythms are hard to resolve with linear-time STFT.
- Idea: √t time-warp concentrates slow rhythms; yields τ-band fingerprints and ML features.

2) Method & Baselines
- √t transform (u=√t), Gaussian/Morlet windows, FFT; τ={5.5,24.5,104}.
- Baselines: STFT, multitaper STFT, reassigned spectrogram, synchrosqueezed CWT.

3) Results (Representative)
- Species τ-fraction profiles and spikes; √t concentration > STFT (1.14–2.08×).
- Cross-species table; CI bands; ablations; controls (phase-randomized).

4) Reproducibility & ML
- Results packaged; CSV-first regeneration; CI summaries; LOFO/LOCO CV (pipeline).

5) Plan, Risks, Resources
- 12–18 month milestones; mitigations; equipment; collaborators; call to action.
