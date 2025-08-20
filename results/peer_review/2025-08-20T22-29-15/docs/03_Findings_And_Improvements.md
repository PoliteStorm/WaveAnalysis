## Findings so far (simple summary)
- √t bands highlight slow/very‑slow rhythms that standard STFT smears over long durations.
- Features from √t (band fractions, k‑centroid/width/entropy) combine with spike stats to differentiate species/states.
- On limited data, per‑channel CV (leave‑one‑channel‑out) is preferable to per‑file CV.

How it recognizes biological patterns
- Spikes: baseline‑subtracted thresholding → rate, ISI, amplitude structure.
- Rhythms: √t‑warped windows stabilize long‑time oscillations, making peaks compact in k.
- Combining both gives a multi‑scale “signature.”

Planned improvements
- Parameter sweeps for τ grid; Morlet windows; u‑domain detrend; confidence intervals via bootstrapping; caching all intermediates; skipping low‑power windows to save CPU.

Bibliography (starter)
- Adamatzky, A. (2022). Fungal networks. `https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/`
- Jones, D. et al. (2023). Electrical spiking in fungi. `https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/`
- Olsson, H., Hansson, B. (2021). Signal processing in biological systems. `https://www.sciencedirect.com/science/article/pii/S0303264721000307`
- Nature (2018): Rhythms and complexity in living systems. `https://www.nature.com/articles/s41598-018-26007-1`
