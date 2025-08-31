## Composites and CSV Index

- species_gallery.png: 2×2 gallery of latest summary panels (one per species).
- csv_index.csv: table listing each run’s tau_band_timeseries.csv and spike_times_s.csv.

How to read the visuals
- Spikes overlay: raw voltage with detected spikes (red). Many red markers → higher spike rate.
- τ-band heatmap: rows=time (s), cols=τ values; brighter = more √t-band power. Track how bands change over time.
- τ-band 3D: same data as heatmap shown as a surface for quick peak spotting.
- STFT vs √t: line spectra for one window. Narrower √t peak vs broader STFT indicates better concentration in √t domain.

How to use the CSV time series
- tau_band_timeseries.csv: time_s plus per-τ power and per-τ normalized power (sum to 1 per row). Use normalized columns to compare bands fairly.
- spike_times_s.csv: one spike timestamp per row (seconds). Align with tau series to examine pre/post changes.

Typical interpretations
- Dominant very‑slow τ over long spans → slow modulatory state or drift.
- Alternating fast/slow bands → switching regimes, possibly stimulus or moisture related.
- Rising power toward later windows → long‑term ramp; consider detrending or verifying baseline.

Reproduce/update
- Generate panels per run using analyze_metrics.py with --plot (and --export_csv for CSVs).
- Rebuild gallery: 
  - scripts/make_species_gallery.py
- Rebuild CSV index:
  - scripts/manage_csvs.py
