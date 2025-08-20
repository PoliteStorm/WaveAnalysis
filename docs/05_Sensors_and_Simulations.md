## Practical sensor ideas
- Electrode arrays: 4–16 channels, Ag/AgCl or stainless steel; 1 Hz sampling is adequate for slow fungal rhythms; higher for spikes.
- Substrate & environment: agar/wood substrate, moisture and temperature sensors logged alongside voltage.
- Stimuli: moisture pulses, light, nutrients; mark events in a sidecar log.
- Low‑power setup: microcontroller or SBC logging to CSV; rotate files hourly to keep RAM/CPU low.
- On‑device pre‑processing: baseline subtraction + simple spike counter; defer heavy transforms to laptop.

## From signals to features
- √t bands: per window, compute normalized τ‑band powers and k‑features (centroid, width, entropy, peaks).
- Spikes: rate, amplitude/ISI distributions, entropy, skewness, kurtosis.
- Metadata: channel geometry, substrate, moisture/temp, stimuli timing.

## Biocomputing simulations (ideas)
- Logic via thresholds: treat band power > θ as logical 1; design AND/OR from two bands/channels.
- Reservoir computing: treat channels × bands as a fading‑memory reservoir; train a linear readout.
- Routing/memory: modulate substrate moisture to gate conduction; read with √t features.
- Network analogies: simulate per‑edge conductance varying with slow bands; test information routing tasks.

## Evaluation
- Use LOCO CV on channels; report accuracy, F1, and calibration. Bootstrap windows to attach CIs.
- Compare √t vs STFT features under identical windows to show added value.

## Next steps
- Fieldable sensor: rugged electrodes + moisture/temp + timestamped logs.
- Online event flagging: “interesting” window alert if spectral concentration or spike SNR exceeds baseline.
