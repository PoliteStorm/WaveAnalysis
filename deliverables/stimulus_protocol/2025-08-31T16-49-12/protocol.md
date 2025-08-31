# Controlled Stimulus Validation Protocol

- Created: 2025-08-31T16-49-12
- Species: Schizophyllum_commune
- Author: joe knowles

## Stimuli
- Moisture: 0.5 mL sterile water at t=+0 s (baseline 10 min, post 20 min)
- Light: 1000 lux LED for 60 s (baseline 10 min, post 20 min)
- Temperature: +2°C air pulse for 60 s (baseline 10 min, post 20 min)

## Recording
- Sampling rate: 1 Hz (match configs)
- Channels: differential pairs as available
- Environment: stable humidity/temperature

## CSV Template
- Columns: time_s,stimulus_type,intensity,notes
- time_s: seconds from recording start
- stimulus_type: moisture | light | temperature
- intensity: free text (e.g., 0.5ml, 1000lux, +2C)
