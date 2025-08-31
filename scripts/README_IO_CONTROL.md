# IO-Control Experiments (Proof-of-Concept)

Author: Joe Knowles

## Overview
- Fit stimulus→response kernels on √t band powers (FIR) and spike counts (Poisson GLM).
- Simulate closed-loop control of band powers using learned kernels.
- Timestamped, data-driven, reproducible runs.

## Quick start
```bash
# 1) Fit models
python3 scripts/io_fit_kernels.py
# outputs under results/io_control/<timestamp>/

# 2) Simulate closed loop
python3 scripts/io_simulate_closed_loop.py
# outputs under results/io_control/simulations/<timestamp>/
```

## Inputs
- Replace placeholder loaders in `io_fit_kernels.py` with links to your real stimulus protocols and √t band outputs.
- Add spike time/count loading where available.

## Outputs
- `run_metadata.json`: author, timestamp, git hash
- `io_models.json`: FIR coefficients per √t band, Poisson GLM weights
- `fir_kernel_summaries.png`: quick visualization of stimulus contributions
- Simulation PNGs: predicted mid-band trajectories vs stimulus scaling

## Notes
- Report effect sizes and 95% CIs for response detection.
- Include negative controls and phase-randomized surrogates alongside main analyses.
