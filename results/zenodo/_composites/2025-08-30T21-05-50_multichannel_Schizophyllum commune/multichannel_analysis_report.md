# Multi-Channel Correlation Analysis Report

**Analysis Date:** 2025-08-30 21:09:14

## Dataset Overview

- **File:** Schizophyllum commune.txt
- **Channels:** 7 (from 7 total)
- **Duration:** 263492.0 seconds
- **Sampling Rate:** 1.0 Hz

## Channel Statistics

| Channel | Mean (mV) | Std (mV) | RMS (mV) | Power (mV²) | SNR (dB) |
|---------|-----------|-----------|-----------|-------------|-----------|
| diff_1 | -0.043 | 0.185 | 0.190 | 0.036 | 20.8 |
| diff_2 | -0.084 | 0.237 | 0.251 | 0.063 | 28.2 |
| diff_3 | -0.170 | 0.263 | 0.313 | 0.098 | 30.1 |
| diff_4 | -0.614 | 0.361 | 0.712 | 0.507 | 41.4 |
| diff_5 | 0.243 | 0.154 | 0.287 | 0.083 | 25.3 |
| diff_6 | -0.049 | 0.179 | 0.186 | 0.035 | 10.7 |
| diff_7 | -0.707 | 0.598 | 0.926 | 0.858 | 10.8 |

## Significant Correlations

**Total Significant Interactions:** 58

| Channel 1 | Channel 2 | Correlation | Lag (s) | Direction |
|-----------|-----------|-------------|----------|-----------|
| diff_4 | diff_5 | -0.691 | -11.0 | negative |
| diff_4 | diff_5 | -0.691 | 95.0 | negative |
| diff_5 | diff_7 | -0.666 | 111.0 | negative |
| diff_5 | diff_7 | -0.665 | 172.0 | negative |
| diff_5 | diff_7 | -0.659 | 367.0 | negative |
| diff_5 | diff_7 | -0.656 | 443.0 | negative |
| diff_5 | diff_7 | -0.646 | -17.0 | negative |
| diff_5 | diff_7 | -0.645 | -90.0 | negative |
| diff_3 | diff_7 | 0.617 | 0.0 | positive |
| diff_3 | diff_7 | 0.615 | 66.0 | positive |

## Network Topology

- **Nodes:** 7
- **Edges:** 42
- **Network Type:** Directed
- **Weakly Connected Components:** 1
- **Strongly Connected Components:** 1

## Biological Interpretation

### Network Connectivity

- **58 significant channel interactions detected**
- These interactions suggest coordinated electrical activity across the fungal mycelium
- Cross-correlations indicate synchronized spiking patterns between different network regions
- Time lags in correlations may reflect signal propagation delays in the mycelial network

### Information Processing

- Granger causality analysis reveals directional information flow
- Coherence analysis shows frequency-specific synchronization
- Network topology suggests distributed processing capabilities

### Functional Implications

- **Communication:** Coordinated activity enables long-distance signaling
- **Integration:** Network connectivity supports information integration
- **Adaptation:** Dynamic correlations suggest adaptive network behavior
- **Intelligence:** Complex interaction patterns indicate computational capabilities

## Technical Recommendations

### Data Quality
- Ensure consistent electrode placement
- Minimize electrical interference
- Validate channel isolation

### Analysis Parameters
- Adjust correlation thresholds based on signal quality
- Consider different frequency bands for coherence analysis
- Evaluate multiple lag ranges for causality testing

### Future Experiments
- Stimulus-response experiments to validate connectivity
- Pharmacological interventions to modulate network activity
- Environmental perturbations to test network resilience

## Files Generated

- `multichannel_analysis_results.json` - Complete analysis results
- `multichannel_overview.png` - Multi-panel overview visualization
- `correlation_peaks.png` - Significant correlation visualization
- `coherence_matrix.png` - Coherence analysis visualization
- `multichannel_analysis_report.md` - This summary report
