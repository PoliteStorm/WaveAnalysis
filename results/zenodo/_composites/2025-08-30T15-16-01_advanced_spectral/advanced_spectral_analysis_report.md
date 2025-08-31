# Advanced Spectral Analysis Comparison Report

**Analysis Date:** 2025-08-30 15:16:05

**Species:** Schizophyllum_commune
**Methods Compared:** √t Transform, Synchrosqueezing, Multitaper, Hilbert-Huang

## Performance Comparison

| Method | Status | Spectral Concentration | Peak Frequency (Hz) | Notes |
|--------|--------|----------------------|-------------------|--------|
| sqrt_transform | ✅ Success | 0.4302 | 0.010 | High resolution |
| synchrosqueezing | ✅ Success | 0.3380 | 0.012 | High resolution |
| multitaper | ❌ Failed | N/A | N/A | cannot import name 'dpss' from 'scipy.signal' (/usr/lib/python3/dist-packages/scipy/signal/__init__.py) |
| hilbert_huang | ⚠️ partial | N/A | N/A | Limited data |

## Key Findings

- **Best Performance:** sqrt_transform with spectral concentration of 0.4302
- **Synchrosqueezing:** Provides enhanced frequency resolution for time-varying spectra
- **Multitaper:** Offers robust spectral estimation with reduced variance
- **Hilbert-Huang:** Well-suited for non-stationary signals with multiple oscillatory modes
- **√t Transform:** Specialized for sublinear temporal dynamics in fungal signals

## Recommendations

1. **For fungal electrophysiology:** √t transform remains the method of choice due to its specialization for sublinear dynamics
2. **For enhanced resolution:** Consider synchrosqueezing for detailed frequency analysis
3. **For robustness:** Multitaper methods provide stable spectral estimates
4. **For complex signals:** Hilbert-Huang transform offers adaptive decomposition

## Files Generated

- `advanced_spectral_comparison.json` - Complete analysis results
- `advanced_spectral_comparison.png` - Comparative visualizations
- `advanced_spectral_analysis_report.md` - This summary report
