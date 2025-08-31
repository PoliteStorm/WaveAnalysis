---
Title: "Biological Validity Audit"
Date: "2025-08-31"
---

## Scope
Systems audited: data acquisition, preprocessing, spike detection, transforms (√t, STFT, advanced), feature extraction, spike-train metrics, multiscale entropy, stimulus–response validation, multichannel/network analysis, ML classifiers, sonification + cross-modal validation, fungal computing simulator, sensor prototype, symbolic language, data management/reproducibility.

## Rating Legend
- Pass: evidence supports biological validity
- Caution: partially supported; add checks
- Needs Work: add validation experiments/controls

## 1) Data Acquisition and Preprocessing
- Checks
  - Sampling rate, gain, reference configuration documented (metadata present)
  - Artifact rejection: motion, mains hum, electrode drift addressed (filters/notch/detrend)
  - Segment stationarity assessed (ADF/KPSS; visual QC)
  - Channel mapping consistent across runs
- Risks
  - Underdocumented hardware settings; mains/EMI contamination; temperature/humidity confounds
- Actions
  - Add impedance logs; environment sensors; standardized notch + bandlimit; per-run QC plots
- Status: Caution

## 2) Spike Detection
- Checks
  - Thresholding with robust noise estimate (MAD), refractory period enforced
  - Cross-validated against wavelet/template methods on annotated subsets
  - Spike width/amplitude distributions plausible for species/tissue
- Risks
  - Threshold bias under drift; misclassification of slow waves as spikes
- Actions
  - Add adaptive threshold with local baseline; manual spot-annotation set; compare two detectors
- Status: Caution

## 3) Time–Frequency Transforms (√t, STFT, Advanced)
- Checks
  - Transform ablation table (detrend, mother, windows) includes SNR/concentration metrics
  - Synthetic benchmark (known components) recovered without spurious peaks
  - Agreement across methods (√t vs STFT vs multitaper/HHT) on real data
- Risks
  - √t warp overemphasizing early-time dynamics; edge artifacts
- Actions
  - Add bootstrap CIs on peak metrics; mirror-padding; report bias tests on synthetic ramps
- Status: Caution

## 4) Feature Extraction (Spectral + Temporal)
- Checks
  - MFCC/basic spectral computed on valid windows mapped via speed factor
  - No NaNs; standardized scaling; dimensionality documented
- Risks
  - Window/hop mismatch with electrophysiology epochs; loudness normalization suppressing dynamics
- Actions
  - Validate time alignment on a test file; export per-window QA CSV; loudness cap with headroom
- Status: Pass

## 5) Spike-Train Metrics
- Checks
  - LV, CV², Fano, Victor distance, burst index distributions within expected neuro-like ranges
  - Refractory consistency; inter-spike interval histograms inspected
- Risks
  - Low-count bias; burst detection threshold sensitivity
- Actions
  - Bias-corrected estimators; sensitivity sweep; report CIs via bootstrapping
- Status: Caution

## 6) Multiscale Entropy (MSE) and Complexity
- Checks
  - Scale factors and embedding parameters documented; CI via surrogate shuffles
  - Complexity index correlates with qualitative dynamics per species
- Risks
  - Short-record bias; parameter overfitting
- Actions
  - Minimum-length guard; permutation-based baselines; cross-run reproducibility chart
- Status: Caution

## 7) Stimulus–Response Validation
- Checks
  - Pre/post windows, t-test, Mann–Whitney U, Cohen’s d computed with multiple-comparison control
  - Effect direction consistent across repetitions; permutation test for onsets
- Risks
  - Low N; stimulus timing jitter; regression-to-mean
- Actions
  - Randomized sham stimuli; block design; pre-registration of primary endpoint
- Status: Caution

## 8) Multichannel/Network Analysis
- Checks
  - Granger lags limited; stationarity checked; false-positive control via permutations
  - Graph metrics (degree, modularity) stable across splits
- Risks
  - Volume conduction/common driver; nonstationarity; computational kills for large lags
- Actions
  - Partial coherence; time-varying GC; surrogate phase-randomization controls
- Status: Caution

## 9) Machine Learning Classifiers
- Checks
  - Leakage avoided; stratified CV; calibration plots; permutation-label baseline
  - Confusion matrices balanced; feature importance stable
- Risks
  - Small dataset; class imbalance; optimistic CV due to segmentation
- Actions
  - Nested CV where feasible; grouped splits by recording; learning curves; external test holdout
- Status: Caution

## 10) Sonification + Cross-Modal Validation
- Checks
  - Continuous sonification uses faithful amplitude modulation; calibration tone present
  - MFCC+CCA correlations high across species; permutation p-values computed
- Risks
  - Psychoacoustic distortion; mapping nonlinearity vs biology
- Actions
  - Increase permutations to ≥200; add bootstrap CIs; add alternative features (chroma, modulation)
- Status: Pass (with p-value upgrade pending)

## 11) Fungal Computing Simulator
- Checks
  - Parameters constrained by literature (conductance ranges, time constants)
  - Reproduces qualitative behaviors (adaptation, bursting) seen in data
- Risks
  - Overfitting to observed datasets; missing biophysical constraints
- Actions
  - Parameter priors; validation against unseen species; unit tests on dynamical regimes
- Status: Caution

## 12) Sensor Prototype (Hardware)
- Checks
  - Concept includes referencing, shielding, anti-aliasing, notch
  - Data pathways documented; impedance targets defined
- Risks
  - EMI; thermal drift; electrode polarization
- Actions
  - Add hardware BOM; bench tests with saline/phantom; impedance spectroscopy logs
- Status: Needs Work

## 13) Symbolic Language (FSL)
- Checks
  - Operators map to measurable primitives (spike counts, coincidence, sustained power)
  - Example programs compile to analyzable pipelines
- Risks
  - Semantics drift from biology; compositionality without empirical grounding
- Actions
  - Define formal semantics with unit tests; align to simulator + real data motifs
- Status: Caution

## 14) Data Management and Reproducibility
- Checks
  - Timestamped outputs; JSON/CSV/HTML reports; Git-tracked; paper sections updated
- Risks
  - Large pushes failing; metadata omissions in some runs
- Actions
  - Chunked pushes; metadata schema validation; nightly summary index regeneration
- Status: Pass

## Key Evidence Pointers
- Cross-modal summary: `results/cross_modal_mfcc_cca/2025-08-31T08-10-27/summary.json`
- Audio dashboards: `results/audio_continuous/_indexes/index_2025-08-30T22-53-05.html`
- Paper sections: `docs/paper/paper.md`

## Immediate Action Items
- Increase MFCC+CCA permutations to 200 and add bootstrap CIs
- Add adaptive threshold spike detector and small annotated ground-truth set
- Add synthetic transform sanity tests and report bias metrics
- Add sham stimuli and multiple-comparison controls in stimulus framework
- Add grouped CV and permutation-label baselines for ML
- Add sensor BOM + bench protocol; record impedance logs per run

## Certification (Provisional)
Overall biological validity: Caution. Strong evidence for cross-modal alignment; remaining items require targeted experiments and controls to elevate to Pass.
