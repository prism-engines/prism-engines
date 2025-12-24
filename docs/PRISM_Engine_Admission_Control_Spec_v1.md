# PRISM Engine Admission Control Spec (v1)

Purpose
- Prevent any single engine from corrupting consensus geometry.
- Make every engine "earn" inclusion through: (a) validity gates, (b) null calibration, (c) stability scoring.

Core Principles
1) Hard gates first. If an engine fails a required gate, it produces NO consensus contribution for that window.
2) Every engine must output: (a) primary metric(s), (b) QC diagnostics, (c) confidence (or p-value / CI), (d) a stability score.
3) Consensus uses weighted fusion. Default weight is NOT equal-weight; it is learned/earned per engine by benchmark + stability.

--------------------------------------------------------------------
A. Standard Inputs (all engines)
--------------------------------------------------------------------

A1. Window definition
- Window sizes: 21 to 252 trading days (or equivalent monthly/weekly counts).
- Each engine declares the MIN window length it requires (see Engine Cards).

A2. Frequency alignment (required)
- All series in a run MUST share the same timestamp grid for that run.
- Mixed-frequency indicators must be:
  - upsampled with explicit hold/forward-fill policy OR
  - downsampled with explicit aggregation policy (mean/median/last), AND
  - flagged in metadata.

A3. Missing data policy (required)
- Each engine must declare how it handles missingness:
  - drop rows, pairwise deletion, imputation (and which type), or masked algorithms.
- Missingness above threshold must gate off the engine:
  - default: missing_rate_window <= 5% for multivariate engines
  - default: missing_rate_window <= 10% for univariate engines

A4. Outlier handling (required for finance; recommended otherwise)
- Default robust transform for prices: log returns (or pct change), winsorize at 1%/99% (configurable).
- For indices already bounded (ENSO/NAO/PDO), do NOT winsorize by default; instead use robust scaling if needed.

A5. Standard scaling options (engine must specify one)
- "zscore" (mean/std), "robust_z" (median/MAD), "rank" (empirical CDF / percent rank)
- Scaling must be consistent within an engine across windows.

--------------------------------------------------------------------
B. Null Calibration (required for any engine that can hallucinate)
--------------------------------------------------------------------

B1. Null families
At least one null test is REQUIRED for:
- Mutual information, Granger, Copulas, DTW, RQA, Lyapunov, Cointegration (rolling), Hurst (rolling)

Recommended nulls (choose 1-2):
1) Permutation null:
   - random permutation of time index (breaks temporal structure entirely)
2) Block bootstrap null:
   - shuffle blocks (preserves local autocorrelation)
3) Phase randomization surrogate:
   - preserves power spectrum, destroys phase coupling (good for dependence tests)
4) Pairwise offset null:
   - circularly shift one series by random lag (destroys alignment, keeps marginal)

B2. Null outputs
Engines must report:
- null_mean, null_std (or full null distribution summary)
- empirical_p_value = (count(null >= observed) + 1) / (N_null + 1)
- effect_size = (observed - null_mean) / (null_std + eps)

B3. Gate rule (default)
- Require empirical_p_value <= 0.05 (configurable) OR effect_size >= 2.0
- If not, engine contributes 0 weight in that window.

--------------------------------------------------------------------
C. Stability Scoring (required for consensus eligibility)
--------------------------------------------------------------------

C1. Stability definitions
Each engine must compute a Stability Score S in [0, 1] for each window, based on at least 2 of:
- Parameter stability: sensitivity to small hyperparameter changes
- Resample stability: bootstrap variance of outputs
- Temporal stability: smoothness vs expected volatility of metric
- Cross-validation stability: out-of-sample consistency (when meaningful)

C2. Default stability recipe (practical)
- Compute the metric on:
  - the window,
  - a 90% subsample (drop random 10% time points or blocks),
  - a second 90% subsample,
- Let d = mean absolute difference across these runs (normalized).
- Map to S via: S = exp(-k * d), k chosen so moderate drift -> ~0.5.

C3. Gate rule (default)
- Require S >= 0.6 to contribute to consensus for that window.
- Engines below threshold may still be logged for diagnostics, but not fused.

--------------------------------------------------------------------
D. Consensus Fusion
--------------------------------------------------------------------

D1. Engine contribution weight (per window)
For engine e at time t:
- Gate(e,t) in {0,1}
- NullPass(e,t) in {0,1}
- Stability(e,t) in [0,1]
- BenchmarkScore(e) in [0,1] (global, from synthetic tests)
- DomainFit(e,domain) in [0,1] (finance/climate/epi)

Default:
Weight(e,t) =
  Gate(e,t) * NullPass(e,t) * Stability(e,t) * BenchmarkScore(e) * DomainFit(e,domain)

D2. Prevent domination
- Apply cap: Weight(e,t) <= w_cap (default 0.25)
- Renormalize weights across eligible engines to sum to 1.0.

D3. Output
Consensus geometry at t must include:
- the fused geometry metrics,
- the contributing engines and their weights,
- a diagnostics section: which engines were gated off and why.

--------------------------------------------------------------------
E. Engine Cards (v1 recommendations)
--------------------------------------------------------------------
Notation:
- MIN_N: minimum usable window length
- Required: must pass to contribute to consensus
- Conditional: can run, but contributes only if gates pass
- Exclude: not used for consensus (diagnostic only allowed)

--------------------------------------------------------------------
E1. PCA (multivariate)
Measures
- Dominant shared factor directions (co-movement modes).

MIN_N
- MIN_N >= max(60, 5 * n_features) unless using shrinkage / randomized PCA.

Required preprocessing
- Robust scaling (robust_z recommended)
- Missingness <= 5% in window (after alignment)
- Robust covariance or shrinkage if p is not small vs n

Gates
- Condition number / explained variance sanity
- Loading stability across bootstrap subsamples: S >= 0.6

Null
- Not required by default, but recommended: compare EVR against shuffled-column null.

Verdict
- INCLUDE (robust + shrinkage strongly recommended)

--------------------------------------------------------------------
E2. Rolling Correlation (pairwise)
Measures
- Linear dependence across indicators.

MIN_N
- MIN_N >= 60 recommended; MIN_N >= 21 allowed with robust correlation and warnings.

Required preprocessing
- Use returns/diffs for trending series (prices)
- Prefer Spearman by default for mixed distributions

Gates
- Outlier rate threshold
- Stability across subsamples

Null
- Optional: shift-null to detect spurious alignment.

Verdict
- INCLUDE (robust / Spearman default)

--------------------------------------------------------------------
E3. Granger Causality (VAR-based)
Measures
- Predictive lead-lag (not true causality).

MIN_N
- MIN_N >= 120 strongly recommended (depends on max lag).

Required preprocessing (REQUIRED)
- Stationarity enforced (ADF/KPSS checks)
- Seasonality controls for climate/epi (monthly/weekly)
- Lag selection (AIC/BIC) and max_lag cap
- Multiple testing control (FDR)

Gates (REQUIRED)
- Stationarity pass for both series
- FDR-adjusted p-value <= 0.05
- Edge stability across resamples

Null (REQUIRED)
- Shift-null or block bootstrap to estimate false edges.

Verdict
- CONDITIONAL (high discipline required)

--------------------------------------------------------------------
E4. Hurst Exponent (univariate)
Measures
- Long-memory / persistence vs mean reversion.

MIN_N
- MIN_N >= 500 preferred; rolling 21-252 is often too short.

Required preprocessing
- Detrend and remove seasonality if present
- Confidence intervals (bootstrap)

Gates
- CI width below threshold; otherwise do not contribute

Null (REQUIRED)
- Compare to matched ARMA/GARCH null or phase-randomized null.

Verdict
- EXCLUDE from consensus (diagnostic only)

--------------------------------------------------------------------
E5. Entropy (Permutation preferred)
Measures
- Complexity / predictability.

MIN_N
- Depends on embedding dimension m:
  - rule of thumb: MIN_N >= 5 * m! (permutation patterns) and practical MIN_N >= 100.

Required preprocessing
- Standardize
- Detrend/seasonal adjust when appropriate
- Fixed (m, tau) policy across runs

Gates
- Sample adequacy for chosen m
- Stability across subsamples

Null (REQUIRED for tight claims; recommended otherwise)
- Permutation null or phase-randomized null.

Verdict
- INCLUDE (conditional on sample adequacy)

--------------------------------------------------------------------
E6. Mutual Information (pairwise)
Measures
- Nonlinear dependence.

MIN_N
- MIN_N >= 200 recommended for kNN MI; more is better.

Required preprocessing (REQUIRED)
- Rank transform (empirical CDF) OR standardized inputs
- Fixed estimator choice + bias control
- Missingness policy

Gates (REQUIRED)
- Null significance (empirical p-value)
- Stability across resamples

Null (REQUIRED)
- Shift-null / phase-randomized / block bootstrap

Verdict
- CONDITIONAL (must be null-calibrated)

--------------------------------------------------------------------
E7. DTW (similarity with warping)
Measures
- Shape similarity allowing temporal misalignment.

MIN_N
- MIN_N >= 60 recommended.

Required preprocessing (REQUIRED)
- Standardize
- Warping constraint (Sakoe-Chiba band, e.g., 5-10% of window)
- Optional detrend depending on domain

Gates (REQUIRED)
- Warping path length constraint
- Compare DTW distance to Euclidean distance ratio (detect over-warping)
- Stability across resamples

Null (REQUIRED)
- Shift-null for baseline similarity

Verdict
- CONDITIONAL (tight constraints required)

--------------------------------------------------------------------
E8. Cointegration (Johansen) (multivariate, long-horizon)
Measures
- Long-run equilibrium among I(1) series.

MIN_N
- MIN_N >= 500-1000 typical; rolling 21-252 is not appropriate.

Required preprocessing (REQUIRED)
- Unit root tests; correct deterministic terms; lag selection
- Use LEVELS (not differenced) for test, but verify I(1)

Gates
- Rank stability across adjacent windows
- Out-of-sample validation

Null
- Recommended

Verdict
- EXCLUDE from short-window consensus; keep as long-horizon module only

--------------------------------------------------------------------
E9. Copula Analysis (pairwise)
Measures
- Joint dependence structure, including tail dependence.

MIN_N
- MIN_N >= 250 recommended for tail estimates; more is better.

Required preprocessing (REQUIRED)
- Transform marginals to U(0,1) via ranks / empirical CDF
- Handle ties; bootstrap CI

Gates (REQUIRED)
- Tail dependence CI not exploding
- Family selection sanity

Null (REQUIRED)
- Bootstrap / permutation to estimate tail dependence under independence

Verdict
- CONDITIONAL (larger windows; bootstrap required)

--------------------------------------------------------------------
E10. HMM
Measures
- Hidden regimes with transition structure.

MIN_N
- MIN_N >= 200 recommended (varies with number of states and dimension).

Required preprocessing (REQUIRED)
- Standardize
- State-count selection policy (IC + out-of-sample)
- Minimum dwell-time constraint (avoid rapid flipping)

Gates (REQUIRED)
- Regime persistence: average dwell >= threshold
- Transition matrix sanity (no near-uniform random flipping)
- Stability across random initializations

Null
- Optional but recommended: compare to shuffled time order.

Verdict
- INCLUDE (conditional on stability)

--------------------------------------------------------------------
E11. Wavelets
Measures
- Time-localized scale/frequency content.

MIN_N
- Depends on lowest frequency scale; for multi-scale: MIN_N >= 128 recommended.

Required preprocessing (REQUIRED)
- Uniform sampling
- Detrend recommended
- Boundary handling (padding policy)

Gates
- Edge-effect diagnostics
- Stability of bandpower summaries across resamples

Null
- Optional: phase randomization for significance of bandpower changes.

Verdict
- INCLUDE (strong cross-domain engine)

--------------------------------------------------------------------
E12. RQA
Measures
- Recurrence structure in reconstructed phase space.

MIN_N
- MIN_N >= 300 recommended (parameter dependent).

Required preprocessing (REQUIRED)
- Standardize; detrend/seasonal adjust as needed
- Systematic embedding selection (tau, m) with locked policy
- Fixed recurrence threshold policy

Gates (REQUIRED)
- Parameter sensitivity test (small changes in m/tau/eps should not swing results wildly)
- Null significance vs phase-randomized surrogate

Null (REQUIRED)
- Phase randomization or block bootstrap

Verdict
- CONDITIONAL leaning EXCLUDE (keep out of consensus until benchmarked)

--------------------------------------------------------------------
E13. Lyapunov Exponent
Measures
- Sensitive dependence / chaos indicator.

MIN_N
- Typically needs long, low-noise series; rolling 21-252 is not suitable.

Required preprocessing
- Denoising + careful embedding (not realistic for your pipeline).

Verdict
- EXCLUDE from consensus

--------------------------------------------------------------------
E14. DMD
Measures
- Coherent modes + approximate linear operator over the window.

MIN_N
- MIN_N >= 100 recommended; higher for multivariate panels.

Required preprocessing (REQUIRED)
- Standardize
- Rank truncation policy (energy threshold)
- Regularization (TLS-DMD or ridge) recommended

Gates (REQUIRED)
- Eigenvalue stability across bootstrap/resamples
- Mode energy concentration sanity

Null
- Optional: compare operator spectrum to shuffled-time baseline.

Verdict
- INCLUDE (conditional, stabilized)

--------------------------------------------------------------------
E15. Rolling Beta (finance only)
Measures
- Linear exposure to benchmark.

MIN_N
- MIN_N >= 60 recommended.

Required preprocessing (REQUIRED)
- Returns (not prices)
- Robust regression recommended; outlier control

Gates
- Benchmark relevance check (R^2 floor)
- Stability across resamples

Null
- Optional

Verdict
- CONDITIONAL (finance sub-geometry only)

--------------------------------------------------------------------
E16. FFT (spectral)
Measures
- Global periodicities in window.

MIN_N
- MIN_N >= 128 recommended for meaningful resolution.

Required preprocessing (REQUIRED)
- Detrend; taper/windowing
- Even sampling

Gates
- Leakage diagnostics; peak stability across resamples

Null
- Recommended: phase randomization

Verdict
- CONDITIONAL (often subordinate to wavelets)

--------------------------------------------------------------------
F. Minimal Consensus Set (recommended v1)
--------------------------------------------------------------------
If you want a compact, robust geometry core:
1) PCA (robust + shrinkage)
2) Robust dependence (Spearman/robust correlation)
3) Wavelets (bandpower summaries)
4) DMD (regularized, rank-controlled)

Add later (after benchmark suite passes):
- HMM (regimes)
- Permutation entropy (complexity)
- Copula tail module (finance crises)

--------------------------------------------------------------------
G. Required Benchmark Suite (must pass before publication claims)
--------------------------------------------------------------------
1) Regime-switch factor model (known states) -> validate PCA/corr/HMM/DMD responses
2) Nonlinear dependence with zero linear correlation -> validate MI module
3) Known lead-lag VAR network + confounder -> validate Granger + FDR + stability
4) Time-local frequency change (chirp / piecewise cycles) -> validate wavelets > FFT
5) Tail dependence stress (Gaussian vs t-copula) -> validate copula module
6) Null tests on phase-randomized and block-bootstrapped surrogates -> estimate false positive rates

--------------------------------------------------------------------
H. What PRISM should always report (for trust)
--------------------------------------------------------------------
For each run/window:
- Engine eligibility report: pass/fail of each gate with reasons
- Engine weights used in consensus (post-cap, normalized)
- Null calibration summaries for engines that require it
- Stability scores per engine
- Sensitivity report: does consensus change if one engine is removed?

End of Spec
