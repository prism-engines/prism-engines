# Bench: `normalize_simple`

**Group:** normalize

**Description:** Baseline single-column normalization correctness

## Data Type
Deterministic synthetic time series

## Methodology
Generate a single-column deterministic synthetic series and apply standard normalization (mean centering and variance scaling).

## Rationale
This benchmark establishes the baseline correctness of the normalization procedure before testing higher-order invariants.

## Expected Outcome
The normalized series has zero mean and unit variance within numerical tolerance, with no NaN values.

## Interpretation of Failure
Deviation from zero mean or unit variance indicates incorrect centering or scaling, invalidating downstream analyses.

**Execution Time:** `0.002499` seconds


---

# Bench: `normalize_multicolumn`

**Group:** normalize

**Description:** Multi-column normalization correctness

## Data Type
Deterministic synthetic multivariate time series

## Methodology
Generate a deterministic synthetic dataset containing multiple numeric columns and apply standard normalization independently to each column.

## Rationale
Normalization must operate per-dimension without leakage across columns. This benchmark verifies dimensional independence.

## Expected Outcome
Each numeric column is normalized to zero mean and unit variance independently, with no cross-column interference.

## Interpretation of Failure
Deviation indicates improper shared statistics across columns, which would distort multivariate analyses.

**Execution Time:** `0.000845` seconds


---

# Bench: `normalize_noise`

**Group:** normalize

**Description:** Normalization robustness under additive noise

## Data Type
Stochastic synthetic multivariate time series

## Methodology
Generate a deterministic synthetic dataset, inject Gaussian noise into numeric columns, and apply standard normalization to the perturbed data.

## Rationale
Real-world data contains measurement noise. Normalization should remain stable under moderate stochastic perturbations.

## Expected Outcome
After normalization, noisy data exhibits zero mean and unit variance per column, with no NaN propagation.

## Interpretation of Failure
Sensitivity to noise indicates numerical instability or improper handling of variance under stochastic conditions.

**Execution Time:** `0.000954` seconds


---

# Bench: `normalize_affine`

**Group:** normalize

**Description:** Normalization affine invariance (scale + shift)

## Data Type
Deterministic synthetic time series

## Methodology
Generate a deterministic synthetic series x, apply an affine transform x' = a·x + b, then normalize both series independently.

## Rationale
Standard normalization should be invariant to affine transforms. Sensitivity to scale or offset indicates biased normalization.

## Expected Outcome
Normalized outputs from original and affine-transformed data are numerically identical within floating-point tolerance.

## Interpretation of Failure
Deviation indicates improper centering or scaling, leading to non-comparable normalized series.

**Execution Time:** `0.001358` seconds


---

# Bench: `normalize_idempotence`

**Group:** normalize

**Description:** Normalization idempotence (normalize twice = same result)

## Data Type
Deterministic synthetic multivariate time series

## Methodology
Apply standard normalization to a synthetic dataset, then re-apply normalization to the already-normalized output.

## Rationale
A properly implemented normalization operator should be idempotent. Reapplying it must not further alter the data.

## Expected Outcome
The second normalization produces output numerically identical to the first within floating-point tolerance.

## Interpretation of Failure
Violation indicates unstable normalization logic or cumulative rescaling effects that distort repeated analyses.

**Execution Time:** `0.001172` seconds


---

# Bench: `normalize_missing_data`

**Group:** normalize

**Description:** Normalization behavior with missing data (NaNs)

## Data Type
Stochastic synthetic multivariate time series with missing values

## Methodology
Generate a synthetic dataset, randomly inject missing values (NaNs) into numeric columns, and apply standard normalization.

## Rationale
Real-world economic and scientific datasets contain missing observations. Normalization must preserve missingness patterns while correctly scaling observed values.

## Expected Outcome
NaN locations are preserved exactly, while non-missing values are normalized to zero mean and unit variance.

## Interpretation of Failure
Altered NaN masks indicate data corruption. Deviations in mean or variance indicate improper handling of missing data.

**Execution Time:** `0.000982` seconds

