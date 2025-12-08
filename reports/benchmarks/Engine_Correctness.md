# Engine Correctness Benchmarks

- **Date**: 2025-12-08
- **Commit**: `cd6ee0e`
- **Generated**: 2025-12-08 21:43:40

## Configuration

- **Lenses tested**: magnitude, pca, influence, clustering, network
- **Scenarios**: 7
- **Data type**: Synthetic signals

## Methods

Each synthetic scenario generates data with known statistical properties. The temporal engine and analytical lenses process the data, then validation checks confirm expected behaviors are detected.

Scenarios tested:
- **Sine wave**: Periodic signal detection
- **Trend + noise**: Trend and decomposition detection
- **Step change**: Regime shift detection
- **White noise**: Null baseline (no structure)
- **Correlated pairs**: Correlation and influence detection
- **Mean reverting**: Stationarity detection
- **Random walk**: Non-stationary behavior

## Results

| Scenario | Rows | Cols | Lenses | Checks Passed | Runtime | Status |
|----------|------|------|--------|---------------|---------|--------|
| sine_wave | 252 | 5 | 5 | 1/1 | 2.29s | PASS |
| trend_noise | 252 | 5 | 5 | 1/1 | 0.19s | PASS |
| step_change | 252 | 5 | 5 | 0/0 | 0.18s | PASS |
| white_noise | 252 | 5 | 5 | 1/1 | 0.19s | PASS |
| correlated_pairs | 252 | 8 | 5 | 0/0 | 0.18s | PASS |
| mean_reverting | 252 | 5 | 5 | 0/0 | 0.18s | PASS |
| random_walk | 252 | 5 | 5 | 0/0 | 0.17s | PASS |

## Detailed Results

### Sine Wave

**Metrics:**

- mean_mean: 0.0005
- mean_std: 0.7057
- column_correlation: 1.0000
- pca_first_component_variance: 1.0000
- n_clusters_found: 2

**Validation Checks:**

- pca_captures_structure: PASS

**Notes:**

- Passed 1/1 validation checks

### Trend Noise

**Metrics:**

- mean_mean: 1.2558
- mean_std: 0.7372
- column_correlation: 0.9833
- pca_first_component: 0.9866

**Validation Checks:**

- pca_captures_trend: PASS

**Notes:**

- Passed 1/1 validation checks

### Step Change

**Metrics:**

- mean_mean: 0.4994
- mean_std: 0.5124
- column_correlation: 0.9611

**Notes:**

- Passed 0/0 validation checks

### White Noise

**Metrics:**

- mean_mean: -0.0165
- mean_std: 0.9914
- column_correlation: -0.0049
- pca_first_component: 0.2290

**Validation Checks:**

- no_dominant_component: PASS

**Notes:**

- Passed 1/1 validation checks

### Correlated Pairs

**Metrics:**

- mean_mean: 0.0269
- mean_std: 1.0050
- column_correlation: 0.0647
- n_clusters: 2

**Notes:**

- Passed 0/0 validation checks

### Mean Reverting

**Metrics:**

- mean_mean: 0.0188
- mean_std: 0.2267
- column_correlation: -0.0433

**Notes:**

- Passed 0/0 validation checks

### Random Walk

**Metrics:**

- mean_mean: 110.8511
- mean_std: 7.5969
- column_correlation: 0.0960

**Notes:**

- Passed 0/0 validation checks

## Summary Statistics

- **Total scenarios**: 7
- **Total checks**: 3/3 passed
- **Total runtime**: 3.38s
- **Pass rate**: 100.0%
