# PRISM Engine Benchmark Suite Report

**Generated**: 2025-12-08_204727
**Datasets Analyzed**: 6
**Total Runtime**: 0.0 seconds

## Executive Summary

This report presents the results of running the PRISM analytical engine across
six synthetic benchmark datasets with known statistical properties. Each dataset
tests specific aspects of the engine's detection capabilities.

### Quick Summary

| Dataset | Indicators | Observations | Lens Agreement | Structure Detected | Confidence |
|---------|------------|--------------|----------------|-------------------|------------|
| anomalies | 0 | 0 | 0.00 | No | low |
| clear_leader | 0 | 0 | 0.00 | No | low |
| clusters | 0 | 0 | 0.00 | No | low |
| periodic | 0 | 0 | 0.00 | No | low |
| pure_noise | 0 | 0 | 0.00 | No | low |
| two_regimes | 0 | 0 | 0.00 | No | low |

## Detailed Results

### Anomalies

**Description**: Random spikes and outliers embedded in normal behavior

#### Dataset Statistics

- **Indicators**: 0
- **Observations**: 0
- **Date Range**:  to 
- **Runtime**: 0.00s

#### Expected Structure

Anomaly lens should flag outlier points; magnitude lens spikes

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.000
- **Mean Rank Fluctuation**: 0.000

#### Interpretation



#### Potential Failure Modes

- Missed anomalies
- False positive rate too high

#### Confidence Evaluation: **LOW**

> **Error**: No data found for benchmark group: anomalies

#### Visualizations

- `anomalies_timeseries.png` - Time series overview
- `anomalies_pca_scatter.png` - PCA projection
- `anomalies_correlation.png` - Correlation matrix
- `anomalies_lens_agreement.png` - Lens ranking comparison
- `anomalies_mrf_timeline.png` - MRF timeline

---

### Clear Leader

**Description**: One dominant indicator with subordinate followers

#### Dataset Statistics

- **Indicators**: 0
- **Observations**: 0
- **Date Range**:  to 
- **Runtime**: 0.00s

#### Expected Structure

Single indicator should rank #1 consistently across all lenses

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.000
- **Mean Rank Fluctuation**: 0.000

#### Interpretation



#### Potential Failure Modes

- Leader not detected
- False positives in followers

#### Confidence Evaluation: **LOW**

> **Error**: No data found for benchmark group: clear_leader

#### Visualizations

- `clear_leader_timeseries.png` - Time series overview
- `clear_leader_pca_scatter.png` - PCA projection
- `clear_leader_correlation.png` - Correlation matrix
- `clear_leader_lens_agreement.png` - Lens ranking comparison
- `clear_leader_mrf_timeline.png` - MRF timeline

---

### Clusters

**Description**: Three correlated clusters of indicators

#### Dataset Statistics

- **Indicators**: 0
- **Observations**: 0
- **Date Range**:  to 
- **Runtime**: 0.00s

#### Expected Structure

Clustering lens should identify 3 groups; network lens shows dense subgraphs

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.000
- **Mean Rank Fluctuation**: 0.000

#### Interpretation



#### Potential Failure Modes

- Cluster merging
- Spurious cluster detection

#### Confidence Evaluation: **LOW**

> **Error**: No data found for benchmark group: clusters

#### Visualizations

- `clusters_timeseries.png` - Time series overview
- `clusters_pca_scatter.png` - PCA projection
- `clusters_correlation.png` - Correlation matrix
- `clusters_lens_agreement.png` - Lens ranking comparison
- `clusters_mrf_timeline.png` - MRF timeline

---

### Periodic

**Description**: Sinusoidal signals with harmonics

#### Dataset Statistics

- **Indicators**: 0
- **Observations**: 0
- **Date Range**:  to 
- **Runtime**: 0.00s

#### Expected Structure

Wavelet/spectral lenses detect periodic components; PCA captures oscillation

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.000
- **Mean Rank Fluctuation**: 0.000

#### Interpretation



#### Potential Failure Modes

- Frequency aliasing
- Phase drift not captured

#### Confidence Evaluation: **LOW**

> **Error**: No data found for benchmark group: periodic

#### Visualizations

- `periodic_timeseries.png` - Time series overview
- `periodic_pca_scatter.png` - PCA projection
- `periodic_correlation.png` - Correlation matrix
- `periodic_lens_agreement.png` - Lens ranking comparison
- `periodic_mrf_timeline.png` - MRF timeline

---

### Pure Noise

**Description**: Pure white noise with no structure (null baseline)

#### Dataset Statistics

- **Indicators**: 0
- **Observations**: 0
- **Date Range**:  to 
- **Runtime**: 0.00s

#### Expected Structure

No lens should show persistent structure; uniform rankings expected

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.000
- **Mean Rank Fluctuation**: 0.000

#### Interpretation



#### Potential Failure Modes

- Spurious structure detected
- Overfitting to noise

#### Confidence Evaluation: **LOW**

> **Error**: No data found for benchmark group: pure_noise

#### Visualizations

- `pure_noise_timeseries.png` - Time series overview
- `pure_noise_pca_scatter.png` - PCA projection
- `pure_noise_correlation.png` - Correlation matrix
- `pure_noise_lens_agreement.png` - Lens ranking comparison
- `pure_noise_mrf_timeline.png` - MRF timeline

---

### Two Regimes

**Description**: Two distinct market regimes with different correlation structures

#### Dataset Statistics

- **Indicators**: 0
- **Observations**: 0
- **Date Range**:  to 
- **Runtime**: 0.00s

#### Expected Structure

Regime-switching lenses should detect transition points

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.000
- **Mean Rank Fluctuation**: 0.000

#### Interpretation



#### Potential Failure Modes

- Regime boundaries missed
- Over-segmentation

#### Confidence Evaluation: **LOW**

> **Error**: No data found for benchmark group: two_regimes

#### Visualizations

- `two_regimes_timeseries.png` - Time series overview
- `two_regimes_pca_scatter.png` - PCA projection
- `two_regimes_correlation.png` - Correlation matrix
- `two_regimes_lens_agreement.png` - Lens ranking comparison
- `two_regimes_mrf_timeline.png` - MRF timeline

---

## Methodology

- **Analysis Window**: 63 days
- **Step Size**: 21 days
- **Lenses Used**: magnitude, pca, influence, clustering, network, anomaly

Each dataset is processed through the PRISM lens pipeline. Rankings are computed
per lens, then aggregated to compute cross-lens agreement (Kendall's tau) and
mean rank fluctuation (MRF) metrics.

## Conclusions

The PRISM engine detected structure in **0/6** datasets.
High-confidence detections: **0/6**

- **pure_noise**: Correctly identified as unstructured (no false positives)
- **clear_leader**: Leader detection below expected confidence

---

*Report generated by PRISM Benchmark Suite v1.0*