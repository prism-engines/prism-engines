# PRISM Engine Benchmark Suite Report

**Generated**: 2025-12-08_220955
**Datasets Analyzed**: 6
**Total Runtime**: 4.0 seconds

## Executive Summary

This report presents the results of running the PRISM analytical engine across
six synthetic benchmark datasets with known statistical properties. Each dataset
tests specific aspects of the engine's detection capabilities.

### Quick Summary

| Dataset | Indicators | Observations | Lens Agreement | Structure Detected | Confidence |
|---------|------------|--------------|----------------|-------------------|------------|
| anomalies | 5 | 1000 | 0.31 | Yes | medium |
| clear_leader | 6 | 1000 | 0.00 | Yes | low |
| clusters | 8 | 1000 | 0.14 | Yes | low |
| periodic | 5 | 1000 | 0.01 | Yes | low |
| pure_noise | 6 | 1000 | -0.06 | Yes | low |
| two_regimes | 6 | 1000 | 0.31 | Yes | medium |

## Detailed Results

### Anomalies

**Description**: Random spikes and outliers embedded in normal behavior

#### Dataset Statistics

- **Indicators**: 5
- **Observations**: 1000
- **Date Range**: 2020-01-01 to 2022-09-26
- **Runtime**: 0.58s

#### Expected Structure

Anomaly lens should flag outlier points; magnitude lens spikes

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.307
- **Mean Rank Fluctuation**: 0.986
- **PCA Explained Variance (PC1-3)**: 35.2%, 20.2%, 17.1%
- **Top Indicators**: anomalies_E, anomalies_C, anomalies_B, anomalies_A, anomalies_D

#### Interpretation

Structure detected in anomalies dataset. First principal component explains 35.2% of variance. Top-ranked indicators: anomalies_E, anomalies_C, anomalies_B Expected: Anomaly lens should flag outlier points; magnitude lens spikes

#### Potential Failure Modes

- Missed anomalies
- False positive rate too high

#### Confidence Evaluation: **MEDIUM**

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

- **Indicators**: 6
- **Observations**: 1000
- **Date Range**: 2020-01-01 to 2022-09-26
- **Runtime**: 0.29s

#### Expected Structure

Single indicator should rank #1 consistently across all lenses

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.004
- **Mean Rank Fluctuation**: 1.495
- **PCA Explained Variance (PC1-3)**: 40.5%, 17.6%, 15.9%
- **Top Indicators**: clear_leader_A, clear_leader_F, clear_leader_E, clear_leader_D, clear_leader_C

#### Interpretation

Structure detected in clear_leader dataset. First principal component explains 40.5% of variance. Low lens agreement (tau=0.00) indicates ambiguous structure. Top-ranked indicators: clear_leader_A, clear_leader_F, clear_leader_E Expected: Single indicator should rank #1 consistently across all lenses

#### Potential Failure Modes

- Leader not detected
- False positives in followers

#### Confidence Evaluation: **LOW**

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

- **Indicators**: 8
- **Observations**: 1000
- **Date Range**: 2020-01-01 to 2022-09-26
- **Runtime**: 0.32s

#### Expected Structure

Clustering lens should identify 3 groups; network lens shows dense subgraphs

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.138
- **Mean Rank Fluctuation**: 1.826
- **PCA Explained Variance (PC1-3)**: 22.8%, 16.9%, 14.2%
- **Top Indicators**: clusters_D, clusters_B, clusters_E, clusters_F, clusters_A

#### Interpretation

Structure detected in clusters dataset. First principal component explains 22.8% of variance. Low lens agreement (tau=0.14) indicates ambiguous structure. Top-ranked indicators: clusters_D, clusters_B, clusters_E Expected: Clustering lens should identify 3 groups; network lens shows dense subgraphs

#### Potential Failure Modes

- Cluster merging
- Spurious cluster detection

#### Confidence Evaluation: **LOW**

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

- **Indicators**: 5
- **Observations**: 1000
- **Date Range**: 2020-01-01 to 2022-09-26
- **Runtime**: 0.27s

#### Expected Structure

Wavelet/spectral lenses detect periodic components; PCA captures oscillation

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.013
- **Mean Rank Fluctuation**: 1.255
- **PCA Explained Variance (PC1-3)**: 40.4%, 23.4%, 18.9%
- **Top Indicators**: periodic_E, periodic_B, periodic_A, periodic_C, periodic_D

#### Interpretation

Structure detected in periodic dataset. First principal component explains 40.4% of variance. Low lens agreement (tau=0.01) indicates ambiguous structure. Top-ranked indicators: periodic_E, periodic_B, periodic_A Expected: Wavelet/spectral lenses detect periodic components; PCA captures oscillation

#### Potential Failure Modes

- Frequency aliasing
- Phase drift not captured

#### Confidence Evaluation: **LOW**

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

- **Indicators**: 6
- **Observations**: 1000
- **Date Range**: 2020-01-01 to 2022-09-26
- **Runtime**: 0.28s

#### Expected Structure

No lens should show persistent structure; uniform rankings expected

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: -0.058
- **Mean Rank Fluctuation**: 1.599
- **PCA Explained Variance (PC1-3)**: 57.3%, 24.5%, 10.8%
- **Top Indicators**: pure_noise_E, pure_noise_F, pure_noise_C, pure_noise_B, pure_noise_D

#### Interpretation

Structure detected in pure_noise dataset. First principal component explains 57.3% of variance. Low lens agreement (tau=-0.06) indicates ambiguous structure. Top-ranked indicators: pure_noise_E, pure_noise_F, pure_noise_C Expected: No lens should show persistent structure; uniform rankings expected

#### Potential Failure Modes

- Spurious structure detected
- Overfitting to noise

#### Confidence Evaluation: **LOW**

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

- **Indicators**: 6
- **Observations**: 1000
- **Date Range**: 2020-01-01 to 2022-09-26
- **Runtime**: 0.28s

#### Expected Structure

Regime-switching lenses should detect transition points

#### Engine Outputs

- **Lens Agreement (Kendall's tau)**: 0.307
- **Mean Rank Fluctuation**: 1.123
- **PCA Explained Variance (PC1-3)**: 61.9%, 26.4%, 6.5%
- **Top Indicators**: two_regimes_C, two_regimes_A, two_regimes_B, two_regimes_D, two_regimes_E

#### Interpretation

Structure detected in two_regimes dataset. First principal component explains 61.9% of variance. Top-ranked indicators: two_regimes_C, two_regimes_A, two_regimes_B Expected: Regime-switching lenses should detect transition points

#### Potential Failure Modes

- Regime boundaries missed
- Over-segmentation

#### Confidence Evaluation: **MEDIUM**

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

The PRISM engine detected structure in **6/6** datasets.
High-confidence detections: **0/6**

- **pure_noise**: WARNING - Structure falsely detected in noise baseline
- **clear_leader**: Leader detection below expected confidence

---

*Report generated by PRISM Benchmark Suite v1.0*