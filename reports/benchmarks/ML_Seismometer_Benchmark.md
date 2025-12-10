# ML / Seismometer Benchmarks

- **Date**: 2025-12-10
- **Commit**: `a278a62`
- **Generated**: 2025-12-10 11:50:48

## Configuration

These benchmarks test the ML components of the PRISM Engine:
- **Seismometer**: Ensemble instability detection
- **Regime Classification**: GMM, K-means, and HMM-based regime detection
- **Detector Components**: Individual seismometer detectors

## Methods

Each ML component is tested end-to-end:
1. Load feature data from the database
2. Split into baseline (training) and evaluation periods
3. Fit models on baseline data
4. Score/predict on evaluation data
5. Verify output structure and ranges

## Results

| Component | Samples | Features | Subcomponents | Runtime | Status |
|-----------|---------|----------|---------------|---------|--------|
| seismometer | 1097 | 35 | 0 | 0.41s | FAIL |
| regime_classification | 1097 | 35 | 0 | 0.41s | FAIL |
| detector_components | 732 | 35 | 0 | 0.40s | PASS |

## Detailed Results

### Seismometer

- **Runtime**: 0.41s
- **Samples**: 1097
- **Features**: 35

**Notes:**
- Baseline: 2022-12-09 to 2024-12-09
- Error: No module named 'engine_core.seismometer'

**Error**: No module named 'engine_core.seismometer'

### Regime Classification

- **Runtime**: 0.41s
- **Samples**: 1097
- **Features**: 35

**Notes:**
- Error: Input X contains infinity or a value too large for dtype('float64').

**Error**: Input X contains infinity or a value too large for dtype('float64').

### Detector Components

- **Runtime**: 0.40s
- **Samples**: 732
- **Features**: 35

**Output snapshot:**
- detector_times: {}

**Notes:**
- ClusteringDriftDetector failed: No module named 'engine_core.seismometer'
- ReconstructionErrorDetector failed: No module named 'engine_core.seismometer'
- CorrelationGraphDetector failed: No module named 'engine_core.seismometer'
- Total detector time: 0.00s

## Seismometer Stability Index Reference

The Seismometer produces a stability index from 0 to 1:

| Range | Level | Interpretation |
|-------|-------|----------------|
| 0.90-1.00 | Stable | Normal market behavior |
| 0.70-0.90 | Elevated | Increased noise, heightened attention |
| 0.50-0.70 | Pre-instability | Structural stress emerging |
| 0.30-0.50 | Divergence | Active structural breakdown |
| 0.00-0.30 | High-risk | Imminent or ongoing instability |

## Interpretation

2 component(s) had issues. See error messages above for details.
