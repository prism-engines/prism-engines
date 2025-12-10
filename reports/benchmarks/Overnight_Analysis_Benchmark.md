# Overnight Analysis Benchmarks

- **Date**: 2025-12-10
- **Commit**: `a278a62`
- **Generated**: 2025-12-10 11:50:48

## Configuration

These benchmarks test the overnight analysis scripts by running their core components with reduced parameters for faster testing.

**Scripts tested:**
- `overnight_lite.py`: Fast version (30-60 min in production)
- `overnight_analysis.py`: Full version (6-12 hours in production)

## Methods

The benchmarks exercise each major component of the overnight scripts:
- Multi-resolution analysis (rolling windows at multiple time scales)
- Bootstrap confidence intervals
- Monte Carlo null distribution generation
- HMM regime analysis (if hmmlearn available)
- GMM regime detection

## Results

| Script | Mode | Indicators | Windows | Components | Runtime | Status |
|--------|------|------------|---------|------------|---------|--------|
| overnight_lite.py | benchmark | 35 | 150 | 3 | 0.47s | PASS |
| overnight_analysis.py | benchmark | 35 | 114 | 5 | 0.47s | PASS |

## Detailed Results

### overnight_lite.py (benchmark)

- **Runtime**: 0.47s
- **Indicators**: 35
- **Windows**: 150
- **Parallel**: Yes
- **Workers**: 11

**Components tested:**
- multiresolution
- bootstrap
- regime_detection

**Notes:**
- Years back: 3
- Window sizes: [21, 63, 126]
- Step size: 10
- Components: multiresolution, bootstrap, regime_detection

### overnight_analysis.py (benchmark)

- **Runtime**: 0.47s
- **Indicators**: 35
- **Windows**: 114
- **Parallel**: Yes
- **Workers**: 11

**Components tested:**
- multiresolution
- bootstrap
- monte_carlo
- hmm
- gmm_regime

**Notes:**
- Years back: 2
- Window sizes: [21, 63, 126, 252]
- Monte Carlo sims: 5
- HMM available and tested
- Components: multiresolution, bootstrap, monte_carlo, hmm, gmm_regime

## Interpretation

All overnight script components executed successfully. The benchmark parameters were reduced for testing speed, but the same code paths used in production were exercised.

**Component coverage**: bootstrap, gmm_regime, hmm, monte_carlo, multiresolution, regime_detection
