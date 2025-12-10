# Engine Performance Benchmarks

- **Date**: 2025-12-10
- **Commit**: `a278a62`
- **Database**: `/home/rudderjason/prism_data/prism.db`
- **Generated**: 2025-12-10 11:50:48

## Configuration

**Benchmark Profiles:**

- **standard**: Baseline full run with core lenses
  - Lenses: magnitude, pca, influence, clustering
  - Window: 252 days, Step: 21 days
- **powerful**: Full lens suite with finer stepping
  - Lenses: magnitude, pca, influence, clustering, network
  - Window: 252 days, Step: 10 days
- **reduced**: Reduced panel for scaling comparison
  - Lenses: magnitude, pca, influence
  - Window: 252 days, Step: 21 days
- **minimal**: Minimal config for quick testing
  - Lenses: magnitude, pca
  - Window: 126 days, Step: 42 days

## Methods

Performance benchmarks load real data from the database and run temporal analysis with different configurations. Each profile tests a different combination of lenses, window sizes, and step intervals.

Metrics measured:
- **Runtime**: Wall-clock time for full analysis
- **Panel size**: Memory footprint of input data
- **Result size**: Memory footprint of output
- **Windows**: Number of rolling windows analyzed

## Results

| Profile | Indicators | Lenses | Windows | Runtime | Panel | Result | Status |
|---------|------------|--------|---------|---------|-------|--------|--------|
| minimal | 15 | 2 | 0 | 0.06s | 1.2 MB | 0.001 MB | PASS |
| standard | 61 | 4 | 0 | 0.13s | 4.5 MB | 0.003 MB | PASS |

## Detailed Results

### minimal

- **Profile**: minimal
- **Indicators**: 15
- **Lenses**: magnitude, pca
- **Windows analyzed**: 0
- **Runtime**: 0.06s
- **Panel size**: 1.2 MB
- **Result size**: 0.001 MB
- **Date range**: 2000-01-01 to 2025-12-09

**Notes:**
- DB: /home/rudderjason/prism_data/prism.db
- Load time: 0.53s

### standard

- **Profile**: standard
- **Indicators**: 61
- **Lenses**: magnitude, pca, influence, clustering
- **Windows analyzed**: 0
- **Runtime**: 0.13s
- **Panel size**: 4.5 MB
- **Result size**: 0.003 MB
- **Date range**: 2000-01-01 to 2025-12-09

**Notes:**
- DB: /home/rudderjason/prism_data/prism.db
- Load time: 0.40s

## Interpretation

- **Fastest**: minimal (0.06s)
- **Slowest**: standard (0.13s)
- **Average result/panel ratio**: 0.00x
