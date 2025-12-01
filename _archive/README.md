# Archived Code

This directory contains old exploratory code that is **not part of the main PRISM pipeline**.

These files are preserved for reference but are not integrated into the current model.

## Directory Structure

### `exploratory-validation/`
Experimental validation frameworks developed during research phase:
- `backtester.py` - Historical backtesting against known market events
- `bootstrap_analysis.py` - Bootstrap confidence interval analysis
- `lens_validator.py` - Synthetic data validation (exploratory version)
- `permutation_tests.py` - Permutation/shuffle testing
- `validation_report.py` - Comprehensive validation reporting
- `system-tests.md` - Validation framework design notes

**Note:** The main validation code is in `/validation/lens_validator.py`

### `exploratory-notebooks/`
Quick-start and analysis notebooks for exploration:
- `PRISM_QuickStart.ipynb` - Getting started guide
- `PRISM_Temporal_Analysis.ipynb` - Temporal analysis examples

**Note:** Main notebooks are in project root: `PRISM.ipynb`, `PRISM_14lens.ipynb`

### `benchmark-data/`
Synthetic benchmark datasets with known ground truth for testing:
- `benchmark_generator.py` - Script that generates test data
- `benchmark_01_clear_leader.csv` - Known causal leader pattern
- `benchmark_02_two_regimes.csv` - Known regime change
- `benchmark_03_clusters.csv` - Known cluster structure
- `benchmark_04_periodic.csv` - Known periodicities
- `benchmark_05_anomalies.csv` - Known anomaly injection
- `benchmark_06_pure_noise.csv` - Control (no structure)

### `alternative-entry-points/`
Alternative scripts that duplicate functionality in main modules:
- `main.py` - Alternative entry point (use `loader.py` instead)
- `sql_helper.py` - Database query helper utility

## Main Pipeline (NOT archived)

The integrated pipeline remains in place:
```
01_fetch/          -> Data fetching
03_cleaning/       -> Data cleaning
05_engine/         -> Core analysis (14 lenses)
06_output/         -> Results storage
```

Entry points:
- `loader.py` - Import helper and lens runner
- `fetcher.py` - Data fetching
- `start/temporal_runner.py` - Temporal analysis
- `PRISM.ipynb` / `PRISM_14lens.ipynb` - Notebooks

## Restoration

To restore any archived code, simply move files back to their original locations.

---
*Archived: 2024-11-29*
