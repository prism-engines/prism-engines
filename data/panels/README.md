# PRISM Master Panels

This directory contains analysis-ready master panel datasets for different analysis scenarios.
Each panel is a consolidated CSV with indicators aligned on a common date index.

## Panel Naming Convention

- `master_panel.csv` - Default panel (current US markets)
- `master_panel_{name}.csv` - Named variant panels

## Available Panels

### master_panel.csv (default)
- **Description**: US market indicators for core analysis
- **Indicators**: 32 indicators
- **Date Range**: 1970-2025
- **Sources**: FRED, Yahoo Finance
- **Use Case**: Standard PRISM temporal analysis

### master_panel_climate.csv
- **Description**: Climate and environmental indicators
- **Indicators**: TBD
- **Date Range**: TBD
- **Sources**: NOAA, EPA, World Bank
- **Use Case**: Climate risk analysis

### master_panel_global.csv
- **Description**: International market indicators
- **Indicators**: TBD
- **Date Range**: TBD
- **Sources**: World Bank, IMF, regional exchanges
- **Use Case**: Cross-market analysis

### master_panel_test1.csv
- **Description**: Engine validation test dataset
- **Indicators**: Synthetic + subset of real indicators
- **Date Range**: Matches production data
- **Sources**: Generated/sampled
- **Use Case**: Engine validation, CI/CD testing

## Usage

```bash
# Use default panel (master_panel.csv)
python temporal_runner.py --increment 1

# Use specific panel
python temporal_runner.py --increment 1 --panel climate
python temporal_runner.py --increment 1 --panel global
python temporal_runner.py --increment 1 --panel test1
```

The `--panel` argument maps to `data/panels/master_panel_{name}.csv`.

## Creating New Panels

1. Prepare a CSV with:
   - First column: Date index (datetime parseable)
   - Remaining columns: Indicator values (numeric)

2. Save as `master_panel_{your_name}.csv` in this directory

3. Update this README with panel metadata

4. Run temporal analysis:
   ```bash
   python temporal_runner.py --panel your_name
   ```

## Panel Requirements

- **Date Column**: Must be the first column, datetime-parseable
- **Numeric Values**: All indicator columns must be numeric
- **NaN Handling**: Missing values are forward-filled, then backward-filled
- **Minimum Data**: At least 100 data points per analysis window
