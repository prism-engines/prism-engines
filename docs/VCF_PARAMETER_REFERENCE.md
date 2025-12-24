# VCF Parameter Reference Guide
## Adjustable Parameters by Script and Module

**Version:** 2025-12-08  
**Purpose:** Complete reference for all configurable parameters across the VCF/PRISM system

---

## Table of Contents

1. [CLI Scripts (Command Line)](#cli-scripts)
2. [Core Engine Parameters](#core-engine-parameters)
3. [Lens Parameters](#lens-parameters)
4. [Seismometer Parameters](#seismometer-parameters)
5. [ML Classifier Parameters](#ml-classifier-parameters)
6. [Config File Settings](#config-file-settings)
7. [Quick Reference Card](#quick-reference-card)

---

## CLI Scripts

### 1. `start/fetcher.py`

**Purpose:** Fetch raw data from FRED, Yahoo, Stooq

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| market | `--market`, `-m` | flag | false | Fetch market data (Yahoo) |
| economic | `--economic`, `-e` | flag | false | Fetch economic data (FRED) |
| all | `--all`, `-a` | flag | false | Fetch all data types |
| start_date | `--start-date`, `-s` | string | None | Start date (YYYY-MM-DD) |
| end_date | `--end-date`, `-E` | string | None | End date (YYYY-MM-DD) |
| no_db | `--no-db` | flag | false | Don't write to database |
| test | `--test`, `-t` | flag | false | Test connections only |
| status | `--status` | flag | false | Show database status |
| verbose | `--verbose`, `-v` | flag | false | Verbose output |

**Examples:**
```bash
python start/fetcher.py --all
python start/fetcher.py --market --start-date 2020-01-01
python start/fetcher.py --economic --no-db
python start/fetcher.py --test
```

---

### 2. `start/update_all.py`

**Purpose:** Update all data sources

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| *(inherits from fetcher)* | | | | |

---

### 3. `panel/build_panel.py`

**Purpose:** Build unified analysis panel from data sources

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| output | `--output`, `-o` | string | None | Output CSV path |
| start_date | `--start-date`, `-s` | string | None | Start date (YYYY-MM-DD) |
| end_date | `--end-date`, `-e` | string | None | End date (YYYY-MM-DD) |
| no_save | `--no-save` | flag | false | Don't save output |

**Examples:**
```bash
python panel/build_panel.py --output data/panels/my_panel.csv
python panel/build_panel.py --start-date 2010-01-01 --end-date 2024-01-01
```

---

### 4. `engine_core/orchestration/temporal_runner.py`

**Purpose:** Run all lenses across rolling time windows

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| start | `--start`, `-s` | int | 2005 | Start year |
| end | `--end`, `-e` | int | current | End year |
| increment | `--increment`, `-i` | int | 1 | Years per window |
| panel | `--panel` | string | None | Panel name (maps to master_panel_{name}.csv) |
| data | `--data` | string | None | Direct path to panel CSV |
| output | `--output`, `-o` | string | None | Output directory |
| parallel | `--parallel`, `-p` | flag | false | Use parallel processing |
| workers | `--workers`, `-w` | int | CPU count | Max parallel workers |
| lenses | `--lenses`, `-l` | list | all 14 | Specific lenses to run |
| list_lenses | `--list-lenses` | flag | false | List available lenses |
| export_csv | `--export-csv` | flag | false | Export CSV (default: SQLite only) |

**Examples:**
```bash
python engine_core/orchestration/temporal_runner.py --increment 1
python engine_core/orchestration/temporal_runner.py --start 2010 --end 2024 --increment 2
python engine_core/orchestration/temporal_runner.py --lenses magnitude pca influence --parallel
python engine_core/orchestration/temporal_runner.py --panel climate
```

---

### 5. `start/temporal_runner.py` (Simplified Wrapper)

**Purpose:** Simplified temporal analysis with performance profiles

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| profile | string | 'standard' | Performance profile |
| window_years | float | from profile | Window size in years |
| step_months | float | from profile | Step size in months |
| lenses | list | from profile | Lenses to run |
| verbose | bool | True | Print progress |

**Performance Profiles:**

| Profile | Lenses | Step (months) | Window (years) | Use Case |
|---------|--------|---------------|----------------|----------|
| `chromebook` | 3 (magnitude, pca, influence) | 2.0 | 1.0 | Limited hardware |
| `standard` | 4 (+ clustering) | 1.0 | 1.0 | Normal use |
| `powerful` | 5 (+ decomposition) | 0.5 | 1.0 | High resolution |

**Python Usage:**
```python
from start.temporal_runner import run_temporal_analysis

results = run_temporal_analysis(
    panel,
    profile='standard',       # 'chromebook', 'standard', 'powerful'
    window_years=1.0,         # Override: analysis window
    step_months=1.0,          # Override: step between windows
    lenses=['magnitude', 'pca'],  # Override: specific lenses
    verbose=True
)
```

---

### 6. `engine_core/orchestration/temporal_aggregator.py`

**Purpose:** Aggregate temporal results by regime or time period

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| group | `--group`, `-g` | string | 'regime' | Grouping method |
| db | `--db` | string | auto | Path to temporal.db |
| output | `--output`, `-o` | string | auto | Output directory |

**Group Options:**
- `regime` - Group by detected regime
- `decade` - Group by decade
- `5year` - Group by 5-year periods
- `<integer>` - Custom period (e.g., `3` for 3-year)

**Examples:**
```bash
python engine_core/orchestration/temporal_aggregator.py --group regime
python engine_core/orchestration/temporal_aggregator.py --group decade
python engine_core/orchestration/temporal_aggregator.py --group 5
```

---

### 7. `scripts/run_interpret.py`

**Purpose:** Generate AI interpretations of results

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| command | positional | string | required | Command: window, event, regime, indicator, ask, compare, list, feedback, stats |
| args | positional | list | [] | Arguments for command |
| backend | `--backend` | string | 'manual' | AI backend: manual, claude, openai, ollama |
| model | `--model` | string | auto | Model to use |
| db | `--db` | string | auto | Path to temporal database |
| save | `--save` | flag | true | Save interpretation to database |
| no_save | `--no-save` | flag | false | Don't save |
| output | `--output`, `-o` | string | None | Output file path |
| prompt_only | `--prompt-only` | flag | false | Show prompt only, don't call AI |

**Examples:**
```bash
python scripts/run_interpret.py window 2020
python scripts/run_interpret.py regime --backend claude
python scripts/run_interpret.py indicator spy_close --output interpretation.md
python scripts/run_interpret.py ask "What does the current regime suggest?" --backend openai
```

---

## Core Engine Parameters

### `engine_core/orchestration/temporal_analysis.py`

**TemporalEngine Class:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| panel | DataFrame | required | Input data panel |
| lenses | list | all | Lenses to use |
| min_periods | int | 60 | Minimum data points per window |

**run_rolling_analysis() Method:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| window_days | int | 252 | Window size in trading days |
| step_days | int | 21 | Step size in trading days |
| progress_callback | callable | None | Progress reporting function |

**Conversion Reference:**
```
1 year  = 252 trading days
1 month = 21 trading days
1 week  = 5 trading days
```

---

## Lens Parameters

Each lens has specific parameters passed via `**kwargs` in the `analyze()` method.

### PCA Lens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_components | int | all | Number of principal components |

### Granger Lens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_lag | int | 5 | Maximum lag to test |
| significance | float | 0.05 | Significance level (p-value threshold) |
| max_columns | int | 20 | Maximum columns to analyze (performance) |

### Clustering Lens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_clusters | int | auto | Number of clusters (auto = silhouette optimization) |
| method | string | 'kmeans' | Clustering method |

### Wavelet Lens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| wavelet | string | 'db4' | Wavelet type |
| levels | int | auto | Decomposition levels |

### Network Lens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| threshold | float | 0.5 | Correlation threshold for edges |
| method | string | 'correlation' | Edge weight method |

### Regime Switching Lens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_regimes | int | 3 | Number of regimes to detect |
| method | string | 'hmm' | Detection method |

### Anomaly Lens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| contamination | float | 0.05 | Expected proportion of anomalies |
| method | string | 'isolation_forest' | Detection method |

### TDA Lens (Topological Data Analysis)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_dimension | int | 2 | Maximum homology dimension |
| n_points | int | 100 | Points for persistence diagram |

---

## Seismometer Parameters

### `engine_core/seismometer/ensemble.py`

**SeismometerEnsemble Class:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| conn | connection | None | Database connection |
| weights | dict | see below | Detector weights |
| n_clusters | int | 5 | Clusters for clustering detector |
| encoding_dim | int | 8 | Autoencoder encoding dimension |
| correlation_window | int | 60 | Window for correlation detector |

**Default Weights:**
```python
DEFAULT_WEIGHTS = {
    'clustering': 0.3,
    'autoencoder': 0.25,
    'correlation_graph': 0.25,
    'threshold': 0.2
}
```

---

## ML Classifier Parameters

### `ml/regime_classifier_ensemble.py`

**Hardcoded Settings (modify in file):**

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| PANEL_PATH | line ~45 | data/panels/master_panel.csv | Input panel |
| LABELS_PATH | line ~46 | data/labels/prism_regimes.csv | Regime labels |
| MODEL_PATH | line ~48 | models/regime_ensemble.pkl | Output model |
| windows | _add_rolling_features() | (21, 63, 252) | Rolling window sizes |
| test_size | main() | 0.2 | Test set proportion |

**Ensemble Components:**
- RandomForestClassifier (n_estimators=200)
- GradientBoostingClassifier (n_estimators=100)
- LogisticRegression (max_iter=1000)

---

## Config File Settings

### `config.yaml`

```yaml
# Main configuration file
# Location: project_root/config.yaml

pipeline:
  default_domain: financial
  checkpoint_enabled: true
  log_level: INFO              # DEBUG, INFO, WARNING, ERROR

data:
  raw_dir: 02_data_raw
  clean_dir: 04_data_clean
  default_frequency: daily     # daily, weekly, monthly
  date_column: date

fetch:
  retry_attempts: 3
  retry_delay_seconds: 2
  timeout_seconds: 30

cleaning:
  default_nan_strategy: ffill  # ffill, bfill, interpolate, drop
  max_nan_pct: 10.0            # Max NaN percentage before warning
  outlier_detection: true
  outlier_method: iqr          # iqr, zscore
  outlier_threshold: 3.0

engine:
  default_mode: basic          # basic, advanced, all
  basic_lenses:
    - magnitude
    - pca
    - influence
    - clustering
    - decomposition
  advanced_lenses:
    - granger
    - mutual_info
    - wavelet
    - network
    - regime
    - anomaly
  consensus_method: borda      # borda, average, median

output:
  dir: 06_output
  archive_enabled: true
  formats:
    - json
    - csv

visualization:
  figure_format: png           # png, svg, pdf
  figure_dpi: 150
  default_figsize: [12, 8]
  color_scheme: viridis

validation:
  n_trials: 5
  min_accuracy: 0.6
```

---

## Quick Reference Card

### Most Common Adjustments

| What You Want | Where to Change | Parameter |
|---------------|-----------------|-----------|
| **Date range** | CLI: `--start-date`, `--end-date` | Any fetcher/panel script |
| **Analysis window** | `temporal_runner.py` | `window_years` or `window_days` |
| **Step size** | `temporal_runner.py` | `step_months` or `step_days` |
| **Which lenses** | `temporal_runner.py` | `--lenses` or `lenses=[]` |
| **Performance mode** | `start/temporal_runner.py` | `profile='chromebook'` |
| **Parallel processing** | `temporal_runner.py` | `--parallel`, `--workers` |
| **Output location** | Any script | `--output`, `-o` |
| **Significance level** | Lens kwargs | `significance=0.05` |
| **Number of regimes** | Seismometer/lens | `n_regimes=4` |

### Resolution Presets (New)

| Resolution | Frequency | Window | Step | Use Case |
|------------|-----------|--------|------|----------|
| weekly | W-FRI | 52 weeks | 13 weeks | Tactical |
| monthly | M | 60 months | 15 months | Strategic (default) |
| quarterly | Q | 40 quarters | 10 quarters | Structural |

### Trading Day Conversions

```
1 year   = 252 days
1 quarter = 63 days
1 month  = 21 days
1 week   = 5 days
```

---

## Parameter Modification Examples

### Example 1: Quick 10-Year Monthly Analysis

```python
from engine_core.orchestration.temporal_runner import TemporalRunner

runner = TemporalRunner()
results = runner.run(
    start_year=2015,
    end_year=2025,
    window_days=252,      # 1 year window
    step_days=21,         # 1 month steps
    lenses=['magnitude', 'pca', 'regime']
)
```

### Example 2: High-Resolution Recent Analysis

```python
results = run_temporal_analysis(
    panel,
    profile='powerful',
    window_years=0.5,     # 6-month window
    step_months=0.25,     # ~1 week steps
    lenses=['magnitude', 'pca', 'influence', 'regime', 'anomaly']
)
```

### Example 3: Custom Lens Configuration

```python
from engine_core.lenses.granger_lens import GrangerLens

granger = GrangerLens()
results = granger.analyze(
    df,
    max_lag=10,           # Test up to 10 lags
    significance=0.01,    # Stricter significance
    max_columns=30        # Analyze more columns
)
```

### Example 4: Custom Seismometer

```python
from engine_core.seismometer.ensemble import SeismometerEnsemble

seis = SeismometerEnsemble(
    n_clusters=4,
    encoding_dim=16,
    correlation_window=90,
    weights={
        'clustering': 0.4,
        'autoencoder': 0.3,
        'correlation_graph': 0.2,
        'threshold': 0.1
    }
)
regime = seis.detect_current_regime()
```

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `FRED_API_KEY` | FRED data access | `export FRED_API_KEY=your_key` |
| `ANTHROPIC_API_KEY` | Claude interpretation | `export ANTHROPIC_API_KEY=your_key` |
| `OPENAI_API_KEY` | OpenAI interpretation | `export OPENAI_API_KEY=your_key` |
| `PRISM_DB_PATH` | Override database path | `export PRISM_DB_PATH=/path/to/db` |

---

*"When in doubt, start with defaults. Adjust only when you understand why."*
