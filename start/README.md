# PRISM Start Scripts

Entry points for running PRISM analysis. Run all commands from this directory.

## Setup

```bash
cd ~/prism-engine
source venv/bin/activate
cd start
```

## Scripts

### 1. update_all.py - Fetch Data

Fetches economic/market data from FRED, Tiingo, Yahoo.

```bash
python update_all.py              # Update all indicators
python update_all.py --force      # Force refresh (ignore cache)
```

**Requires**: API keys in `.env` file at project root:
```
FRED_API_KEY=your_key
TIINGO_API_KEY=your_key
```

---

### 2. analyze.py - Main Analysis

Runs all lenses on current data, saves to database.

```bash
python analyze.py                           # Standard run (13 lenses)
python analyze.py --weighted combined       # With geometry weights
python analyze.py --weighted independence   # Independence weights only
python analyze.py --list                    # Show available lenses
python analyze.py --lenses granger pca      # Run specific lenses
```

**Output**: Results saved to `data/prism.db`, summary printed to terminal.

**Note**: Skip TDA lens on low-memory machines (kills Chromebook):
```bash
python analyze.py --lenses granger regime clustering wavelet anomaly pca network transfer_entropy mutual_info magnitude decomposition dmd influence
```

---

### 3. lens_geometry.py - Compute Lens Weights

Analyzes lens correlations and computes independence weights.

```bash
python lens_geometry.py           # Analyze latest run
python lens_geometry.py --save    # Save weights to database
python lens_geometry.py --run 2   # Analyze specific run
```

**Output**: Correlation matrix, clusters, redundancy analysis, weights.

Run this AFTER `analyze.py` to compute weights for `--weighted` flag.

---

### 4. compare_consensus.py - Weighted vs Unweighted

Shows how weighting changes rankings.

```bash
python compare_consensus.py           # Latest run
python compare_consensus.py --run 3   # Specific run
python compare_consensus.py --export  # Export to CSV
```

**Output**: Side-by-side rankings, biggest risers/fallers.

---

### 5. query_results.py - Database Queries

Query historical analysis runs.

```bash
python query_results.py               # List all runs
python query_results.py --run 3       # Details for run 3
python query_results.py --indicator spy   # History for specific indicator
```

---

### 6. benchmark.py - Validate Lenses

Tests lenses against synthetic data with known ground truth.

```bash
python benchmark.py           # Run all benchmarks
python benchmark.py --v1      # Original 6 lenses
python benchmark.py --v2      # Newer 4 lenses
python benchmark.py --regen   # Regenerate test data first
```

**Expected**: 10/10 passing (or 13/14 with TDA skipped).

---

### 7. temporal.py - Rolling Window Analysis

Runs analysis across sliding time windows.

```bash
python temporal.py                    # Use config defaults
python temporal.py --windows 63 126   # Custom window sizes (days)
python temporal.py --step 10          # Step size between windows
```

---

### 8. full_monty.py - Comprehensive Analysis

Runs everything: update, analyze, geometry, temporal.

```bash
python full_monty.py          # Full pipeline
```

**Warning**: Heavy compute. Use on Mac, not Chromebook.

---

### 9. analyze_lean.py - Lightweight Version

7-lens subset for quick testing.

```bash
python analyze_lean.py        # Fast analysis
```

---

## Typical Workflow

```bash
# 1. First time setup
python update_all.py

# 2. Run analysis
python analyze.py

# 3. Compute lens weights
python lens_geometry.py --save

# 4. Run weighted analysis
python analyze.py --weighted combined

# 5. Compare results
python compare_consensus.py
```

## Configuration

Edit `prism_config.yaml` to change:
- Date range
- Indicator list
- Lens selection
- Window parameters

## Output

- Results: `../data/prism.db` (SQLite)
- Charts: `output/` folder
- Logs: Terminal output
