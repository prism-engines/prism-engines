# PRISM Engine Quick Start Guide

## Before You Start (One-Time Setup)

Make sure these are installed (you already did this):
```bash
pip install pandas numpy scipy matplotlib scikit-learn tqdm
```

---

## Step-by-Step: Running an Analysis

### Step 1: Open Terminal

- Press `Cmd + Space`
- Type `Terminal`
- Press Enter

### Step 2: Navigate to PRISM Folder

Copy and paste this exact line:
```bash
cd "/Users/jasonrudder/Library/CloudStorage/GoogleDrive-rudder.jason@gmail.com/My Drive/prism-engine/prism-engine"
```

Press Enter.

### Step 3: Choose Your Data (Optional)

**If using default market data:** Skip to Step 4.

**If using different data:**
```bash
# See what panels are available
ls data/panels/

# Copy your desired panel to the active location
cp data/panels/master_panel_climate.csv data/raw/master_panel.csv
```

### Step 4: Run the Analysis

**Standard run (1-year windows, 1970-2025):**
```bash
python 05_engine/orchestration/temporal_runner.py --start 1970 --end 2025 --export-csv
```

**Custom date range:**
```bash
python 05_engine/orchestration/temporal_runner.py --start 1990 --end 2025 --export-csv
```

**Different window size (e.g., 5-year):**
```bash
python 05_engine/orchestration/temporal_runner.py --start 1970 --end 2025 --increment 5 --export-csv
```

Wait for it to finish. You'll see progress like:
```
[1/55] Processing 1970-1971...
[2/55] Processing 1971-1972...
...
COMPLETE
```

### Step 5: Generate Visualizations

```bash
python 05_engine/orchestration/temporal_visualizer.py --top 15
```

### Step 6: View Summary

```bash
python 05_engine/orchestration/temporal_visualizer.py --summary
```

### Step 7: View Your Results

**Open the plots folder:**
```bash
open 06_output/temporal/plots/
```

**Open the database in DB Browser:**
```bash
open 06_output/temporal/prism_temporal.db
```

**View CSV files:**
```bash
open 06_output/temporal/
```

---

## Quick Reference Commands

| What You Want | Command |
|---------------|---------|
| Run default analysis | `python 05_engine/orchestration/temporal_runner.py --export-csv` |
| Run with 5-year windows | `python 05_engine/orchestration/temporal_runner.py --increment 5 --export-csv` |
| See summary stats | `python 05_engine/orchestration/temporal_visualizer.py --summary` |
| Generate plots | `python 05_engine/orchestration/temporal_visualizer.py --top 15` |
| Open results folder | `open 06_output/temporal/` |
| Open plots | `open 06_output/temporal/plots/` |
| Check regime stability | `cat 06_output/temporal/regime_stability_1yr.csv` |

---

## Switching Data Panels

**To switch to climate data:**
```bash
cp data/panels/master_panel_climate.csv data/raw/master_panel.csv
python 05_engine/orchestration/temporal_runner.py --export-csv
```

**To switch back to market data:**
```bash
cp data/panels/master_panel_markets.csv data/raw/master_panel.csv
python 05_engine/orchestration/temporal_runner.py --export-csv
```

---

## Troubleshooting

**"No module named X"**
```bash
pip install X
```

**"No such file or directory"**
Make sure you're in the right folder:
```bash
cd "/Users/jasonrudder/Library/CloudStorage/GoogleDrive-rudder.jason@gmail.com/My Drive/prism-engine/prism-engine"
pwd
```
Should show the prism-engine path.

**"master_panel.csv not found"**
```bash
ls data/raw/
```
Make sure master_panel.csv exists there.

---

## Output Files Explained

| File | What It Contains |
|------|------------------|
| `prism_temporal.db` | Full database (query with SQL) |
| `temporal_results_1yr.csv` | All rankings, all windows, long format |
| `rank_evolution_1yr.csv` | Rankings pivoted (indicators × windows) |
| `regime_stability_1yr.csv` | Spearman correlations between windows |
| `plots/rank_trajectory.png` | Line chart of rank changes |
| `plots/rank_heatmap.png` | Heatmap of all rankings |
| `plots/regime_stability.png` | Line chart of structural stability |

---

## Using the SQL Helper

After running analysis, you can query results in Python:

```python
from prism_sql_helper import connect, top_indicators, indicator_history, regime_shifts

conn = connect()

# Top 10 for a specific window
top_indicators(conn, window_start=2005)

# Track one indicator over time
indicator_history(conn, 'walcl')

# See regime stability scores
regime_shifts(conn)
```

---

## Running Different Scenarios

### Scenario 1: US Markets (Default)
```bash
python 05_engine/orchestration/temporal_runner.py --start 1970 --end 2025 --export-csv
```

### Scenario 2: Climate Indicators
```bash
cp data/panels/master_panel_climate.csv data/raw/master_panel.csv
python 05_engine/orchestration/temporal_runner.py --start 1950 --end 2025 --export-csv
```

### Scenario 3: Global Markets
```bash
cp data/panels/master_panel_global.csv data/raw/master_panel.csv
python 05_engine/orchestration/temporal_runner.py --start 1990 --end 2025 --export-csv
```

### Scenario 4: Test/Validation
```bash
cp data/panels/master_panel_test1.csv data/raw/master_panel.csv
python 05_engine/orchestration/temporal_runner.py --start 2000 --end 2025 --export-csv
```

---

## Full Workflow Example

```bash
# 1. Open terminal and navigate
cd "/Users/jasonrudder/Library/CloudStorage/GoogleDrive-rudder.jason@gmail.com/My Drive/prism-engine/prism-engine"

# 2. Run analysis
python 05_engine/orchestration/temporal_runner.py --start 1970 --end 2025 --export-csv

# 3. Generate plots
python 05_engine/orchestration/temporal_visualizer.py --top 15

# 4. View summary
python 05_engine/orchestration/temporal_visualizer.py --summary

# 5. Open results
open 06_output/temporal/plots/
```

---

*PRISM Engine v1.0*
