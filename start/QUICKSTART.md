# PRISM Engine Quick Start Guide

## 1. Open Terminal and Navigate to PRISM
```bash
cd ~/Library/CloudStorage/GoogleDrive-rudder.jason@gmail.com/My\ Drive/prism-engine/prism-engine
```

## 2. Run the Analysis

**Default (1-year windows, full history):**
```bash
python 05_engine/orchestration/temporal_runner.py --start 1970 --end 2025 --export-csv
```

**With a different panel:**
```bash
python 05_engine/orchestration/temporal_runner.py --panel climate --start 1970 --end 2025 --export-csv
```

**Shorter time range:**
```bash
python 05_engine/orchestration/temporal_runner.py --start 2000 --end 2025 --export-csv
```

## 3. Generate Visualizations
```bash
python 05_engine/orchestration/temporal_visualizer.py --increment 1 --top 15
```

## 4. View Results

**Open plots:**
```bash
open 06_output/temporal/plots/
```

**View top indicators:**
```bash
cat 06_output/temporal/temporal_results_1yr.csv | head -20
```

**View regime stability:**
```bash
cat 06_output/temporal/regime_stability_1yr.csv
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Run full analysis | `python 05_engine/orchestration/temporal_runner.py --start 1970 --end 2025 --export-csv` |
| Use climate panel | `--panel climate` |
| Generate plots | `python 05_engine/orchestration/temporal_visualizer.py --increment 1 --top 15` |
| Run tests | `cd tests && pytest . -v` |
| Aggregate by regime | `python 05_engine/orchestration/temporal_aggregator.py --group regime` |

---

## Available Panels

Located in `data/panels/`:
- `master_panel.csv` (default - US markets)
- `master_panel_climate.csv`
- `master_panel_global.csv`
- `master_panel_test1.csv`

---

## Output Files

After running, find results in `06_output/temporal/`:
- `prism_temporal.db` - SQLite database (primary)
- `temporal_results_1yr.csv` - Consensus rankings
- `rank_evolution_1yr.csv` - Rank changes over time
- `regime_stability_1yr.csv` - Year-over-year correlations
- `plots/` - Visualization PNGs

---

## Troubleshooting

**"No module named X":**
```bash
pip install X --break-system-packages
```

**FRED API errors:**
```bash
export FRED_API="3fd12c9d0fa4d7fd3c858b72251e3388"
```

**Permission errors on Google Drive path:**
Make sure Google Drive is running and synced.
