# PRISM Engine - Repository Cleanup Audit Report

**Generated:** 2025-12-07
**Repository:** prism-engine
**Current Size:** 144 MB total (87 MB excluding .git)

---

## Executive Summary

Your repository is **significantly bloated** due to generated output files, trained ML models, and log files that were committed before `.gitignore` rules were added. The actual source code is only ~2-3 MB.

| Category | Size | % of Repo | Recommendation |
|----------|------|-----------|----------------|
| Git History | 57 MB | 40% | Consider shallow clone for distribution |
| ML Models (.pkl) | 50 MB | 35% | Move to artifact storage |
| Output Files | 27 MB | 19% | Remove from tracking |
| Data Files | 4.9 MB | 3% | Keep selectively |
| Log Files | 2.6 MB | 2% | Remove from tracking |
| **Source Code** | **~3 MB** | **2%** | Keep |

---

## Tier 1: Remove Immediately (Safe) - ~30 MB Savings

These files are generated outputs and can be safely removed from git tracking.

### 1.1 Output Directory (27 MB)
The entire `output/` directory contains generated analysis results. All files can be regenerated.

| File | Size | Description |
|------|------|-------------|
| `output/overnight_lite/multiresolution.csv` | 12 MB | Generated analysis data |
| `output/correlation_movie/correlation_evolution.gif` | 7.5 MB | Generated animation |
| `output/full_40y_analysis/consensus_timeline_40y.png` | 1.8 MB | Generated chart |
| `output/overnight_lite/multiresolution_heatmap.png` | 1.7 MB | Generated heatmap |
| `output/lens_analysis/lens_contributions.png` | 1.2 MB | Generated chart |
| `output/contributions/indicator_contributions.png` | 890 KB | Generated chart |
| `output/full_40y_analysis/lens_scores_40y.csv` | 445 KB | Generated scores |
| `output/full_40y_analysis/lens_normalized_40y.csv` | 388 KB | Generated data |
| `output/coherence/coherence_timeline.png` | 356 KB | Generated chart |
| `output/correlation/correlation_heatmap.png` | 290 KB | Generated chart |
| `output/overnight_lite/early_warning_lines.png` | 289 KB | Generated chart |
| + 15 more files... | ~500 KB | Various outputs |

**Action:**
```bash
# Remove from git tracking (keeps local files)
git rm -r --cached output/
git commit -m "Remove generated output files from tracking"
```

### 1.2 Log Files (2.6 MB)

| File | Size | Description |
|------|------|-------------|
| `overnight.log` | 2.6 MB | Runtime log file |

**Action:**
```bash
git rm --cached overnight.log
git commit -m "Remove log file from tracking"
```

### 1.3 Backup Directories (~90 KB)

| Directory | Size | Description |
|-----------|------|-------------|
| `.backup_20251206_163000/` | 64 KB | Old backup with deprecated code |
| `fetch/backups/` | 26 KB | Backup fetcher scripts |

**Action:**
```bash
git rm -r --cached .backup_20251206_163000/
git rm -r --cached fetch/backups/
git commit -m "Remove backup directories from tracking"
```

---

## Tier 2: Consider Removing - ~50 MB Savings

These files require a decision based on your workflow.

### 2.1 ML Model Files (50 MB)

| File | Size | Description |
|------|------|-------------|
| `models/regime_rf_pro.pkl` | 27 MB | Random Forest model (pro) |
| `models/regime_rf.pkl` | 7.8 MB | Random Forest model |
| `models/regime_ensemble.pkl` | 7.4 MB | Ensemble model |
| `models/regime_ensemble_market.pkl` | 4.4 MB | Market ensemble |
| `models/regime_ensemble_post_etf.pkl` | 2.5 MB | Post-ETF ensemble |
| `models/regime_rf_v2.pkl` | 189 KB | RF model v2 |
| `models/prism_ml_engine.pkl` | 69 KB | ML engine state |

**Options:**
1. **Keep in repo** - If models rarely change and are needed for reproducibility
2. **Move to releases** - Attach as GitHub release assets
3. **Use Git LFS** - Large File Storage for better handling
4. **External storage** - S3, GCS, or model registry (MLflow, DVC)

**If removing:**
```bash
git rm -r --cached models/*.pkl
git commit -m "Move ML models to artifact storage"
```

### 2.2 Data Panel Files (4.8 MB)

| File | Size | Description |
|------|------|-------------|
| `data/panels/master_panel.csv` | 3.6 MB | Master data panel |
| `data/labels/prism_regimes.csv` | 657 KB | Regime labels |
| `data/labels/prism_ml_states.csv` | 515 KB | ML state labels |

**Recommendation:** Keep `data/` if this data cannot be regenerated from APIs (FRED, Yahoo, etc.). If it can be fetched fresh, consider removing.

---

## Tier 3: Keep (Source Code & Config)

These should remain in the repository:

| Directory | Size | Description |
|-----------|------|-------------|
| `engine_core/` | 320 KB | Core engine code |
| `start/` | 451 KB | CLI entry points |
| `tests/` | 190 KB | Test suite |
| `utils/` | 151 KB | Utility functions |
| `fetch/` | 96 KB | Data fetchers |
| `interpretation/` | 115 KB | Analysis interpretation |
| `engine/` | 65 KB | Engine modules |
| `validation/` | 62 KB | Validation code |
| `cleaning/` | 60 KB | Data cleaning |
| `ml/` | 59 KB | ML modules |
| `visualization/` | 29 KB | Visualization code |
| `panel/` | 36 KB | Panel management |
| `scripts/` | 23 KB | Utility scripts |
| `memory/` | 15 KB | Memory management |
| `data/registry/` | 32 KB | Config registries (JSON) |

---

## Tier 4: Optional - Documentation Assets (2 MB)

| File | Size | Description |
|------|------|-------------|
| `documentation/reports/PRISM_Report.pdf` | 926 KB | Main report PDF |
| `documentation/reports/*.png` | 771 KB | Report figures |
| `documentation/*.md` | ~200 KB | Markdown docs |
| `docs/` | 45 KB | Additional docs |

**Recommendation:** Keep documentation. Consider if the PDF and PNGs in `documentation/reports/` need to be tracked, or if they can be generated from source.

---

## Issue: Files Already in .gitignore But Still Tracked

Your `.gitignore` already includes rules for:
- `output/`
- `*.csv`
- `*.png`
- `*.log`
- `*.db`

However, these files are still tracked because they were committed before the rules were added. Git continues tracking files that were already added, regardless of `.gitignore`.

---

## Recommended Cleanup Commands

### Quick Cleanup (Tier 1 Only - 30 MB)
```bash
# Remove generated files from tracking
git rm -r --cached output/
git rm --cached overnight.log
git rm -r --cached .backup_20251206_163000/
git rm -r --cached fetch/backups/

# Commit the removal
git commit -m "Remove generated outputs, logs, and backups from tracking"

# Push changes
git push
```

### Full Cleanup (Tier 1 + 2 - 80 MB)
```bash
# Remove output files
git rm -r --cached output/
git rm --cached overnight.log
git rm -r --cached .backup_20251206_163000/
git rm -r --cached fetch/backups/

# Remove ML models (only if you have alternative storage)
git rm -r --cached models/*.pkl

# Commit
git commit -m "Remove generated files and move models to external storage"

# Push
git push
```

### Reduce Git History Size (Advanced)
If you want to remove these files from history entirely to reduce the .git folder size:
```bash
# WARNING: This rewrites history - coordinate with collaborators
git filter-branch --force --index-filter \
  'git rm -r --cached --ignore-unmatch output/ overnight.log models/*.pkl' \
  --prune-empty -- --all

# Or use BFG Repo Cleaner (faster):
# bfg --delete-folders output --delete-files '*.pkl'
# git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

---

## Expected Results After Cleanup

| Scenario | Repo Size | ZIP Size (est.) |
|----------|-----------|-----------------|
| Current | 144 MB | ~40 MB |
| After Tier 1 | ~114 MB | ~25 MB |
| After Tier 1+2 | ~64 MB | ~10 MB |
| After History Rewrite | ~10 MB | ~3 MB |

---

## Updated .gitignore Additions

Your `.gitignore` is already well-configured. Just ensure these patterns remain:

```gitignore
# Already present - good!
output/
*.csv
*.png
*.log
*.db

# Consider adding for ML models
models/*.pkl
!models/.gitkeep

# Backup directories
.backup_*/
**/backups/
```

---

## Summary

| Action | Effort | Size Saved | Risk |
|--------|--------|------------|------|
| Remove `output/` | Low | 27 MB | None |
| Remove `overnight.log` | Low | 2.6 MB | None |
| Remove backups | Low | 90 KB | None |
| Remove model `.pkl` files | Medium | 50 MB | Need alt storage |
| Rewrite git history | High | 57 MB | Breaks forks/clones |

**Recommended:** Start with Tier 1 cleanup for immediate 30 MB reduction with zero risk.
