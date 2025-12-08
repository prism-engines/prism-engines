# PRISM Engine Refactor Report v1

**Generated:** 2025-12-07
**Branch:** claude/audit-remove-legacy-01WhhYm2h4K1psGJ7dD6vSz5
**Scope:** Domain-Agnostic Audit, Legacy Removal, Repository Cleanup

---

## Executive Summary

This audit identifies legacy finance-specific terminology, broken import paths, deprecated SQL references, and files/branches recommended for removal. The goal is to transition PRISM to a fully **domain-agnostic** architecture supporting finance, climate, biology, energy, and other domains.

---

## 1. Non-Domain-Agnostic Logic Found

### 1.1 Legacy Regime Terminology (HIGH PRIORITY)

These files contain finance-specific regime labels that violate domain-agnostic principles:

| File | Line(s) | Issue | Recommended Action |
|------|---------|-------|-------------------|
| `ml/regime_classifier_rf.py` | 7-9, 124-143, 242-243 | Uses `risk_on`, `risk_off`, `neutral` based on SPY returns | **Replace with domain-agnostic states** |
| `ml/regime_ensemble_post_etf.py` | 252-254 | Uses `risk_on`, `risk_off`, `neutral` | Replace with generic state labels |
| `ml/build_prism_labels.py` | 8-17 | `compute_regime_from_mrf()` returns `risk_on`, `risk_off`, `neutral` | Rename to `compute_state_from_signal()` |
| `engine_core/lenses/regime_switching_lens.py` | 22, 28, 96 | References "bull/bear", "Market regime detection" | Update to "system state detection" |

**Recommended Terminology Mapping:**
```
risk_on    -> elevated / expansion / high_activity
risk_off   -> depressed / contraction / low_activity
neutral    -> baseline / stable / moderate
bull/bear  -> expansion/contraction or positive/negative
regime     -> state / phase / mode
```

### 1.2 Hardcoded Market Indexes in Engine Logic

| File | Line(s) | Issue |
|------|---------|-------|
| `dashboard/app.py` | 437-440, 642-648 | Hardcoded `sp500`, `vix` as defaults |
| `start/run_calibrated.py` | 72, 362-381, 543-660 | Hardcoded `sp500`, `vix`, `T10Y2Y` indicators |
| `panel/runtime_loader.py` | 16 | Default `["sp500", "vix", "t10y2y"]` |
| `panel/__init__.py` | 14 | Default `["sp500", "vix", "t10y2y"]` |
| `start/engine_contributions.py` | 67-68, 119-122 | Hardcoded `VIXCLS`, `SP500`, etc. |

**Fix:** These should be loaded from domain registry, not hardcoded.

### 1.3 "Market-Only" References

| File | Line(s) | Issue |
|------|---------|-------|
| `ml/regime_classifier_market_only.py` | 12, 36 | Script name and logic assume "market-only" |
| `ml/regime_ensemble_post_etf.py` | 3, 41, 91, 123, 164 | "Market-Only Ensemble" terminology |
| `ml/prism_ml_engine.py` | 9 | References "market-only columns" |

### 1.4 Finance-Specific Documentation Strings

| File | Line | Description Reference |
|------|------|----------------------|
| `engine_core/lenses/regime_switching_lens.py` | 28 | `description = "Market regime detection and analysis"` |
| `start/run_tuned_sql.py` | 234-235 | "Recession indicator" hardcoded |
| `start/overnight_analysis.py` | 648, 881 | `['Crisis', 'Normal', 'Bull']` state names |
| `start/early_warning_chart.py` | 145, 305 | References "2022 Bear" |

---

## 2. Broken Import Paths & Missing Directories

### 2.1 Missing `data_fetch/` Directory (CRITICAL)

Multiple files reference `data_fetch/system_registry.json` but **this directory does not exist**:

| File | Line(s) | Reference |
|------|---------|-----------|
| `utils/db_registry.py` | 46, 51 | `data_fetch/system_registry.json` |
| `utils/registry_validator.py` | 105, 184, 248, 305-307 | `data_fetch/*.json` |
| `utils/fetch_validator.py` | 676, 793, 901, 967-969 | `data_fetch/*.json` |

**Fix:** Update these paths to point to `data/registry/system_registry.json` which exists.

### 2.2 Deprecated Table References

These files still use deprecated `market_prices` and `econ_values` tables:

| File | Issue |
|------|-------|
| `start/lens_contributions.py` | Direct SQL to `market_prices`, `econ_values` |
| `start/calibration_lens_benchmark.py` | SQL queries to deprecated tables |
| `start/run_calibrated.py` | Falls back to deprecated tables |
| `start/run_regime_analysis.py` | Uses deprecated tables |
| `start/overnight_lite.py` | Uses deprecated tables |
| `start/run_tuned.py` | Uses deprecated tables |
| `start/correlation_movie.py` | Uses deprecated tables |
| `start/early_warning_clean.py` | Uses deprecated tables |
| `start/check_coverage.py` | Uses deprecated tables |
| `scripts/validate_schema.py` | Tests deprecated tables |

**Fix:** Migrate to `indicator_values` table (the unified schema).

---

## 3. Files Recommended for Deletion

### 3.1 Immediate Deletion (Safe)

| File/Directory | Reason |
|----------------|--------|
| `fetcher again.zip` | Temporary archive in root |
| `fetcher_yahoo_fixed.py` | Backup script in root |
| `overnight.log` | 2.6MB log file |
| `.backup_20251206_163000/` | Old backup directory |
| `cladue_sandbox/` | Typo directory (duplicate of claude_sandbox) |
| `update_start_scripts.py` | One-time migration script in root |

### 3.2 claude_sandbox/ Contents (Review Before Delete)

| File | Type | Recommendation |
|------|------|----------------|
| `claude_sandbox/*.zip` (12 files) | Phase archives | **Delete** - 350KB total |
| `claude_sandbox/calibratedengine` | Text dump | Delete |
| `claude_sandbox/dashboard_update_*` | Text dumps | Delete |
| `claude_sandbox/databaselocation` | Text dump | Delete |
| `claude_sandbox/*.md` | Notes | Review, then delete |

### 3.3 legacy/ Directory (Mark Deprecated)

| File | Status |
|------|--------|
| `legacy/panel_builders/build_panel.py` | Already deprecated (has warning) |
| `legacy/panel_builders/build_climate_panel.py` | Already deprecated (has warning) |
| `legacy/panel_builders/transforms_econ.py` | Review for removal |
| `legacy/panel_builders/transforms_market.py` | Review for removal |

### 3.4 Root-Level Test Files (Move to tests/)

| File | Recommendation |
|------|----------------|
| `test_phase1.py` | Move to `tests/integration/` |
| `test_phase2.py` | Move to `tests/integration/` |
| `test_phase3_cli.py` | Move to `tests/integration/` |
| `test_phase4_html.py` | Move to `tests/integration/` |
| `test_phase5_web.py` | Move to `tests/integration/` |
| `test_phase6_plugins.py` | Move to `tests/integration/` |
| `test_phase7_universal.py` | Move to `tests/integration/` |

### 3.5 ml/ Directory (Legacy ML Models)

All files in `ml/` contain legacy finance-specific regime classification:

| File | Issue |
|------|-------|
| `ml/regime_classifier_rf.py` | SPY-based regime labels |
| `ml/regime_classifier_rf_v2.py` | SPY-based regime labels |
| `ml/regime_classifier_rf_pro.py` | SPY-based regime labels |
| `ml/regime_classifier_ensemble.py` | Market-specific |
| `ml/regime_classifier_market_only.py` | "Market-only" in name |
| `ml/regime_ensemble_post_etf.py` | ETF-era specific |
| `ml/build_prism_labels.py` | Uses risk_on/off terminology |

**Recommendation:** Archive entire `ml/` directory or refactor for domain-agnostic state classification.

---

## 4. Branch Audit

### 4.1 Current Branches

```
Local:
  claude/add-engine-series-visualization-012Wx5nWV5LCRhzfMRQ9U2iV
* claude/audit-remove-legacy-01WhhYm2h4K1psGJ7dD6vSz5

Remote:
  origin/claude/add-engine-series-visualization-012Wx5nWV5LCRhzfMRQ9U2iV
  origin/claude/audit-remove-legacy-01WhhYm2h4K1psGJ7dD6vSz5
```

### 4.2 Recommendations

| Branch | Status | Action |
|--------|--------|--------|
| `claude/add-engine-series-visualization-*` | Merged (PR #101) | **Safe to delete** |
| `claude/audit-remove-legacy-*` | Current work | Keep until merged |

The repository appears clean with only 2 active branches. No stale branches detected.

---

## 5. SQL Schema Validation

### 5.1 Current Schema Status

The schema in `data/sql/schema.sql` is **correctly domain-agnostic**:
- `systems` table with multi-domain support (finance, climate, biology, etc.)
- `indicators` table with domain-agnostic metadata
- `indicator_values` as unified data table
- `fetch_log` for auditing

### 5.2 Deprecated Tables (Still Present)

| Table | Status | Action |
|-------|--------|--------|
| `market_prices` | Deprecated | Remove after migration |
| `econ_values` | Deprecated | Remove after migration |
| `timeseries` view | Backward compat | Keep temporarily |

### 5.3 Files Still Using Deprecated Tables

See Section 2.2 above. These need migration to `indicator_values`.

---

## 6. Terminology Replacements Required

### 6.1 State/Regime Terminology

| Current | Domain-Agnostic Replacement |
|---------|---------------------------|
| `risk_on` | `elevated` or `expansion` |
| `risk_off` | `depressed` or `contraction` |
| `neutral` | `baseline` or `stable` |
| `bull` | `expansion` or `positive_trend` |
| `bear` | `contraction` or `negative_trend` |
| `regime` | `state` or `phase` |
| `crisis` | `instability` or `disruption` |

### 6.2 Domain References

| Current | Domain-Agnostic Replacement |
|---------|---------------------------|
| `market regime detection` | `system state detection` |
| `market_prices` | `indicator_values` |
| `econ_values` | `indicator_values` |
| `market-only` | `single-domain` or remove |
| `recession indicator` | `contraction signal` |

---

## 7. Import Path Corrections Required

### 7.1 data_fetch -> data/registry

```python
# OLD (broken)
registry_path = _get_project_root() / "data_fetch" / "system_registry.json"

# NEW (correct)
registry_path = _get_project_root() / "data" / "registry" / "system_registry.json"
```

Files requiring this fix:
- `utils/db_registry.py`
- `utils/registry_validator.py`
- `utils/fetch_validator.py`

---

## 8. Summary Action Items

### 8.1 Immediate Actions (This PR)

| Priority | Task | Files Affected |
|----------|------|----------------|
| HIGH | Delete `fetcher again.zip`, `fetcher_yahoo_fixed.py` | Root |
| HIGH | Delete `cladue_sandbox/` (typo directory) | Root |
| HIGH | Delete `.backup_20251206_163000/` | Root |
| HIGH | Fix `data_fetch` -> `data/registry` paths | 3 utils files |
| MEDIUM | Delete `claude_sandbox/*.zip` files | 12 files |
| MEDIUM | Move `test_phase*.py` to `tests/integration/` | 7 files |
| MEDIUM | Delete `overnight.log` | Root |

### 8.2 Follow-Up PRs (Future Work)

| Priority | Task | Scope |
|----------|------|-------|
| HIGH | Refactor `ml/` for domain-agnostic states | 8 files |
| HIGH | Migrate deprecated SQL table usage | 10+ start scripts |
| MEDIUM | Update lens descriptions to domain-agnostic | engine_core/lenses |
| MEDIUM | Remove hardcoded indicator defaults | panel/, dashboard/ |
| LOW | Archive or remove legacy/ directory | 5 files |

### 8.3 Branch Cleanup

```bash
# After merging PR #101
git branch -d claude/add-engine-series-visualization-012Wx5nWV5LCRhzfMRQ9U2iV
git push origin --delete claude/add-engine-series-visualization-012Wx5nWV5LCRhzfMRQ9U2iV
```

---

## 9. Files Updated in This PR

### 9.1 Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `PRISM_REFACTOR_REPORT_v1.md` | Added | This comprehensive audit report |
| `utils/db_registry.py` | Modified | Fixed `data_fetch/` -> `data/registry/` path |
| `utils/registry_validator.py` | Modified | Fixed `data_fetch/` -> `data/registry/` paths (4 locations) |
| `utils/fetch_validator.py` | Modified | Fixed `data_fetch/` -> `data/registry/` paths (4 locations) |

### 9.2 Files Deleted

| File/Directory | Size | Reason |
|----------------|------|--------|
| `fetcher again.zip` | 9KB | Temporary archive in root |
| `fetcher_yahoo_fixed.py` | 13KB | Backup/duplicate script in root |
| `overnight.log` | 2.6MB | Generated log file |
| `update_start_scripts.py` | 6KB | One-time migration script |
| `.backup_20251206_163000/` | 64KB | Old backup directory (10 files) |
| `cladue_sandbox/` | 80KB | Typo directory - duplicate of claude_sandbox (10 files) |
| `claude_sandbox/*.zip` | 350KB | Phase archives (12 ZIP files) |
| `claude_sandbox/calibratedengine` | 1KB | Text dump |
| `claude_sandbox/dashboard_update_*` | 37KB | Text dumps (2 files) |
| `claude_sandbox/databaselocation` | 1KB | Text dump |

**Total Space Freed:** ~3.2 MB (before git gc)

---

## 10. Known Future Cleanup Recommendations

1. **Model Refactoring**: The entire `ml/` directory needs refactoring to use domain-agnostic state classification rather than finance-specific regime labels.

2. **SQL Migration**: Complete migration from `market_prices`/`econ_values` to unified `indicator_values` table, then remove deprecated tables from schema.

3. **Registry Consolidation**: Ensure all registry references point to `data/registry/` and remove any references to non-existent `data_fetch/` directory.

4. **Documentation Update**: Update all docstrings and comments that reference "market", "regime", "bull/bear" to use domain-agnostic terminology.

5. **Test Coverage**: Add tests to validate domain-agnostic behavior and flag any finance-specific assumptions.

---

**End of Report**
