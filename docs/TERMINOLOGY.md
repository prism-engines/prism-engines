# PRISM Terminology

**Canonical reference for all naming conventions.**

---

## Core Terms

| Term | Definition |
|------|------------|
| **Engine** | A mathematical method (PCA, Granger, Hurst, Wavelet, Entropy, etc.) |
| **Phase** | A processing stage in the analysis pipeline |
| **Indicator** | A time series data source (GDP, S&P 500, Treasury Yield, etc.) |

---

## The Three Phases

| Phase | Name | Purpose | Input | Output |
|-------|------|---------|-------|--------|
| 1 | **Unbound** | Indicator behavior without system context | Indicator time series | Engine outputs per indicator |
| 2 | **Structure** | Emergent system geometry | Phase 1 outputs | System state boundaries |
| 3 | **Bounded** | Indicator behavior within geometry | Phase 1 outputs + Phase 2 structure | Positioned indicators |

---

## Data Treatments

| Treatment | Description | Table Prefix |
|-----------|-------------|--------------|
| **Raw** | As fetched from source, immutable | `raw` |
| **Cleaned** | Missing values handled, outliers addressed | `cleaned` |
| **Normalized** | Scaled/transformed for analysis | `normalized` |

---

## DuckDB Schema Names

### Indicator Data (Immutable Layers)

| Schema | Contents |
|--------|----------|
| `raw` | Fetched data — never modified |
| `cleaned` | Cleaned data — derived from raw |
| `normalized` | Normalized data — derived from cleaned |

### Phase Outputs (By Data Treatment)

| Schema | Phase | Treatment |
|--------|-------|-----------|
| `unbound_raw` | 1 | Raw |
| `unbound_cleaned` | 1 | Cleaned |
| `unbound_norm` | 1 | Normalized |
| `structure_raw` | 2 | Raw |
| `structure_cleaned` | 2 | Cleaned |
| `structure_norm` | 2 | Normalized |
| `bounded_raw` | 3 | Raw |
| `bounded_cleaned` | 3 | Cleaned |
| `bounded_norm` | 3 | Normalized |

---

## Deprecated Terms (DO NOT USE)

| Deprecated | Replacement | Reason |
|------------|-------------|--------|
| ~~Lens~~ | Engine | Clarity |
| ~~Fingerprint~~ | (describe output directly) | Ambiguous |
| ~~Observe~~ | — | Legacy, no purpose |
| ~~Coherence~~ | — | GPT artifact |
| ~~VCF~~ | PRISM | Rebranded |

---

## Component Naming

| Component | Naming Pattern | Example |
|-----------|----------------|---------|
| Fetcher | `{source}_fetcher.py` or `{source}.py` | `fred.py`, `tiingo.py` |
| Engine | `{method}.py` | `pca.py`, `granger.py`, `hurst.py` |
| Phase | `{phase_name}.py` | `unbound.py`, `structure.py`, `bounded.py` |

---

## Method Reference

When uncertain about terminology, consult this hierarchy:

1. **This document** (TERMINOLOGY.md)
2. **ARCHITECTURE.md**
3. **Ask before inventing**
