# PRISM Engines

Mathematical analysis engines for the PRISM system.

## Overview

PRISM engines analyze indicator behavior through multiple mathematical "lenses." Each engine captures different aspects of the data — correlation structure, causality, frequency content, complexity, regime dynamics, etc.

The weighted consensus of engine outputs produces robust behavioral fingerprints that power PRISM's geometry visualization.

## Architecture

```
clean.indicators (DB)
        │
        ▼
┌───────────────────┐
│   BaseEngine      │ ◄── Shared: data loading, normalization, result storage
└───────────────────┘
        │
        ├── PCAEngine
        ├── CrossCorrelationEngine
        ├── ClusteringEngine
        ├── HurstEngine
        ├── GrangerEngine
        ├── ...
        │
        ▼
┌───────────────────┐
│  Phase Schemas    │
│  - unbound.*      │ ◄── Individual indicator analysis
│  - structure.*    │ ◄── System-level geometry
│  - bounded.*      │ ◄── Indicators within structure
└───────────────────┘
```

## Quick Start

```python
from prism.engines import get_engine, list_engines

# List available engines
for name, info in list_engines().items():
    print(f"{name}: {info['description']}")

# Run an engine
pca = get_engine("pca")
result = pca.execute(
    indicator_ids=["SPY", "AGG", "XLU", "GLD"],
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
)

print(result.summary())
# Engine: pca
# Run ID: pca_20241218_143052_a1b2c3d4
# Status: SUCCESS
# Runtime: 0.34s
# Window: 2020-01-02 to 2023-12-29
# Normalization: zscore
# Metrics: {'n_indicators': 4, 'variance_pc1': 0.52, ...}
```

## CLI Usage

```bash
# List all engines
python scripts/run_engine.py --list

# Run single engine
python scripts/run_engine.py --engine pca

# Run with specific indicators
python scripts/run_engine.py --engine granger --indicators SPY AGG XLU

# Run with date range
python scripts/run_engine.py --engine clustering --start 2020-01-01 --end 2024-01-01

# Run multiple engines
python scripts/run_engine.py --engines pca clustering hurst

# Run all core engines (priority 1)
python scripts/run_engine.py --priority 1

# Run with custom parameters
python scripts/run_engine.py --engine hmm --params n_states=4

# Override normalization
python scripts/run_engine.py --engine pca --normalization returns
```

## Engine Catalog

### Phase 1: Core (Priority 1)

| Engine | Category | Measures | Normalization |
|--------|----------|----------|---------------|
| `pca` | Structure | Variance explained, loadings, dimensionality | Z-score |
| `cross_correlation` | Correlation | Lead/lag relationships, optimal lags | Z-score |
| `rolling_beta` | Regression | Time-varying sensitivity to reference | Returns |
| `clustering` | Structure | Behavioral groupings, cluster quality | Z-score |
| `hurst` | Complexity | Long-term memory, persistence (H exponent) | None |

### Phase 2: Information & Causality (Priority 2)

| Engine | Category | Measures | Normalization |
|--------|----------|----------|---------------|
| `granger` | Causality | Directional predictive relationships | Diff (stationary) |
| `mutual_information` | Information | Non-linear dependence | Discretize |
| `transfer_entropy` | Information | Directional information flow (bits) | Discretize |

### Phase 3: Frequency & Complexity (Priority 3)

| Engine | Category | Measures | Normalization |
|--------|----------|----------|---------------|
| `wavelet` | Frequency | Time-frequency coherence, phase | None |
| `entropy` | Complexity | Shannon, permutation, sample entropy | Varies |
| `spectral` | Frequency | Dominant cycles, spectral peaks | Diff |

### Phase 4: Advanced (Priority 4)

| Engine | Category | Measures | Normalization |
|--------|----------|----------|---------------|
| `garch` | Volatility | Conditional variance, persistence | Returns |
| `hmm` | Regime | Hidden states, transitions, durations | Z-score |
| `dmd` | Dynamics | Eigenvalues, modes, frequencies, stability | Z-score |

## Engine Details

### PCA (Principal Component Analysis)

**Purpose:** Understand variance structure and dimensionality.

**Key Outputs:**
- `variance_pc1`: How much variance PC1 explains (market factor)
- `components_for_90pct`: Effective dimensionality
- Loading matrix: Which indicators drive each component

**Interpretation:**
- High PC1 variance → Market dominated by single factor
- Low PC1 variance → Diverse, uncorrelated indicators
- Loading signs → Which indicators move together

---

### Hurst Exponent

**Purpose:** Detect trending vs mean-reverting behavior.

**Key Outputs:**
- `hurst`: The H exponent (0-1)
  - H > 0.5: Trending (momentum)
  - H = 0.5: Random walk
  - H < 0.5: Mean-reverting

**Interpretation:**
- High H indicators may continue trends
- Low H indicators may revert to mean
- System-wide H changes signal regime shifts

---

### Granger Causality

**Purpose:** Find directional predictive relationships.

**Key Outputs:**
- F-statistic and p-value per pair
- Optimal lag for each relationship
- Causality network (who predicts whom)

**Interpretation:**
- Significant X→Y: Past X improves Y prediction
- Use for lead/lag identification
- Build influence networks

---

### HMM (Hidden Markov Model)

**Purpose:** Detect latent regimes from behavior.

**Key Outputs:**
- State assignments over time
- Transition matrix (regime switching probabilities)
- Regime duration statistics

**Interpretation:**
- States represent market regimes (bull/bear/transition)
- High self-transition → Stable regime
- Cross-check against geometric regimes

---

### DMD (Dynamic Mode Decomposition)

**Purpose:** Analyze system dynamics via modes.

**Key Outputs:**
- Eigenvalues (growth/decay rates, frequencies)
- Mode energies
- Reconstruction error

**Interpretation:**
- Stable modes (|λ| ≤ 1): Sustainable dynamics
- Unstable modes: Regime breakdown
- Dominant frequency: System's natural oscillation
- Aligns with PRISM's "vibrations" framing

## Normalization Reference

| Method | Code | When to Use |
|--------|------|-------------|
| Z-score | `zscore` | Most methods, comparability across scales |
| Min-Max | `minmax` | When 0-1 range needed |
| Returns | `returns` | Volatility, GARCH, time-series regression |
| Rank | `rank` | Copula, non-parametric methods |
| Difference | `diff` | Ensure stationarity |
| Discretize | (internal) | Information-theoretic methods |

## Output Storage

Engines write results to phase-specific schemas:

```
unbound.*          → Individual indicator analysis
  - pca_loadings
  - pca_variance
  - correlation_matrix
  - granger_causality
  - geometry_fingerprints

structure.*        → System-level patterns
  - regimes
  - clusters
  - centroids
  - similarity_matrix

bounded.*          → Indicators within structure
  - indicator_positions
  - leader_laggard
  - regime_membership
```

## Adding New Engines

1. Create `prism/engines/your_engine.py`:

```python
from .base import BaseEngine

class YourEngine(BaseEngine):
    name = "your_engine"
    phase = "unbound"  # or "structure" or "bounded"
    default_normalization = "zscore"  # or None
    
    def run(self, df, run_id, **params):
        # df: DataFrame with DatetimeIndex, indicators as columns
        # Already normalized if default_normalization is set
        
        # Your analysis here
        results = do_analysis(df)
        
        # Store results
        self.store_results("your_table", results_df, run_id)
        
        # Return summary metrics
        return {
            "metric1": value1,
            "metric2": value2,
        }
```

2. Register in `__init__.py`:

```python
from .your_engine import YourEngine

ENGINE_REGISTRY["your_engine"] = YourEngine
ENGINE_INFO["your_engine"] = {
    "phase": "unbound",
    "category": "your_category",
    "description": "What it does",
    "normalization": "zscore",
    "priority": 3,
}
```

3. Add schema table if needed in `schema.sql`

## Dependencies

**Required:**
- numpy
- pandas
- scipy
- scikit-learn
- duckdb

**Optional (enhanced functionality):**
- `arch` — GARCH models (fallback available)
- `hmmlearn` — HMM (fallback available)
- `pywt` — Wavelets (fallback available)
- `statsmodels` — Additional statistical tests

Install optional:
```bash
pip install arch hmmlearn pywt statsmodels
```

## Testing

```bash
# Run all benchmarks
python tests/benchmark_engines.py

# Run specific engine test
python tests/benchmark_engines.py --engine pca

# Verbose output
python tests/benchmark_engines.py -v
```

## Design Principles

1. **Modular, on-demand execution** — Run engines one at a time or in small batches
2. **Minimal persistence** — Store only what's needed for reproducibility
3. **Agent-guided selection** — AI agents recommend which engines matter for each question
4. **Observability** — Every run produces clear logs and summary metrics
5. **Reproducibility** — Deterministic seeds, logged parameters, versioned outputs

## References

### Key Papers
- Mandelbrot (1963) — Hurst exponent, long-term memory
- Granger (1969) — Causality testing
- Schreiber (2000) — Transfer entropy
- Torrence & Compo (1998) — Wavelet analysis
- Schmid (2010) — Dynamic Mode Decomposition

### Python Ecosystem
- `sklearn` — PCA, clustering, mutual information
- `scipy` — FFT, correlation, signal processing
- `statsmodels` — Granger, cointegration
- `arch` — GARCH volatility models
- `hmmlearn` — Hidden Markov Models
- `pywt` — Wavelet transforms

---

*PRISM Engine Module v1.0*
*December 2024*
