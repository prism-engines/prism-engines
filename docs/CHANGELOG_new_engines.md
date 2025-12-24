# PRISM New Engines - Change Log

**Date:** 2024-12-18
**Summary:** Added 6 new analysis engines to complete the PRISM lens complement (20 total)

---

## Overview

Added 6 engines from the PRISM_Lens_Reference.md specification that were not yet implemented:

| # | Engine | Category | Priority | Status |
|---|--------|----------|----------|--------|
| 1 | DTW | Similarity | 3 | ✓ Added |
| 2 | Copula | Dependence | 4 | ✓ Added |
| 3 | RQA | Complexity | 4 | ✓ Added |
| 4 | Cointegration | Causality | 3 | ✓ Added |
| 5 | Lyapunov | Complexity | 4 | ✓ Added |
| 6 | Distance | Geometry | 2 | ✓ Added |

---

## New Engine Details

### 1. DTW (Dynamic Time Warping)

**File:** `prism/engines/dtw.py`

**Purpose:** Measures shape similarity between time series, allowing for temporal warping/alignment. Useful for comparing indicators that may be phase-shifted or have different speeds.

**Algorithm:** Classic DTW with optional Sakoe-Chiba band constraint for computational efficiency.

**Key Outputs:**
- `dtw_distance`: Raw DTW distance between pairs
- `dtw_similarity`: Normalized similarity (0-1)
- `avg_dtw_distance`: System-wide average
- `n_pairs`: Number of pairs analyzed

**Usage:**
```python
from prism.engines import get_engine
dtw = get_engine("dtw")
metrics = dtw.run(df_normalized, run_id="test")
```

**Normalization:** Z-score preferred

---

### 2. Copula

**File:** `prism/engines/copula.py`

**Purpose:** Analyzes tail dependence and non-linear dependence structure. Captures how indicators behave together in extreme conditions (crashes, rallies).

**Algorithm:** Empirical copula estimation with threshold-based tail dependence.

**Key Outputs:**
- `lower_tail_dependence`: Joint crash behavior
- `upper_tail_dependence`: Joint rally behavior
- `kendall_tau`: Rank correlation (robust to outliers)
- `spearman_rho`: Rank correlation
- `tail_asymmetry`: Difference between upper and lower tail dependence

**Usage:**
```python
from prism.engines import get_engine
copula = get_engine("copula")
metrics = copula.run(df, run_id="test")  # Uses raw data, transforms internally
```

**Normalization:** Rank transform (applied internally)

---

### 3. RQA (Recurrence Quantification Analysis)

**File:** `prism/engines/rqa.py`

**Purpose:** Analyzes system dynamics through recurrence plots. Detects determinism, laminarity, and regime structure in time series.

**Algorithm:** Time-delay embedding + recurrence matrix + line structure analysis.

**Key Outputs:**
- `recurrence_rate`: Fraction of recurrent points
- `determinism`: Fraction in diagonal structures (predictability)
- `laminarity`: Fraction in vertical structures (regime persistence)
- `entropy`: Complexity of diagonal line distribution
- `avg_diagonal_length`: Average predictability horizon
- `max_diagonal_length`: Maximum predictability horizon

**Interpretation:**
- High determinism → Predictable dynamics
- High laminarity → Regime persistence
- High entropy → Complex dynamics

**Usage:**
```python
from prism.engines import get_engine
rqa = get_engine("rqa")
metrics = rqa.run(df_normalized, run_id="test", embedding_dim=3, time_delay=1)
```

**Normalization:** Z-score preferred

---

### 4. Cointegration

**File:** `prism/engines/cointegration.py`

**Purpose:** Tests for long-run equilibrium relationships between series using Engle-Granger methodology. Identifies pairs that move together over time.

**Algorithm:** Two-step Engle-Granger: OLS regression + ADF test on residuals.

**Key Outputs:**
- `adf_stat`: Augmented Dickey-Fuller test statistic
- `p_value`: Significance of cointegration
- `hedge_ratio`: Cointegrating coefficient (for spread construction)
- `half_life`: Mean reversion speed (if cointegrated)
- `n_cointegrated`: Number of cointegrated pairs
- `cointegration_rate`: Fraction of pairs cointegrated

**Usage:**
```python
from prism.engines import get_engine
coint = get_engine("cointegration")
metrics = coint.run(df_levels, run_id="test")  # Use levels, not returns
```

**Normalization:** None (uses levels)

---

### 5. Lyapunov

**File:** `prism/engines/lyapunov.py`

**Purpose:** Estimates the largest Lyapunov exponent (LLE) to characterize chaos vs stability. Measures sensitivity to initial conditions.

**Algorithm:** Rosenstein's method (1993) - tracks divergence of nearby trajectories in embedded phase space.

**Key Outputs:**
- `lyapunov_exponent`: The LLE value per indicator
- `is_chaotic`: Boolean (LLE > 0.01)
- `is_stable`: Boolean (LLE < -0.01)
- `chaotic_fraction`: System-wide chaos prevalence

**Interpretation:**
- LLE > 0: **Chaotic** - exponential divergence, limited predictability
- LLE ≈ 0: **Marginal** - neutral stability
- LLE < 0: **Stable** - convergent, predictable

**Usage:**
```python
from prism.engines import get_engine
lyap = get_engine("lyapunov")
metrics = lyap.run(df_normalized, run_id="test", embedding_dim=3)
```

**Normalization:** Z-score preferred

---

### 6. Distance

**File:** `prism/engines/distance.py`

**Purpose:** Computes multiple geometric distance metrics between indicators. Provides different views of indicator separation.

**Methods:**
1. **Euclidean**: Standard L2 distance in time-space
2. **Mahalanobis**: Covariance-adjusted distance
3. **Cosine**: Directional similarity (1 - cosine similarity)
4. **Correlation**: 1 - |correlation|

**Key Outputs:**
- `avg_euclidean_distance`: Average Euclidean separation
- `avg_mahalanobis_distance`: Average covariance-adjusted separation
- `avg_cosine_distance`: Average directional dissimilarity
- `avg_correlation_distance`: Average decorrelation

**Usage:**
```python
from prism.engines import get_engine
dist = get_engine("distance")
metrics = dist.run(df_normalized, run_id="test", methods=["euclidean", "cosine"])
```

**Normalization:** Z-score preferred

---

## Files Modified

### `prism/engines/__init__.py`

Added imports and registry entries for all 6 new engines:

```python
# New imports
from .dtw import DTWEngine
from .copula import CopulaEngine
from .rqa import RQAEngine
from .cointegration import CointegrationEngine
from .lyapunov import LyapunovEngine
from .distance import DistanceEngine

# Added to ENGINE_REGISTRY
"dtw": DTWEngine,
"copula": CopulaEngine,
"rqa": RQAEngine,
"cointegration": CointegrationEngine,
"lyapunov": LyapunovEngine,
"distance": DistanceEngine,

# Added to ENGINE_INFO with metadata
```

### `tests/benchmark_engines.py`

Added 6 new test functions:
- `test_dtw_basic()`
- `test_copula_basic()`
- `test_rqa_basic()`
- `test_cointegration_basic()`
- `test_lyapunov_basic()`
- `test_distance_basic()`

Test count increased from 20 to 26.

---

## Test Results

```
============================================================
PRISM ENGINE BENCHMARK SUITE
============================================================
Total:  26
Passed: 26
Failed: 0
Time:   3.30s
============================================================
```

All new engines pass validation:

| Engine | Test | Time | Status |
|--------|------|------|--------|
| dtw | basic | 0.064s | ✓ PASS |
| copula | basic | 0.013s | ✓ PASS |
| cointegration | basic | 0.007s | ✓ PASS |
| distance | basic | 0.002s | ✓ PASS |
| rqa | basic | 0.230s | ✓ PASS |
| lyapunov | basic | 0.247s | ✓ PASS |

---

## Complete Engine Catalog (20 Engines)

### Priority 1 - Core
| Engine | Category | Description |
|--------|----------|-------------|
| pca | structure | Principal Component Analysis |
| cross_correlation | correlation | Lead/lag relationships |
| rolling_beta | regression | Time-varying sensitivity |
| clustering | structure | Behavioral grouping |
| hurst | complexity | Long-term memory |

### Priority 2 - Foundation
| Engine | Category | Description |
|--------|----------|-------------|
| distance | geometry | Euclidean, Mahalanobis, Cosine |
| granger | causality | Directional predictive relationships |
| mutual_information | information | Non-linear dependence |
| transfer_entropy | information | Directional information flow |

### Priority 3 - Intermediate
| Engine | Category | Description |
|--------|----------|-------------|
| dtw | similarity | Dynamic Time Warping |
| cointegration | causality | Long-run equilibrium |
| wavelet | frequency | Time-frequency coherence |
| entropy | complexity | Shannon, permutation entropy |
| spectral | frequency | Dominant cycles |

### Priority 4 - Advanced
| Engine | Category | Description |
|--------|----------|-------------|
| copula | dependence | Tail dependence structure |
| rqa | complexity | Recurrence Quantification Analysis |
| lyapunov | complexity | Chaos detection |
| garch | volatility | Conditional variance dynamics |
| hmm | regime | Hidden Markov Model |
| dmd | dynamics | Dynamic Mode Decomposition |

---

## Quick Reference

### List All Engines
```python
from prism.engines import list_engines
for name, info in list_engines().items():
    print(f"{name}: {info['description']}")
```

### Run Single Engine
```python
from prism.engines import get_engine
engine = get_engine("dtw")
metrics = engine.run(df, run_id="analysis_001")
```

### Run by Priority
```python
from prism.engines import get_engines_by_priority
core_engines = get_engines_by_priority(1)  # Priority 1 only
all_engines = get_engines_by_priority(4)   # All engines
```

### CLI
```bash
# List engines
python scripts/run_engine.py --list

# Run specific engine
python scripts/run_engine.py --engine dtw

# Run benchmarks
python tests/benchmark_engines.py -v
```

---

## Dependencies

All new engines use standard scientific Python stack:
- `numpy` - Core numerics
- `pandas` - Data handling
- `scipy` - Statistical functions

No additional dependencies required.

---

## References

### DTW
- Sakoe & Chiba (1978) - Dynamic programming algorithm optimization

### Copula
- Sklar (1959) - Copula theory foundations
- Joe (1997) - Multivariate models and dependence concepts

### RQA
- Eckmann et al. (1987) - Recurrence plots
- Marwan et al. (2007) - Recurrence plots for complex systems

### Cointegration
- Engle & Granger (1987) - Cointegration and error correction
- MacKinnon (1991) - Critical values for cointegration tests

### Lyapunov
- Rosenstein et al. (1993) - Practical method for calculating largest Lyapunov exponents
- Wolf et al. (1985) - Determining Lyapunov exponents from a time series

### Distance Metrics
- Mahalanobis (1936) - Generalized distance in statistics

---

*PRISM Engine Module v1.1*
*December 2024*
