# =============================================================================
# PRISM Lens Engine Reference
# =============================================================================
#
# Comprehensive catalog of mathematical methods for the PRISM analytical engine.
# Each lens produces a behavioral fingerprint for indicators.
#
# Version: 1.0.0
# Status: Reference for Engine Development
#
# =============================================================================

## Overview

PRISM uses multiple mathematical "lenses" to analyze indicator behavior. Each lens
captures different aspects of the data — correlation structure, causality, 
frequency content, complexity, etc. The weighted consensus of lens outputs
produces robust behavioral fingerprints.

This document catalogs:
- 18 candidate lenses (select 10-14 for implementation)
- Data requirements for each
- Normalization/cleaning specifications
- Python implementation references

---

## Quick Reference Table

| #  | Lens                    | Category        | Measures                          | Stationarity | Normalization |
|----|-------------------------|-----------------|-----------------------------------|--------------|---------------|
| 1  | PCA                     | Structure       | Variance explained, loadings      | Preferred    | Z-score       |
| 2  | Granger Causality       | Causality       | Predictive relationships          | Required     | None/Z-score  |
| 3  | Wavelet Coherence       | Frequency       | Time-frequency co-movement        | No           | None          |
| 4  | Transfer Entropy        | Information     | Directional information flow      | Preferred    | Discretize    |
| 5  | Dynamic Time Warping    | Similarity      | Shape similarity across lags      | No           | Z-score       |
| 6  | Hurst Exponent          | Complexity      | Long-term memory / persistence    | No           | None          |
| 7  | Spectral Density        | Frequency       | Dominant cycles                   | Required     | None          |
| 8  | Mutual Information      | Information     | Non-linear dependence             | Preferred    | Discretize    |
| 9  | Copula Dependence       | Structure       | Tail dependence                   | No           | Rank/Uniform  |
| 10 | Recurrence Analysis     | Complexity      | System dynamics, regime proximity | No           | Z-score       |
| 11 | Cross-Correlation       | Correlation     | Lead/lag relationships            | Preferred    | Z-score       |
| 12 | Rolling Beta            | Regression      | Time-varying sensitivity          | Preferred    | None          |
| 13 | Entropy (Shannon/Sample)| Complexity      | Irregularity, predictability      | No           | Discretize    |
| 14 | Cluster Membership      | Structure       | Behavioral grouping               | No           | Z-score       |
| 15 | Cointegration           | Causality       | Long-run equilibrium              | Non-stationary OK | None      |
| 16 | GARCH Volatility        | Volatility      | Conditional variance dynamics     | No           | Returns       |
| 17 | Lyapunov Exponent       | Complexity      | Chaos / sensitivity to conditions | No           | Z-score       |
| 18 | Graph Centrality        | Network         | Systemic importance               | Preferred    | Z-score       |

---

## Detailed Lens Specifications

---

### 1. Principal Component Analysis (PCA)

**Category:** Dimensionality Reduction / Structure

**What it measures:**
- How much variance each indicator explains
- Which indicators move together (loadings)
- Effective dimensionality of the system

**Key outputs:**
- Explained variance ratio per component
- Loading matrix (indicator weights per PC)
- Cumulative variance explained

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Preferred (use returns or differences)             |
| Missing values   | Not allowed — impute or drop                       |
| Normalization    | **Z-score required** (mean=0, std=1)               |
| Minimum samples  | 3× number of indicators                            |
| Frequency        | Uniform time steps                                 |

**Normalization procedure:**
```python
# Z-score normalization (per indicator, within window)
z = (x - x.mean()) / x.std()
```

**Python libraries:**
- `sklearn.decomposition.PCA`
- `numpy.linalg.eig` (manual)

**Notes:**
- Run on returns, not levels
- Consider robust PCA for outlier resistance
- Track loading stability across windows for regime detection

---

### 2. Granger Causality

**Category:** Causality

**What it measures:**
- Whether past values of X improve prediction of Y
- Directional predictive relationships
- Lead/lag structure between indicators

**Key outputs:**
- F-statistic and p-value per pair
- Optimal lag order
- Causality network (directed graph)

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | **Required** — use ADF test, difference if needed  |
| Missing values   | Not allowed                                        |
| Normalization    | Optional (Z-score for comparability)               |
| Minimum samples  | 50+ per indicator recommended                      |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# 1. Test stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(series)
if result[1] > 0.05:  # non-stationary
    series = series.diff().dropna()

# 2. Optional z-score
z = (series - series.mean()) / series.std()
```

**Python libraries:**
- `statsmodels.tsa.stattools.grangercausalitytests`

**Notes:**
- Test multiple lag orders (1-10 typical)
- Bonferroni correction for multiple comparisons
- Consider VAR-based Granger for efficiency

---

### 3. Wavelet Coherence

**Category:** Frequency / Time-Frequency

**What it measures:**
- Co-movement at different frequencies (scales)
- Time-localized correlation
- Phase relationships (lead/lag at each frequency)

**Key outputs:**
- Coherence matrix (time × frequency)
- Phase difference matrix
- Cone of influence (edge effects)

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | **Not required** — handles non-stationarity well   |
| Missing values   | Not allowed — interpolate if sparse                |
| Normalization    | **None required** — wavelet is scale-invariant     |
| Minimum samples  | 128+ recommended for frequency resolution          |
| Frequency        | Uniform time steps required                        |

**Pre-processing:**
```python
# Just ensure no NaN and uniform spacing
series = series.interpolate(method='linear')
```

**Python libraries:**
- `pywt` (PyWavelets)
- `pycwt` (continuous wavelet)

**Notes:**
- Morlet wavelet typical for financial data
- Computational cost: O(n log n) per pair
- Rich output — consider summarizing to key frequencies

---

### 4. Transfer Entropy

**Category:** Information Theory / Causality

**What it measures:**
- Directional information flow (X → Y)
- Non-linear causality (beyond Granger)
- Bits of information transferred

**Key outputs:**
- Transfer entropy value (bits)
- Effective transfer entropy (bias-corrected)
- Significance via shuffled surrogates

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Preferred but not strict                           |
| Missing values   | Not allowed                                        |
| Normalization    | **Discretization required**                        |
| Minimum samples  | 500+ for reliable estimation                       |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Discretization (binning or symbolic)
import numpy as np

# Option 1: Equal-width bins
n_bins = 8  # or use Freedman-Diaconis rule
bins = np.linspace(series.min(), series.max(), n_bins + 1)
discrete = np.digitize(series, bins)

# Option 2: Quantile bins (recommended)
discrete = pd.qcut(series, q=n_bins, labels=False, duplicates='drop')

# Option 3: Symbolic (sign of returns)
discrete = np.sign(series.diff())
```

**Python libraries:**
- `pyinform`
- `dit`
- `JIDT` (Java, Python wrapper)

**Notes:**
- Sensitive to bin count — test robustness
- Use shuffled surrogates for significance testing
- Computationally intensive for large windows

---

### 5. Dynamic Time Warping (DTW)

**Category:** Similarity / Pattern Matching

**What it measures:**
- Shape similarity allowing for time distortion
- Alignment path between series
- Similarity even with lead/lag

**Key outputs:**
- DTW distance (lower = more similar)
- Optimal warping path
- Alignment indices

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Not required                                       |
| Missing values   | Not allowed                                        |
| Normalization    | **Z-score required** (amplitude invariance)        |
| Minimum samples  | 50+ recommended                                    |
| Frequency        | Uniform preferred, not strict                      |

**Pre-processing:**
```python
# Z-score for amplitude invariance
z = (x - x.mean()) / x.std()

# Optional: Use returns instead of levels
returns = x.pct_change().dropna()
z = (returns - returns.mean()) / returns.std()
```

**Python libraries:**
- `dtaidistance`
- `tslearn.metrics.dtw`
- `fastdtw`

**Notes:**
- O(n²) complexity — use window constraints for long series
- Consider DTW Barycenter Averaging for cluster centroids
- Shape-based, good complement to correlation

---

### 6. Hurst Exponent (Mandelbrot)

**Category:** Complexity / Fractal Analysis

**What it measures:**
- Long-term memory in time series
- Persistence (H > 0.5) vs mean-reversion (H < 0.5)
- Self-similarity / fractal dimension

**Key outputs:**
- H value (0 to 1)
  - H ≈ 0.5: Random walk
  - H > 0.5: Trending / persistent
  - H < 0.5: Mean-reverting / anti-persistent
- Confidence interval

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | **Not required** — designed for non-stationary     |
| Missing values   | Not allowed                                        |
| Normalization    | **None required**                                  |
| Minimum samples  | 256+ recommended for R/S method                    |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Minimal — just clean NaN
series = series.dropna()
```

**Python libraries:**
- `hurst` (pip install hurst)
- `nolds.hurst_rs`

**Methods:**
- R/S (Rescaled Range) — classical
- DFA (Detrended Fluctuation Analysis) — more robust
- Wavelet-based — multi-scale

**Notes:**
- Use DFA for short series
- Track H over rolling windows for regime detection
- Core to fractal/Mandelbrot analysis

---

### 7. Spectral Density (Fourier)

**Category:** Frequency Domain

**What it measures:**
- Dominant periodicities/cycles
- Power distribution across frequencies
- Cyclical vs random components

**Key outputs:**
- Power spectral density (PSD)
- Dominant frequencies
- Spectral peaks

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | **Required** — use returns or detrend              |
| Missing values   | Not allowed                                        |
| Normalization    | **None required** (or mean-center)                 |
| Minimum samples  | Power of 2 ideal (128, 256, 512...)                |
| Frequency        | Uniform time steps required                        |

**Pre-processing:**
```python
# Detrend and/or use returns
from scipy.signal import detrend
series_detrended = detrend(series)

# Or simply use returns
returns = series.pct_change().dropna()
```

**Python libraries:**
- `scipy.signal.welch` (Welch's method — recommended)
- `scipy.fft.fft`
- `numpy.fft.fft`

**Notes:**
- Welch's method reduces noise via averaging
- Consider log-log plot for power law detection
- Cross-spectral density for pairs

---

### 8. Mutual Information

**Category:** Information Theory

**What it measures:**
- Non-linear dependence between variables
- Shared information (bits)
- Captures relationships correlation misses

**Key outputs:**
- MI value (≥ 0, higher = more dependence)
- Normalized MI (0-1 scale)

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Preferred                                          |
| Missing values   | Not allowed                                        |
| Normalization    | **Discretization or KDE required**                 |
| Minimum samples  | 200+ for reliable estimation                       |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Option 1: Discretization (same as Transfer Entropy)
discrete = pd.qcut(series, q=8, labels=False, duplicates='drop')

# Option 2: Use KNN-based estimator (no discretization needed)
from sklearn.feature_selection import mutual_info_regression
# This handles continuous data directly
```

**Python libraries:**
- `sklearn.metrics.mutual_info_score` (discrete)
- `sklearn.feature_selection.mutual_info_regression` (continuous, KNN)
- `dit`

**Notes:**
- KNN estimator (Kraskov) works on continuous data
- Compare to Pearson correlation — MI captures non-linear
- Symmetric: MI(X,Y) = MI(Y,X)

---

### 9. Copula Dependence

**Category:** Dependence Structure

**What it measures:**
- Tail dependence (joint extremes)
- Dependence structure separate from marginals
- Asymmetric dependence (left vs right tail)

**Key outputs:**
- Tail dependence coefficients (lower, upper)
- Copula family and parameters
- Kendall's tau (rank correlation)

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Not required                                       |
| Missing values   | Not allowed                                        |
| Normalization    | **Rank transform to uniform [0,1]**                |
| Minimum samples  | 250+ for tail estimation                           |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Probability integral transform (rank to uniform)
from scipy.stats import rankdata

def to_uniform(x):
    ranks = rankdata(x)
    return ranks / (len(x) + 1)  # Avoid 0 and 1

u = to_uniform(series_x)
v = to_uniform(series_y)
```

**Python libraries:**
- `copulas`
- `pyvinecopulib`
- `statsmodels.distributions.copula`

**Notes:**
- Critical for risk — correlations understate tail dependence
- Fit multiple families (Gaussian, Clayton, Gumbel, Frank)
- Clayton captures lower tail, Gumbel captures upper

---

### 10. Recurrence Quantification Analysis (RQA)

**Category:** Complexity / Nonlinear Dynamics

**What it measures:**
- Recurrence of system states
- Determinism vs stochasticity
- Regime proximity and transitions

**Key outputs:**
- Recurrence rate (RR)
- Determinism (DET)
- Entropy of diagonal lines
- Laminarity (LAM)

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Not required — captures non-stationarity           |
| Missing values   | Not allowed                                        |
| Normalization    | **Z-score recommended**                            |
| Minimum samples  | 500+ for reliable RQA                              |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Z-score normalization
z = (series - series.mean()) / series.std()

# Embedding (time-delay)
def embed(x, dim=3, tau=1):
    n = len(x) - (dim - 1) * tau
    return np.array([x[i:i + dim * tau:tau] for i in range(n)])

embedded = embed(z, dim=3, tau=1)
```

**Python libraries:**
- `pyrqa`
- `pyunicorn.timeseries.RecurrencePlot`

**Notes:**
- Requires embedding dimension and delay selection
- Use mutual information for delay, FNN for dimension
- Cross-recurrence for pairs

---

### 11. Cross-Correlation Function (CCF)

**Category:** Correlation / Lead-Lag

**What it measures:**
- Correlation at various time lags
- Lead/lag relationships
- Optimal lag for maximum correlation

**Key outputs:**
- CCF values at each lag
- Lag at maximum correlation
- Significance bounds

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | **Preferred** — pre-whiten if needed               |
| Missing values   | Not allowed                                        |
| Normalization    | **Z-score recommended**                            |
| Minimum samples  | 50+ (more for longer lags)                         |
| Frequency        | Uniform time steps required                        |

**Pre-processing:**
```python
# Z-score (makes CCF values comparable)
z_x = (x - x.mean()) / x.std()
z_y = (y - y.mean()) / y.std()

# Optional: Pre-whitening (remove autocorrelation)
from statsmodels.tsa.ar_model import AutoReg
model = AutoReg(z_x, lags=1).fit()
residuals = model.resid
```

**Python libraries:**
- `numpy.correlate`
- `scipy.signal.correlate`
- `statsmodels.tsa.stattools.ccf`

**Notes:**
- Simple but effective
- Combine with significance testing
- Consider partial cross-correlation

---

### 12. Rolling Beta (Time-Varying Regression)

**Category:** Regression / Sensitivity

**What it measures:**
- Time-varying sensitivity to a reference
- Beta stability/instability
- Regime-dependent relationships

**Key outputs:**
- Rolling beta coefficient
- Rolling R²
- Beta volatility

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Preferred (use returns)                            |
| Missing values   | Not allowed                                        |
| Normalization    | **None required** (use returns)                    |
| Minimum samples  | Window size × 3 minimum                            |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Convert to returns
returns_x = prices_x.pct_change().dropna()
returns_y = prices_y.pct_change().dropna()
```

**Python libraries:**
- `statsmodels.regression.rolling.RollingOLS`
- `pandas.DataFrame.rolling` + manual OLS

**Notes:**
- Track beta of beta (second-order stability)
- Compare to Kalman filter for smooth estimates
- Window size selection matters

---

### 13. Entropy (Shannon / Sample / Approximate)

**Category:** Complexity / Predictability

**What it measures:**
- Irregularity and unpredictability
- Information content
- Complexity of dynamics

**Key outputs:**
- Entropy value (bits or nats)
- Relative entropy (vs uniform/normal)

**Variants:**
| Variant            | Use case                              |
|--------------------|---------------------------------------|
| Shannon Entropy    | Discrete data, probability dist      |
| Sample Entropy     | Continuous, regularity of patterns   |
| Approximate Entropy| Similar to Sample, faster            |
| Permutation Entropy| Ordinal patterns, robust to noise    |

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Not strictly required                              |
| Missing values   | Not allowed                                        |
| Normalization    | **Discretization for Shannon**                     |
| Minimum samples  | 200+ for Sample/Approximate Entropy                |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# For Shannon: discretize
discrete = pd.qcut(series, q=10, labels=False, duplicates='drop')

# For Sample/Approximate: z-score recommended
z = (series - series.mean()) / series.std()
```

**Python libraries:**
- `scipy.stats.entropy` (Shannon)
- `antropy` (Sample, Approximate, Permutation)
- `nolds.sampen`

**Notes:**
- Permutation entropy is robust and fast
- Compare across indicators for relative complexity
- Low entropy = more predictable

---

### 14. Cluster Analysis (K-Means, Hierarchical, DBSCAN)

**Category:** Structure / Grouping

**What it measures:**
- Natural groupings of indicators
- Behavioral similarity clusters
- Cluster stability over time

**Key outputs:**
- Cluster assignments
- Cluster centroids
- Silhouette scores

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Preferred for returns-based                        |
| Missing values   | Not allowed                                        |
| Normalization    | **Z-score required**                               |
| Minimum samples  | 50+ per clustering window                          |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Z-score each indicator
for col in df.columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# Or use correlation matrix as distance
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

corr_matrix = df.corr()
distance_matrix = 1 - corr_matrix  # Convert correlation to distance
condensed = squareform(distance_matrix, checks=False)
linkage_matrix = linkage(condensed, method='ward')
```

**Python libraries:**
- `sklearn.cluster.KMeans`
- `scipy.cluster.hierarchy`
- `sklearn.cluster.DBSCAN`

**Notes:**
- Track cluster membership changes for regime detection
- Use silhouette score to choose k
- Hierarchical gives dendrogram for visualization

---

### 15. Cointegration (Engle-Granger, Johansen)

**Category:** Long-Run Equilibrium

**What it measures:**
- Long-run equilibrium relationships
- Pairs that move together despite short-term divergence
- Mean-reverting spreads

**Key outputs:**
- Cointegration test statistic
- Cointegrating vector (hedge ratios)
- Error correction speed

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | **Non-stationary levels OK** (that's the point)    |
| Missing values   | Not allowed                                        |
| Normalization    | **None** — use levels                              |
| Minimum samples  | 250+ recommended                                   |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Use levels (prices), not returns
# Ensure same time alignment
aligned = pd.concat([series_x, series_y], axis=1).dropna()
```

**Python libraries:**
- `statsmodels.tsa.stattools.coint` (Engle-Granger)
- `statsmodels.tsa.vector_ar.vecm.coint_johansen` (Johansen)

**Notes:**
- Engle-Granger for pairs
- Johansen for multiple series
- Time-varying cointegration via rolling windows

---

### 16. GARCH Volatility

**Category:** Volatility Dynamics

**What it measures:**
- Conditional volatility (time-varying)
- Volatility clustering
- Persistence of shocks

**Key outputs:**
- Conditional variance series
- GARCH parameters (α, β)
- Volatility forecasts

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Use returns (typically stationary)                 |
| Missing values   | Not allowed                                        |
| Normalization    | **None** — use raw returns                         |
| Minimum samples  | 500+ for stable estimation                         |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Convert to returns (percentage or log)
returns = 100 * prices.pct_change().dropna()  # Percentage returns
# or
log_returns = 100 * np.log(prices / prices.shift(1)).dropna()
```

**Python libraries:**
- `arch` (pip install arch)

**Notes:**
- Compare GARCH(1,1), EGARCH, GJR-GARCH
- Track conditional correlation via DCC-GARCH
- Volatility of volatility for regime detection

---

### 17. Lyapunov Exponent

**Category:** Chaos / Nonlinear Dynamics

**What it measures:**
- Sensitivity to initial conditions
- Chaotic vs stable dynamics
- Predictability horizon

**Key outputs:**
- Largest Lyapunov exponent (λ)
  - λ > 0: Chaotic
  - λ ≈ 0: Edge of chaos
  - λ < 0: Stable

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Not required                                       |
| Missing values   | Not allowed                                        |
| Normalization    | **Z-score recommended**                            |
| Minimum samples  | 1000+ for reliable estimation                      |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Z-score normalization
z = (series - series.mean()) / series.std()

# Requires embedding (same as RQA)
# Use mutual information for delay, FNN for dimension
```

**Python libraries:**
- `nolds.lyap_r` (Rosenstein)
- `nolds.lyap_e` (Eckmann)

**Notes:**
- Computationally intensive
- Requires careful embedding parameter selection
- Compare across indicators for relative chaos

---

### 18. Graph Centrality (Network Analysis)

**Category:** Network / Systemic Importance

**What it measures:**
- Indicator importance in the network
- Information flow hubs
- Systemic risk contributors

**Key outputs:**
- Degree centrality
- Betweenness centrality
- Eigenvector centrality
- PageRank

**Data requirements:**
| Requirement      | Specification                                      |
|------------------|----------------------------------------------------|
| Stationarity     | Preferred for correlation-based networks           |
| Missing values   | Not allowed                                        |
| Normalization    | **Z-score for returns**                            |
| Minimum samples  | 60+ for stable correlation estimation              |
| Frequency        | Uniform time steps                                 |

**Pre-processing:**
```python
# Build network from correlation matrix
import networkx as nx

# Z-score returns
returns_z = returns.apply(lambda x: (x - x.mean()) / x.std())

# Correlation matrix
corr = returns_z.corr()

# Threshold to create adjacency
threshold = 0.5
adj = (corr.abs() > threshold).astype(int)
np.fill_diagonal(adj.values, 0)

# Create graph
G = nx.from_pandas_adjacency(adj)
```

**Python libraries:**
- `networkx`
- `igraph`

**Notes:**
- MST (Minimum Spanning Tree) for filtered network
- Track centrality changes for regime detection
- PMFG as alternative to MST

---

## Normalization Summary Table

| Method                | Primary Normalization | Alternative           | Notes                          |
|-----------------------|-----------------------|-----------------------|--------------------------------|
| PCA                   | Z-score               | —                     | Required for comparability     |
| Granger Causality     | None                  | Z-score               | Stationarity more important    |
| Wavelet Coherence     | None                  | —                     | Scale-invariant                |
| Transfer Entropy      | Discretization        | KDE                   | Binning required               |
| DTW                   | Z-score               | Min-Max               | Amplitude invariance           |
| Hurst Exponent        | None                  | —                     | Works on raw levels            |
| Spectral Density      | Detrend               | Returns               | Remove trend                   |
| Mutual Information    | Discretization        | KNN (continuous)      | Binning or KDE                 |
| Copula                | Rank → Uniform        | —                     | Probability integral transform |
| RQA                   | Z-score               | —                     | Embedding requires scaling     |
| Cross-Correlation     | Z-score               | Pre-whitening         | Comparability                  |
| Rolling Beta          | Returns               | —                     | Use returns, not levels        |
| Entropy               | Discretization        | Z-score for Sample    | Depends on variant             |
| Clustering            | Z-score               | Correlation distance  | Comparability across features  |
| Cointegration         | None                  | —                     | Uses levels                    |
| GARCH                 | Returns               | —                     | Percentage returns typical     |
| Lyapunov Exponent     | Z-score               | —                     | Embedding requires scaling     |
| Graph Centrality      | Z-score               | —                     | For correlation network        |

---

## Data Cleaning Requirements

### Universal Requirements (All Lenses)

```python
def clean_for_lens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universal data cleaning for PRISM lenses.
    
    Args:
        df: DataFrame with DatetimeIndex, indicators as columns
    
    Returns:
        Cleaned DataFrame
    """
    # 1. Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 2. Sort by time
    df = df.sort_index()
    
    # 3. Remove exact duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # 4. Handle missing values
    #    Option A: Forward fill then back fill (conservative)
    df = df.ffill().bfill()
    #    Option B: Interpolate (for smoother series)
    #    df = df.interpolate(method='time')
    #    Option C: Drop (if missing rate is high)
    #    df = df.dropna()
    
    # 5. Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    # 6. Ensure uniform frequency (resample if needed)
    # freq = pd.infer_freq(df.index)
    # if freq is None:
    #     df = df.resample('D').last().ffill()
    
    return df
```

### Outlier Handling

```python
def winsorize(series: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """
    Clip extreme values to specified quantiles.
    """
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower, upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Return boolean mask of outliers based on z-score.
    """
    z = (series - series.mean()) / series.std()
    return z.abs() > threshold


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Return boolean mask of outliers based on IQR.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)
```

---

## Implementation Priority

### Phase 1: Core Lenses (Start Here)
1. **PCA** — Foundational, well-understood
2. **Cross-Correlation** — Simple lead/lag
3. **Rolling Beta** — Sensitivity dynamics
4. **Clustering** — Behavioral grouping
5. **Hurst Exponent** — Persistence/Mandelbrot

### Phase 2: Information & Causality
6. **Granger Causality** — Directional prediction
7. **Mutual Information** — Non-linear dependence
8. **Transfer Entropy** — Information flow

### Phase 3: Frequency & Complexity
9. **Wavelet Coherence** — Time-frequency
10. **Entropy** (Permutation) — Complexity
11. **Spectral Density** — Dominant cycles

### Phase 4: Advanced / Research
12. **Copula Dependence** — Tail risk
13. **RQA** — Regime dynamics
14. **Lyapunov Exponent** — Chaos detection
15. **Graph Centrality** — Network importance
16. **Cointegration** — Long-run equilibrium
17. **GARCH** — Volatility dynamics
18. **DTW** — Shape similarity

---

## Agent Lens Selection Logic

The agent can optimize lens selection based on:

```python
def select_lenses(
    data_characteristics: dict,
    computational_budget: float,
    analysis_goal: str
) -> list:
    """
    Agent-driven lens selection.
    
    Args:
        data_characteristics: {
            'n_samples': int,
            'n_indicators': int,
            'frequency': str,  # 'daily', 'monthly'
            'has_trends': bool,
            'has_seasonality': bool,
        }
        computational_budget: float (seconds)
        analysis_goal: str  # 'structure', 'causality', 'regime', 'all'
    
    Returns:
        List of lens names to run
    """
    lenses = []
    
    # Always include foundational
    lenses.append('pca')
    lenses.append('cross_correlation')
    
    # Structure analysis
    if analysis_goal in ('structure', 'all'):
        lenses.extend(['clustering', 'copula'])
    
    # Causality analysis
    if analysis_goal in ('causality', 'all'):
        if data_characteristics['n_samples'] > 100:
            lenses.append('granger')
        if data_characteristics['n_samples'] > 500:
            lenses.append('transfer_entropy')
    
    # Regime detection
    if analysis_goal in ('regime', 'all'):
        lenses.extend(['hurst', 'entropy'])
        if data_characteristics['n_samples'] > 500:
            lenses.append('rqa')
    
    # Frequency analysis (if enough samples)
    if data_characteristics['n_samples'] >= 128:
        lenses.append('wavelet')
    
    # Trim based on computational budget
    # (order by information value / compute cost)
    
    return lenses
```

---

## References

### Key Papers
- Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices"
- Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models"
- Schreiber, T. (2000). "Measuring Information Transfer" (Transfer Entropy)
- Torrence & Compo (1998). "A Practical Guide to Wavelet Analysis"

### Python Ecosystem
- `statsmodels` — Granger, cointegration, ARIMA
- `sklearn` — PCA, clustering, mutual information
- `scipy` — FFT, correlation, clustering
- `arch` — GARCH models
- `pywt` — Wavelets
- `nolds` — Hurst, Lyapunov, DFA
- `antropy` — Entropy measures
- `networkx` — Graph analysis

---

*Document Version: 1.0.0*
*Generated: December 2024*
*For: PRISM Engine Development*
