# PRISM Technical Review
## Probabilistic Regime Identification through Systematic Measurement
### December 19, 2025

---

## Executive Summary

PRISM is a multi-domain geometric measurement framework for detecting structural regimes in complex systems. Unlike traditional approaches that use hardcoded thresholds to define regimes, PRISM discovers geometric states directly from data using 20 mathematical "lenses" and allows behavior analysis bounded within those states.

**Current Status:** Operational observatory with 3 domains, 45 indicators, 155,000+ data rows spanning 77 years.

**Key Innovation:** Domain-agnostic measurement that enables cross-domain structural comparison without predefined rules.

**Philosophical Foundation:** "Observatory, not oracle" — PRISM measures and reports geometric coordinates without interpretation. The system outputs numbers, not labels like "risk-on" or "fear." Interpretation is left to the observer (human or AI).

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRISM OBSERVATORY                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   FINANCE   │    │   CLIMATE   │    │ EPIDEMIOLOGY│             │
│  │   Agent     │    │   Agent     │    │   Agent     │             │
│  │ 15 indicators│   │ 18 indicators│   │ 12 indicators│            │
│  │  2005-2025  │    │  1948-2025  │    │  2005-2025  │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                  │                  │                     │
│         ▼                  ▼                  ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    UNIFIED DATA LAYER                        │   │
│  │                    DuckDB: 155,000+ rows                     │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│         ┌───────────────────┼───────────────────┐                  │
│         ▼                   ▼                   ▼                  │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐           │
│  │  PHASE 1   │      │  PHASE 2   │      │  PHASE 3   │           │
│  │  UNBOUND   │ ───▶ │ STRUCTURE  │ ───▶ │  BOUNDED   │           │
│  │ 20 Engines │      │ Extraction │      │ Projection │           │
│  └────────────┘      └────────────┘      └────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Three-Phase Methodology

### Phase 1: Unbound Analysis
Run 20 mathematical engines on raw data to produce a multi-dimensional "fingerprint" of system state. No regime labels — just measurements.

### Phase 2: Structure Extraction  
Apply dimensionality reduction and clustering to the fingerprint time series. Discover natural geometric states from the data itself, not from predefined thresholds.

### Phase 3: Bounded Analysis
Project indicators into behavioral space WITHIN each geometric state. The same indicator behaves differently in different states. This reveals conditional dynamics invisible to unconditional analysis.

---

## 3. The 20 Mathematical Engines

Each engine examines the data through a different mathematical lens. Together, they produce a high-dimensional fingerprint that captures system structure.

---

### 3.1 PRINCIPAL COMPONENT ANALYSIS (PCA)

**Purpose:** Measure how variance is distributed across dimensions.

**Eigendecomposition:**
```
Σ = VΛVᵀ
```
Where Σ = covariance matrix, V = eigenvectors, Λ = eigenvalues

**Variance Explained:**
```
VE_k = λ_k / Σᵢλᵢ
```

**Effective Dimensionality:**
```
D_eff = exp(-Σᵢ pᵢ log(pᵢ))   where pᵢ = λᵢ / Σⱼλⱼ
```

**Outputs:** variance_pc1, effective_dimensionality, n_components_90

---

### 3.2 CROSS-CORRELATION ANALYSIS

**Purpose:** Measure pairwise relationships between indicators.

**Pearson Correlation:**
```
ρ_xy = Cov(X,Y) / (σ_x · σ_y)
```

**Average Absolute Correlation:**
```
ρ̄ = (2 / n(n-1)) · Σᵢ<ⱼ |ρᵢⱼ|
```

**Outputs:** avg_abs_correlation, max_correlation, correlation_dispersion

---

### 3.3 HURST EXPONENT

**Purpose:** Measure long-term memory and persistence.

**R/S Scaling:**
```
E[R(n)/S(n)] ~ C · nᴴ

H = slope of log(R/S) vs log(n)
```

**Interpretation:**
- H > 0.5: Persistent (trends continue)
- H = 0.5: Random walk
- H < 0.5: Mean-reverting

**Outputs:** hurst_exponent, hurst_confidence

---

### 3.4 LYAPUNOV EXPONENT

**Purpose:** Measure sensitivity to initial conditions (chaos).

**Definition:**
```
|δZ(t)| ≈ |δZ₀| · eᵏᵗ

λ = lim[t→∞] (1/t) · ln(|δZ(t)| / |δZ₀|)
```

**Interpretation:**
- λ > 0: Chaotic
- λ < 0: Stable

**Outputs:** avg_lyapunov, max_lyapunov, n_chaotic, n_stable

---

### 3.5 ENTROPY ANALYSIS

**Purpose:** Measure information content and unpredictability.

**Shannon Entropy:**
```
H(X) = -Σᵢ p(xᵢ) · log₂(p(xᵢ))
```

**Normalized Entropy:**
```
H_norm = H(X) / log₂(n_bins)
```

**Outputs:** avg_entropy, min_entropy, max_entropy, entropy_dispersion

---

### 3.6 COPULA ANALYSIS

**Purpose:** Measure tail dependence — do extremes happen together?

**Lower Tail Dependence:**
```
λ_L = lim[u→0⁺] C(u,u) / u
```

**Upper Tail Dependence:**
```
λ_U = lim[u→1⁻] (1 - 2u + C(u,u)) / (1-u)
```

**Outputs:** avg_lower_tail, avg_upper_tail, tail_asymmetry

---

### 3.7 ROLLING VOLATILITY

**Purpose:** Measure realized volatility over time.

**Annualized Volatility:**
```
σ = std(returns) · √252
```

**Outputs:** avg_volatility, max_volatility, vol_of_vol

---

### 3.8 GARCH ANALYSIS

**Purpose:** Model conditional volatility clustering.

**GARCH(1,1):**
```
σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²
```

**Persistence:** α + β (close to 1 = highly persistent)

**Outputs:** avg_persistence, avg_alpha, avg_beta

---

### 3.9 HIDDEN MARKOV MODEL (HMM)

**Purpose:** Infer latent regime states.

**Transition Matrix:**
```
A[i,j] = P(sₜ = j | sₜ₋₁ = i)
```

**Outputs:** n_states, current_state, state_stability, transition_frequency

---

### 3.10 WAVELET COHERENCE

**Purpose:** Measure correlation across different time scales.

**Continuous Wavelet Transform:**
```
W_x(a,b) = (1/√a) · ∫ x(t) · ψ*((t-b)/a) dt
```

**Coherence:**
```
R²_xy(a,b) = |S(W_xy)|² / [S(|W_x|²) · S(|W_y|²)]
```

**Outputs:** avg_coherence_short, avg_coherence_long, multi_frequency_coupling

---

### 3.11 GRANGER CAUSALITY

**Purpose:** Test whether one series helps predict another.

**F-Test:**
```
H₀: X does not Granger-cause Y
```

**Outputs:** n_causal_pairs, causal_density, bidirectional_pairs

---

### 3.12 SPECTRAL ANALYSIS

**Purpose:** Decompose into frequency components.

**Power Spectral Density:**
```
P(f) = |FFT(x)|²
```

**Outputs:** dominant_frequency, dominant_period, spectral_entropy

---

### 3.13 RECURRENCE ANALYSIS

**Purpose:** Quantify how often system returns to previous states.

**Recurrence Rate:**
```
RR = (1/N²) · Σᵢ,ⱼ R[i,j]
```

**Outputs:** recurrence_rate, determinism, avg_diagonal_length

---

### 3.14 NETWORK ANALYSIS

**Purpose:** Treat indicators as network nodes.

**Degree Centrality:**
```
C_D(i) = Σⱼ A[i,j] / (n-1)
```

**Outputs:** network_density, avg_degree, clustering_coefficient

---

### 3.15 REGIME CHANGE DETECTION (CUSUM/PELT)

**Purpose:** Detect structural breaks.

**PELT Objective:**
```
Minimize: Σᵢ C(segment_i) + β·n_changepoints
```

**Outputs:** n_changepoints, last_changepoint, avg_segment_length

---

### 3.16 COINTEGRATION ANALYSIS

**Purpose:** Test for long-run equilibrium relationships.

**Engle-Granger:**
```
Yₜ = α + βXₜ + εₜ   (test εₜ for stationarity)
```

**Outputs:** n_cointegrated_pairs, cointegration_density, avg_half_life

---

### 3.17 MUTUAL INFORMATION

**Purpose:** Measure nonlinear dependence.

**Mutual Information:**
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

**Outputs:** avg_mutual_info, max_mutual_info, mi_vs_correlation_ratio

---

### 3.18 DISTRIBUTION MOMENTS

**Purpose:** Characterize distribution shape.

**Skewness:**
```
γ₁ = E[(X-μ)³] / σ³
```

**Kurtosis:**
```
γ₂ = E[(X-μ)⁴] / σ⁴
```

**Outputs:** avg_skewness, avg_kurtosis, max_kurtosis

---

### 3.19 AUTOCORRELATION ANALYSIS

**Purpose:** Measure self-similarity over time.

**Autocorrelation at lag k:**
```
ρ(k) = Cov(Xₜ, Xₜ₋ₖ) / Var(Xₜ)
```

**Outputs:** avg_autocorr_1, avg_autocorr_5, avg_autocorr_21

---

### 3.20 TREND STRENGTH

**Purpose:** Measure trend vs. noise.

**R² of Linear Fit:**
```
R² = 1 - (SS_res / SS_tot)
```

**Outputs:** avg_trend_strength, avg_trend_slope, n_trending_up

---

## 4. Current Data Holdings

| Domain | Indicators | Rows | Date Range |
|--------|------------|------|------------|
| Finance | 15 | 126,661 | 2005-2025 |
| Climate | 18 | 15,644 | 1948-2025 |
| Epidemiology | 12 | 12,288 | 2005-2025 |
| **TOTAL** | **45** | **154,593** | **77 years** |

---

## 5. Key Results

### Pre-2008 Deformation
```
2007-06-21: First Stress State entry (452 days before Lehman)
2008-06-25: Re-entry (82 days before Lehman)
WARNING LEAD TIME: 15 months
```

### Current State (Dec 2025)
```
State: 4 (Not Stress)
PC1 Variance: 0.6549 (healthy)
Effective Dim: 2.09 (moderate)
Pattern: Bouncing, not sustained deformation
```

### GFC vs COVID Signatures
| Metric | GFC | COVID |
|--------|-----|-------|
| Duration | 517 days | 33 days |
| States traversed | 4 | 1 |
| Signature | Slow internal | Instant external |

---

## 6. Infrastructure

| Component | Specification |
|-----------|---------------|
| Server | Hetzner CPX41 (8 vCPU, 16GB) |
| Cost | $34/month |
| Database | DuckDB |
| Stack | Python 3.12, SciPy, Scikit-learn |

---

## 7. Key Differentiators

1. **Measures, doesn't predict** — no overfitting
2. **Discovers structure from data** — no hardcoded thresholds
3. **Cross-domain comparison** — unified geometric framework
4. **Complements AI** — provides transparent inputs
5. **Validates against known events** — El Niño, GFC, COVID

---

## 8. Roadmap

| Phase | Status |
|-------|--------|
| Core Engines | ✅ Complete |
| Multi-Domain Agents | ✅ Complete |
| Cross-Domain Analysis | 🔄 Running |
| Sub-Geometry | 📋 Planned |
| Real-Time Pipeline | 📋 Planned |
| Publication | 📋 Planned |

---

*PRISM Observatory v0.2.0 — December 19, 2025*
