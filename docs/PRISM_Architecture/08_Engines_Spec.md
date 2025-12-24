
# PRISM Math Engine Complement — Lens & System Engines

## Scope

This document specifies the **mathematical engine complement** used by PRISM for:

1. Indicator-level engines
2. Lens-level (cross-indicator) engines
3. System-level engines (indicator significance within system)

This is a **normative specification**.  
Engines defined here must be implemented exactly once and reused across contexts.

All engines operate on **time-indexed data**.

---

## 1. Shared Mathematical Substrate

### 1.1 Canonical Input Form

All engines consume data in the form:

x_i(t)

Where:
- i = indicator index
- t = time (DATE or TIMESTAMP)
- x = observed or derived scalar value

For multi-indicator contexts, inputs form a vector:

X(t) = [x_1(t), x_2(t), ..., x_n(t)]^T

Time alignment is performed **prior** to engine execution using SQL.

---

## 2. Indicator Engines (Single-Series Context)

Indicator engines operate on a single series x(t).

### 2.1 Normalization Engines

#### Z-Score Normalization

z(t) = (x(t) - μ) / σ

Where:
- μ = mean over window W
- σ = standard deviation over window W

#### Min-Max Scaling

x_norm(t) = (x(t) - min_W) / (max_W - min_W)

---

### 2.2 Trend & Slope Engines

#### Linear Trend (OLS)

x(t) = a t + b

Slope a is retained as the trend metric.

#### Rolling Difference

Δx(t) = x(t) - x(t - k)

---

### 2.3 Volatility Engines

#### Rolling Variance

Var_W(t) = E[(x - μ)^2]

#### Rolling Standard Deviation

σ_W(t) = sqrt(Var_W(t))

---

## 3. Lens Engines (Multi-Series Geometry)

Lens engines apply the **same math** to vector inputs X(t).

### 3.1 Correlation Engine

For indicators i, j:

ρ_ij = Cov(x_i, x_j) / (σ_i σ_j)

Produces a correlation matrix R(t).

---

### 3.2 Distance Engines

#### Euclidean Distance

d_ij(t) = ||x_i(t) - x_j(t)||_2

#### Mahalanobis Distance

d_i(t) = sqrt((x_i - μ)^T Σ^{-1} (x_i - μ))

Where Σ is the covariance matrix.

---

### 3.3 Principal Components (PCA)

X_c = X - μ

Σ = (1 / n) X_c^T X_c

Eigen decomposition:

Σ v_k = λ_k v_k

Outputs:
- eigenvalues λ_k (variance explained)
- eigenvectors v_k (axes of geometry)

---

### 3.4 Coherence / Alignment

Given two vectors:

cos(θ_ij) = (x_i · x_j) / (||x_i|| ||x_j||)

Measures directional alignment.

---

## 4. System Engines (Indicator Significance)

System engines quantify **influence of indicators relative to the system**.

### 4.1 Contribution via Projection

Let:
- v_k = dominant system eigenvector
- x_i = indicator vector

Contribution score:

C_i = |x_i · v_k| / ||v_k||

---

### 4.2 Energy / Magnitude Contribution

Given system energy:

E_sys = Σ_k λ_k

Indicator contribution:

E_i = Σ_k (λ_k * |v_{k,i}|)

Normalized:

w_i = E_i / Σ_i E_i

---

### 4.3 Sensitivity (Perturbation)

Remove indicator i:

ΔE_i = E_sys - E_sys(-i)

Measures marginal influence.

---

## 5. Temporal Evolution

All metrics are functions of time:

Metric(t, W)

Where W is an explicit rolling window.

No engine operates on implicit or global time.

---

## 6. Engine Reuse Rule (Critical)

Each engine:
- is implemented once
- accepts a context flag:
  - single-series
  - multi-series
- does not branch mathematically based on domain

Only input dimensionality changes.

---

## 7. Persistence

All outputs are persisted with:

- run_id
- engine_id
- parameter_hash
- time
- indicator_id (if applicable)

---

## 8. Prohibited Behavior

- No ad-hoc math
- No reimplementation per subsystem
- No domain-specific tuning inside engines
- No hidden normalization

---

## 9. Summary

Indicator Engines:
- measure properties of x(t)

Lens Engines:
- measure geometry of X(t)

System Engines:
- measure influence of x_i(t) within X(t)

All share:
- time
- math
- parameters
- persistence rules

This completes the PRISM math engine specification.
