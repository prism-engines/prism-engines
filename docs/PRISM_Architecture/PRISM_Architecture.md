# PRISM Architecture — Refined Specification

**Version:** 2.0  
**Status:** Draft for Review  
**Purpose:** Canonical reference for all PRISM subsystems

---

## Document Map

| Section | Purpose |
|---------|---------|
| 00 | Core Primitives (NEW) |
| 01 | Persistence |
| 02 | Indicator Math |
| 03 | Lens Math |
| 04 | System Math |
| 05 | Runners (EXPANDED) |
| 06 | Data Lineage |
| 07 | Fetchers |
| 08 | Failure Handling (NEW) |

---

# 00 — Core Primitives

## Purpose
Define canonical data structures shared across all subsystems.
These primitives are referenced by name throughout the architecture.

---

## 00.1 Window Specification

A **Window** defines a bounded time range for computation.

### Canonical Structure

```python
@dataclass
class Window:
    window_id: str           # unique identifier (hash of params)
    anchor: date             # reference point
    lookback_days: int       # days before anchor (inclusive)
    lookahead_days: int      # days after anchor (usually 0)
    window_type: str         # "rolling" | "fixed" | "expanding"
    
    @property
    def start_date(self) -> date:
        return self.anchor - timedelta(days=self.lookback_days)
    
    @property
    def end_date(self) -> date:
        return self.anchor + timedelta(days=self.lookahead_days)
```

### Window Types

| Type | Behavior | Use Case |
|------|----------|----------|
| `rolling` | Slides forward, fixed width | Moving averages, rolling volatility |
| `fixed` | Static boundaries | Era analysis, regime comparison |
| `expanding` | Start fixed, end grows | Cumulative statistics |

### Window ID Computation

```python
def compute_window_id(anchor: date, lookback: int, lookahead: int, wtype: str) -> str:
    payload = f"{anchor.isoformat()}|{lookback}|{lookahead}|{wtype}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

### Constraints
- All engines must accept `Window` objects, not raw date pairs
- Windows are immutable once created
- `lookahead_days` must be 0 for any causal analysis

---

## 00.2 Parameter Hash Specification

Every computation must be traceable to its exact parameters.

### Hash Computation (Canonical)

```python
import hashlib
import json

def compute_parameter_hash(params: dict) -> str:
    """
    Canonical parameter hash computation.
    
    Rules:
    - Keys sorted alphabetically
    - Values JSON-serialized with sort_keys=True
    - Dates converted to ISO strings
    - Floats rounded to 10 decimal places
    - None values included explicitly
    """
    normalized = {}
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, date):
            value = value.isoformat()
        elif isinstance(value, float):
            value = round(value, 10)
        normalized[key] = value
    
    payload = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

### Required Parameter Sets by Subsystem

| Subsystem | Required Parameters |
|-----------|---------------------|
| Fetcher | `indicator_id`, `source`, `fetch_params` |
| Indicator Math | `indicator_id`, `window`, `engine_id`, `engine_params` |
| Lens Math | `indicator_ids` (sorted list), `window`, `lens_id`, `lens_params` |
| System Math | `lens_output_ids`, `window`, `system_engine_id` |

### Constraints
- Same parameters MUST produce same hash across all subsystems
- Hash computation is implemented ONCE in `prism/core/hashing.py`
- No subsystem may implement its own hash function

---

## 00.3 Engine Versioning

Engines evolve. Old results must remain interpretable.

### Version Structure

```python
@dataclass
class EngineVersion:
    engine_id: str          # e.g., "zscore", "pca", "correlation"
    major: int              # breaking changes to output schema
    minor: int              # changes to algorithm (same schema)
    patch: int              # bug fixes (same algorithm)
    
    def __str__(self) -> str:
        return f"{self.engine_id}@{self.major}.{self.minor}.{self.patch}"
```

### Version Semantics

| Change Type | Version Bump | Backward Compatible |
|-------------|--------------|---------------------|
| Output column added/removed | Major | No |
| Algorithm improvement | Minor | Schema yes, values no |
| Bug fix | Patch | Yes |

### Lineage Requirement

All persisted outputs must include:
```
engine_version: str  # e.g., "pca@2.1.0"
```

### Comparison Rules
- Results from different major versions are NOT directly comparable
- Results from different minor versions may be compared with documented caveats
- Patch versions are always comparable

---

## 00.4 Missing Data Specification

Reality has gaps. PRISM handles them explicitly.

### Missing Data Taxonomy

| Code | Meaning | Source |
|------|---------|--------|
| `NULL` | Value not available | Data gap in source |
| `NaN` | Computation undefined | Division by zero, log(negative) |
| `HOLIDAY` | Market closed | Known non-trading day |
| `PREFETCH` | Before series start | Indicator didn't exist yet |

### Handling Rules by Subsystem

**Fetchers:**
- Return `NULL` for missing dates
- Never interpolate
- Never forward-fill
- Document gaps in fetch metadata

**Indicator Math:**
- Skip `NULL` values in rolling computations
- Require minimum observation threshold (configurable, default 70%)
- Output `NaN` if threshold not met
- Propagate `NULL` to output if input is `NULL`

**Lens Math:**
- Require time alignment BEFORE lens execution
- Options for missing data (specified per-lens):
  - `skip`: Exclude timepoint from computation
  - `fail`: Raise error if any indicator missing
  - `pairwise`: Use available pairs (correlation only)
- Default: `skip`

**System Math:**
- Operates only on complete lens outputs
- Missing lens outputs cause run failure (no partial system metrics)

### Minimum Observation Threshold

```python
def check_observation_threshold(
    series: pd.Series, 
    window: Window, 
    min_pct: float = 0.70
) -> bool:
    """Returns True if sufficient observations exist."""
    expected = (window.end_date - window.start_date).days + 1
    actual = series.notna().sum()
    return (actual / expected) >= min_pct
```

### Constraints
- Interpolation is NEVER performed implicitly
- Forward-fill is NEVER performed implicitly
- If smoothing/filling is needed, it must be an explicit engine with its own `engine_id`

---

# 01 — Persistence Subsystem

## Purpose
The Persistence Subsystem is responsible for **durable state, lineage, and replayability**.
It is the only subsystem permitted to interact directly with DuckDB and execute SQL.

## Technologies
- DuckDB (embedded OLAP database)
- SQL (DuckDB dialect)
- Python (duckdb, pandas)

## Responsibilities
- Persist all fetched, derived, and computed data
- Persist execution metadata and failures
- Provide reproducible historical state
- Enforce schema and type consistency

## Forbidden Responsibilities
- No fetching
- No mathematical computation
- No interpretation or domain logic
- No orchestration decisions

## Core Tables

### fetch_runs
```sql
CREATE TABLE fetch_runs (
    run_id          VARCHAR NOT NULL,
    indicator_id    VARCHAR NOT NULL,
    source          VARCHAR NOT NULL,
    status          VARCHAR NOT NULL,  -- 'OK' | 'FAIL' | 'PARTIAL'
    rows_fetched    INTEGER,
    error_message   VARCHAR,
    parameter_hash  VARCHAR NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    PRIMARY KEY (run_id, indicator_id)
);
```

### indicator_values
```sql
CREATE TABLE indicator_values (
    indicator_id    VARCHAR NOT NULL,
    date            DATE NOT NULL,
    value           DOUBLE,
    source          VARCHAR NOT NULL,
    fetch_run_id    VARCHAR NOT NULL,
    fetched_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (indicator_id, date)
);
```

### indicator_derived
```sql
CREATE TABLE indicator_derived (
    run_id          VARCHAR NOT NULL,
    indicator_id    VARCHAR NOT NULL,
    window_id       VARCHAR NOT NULL,
    engine_id       VARCHAR NOT NULL,
    engine_version  VARCHAR NOT NULL,
    metric_name     VARCHAR NOT NULL,
    date            DATE NOT NULL,
    value           DOUBLE,
    parameter_hash  VARCHAR NOT NULL,
    computed_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, indicator_id, window_id, engine_id, metric_name, date)
);
```

### lens_outputs
```sql
CREATE TABLE lens_outputs (
    run_id          VARCHAR NOT NULL,
    lens_id         VARCHAR NOT NULL,
    lens_version    VARCHAR NOT NULL,
    window_id       VARCHAR NOT NULL,
    indicator_i     VARCHAR NOT NULL,
    indicator_j     VARCHAR,           -- NULL for single-indicator metrics
    metric_name     VARCHAR NOT NULL,
    date            DATE NOT NULL,
    value           DOUBLE,
    parameter_hash  VARCHAR NOT NULL,
    computed_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, lens_id, window_id, indicator_i, indicator_j, metric_name, date)
);
```

### system_metrics
```sql
CREATE TABLE system_metrics (
    run_id              VARCHAR NOT NULL,
    system_engine_id    VARCHAR NOT NULL,
    engine_version      VARCHAR NOT NULL,
    window_id           VARCHAR NOT NULL,
    indicator_id        VARCHAR NOT NULL,
    metric_name         VARCHAR NOT NULL,
    date                DATE NOT NULL,
    value               DOUBLE,
    parameter_hash      VARCHAR NOT NULL,
    computed_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, system_engine_id, window_id, indicator_id, metric_name, date)
);
```

### run_failures
```sql
CREATE TABLE run_failures (
    failure_id      VARCHAR PRIMARY KEY,
    run_id          VARCHAR NOT NULL,
    subsystem       VARCHAR NOT NULL,  -- 'fetch' | 'indicator' | 'lens' | 'system'
    component_id    VARCHAR NOT NULL,  -- indicator_id, lens_id, etc.
    error_type      VARCHAR NOT NULL,
    error_message   VARCHAR,
    stack_trace     VARCHAR,
    failed_at       TIMESTAMP NOT NULL,
    recovered       BOOLEAN DEFAULT FALSE
);
```

## DuckDB Usage
- Single-writer model per run
- SQL used for:
  - time alignment
  - joins
  - filtering
  - window materialization
- No SQL logic embedded in math engines

## Lineage
Every table must include:
- `run_id`
- `indicator_id` (where applicable)
- `parameter_hash`
- execution timestamp

Persistence stores facts only.

---

# 02 — Indicator Math Subsystem

## Purpose
Compute **per-indicator analytical measurements** from raw time-series data.

## Technologies
- Python
- pandas
- NumPy (internal only)

## Input Contract

### Tensor Shape
```
x(t) ∈ ℝ  (scalar time series)
```

### Preconditions (enforced by runner, NOT by engine)
- Time alignment: complete
- Window: materialized as `Window` object
- Missing data: strategy resolved; engine receives clean series or `NaN` markers
- Data type: `float64`

### Input DataFrame Schema
```
indicator_id: str    (constant for call)
date: DATE
value: float64       (may contain NaN per missing data rules)
```

## Output
- `indicator_values` from DuckDB (via runner)
- Single `indicator_id`
- `Window` object
- Engine parameters

## Output
- pandas DataFrame with columns:
  - `indicator_id`
  - `date`
  - `window_id`
  - `engine_id`
  - `metric_name`
  - `value`

## Engines

### Normalization Engines

**zscore** (v1.0.0)
```
z(t) = (x(t) - μ_W) / σ_W
```
Parameters: `window`
Output metrics: `zscore`

**minmax** (v1.0.0)
```
x_norm(t) = (x(t) - min_W) / (max_W - min_W)
```
Parameters: `window`
Output metrics: `minmax_scaled`

### Trend Engines

**linear_trend** (v1.0.0)
```
x(t) = a·t + b (OLS fit)
```
Parameters: `window`
Output metrics: `slope`, `intercept`, `r_squared`

**rolling_diff** (v1.0.0)
```
Δx(t) = x(t) - x(t - k)
```
Parameters: `window`, `lag_days`
Output metrics: `diff`

### Volatility Engines

**rolling_std** (v1.0.0)
```
σ_W(t) = sqrt(Var_W(x))
```
Parameters: `window`
Output metrics: `rolling_std`

**rolling_var** (v1.0.0)
```
Var_W(t) = E[(x - μ)²]
```
Parameters: `window`
Output metrics: `rolling_var`

## Constraints
- Operates on one indicator at a time
- No cross-indicator assumptions
- No DuckDB access
- No SQL execution
- No persistence decisions
- Must respect minimum observation threshold

## Persistence
Outputs returned to runner, written to `indicator_derived`.

---

# 03 — Lens Math Subsystem

## Purpose
Apply **shared mathematical engines** across multiple indicators to measure
relationships and geometry.

## Key Constraint
Indicator math engines and lens math engines **share implementations where applicable**.

## Input Contract

### Tensor Shape
```
X(t) ∈ ℝⁿ  (vector time series, n = number of indicators)
```

### Preconditions (enforced by runner, NOT by engine)
- Time alignment: **complete** (all indicators share identical date index)
- Window: materialized as `Window` object
- Missing data: strategy resolved BEFORE engine call
  - `skip`: rows with any NaN removed
  - `fail`: no NaN present (runner would have aborted)
  - `pairwise`: NaN handling delegated to engine (correlation only)
- Data type: `float64`
- Indicator order: sorted alphabetically by `indicator_id` (deterministic)

### Input DataFrame Schema
```
date: DATE           (index)
{indicator_id_1}: float64
{indicator_id_2}: float64
...
{indicator_id_n}: float64
```

## Output
- `indicator_derived` datasets (via runner)
- Time-aligned via SQL BEFORE engine execution
- `Window` object
- Lens parameters
- Missing data strategy: `skip` | `fail` | `pairwise`

## Output
- pandas DataFrame with columns:
  - `lens_id`
  - `window_id`
  - `indicator_i`
  - `indicator_j` (NULL for vector metrics)
  - `date`
  - `metric_name`
  - `value`

## Engines

### Correlation Engine (v1.0.0)
```
ρ_ij = Cov(x_i, x_j) / (σ_i · σ_j)
```
Parameters: `window`, `missing_strategy`
Output metrics: `correlation`

### Distance Engines

**euclidean** (v1.0.0)
```
d_ij(t) = ||x_i(t) - x_j(t)||_2
```
Parameters: `window`
Output metrics: `euclidean_distance`

**mahalanobis** (v1.0.0)
```
d_i(t) = sqrt((x_i - μ)ᵀ Σ⁻¹ (x_i - μ))
```
Parameters: `window`
Output metrics: `mahalanobis_distance`
Note: Requires invertible covariance matrix; fails gracefully if singular.

### PCA Engine (v1.0.0)
```
Σ = (1/n) X_cᵀ X_c
Σ v_k = λ_k v_k
```
Parameters: `window`, `n_components`

**Canonical Outputs (lens_outputs):**

| metric_name | Description | Shape |
|-------------|-------------|-------|
| `eigenvalue_{k}` | Raw eigenvalue for component k | scalar per k |
| `variance_explained_{k}` | `λ_k / Σ_j λ_j` | scalar per k |
| `loading_{indicator}_{k}` | Loading of indicator on component k | scalar per indicator × k |

**Invariants:**
- `Σ_k variance_explained_k = 1.0 ± 1e-9`
- `Σ_k eigenvalue_k = trace(Σ) ± 1e-9`
- Loadings are normalized: `||v_k|| = 1`

**Stability:**
- If `cond(Σ) > CONDITION_MAX`: fail with `COMPUTE_SINGULAR_MATRIX`
- If any `eigenvalue_k < EPSILON`: emit warning, retain in output

**System Engine Contract:**
System engines consume ONLY these canonical fields. No recomputation of PCA inside system engines.

### Coherence Engine (v1.0.0)
```
cos(θ_ij) = (x_i · x_j) / (||x_i|| · ||x_j||)
```
Parameters: `window`
Output metrics: `coherence`

## Constraints
- No fetching
- No raw data access
- No persistence logic
- No domain interpretation
- Time alignment is PRECONDITION, not responsibility

## Persistence
Outputs returned to runner, written to `lens_outputs`.

---

# 04 — System Math Subsystem

## Purpose
Quantify **indicator significance within the full system context**.

## Input Contract

### Tensor Shape
```
Lens outputs: structured records from lens_outputs table
Indicator derived: structured records from indicator_derived table
```

### Preconditions (enforced by runner)
- PCA outputs exist for requested window
- All canonical PCA fields present (eigenvalues, loadings, variance_explained)
- PCA invariants verified: `Σ variance_explained = 1 ± ε`

### Time Semantics (Critical)

**System engines evaluate at the same time index as lens outputs.**

Rules:
- No aggregation across time inside system engines
- No hidden smoothing or averaging
- If temporal reduction is needed, it must be:
  - Declared as a separate engine
  - Emit a distinct `metric_name` (e.g., `mean_contribution_30d`)
  - Document the reduction explicitly

Example of **prohibited** behavior:
```python
# WRONG: hidden temporal averaging
contribution = lens_outputs['loading'].rolling(30).mean()
```

Example of **correct** behavior:
```python
# RIGHT: point-in-time evaluation
contribution = compute_contribution(lens_outputs.loc[date])

# OR: explicit temporal reduction engine
class RollingContributionEngine:
    engine_id = "rolling_contribution"
    # ... explicit declaration of temporal semantics
```

## Input
- `lens_outputs` (specifically PCA results)
- `indicator_derived`
- System composition metadata
- `Window` object

## Output
- pandas DataFrame with columns:
  - `system_engine_id`
  - `window_id`
  - `indicator_id`
  - `date`
  - `metric_name`
  - `value`

## Engines

### Projection Contribution (v1.0.0)
```
C_i = |x_i · v_k| / ||v_k||
```
Where v_k is the dominant eigenvector.
Parameters: `window`, `component_index`
Output metrics: `projection_contribution`

### Energy Contribution (v1.0.0)
```
E_i = Σ_k (λ_k · |v_{k,i}|)
w_i = E_i / Σ_i E_i
```
Parameters: `window`
Output metrics: `energy_contribution`, `normalized_weight`

### Perturbation Sensitivity (v1.0.0)
```
ΔE_i = E_sys - E_sys(-i)
```
Parameters: `window`
Output metrics: `perturbation_sensitivity`

**Computational Complexity:**
```
O(n × PCA_cost)
```
Where n = number of indicators. For n=32 indicators, this means 33 PCA computations per window.

**Caching Rule (Mandatory):**
- Baseline PCA for the full system MUST be computed once and reused
- Do NOT recompute full-system PCA for each indicator removal
- Implementation pattern:
  ```python
  baseline_energy = compute_system_energy(full_pca_result)
  for indicator in indicators:
      reduced_pca = compute_pca(X.drop(indicator))
      reduced_energy = compute_system_energy(reduced_pca)
      sensitivity[indicator] = baseline_energy - reduced_energy
  ```

**Failure Rules:**
- If covariance singular after indicator removal → emit `COMPUTE_SINGULAR_MATRIX` for that indicator only
- If >50% of indicators fail → abort engine, emit `COMPUTE_BATCH_FAILURE`
- Singular indicators output `NaN`, do not abort entire computation

**Runtime Guard:**
- If n > 100: emit warning, require explicit `allow_large_n=True` parameter
- If estimated runtime > 60s: emit warning to stderr before starting

## Constraints
- No raw data access
- No fetching
- No lens re-computation
- Operates only on persisted analytical outputs

## Persistence
Outputs returned to runner, written to `system_metrics`.

---

# 05 — Runners

## Purpose
Coordinate execution across subsystems with explicit control flow.

## Technologies
- Python
- YAML (configuration)
- UUID (run identification)

## Responsibilities
- Read and validate YAML configuration
- Generate `run_id` at execution start
- Resolve indicators to sources
- Construct `Window` objects
- Invoke subsystems in correct order
- Handle failures according to policy
- Invoke persistence layer

## Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         RUNNER                               │
├─────────────────────────────────────────────────────────────┤
│  1. Load pipeline.yaml                                       │
│  2. Generate run_id                                          │
│  3. Validate configuration                                   │
│  4. For each indicator:                                      │
│     ├─ Invoke fetcher → persist to indicator_values          │
│     └─ On failure: log to run_failures, apply policy         │
│  5. For each indicator + window:                             │
│     ├─ Invoke indicator math → persist to indicator_derived  │
│     └─ On failure: log, apply policy                         │
│  6. Time-align indicators via SQL                            │
│  7. For each lens + window:                                  │
│     ├─ Invoke lens math → persist to lens_outputs            │
│     └─ On failure: log, apply policy                         │
│  8. For each system engine:                                  │
│     ├─ Invoke system math → persist to system_metrics        │
│     └─ On failure: log, apply policy                         │
│  9. Finalize run metadata                                    │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Schema

```yaml
# pipeline.yaml
run:
  name: "quarterly_analysis"
  description: "Q4 2024 indicator analysis"

windows:
  - id: "rolling_252"
    type: "rolling"
    lookback_days: 252
    lookahead_days: 0
  - id: "fixed_2024"
    type: "fixed"
    start_date: "2024-01-01"
    end_date: "2024-12-31"

indicators:
  - id: "SP500"
    source: "yahoo"
    params:
      ticker: "^GSPC"
  - id: "TNX"
    source: "yahoo"
    params:
      ticker: "^TNX"
  - id: "M2"
    source: "fred"
    params:
      series_id: "M2SL"

indicator_engines:
  - engine_id: "zscore"
    apply_to: "all"  # or list of indicator_ids
    windows: ["rolling_252"]
  - engine_id: "rolling_std"
    apply_to: ["SP500", "TNX"]
    windows: ["rolling_252"]

lenses:
  - lens_id: "correlation"
    indicators: "all"
    windows: ["rolling_252"]
    params:
      missing_strategy: "pairwise"
  - lens_id: "pca"
    indicators: "all"
    windows: ["rolling_252", "fixed_2024"]
    params:
      n_components: 3

system_engines:
  - engine_id: "energy_contribution"
    windows: ["rolling_252"]

failure_policy:
  fetch_failure: "continue"      # continue | abort | retry
  indicator_failure: "continue"
  lens_failure: "continue"
  system_failure: "abort"
  max_retries: 3
  retry_delay_seconds: 60
```

## Parameter Flow

```
pipeline.yaml
    │
    ▼
┌─────────────────┐
│  Runner parses  │
│  and validates  │
└────────┬────────┘
         │
         ├──► Window objects constructed
         │
         ├──► Fetcher params extracted
         │    └──► parameter_hash computed
         │
         ├──► Engine params extracted
         │    └──► parameter_hash computed
         │
         └──► Lens params extracted
              └──► parameter_hash computed
```

## Failure Propagation

| Failure At | Default Policy | Dependent Subsystems |
|------------|----------------|----------------------|
| Fetch | Continue | Indicator math skipped for that indicator |
| Indicator Math | Continue | Lens excludes indicator |
| Lens | Continue | System math may fail if input missing |
| System | Abort | Run marked incomplete |

### Dependency Graph

```
fetch(A) ──► indicator_math(A) ──┐
fetch(B) ──► indicator_math(B) ──┼──► lens(A,B,C) ──► system_math
fetch(C) ──► indicator_math(C) ──┘
```

If `fetch(B)` fails:
- `indicator_math(B)` is skipped
- `lens(A,B,C)` executes with only A and C (if `missing_strategy: skip`)
- OR `lens(A,B,C)` fails (if `missing_strategy: fail`)

## Constraints
- No math implementation
- No SQL schema definition
- No data interpretation
- Runner is stateless between runs

## Run Metadata

At run completion, persist:
```python
{
    "run_id": str,
    "started_at": timestamp,
    "completed_at": timestamp,
    "status": "complete" | "partial" | "failed",
    "config_hash": str,  # hash of pipeline.yaml
    "indicators_processed": int,
    "indicators_failed": int,
    "lenses_processed": int,
    "lenses_failed": int
}
```

---

# 06 — Data Lineage

## Purpose
Ensure traceability, reproducibility, and auditability.

## Lineage Fields (Required on All Tables)

| Field | Type | Purpose |
|-------|------|---------|
| `run_id` | VARCHAR | Groups all outputs from single execution |
| `indicator_id` | VARCHAR | Source indicator (where applicable) |
| `parameter_hash` | VARCHAR(16) | Exact params that produced this row |
| `engine_version` | VARCHAR | Engine that produced this output |
| `computed_at` | TIMESTAMP | When computation occurred |

## Reproducibility Contract

Given:
- Same `indicator_values` (by date range)
- Same `parameter_hash`
- Same `engine_version`

PRISM MUST produce identical outputs.

## SQL Role
- Join datasets across time
- Align indicators to common timestamps
- Materialize windows
- Preserve historical state

## DuckDB Role
- Embedded analytical store
- Deterministic replay
- Local + cloud portability

## pandas Role
- In-memory transformation
- Math execution substrate

## Transition Rules

All transitions between SQL and pandas are explicit:

```python
# SQL → pandas (via runner)
df = conn.execute("SELECT * FROM indicator_values WHERE ...").fetchdf()

# pandas → SQL (via runner)
conn.execute("INSERT INTO indicator_derived SELECT * FROM df")
```

Math engines NEVER execute SQL.
Persistence layer NEVER executes math.

---

## 00.5 Numeric Stability Bounds

Engines involving division or matrix inversion must enforce stability bounds.

### Global Constants

```python
# Defined once in prism/core/numeric.py
EPSILON = 1e-12          # Minimum denominator threshold
CONDITION_MAX = 1e12     # Maximum condition number for matrix inversion
```

### Stability Rules

| Operation | Condition | Result |
|-----------|-----------|--------|
| Division by σ | `σ < EPSILON` | Output `NaN`, emit `COMPUTE_INSUFFICIENT_VARIANCE` |
| Correlation | `σ_i < EPSILON` or `σ_j < EPSILON` | Output `NaN` for that pair |
| Matrix inversion | `cond(Σ) > CONDITION_MAX` | Fail with `COMPUTE_SINGULAR_MATRIX` |
| Log transform | `x ≤ 0` | Output `NaN`, emit `COMPUTE_DOMAIN_ERROR` |

### Implementation Pattern

```python
def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < EPSILON:
        return float('nan')
    return numerator / denominator

def check_condition_number(matrix: np.ndarray) -> bool:
    """Returns True if matrix is invertible within stability bounds."""
    cond = np.linalg.cond(matrix)
    return cond < CONDITION_MAX
```

### Propagation Rule

If any intermediate computation produces `NaN` due to stability bounds:
- The engine completes (does not raise)
- Output contains `NaN` for affected metrics
- A `data_quality` failure is logged with specific error code
- Downstream engines receive `NaN` and must handle accordingly

---

# 07 — Fetchers

## Purpose
Retrieve raw time series data from external sources.

## Design Philosophy
1. Domain-agnostic
2. Source-specific
3. Mechanics centralized
4. Observable at runtime
5. Silent failure is forbidden

## File Structure

```
prism/fetchers/
├── base_fetcher.py    # Abstract base class
├── fred.py            # FRED API
├── yahoo.py           # Yahoo Finance
├── tiingo.py          # Tiingo API
└── registry.py        # Source → Fetcher mapping
```

## BaseFetcher Contract

```python
class BaseFetcher(ABC):
    def __init__(self, indicator_id: str, source: str, params: dict):
        self.indicator_id = indicator_id
        self.source = source
        self.params = params
        self.parameter_hash = compute_parameter_hash(params)
    
    @abstractmethod
    def _fetch_once(self) -> pd.DataFrame:
        """
        Fetch data from source.
        
        Returns:
            DataFrame with columns: date, value
        
        Raises:
            FetchError on failure
        """
        pass
    
    def fetch(self, max_retries: int = 3) -> FetchResult:
        """
        Fetch with retry logic and observability.
        
        Emits to stderr:
            [FETCH START ] {indicator_id}
            [FETCH OK    ] {indicator_id} — rows={n}
            [FETCH FAIL  ] {indicator_id} — {error}
        
        Returns:
            FetchResult with status, data, error
        """
        pass
```

## FetchResult Structure

```python
@dataclass
class FetchResult:
    indicator_id: str
    source: str
    status: str           # 'OK' | 'FAIL' | 'PARTIAL'
    data: pd.DataFrame    # Empty on failure
    rows_fetched: int
    error_message: Optional[str]
    parameter_hash: str
    started_at: datetime
    completed_at: datetime
```

## Source Fetcher Requirements

Each concrete fetcher:
- Subclasses `BaseFetcher`
- Implements `_fetch_once()` only
- Returns DataFrame with `date` and `value` columns
- Does NOT handle retries (base class does)
- Does NOT emit status (base class does)
- Does NOT persist data (runner does)

## Constraints
- Fetchers do not know about domains
- Fetchers do not write to DuckDB
- Fetchers do not select indicators
- Fetchers do not read pipeline.yaml

---

# 08 — Failure Handling

## Purpose
Define explicit failure semantics across all subsystems.

## Failure Categories

| Category | Examples | Recovery |
|----------|----------|----------|
| Transient | Network timeout, rate limit | Retry with backoff |
| Permanent | Invalid indicator, auth failure | Skip, log, continue |
| Data Quality | Insufficient observations | Skip computation, flag |
| System | Out of memory, disk full | Abort run |

## Failure Record Schema

```python
@dataclass
class FailureRecord:
    failure_id: str          # UUID
    run_id: str
    subsystem: str           # 'fetch' | 'indicator' | 'lens' | 'system'
    component_id: str        # indicator_id, lens_id, etc.
    error_type: str          # 'transient' | 'permanent' | 'data_quality' | 'system'
    error_code: str          # Specific error code
    error_message: str
    stack_trace: Optional[str]
    context: dict            # Additional debugging info
    failed_at: datetime
    retry_count: int
    recovered: bool
```

## Retry Policy

```python
@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    
    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay_seconds * (self.exponential_base ** attempt)
        return min(delay, self.max_delay_seconds)
```

## Error Codes

### Fetch Errors
| Code | Meaning | Retryable |
|------|---------|-----------|
| `FETCH_TIMEOUT` | Request timed out | Yes |
| `FETCH_RATE_LIMIT` | API rate limit hit | Yes (with delay) |
| `FETCH_AUTH` | Authentication failed | No |
| `FETCH_NOT_FOUND` | Indicator not found | No |
| `FETCH_PARSE` | Response parse error | No |

### Computation Errors
| Code | Meaning | Retryable |
|------|---------|-----------|
| `COMPUTE_INSUFFICIENT_DATA` | Below observation threshold | No |
| `COMPUTE_SINGULAR_MATRIX` | Matrix not invertible | No |
| `COMPUTE_NUMERIC_OVERFLOW` | Value exceeded bounds | No |
| `COMPUTE_TIMEOUT` | Computation took too long | Maybe |

## Cascade Behavior

When a failure occurs, the runner must decide how to handle dependent computations:

```python
class CascadePolicy(Enum):
    SKIP_DEPENDENTS = "skip"      # Skip all dependent computations
    PARTIAL_CONTINUE = "partial"  # Continue with available data
    ABORT_RUN = "abort"           # Stop entire run
```

### Default Cascade Rules

| Failure Point | Cascade Policy | Rationale |
|---------------|----------------|-----------|
| Single indicator fetch | PARTIAL_CONTINUE | Other indicators still valid |
| >50% indicators failed | ABORT_RUN | Insufficient data for analysis |
| Lens computation | PARTIAL_CONTINUE | Other lenses still valid |
| System math | SKIP_DEPENDENTS | No downstream consumers |

## Observability Requirements

All failures emit to stderr:
```
[FAIL {subsystem}] {component_id} — {error_code}: {message}
```

All failures are persisted to `run_failures` table.

All runs include failure summary in metadata:
```python
{
    "failures": {
        "fetch": 2,
        "indicator": 0,
        "lens": 1,
        "system": 0
    },
    "failure_ids": ["uuid1", "uuid2", "uuid3"]
}
```

---

# Appendix A — Engine Registry

## Engine Complement Closure (Normative)

The engine complement is **closed**. Only engines listed in this appendix are valid PRISM engines.

### Adding a New Engine

Introducing a new engine requires:

1. **New `engine_id`** — unique, lowercase, snake_case
2. **Versioned specification** — equations, input/output schema, preconditions
3. **Schema impact review** — does it require new columns in persistence tables?
4. **Migration note** — explicit entry in change log
5. **Benchmark validation** — synthetic data test demonstrating correctness

### Rationale

This closure prevents:
- Metric sprawl ("just add one more")
- Untested computations entering production
- Schema drift from ad-hoc outputs
- Reproducibility failures from undocumented engines

If an engine is not in Appendix A, it does not exist in PRISM.

---

## Indicator Engines

| engine_id | Version | Input | Output Metrics |
|-----------|---------|-------|----------------|
| `zscore` | 1.0.0 | x(t), window | zscore |
| `minmax` | 1.0.0 | x(t), window | minmax_scaled |
| `linear_trend` | 1.0.0 | x(t), window | slope, intercept, r_squared |
| `rolling_diff` | 1.0.0 | x(t), window, lag | diff |
| `rolling_std` | 1.0.0 | x(t), window | rolling_std |
| `rolling_var` | 1.0.0 | x(t), window | rolling_var |

## Lens Engines

| lens_id | Version | Input | Output Metrics |
|---------|---------|-------|----------------|
| `correlation` | 1.0.0 | X(t), window | correlation |
| `euclidean` | 1.0.0 | X(t), window | euclidean_distance |
| `mahalanobis` | 1.0.0 | X(t), window | mahalanobis_distance |
| `pca` | 1.0.0 | X(t), window, n_components | eigenvalue_k, loading_i_k, variance_explained_k |
| `coherence` | 1.0.0 | X(t), window | coherence |

## System Engines

| engine_id | Version | Input | Output Metrics |
|-----------|---------|-------|----------------|
| `projection_contribution` | 1.0.0 | lens_outputs, window | projection_contribution |
| `energy_contribution` | 1.0.0 | lens_outputs, window | energy_contribution, normalized_weight |
| `perturbation_sensitivity` | 1.0.0 | lens_outputs, window | perturbation_sensitivity |

---

# Appendix B — File Locations

```
prism/
├── core/
│   ├── primitives.py      # Window, EngineVersion, canonical types
│   ├── hashing.py         # compute_parameter_hash (SINGLE SOURCE OF TRUTH)
│   └── errors.py          # Error codes, FailureRecord, failure semantics
│
├── db/
│   ├── schema.sql         # All DuckDB table definitions
│   ├── migrations/        # Schema evolution scripts
│   └── persistence.py    # DuckDB access layer (NO orchestration)
│
├── engines/
│   ├── indicator/         # Indicator-level math engines
│   ├── lens/              # Lens-level math engines (shared math)
│   └── system/            # System-level math engines
│
├── fetchers/
│   ├── base_fetcher.py    # Abstract fetcher (retry, validation, status)
│   ├── fred.py            # FRED source fetcher
│   ├── yahoo.py           # Yahoo source fetcher
│   └── registry.py        # Source → Fetcher class mapping
│
├── runners/
│   ├── fetch_runner.py    # Data acquisition orchestration
│   ├── pipeline_config.py # pipeline.yaml parsing ONLY
│   └── cascade.py         # Failure cascade / stop-continue logic
│
├── entry_points/
│   ├── fetch_domain.py    # CLI / GUI adapter → FetchRunner
│   ├── run_pipeline.py    # CLI adapter → pipeline runner
│   └── __main__.py        # `python -m prism` entry
│
└── observe/
    └── status.py          # stderr-only status emission (START / OK / FAIL)

```

---

# Appendix C — Change Log

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2024-XX-XX | Added: Core Primitives (00), Failure Handling (08), expanded Runners (05), engine versioning, missing data spec, parameter hash spec |
| 2.1 | 2024-XX-XX | Added: Engine Complement Closure (Appendix A), input tensor contracts, PCA canonical outputs with invariants, time semantics for system engines, perturbation complexity rules, numeric stability bounds (00.5), Engine Interface Contract (Appendix D) |
| 1.0 | Original | Initial specification |

---

# Appendix D — Engine Interface Contract

## Purpose
Provide a machine-checkable contract for all PRISM engines.

## Protocol Definition

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class EngineVersion:
    """Immutable engine version identifier."""
    engine_id: str
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.engine_id}@{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible_with(self, other: 'EngineVersion') -> bool:
        """Major version must match for compatibility."""
        return (
            self.engine_id == other.engine_id 
            and self.major == other.major
        )


@runtime_checkable
class MathEngine(Protocol):
    """
    Base protocol for all PRISM math engines.
    
    Engines MUST:
    - Be stateless (no instance variables modified during compute)
    - Be deterministic (same inputs → same outputs)
    - Handle NaN according to documented rules
    - Respect numeric stability bounds
    
    Engines MUST NOT:
    - Access databases
    - Perform I/O
    - Modify input DataFrames
    - Catch and suppress errors silently
    """
    
    engine_id: str
    version: EngineVersion
    
    def compute(
        self,
        df: pd.DataFrame,
        window: 'Window',
        params: dict
    ) -> pd.DataFrame:
        """
        Execute engine computation.
        
        Args:
            df: Input data (shape depends on engine context)
                - Indicator engines: single column + date index
                - Lens engines: n columns + date index
                - System engines: lens outputs structure
            window: Materialized Window object
            params: Engine-specific parameters
        
        Returns:
            DataFrame with columns:
                - All context identifiers (indicator_id, lens_id, etc.)
                - date
                - metric_name
                - value
        
        Raises:
            ComputeError: On unrecoverable computation failure
        """
        ...
    
    def validate_inputs(
        self,
        df: pd.DataFrame,
        window: 'Window',
        params: dict
    ) -> list[str]:
        """
        Validate inputs before computation.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        ...
    
    def get_output_schema(self) -> dict[str, type]:
        """
        Declare output column names and types.
        
        Returns:
            Dict mapping column name → Python type
        """
        ...


@runtime_checkable  
class IndicatorEngine(MathEngine, Protocol):
    """Engine operating on single indicator time series."""
    
    context: str = "indicator"
    input_shape: str = "x(t) ∈ ℝ"


@runtime_checkable
class LensEngine(MathEngine, Protocol):
    """Engine operating on multiple aligned indicators."""
    
    context: str = "lens"
    input_shape: str = "X(t) ∈ ℝⁿ"
    missing_strategy: str  # 'skip' | 'fail' | 'pairwise'


@runtime_checkable
class SystemEngine(MathEngine, Protocol):
    """Engine operating on lens outputs to compute indicator significance."""
    
    context: str = "system"
    input_shape: str = "lens_outputs + indicator_derived"
    requires_pca: bool = True
```

## Example Implementation

```python
class ZScoreEngine:
    """Z-score normalization engine."""
    
    engine_id = "zscore"
    version = EngineVersion("zscore", 1, 0, 0)
    context = "indicator"
    input_shape = "x(t) ∈ ℝ"
    
    def compute(
        self,
        df: pd.DataFrame,
        window: Window,
        params: dict
    ) -> pd.DataFrame:
        series = df['value']
        
        # Respect stability bounds
        mu = series.mean()
        sigma = series.std()
        
        if sigma < EPSILON:
            zscore = pd.Series([float('nan')] * len(series), index=series.index)
        else:
            zscore = (series - mu) / sigma
        
        return pd.DataFrame({
            'indicator_id': df['indicator_id'].iloc[0],
            'date': df.index,
            'window_id': window.window_id,
            'engine_id': self.engine_id,
            'metric_name': 'zscore',
            'value': zscore.values
        })
    
    def validate_inputs(
        self,
        df: pd.DataFrame,
        window: Window,
        params: dict
    ) -> list[str]:
        errors = []
        if 'value' not in df.columns:
            errors.append("Missing 'value' column")
        if df['value'].isna().all():
            errors.append("All values are NaN")
        return errors
    
    def get_output_schema(self) -> dict[str, type]:
        return {
            'indicator_id': str,
            'date': 'date',
            'window_id': str,
            'engine_id': str,
            'metric_name': str,
            'value': float
        }
```

## Registration Pattern

```python
# prism/engines/registry.py

from typing import Type

ENGINE_REGISTRY: dict[str, Type[MathEngine]] = {}

def register_engine(cls: Type[MathEngine]) -> Type[MathEngine]:
    """Decorator to register an engine."""
    if cls.engine_id in ENGINE_REGISTRY:
        raise ValueError(f"Engine {cls.engine_id} already registered")
    ENGINE_REGISTRY[cls.engine_id] = cls
    return cls

def get_engine(engine_id: str) -> MathEngine:
    """Retrieve engine instance by ID."""
    if engine_id not in ENGINE_REGISTRY:
        raise KeyError(f"Unknown engine: {engine_id}. Valid engines: {list(ENGINE_REGISTRY.keys())}")
    return ENGINE_REGISTRY[engine_id]()
```

## Compliance Checklist

Before an engine is added to Appendix A, verify:

- [ ] Implements `MathEngine` protocol
- [ ] `engine_id` is unique and follows naming convention
- [ ] `version` follows semantic versioning
- [ ] `compute()` is deterministic
- [ ] `compute()` respects numeric stability bounds
- [ ] `compute()` handles NaN per documented rules
- [ ] `validate_inputs()` catches malformed inputs
- [ ] `get_output_schema()` matches actual output
- [ ] Unit tests pass with synthetic data
- [ ] Integration test with real data sample
- [ ] Documentation complete (equations, parameters, outputs)
