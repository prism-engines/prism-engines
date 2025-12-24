# PRISM Engines

Geometric analysis engines for time series data.

## Installation

```bash
pip install prism-engines
```

For plotting support:
```bash
pip install prism-engines[plot]
```

## Quick Start

```python
import prism_engines as prism

# Load CSV and run all engines
results = prism.run("your_data.csv")

# Print analysis report
results.print_report()

# Generate visualization
results.plot()

# Save results
results.save("output/", name="my_analysis")
```

## What It Does

PRISM Engines analyzes multi-variate time series to reveal geometric structure:

| Engine | What It Measures |
|--------|-----------------|
| **Correlation** | Pairwise relationships between series |
| **PCA** | Dimensionality and dominant modes |
| **Hurst** | Memory/persistence in each series |

## Example Output

```
============================================================
PRISM ENGINES ANALYSIS REPORT
============================================================

Data: (100, 5) (5 series)
Range: 2024-01-01 to 2024-04-10

--- CORRELATION ---
  mean_abs_correlation: 0.4523
  max_correlation: {'pair': ('AAPL', 'MSFT'), 'value': 0.892}

--- PCA ---
  explained_variance_ratio: [0.523, 0.218, 0.142, 0.089, 0.028]
  effective_dimension: 3
  global_forcing_metric: 0.523

--- HURST ---
  mean_hurst: 0.612
  persistence_classification: {'AAPL': 'persistent', ...}
============================================================
```

## Detailed Usage

### Loading Data

```python
from prism_engines import load_csv

# Load and validate CSV
df = load_csv("data.csv")
```

Your CSV should have:
- First column: dates (will become index)
- Remaining columns: numeric time series

### Running Specific Engines

```python
from prism_engines import run_engines

# Run only PCA and correlation
results = run_engines(df, engines=["pca", "correlation"])
```

### Accessing Results

```python
# Get specific engine result
pca = results["pca"]

# Access metrics
print(pca.metrics["global_forcing_metric"])
print(pca.metrics["effective_dimension"])

# Check all available metrics
print(pca.metrics.keys())
```

### Available Engines

```python
from prism_engines import list_engines

print(list_engines())
# ['correlation', 'pca', 'hurst']
```

## Metrics Reference

### Correlation Engine
- `correlation_matrix`: NxN correlation matrix
- `mean_abs_correlation`: Average |correlation|
- `max_correlation`: Strongest correlated pair
- `min_correlation`: Weakest correlated pair

### PCA Engine
- `explained_variance_ratio`: Variance per component
- `effective_dimension`: Components for 90% variance
- `global_forcing_metric`: PC1 dominance (higher = more shared movement)
- `pc1_loadings`: How each series loads on PC1

### Hurst Engine
- `hurst_exponents`: H value per series
- `mean_hurst`: Average H
- `persistence_classification`: Category per series
  - `H < 0.4`: anti-persistent (mean-reverting)
  - `0.4 ≤ H ≤ 0.6`: random walk
  - `H > 0.6`: persistent (trending)

## License

MIT
