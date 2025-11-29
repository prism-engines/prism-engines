# Mathematical Lens Framework - Usage Guide

## What This Is

**Not**: A market timing system, regime classifier, or trading signal  
**Is**: Pure mathematical exploration - applying different analytical methods to the same data and seeing what each reveals

## Core Philosophy

> "If I look at the same data through different mathematical lenses, what different patterns emerge? Where do the methods agree? Disagree? What does each uniquely see?"

**No imposed structure. No regime labels. Just mathematical curiosity.**

---

## Quick Start

### 1. Load Your Data

```python
import pandas as pd
from mathematical_lenses import run_full_lens_analysis

# Your indicator panel (any number of indicators, any timeframe)
panel = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)

# Just needs to be: rows = time, columns = indicators
print(panel.shape)  # e.g., (200, 9) = 200 months, 9 indicators
```

### 2. Run All Lenses

```python
# One command to run all mathematical analyses
comparator = run_full_lens_analysis(
    panel, 
    date_to_analyze='2020-03-09'  # Optional: focus on specific date
)

# This runs:
# - Vector Magnitude
# - PCA
# - Granger Causality  
# - Dynamic Mode Decomposition
# - Rolling Influence
# - Mutual Information
```

### 3. See What Each Lens Found

The output automatically shows:

**Lens Agreement Matrix**: How much do different methods agree?
```
           Magnitude   PCA  Granger   DMD  Influence  MutualInfo
Magnitude      1.00  0.82     0.45  0.71       0.89        0.76
PCA            0.82  1.00     0.51  0.79       0.85        0.83
Granger        0.45  0.51     1.00  0.39       0.48        0.55
...
```

**Consensus Indicators**: Which indicators do MOST lenses rank highly?
```
         Magnitude   PCA  Granger   DMD  Influence  mean_score  n_lenses
VIX_US        0.85  0.92     0.95  0.78       0.88        0.88         5
SPY_US        0.72  0.81     0.42  0.82       0.79        0.71         5
CPI_US        0.58  0.65     0.38  0.61       0.64        0.57         5
```

**Unique Insights**: What does each lens see that others don't?
```
Magnitude: []
PCA: ['M2_US']
Granger: ['VIX_US', 'DGS10_US']  ← Granger sees VIX as uniquely causal!
DMD: ['GDP_US']
Influence: []
MutualInfo: ['T10Y2Y_US']
```

---

## What Each Lens Reveals

### Lens 1: Vector Magnitude
**Question**: "How much is happening overall?"  
**Method**: L2 norm across all indicators  
**Insight**: Overall system volatility/activity  
**Unique**: Treats all indicators equally, pure Euclidean distance

### Lens 2: PCA
**Question**: "What are the natural factors?"  
**Method**: Linear dimensionality reduction  
**Insight**: Discovers underlying structure (e.g., "3 factors explain 90% of variance")  
**Unique**: Shows if data is actually high or low dimensional

### Lens 3: Granger Causality
**Question**: "Which indicators predict/cause others?"  
**Method**: Temporal lead/lag regression  
**Insight**: Causal structure (who drives whom)  
**Unique**: Only method that reveals directional causality

### Lens 4: Dynamic Mode Decomposition (DMD)
**Question**: "What are the dominant oscillatory patterns?"  
**Method**: Eigendecomposition of dynamics  
**Insight**: Frequencies and growth/decay modes  
**Unique**: Reveals temporal periodicities (e.g., "12-month cycle")

### Lens 5: Rolling Influence
**Question**: "Which indicators are most active RIGHT NOW?"  
**Method**: Time-varying volatility × current deviation  
**Insight**: Shows influence shifts over time  
**Unique**: Only time-varying method - catches regime transitions

### Lens 6: Mutual Information
**Question**: "Which indicators share the most information?"  
**Method**: Information-theoretic dependence  
**Insight**: Non-linear relationships, information redundancy  
**Unique**: Catches non-linear dependencies linear methods miss

---

## Example: COVID Crash Through Different Lenses

```python
# Focus on March 2020
covid_comparison = comparator.compare_at_date(
    pd.Timestamp('2020-03-09'),
    n_top=5
)
print(covid_comparison)
```

**Hypothetical Output**:
```
            Magnitude  PCA  Granger  DMD  Influence  MutualInfo
VIX_US           0.95  0.92     0.98  0.78       0.94        0.82
SPY_US           0.88  0.89     0.42  0.85       0.91        0.79
DGS10_US         0.45  0.52     0.67  0.41       0.48        0.51
M2_US            0.32  0.28     0.15  0.52       0.29        0.33
CPI_US           0.18  0.22     0.08  0.19       0.21        0.25
```

**What This Tells You**:

✅ **Strong Consensus**: VIX and SPY - ALL methods agree these dominated  
✅ **Granger Unique Insight**: VIX scored 0.98 in causality (was the DRIVER, not just correlated)  
✅ **DMD Unique Insight**: M2 scored higher (0.52) - suggests liquidity pulse mattered in oscillatory dynamics  
✅ **Disagreement**: Granger ranks DGS10 high (0.67) but others don't - means rates PREDICTED but didn't PARTICIPATE in magnitude

**Interpretation**: All methods see risk/equity dominance, but Granger uniquely identifies VIX as CAUSAL driver, while DMD uniquely sees liquidity's oscillatory role.

---

## Advanced Usage

### Add Your Own Custom Lens

```python
class MyCustomLens:
    def __init__(self, name: str = "MyLens"):
        self.name = name
    
    def analyze(self, panel: pd.DataFrame) -> Dict:
        # Your custom mathematical analysis
        # Return a dict with your results
        return {
            'your_metric': your_calculation,
            'method': 'Description of your method'
        }
    
    def top_indicators(self, result: Dict, date, n: int = 5):
        # Return list of (indicator, score) tuples
        return [('indicator_name', score), ...]

# Add it
comparator = LensComparator(panel)
comparator.add_lens(MyCustomLens())
comparator.add_lens(MagnitudeLens())  # etc
comparator.run_all()
```

### Compare Over Time

```python
# See how agreement changes
dates_to_check = pd.date_range('2020-01-01', '2020-12-31', freq='M')

agreement_over_time = []
for date in dates_to_check:
    comparison = comparator.compare_at_date(date, n_top=3)
    # Calculate agreement (e.g., overlap in top 3)
    agreement_over_time.append(...)

# Plot: Does lens agreement drop during crises?
```

### Extract Specific Lens Results

```python
# Get detailed results from one lens
pca_results = comparator.results['PCA']

print(f"Number of components: {pca_results['n_components']}")
print(f"Explained variance: {pca_results['explained_variance']}")
print(f"Loadings:\n{pca_results['loadings']}")

# Get Granger causality matrix
granger_results = comparator.results['Granger']
print(f"Causality matrix:\n{granger_results['causality_matrix']}")
print(f"Top drivers: {granger_results['out_degree'].head()}")
```

---

## Interesting Questions You Can Explore

### 1. "Do all methods agree on dimensionality?"
```python
# PCA: How many components?
n_pca = comparator.results['PCA']['n_components']

# DMD: How many significant modes?
n_dmd = comparator.results['DMD']['n_modes']

# Mutual Info: Effective dimensionality from entropy
# (add custom calculation)

print(f"PCA sees {n_pca} dimensions")
print(f"DMD sees {n_dmd} modes")
# Do they agree? What does disagreement mean?
```

### 2. "Are market 'leaders' also causal drivers?"
```python
# Influence lens: Who has high influence?
influence_top = comparator.results['Influence']['influence_scores'].mean().sort_values(ascending=False).head(5)

# Granger lens: Who causes others?
granger_top = comparator.results['Granger']['out_degree'].head(5)

# Compare
print("High influence:", list(influence_top.index))
print("High causality:", list(granger_top.index))
# Different? Means high-variance ≠ causal!
```

### 3. "Which indicators are redundant?"
```python
# Mutual Information: High MI = redundant
mi_matrix = comparator.results['MutualInfo']['mi_matrix']

# Find pairs with highest MI
redundant_pairs = []
for i in mi_matrix.index:
    for j in mi_matrix.columns:
        if i < j:  # Avoid duplicates
            redundant_pairs.append((i, j, mi_matrix.loc[i, j]))

redundant_pairs.sort(key=lambda x: x[2], reverse=True)
print("Most redundant pairs:", redundant_pairs[:5])
# Maybe CPI and PPI have MI = 0.9 → choose one!
```

### 4. "Does lens agreement predict stability?"
```python
# Hypothesis: When lenses agree → stable period
#            When lenses disagree → transition period

agreement_scores = []
for date in panel.index[50:]:  # Skip first 50 for rolling calcs
    comparison = comparator.compare_at_date(date, n_top=3)
    # Calculate std across lenses for top indicators
    agreement_score = comparison.std(axis=1).mean()
    agreement_scores.append(agreement_score)

# Plot: Low agreement → crisis periods?
```

---

## What NOT to Do

❌ Don't use this for trading signals  
❌ Don't pick one "best" lens and ignore others  
❌ Don't force interpretations onto disagreements  
❌ Don't expect all lenses to agree (disagreement is interesting!)

## What TO Do

✅ Be curious about where methods agree (robust findings)  
✅ Be curious about where methods disagree (reveals different aspects)  
✅ Look for what each lens uniquely sees  
✅ Let patterns emerge rather than forcing them  
✅ Share interesting findings with the quant community

---

## Output Files

The framework can save results for later analysis:

```python
# Save comparison for a specific date
comparison = comparator.compare_at_date('2020-03-09')
comparison.to_csv('covid_lens_comparison.csv')

# Save agreement matrix
agreement = comparator.agreement_matrix()
agreement.to_csv('lens_agreement_matrix.csv')

# Save consensus rankings
consensus = comparator.consensus_indicators(n_top=20)
consensus.to_csv('consensus_important_indicators.csv')
```

---

## For Your Actual Data

```python
# With your 9 current indicators
panel = pd.read_csv('normalized_panel_monthly.csv', index_col='date', parse_dates=True)
comparator = run_full_lens_analysis(panel)

# Key dates to examine
key_dates = [
    '2008-09-15',  # Lehman
    '2020-03-09',  # COVID crash
    '2025-11-26',  # Current
]

for date in key_dates:
    print(f"\n{'='*70}")
    print(f"Analysis for {date}")
    print(f"{'='*70}")
    comparison = comparator.compare_at_date(pd.Timestamp(date), n_top=5)
    print(comparison)
```

---

## The Beautiful Thing

**Different mathematical perspectives on the same reality.**

Like different instruments in an orchestra - each one hears something slightly different in the same piece of music. The violin hears melody, the drums hear rhythm, the bass hears harmony.

None is "right" or "wrong" - they're all valid perspectives.

**The interesting questions are:**
- Where do they all agree? (robust signal)
- Where do they disagree? (reveals complexity)
- What does each uniquely hear? (method-specific insights)

This is pure mathematical exploration. Enjoy the journey!
