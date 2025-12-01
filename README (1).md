# PRISM Engine

**Pattern Recognition through Integrated Signal Methods**

---

## How to Use (The Simple Version)

```bash
# 1. Put your data in data/clean/
# 2. Run:
python run.py

# That's it. Read results/latest/summary.txt
```

---

## What It Does

PRISM takes your time series data and:

1. **Groups** indicators into categories (monetary, employment, housing, etc.)
2. **Weights** them algorithmically (no human bias)
3. **Runs ALL lenses** (correlation, PCA, wavelets, etc.)
4. **Calculates coherence** (do the lenses agree?)
5. **Tells you what matters** (ranked indicators + warnings)

---

## The Key Insight

> **It's not what any one lens says. It's when they ALL agree.**

High coherence = All lenses see the same thing  
Low coherence = Lenses disagree (often healthy - diversity)

A sudden SPIKE in coherence? That's the signal. That's the herd running for the exit.

---

## Commands

```bash
# Full analysis (default)
python run.py

# Quick mode (3 windows, faster)
python run.py --quick

# Focus on one category
python run.py --focus monetary

# Only recent data
python run.py --since 2020

# Generate HTML report
python run.py --report

# Override weighting method
python run.py --weights equal    # Equal weights
python run.py --weights pca      # PCA loadings
python run.py --weights variance # Inverse variance
python run.py --weights auto     # Let PRISM decide (default)
```

---

## Folder Structure

```
prism-engine/
│
├── run.py              ← THE ONLY FILE YOU RUN
│
├── data/
│   ├── clean/          ← Put your data here (CSV)
│   └── indicators.yaml ← (optional) Custom categories
│
├── engine/             ← The machinery (you don't touch this)
│   ├── consolidate.py
│   ├── temporal.py
│   ├── coherence.py
│   └── report.py
│
├── results/
│   ├── latest/         ← Most recent run
│   │   ├── summary.txt ← READ THIS
│   │   ├── coherence.csv
│   │   └── rankings.csv
│   │
│   └── archive/        ← Past runs (timestamped)
│
└── reports/
    └── latest_report.html  ← Visual report (--report flag)
```

---

## Data Format

Your input CSV should look like:

```
Date,M2SL,PAYEMS,VIX,SP500,...
2020-01-01,15000,150000,12.5,3200,...
2020-01-02,15010,150100,13.0,3210,...
```

- First column: Date (will be parsed)
- Other columns: Your indicators (any names)
- Missing values: OK, PRISM handles them

---

## Weighting Methods

PRISM removes human judgment. Here's how:

| Method | Logic | Use When |
|--------|-------|----------|
| `equal` | All indicators same weight | Baseline comparison |
| `variance` | Stable = more weight | Some indicators noisy |
| `entropy` | High info = more weight | Default good choice |
| `pca` | Main signal drivers | Want to find what matters |
| `auto` | PRISM picks best | Recommended |

---

## Understanding Results

### summary.txt

```
COHERENCE INDEX
  Overall: 0.523
  Trend: rising
  Interpretation: MODERATE - Reasonable agreement

TOP INDICATORS
  1. M2SL                    (2.341)
  2. WALCL                   (2.892)
  ...
```

### What the numbers mean

- **Coherence 0.0-0.3**: Lenses disagree (diverse, often healthy)
- **Coherence 0.3-0.5**: Some alignment (normal)
- **Coherence 0.5-0.7**: Notable agreement (pay attention)
- **Coherence 0.7+**: Strong alignment (investigate!)

### Warning flags

If you see:
- "Coherence spike" → Something forced alignment
- "High coherence" → Potential regime shift
- "Very low coherence" → Check data quality

---

## Adding Your Own Lenses

Edit `engine/temporal.py`:

```python
@lens("my_custom_lens")
def lens_my_custom(df: pd.DataFrame, config: dict) -> pd.Series:
    # Your logic here
    # Return: Series with indicator names as index, importance as values
    return rankings.sort_values(ascending=False)
```

That's it. PRISM will automatically include it.

---

## Philosophy

PRISM is **type-agnostic** and **time-agnostic**:

- Works on any time series (markets, climate, chemistry, biology)
- Works at any time scale (milliseconds to megayears)
- The math doesn't know what a "stock" is - it just sees patterns

The same coherence signal that warns of a market crash could warn of:
- Ecosystem collapse
- Reactor runaway  
- Political instability
- Phase transition

---

## Questions?

Read the code. It's commented for humans, not AI systems.

Or run `python run.py --help`
