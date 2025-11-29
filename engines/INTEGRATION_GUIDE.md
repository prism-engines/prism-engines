# Integration Guide: Adding Influence Analysis to VCF-RESEARCH

## Quick Start

### Where to Put the New Files

```
VCF-RESEARCH/
├── code/
│   ├── math/                          # ← Your existing math engine
│   │   └── (your current files)
│   │
│   ├── influence/                     # ← NEW DIRECTORY
│   │   ├── __init__.py
│   │   ├── influence_analysis.py     # Core influence module
│   │   ├── engine_comparison.py      # Engine comparison
│   │   └── ranking.py                # (optional: break out ranking logic)
│   │
│   ├── regime_engine/                 # ← Your Phase I work
│   ├── sector_engine/
│   ├── unified_engine/
│   ├── wavelit_engine/
│   ├── shared/
│   │
│   └── tests/                         # ← NEW DIRECTORY
│       ├── __init__.py
│       ├── test_framework.py
│       ├── test_influence.py
│       └── test_integration.py
```

---

## Step-by-Step Integration

### 1. Create Directory Structure

```bash
cd /path/to/VCF-RESEARCH/code
mkdir -p influence
mkdir -p tests
```

### 2. Move New Files

```bash
# Copy the new modules
cp influence_analysis.py influence/
cp engine_comparison.py influence/
cp test_framework.py tests/

# Create __init__.py files
touch influence/__init__.py
touch tests/__init__.py
```

### 3. Update `influence/__init__.py`

```python
"""
VCF Influence Analysis Module

Provides tools for determining which indicators drive system dynamics
"""

from .influence_analysis import (
    InfluenceRanking,
    PairwiseCoherence,
    TemporalInfluenceTracking,
    InfluenceAnalysisEngine,
    ComparativeInfluenceAnalysis
)

from .engine_comparison import (
    EngineComparator,
    EngineEvaluator
)

__all__ = [
    'InfluenceRanking',
    'PairwiseCoherence',
    'TemporalInfluenceTracking',
    'InfluenceAnalysisEngine',
    'ComparativeInfluenceAnalysis',
    'EngineComparator',
    'EngineEvaluator'
]
```

---

## Connecting to Your Existing Code

### Option A: Standalone Usage (Recommended for Pilot)

Keep influence analysis separate, load data directly:

```python
# pilot_analysis.py

import pandas as pd
from code.influence.influence_analysis import InfluenceAnalysisEngine

# Load your pilot data
pilot_data = pd.read_csv('data/data_normalized/pilot_4_indicators.csv', 
                         index_col=0, parse_dates=True)

# Run influence analysis
engine = InfluenceAnalysisEngine(pilot_data)
report = engine.full_influence_report()

# Save results
report['top_influencers'].to_csv('outputs/pilot_top_influencers.csv')
drivers = engine.detect_what_drives_dynamics()
drivers.to_csv('outputs/pilot_drivers_over_time.csv')
```

### Option B: Integrate with Existing Engines

Modify your existing engines to also compute influence:

```python
# In code/unified_engine/unified_engine.py

from code.influence.influence_analysis import InfluenceRanking

class UnifiedEngine:
    def __init__(self, panel_df):
        self.panel = panel_df
        self.influence_ranker = InfluenceRanking(panel_df)  # ADD THIS
        # ... your existing code
    
    def run_analysis(self):
        # Your existing analysis
        regime_results = self.detect_regime()
        vector_results = self.vector_analysis()
        
        # ADD: Influence analysis
        influence_results = self.influence_ranker.composite_influence_score()
        top_influencers = self.influence_ranker.top_influencers(n_top=10)
        
        return {
            'regime': regime_results,
            'vector': vector_results,
            'influence': influence_results,  # NEW
            'top_influencers': top_influencers  # NEW
        }
```

### Option C: Create New Standalone Script

Most flexible for comparing approaches:

```python
# scripts/compare_engines.py

import sys
sys.path.append('..')

import pandas as pd
from code.influence.influence_analysis import InfluenceAnalysisEngine
from code.influence.engine_comparison import EngineComparator

# Load data
panel = pd.read_csv('../data/data_normalized/all_56_indicators.csv',
                    index_col=0, parse_dates=True)

# Define your engine
def vcf_vector_engine(panel_df, **kwargs):
    """Your current VCF approach"""
    from code.math.vcf_math_models import VCFMathEngine
    
    math_engine = VCFMathEngine(panel_df)
    results = math_engine.full_vector_analysis()
    
    # Convert to influence format
    # (You'll need to map your results to the expected format)
    return {
        'top_influencers': extract_top_influencers(results),
        'temporal_drivers': extract_temporal_drivers(results)
    }

# Define influence-based engine
def vcf_influence_engine(panel_df, **kwargs):
    """New influence-based approach"""
    engine = InfluenceAnalysisEngine(panel_df)
    return {
        'top_influencers': engine.ranker.top_influencers(n_top=10),
        'temporal_drivers': engine.detect_what_drives_dynamics()
    }

# Compare
comparator = EngineComparator(panel)
comparator.register_engine('VCF_Vector', vcf_vector_engine)
comparator.register_engine('VCF_Influence', vcf_influence_engine)

report = comparator.generate_comparison_report(
    output_file='../outputs/vcf_engine_comparison.json'
)

print("Comparison complete! See outputs/vcf_engine_comparison.json")
```

---

## Using with Your 56 Indicators

### Step 1: Prepare Data

```python
# scripts/prepare_56_indicators.py

import pandas as pd
import glob

# Load all your indicators
indicator_files = glob.glob('data/raw/*.csv')

all_indicators = {}
for file in indicator_files:
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    indicator_name = file.split('/')[-1].replace('.csv', '')
    all_indicators[indicator_name] = df

# Combine into panel
panel = pd.DataFrame(all_indicators)

# Normalize (you might already have this)
panel_normalized = (panel - panel.mean()) / panel.std()

# Save
panel_normalized.to_csv('data/data_normalized/all_56_indicators.csv')
print(f"Prepared {len(panel_normalized.columns)} indicators")
```

### Step 2: Run Influence Analysis

```python
# scripts/analyze_56_indicators.py

import pandas as pd
from code.influence.influence_analysis import InfluenceAnalysisEngine
from code.tests.test_framework import VCFTestSuite

# Load data
panel = pd.read_csv('data/data_normalized/all_56_indicators.csv',
                    index_col=0, parse_dates=True)

# Validate data first
print("Running data validation...")
test_suite = VCFTestSuite(panel)
validation_results = test_suite.run_all_tests()

if not all(r.get('passed', False) for r in validation_results.values()):
    print("WARNING: Some validation tests failed")
    print("Review test_results.json before proceeding")

# Run influence analysis
print("\nRunning influence analysis on 56 indicators...")
engine = InfluenceAnalysisEngine(panel)

# Get results
print("Computing influence scores...")
influence_scores = engine.ranker.composite_influence_score(window=12)
influence_scores.to_csv('outputs/influence_scores_all_indicators.csv')

print("Identifying top influencers over time...")
drivers = engine.detect_what_drives_dynamics(window=12)
drivers.to_csv('outputs/primary_drivers_over_time.csv')

print("Computing coherent clusters...")
clusters = engine.pairwise.coherent_clusters(threshold=0.7)
with open('outputs/coherent_clusters.txt', 'w') as f:
    for i, cluster in enumerate(clusters):
        f.write(f"Cluster {i+1}: {', '.join(cluster)}\n")

print("\nAnalysis complete! Results saved to outputs/")
```

### Step 3: Visualize Results

```python
# scripts/visualize_influence.py

import pandas as pd
import matplotlib.pyplot as plt

# Load results
drivers = pd.read_csv('outputs/primary_drivers_over_time.csv', 
                      index_col=0, parse_dates=True)

# Plot primary driver over time
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top driver identity
ax = axes[0]
# Create categorical plot showing which indicator was dominant
unique_drivers = drivers['primary_driver'].unique()
driver_codes = {d: i for i, d in enumerate(unique_drivers)}
drivers['driver_code'] = drivers['primary_driver'].map(driver_codes)

ax.scatter(drivers['date'], drivers['driver_code'], 
          c=drivers['primary_magnitude'], cmap='viridis', s=50)
ax.set_yticks(range(len(unique_drivers)))
ax.set_yticklabels(unique_drivers)
ax.set_title('Primary Driver Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Indicator')

# Influence concentration
ax = axes[1]
ax.plot(drivers['date'], drivers['concentration'])
ax.axhline(y=0.5, color='r', linestyle='--', label='High concentration threshold')
ax.set_title('Influence Concentration (How dominant is top driver?)')
ax.set_xlabel('Date')
ax.set_ylabel('Concentration')
ax.legend()

plt.tight_layout()
plt.savefig('outputs/influence_evolution.png', dpi=300)
print("Visualization saved to outputs/influence_evolution.png")
```

---

## Comparing with Gemini's Results

### Step 1: Format Gemini's Output

```python
# scripts/format_gemini_results.py

import pandas as pd

# Load Gemini's results (format will depend on what Gemini gave you)
gemini_raw = pd.read_csv('data/gemini_results.csv')

# Convert to standard format expected by EngineComparator
# This is the format: DataFrame with columns ['indicator', 'influence_score', 'rank']

gemini_formatted = pd.DataFrame({
    'indicator': gemini_raw['indicator_name'],
    'influence_score': gemini_raw['importance_score'],  # or whatever Gemini calls it
    'rank': range(1, len(gemini_raw) + 1)
})

gemini_formatted.to_csv('data/gemini_results_formatted.csv', index=False)
```

### Step 2: Run Comparison

```python
# scripts/compare_with_gemini.py

import pandas as pd
from code.influence.engine_comparison import EngineComparator
from code.influence.influence_analysis import InfluenceAnalysisEngine

# Load data
panel = pd.read_csv('data/data_normalized/all_56_indicators.csv',
                    index_col=0, parse_dates=True)

# Your engine
def your_engine(panel_df, **kwargs):
    engine = InfluenceAnalysisEngine(panel_df)
    return {
        'top_influencers': engine.ranker.top_influencers(n_top=20)
    }

# Gemini's engine (wrapper for pre-computed results)
def gemini_engine(panel_df, **kwargs):
    # Load Gemini's pre-computed results
    gemini_results = pd.read_csv('data/gemini_results_formatted.csv')
    return {
        'top_influencers': gemini_results
    }

# Compare
comparator = EngineComparator(panel)
comparator.register_engine('Your_VCF', your_engine)
comparator.register_engine('Gemini', gemini_engine)

# Generate comprehensive comparison
report = comparator.generate_comparison_report(
    output_file='outputs/vcf_vs_gemini_comparison.json'
)

# Print summary
print("=" * 70)
print("VCF vs GEMINI COMPARISON")
print("=" * 70)
print(f"\nCorrelation: {report['agreement_analysis']['correlation_matrix']}")
print(f"\nHigh agreement indicators:")
for ind in report['agreement_analysis']['high_agreement_indicators'][:10]:
    print(f"  - {ind}")
print(f"\nBiggest disagreements:")
disagreements = report['top_disagreements']['indicator']
for ind in list(disagreements)[:5]:
    print(f"  - {ind}")
```

---

## Running the Pilot (4 Indicators)

Quick script for your pilot run:

```python
# scripts/pilot_analysis.py

import pandas as pd
from code.influence.influence_analysis import InfluenceAnalysisEngine

# Load pilot data
# Assuming you have: S&P 500 MA diff, 10Y yield, DXY, AGG
pilot = pd.read_csv('data/pilot_data.csv', index_col=0, parse_dates=True)

print(f"Loaded pilot data: {pilot.shape}")
print(f"Indicators: {list(pilot.columns)}")
print(f"Date range: {pilot.index[0]} to {pilot.index[-1]}")

# Run influence analysis
engine = InfluenceAnalysisEngine(pilot)

# Current snapshot
report = engine.full_influence_report()

print("\n" + "=" * 70)
print("PILOT RESULTS: TOP INFLUENCERS")
print("=" * 70)
print(report['top_influencers'].to_string(index=False))

print("\n" + "=" * 70)
print("MOST COHERENT PAIRS")
print("=" * 70)
print(report['coherent_pairs'].to_string(index=False))

# Temporal evolution
drivers = engine.detect_what_drives_dynamics()

print("\n" + "=" * 70)
print("DRIVERS OVER TIME (Last 12 periods)")
print("=" * 70)
print(drivers.tail(12)[['date', 'primary_driver', 'primary_magnitude', 
                        'secondary_driver', 'concentration']].to_string(index=False))

# Save
drivers.to_csv('outputs/pilot_drivers.csv')
report['top_influencers'].to_csv('outputs/pilot_top_influencers.csv')

print("\nResults saved to outputs/")
```

---

## Testing Before Full Run

Always test first:

```python
# scripts/run_tests.py

from code.tests.test_framework import VCFTestSuite
import pandas as pd

# Test on pilot data first
pilot = pd.read_csv('data/pilot_data.csv', index_col=0, parse_dates=True)

print("Testing pilot data...")
test_suite = VCFTestSuite(pilot)
results = test_suite.run_all_tests()

if results['data_validation']['missing_values']['all_passed']:
    print("\n✓ Pilot data validated")
    print("Ready to run on 56 indicators")
else:
    print("\n✗ Issues found in pilot data")
    print("Fix issues before proceeding")
```

---

## Next Steps

1. **Run pilot** with 4 indicators using `scripts/pilot_analysis.py`
2. **Compare with Gemini's pilot results** using `scripts/compare_with_gemini.py`
3. **Validate** any disagreements - where do you disagree and why?
4. **Scale to 56 indicators** once pilot is validated
5. **Track influence over historical periods** (1970s, 2008, 2020, etc.)

---

## File Manifest

Place these files in your repo:

```
VCF-RESEARCH/
├── code/
│   ├── influence/
│   │   ├── __init__.py                     # Package init
│   │   ├── influence_analysis.py           # Core influence module
│   │   └── engine_comparison.py            # Engine comparison
│   └── tests/
│       ├── __init__.py
│       └── test_framework.py               # Test suite
│
├── scripts/                                 # NEW: Analysis scripts
│   ├── prepare_56_indicators.py
│   ├── analyze_56_indicators.py
│   ├── visualize_influence.py
│   ├── compare_with_gemini.py
│   ├── pilot_analysis.py
│   └── run_tests.py
│
├── outputs/                                 # Results go here
│   ├── influence_scores_all_indicators.csv
│   ├── primary_drivers_over_time.csv
│   ├── coherent_clusters.txt
│   ├── vcf_vs_gemini_comparison.json
│   └── influence_evolution.png
│
└── docs/
    └── README_INFLUENCE_FRAMEWORK.md        # Main documentation
```

---

## Quick Reference

### Import Patterns

```python
# For influence analysis
from code.influence.influence_analysis import InfluenceAnalysisEngine

# For engine comparison
from code.influence.engine_comparison import EngineComparator

# For testing
from code.tests.test_framework import VCFTestSuite
```

### Common Operations

```python
# Load data
panel = pd.read_csv('data/data_normalized/indicators.csv', index_col=0, parse_dates=True)

# Run influence analysis
engine = InfluenceAnalysisEngine(panel)

# Get top 10 influencers NOW
top = engine.ranker.top_influencers(n_top=10)

# Get drivers over time
drivers = engine.detect_what_drives_dynamics()

# Full report
report = engine.full_influence_report()

# Compare engines
comparator = EngineComparator(panel)
comparator.register_engine('Engine1', func1)
comparator.register_engine('Engine2', func2)
comparison = comparator.generate_comparison_report()
```

---

Ready to integrate! Start with the pilot, then scale to 56 indicators.
