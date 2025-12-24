# Temporal Resolution Specification
## Market + Economic Domain Analysis

**Version:** 2025-12-08  
**Status:** Active Specification  
**Scope:** Single-domain market/economic analysis (cross-domain deferred)

---

## 1. Purpose

Define how VCF determines time resolution for temporal analysis. The system should:

- Let data determine appropriate resolution
- Match resolution to analysis goal
- Never fabricate granularity that doesn't exist
- Remove device-dependent profiles

---

## 2. Core Principles

### 2.1 Never Upsample

Monthly data does not become daily data. Forward-filling CPI across 22 trading days creates 21 fake datapoints with artificial autocorrelation.

**Rule:** Aggregate DOWN to slower frequencies. Never interpolate UP to faster frequencies.

### 2.2 Resolution Matches the Question

| Question Type | Timescale | Resolution |
|---------------|-----------|------------|
| "Is something breaking right now?" | Days-weeks | Weekly |
| "Where are we in the cycle?" | Months | Monthly |
| "Is this a secular shift?" | Years | Quarterly |

### 2.3 Information Content Determines Floor

If your panel includes quarterly GDP, the slowest meaningful resolution is quarterly (or monthly with forward-fill acknowledged).

---

## 3. Resolution Presets

### 3.1 Tactical (Weekly)

```yaml
frequency: W
lookback: 2Y
use_case: Recent regime shifts, short-term monitoring
when_to_use:
  - Monitoring for emerging stress
  - Post-event analysis
  - Short-term tactical tilts
```

### 3.2 Strategic (Monthly) — DEFAULT

```yaml
frequency: M
lookback: 10Y
use_case: Cycle positioning, portfolio tilting
when_to_use:
  - Standard regime analysis
  - Economic cycle positioning
  - Most FRED data releases monthly
```

### 3.3 Structural (Quarterly)

```yaml
frequency: Q
lookback: 30Y+
use_case: Secular trends, long-term patterns
when_to_use:
  - Multi-decade analysis
  - GDP/productivity cycles
  - Structural regime identification
```

---

## 4. Aggregation Rules

When converting higher-frequency data to lower frequency:

| Data Type | Aggregation Method | Rationale |
|-----------|-------------------|-----------|
| Prices (SPY, ETFs) | `last` | Month-end snapshot |
| Yields/Rates (DGS10) | `mean` | Average conditions |
| Volumes/Flows | `sum` | Total activity |
| Ratios (MA ratio) | `last` | Point-in-time position |
| Volatility (VIX) | `mean` or `max` | Average or peak stress |

### 4.1 Implementation

```python
AGGREGATION_RULES = {
    'price': 'last',
    'yield': 'mean', 
    'rate': 'mean',
    'ratio': 'last',
    'volume': 'sum',
    'flow': 'sum',
    'volatility': 'mean',
    'index': 'last',
    'level': 'last',
    'change': 'sum',
    'default': 'last'
}

def aggregate_series(series, target_freq, data_type='default'):
    method = AGGREGATION_RULES.get(data_type, 'last')
    return series.resample(target_freq).agg(method)
```

---

## 5. Handling Mixed Frequencies

### 5.1 Daily + Monthly (Common Case)

**Scenario:** SPY (daily) + CPI (monthly)

**Resolution:** Monthly

**Process:**
1. Aggregate SPY daily → monthly (last trading day)
2. CPI already monthly → no change
3. Align on month-end dates
4. Run analysis

### 5.2 Daily + Monthly + Quarterly

**Scenario:** SPY (daily) + CPI (monthly) + GDP (quarterly)

**For Monthly Analysis:**
1. Aggregate SPY daily → monthly
2. CPI stays monthly
3. GDP forward-filled across months within quarter
4. Flag: `gdp.forward_filled = True`

**For Quarterly Analysis:**
1. Aggregate SPY daily → quarterly (quarter-end)
2. Aggregate CPI monthly → quarterly (quarter average)
3. GDP stays quarterly

---

## 6. Window and Stride

### 6.1 Window Selection

Window size based on resolution:

| Resolution | Default Window | Rationale |
|------------|----------------|-----------|
| Weekly | 52 weeks (1Y) | Captures annual cycle |
| Monthly | 60 months (5Y) | Captures business cycle |
| Quarterly | 40 quarters (10Y) | Captures long cycles |

**Override:** If dominant cycle detected via spectral analysis:
```
window = 2 × dominant_cycle_length
```

### 6.2 Stride Selection

```
stride = window / 4
```

Ensures 75% overlap for smooth temporal evolution.

---

## 7. API Specification

### 7.1 New Interface

```python
from engine_core.orchestration.temporal_analysis import TemporalAnalyzer

analyzer = TemporalAnalyzer(
    resolution='monthly',      # 'weekly', 'monthly', 'quarterly'
    lookback='10Y',            # or specific date range
    window=None,               # auto-detect if None
    stride=None,               # auto = window/4
)

result = analyzer.run(
    panel=panel_df,
    lenses='all',              # or list of specific lenses
    indicators=None,           # None = all
)
```

### 7.2 Result Structure

```python
TemporalResult(
    scores: pd.DataFrame,          # indicator × lens scores
    rankings: pd.DataFrame,        # indicator × lens rankings  
    temporal_evolution: pd.DataFrame,  # scores over time windows
    consensus: pd.DataFrame,       # cross-lens agreement
    metadata: dict,                # config, timestamps, etc.
    config: TemporalConfig,        # resolution, window, stride used
)
```

---

## 8. Configuration File

```yaml
# temporal_config.yaml

default_resolution: monthly

resolutions:
  weekly:
    frequency: W-FRI          # Week ending Friday
    lookback_default: 2Y
    window_periods: 52
    stride_divisor: 4
    
  monthly:
    frequency: M              # Month end
    lookback_default: 10Y
    window_periods: 60
    stride_divisor: 4
    
  quarterly:
    frequency: Q              # Quarter end
    lookback_default: 30Y
    window_periods: 40
    stride_divisor: 4

aggregation_by_type:
  price: last
  yield: mean
  rate: mean
  ratio: last
  volume: sum
  flow: sum
  volatility: mean
  index: last
  default: last

# Future: cross-domain multi-resolution
multi_resolution:
  enabled: false
  layers: [fast, medium, slow]
```

---

## 9. What This Replaces

### Remove

- `device_profile` parameter (chromebook, standard, powerful)
- Hardcoded window sizes
- Hardcoded stride values
- Upsampling / interpolation to higher frequencies
- Device-based lens filtering

### Keep

- Lens selection (all 14+ lenses available)
- Parallel execution option
- Output formats (CSV, JSON, HTML)

---

## 10. Migration Path

### Phase 1 (Now)
- Implement resolution presets (weekly, monthly, quarterly)
- Implement aggregation rules
- Remove device profiles
- Default to monthly

### Phase 2 (Later)
- Auto-detect dominant cycle for window sizing
- Spectral/wavelet-based window optimization

### Phase 3 (Cross-Domain, Future)
- Multi-resolution parallel engine
- Cross-scale coherence analysis
- Domain-specific resolution detection

---

## 11. Examples

### Example 1: Standard Regime Analysis

```python
analyzer = TemporalAnalyzer(resolution='monthly')
result = analyzer.run(panel)

# Uses:
# - Monthly resolution
# - 5-year window (60 months)
# - 15-month stride
# - All lenses
```

### Example 2: Tactical Monitoring

```python
analyzer = TemporalAnalyzer(
    resolution='weekly',
    lookback='1Y'
)
result = analyzer.run(panel)

# Uses:
# - Weekly resolution  
# - 1-year window (52 weeks)
# - 13-week stride
```

### Example 3: Long-Term Structure

```python
analyzer = TemporalAnalyzer(
    resolution='quarterly',
    lookback='40Y'
)
result = analyzer.run(panel)

# Uses:
# - Quarterly resolution
# - 10-year window (40 quarters)
# - 10-quarter stride
```

---

## 12. Summary

| Principle | Implementation |
|-----------|----------------|
| Data determines resolution | Aggregate to slowest meaningful frequency |
| Question determines choice | Weekly/Monthly/Quarterly presets |
| No fake granularity | Never upsample, only downsample |
| No device dependence | Remove chromebook/standard/powerful |
| Sensible defaults | Monthly for market+economic |

---

*"Match the resolution to the question, not to the fastest data available."*
