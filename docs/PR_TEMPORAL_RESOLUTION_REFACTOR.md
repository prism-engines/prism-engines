# PR: Temporal Resolution Refactor
## Remove Device Profiles, Implement Resolution Presets

**PR Type:** Refactor  
**Priority:** High  
**Breaking Changes:** Yes (removes device_profile parameter)

---

## Summary

Replace device-dependent temporal configuration (chromebook/standard/powerful) with analysis-driven resolution presets (weekly/monthly/quarterly). The system should determine temporal parameters based on the analytical question, not hardware assumptions.

---

## Files to Modify

### 1. `engine_core/orchestration/temporal_runner.py`

**Remove:**
```python
# DELETE these device profile references
DEVICE_PROFILES = {
    'chromebook': {...},
    'standard': {...},
    'powerful': {...}
}

def get_device_profile(profile_name):
    ...
```

**Add:**
```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal

RESOLUTION_PRESETS = {
    'weekly': {
        'frequency': 'W-FRI',
        'window_periods': 52,
        'stride_divisor': 4,
        'lookback_default': '2Y',
    },
    'monthly': {
        'frequency': 'M',
        'window_periods': 60,
        'stride_divisor': 4,
        'lookback_default': '10Y',
    },
    'quarterly': {
        'frequency': 'Q',
        'window_periods': 40,
        'stride_divisor': 4,
        'lookback_default': '30Y',
    }
}

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

@dataclass
class TemporalConfig:
    resolution: Literal['weekly', 'monthly', 'quarterly'] = 'monthly'
    frequency: str = 'M'
    window_periods: int = 60
    stride_periods: int = 15
    lookback: str = '10Y'
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    @classmethod
    def from_resolution(cls, resolution: str = 'monthly', **overrides):
        preset = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS['monthly'])
        config = cls(
            resolution=resolution,
            frequency=preset['frequency'],
            window_periods=preset['window_periods'],
            stride_periods=preset['window_periods'] // preset['stride_divisor'],
            lookback=preset['lookback_default'],
        )
        # Apply any overrides
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        return config
```

**Modify `TemporalRunner.__init__`:**
```python
# OLD
def __init__(self, device_profile='standard', ...):
    self.profile = get_device_profile(device_profile)
    ...

# NEW  
def __init__(
    self, 
    resolution: str = 'monthly',
    lookback: Optional[str] = None,
    window: Optional[int] = None,
    stride: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
):
    self.config = TemporalConfig.from_resolution(
        resolution=resolution,
        lookback=lookback or None,
        window_periods=window,
        stride_periods=stride,
        start_date=start_date,
        end_date=end_date,
    )
    ...
```

---

### 2. `engine_core/orchestration/temporal_aggregator.py`

**Add aggregation helper:**
```python
import pandas as pd
from typing import Dict, Optional

def get_indicator_data_type(indicator_name: str, registry: Dict) -> str:
    """Look up data type from registry, default to 'default'."""
    if indicator_name in registry:
        return registry[indicator_name].get('data_type', 'default')
    
    # Heuristic fallbacks
    name_lower = indicator_name.lower()
    if any(x in name_lower for x in ['spy', 'qqq', 'etf', 'close', 'price']):
        return 'price'
    if any(x in name_lower for x in ['dgs', 'yield', 'rate']):
        return 'yield'
    if any(x in name_lower for x in ['volume', 'vol_']):
        return 'volume'
    if any(x in name_lower for x in ['vix', 'volatility']):
        return 'volatility'
    if any(x in name_lower for x in ['ratio', 'ma_']):
        return 'ratio'
    return 'default'


def aggregate_panel_to_frequency(
    panel: pd.DataFrame,
    target_frequency: str,
    registry: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Aggregate panel to target frequency using appropriate methods per column.
    
    Args:
        panel: DataFrame with DatetimeIndex
        target_frequency: pandas frequency string ('W-FRI', 'M', 'Q')
        registry: Optional indicator registry for data type lookup
    
    Returns:
        Aggregated DataFrame at target frequency
    """
    registry = registry or {}
    
    aggregated = {}
    for col in panel.columns:
        data_type = get_indicator_data_type(col, registry)
        method = AGGREGATION_RULES.get(data_type, 'last')
        
        try:
            aggregated[col] = panel[col].resample(target_frequency).agg(method)
        except Exception as e:
            # Fallback to last if aggregation fails
            aggregated[col] = panel[col].resample(target_frequency).last()
    
    result = pd.DataFrame(aggregated)
    result = result.dropna(how='all')  # Remove empty rows
    return result


def detect_native_frequency(series: pd.Series) -> str:
    """Detect the native frequency of a series."""
    if len(series) < 2:
        return 'unknown'
    
    # Calculate median gap between observations
    gaps = series.dropna().index.to_series().diff().dropna()
    if len(gaps) == 0:
        return 'unknown'
    
    median_gap = gaps.median()
    
    if median_gap <= pd.Timedelta(days=1):
        return 'daily'
    elif median_gap <= pd.Timedelta(days=7):
        return 'weekly'
    elif median_gap <= pd.Timedelta(days=32):
        return 'monthly'
    elif median_gap <= pd.Timedelta(days=95):
        return 'quarterly'
    else:
        return 'annual'
```

---

### 3. `start/temporal_runner.py`

**Update CLI interface:**
```python
# OLD
import argparse
parser.add_argument('--profile', choices=['chromebook', 'standard', 'powerful'], 
                    default='standard')

# NEW
import argparse

parser = argparse.ArgumentParser(description='Run temporal analysis')
parser.add_argument(
    '--resolution', '-r',
    choices=['weekly', 'monthly', 'quarterly'],
    default='monthly',
    help='Analysis resolution (default: monthly)'
)
parser.add_argument(
    '--lookback', '-l',
    type=str,
    default=None,
    help='Lookback period, e.g., "5Y", "10Y", "2020-01-01" (default: auto)'
)
parser.add_argument(
    '--window', '-w',
    type=int,
    default=None,
    help='Window size in periods (default: auto based on resolution)'
)
parser.add_argument(
    '--lenses',
    type=str,
    default='all',
    help='Comma-separated lens names or "all" (default: all)'
)
parser.add_argument(
    '--output', '-o',
    type=str,
    default='output/latest',
    help='Output directory'
)

def main():
    args = parser.parse_args()
    
    runner = TemporalRunner(
        resolution=args.resolution,
        lookback=args.lookback,
        window=args.window,
    )
    
    lenses = None if args.lenses == 'all' else args.lenses.split(',')
    
    result = runner.run_all_lenses(lenses=lenses)
    runner.save_results(result, output_dir=args.output)
    
    print(f"Analysis complete. Results saved to {args.output}/")
    print(f"Resolution: {args.resolution}")
    print(f"Window: {runner.config.window_periods} periods")
    print(f"Stride: {runner.config.stride_periods} periods")
```

---

### 4. `config.yaml` (or create `temporal_config.yaml`)

**Add:**
```yaml
# temporal_config.yaml

temporal:
  default_resolution: monthly
  
  resolutions:
    weekly:
      frequency: W-FRI
      window_periods: 52
      stride_divisor: 4
      lookback_default: 2Y
      
    monthly:
      frequency: M
      window_periods: 60
      stride_divisor: 4
      lookback_default: 10Y
      
    quarterly:
      frequency: Q
      window_periods: 40
      stride_divisor: 4
      lookback_default: 30Y

  aggregation:
    price: last
    yield: mean
    rate: mean
    ratio: last
    volume: sum
    flow: sum
    volatility: mean
    index: last
    level: last
    change: sum
    default: last
    
  # Future: multi-resolution (disabled for now)
  multi_resolution:
    enabled: false
```

---

### 5. Update Registry Files

**Modify `data_fetch/market_registry.json`:**

Add `data_type` field to each indicator:
```json
{
  "spy": {
    "source": "yahoo",
    "symbol": "SPY",
    "data_type": "price",
    "native_frequency": "daily"
  },
  "vix": {
    "source": "fred",
    "symbol": "VIXCLS",
    "data_type": "volatility",
    "native_frequency": "daily"
  }
}
```

**Modify `data_fetch/economic_registry.json`:**
```json
{
  "cpi": {
    "source": "fred",
    "symbol": "CPIAUCSL",
    "data_type": "index",
    "native_frequency": "monthly"
  },
  "gdp": {
    "source": "fred",
    "symbol": "GDP",
    "data_type": "level",
    "native_frequency": "quarterly"
  },
  "dgs10": {
    "source": "fred", 
    "symbol": "DGS10",
    "data_type": "yield",
    "native_frequency": "daily"
  }
}
```

---

## Files to Delete

```
# Remove if they exist as separate device profile configs
config/profiles/chromebook.yaml
config/profiles/standard.yaml  
config/profiles/powerful.yaml
```

---

## Tests to Add

### `tests/test_temporal_resolution.py`

```python
import pytest
import pandas as pd
import numpy as np
from engine_core.orchestration.temporal_runner import TemporalConfig, TemporalRunner
from engine_core.orchestration.temporal_aggregator import (
    aggregate_panel_to_frequency,
    detect_native_frequency,
    get_indicator_data_type
)

class TestTemporalConfig:
    
    def test_default_resolution_is_monthly(self):
        config = TemporalConfig.from_resolution()
        assert config.resolution == 'monthly'
        assert config.frequency == 'M'
        assert config.window_periods == 60
    
    def test_weekly_resolution(self):
        config = TemporalConfig.from_resolution('weekly')
        assert config.resolution == 'weekly'
        assert config.frequency == 'W-FRI'
        assert config.window_periods == 52
        assert config.stride_periods == 13
    
    def test_quarterly_resolution(self):
        config = TemporalConfig.from_resolution('quarterly')
        assert config.resolution == 'quarterly'
        assert config.frequency == 'Q'
        assert config.window_periods == 40
    
    def test_override_window(self):
        config = TemporalConfig.from_resolution('monthly', window_periods=24)
        assert config.window_periods == 24


class TestAggregation:
    
    @pytest.fixture
    def daily_panel(self):
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        return pd.DataFrame({
            'spy_close': np.random.randn(len(dates)).cumsum() + 100,
            'dgs10': np.random.randn(len(dates)) * 0.5 + 2.0,
            'volume': np.random.randint(1000, 10000, len(dates)),
        }, index=dates)
    
    def test_aggregate_to_monthly(self, daily_panel):
        result = aggregate_panel_to_frequency(daily_panel, 'M')
        assert len(result) == 12  # 12 months
        assert all(col in result.columns for col in daily_panel.columns)
    
    def test_aggregate_to_weekly(self, daily_panel):
        result = aggregate_panel_to_frequency(daily_panel, 'W-FRI')
        assert len(result) >= 50  # ~52 weeks
    
    def test_price_uses_last(self, daily_panel):
        result = aggregate_panel_to_frequency(daily_panel, 'M')
        # Last value of January should match
        jan_last = daily_panel.loc['2020-01-31', 'spy_close']
        assert result.loc['2020-01-31', 'spy_close'] == jan_last


class TestFrequencyDetection:
    
    def test_detect_daily(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        series = pd.Series(range(100), index=dates)
        assert detect_native_frequency(series) == 'daily'
    
    def test_detect_monthly(self):
        dates = pd.date_range('2020-01-01', periods=24, freq='M')
        series = pd.Series(range(24), index=dates)
        assert detect_native_frequency(series) == 'monthly'


class TestDataTypeDetection:
    
    def test_spy_is_price(self):
        assert get_indicator_data_type('spy_close', {}) == 'price'
    
    def test_dgs10_is_yield(self):
        assert get_indicator_data_type('dgs10', {}) == 'yield'
    
    def test_vix_is_volatility(self):
        assert get_indicator_data_type('vix', {}) == 'volatility'
    
    def test_unknown_is_default(self):
        assert get_indicator_data_type('unknown_indicator', {}) == 'default'
```

---

## Migration Guide

### For Existing Scripts

**Before:**
```python
runner = TemporalRunner(device_profile='standard')
```

**After:**
```python
runner = TemporalRunner(resolution='monthly')
```

### CLI Changes

**Before:**
```bash
python start/temporal_runner.py --profile standard
```

**After:**
```bash
python start/temporal_runner.py --resolution monthly
```

---

## Verification Checklist

- [ ] Device profile code removed from `temporal_runner.py`
- [ ] Resolution presets implemented
- [ ] Aggregation rules implemented
- [ ] Registry files updated with `data_type` fields
- [ ] CLI updated to use `--resolution`
- [ ] Tests pass
- [ ] Existing outputs regenerate correctly with new system
- [ ] Documentation updated

---

## Out of Scope (Future PRs)

- Multi-resolution parallel analysis
- Auto-detection of dominant cycle for window sizing
- Cross-domain frequency handling
- Spectral-based window optimization

These are deferred to Phase 10+.

---

## Notes for Implementation

1. **Backward Compatibility:** If `device_profile` is passed, log a deprecation warning and map to resolution:
   - chromebook → weekly
   - standard → monthly
   - powerful → monthly

2. **Default Behavior:** If no resolution specified, default to monthly. This matches the most common use case (market + economic analysis).

3. **Aggregation Edge Cases:** Handle NaN-heavy series gracefully. If a series is >50% NaN after aggregation, flag it in metadata.

4. **Registry Updates:** The `data_type` field in registries is optional. The heuristic fallback based on indicator names should handle most cases.
