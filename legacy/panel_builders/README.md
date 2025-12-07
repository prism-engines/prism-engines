# Legacy Panel Builders (DEPRECATED)

These files have been moved here for historical reference only.
They are **NOT** imported or used by the active codebase.

## Why deprecated?

The new PRISM architecture is **domain-agnostic**:

- **OLD**: Panels were defined in files with domain-specific logic (market, economic, climate)
- **NEW**: Panels are defined at runtime by the UI as simple lists of indicator names

## New approach

```python
from panel.runtime_loader import load_panel

# Panel = list of indicator names (defined by UI at runtime)
selected_indicators = ["sp500", "vix", "t10y2y"]
panel = load_panel(selected_indicators)
```

## Files in this folder

| File | Was | Now use |
|------|-----|---------|
| `build_panel.py` | Domain-specific panel builder | `panel.runtime_loader.load_panel()` |
| `build_climate_panel.py` | Climate-specific panel | `panel.runtime_loader.load_panel()` |
| `transforms_market.py` | Market-specific transforms | Generic transforms |
| `transforms_econ.py` | Economic-specific transforms | Generic transforms |

## Do not import

These files contain `DeprecationWarning` statements and should not be imported.
They are retained only for reference during the migration period.
