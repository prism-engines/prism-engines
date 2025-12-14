# PR: Introduce Lens Architecture and Geometry Schema

## Summary

This PR introduces a clean separation between **data transforms** (pipeline mutations) and **analytical lenses** (geometry observations). It establishes the canonical artifact schema for PRISM outputs.

## Motivation

The existing engines in `/prism/engine/` had a contract mismatch:
- `BaseEngine` expected `run(df) -> DataFrame`
- Actual engines implemented `run(indicator, df) -> dict`

This PR resolves that by introducing `LensBase` — a distinct abstraction for analytical observation that reflects what the engines actually do.

Per the PRISM architecture document:
> "PRISM is an observational instrument... it reveals structure that is otherwise invisible."

Lenses observe. Transforms mutate. These are now cleanly separated.

## What's New

### 1. `LensBase` (`prism/engine/lens_base.py`)

Abstract base class for analytical lenses:
- Input: `(indicator: str, df: DataFrame)` where df has a `value` column
- Output: `dict` of metric_name → numeric value
- Stateless, deterministic, no side effects

```python
class EntropyLens(LensBase):
    name = "entropy"
    
    def _observe(self, indicator: str, df):
        # ... compute entropy ...
        return {"entropy": float(entropy), "length": len(values)}
```

### 2. Geometry Schema (`prism/schema/`)

Three-level artifact hierarchy:

| Level | Class | Description |
|-------|-------|-------------|
| 1 | `LensObservation` | Single lens on single indicator |
| 2 | `IndicatorGeometry` | All lenses on single indicator |
| 3 | `SystemSnapshot` | All indicators, all lenses, one moment |

All immutable. All with provenance (timestamps, hashes).

### 3. System Metrics (`prism/schema/system_metrics.py`)

Derived geometry at the system level:
- `effective_dim` — How many independent dimensions? (via SVD)
- `lens_agreement` — Do lenses rank indicators similarly? (Spearman correlation)
- `indicator_concentration` — Is variance concentrated? (HHI)

These are **observations**, not interpretations.

### 4. `Observer` (`prism/engine/observer.py`)

Orchestrates lenses and produces `SystemSnapshot`:

```python
observer = Observer([
    EntropyLens(),
    CurvatureLens(),
    GapLens(),
])

snapshot = observer.observe({
    "SP500": sp500_df,
    "DXY": dxy_df,
})

print(snapshot.system_metrics["effective_dim"])
```

### 5. `ArtifactWriter` (`prism/io/writer.py`)

Exports geometry artifacts:
- `observations.csv` (or `.parquet`) — flat observation data
- `manifest.json` — provenance and system metrics

```python
writer = ArtifactWriter("./output/2024-01-15")
writer.write(snapshot, format="csv")
```

### 6. Updated Lenses

All engines converted to `LensBase`:
- `GeometryLens` — mean, std, slope
- `EntropyLens` — distributional complexity
- `CurvatureLens` — second derivative stats
- `GapLens` — curvature volatility
- `ChangepointLens` — structural breaks
- `SparsityLens` — jump behavior
- `CompareLens` — composite metrics

### 7. Updated Registry

New `LENS_REGISTRY` with helper functions:
- `get_lens_class(name)` — returns class
- `get_lens(name)` — returns instance
- `list_lenses()` — available lens names

Legacy `ENGINE_REGISTRY` preserved for backward compatibility.

## Files Added

```
prism/
├── engine/
│   ├── lens_base.py      # NEW: LensBase abstraction
│   ├── observer.py       # NEW: Observation orchestrator
│   ├── registry.py       # UPDATED: Added LENS_REGISTRY
│   ├── entropy.py        # UPDATED: Now extends LensBase
│   ├── geometry.py       # UPDATED: Now extends LensBase
│   ├── gap.py            # UPDATED: Now extends LensBase
│   ├── curvature.py      # UPDATED: Now extends LensBase
│   ├── changepoint.py    # UPDATED: Now extends LensBase
│   ├── sparsity.py       # UPDATED: Now extends LensBase
│   └── compare.py        # UPDATED: Now extends LensBase
├── schema/
│   ├── __init__.py       # NEW
│   ├── observation.py    # NEW: LensObservation, IndicatorGeometry, SystemSnapshot
│   └── system_metrics.py # NEW: effective_dim, lens_agreement, concentration
├── io/
│   ├── __init__.py       # NEW
│   └── writer.py         # NEW: ArtifactWriter, ArtifactReader
examples/
└── observe_demo.py       # NEW: Usage demonstration
```

## Files Not Changed

- `prism/engine/base.py` — Preserved (may still be used elsewhere)
- `prism/engine/cluster.py` — Not converted (different pattern, uses store)
- `prism/engine/normalize.py` — Not converted (utility function, not a lens)
- `prism/core/*` — Pipeline infrastructure unchanged

## Breaking Changes

None. This is additive. Existing code paths unchanged.

## Migration Path

Old pattern (still works):
```python
from prism.engine.entropy import EntropyEngine
engine = EntropyEngine()
result = engine.run(indicator, df)  # Returns dict
```

New pattern (recommended):
```python
from prism.engine.observer import Observer
from prism.engine.registry import get_lens

observer = Observer([get_lens("entropy")])
snapshot = observer.observe({"indicator": df})
```

## Testing

Run the demo:
```bash
cd prism
python -m examples.observe_demo
```

Or import directly:
```python
from prism.engine.observer import Observer
from prism.engine.entropy import EntropyLens
# ...
```

## Output Example

**observations.csv:**
```
indicator,lens,window_start,window_end,observed_at,data_points,m_entropy,m_std,m_mean,snapshot_hash
SP500,entropy,2020-01-01,2021-05-15,2024-01-15T10:23:45,500,2.34,,,a1b2c3d4
SP500,geometry,2020-01-01,2021-05-15,2024-01-15T10:23:45,500,,234.5,3250.2,a1b2c3d4
...
```

**manifest.json:**
```json
{
  "snapshot_hash": "a1b2c3d4",
  "observed_at": "2024-01-15T10:23:45+00:00",
  "window_start": "2020-01-01",
  "window_end": "2021-05-15",
  "indicators": ["SP500", "TNX", "DXY", "AGG"],
  "lenses": ["entropy", "geometry", "curvature", "gap"],
  "system_metrics": {
    "effective_dim": 2.73,
    "lens_agreement": 0.81,
    "indicator_concentration": 0.34
  }
}
```

## North Star Check

> A scientist runs PRISM and says: "Something moved — now I want to understand why."

✓ Effective dimensionality dropped? Something collapsed.  
✓ Lens agreement spiked? Structure converging.  
✓ Concentration increased? Fewer drivers.  

PRISM observes. Scientists interpret.

---

## Next Steps (Future PRs)

1. Time-indexed snapshots (rolling windows)
2. Snapshot comparison / delta metrics
3. Database persistence for geometry history
4. Additional lenses (wavelets, Granger, etc.)
