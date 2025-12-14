# prism/schema/observation.py
"""
Geometry observation schemas.

Three levels of structure:
1. LensObservation - single lens on single indicator
2. IndicatorGeometry - all lenses on single indicator
3. SystemSnapshot - all indicators, all lenses, one moment

These are OBSERVATIONS, not interpretations.
A scientist looks at these and asks "why?" — PRISM does not answer that.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import hashlib
import json


def _now() -> str:
    """UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _hash_manifest(manifest: dict) -> str:
    """Stable hash for provenance tracking."""
    payload = json.dumps(manifest, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Level 1: Single lens observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LensObservation:
    """
    Single lens applied to single indicator at a point in time.

    This is the atomic unit of PRISM output.
    Immutable. No interpretation.
    """

    # What was observed
    indicator: str
    lens: str

    # The observation window
    window_start: str  # ISO date
    window_end: str    # ISO date

    # Raw metrics (lens-specific, all numeric)
    metrics: Dict[str, float]

    # Provenance
    observed_at: str = field(default_factory=_now)
    data_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict for storage/serialization."""
        base = {
            "indicator": self.indicator,
            "lens": self.lens,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "observed_at": self.observed_at,
            "data_points": self.data_points,
        }
        # Flatten metrics with prefix to avoid collisions
        for k, v in self.metrics.items():
            base[f"m_{k}"] = v
        return base


# ---------------------------------------------------------------------------
# Level 2: Indicator geometry (one indicator, all lenses)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndicatorGeometry:
    """
    All lens observations for a single indicator.

    Captures how one signal looks through every lens.
    """

    indicator: str
    window_start: str
    window_end: str

    observations: Tuple[LensObservation, ...]  # Immutable sequence

    observed_at: str = field(default_factory=_now)

    @property
    def lenses_applied(self) -> Tuple[str, ...]:
        """Names of all lenses that observed this indicator."""
        return tuple(o.lens for o in self.observations)

    def metrics_by_lens(self) -> Dict[str, Dict[str, float]]:
        """Get metrics organized by lens name."""
        return {o.lens: o.metrics for o in self.observations}

    def get_metric(self, lens: str, metric: str) -> float:
        """Get a specific metric from a specific lens."""
        for o in self.observations:
            if o.lens == lens and metric in o.metrics:
                return o.metrics[metric]
        raise KeyError(f"No metric '{metric}' from lens '{lens}'")

    def to_rows(self) -> List[Dict[str, Any]]:
        """One row per lens observation."""
        return [o.to_dict() for o in self.observations]


# ---------------------------------------------------------------------------
# Level 3: System snapshot (all indicators, all lenses, one moment)
# ---------------------------------------------------------------------------

@dataclass
class SystemSnapshot:
    """
    Complete geometry state at a moment in time.

    This is what PRISM emits. A scientist can:
    - Inspect any indicator
    - Compare across lenses
    - Track changes over time

    No interpretation. Just structure.
    """

    window_start: str
    window_end: str

    indicators: Dict[str, IndicatorGeometry]

    # System-level derived metrics (computed, not interpreted)
    system_metrics: Dict[str, float] = field(default_factory=dict)

    # Provenance
    observed_at: str = field(default_factory=_now)
    manifest: Dict[str, Any] = field(default_factory=dict)
    snapshot_hash: str = field(init=False, default="")

    def __post_init__(self):
        """Compute snapshot hash from manifest."""
        self.snapshot_hash = _hash_manifest({
            "window": (self.window_start, self.window_end),
            "indicators": sorted(self.indicators.keys()),
            "manifest": self.manifest,
        })

    # ---- Accessors ----

    @property
    def indicator_names(self) -> List[str]:
        """All indicator names in this snapshot."""
        return list(self.indicators.keys())

    @property
    def lens_names(self) -> List[str]:
        """All unique lens names used in this snapshot."""
        lenses = set()
        for ig in self.indicators.values():
            for obs in ig.observations:
                lenses.add(obs.lens)
        return sorted(lenses)

    def get_metric_matrix(self, metric: str) -> Dict[str, Dict[str, float]]:
        """
        indicator -> lens -> value for a specific metric.

        Useful for comparing how different indicators score
        on the same metric across lenses.
        """
        matrix = {}
        for ind_name, ind_geo in self.indicators.items():
            matrix[ind_name] = {}
            for obs in ind_geo.observations:
                if metric in obs.metrics:
                    matrix[ind_name][obs.lens] = obs.metrics[metric]
        return matrix

    def get_lens_vector(self, lens: str) -> Dict[str, Dict[str, float]]:
        """
        indicator -> metrics for a specific lens.

        Useful for seeing how all indicators look through one lens.
        """
        vector = {}
        for ind_name, ind_geo in self.indicators.items():
            for obs in ind_geo.observations:
                if obs.lens == lens:
                    vector[ind_name] = obs.metrics
        return vector

    def get_indicator(self, name: str) -> IndicatorGeometry:
        """Get geometry for a specific indicator."""
        if name not in self.indicators:
            raise KeyError(f"No indicator '{name}' in snapshot")
        return self.indicators[name]

    # ---- Export ----

    def to_flat_rows(self) -> List[Dict[str, Any]]:
        """
        Flatten everything to rows for CSV/Parquet.
        One row per (indicator, lens) pair.
        """
        rows = []
        for ind_geo in self.indicators.values():
            for obs in ind_geo.observations:
                row = obs.to_dict()
                row["snapshot_hash"] = self.snapshot_hash
                rows.append(row)
        return rows

    def to_manifest(self) -> Dict[str, Any]:
        """
        Provenance record. What went into this snapshot.
        """
        return {
            "snapshot_hash": self.snapshot_hash,
            "observed_at": self.observed_at,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "indicators": sorted(self.indicators.keys()),
            "lenses": self.lens_names,
            "system_metrics": self.system_metrics,
            **self.manifest,
        }
