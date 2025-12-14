# prism/engine/observer.py
"""
Observer - The sensor array that runs lenses and produces geometry.

The Observer:
- Accepts prepared data (from pipeline)
- Runs all configured lenses on all indicators
- Produces a SystemSnapshot with full provenance

This is the boundary between data preparation and geometry observation.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from prism.engine.lens_base import LensBase
from prism.schema.observation import (
    LensObservation,
    IndicatorGeometry,
    SystemSnapshot,
)
from prism.schema.system_metrics import compute_system_metrics


class Observer:
    """
    Runs lenses and produces geometry artifacts.

    Usage:
        observer = Observer([EntropyLens(), CurvatureLens(), GapLens()])
        snapshot = observer.observe({"SP500": df1, "DXY": df2})
    """

    def __init__(self, lenses: List[LensBase]):
        """
        Initialize with a list of lenses.

        Args:
            lenses: List of LensBase instances to apply
        """
        if not lenses:
            raise ValueError("Observer requires at least one lens")

        self.lenses = lenses

    @property
    def lens_names(self) -> List[str]:
        """Names of all configured lenses."""
        return [lens.name for lens in self.lenses]

    def observe(
        self,
        data: Dict[str, pd.DataFrame],
        manifest: Optional[Dict[str, Any]] = None,
    ) -> SystemSnapshot:
        """
        Observe all indicators through all lenses.

        Args:
            data: Dict mapping indicator name to DataFrame.
                  Each DataFrame must have 'date' and 'value' columns.
            manifest: Optional metadata to include in provenance.

        Returns:
            SystemSnapshot with complete geometry observations.
        """
        if not data:
            raise ValueError("No data provided to observe")

        manifest = manifest or {}

        # Validate input DataFrames
        for name, df in data.items():
            if "value" not in df.columns:
                raise ValueError(f"DataFrame for '{name}' missing 'value' column")
            if "date" not in df.columns:
                raise ValueError(f"DataFrame for '{name}' missing 'date' column")

        # Determine observation window from data
        all_dates = []
        for df in data.values():
            dates = pd.to_datetime(df["date"])
            all_dates.extend(dates.tolist())

        window_start = str(min(all_dates).date())
        window_end = str(max(all_dates).date())

        # Build indicator geometries
        indicators = {}

        for indicator, df in data.items():
            observations = self._observe_indicator(
                indicator, df, window_start, window_end
            )

            indicators[indicator] = IndicatorGeometry(
                indicator=indicator,
                window_start=window_start,
                window_end=window_end,
                observations=tuple(observations),
            )

        # Build snapshot
        snapshot = SystemSnapshot(
            window_start=window_start,
            window_end=window_end,
            indicators=indicators,
            manifest={
                "lenses": self.lens_names,
                "indicators": list(data.keys()),
                **manifest,
            },
        )

        # Compute system-level metrics
        snapshot.system_metrics = compute_system_metrics(snapshot)

        return snapshot

    def _observe_indicator(
        self,
        indicator: str,
        df: pd.DataFrame,
        window_start: str,
        window_end: str,
    ) -> List[LensObservation]:
        """
        Run all lenses on a single indicator.

        Returns list of LensObservation objects.
        """
        observations = []

        for lens in self.lenses:
            # Run the lens
            raw_metrics = lens.observe(indicator, df)

            # Extract clean metrics (numeric only, no underscore-prefixed keys)
            clean_metrics = {
                k: v
                for k, v in raw_metrics.items()
                if not k.startswith("_") and isinstance(v, (int, float))
            }

            obs = LensObservation(
                indicator=indicator,
                lens=lens.name,
                window_start=window_start,
                window_end=window_end,
                metrics=clean_metrics,
                data_points=len(df),
            )
            observations.append(obs)

        return observations

    def observe_one(
        self,
        indicator: str,
        df: pd.DataFrame,
    ) -> IndicatorGeometry:
        """
        Convenience method to observe a single indicator.

        Returns IndicatorGeometry (not a full SystemSnapshot).
        """
        if "value" not in df.columns:
            raise ValueError("DataFrame missing 'value' column")
        if "date" not in df.columns:
            raise ValueError("DataFrame missing 'date' column")

        dates = pd.to_datetime(df["date"])
        window_start = str(dates.min().date())
        window_end = str(dates.max().date())

        observations = self._observe_indicator(
            indicator, df, window_start, window_end
        )

        return IndicatorGeometry(
            indicator=indicator,
            window_start=window_start,
            window_end=window_end,
            observations=tuple(observations),
        )
