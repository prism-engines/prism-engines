# prism/engine/lens_base.py
"""
LensBase - Core abstraction for analytical lenses.

Lenses are NOT transforms. They do not mutate data.
They observe a time series and emit geometry metrics.

This is distinct from TransformBase which is for pipeline mutations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class LensBase(ABC):
    """
    Base class for analytical lenses.

    Contract:
    - Input: indicator name + DataFrame with 'value' column
    - Output: dict of metric_name -> scalar
    - Lenses are stateless and deterministic
    - Lenses do NOT interpret, they observe

    Implementation:
    - Subclass and implement _observe()
    - Set `name` class attribute
    """

    name: str = "unnamed_lens"

    def __call__(self, indicator: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Allow lens(indicator, df) syntax."""
        return self.observe(indicator, df)

    def observe(self, indicator: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Public entry point with validation.

        Args:
            indicator: Name of the indicator being observed
            df: DataFrame with at least a 'value' column

        Returns:
            Dict of metric_name -> numeric value
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Lens expects DataFrame, got {type(df)}")

        if "value" not in df.columns:
            raise ValueError("DataFrame must have 'value' column")

        result = self._observe(indicator, df)

        # Enforce output contract
        if not isinstance(result, dict):
            raise TypeError(f"Lens must return dict, got {type(result)}")

        # Attach provenance (prefixed to avoid collision)
        result["_lens"] = self.name
        result["_indicator"] = indicator

        return result

    @abstractmethod
    def _observe(self, indicator: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Implement lens logic here.

        Return a flat dict of metric_name -> numeric value.
        Do not include indicator/lens names; those are added automatically.
        """
        raise NotImplementedError


# Compatibility alias
Lens = LensBase
