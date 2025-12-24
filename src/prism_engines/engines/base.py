"""
Base classes for PRISM engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class EngineResult:
    """Result from a single engine run."""
    engine_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        if self.error:
            return f"EngineResult({self.engine_name}, error={self.error})"
        return f"EngineResult({self.engine_name}, metrics={list(self.metrics.keys())})"


class BaseEngine(ABC):
    """
    Abstract base class for all PRISM engines.

    An engine takes a DataFrame and computes geometric/statistical metrics.
    """

    name: str = "base"
    description: str = "Base engine"

    @abstractmethod
    def run(self, df: pd.DataFrame) -> EngineResult:
        """
        Run the engine on a DataFrame.

        Args:
            df: DataFrame with numeric columns (each column is a time series)

        Returns:
            EngineResult with computed metrics
        """
        pass

    def validate(self, df: pd.DataFrame) -> Optional[str]:
        """
        Validate input data. Returns error message if invalid, None if valid.
        """
        if df.empty:
            return "DataFrame is empty"
        if len(df.columns) < 2:
            return "Need at least 2 columns for analysis"
        if len(df) < 10:
            return "Need at least 10 rows for analysis"
        return None
