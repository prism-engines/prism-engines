"""
Base Lens - Abstract interface for all analytical lenses
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseLens(ABC):
    """
    Abstract base class for all PRISM lenses.

    Each lens provides a different mathematical perspective on the data.
    All lenses must implement:
    - analyze(): Run the analysis
    - rank_indicators(): Rank indicators by importance
    """

    # Lens metadata
    name: str = "base"
    description: str = "Base lens class"
    category: str = "base"  # 'basic' or 'advanced'

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize lens.

        Args:
            checkpoint_dir: Directory for saving checkpoints
        """
        self.checkpoint_dir = checkpoint_dir or Path("engine_core/checkpoints/lens_outputs")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_result: Optional[Dict] = None
        self._computation_time: float = 0.0

    @abstractmethod
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Run the lens analysis.

        Args:
            df: Input DataFrame with 'date' column and indicator columns
            **kwargs: Lens-specific parameters

        Returns:
            Dictionary with analysis results
        """
        pass

    @abstractmethod
    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by importance according to this lens.

        Args:
            df: Input DataFrame
            **kwargs: Lens-specific parameters

        Returns:
            DataFrame with columns ['indicator', 'score', 'rank']
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            True if valid

        Raises:
            ValueError: If invalid
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")

        if "date" not in df.columns:
            raise ValueError("DataFrame must have 'date' column")

        # Check for at least 2 indicator columns
        value_cols = [c for c in df.columns if c != "date"]
        if len(value_cols) < 2:
            raise ValueError("DataFrame must have at least 2 indicator columns")

        return True

    def get_value_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of value columns (excluding date)."""
        return [c for c in df.columns if c != "date"]

    def prepare_data(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """
        Prepare data for analysis.

        Args:
            df: Input DataFrame
            dropna: Whether to drop rows with NaN

        Returns:
            Prepared DataFrame
        """
        result = df.copy()

        # Ensure date is datetime
        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"])
            result = result.sort_values("date")

        # Handle NaN
        if dropna:
            result = result.dropna()

        return result

    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = "zscore"
    ) -> pd.DataFrame:
        """
        Normalize data for analysis.

        Args:
            df: Input DataFrame
            method: 'zscore', 'minmax', or 'robust'

        Returns:
            Normalized DataFrame
        """
        result = df.copy()
        value_cols = self.get_value_columns(df)

        for col in value_cols:
            if method == "zscore":
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - mean) / std
            elif method == "minmax":
                min_val = result[col].min()
                max_val = result[col].max()
                if max_val > min_val:
                    result[col] = (result[col] - min_val) / (max_val - min_val)
            elif method == "robust":
                median = result[col].median()
                iqr = result[col].quantile(0.75) - result[col].quantile(0.25)
                if iqr > 0:
                    result[col] = (result[col] - median) / iqr

        return result

    def save_checkpoint(self, result: Dict, name: Optional[str] = None) -> Path:
        """
        Save analysis result as checkpoint.

        Args:
            result: Analysis result dictionary
            name: Optional custom name

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or self.name
        checkpoint_path = self.checkpoint_dir / f"{name}_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        import json

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        serializable = {k: convert(v) for k, v in result.items()}

        with open(checkpoint_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def get_metadata(self) -> Dict[str, Any]:
        """Get lens metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "last_computation_time": self._computation_time,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
