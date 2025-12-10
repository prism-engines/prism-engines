"""
Base Cleaner - Abstract interface for data cleaning
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseCleaner(ABC):
    """
    Abstract base class for data cleaners.

    Subclasses implement specific cleaning strategies for different data types.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize cleaner.

        Args:
            checkpoint_dir: Directory to save cleaning checkpoints
        """
        self.checkpoint_dir = checkpoint_dir or Path("cleaning/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cleaning_log: List[Dict[str, Any]] = []

    @abstractmethod
    def clean(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Clean the input DataFrame.

        Args:
            df: Input DataFrame with 'date' column
            **kwargs: Strategy-specific parameters

        Returns:
            Cleaned DataFrame
        """
        pass

    def log_operation(
        self,
        operation: str,
        column: str,
        before_count: int,
        after_count: int,
        details: Optional[Dict] = None
    ) -> None:
        """Log a cleaning operation for audit trail."""
        self.cleaning_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "column": column,
            "before_nan": before_count,
            "after_nan": after_count,
            "filled": before_count - after_count,
            "details": details or {}
        })

    def get_nan_summary(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get summary of NaN values per column.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with NaN statistics per column
        """
        summary = {}
        for col in df.columns:
            if col == "date":
                continue
            n_total = len(df)
            n_nan = df[col].isna().sum()
            summary[col] = {
                "total": n_total,
                "nan_count": int(n_nan),
                "nan_pct": round(n_nan / n_total * 100, 2) if n_total > 0 else 0,
                "valid_count": int(n_total - n_nan),
            }
        return summary

    def save_before_after(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        name: str
    ) -> Path:
        """
        Save before/after comparison for inspection.

        Args:
            before: DataFrame before cleaning
            after: DataFrame after cleaning
            name: Name for the checkpoint

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.checkpoint_dir / "before_after" / f"{name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        before.to_parquet(save_dir / "before.parquet")
        after.to_parquet(save_dir / "after.parquet")

        # Save diff summary
        diff_summary = self._compute_diff(before, after)
        pd.DataFrame(diff_summary).T.to_csv(save_dir / "diff_summary.csv")

        logger.info(f"Before/after saved to {save_dir}")
        return save_dir

    def _compute_diff(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Compute difference statistics between before and after."""
        diff = {}
        for col in before.columns:
            if col == "date":
                continue
            if col in after.columns:
                before_nan = before[col].isna().sum()
                after_nan = after[col].isna().sum()
                diff[col] = {
                    "before_nan": int(before_nan),
                    "after_nan": int(after_nan),
                    "filled": int(before_nan - after_nan),
                    "fill_rate": round((before_nan - after_nan) / before_nan * 100, 2)
                                 if before_nan > 0 else 0
                }
        return diff

    def validate_output(self, df: pd.DataFrame, max_nan_pct: float = 5.0) -> bool:
        """
        Validate cleaned output meets quality thresholds.

        Args:
            df: Cleaned DataFrame
            max_nan_pct: Maximum allowed NaN percentage per column

        Returns:
            True if all columns pass validation
        """
        passed = True
        for col in df.columns:
            if col == "date":
                continue
            nan_pct = df[col].isna().sum() / len(df) * 100
            if nan_pct > max_nan_pct:
                logger.warning(f"{col}: {nan_pct:.1f}% NaN exceeds threshold {max_nan_pct}%")
                passed = False
        return passed
