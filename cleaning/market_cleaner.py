"""
MarketCleaner - concrete cleaner for financial / market time series.

Responsibilities:
- Normalize date handling and column names
- Handle missing values with robust defaults
- Detect and clip extreme outliers
- Log operations via BaseCleaner
- Validate final NaN levels using BaseCleaner.validate_output
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List
from dataclasses import dataclass

import logging
import math

import numpy as np
import pandas as pd

from .cleaner_base import BaseCleaner

logger = logging.getLogger(__name__)


@dataclass
class MarketCleaningConfig:
    """
    Configuration for MarketCleaner.

    This is intentionally simple and internal. We can later extend this
    to load from cleaning/configs/financial_cleaning.yaml without
    breaking the API.
    """
    max_nan_pct_per_column: float = 5.0
    max_nan_pct_before_drop: float = 50.0  # drop columns above this
    outlier_zscore_threshold: float = 6.0  # relatively conservative
    min_history_points: int = 30          # if less, skip outlier logic
    treat_zero_as_nan_for: Optional[Iterable[str]] = None  # e.g. ["volume"]


class MarketCleaner(BaseCleaner):
    """
    Concrete cleaner for market / financial time series.

    Assumptions:
    - DataFrame has a 'date' column OR a DatetimeIndex.
    - All other columns are numeric market series (prices, volumes, etc.).
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        config: Optional[MarketCleaningConfig] = None,
    ) -> None:
        super().__init__(checkpoint_dir=checkpoint_dir)
        self.config = config or MarketCleaningConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Clean market data.

        Steps:
        1. Standardize date & column naming
        2. Optionally treat some zeros as NaN
        3. Handle NaNs (ffill, bfill, then optionally drop bad columns)
        4. Clip extreme outliers using z-score
        5. Validate NaN levels

        Returns:
            Cleaned DataFrame with a 'date' column.
        """
        if df is None or df.empty:
            raise ValueError("MarketCleaner.clean() received empty DataFrame")

        df = df.copy()

        df = self._ensure_date_column(df)
        df = self._standardize_column_names(df)

        df = self._maybe_convert_zeros_to_nan(df)
        before_nan_summary = self.get_nan_summary(df)

        df = self._handle_missing_values(df)
        df = self._clip_outliers(df)

        after_nan_summary = self.get_nan_summary(df)
        self._log_nan_improvement(before_nan_summary, after_nan_summary)

        if not self.validate_output(
            df,
            max_nan_pct=self.config.max_nan_pct_per_column,
        ):
            logger.warning(
                "MarketCleaner.validate_output() failed NaN thresholds. "
                "Downstream engines should treat this panel cautiously."
            )

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure there is a 'date' column of dtype datetime64[ns]."""
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "date"})
            else:
                # Best-effort: try to detect a datetime-like column
                datetime_like = [
                    c
                    for c in df.columns
                    if np.issubdtype(df[c].dtype, np.datetime64)
                ]
                if len(datetime_like) == 1:
                    df = df.rename(columns={datetime_like[0]: "date"})
                else:
                    raise ValueError(
                        "MarketCleaner expected a 'date' column or DatetimeIndex."
                    )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"])

        return df.reset_index(drop=True)

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case, strip spaces, unify simple price/volume naming."""
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            if col == "date":
                continue
            clean = col.strip()
            clean = clean.replace(" ", "_")
            clean = clean.replace("-", "_")
            clean = clean.lower()
            rename_map[col] = clean

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _maybe_convert_zeros_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally treat some zeros as NaN (e.g. volume = 0 on weekends).

        Columns to treat this way are configured via config.treat_zero_as_nan_for.
        """
        cols = list(self.config.treat_zero_as_nan_for or [])
        if not cols:
            return df

        df = df.copy()
        for col in cols:
            if col in df.columns:
                before = int(df[col].isna().sum())
                df.loc[df[col] == 0, col] = np.nan
                after = int(df[col].isna().sum())
                if after > before:
                    self.log_operation(
                        operation="zero_to_nan",
                        column=col,
                        before_count=before,
                        after_count=after,
                        details={"added_nans": after - before},
                    )
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaNs using a conservative strategy:

        - Forward fill within each column
        - Backward fill remaining
        - Drop columns where NaN pct is still above max_nan_pct_before_drop
        """
        df = df.copy()

        numeric_cols = [
            c
            for c in df.columns
            if c != "date" and np.issubdtype(df[c].dtype, np.number)
        ]

        # Before counts
        before_nan = {c: int(df[c].isna().sum()) for c in numeric_cols}

        # FFill then BFill
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # After fill counts
        after_nan = {c: int(df[c].isna().sum()) for c in numeric_cols}

        # Log operations column by column
        for col in numeric_cols:
            self.log_operation(
                operation="nan_fill_ffill_bfill",
                column=col,
                before_count=before_nan[col],
                after_count=after_nan[col],
                details={},
            )

        # Now drop extremely bad columns if any remain
        to_drop: List[str] = []
        n_rows = len(df)
        for col in numeric_cols:
            if n_rows == 0:
                continue
            nan_pct = df[col].isna().sum() / n_rows * 100
            if nan_pct > self.config.max_nan_pct_before_drop:
                logger.warning(
                    "MarketCleaner dropping column '%s' due to %.1f%% NaNs "
                    "(threshold=%.1f%%)",
                    col,
                    nan_pct,
                    self.config.max_nan_pct_before_drop,
                )
                to_drop.append(col)

        if to_drop:
            df = df.drop(columns=to_drop)

        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip extreme outliers using simple z-score logic.

        We avoid modifying the 'date' column and only consider numeric columns.
        """
        df = df.copy()

        numeric_cols = [
            c
            for c in df.columns
            if c != "date" and np.issubdtype(df[c].dtype, np.number)
        ]

        if len(df) < self.config.min_history_points:
            logger.info(
                "MarketCleaner._clip_outliers: not enough history (%d rows) "
                "to safely apply z-score clipping (min=%d). Skipping.",
                len(df),
                self.config.min_history_points,
            )
            return df

        for col in numeric_cols:
            series = df[col]
            if series.isna().all():
                continue

            mean = series.mean()
            std = series.std(ddof=0)

            if std == 0 or math.isnan(std):
                continue

            z = (series - mean) / std
            threshold = self.config.outlier_zscore_threshold

            high_mask = z > threshold
            low_mask = z < -threshold
            any_outliers = bool(high_mask.any() or low_mask.any())

            if any_outliers:
                before_nan = int(series.isna().sum())
                clipped = series.clip(
                    lower=mean - threshold * std,
                    upper=mean + threshold * std,
                )
                df[col] = clipped
                after_nan = int(df[col].isna().sum())

                self.log_operation(
                    operation="zscore_clip",
                    column=col,
                    before_count=before_nan,
                    after_count=after_nan,
                    details={
                        "threshold": threshold,
                        "mean": float(mean),
                        "std": float(std),
                    },
                )

        return df

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_nan_improvement(
        self,
        before: Dict[str, Dict[str, Any]],
        after: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Log summary improvement from before->after cleaning using
        BaseCleaner.get_nan_summary outputs.
        """
        for col, before_stats in before.items():
            if col not in after:
                continue
            after_stats = after[col]
            before_nan = before_stats.get("nan_count", 0)
            after_nan = after_stats.get("nan_count", 0)

            if after_nan < before_nan:
                logger.info(
                    "MarketCleaner NaN improvement for %s: %d → %d "
                    "(%.2f%% → %.2f%%)",
                    col,
                    before_nan,
                    after_nan,
                    before_stats.get("nan_pct", 0.0),
                    after_stats.get("nan_pct", 0.0),
                )

