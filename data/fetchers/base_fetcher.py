"""
Base Fetcher Module
===================

Abstract base class for multi-domain data fetchers.
Provides common functionality for caching, validation, and error handling.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality metrics."""
    n_observations: int
    date_range: Tuple[str, str]
    missing_pct: float
    gaps: List[Tuple[str, str]]  # List of (start, end) gap periods
    outliers: int
    issues: List[str]
    passed: bool


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    success: bool
    data: Optional[pd.DataFrame]
    source: str
    fetch_time: datetime
    cache_hit: bool
    quality_report: Optional[DataQualityReport]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseFetcher(ABC):
    """
    Abstract base class for data fetchers.

    Provides:
    - Caching with TTL
    - Data validation
    - Error handling
    - Quality reporting
    """

    def __init__(self,
                 cache_dir: str = ".cache/fetchers",
                 cache_ttl_hours: int = 24):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory for caching data
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the data source."""
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL for the data source."""
        pass

    @abstractmethod
    def _fetch_raw(self,
                   indicator: str,
                   start_date: str,
                   end_date: str,
                   **kwargs) -> pd.DataFrame:
        """
        Fetch raw data from source.

        Args:
            indicator: Data indicator/series name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Additional source-specific parameters

        Returns:
            Raw DataFrame
        """
        pass

    def _get_cache_key(self, indicator: str, start_date: str,
                       end_date: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        params = {
            'source': self.source_name,
            'indicator': indicator,
            'start': start_date,
            'end': end_date,
            **kwargs
        }
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file."""
        return self.cache_dir / f"{self.source_name}_{cache_key}.parquet"

    def _check_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Check if valid cache exists."""
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        # Check TTL
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.cache_ttl:
            return None

        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None

    def _save_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            data.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _validate_data(self, data: pd.DataFrame) -> DataQualityReport:
        """
        Validate fetched data quality.

        Args:
            data: DataFrame to validate

        Returns:
            DataQualityReport
        """
        issues = []

        if data is None or len(data) == 0:
            return DataQualityReport(
                n_observations=0,
                date_range=('N/A', 'N/A'),
                missing_pct=100.0,
                gaps=[],
                outliers=0,
                issues=['No data returned'],
                passed=False
            )

        n_obs = len(data)

        # Date range
        if 'date' in data.columns:
            date_col = pd.to_datetime(data['date'])
            date_range = (str(date_col.min().date()), str(date_col.max().date()))
        elif isinstance(data.index, pd.DatetimeIndex):
            date_range = (str(data.index.min().date()), str(data.index.max().date()))
        else:
            date_range = ('unknown', 'unknown')
            issues.append("No date column found")

        # Missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            missing_pct = data[numeric_cols].isna().mean().mean() * 100
        else:
            missing_pct = 0.0

        if missing_pct > 5:
            issues.append(f"High missing rate: {missing_pct:.1f}%")

        # Detect gaps (for time series)
        gaps = []
        if 'date' in data.columns or isinstance(data.index, pd.DatetimeIndex):
            try:
                if 'date' in data.columns:
                    dates = pd.to_datetime(data['date']).sort_values()
                else:
                    dates = data.index.sort_values()

                # Expected frequency
                if len(dates) > 1:
                    median_diff = dates.diff().median()
                    large_gaps = dates.diff() > median_diff * 3

                    gap_indices = np.where(large_gaps)[0]
                    for idx in gap_indices:
                        gap_start = str(dates.iloc[idx-1].date())
                        gap_end = str(dates.iloc[idx].date())
                        gaps.append((gap_start, gap_end))
            except Exception:
                pass

        if len(gaps) > 5:
            issues.append(f"Data has {len(gaps)} significant gaps")

        # Outliers (simple IQR method)
        outlier_count = 0
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 10:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((values < q1 - 3*iqr) | (values > q3 + 3*iqr)).sum()
                outlier_count += outliers

        if outlier_count > n_obs * 0.01:
            issues.append(f"Significant outliers detected: {outlier_count}")

        # Determine pass/fail
        passed = len(issues) == 0 or (missing_pct < 10 and len(gaps) < 10)

        return DataQualityReport(
            n_observations=n_obs,
            date_range=date_range,
            missing_pct=missing_pct,
            gaps=gaps[:10],  # Limit reported gaps
            outliers=outlier_count,
            issues=issues,
            passed=passed
        )

    def fetch(self,
              indicator: str,
              start_date: str,
              end_date: str,
              use_cache: bool = True,
              **kwargs) -> FetchResult:
        """
        Fetch data with caching and validation.

        Args:
            indicator: Data indicator/series name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cache
            **kwargs: Additional source-specific parameters

        Returns:
            FetchResult with data and metadata
        """
        fetch_time = datetime.now()
        cache_key = self._get_cache_key(indicator, start_date, end_date, **kwargs)

        # Check cache
        if use_cache:
            cached_data = self._check_cache(cache_key)
            if cached_data is not None:
                quality = self._validate_data(cached_data)
                return FetchResult(
                    success=True,
                    data=cached_data,
                    source=self.source_name,
                    fetch_time=fetch_time,
                    cache_hit=True,
                    quality_report=quality,
                    metadata={'indicator': indicator, 'cached': True}
                )

        # Fetch from source
        try:
            data = self._fetch_raw(indicator, start_date, end_date, **kwargs)

            # Validate
            quality = self._validate_data(data)

            # Cache if valid
            if quality.passed and use_cache:
                self._save_cache(cache_key, data)

            return FetchResult(
                success=True,
                data=data,
                source=self.source_name,
                fetch_time=fetch_time,
                cache_hit=False,
                quality_report=quality,
                metadata={'indicator': indicator}
            )

        except Exception as e:
            logger.error(f"Fetch failed for {self.source_name}/{indicator}: {e}")
            return FetchResult(
                success=False,
                data=None,
                source=self.source_name,
                fetch_time=fetch_time,
                cache_hit=False,
                quality_report=None,
                error=str(e)
            )

    def list_indicators(self) -> List[str]:
        """
        List available indicators from this source.

        Returns:
            List of indicator names
        """
        return []

    def get_metadata(self, indicator: str) -> Dict[str, Any]:
        """
        Get metadata for an indicator.

        Args:
            indicator: Indicator name

        Returns:
            Dictionary with metadata
        """
        return {
            'source': self.source_name,
            'indicator': indicator,
            'base_url': self.base_url
        }


def print_fetch_result(result: FetchResult):
    """Pretty-print fetch result."""
    print("=" * 60)
    print(f"FETCH RESULT: {result.source}")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Cache hit: {result.cache_hit}")
    print(f"Fetch time: {result.fetch_time}")

    if result.error:
        print(f"Error: {result.error}")

    if result.data is not None:
        print(f"\nData shape: {result.data.shape}")
        print(f"Columns: {list(result.data.columns)}")

    if result.quality_report:
        qr = result.quality_report
        print(f"\nQuality Report:")
        print(f"  Observations: {qr.n_observations}")
        print(f"  Date range: {qr.date_range}")
        print(f"  Missing: {qr.missing_pct:.2f}%")
        print(f"  Gaps: {len(qr.gaps)}")
        print(f"  Outliers: {qr.outliers}")
        print(f"  Passed: {qr.passed}")

        if qr.issues:
            print("  Issues:")
            for issue in qr.issues:
                print(f"    - {issue}")

    print("=" * 60)
