"""
PRISM Fetcher Router - Source-Based Routing (Registry v2)
==========================================================

Routes fetch requests to appropriate data sources based on Registry v2 configuration.

Supported sources:
- FRED: Federal Reserve Economic Data (macro indicators)
- Tiingo: Stock/ETF adjusted close prices

API Keys:
- FRED: Set FRED_API_KEY environment variable
- Tiingo: Set TIINGO_API_KEY environment variable

Usage:
    from fetch.fetcher_router import SourceRouter

    router = SourceRouter()

    # Fetch by indicator name (looks up source in registry)
    df = router.fetch("sp500")

    # Fetch by source explicitly
    df = router.fetch_from_source("fred", "SP500")
    df = router.fetch_from_source("tiingo", "SPY")
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Project root for finding registry
PROJECT_ROOT = Path(__file__).parent.parent


class SourceRouter:
    """
    Routes fetch requests to appropriate data source based on Registry v2.

    Each indicator in the registry specifies its source (fred, tiingo).
    The router instantiates the appropriate fetcher and delegates the request.
    """

    def __init__(self):
        """Initialize the source router with lazy-loaded fetchers."""
        self._fred_fetcher = None
        self._tiingo_fetcher = None
        self._registry = None

    @property
    def registry(self) -> Dict[str, Dict[str, Any]]:
        """Lazy-load the indicator registry."""
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    @property
    def fred_fetcher(self):
        """Lazy-load FRED fetcher."""
        if self._fred_fetcher is None:
            from fetch.fetcher_fred import FREDFetcher
            self._fred_fetcher = FREDFetcher()
        return self._fred_fetcher

    @property
    def tiingo_fetcher(self):
        """Lazy-load Tiingo fetcher."""
        if self._tiingo_fetcher is None:
            from fetch.fetcher_tiingo import TiingoFetcher
            self._tiingo_fetcher = TiingoFetcher()
        return self._tiingo_fetcher

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load indicators from Registry v2 (YAML)."""
        registry_path = PROJECT_ROOT / "data" / "registry" / "indicators.yaml"

        if not registry_path.exists():
            logger.error(f"Registry not found: {registry_path}")
            return {}

        with open(registry_path, "r") as f:
            indicators = yaml.safe_load(f)

        # Filter out comments/metadata (any non-dict entries)
        return {k: v for k, v in indicators.items() if isinstance(v, dict)}

    def get_indicator_config(self, indicator_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for an indicator from the registry.

        Args:
            indicator_name: The indicator name (e.g., 'sp500', 'spy')

        Returns:
            Config dict with 'source' and 'source_id', or None if not found
        """
        return self.registry.get(indicator_name.lower())

    def fetch(
        self,
        indicator_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for an indicator by name.

        Looks up the indicator in the registry to determine source and source_id,
        then routes to the appropriate fetcher.

        Args:
            indicator_name: The indicator name from registry (e.g., 'sp500', 'spy')
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with [date, value] columns, or None on failure
        """
        config = self.get_indicator_config(indicator_name)

        if config is None:
            logger.error(f"Indicator not found in registry: {indicator_name}")
            return None

        source = config.get("source")
        source_id = config.get("source_id")

        if not source or not source_id:
            logger.error(f"Invalid registry entry for {indicator_name}: missing source or source_id")
            return None

        return self.fetch_from_source(
            source=source,
            source_id=source_id,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

    def fetch_from_source(
        self,
        source: str,
        source_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data directly from a source.

        Args:
            source: Data source ('fred' or 'tiingo')
            source_id: The ID at the source (e.g., 'SP500' for FRED, 'SPY' for Tiingo)
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with [date, value] columns, or None on failure
        """
        source = source.lower()

        if source == "fred":
            return self._fetch_fred(source_id, start_date, end_date, **kwargs)
        elif source == "tiingo":
            return self._fetch_tiingo(source_id, start_date, end_date, **kwargs)
        else:
            logger.error(f"Unknown source: {source}")
            return None

    def _fetch_fred(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Fetch from FRED."""
        logger.info(f"[Router] FRED -> {series_id}")

        try:
            df = self.fred_fetcher.fetch_single(
                series_id,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )

            if df is not None and not df.empty:
                logger.info(f"[Router] FRED OK -> {series_id} ({len(df)} rows)")
                return df
            else:
                logger.warning(f"[Router] FRED returned no data for {series_id}")
                return None

        except Exception as e:
            logger.error(f"[Router] FRED error for {series_id}: {e}")
            return None

    def _fetch_tiingo(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Fetch from Tiingo."""
        logger.info(f"[Router] Tiingo -> {ticker}")

        try:
            df = self.tiingo_fetcher.fetch_single(
                ticker,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )

            if df is not None and not df.empty:
                logger.info(f"[Router] Tiingo OK -> {ticker} ({len(df)} rows)")
                return df
            else:
                logger.warning(f"[Router] Tiingo returned no data for {ticker}")
                return None

        except Exception as e:
            logger.error(f"[Router] Tiingo error for {ticker}: {e}")
            return None

    def fetch_multiple(
        self,
        indicator_names: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple indicators.

        Args:
            indicator_names: List of indicator names from registry
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary mapping indicator names to DataFrames
        """
        results = {}

        for name in indicator_names:
            df = self.fetch(name, start_date=start_date, end_date=end_date, **kwargs)
            if df is not None:
                results[name] = df

        return results

    def list_indicators_by_source(self) -> Dict[str, List[str]]:
        """
        Group indicators by source.

        Returns:
            Dict mapping source -> list of indicator names
        """
        by_source: Dict[str, List[str]] = {}

        for name, config in self.registry.items():
            source = config.get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(name)

        return by_source

    def test_connections(self) -> Dict[str, bool]:
        """
        Test connections to all data sources.

        Returns:
            Dict mapping source -> connection status
        """
        results = {}

        # Test FRED
        try:
            results["fred"] = self.fred_fetcher.test_connection()
        except Exception as e:
            logger.error(f"FRED connection test failed: {e}")
            results["fred"] = False

        # Test Tiingo
        try:
            results["tiingo"] = self.tiingo_fetcher.test_connection()
        except Exception as e:
            logger.error(f"Tiingo connection test failed: {e}")
            results["tiingo"] = False

        return results


# Backward compatibility - HybridFetcher is now SourceRouter
# This alias allows old code to work while transitioning
HybridFetcher = SourceRouter
