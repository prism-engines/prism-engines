"""
FREDFetcher - Fetch economic time series from FRED
==================================================

This module fetches economic indicators from FRED (Federal Reserve Economic Data)
and writes them into the unified PRISM database schema (via prism_db).

Usage:
    from fetch.fetcher_fred import FREDFetcher

    fetcher = FREDFetcher()
    df = fetcher.fetch_single("DGS10")

    results = fetcher.fetch_all()   # loads registry + writes to DB
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import requests

from fetch.fetcher_base import BaseFetcher


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class FREDFetcher(BaseFetcher):
    """Fetcher for FRED economic indicators."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None, checkpoint_dir: Optional[Path] = None):
        super().__init__(checkpoint_dir)
        self.api_key = api_key or os.getenv("FRED_API_KEY") or os.getenv("FRED_API")

        if not self.api_key:
            logger.warning(
                "No FRED API key found. Set FRED_API_KEY environment variable. "
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    # ------------------------------------------------------------
    # VALIDATE RESPONSE
    # ------------------------------------------------------------
    def validate_response(self, data):
        """
        Validate FRED API response structure.

        Expected structure:
        {
            'id': 'GDP',
            'observation_start': '2000-01-01',
            'observation_end': '2023-01-01',
            'observations': [
                {'date': '2000-01-01', 'value': '12345.6'},
                ...
            ]
        }
        """
        if data is None:
            raise ValueError("FRED API returned no data")

        if not isinstance(data, dict):
            raise ValueError("Invalid FRED response: expected dict")

        if "observations" not in data:
            raise ValueError("Invalid FRED response: missing 'observations' array")

        if not isinstance(data["observations"], list):
            raise ValueError("FRED response 'observations' field is not a list")

        return True

    # ------------------------------------------------------------
    # SINGLE SERIES FETCH
    # ------------------------------------------------------------
    def fetch_single(
        self,
        ticker: str,
        start_date: Optional[str] = "1900-01-01",
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:

        if not self.api_key:
            logger.error("Cannot fetch without FRED API key")
            return None

        params = {
            "series_id": ticker,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
        }

        if end_date:
            params["observation_end"] = end_date

        logger.info(f"Fetching FRED series: {ticker}")

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            obs = response.json().get("observations", [])
            if not obs:
                logger.warning(f"FRED returned no data for {ticker}")
                return None

            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df[["date", "value"]]

        except requests.RequestException as e:
            logger.error(f"Network error fetching {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    # ------------------------------------------------------------
    # MULTIPLE SERIES FETCH
    # ------------------------------------------------------------
    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:

        results = {}
        for ticker in tickers:
            df = self.fetch_single(ticker, start_date, end_date)
            if df is not None and not df.empty:
                results[ticker] = df

        return results

    # ------------------------------------------------------------
    # LOAD ECONOMIC REGISTRY
    # ------------------------------------------------------------
    def load_economic_registry(self) -> List[Dict[str, Any]]:
        registry_path = PROJECT_ROOT / "data" / "registry" / "economic_registry.json"

        if not registry_path.exists():
            logger.error(f"Economic registry not found: {registry_path}")
            return []

        with open(registry_path, "r") as f:
            registry = json.load(f)

        return registry.get("series", [])

    # ------------------------------------------------------------
    # FETCH ALL (MAIN ENTRY POINT)
    # ------------------------------------------------------------
    def fetch_all(
        self,
        write_to_db: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:

        series_list = self.load_economic_registry()
        results = {}

        # Try DB
        if write_to_db:
            try:
                from data.sql.prism_db import (
                    init_database,
                    ensure_indicator,
                    write_values,
                    log_fetch
                )
                init_database()
                db_ready = True
            except Exception as e:
                logger.error(f"Database unavailable: {e}")
                db_ready = False
        else:
            db_ready = False

        for cfg in series_list:
            if not cfg.get("enabled", True):
                continue

            key = cfg["key"]
            ticker = cfg.get("ticker", key.upper())

            logger.info(f"Fetching economic series: {key} ({ticker})")

            try:
                df = self.fetch_single(ticker, start_date, end_date)

                if df is None or df.empty:
                    logger.warning(f"No data returned for {key}")

                    if db_ready:
                        log_fetch(
                            indicator=key,
                            system="economic",
                            source="fred",
                            rows_fetched=0,
                            status="error",
                            error_message="No data returned"
                        )
                    continue

                results[key] = df

                if db_ready:
                    ensure_indicator(
                        name=key,
                        system="economic",
                        frequency=cfg.get("frequency", "daily"),
                        source="fred",
                        description=cfg.get("name", "")
                    )

                    rows = write_values(key, "economic", df)

                    log_fetch(
                        indicator=key,
                        system="economic",
                        source="fred",
                        rows_fetched=rows,
                        status="success"
                    )

                    logger.info(f"  -> Wrote {rows} rows for {key}")

            except Exception as e:
                logger.error(f"Error fetching {key}: {e}")

                if db_ready:
                    log_fetch(
                        indicator=key,
                        system="economic",
                        source="fred",
                        rows_fetched=0,
                        status="error",
                        error_message=str(e)
                    )

        logger.info(f"Completed: {len(results)} economic series fetched")
        return results

    # ------------------------------------------------------------
    # CONNECTION TEST
    # ------------------------------------------------------------
    def test_connection(self) -> bool:
        try:
            df = self.fetch_single("DGS10")
            return df is not None and not df.empty
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


FetcherFRED = FREDFetcher
