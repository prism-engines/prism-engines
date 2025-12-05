#!/usr/bin/env python
"""
PRISM Hybrid Market Ingestion
=============================

- Uses Stooq as PRIMARY source
- Falls back to Yahoo *only* for stubborn symbols (VIX, DXY, etc.)
- Writes clean daily close series into indicator_values
- Respects data/registry/market_registry.json

Usage:
    PYTHONPATH=. python fetch/fetch_all_hybrid.py
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from fetch.hybrid_fetcher import HybridFetcher
from data.sql.db import write_dataframe, log_fetch

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logger = logging.getLogger("fetch_all_hybrid")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "market_registry.json"


# --------------------------------------------------------------------
# Registry helpers
# --------------------------------------------------------------------
def load_registry(path: Path = REGISTRY_PATH) -> List[Dict[str, Any]]:
    """
    Load market registry and return instruments list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instruments = data.get("instruments", [])
    logger.info(f"Loaded registry: {len(instruments)} instruments.")
    return instruments


# --------------------------------------------------------------------
# Ingestion core
# --------------------------------------------------------------------
def ingest_instrument(
    fetcher: HybridFetcher,
    instrument: Dict[str, Any],
) -> int:
    """
    Fetch & write a single instrument.

    Registry fields expected:
        key    → indicator name in DB (lowercase)
        ticker → primary vendor ticker (e.g., SPY.US, VIX.US, DXY.US)
        source → 'stooq' (primary) or anything else (ignored here)
    """
    key = instrument["key"]          # e.g., 'spy'
    ticker = instrument["ticker"]    # e.g., 'SPY.US'
    src = instrument.get("source", "stooq")

    logger.info(f"--- Ingesting {key.upper()} ({ticker}) ---")

    try:
        df = fetcher.fetch_single(ticker)
    except Exception as e:
        logger.error(f"[{key}] Fetch FAILED: {e}")
        log_fetch(
            indicator=key,
            system="market",
            source="hybrid",
            rows_fetched=0,
            status="error",
            error_message=str(e),
        )
        return 0

    if df is None or df.empty:
        logger.warning(f"[{key}] No data returned, skipping.")
        log_fetch(
            indicator=key,
            system="market",
            source="hybrid",
            rows_fetched=0,
            status="empty",
            error_message="no data",
        )
        return 0

    # The HybridFetcher already returns standardized DF:
    #   ['date', 'value', 'value_2', 'adjusted_value']
    # Here we just write it to DB.
    rows = write_dataframe(df, indicator=key, system="market")

    log_fetch(
        indicator=key,
        system="market",
        source="hybrid",
        rows_fetched=rows,
        status="success",
        error_message=None,
    )

    logger.info(f"[{key}] Inserted {rows} rows into indicator_values.")
    return rows


def run_full_hybrid_ingestion() -> None:
    """
    Main entry: iterate all registry instruments using HybridFetcher.
    """
    logger.info("=" * 46)
    logger.info("  PRISM HYBRID MARKET INGESTION START")
    logger.info("=" * 46)
    logger.info("Primary: Stooq  |  Fallback: Yahoo\n")

    instruments = load_registry()
    fetcher = HybridFetcher()

    total_rows = 0
    failures = 0

    for inst in instruments:
        rows = ingest_instrument(fetcher, inst)
        if rows == 0:
            failures += 1
        total_rows += rows

    logger.info("\n" + "=" * 46)
    logger.info("  HYBRID MARKET INGESTION COMPLETE")
    logger.info("=" * 46)
    logger.info(f"Total instruments: {len(instruments)}")
    logger.info(f"Total rows inserted: {total_rows}")
    logger.info(f"Failures (0 rows): {failures}\n")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
if __name__ == "__main__":
    run_full_hybrid_ingestion()