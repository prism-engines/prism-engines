#!/usr/bin/env python3
"""
PRISM Engine – Full Stooq Fetch Pipeline
----------------------------------------
Loads all tickers from market_registry.json (Stooq only),
fetches price history, writes to indicator_values, and logs fetches.
"""

import json
from pathlib import Path
import logging
from datetime import datetime

from fetch.fetcher_stooq import StooqFetcher
from data.sql.db import (
    add_indicator,
    write_dataframe,
    log_fetch,
    connect
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_all_stooq")

PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "market_registry.json"


def load_registry(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_stooq_instruments(reg: dict):
    return [
        instr for instr in reg.get("instruments", [])
        if instr.get("enabled", True) and instr.get("source", "").lower() == "stooq"
    ]


def run_full_stooq_ingestion():
    logger.info("==============================================")
    logger.info("  PRISM STOOQ FULL INGESTION START")
    logger.info("==============================================\n")

    registry = load_registry(REGISTRY_PATH)
    instruments = get_stooq_instruments(registry)

    logger.info(f"Loaded registry: {len(instruments)} Stooq instruments enabled.\n")

    fetcher = StooqFetcher()
    conn = connect()

    total_rows = 0
    failures = 0

    for instr in instruments:
        key = instr["key"]
        ticker = instr["ticker"]
        system = "market"

        logger.info(f"--- Fetching {key.upper()} ({ticker}) ---")

        try:
            df = fetcher.fetch_single(ticker)
            if df is None or df.empty:
                logger.warning(f"NO DATA for {ticker}")
                log_fetch(key, system, "stooq", 0, status="empty", conn=conn)
                failures += 1
                continue

            ind_id = add_indicator(
                key,
                system,
                frequency="daily",
                source="stooq",
                units="price",
                description=f"{key.upper()} from Stooq",
                conn=conn
            )

            rows_inserted = write_dataframe(df, key, system, conn=conn)
            total_rows += rows_inserted

            log_fetch(
                indicator=key,
                system=system,
                source="stooq",
                rows_fetched=rows_inserted,
                status="success",
                conn=conn
            )

            logger.info(f"Inserted {rows_inserted} rows for {key}\n")

        except Exception as e:
            logger.error(f"ERROR fetching {ticker}: {e}")
            log_fetch(
                indicator=key,
                system=system,
                source="stooq",
                rows_fetched=0,
                status="error",
                error_message=str(e),
                conn=conn
            )
            failures += 1

    conn.close()

    logger.info("\n==============================================")
    logger.info("  STOOQ FULL INGESTION COMPLETE")
    logger.info("==============================================")
    logger.info(f"Total instruments: {len(instruments)}")
    logger.info(f"Total rows inserted: {total_rows}")
    logger.info(f"Failures: {failures}\n")


if __name__ == "__main__":
    run_full_stooq_ingestion()