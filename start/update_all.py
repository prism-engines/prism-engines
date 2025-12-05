"""
PRISM Engine - Full Data Pipeline Orchestrator

This script runs the full data pipeline:

1. Load and validate the metric registry
2. Run all fetchers (Yahoo Finance + FRED) and write to the database
3. Build synthetic time series
4. Build technical indicators
5. Verify geometry engine inputs

CLI Usage:

    # Full update from 2000-01-01 to today
    python start/update_all.py

    # Full update with a custom start date
    python start/update_all.py --start-date 1998-01-01

    # Skip fetching (use existing DB data) but rebuild synthetics/technicals/geometry
    python start/update_all.py --skip-fetch

    # Verbose logging
    python start/update_all.py -v
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------
# Path setup so we can run as: python start/update_all.py
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the update script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def step_header(step_num: int, title: str) -> None:
    """Pretty-print a step header."""
    print()
    print("=" * 60)
    print(f"STEP {step_num}: {title}")
    print("=" * 60)


# ---------------------------------------------------------------------
# Registry loading / validation
# ---------------------------------------------------------------------

def load_and_validate_registry() -> dict:
    """
    Load and validate the metric registry.

    Expects data/registry to expose:
        - load_metric_registry()
        - validate_registry(registry)
    """
    from data.registry import load_metric_registry, validate_registry

    registry = load_metric_registry()
    validate_registry(registry)
    return registry


# ---------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------

def run_yahoo_fetcher(
    registry: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """
    Fetch market data from Yahoo Finance and store it in the database.

    Returns:
        Number of indicators successfully updated.
    """
    from fetch.fetcher_yahoo import YahooFetcher

    logger.info("Fetching market data from Yahoo Finance...")

    # Get market tickers from registry
    market_metrics = registry.get("market", [])
    tickers = []
    for metric in market_metrics:
        ticker = metric.get("ticker") or metric.get("name")
        if ticker:
            tickers.append(ticker)

    if not tickers:
        logger.warning("No market tickers found in registry")
        return 0

    fetcher = YahooFetcher()
    count = 0
    cursor = conn.cursor()

    for ticker in tickers:
        try:
            df = fetcher.fetch_single(ticker, start_date=start_date, end_date=end_date)

            if df is None or df.empty:
                logger.warning(f"No data for {ticker}")
                continue

            # Normalize column names
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})

            # Get value column (close price)
            value_col = ticker.lower()
            if value_col not in df.columns:
                # Try to find any value column
                for col in df.columns:
                    if col not in ["date", "index"]:
                        value_col = col
                        break

            if value_col not in df.columns:
                logger.warning(f"No value column found for {ticker}")
                continue

            indicator_df = df[["date", value_col]].copy()
            indicator_df.columns = ["date", "value"]
            indicator_df = indicator_df.dropna()

            if indicator_df.empty:
                continue

            # Get or create indicator
            name = ticker.lower()
            cursor.execute("SELECT id FROM indicators WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                indicator_id = row[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO indicators (name, system, frequency, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, "finance", "daily", "yahoo"),
                )
                indicator_id = cursor.lastrowid

            # Insert values
            for _, r in indicator_df.iterrows():
                date_val = r["date"]
                if hasattr(date_val, "strftime"):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)[:10]

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
                    VALUES (?, ?, ?)
                    """,
                    (indicator_id, date_str, float(r["value"])),
                )

            conn.commit()
            count += 1
            logger.info(f"  {name}: {len(indicator_df)} rows written")

        except Exception as e:
            logger.error(f"Error writing market series {ticker}: {e}")

    return count


def run_fred_fetcher(
    registry: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """
    Fetch economic data from FRED and store it in the database.

    Returns:
        Number of indicators successfully updated.
    """
    try:
        from fetch.fetcher_fred import FREDFetcher
    except ImportError:
        logger.warning("FRED fetcher not available, skipping economic data")
        return 0

    logger.info("Fetching economic data from FRED...")

    # Economic metrics live under registry["economic"]
    economic_metrics = registry.get("economic", [])

    # Mapping from registry metric name -> FRED series ID
    REGISTRY_TO_FRED = {
        "cpi": "CPIAUCSL",
        "ppi": "PPIACO",
        "unrate": "UNRATE",
        "payems": "PAYEMS",
        "m2": "M2SL",
        "gdp": "GDP",
        "dgs10": "DGS10",
        "dgs2": "DGS2",
        "dgs3mo": "DGS3MO",
        "baaffm": "BAAFFM",
        "baa10ym": "BAA10YM",
    }

    fetcher = FREDFetcher()
    cursor = conn.cursor()
    count = 0

    for metric in economic_metrics:
        name = metric.get("name", "").lower()
        freq = metric.get("frequency", "monthly")

        fred_id = REGISTRY_TO_FRED.get(name)
        if not fred_id:
            logger.warning(f"No FRED mapping for metric '{name}'")
            continue

        try:
            df = fetcher.fetch_single(
                fred_id,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                logger.warning(f"No data for {name} ({fred_id})")
                continue

            df = df.reset_index()

            # Normalize columns
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            if fred_id in df.columns:
                df = df.rename(columns={fred_id: "value"})
            elif "value" not in df.columns:
                # Take the first non-date column as value
                value_col = [c for c in df.columns if c != "date"][0]
                df = df.rename(columns={value_col: "value"})

            df = df[["date", "value"]].dropna()
            if df.empty:
                logger.warning(f"All values NaN for {name} ({fred_id})")
                continue

            # Get or create indicator row
            cursor.execute("SELECT id FROM indicators WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                indicator_id = row[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO indicators (name, system, frequency, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, "finance", freq, "fred"),
                )
                indicator_id = cursor.lastrowid

            # Insert values
            for _, r in df.iterrows():
                date_val = r["date"]
                if hasattr(date_val, "strftime"):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)[:10]

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
                    VALUES (?, ?, ?)
                    """,
                    (indicator_id, date_str, float(r["value"])),
                )

            conn.commit()
            count += 1
            logger.info(f"  {name}: {len(df)} rows written")

        except Exception as e:
            logger.error(f"Error fetching/writing economic series {name}: {e}")

    return count


# ---------------------------------------------------------------------
# Synthetic / technical builders and geometry verification
# ---------------------------------------------------------------------

def run_synthetic_builder(
    registry: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Build synthetic time series from base indicators.

    Note: repo-cleanup's build_synthetic_timeseries takes (reg, conn, start, end)
    """
    from data.sql.synthetic_pipeline import build_synthetic_timeseries

    logger.info("Building synthetic time series...")
    return build_synthetic_timeseries(registry, conn, start_date, end_date)


def run_technical_builder(
    registry: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Build technical indicators for sectors and other assets.
    """
    from engine_core.metrics.sector_technicals import build_technical_indicators

    logger.info("Building technical indicators...")
    return build_technical_indicators(registry, conn, start_date, end_date)


def verify_geometry_inputs(
    registry: dict,
    conn: sqlite3.Connection,
) -> int:
    """
    Verify that the geometry engine has the required input series available.
    """
    from engine_core.geometry.inputs import get_geometry_input_series

    logger.info("Verifying geometry engine inputs...")
    inputs = get_geometry_input_series(conn, registry)
    return len(inputs)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main() -> int:
    """Main entry point for the full update pipeline."""
    parser = argparse.ArgumentParser(
        description="Full data pipeline update for PRISM Engine",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip market/economic data fetching; rebuild synthetics/technicals only",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2000-01-01",
        help="Start date for data (YYYY-MM-DD, default: 2000-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging output",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    print()
    print("=" * 60)
    print("PRISM ENGINE - FULL UPDATE PIPELINE")
    print("=" * 60)
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start date:  {args.start_date}")
    print(f"End date:    {args.end_date or 'today'}")
    print(f"Skip fetch:  {args.skip_fetch}")
    print()

    try:
        # Step 1: Load and validate registry
        step_header(1, "LOAD AND VALIDATE REGISTRY")
        registry = load_and_validate_registry()
        print(f"Registry version:     {registry.get('version')}")
        print(f"Market metrics:       {len(registry.get('market', []))}")
        print(f"Economic metrics:     {len(registry.get('economic', []))}")
        print(f"Synthetic metrics:    {len(registry.get('synthetic', []))}")
        print(f"Technical metrics:    {len(registry.get('technical', []))}")

        # Step 2: Initialize DB
        step_header(2, "INITIALIZE DATABASE")
        from data.sql.prism_db import get_db_path, initialize_db

        db_path = get_db_path()
        initialize_db()
        print(f"Database path:        {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Step 3: Fetch data (optional)
        if not args.skip_fetch:
            step_header(3, "FETCH DATA (YAHOO + FRED)")

            yahoo_count = run_yahoo_fetcher(
                registry=registry,
                conn=conn,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            print(f"Yahoo Finance:        {yahoo_count} indicators updated")

            fred_count = run_fred_fetcher(
                registry=registry,
                conn=conn,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            print(f"FRED:                 {fred_count} indicators updated")
        else:
            step_header(3, "FETCH DATA (SKIPPED)")
            print("Skipping data fetch as requested via --skip-fetch")

        # Step 4: Build synthetic series
        step_header(4, "BUILD SYNTHETIC SERIES")
        synthetic_results = run_synthetic_builder(
            registry=registry,
            conn=conn,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        synthetic_ok = sum(1 for v in synthetic_results.values() if v > 0)
        print(f"Synthetic series OK:  {synthetic_ok}/{len(synthetic_results)}")

        # Step 5: Build technical indicators
        step_header(5, "BUILD TECHNICAL INDICATORS")
        technical_results = run_technical_builder(
            registry=registry,
            conn=conn,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        technical_ok = sum(1 for v in technical_results.values() if v > 0)
        print(f"Technical OK:         {technical_ok}/{len(technical_results)}")

        # Step 6: Verify geometry inputs
        step_header(6, "VERIFY GEOMETRY INPUTS")
        geometry_count = verify_geometry_inputs(registry, conn)
        print(f"Geometry input series: {geometry_count}")

        # Summary
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"DB path:    {db_path}")

        conn.close()
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
