"""
PRISM Engine - Full Data Pipeline Orchestrator (Registry v2)

Domain-agnostic data pipeline using indicators.yaml registry.

This script:
1. Loads indicators from data/registry/indicators.yaml
2. Fetches data from sources (FRED, etc.)
3. Writes to unified indicator_values table (Schema v2)
4. Builds synthetic time series
5. Builds technical indicators
6. Verifies geometry engine inputs

CLI Usage:
    python start/update_all.py
    python start/update_all.py --skip-fetch
    python start/update_all.py --start-date 1998-01-01
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------
# Path setup
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
# Registry v2 - Load from YAML (domain-agnostic)
# ---------------------------------------------------------------------

def load_indicators_yaml() -> Dict[str, Dict[str, Any]]:
    """
    Load indicators from data/registry/indicators.yaml (Registry v2).

    Returns:
        Dict mapping indicator_name -> {source, source_id, ...}
    """
    registry_path = PROJECT_ROOT / "data" / "registry" / "indicators.yaml"

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Registry v2 not found: {registry_path}\n"
            "Please create data/registry/indicators.yaml"
        )

    with open(registry_path, "r") as f:
        indicators = yaml.safe_load(f)

    # Filter out comments/metadata (any non-dict entries)
    return {k: v for k, v in indicators.items() if isinstance(v, dict)}


# ---------------------------------------------------------------------
# Unified Fetcher (Schema v2)
# ---------------------------------------------------------------------

def fetch_all_indicators(
    indicators: Dict[str, Dict[str, Any]],
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """
    Fetch all indicators and write to indicator_values (Schema v2).

    Uses indicator_name as the key (no indicator_id).
    Routes to appropriate source (FRED or Tiingo) based on registry.

    Returns:
        Number of indicators successfully fetched.
    """
    from data.sql.db_connector import add_indicator, write_dataframe
    import pandas as pd

    # Group by source
    fred_indicators = {k: v for k, v in indicators.items() if v.get("source") == "fred"}
    tiingo_indicators = {k: v for k, v in indicators.items() if v.get("source") == "tiingo"}

    count = 0

    # -------------------------------------------------------------------------
    # Fetch FRED indicators
    # -------------------------------------------------------------------------
    if fred_indicators:
        logger.info(f"Fetching {len(fred_indicators)} FRED indicators...")
        try:
            from fetch.fetcher_fred import FREDFetcher
            fetcher = FREDFetcher()

            for name, config in fred_indicators.items():
                source_id = config.get("source_id")
                if not source_id:
                    logger.warning(f"No source_id for {name}, skipping")
                    continue

                try:
                    df = fetcher.fetch_single(source_id, start_date=start_date, end_date=end_date)

                    if df is None or df.empty:
                        logger.warning(f"No data for {name} ({source_id})")
                        continue

                    # Normalize to date/value format
                    df = df.reset_index()
                    if "Date" in df.columns:
                        df = df.rename(columns={"Date": "date"})

                    # Find value column
                    if source_id in df.columns:
                        df = df.rename(columns={source_id: "value"})
                    elif "value" not in df.columns:
                        value_cols = [c for c in df.columns if c not in ["date", "index"]]
                        if value_cols:
                            df = df.rename(columns={value_cols[0]: "value"})

                    df = df[["date", "value"]].dropna()

                    if df.empty:
                        logger.warning(f"All values NaN for {name}")
                        continue

                    # Format dates
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                    # Register indicator (Schema v2)
                    add_indicator(name, source="fred")

                    # Write to indicator_values (Schema v2)
                    rows = write_dataframe(
                        df,
                        table="indicator_values",
                        indicator_name=name,
                        provenance="fred"
                    )

                    count += 1
                    logger.info(f"  {name}: {rows} rows written")

                except Exception as e:
                    logger.error(f"Error fetching {name} ({source_id}): {e}")

        except ImportError:
            logger.warning("FRED fetcher not available")

    # -------------------------------------------------------------------------
    # Fetch Tiingo indicators
    # -------------------------------------------------------------------------
    if tiingo_indicators:
        logger.info(f"Fetching {len(tiingo_indicators)} Tiingo indicators...")
        try:
            from fetch.fetcher_tiingo import TiingoFetcher
            fetcher = TiingoFetcher()

            for name, config in tiingo_indicators.items():
                source_id = config.get("source_id")
                if not source_id:
                    logger.warning(f"No source_id for {name}, skipping")
                    continue

                try:
                    df = fetcher.fetch_single(source_id, start_date=start_date, end_date=end_date)

                    if df is None or df.empty:
                        logger.warning(f"No data for {name} ({source_id})")
                        continue

                    # Tiingo fetcher already returns [date, value] format
                    # Just ensure proper date formatting
                    df = df[["date", "value"]].dropna()

                    if df.empty:
                        logger.warning(f"All values NaN for {name}")
                        continue

                    # Format dates
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                    # Register indicator (Schema v2)
                    add_indicator(name, source="tiingo")

                    # Write to indicator_values (Schema v2)
                    rows = write_dataframe(
                        df,
                        table="indicator_values",
                        indicator_name=name,
                        provenance="tiingo"
                    )

                    count += 1
                    logger.info(f"  {name}: {rows} rows written")

                except Exception as e:
                    logger.error(f"Error fetching {name} ({source_id}): {e}")

        except ImportError:
            logger.warning("Tiingo fetcher not available")

    return count


# ---------------------------------------------------------------------
# Synthetic / Technical builders (Schema v2 compatible)
# ---------------------------------------------------------------------

def run_synthetic_builder(
    indicators: Dict[str, Dict[str, Any]],
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Build synthetic time series from base indicators."""
    try:
        from data.sql.synthetic_pipeline import build_synthetic_timeseries

        logger.info("Building synthetic time series...")
        # Convert to legacy format for compatibility
        legacy_registry = {"indicators": list(indicators.keys())}
        return build_synthetic_timeseries(legacy_registry, conn, start_date, end_date)
    except Exception as e:
        logger.warning(f"Synthetic builder not available or failed: {e}")
        return {}


def run_technical_builder(
    indicators: Dict[str, Dict[str, Any]],
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Build technical indicators."""
    try:
        from engine_core.metrics.sector_technicals import build_technical_indicators

        logger.info("Building technical indicators...")
        legacy_registry = {"indicators": list(indicators.keys())}
        return build_technical_indicators(legacy_registry, conn, start_date, end_date)
    except Exception as e:
        logger.warning(f"Technical builder not available or failed: {e}")
        return {}


def verify_geometry_inputs(conn: sqlite3.Connection) -> int:
    """Verify that geometry engine has required inputs."""
    from engine_core.geometry.inputs import get_geometry_input_series

    logger.info("Verifying geometry engine inputs...")
    inputs = get_geometry_input_series(conn, {})
    return len(inputs)


# ---------------------------------------------------------------------
# Runtime Panel Loader (domain-agnostic)
# ---------------------------------------------------------------------

def load_panel(indicator_names: List[str]) -> "pd.DataFrame":
    """
    Load a panel of indicators at runtime.

    This is the domain-agnostic panel loader - panels are defined
    by the UI at runtime, not in files.

    Args:
        indicator_names: List of indicator names to include

    Returns:
        Wide-format DataFrame with date index and indicator columns
    """
    from data.sql.db_connector import load_all_indicators_wide

    return load_all_indicators_wide(indicator_names)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main() -> int:
    """Main entry point for the full update pipeline."""
    parser = argparse.ArgumentParser(
        description="Full data pipeline update for PRISM Engine (Registry v2)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching; rebuild synthetics/technicals only",
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
    print("PRISM ENGINE - FULL UPDATE PIPELINE (Registry v2)")
    print("=" * 60)
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start date:  {args.start_date}")
    print(f"End date:    {args.end_date or 'today'}")
    print(f"Skip fetch:  {args.skip_fetch}")
    print()

    try:
        # Step 1: Load indicators from YAML (Registry v2)
        step_header(1, "LOAD INDICATORS (REGISTRY V2)")
        indicators = load_indicators_yaml()
        print(f"Indicators loaded:    {len(indicators)}")

        # Group by source for display
        by_source = {}
        for name, config in indicators.items():
            source = config.get("source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1

        for source, count in by_source.items():
            print(f"  {source}: {count}")

        # Step 2: Initialize DB
        step_header(2, "INITIALIZE DATABASE")
        from data.sql.db_connector import init_database
        from data.sql.db_path import get_db_path

        db_path = get_db_path()
        init_database()
        print(f"Database path:        {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Step 3: Fetch data (optional)
        if not args.skip_fetch:
            step_header(3, "FETCH DATA (UNIFIED)")

            fetch_count = fetch_all_indicators(
                indicators=indicators,
                conn=conn,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            print(f"Indicators fetched:   {fetch_count}")
        else:
            step_header(3, "FETCH DATA (SKIPPED)")
            print("Skipping data fetch as requested via --skip-fetch")

        # Step 4: Build synthetic series
        step_header(4, "BUILD SYNTHETIC SERIES")
        synthetic_results = run_synthetic_builder(
            indicators=indicators,
            conn=conn,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        synthetic_ok = sum(1 for v in synthetic_results.values() if v > 0) if synthetic_results else 0
        print(f"Synthetic series OK:  {synthetic_ok}/{len(synthetic_results) if synthetic_results else 0}")

        # Step 5: Build technical indicators
        step_header(5, "BUILD TECHNICAL INDICATORS")
        technical_results = run_technical_builder(
            indicators=indicators,
            conn=conn,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        technical_ok = sum(1 for v in technical_results.values() if v > 0) if technical_results else 0
        print(f"Technical OK:         {technical_ok}/{len(technical_results) if technical_results else 0}")

        # Step 6: Verify geometry inputs
        step_header(6, "VERIFY GEOMETRY INPUTS")
        geometry_count = verify_geometry_inputs(conn)
        print(f"Geometry input series: {geometry_count}")

        # Summary
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"DB path:    {db_path}")
        print()

        # Print indicator summary
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM indicators")
        total_indicators = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM indicator_values")
        total_values = cursor.fetchone()[0]

        print(f"Total indicators:    {total_indicators}")
        print(f"Total data points:   {total_values:,}")

        conn.close()
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
