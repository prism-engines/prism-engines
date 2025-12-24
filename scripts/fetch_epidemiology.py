#!/usr/bin/env python3
"""
Fetch Delphi Epidemiological Data and Load into PRISM

This script:
1. Fetches ILI and related data from CMU Delphi Epidata API
2. Aggregates into weekly time series by region
3. Loads into PRISM's DuckDB database
4. Runs basic geometry analysis to validate

Usage:
    python scripts/fetch_epidemiology.py

    # Or with options:
    python scripts/fetch_epidemiology.py --start 2015-01-01 --regions NATIONAL HHS1 HHS2
"""

import sys
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np

from prism.fetch.delphi import DelphiFetcher, REGIONS, SIGNAL_TYPES, get_all_indicator_ids
from prism.db.open import open_prism_db


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_epidemiology_schema(conn: duckdb.DuckDBPyConnection):
    """
    Create epidemiology-specific tables if needed.

    We'll use the same data.indicators table but with source='delphi'.
    Also create a reference table for regions.
    """
    # Create regions reference table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta.epi_regions (
            region_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            description VARCHAR,
            region_code VARCHAR,
            region_type VARCHAR,
            population INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Populate regions
    for name, region in REGIONS.items():
        conn.execute("""
            INSERT OR REPLACE INTO meta.epi_regions
            (region_id, name, description, region_code, region_type, population)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            name, region.name, region.description,
            region.region_code, region.region_type, region.population
        ])

    logger.info(f"Registered {len(REGIONS)} epidemiological regions")


def load_to_duckdb(
    results: list,
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
):
    """
    Load fetched epidemiology data into PRISM's data.indicators table.
    """
    now = datetime.now()
    total_rows = 0

    for result in results:
        if not result.success or result.data is None:
            continue

        df = result.data.copy()
        df["indicator_id"] = result.indicator_id
        df["source"] = "delphi"
        df["frequency"] = "W"  # Weekly
        df["cleaning_method"] = "raw"
        df["fetched_at"] = now
        df["cleaned_at"] = now
        df["run_id"] = run_id

        df = df[[
            "indicator_id", "date", "value", "source", "frequency",
            "cleaning_method", "fetched_at", "cleaned_at", "run_id"
        ]]

        # Drop rows with NULL values
        df = df.dropna(subset=["value"])

        if df.empty:
            continue

        # Remove existing data for this indicator
        conn.execute("""
            DELETE FROM data.indicators
            WHERE indicator_id = ? AND source = 'delphi'
        """, [result.indicator_id])

        # Insert new data
        conn.execute("""
            INSERT INTO data.indicators
            (indicator_id, date, value, source, frequency,
             cleaning_method, fetched_at, cleaned_at, run_id)
            SELECT * FROM df
        """)

        total_rows += len(df)

    logger.info(f"Loaded {total_rows} rows into data.indicators")
    return total_rows


def run_quick_analysis(conn: duckdb.DuckDBPyConnection):
    """
    Run quick analysis to validate the data looks reasonable.
    """
    print("\n" + "=" * 60)
    print("QUICK ANALYSIS: Epidemiology Data Summary")
    print("=" * 60)

    # Count by region
    result = conn.execute("""
        SELECT
            SPLIT_PART(indicator_id, '_ILI', 1) as region,
            COUNT(DISTINCT indicator_id) as indicators,
            MIN(date) as first_date,
            MAX(date) as last_date,
            COUNT(*) as total_obs
        FROM data.indicators
        WHERE source = 'delphi' AND indicator_id LIKE '%_ILI'
        GROUP BY 1
        ORDER BY total_obs DESC
    """).fetchdf()

    print("\nData by Region (ILI):")
    print(result.to_string(index=False))

    # Peak ILI weeks
    result = conn.execute("""
        SELECT
            indicator_id,
            date,
            value as ili_rate
        FROM data.indicators
        WHERE source = 'delphi'
          AND indicator_id LIKE '%_ILI'
          AND value IS NOT NULL
        ORDER BY value DESC
        LIMIT 10
    """).fetchdf()

    print("\n\nTop 10 Peak ILI Weeks:")
    print(result.to_string(index=False))

    # Cross-region correlation preview
    result = conn.execute("""
        WITH weekly_ili AS (
            SELECT
                date,
                MAX(CASE WHEN indicator_id = 'NATIONAL_ILI' THEN value END) as national,
                MAX(CASE WHEN indicator_id = 'HHS1_ILI' THEN value END) as hhs1,
                MAX(CASE WHEN indicator_id = 'HHS4_ILI' THEN value END) as hhs4,
                MAX(CASE WHEN indicator_id = 'HHS9_ILI' THEN value END) as hhs9
            FROM data.indicators
            WHERE source = 'delphi' AND indicator_id LIKE '%_ILI'
            GROUP BY date
        )
        SELECT
            ROUND(CORR(national, hhs1), 3) as nat_hhs1_corr,
            ROUND(CORR(national, hhs4), 3) as nat_hhs4_corr,
            ROUND(CORR(national, hhs9), 3) as nat_hhs9_corr,
            ROUND(CORR(hhs1, hhs4), 3) as hhs1_hhs4_corr,
            ROUND(CORR(hhs4, hhs9), 3) as hhs4_hhs9_corr
        FROM weekly_ili
        WHERE national IS NOT NULL
    """).fetchdf()

    print("\n\nCross-Region Correlations (ILI):")
    print(result.to_string(index=False))
    print("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Delphi epidemiology data for PRISM"
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["NATIONAL", "HHS1", "HHS2", "HHS3", "HHS4", "HHS5",
                 "HHS6", "HHS7", "HHS8", "HHS9", "HHS10"],
        help="Regions to fetch",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (default: today)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip loading to DuckDB (fetch only)",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Also save to CSV file",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = date.today()

    # Validate regions
    for region in args.regions:
        if region not in REGIONS:
            print(f"Error: Unknown region '{region}'")
            print(f"Available: {list(REGIONS.keys())}")
            return 1

    print("=" * 60)
    print("PRISM Delphi Epidemiology Data Fetcher")
    print("=" * 60)
    print(f"Date range:  {start_date} to {end_date}")
    print(f"Regions:     {args.regions}")
    print(f"Signals:     {SIGNAL_TYPES}")
    print(f"Indicators:  {len(args.regions) * len(SIGNAL_TYPES)} total")
    print("=" * 60)

    # Fetch data
    fetcher = DelphiFetcher()
    all_results = []

    for region_name in args.regions:
        print(f"\n[{region_name}] {REGIONS[region_name].description}")
        results = fetcher.fetch_region_all(region_name, start_date, end_date)

        for result in results:
            if result.success:
                print(f"  + {result.indicator_id}: {result.rows} weeks")
                all_results.append(result)
            else:
                print(f"  - {result.indicator_id}: {result.error}")

    if not all_results:
        print("\nNo data fetched!")
        return 1

    # Save to CSV if requested
    if args.csv_output:
        dfs = []
        for result in all_results:
            df = result.data.copy()
            df["indicator_id"] = result.indicator_id
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined[["indicator_id", "date", "value"]]
        combined.to_csv(args.csv_output, index=False)
        print(f"\nSaved to {args.csv_output}")

    # Load to DuckDB
    if not args.skip_load:
        print("\nLoading to DuckDB...")
        run_id = f"delphi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with open_prism_db() as conn:
            # Ensure schema exists
            ensure_epidemiology_schema(conn)

            # Load data
            load_to_duckdb(all_results, conn, run_id)

            # Run quick analysis
            run_quick_analysis(conn)

    print("\n+ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
