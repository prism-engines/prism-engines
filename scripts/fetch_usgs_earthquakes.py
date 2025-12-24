#!/usr/bin/env python3
"""
Fetch USGS Earthquake Data and Load into PRISM

This script:
1. Fetches earthquake data from USGS API
2. Aggregates into daily time series by region
3. Loads into PRISM's DuckDB database
4. Runs basic geometry analysis to validate

Usage:
    python scripts/fetch_usgs_earthquakes.py

    # Or with options:
    python scripts/fetch_usgs_earthquakes.py --start 2010-01-01 --regions SAN_ANDREAS JAPAN
"""

import sys
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import uuid

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np

from prism.fetch.usgs import USGSFetcher, REGIONS, INDICATOR_TYPES, get_all_indicator_ids
from prism.db.open import open_prism_db


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_seismology_schema(conn: duckdb.DuckDBPyConnection):
    """
    Create seismology-specific tables if needed.

    We'll use the same data.indicators table but with source='usgs'.
    Also create a reference table for regions.
    """
    # Create regions reference table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta.seismic_regions (
            region_id VARCHAR PRIMARY KEY,
            description VARCHAR,
            min_lat DOUBLE,
            max_lat DOUBLE,
            min_lon DOUBLE,
            max_lon DOUBLE,
            min_magnitude DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Populate regions
    for name, region in REGIONS.items():
        conn.execute("""
            INSERT OR REPLACE INTO meta.seismic_regions
            (region_id, description, min_lat, max_lat, min_lon, max_lon, min_magnitude)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            name, region.description,
            region.min_lat, region.max_lat,
            region.min_lon, region.max_lon,
            region.min_magnitude
        ])

    logger.info(f"Registered {len(REGIONS)} seismic regions")


def load_to_duckdb(
    results: list,
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
):
    """
    Load fetched earthquake data into PRISM's data.indicators table.
    """
    now = datetime.now()
    total_rows = 0

    for result in results:
        if not result.success or result.data is None:
            continue

        df = result.data.copy()
        df["indicator_id"] = result.indicator_id
        df["source"] = "usgs"
        df["frequency"] = "D"  # Daily
        df["cleaning_method"] = "aggregated"
        df["fetched_at"] = now
        df["cleaned_at"] = now
        df["run_id"] = run_id

        df = df[[
            "indicator_id", "date", "value", "source", "frequency",
            "cleaning_method", "fetched_at", "cleaned_at", "run_id"
        ]]

        # Drop rows with NULL values (days with no earthquakes in region)
        df = df.dropna(subset=["value"])

        # Remove existing data for this indicator
        conn.execute("""
            DELETE FROM data.indicators
            WHERE indicator_id = ? AND source = 'usgs'
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
    print("QUICK ANALYSIS: Earthquake Data Summary")
    print("=" * 60)

    # Count by region
    result = conn.execute("""
        SELECT
            SPLIT_PART(indicator_id, '_COUNT', 1) as region,
            COUNT(DISTINCT indicator_id) as indicators,
            MIN(date) as first_date,
            MAX(date) as last_date,
            COUNT(*) as total_obs
        FROM data.indicators
        WHERE source = 'usgs' AND indicator_id LIKE '%_COUNT'
        GROUP BY 1
        ORDER BY total_obs DESC
    """).fetchdf()

    print("\nData by Region:")
    print(result.to_string(index=False))

    # Find biggest earthquake days
    result = conn.execute("""
        SELECT
            indicator_id,
            date,
            value as max_magnitude
        FROM data.indicators
        WHERE source = 'usgs'
          AND indicator_id LIKE '%_MAXMAG'
          AND value IS NOT NULL
        ORDER BY value DESC
        LIMIT 10
    """).fetchdf()

    print("\n\nTop 10 Earthquake Days (by max magnitude):")
    print(result.to_string(index=False))

    # Cross-region correlation preview
    result = conn.execute("""
        WITH daily_counts AS (
            SELECT
                date,
                MAX(CASE WHEN indicator_id = 'SAN_ANDREAS_COUNT' THEN value END) as san_andreas,
                MAX(CASE WHEN indicator_id = 'JAPAN_COUNT' THEN value END) as japan,
                MAX(CASE WHEN indicator_id = 'INDONESIA_COUNT' THEN value END) as indonesia,
                MAX(CASE WHEN indicator_id = 'CASCADIA_COUNT' THEN value END) as cascadia
            FROM data.indicators
            WHERE source = 'usgs' AND indicator_id LIKE '%_COUNT'
            GROUP BY date
        )
        SELECT
            CORR(san_andreas, japan) as sa_jp_corr,
            CORR(san_andreas, indonesia) as sa_id_corr,
            CORR(japan, indonesia) as jp_id_corr,
            CORR(san_andreas, cascadia) as sa_casc_corr
        FROM daily_counts
        WHERE san_andreas IS NOT NULL
          AND japan IS NOT NULL
          AND indonesia IS NOT NULL
    """).fetchdf()

    print("\n\nCross-Region Correlations (daily counts):")
    print(result.to_string(index=False))
    print("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch USGS earthquake data for PRISM"
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["SAN_ANDREAS", "CASCADIA", "JAPAN", "INDONESIA", "CHILE", "ALASKA"],
        help="Regions to fetch",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2014-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (default: yesterday)",
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
        end_date = date.today() - timedelta(days=1)

    # Validate regions
    for region in args.regions:
        if region not in REGIONS:
            print(f"Error: Unknown region '{region}'")
            print(f"Available: {list(REGIONS.keys())}")
            return 1

    print("=" * 60)
    print("PRISM USGS Earthquake Data Fetcher")
    print("=" * 60)
    print(f"Date range:  {start_date} to {end_date}")
    print(f"Regions:     {args.regions}")
    print(f"Indicators:  {len(args.regions) * len(INDICATOR_TYPES)} total")
    print("=" * 60)

    # Fetch data
    fetcher = USGSFetcher()
    all_results = []

    for region_name in args.regions:
        print(f"\n[{region_name}] {REGIONS[region_name].description}")
        results = fetcher.fetch_region_all(region_name, start_date, end_date)

        for result in results:
            if result.success:
                non_null = result.data["value"].notna().sum()
                print(f"  ✓ {result.indicator_id}: {result.rows} days ({non_null} with data)")
                all_results.append(result)
            else:
                print(f"  ✗ {result.indicator_id}: {result.error}")

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
        run_id = f"usgs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with open_prism_db() as conn:
            # Ensure schema exists
            ensure_seismology_schema(conn)

            # Load data
            load_to_duckdb(all_results, conn, run_id)

            # Run quick analysis
            run_quick_analysis(conn)

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
