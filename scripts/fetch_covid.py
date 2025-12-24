#!/usr/bin/env python3
"""
Fetch COVID-19 Epidemiology Data and Load into PRISM

This script:
1. Fetches COVID-19 data from Our World in Data
2. Creates daily time series by country/region
3. Loads into PRISM's DuckDB database
4. Runs basic geometry analysis to validate

Expected patterns (volatile, unlike seismology):
- Correlation spikes during wave onsets (all countries get hit together)
- Regime changes with variant emergence (Alpha -> Delta -> Omicron)
- Cluster merging during synchronized global waves
- Stringency index coupling with case trajectories

Usage:
    python scripts/fetch_covid.py

    # With options:
    python scripts/fetch_covid.py --regions USA GBR DEU JPN BRA IND
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

from prism.fetch.owid import (
    OWIDFetcher,
    REGIONS,
    INDICATOR_TYPES,
    CORE_INDICATORS,
    get_all_indicator_ids,
)
from prism.db.open import open_prism_db


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_covid_schema(conn: duckdb.DuckDBPyConnection):
    """
    Create COVID-specific reference tables if needed.
    """
    # Create regions reference table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta.covid_regions (
            region_code VARCHAR PRIMARY KEY,
            region_name VARCHAR,
            population BIGINT,
            owid_location VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Populate regions
    for code, region in REGIONS.items():
        conn.execute("""
            INSERT OR REPLACE INTO meta.covid_regions
            (region_code, region_name, population, owid_location)
            VALUES (?, ?, ?, ?)
        """, [code, region.name, region.population, region.owid_location])

    # Create indicator types reference
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta.covid_indicator_types (
            indicator_type VARCHAR PRIMARY KEY,
            description VARCHAR,
            unit VARCHAR,
            owid_column VARCHAR,
            is_core BOOLEAN
        )
    """)

    for itype, info in INDICATOR_TYPES.items():
        conn.execute("""
            INSERT OR REPLACE INTO meta.covid_indicator_types
            (indicator_type, description, unit, owid_column, is_core)
            VALUES (?, ?, ?, ?, ?)
        """, [itype, info["description"], info["unit"], info["owid_column"],
              itype in CORE_INDICATORS])

    logger.info(f"Registered {len(REGIONS)} regions, {len(INDICATOR_TYPES)} indicator types")


def load_to_duckdb(
    results: list,
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
):
    """
    Load fetched COVID data into PRISM's data.indicators table.
    """
    now = datetime.now()
    total_rows = 0

    for result in results:
        if not result.success or result.data is None:
            continue

        df = result.data.copy()

        # Skip if all NaN
        if df["value"].isna().all():
            logger.warning(f"Skipping {result.indicator_id}: all NaN values")
            continue

        # Drop NaN values
        df = df.dropna(subset=["value"])

        if df.empty:
            continue

        df["indicator_id"] = result.indicator_id
        df["source"] = "owid"
        df["frequency"] = "D"  # Daily
        df["cleaning_method"] = "smoothed"  # OWID provides smoothed data
        df["fetched_at"] = now
        df["cleaned_at"] = now
        df["run_id"] = run_id

        df = df[[
            "indicator_id", "date", "value", "source", "frequency",
            "cleaning_method", "fetched_at", "cleaned_at", "run_id"
        ]]

        # Remove existing data for this indicator
        conn.execute("""
            DELETE FROM data.indicators
            WHERE indicator_id = ? AND source = 'owid'
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
    Run quick analysis to validate the data and show COVID-specific patterns.
    """
    print("\n" + "=" * 70)
    print("QUICK ANALYSIS: COVID-19 Epidemiology Data")
    print("=" * 70)

    # Data summary
    result = conn.execute("""
        SELECT
            SPLIT_PART(indicator_id, '_CASES', 1) as region,
            COUNT(DISTINCT indicator_id) as indicators,
            MIN(date) as first_date,
            MAX(date) as last_date,
            COUNT(*) as total_obs
        FROM data.indicators
        WHERE source = 'owid' AND indicator_id LIKE '%_CASES'
        GROUP BY 1
        ORDER BY region
    """).fetchdf()

    print("\nData by Region (Cases):")
    print(result.to_string(index=False))

    # Find peak death days by country
    result = conn.execute("""
        SELECT
            indicator_id,
            date,
            ROUND(value, 1) as deaths_per_million
        FROM data.indicators
        WHERE source = 'owid'
          AND indicator_id LIKE '%_DEATHS_PC'
          AND value IS NOT NULL
        ORDER BY value DESC
        LIMIT 10
    """).fetchdf()

    print("\n\nTop 10 Death Rate Days (deaths per million):")
    print(result.to_string(index=False))

    # Cross-country correlation (cases)
    result = conn.execute("""
        WITH daily_cases AS (
            SELECT
                date,
                MAX(CASE WHEN indicator_id = 'USA_CASES_PC' THEN value END) as usa,
                MAX(CASE WHEN indicator_id = 'GBR_CASES_PC' THEN value END) as gbr,
                MAX(CASE WHEN indicator_id = 'DEU_CASES_PC' THEN value END) as deu,
                MAX(CASE WHEN indicator_id = 'JPN_CASES_PC' THEN value END) as jpn,
                MAX(CASE WHEN indicator_id = 'BRA_CASES_PC' THEN value END) as bra,
                MAX(CASE WHEN indicator_id = 'IND_CASES_PC' THEN value END) as ind
            FROM data.indicators
            WHERE source = 'owid' AND indicator_id LIKE '%_CASES_PC'
            GROUP BY date
        )
        SELECT
            ROUND(CORR(usa, gbr), 3) as usa_gbr,
            ROUND(CORR(usa, deu), 3) as usa_deu,
            ROUND(CORR(usa, jpn), 3) as usa_jpn,
            ROUND(CORR(gbr, deu), 3) as gbr_deu,
            ROUND(CORR(usa, bra), 3) as usa_bra,
            ROUND(CORR(usa, ind), 3) as usa_ind
        FROM daily_cases
        WHERE usa IS NOT NULL AND gbr IS NOT NULL
    """).fetchdf()

    print("\n\nCross-Country Correlations (cases per million):")
    print(result.to_string(index=False))

    # Domain comparison
    seis_count = conn.execute("""
        SELECT COUNT(*) FROM data.indicators WHERE source = 'usgs'
    """).fetchone()[0]

    if seis_count > 0:
        print("\n\nDomain Comparison (mean |correlation|):")
        print("  Epidemiology: High correlation expected (synchronized pandemic waves)")
        print("  Seismology:   Low correlation expected (independent fault systems)")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch COVID-19 epidemiology data for PRISM"
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["USA", "GBR", "DEU", "FRA", "JPN", "BRA", "IND", "ZAF"],
        help="Regions to fetch",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
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
    parser.add_argument(
        "--all-indicators",
        action="store_true",
        help="Fetch all indicators (not just core set)",
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

    indicator_count = len(args.regions) * (
        len(INDICATOR_TYPES) if args.all_indicators else len(CORE_INDICATORS)
    )

    print("=" * 70)
    print("PRISM COVID-19 Epidemiology Data Fetcher")
    print("=" * 70)
    print(f"Date range:  {start_date} to {end_date}")
    print(f"Regions:     {args.regions}")
    print(f"Indicators:  {indicator_count} total")
    print("=" * 70)

    # Fetch data
    fetcher = OWIDFetcher()
    all_results = []

    for region_code in args.regions:
        print(f"\n[{region_code}] {REGIONS[region_code].name}")
        results = fetcher.fetch_region_all(
            region_code,
            start_date,
            end_date,
            core_only=not args.all_indicators
        )

        for result in results:
            if result.success:
                non_null = result.data["value"].notna().sum()
                pct = 100 * non_null / len(result.data) if len(result.data) > 0 else 0
                print(f"  + {result.indicator_id}: {result.rows} days ({non_null} with data, {pct:.0f}%)")
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
        run_id = f"owid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with open_prism_db() as conn:
            # Ensure schema exists
            ensure_covid_schema(conn)

            # Load data
            load_to_duckdb(all_results, conn, run_id)

            # Run quick analysis
            run_quick_analysis(conn)

    print("\n+ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
