#!/usr/bin/env python3
"""
PRISM Database Performance Checker
===================================

Purpose: Measure DB read/write performance & schema health.

This script implements:
- Row count per table
- Time to load large datasets (simulated via repeated reads)
- Time to pivot a wide panel
- Index health check (sqlite_master)
- Detect fragmentation

Output:
    DATABASE PERFORMANCE REPORT
    Table row counts
    Load times
    Pivot times
    Index health status
    Fragmentation metrics

Usage:
    python start/performance/check_db_performance.py

Notes:
    - Read-only: never writes to the database
    - Exits gracefully on errors
"""

import sys
import logging
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from panel.runtime_loader import load_panel
from data.sql.db_connector import get_db_path, load_all_indicators_wide, get_connection
from data.family_manager import FamilyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_table_row_counts() -> Dict[str, int]:
    """
    Get row counts for all tables in the database.

    Returns:
        Dict mapping table name to row count
    """
    counts = {}

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get list of tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row['name'] for row in cursor.fetchall()]

        # Count rows in each table
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                row = cursor.fetchone()
                counts[table] = row['cnt'] if row else 0
            except Exception as e:
                logger.warning(f"Could not count rows in {table}: {e}")
                counts[table] = -1

        conn.close()

    except Exception as e:
        logger.error(f"Error getting table row counts: {e}")

    return counts


def time_repeated_loads(iterations: int = 10) -> Dict[str, float]:
    """
    Time repeated data loads to simulate large dataset handling.

    Args:
        iterations: Number of load iterations

    Returns:
        Dict with timing statistics
    """
    results = {
        "iterations": iterations,
        "load_times_ms": [],
        "total_rows_loaded": 0
    }

    try:
        for i in range(iterations):
            start_time = time.perf_counter()

            # Load all data
            df = load_all_indicators_wide()

            end_time = time.perf_counter()
            load_time_ms = (end_time - start_time) * 1000

            results["load_times_ms"].append(load_time_ms)
            if not df.empty:
                results["total_rows_loaded"] += len(df)

        if results["load_times_ms"]:
            results["avg_load_time_ms"] = sum(results["load_times_ms"]) / len(results["load_times_ms"])
            results["min_load_time_ms"] = min(results["load_times_ms"])
            results["max_load_time_ms"] = max(results["load_times_ms"])

    except Exception as e:
        logger.error(f"Error in repeated loads: {e}")
        results["error"] = str(e)

    return results


def time_pivot_operation() -> Dict[str, Any]:
    """
    Time how long it takes to pivot data to wide format.

    Returns:
        Dict with pivot timing information
    """
    results = {
        "raw_load_time_ms": 0,
        "pivot_time_ms": 0,
        "total_time_ms": 0,
        "resulting_shape": None
    }

    try:
        # First load raw data (long format simulation)
        start_time = time.perf_counter()

        conn = get_connection()
        query = """
            SELECT indicator_name, date, value
            FROM indicator_values
            ORDER BY date
        """

        import pandas as pd
        df_long = pd.read_sql_query(query, conn)
        conn.close()

        raw_load_time = time.perf_counter()
        results["raw_load_time_ms"] = (raw_load_time - start_time) * 1000

        if not df_long.empty:
            # Time pivot operation
            pivot_start = time.perf_counter()
            df_wide = df_long.pivot(index='date', columns='indicator_name', values='value')
            pivot_end = time.perf_counter()

            results["pivot_time_ms"] = (pivot_end - pivot_start) * 1000
            results["total_time_ms"] = (pivot_end - start_time) * 1000
            results["resulting_shape"] = df_wide.shape
        else:
            results["note"] = "No data to pivot"

    except Exception as e:
        logger.error(f"Error in pivot operation: {e}")
        results["error"] = str(e)

    return results


def check_index_health() -> Dict[str, Any]:
    """
    Check health of database indexes using sqlite_master.

    Returns:
        Dict with index health information
    """
    results = {
        "indexes": [],
        "missing_recommended_indexes": [],
        "status": "healthy"
    }

    # Recommended indexes for performance
    recommended_indexes = {
        "indicator_values": ["indicator_name", "date"],
        "indicators": ["name", "source"],
        "fetch_log": ["indicator_name", "timestamp"]
    }

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get all indexes
        cursor.execute("""
            SELECT name, tbl_name, sql
            FROM sqlite_master
            WHERE type='index' AND sql IS NOT NULL
        """)

        indexes = cursor.fetchall()

        for idx in indexes:
            results["indexes"].append({
                "name": idx['name'],
                "table": idx['tbl_name'],
                "sql": idx['sql'][:100] if idx['sql'] else None  # Truncate long SQL
            })

        # Check for missing recommended indexes
        existing_index_tables = set(idx['tbl_name'] for idx in indexes)

        for table, columns in recommended_indexes.items():
            # Check if table has any index
            table_indexes = [idx for idx in indexes if idx['tbl_name'] == table]
            if not table_indexes:
                results["missing_recommended_indexes"].append({
                    "table": table,
                    "recommended_columns": columns
                })

        if results["missing_recommended_indexes"]:
            results["status"] = "warning"

        conn.close()

    except Exception as e:
        logger.error(f"Error checking index health: {e}")
        results["error"] = str(e)
        results["status"] = "error"

    return results


def detect_fragmentation() -> Dict[str, Any]:
    """
    Detect database fragmentation using SQLite analysis.

    Returns:
        Dict with fragmentation metrics
    """
    results = {
        "page_count": 0,
        "page_size": 0,
        "total_size_bytes": 0,
        "freelist_count": 0,
        "fragmentation_ratio": 0.0,
        "status": "unknown"
    }

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get database page info
        cursor.execute("PRAGMA page_count")
        row = cursor.fetchone()
        results["page_count"] = row[0] if row else 0

        cursor.execute("PRAGMA page_size")
        row = cursor.fetchone()
        results["page_size"] = row[0] if row else 0

        cursor.execute("PRAGMA freelist_count")
        row = cursor.fetchone()
        results["freelist_count"] = row[0] if row else 0

        # Calculate metrics
        results["total_size_bytes"] = results["page_count"] * results["page_size"]

        if results["page_count"] > 0:
            results["fragmentation_ratio"] = results["freelist_count"] / results["page_count"]

        # Determine status
        if results["fragmentation_ratio"] < 0.05:
            results["status"] = "healthy"
        elif results["fragmentation_ratio"] < 0.20:
            results["status"] = "moderate"
        else:
            results["status"] = "fragmented"

        conn.close()

    except Exception as e:
        logger.error(f"Error detecting fragmentation: {e}")
        results["error"] = str(e)
        results["status"] = "error"

    return results


def print_report(
    row_counts: Dict[str, int],
    load_results: Dict[str, Any],
    pivot_results: Dict[str, Any],
    index_health: Dict[str, Any],
    fragmentation: Dict[str, Any]
) -> None:
    """Print the database performance report."""

    print("\n" + "=" * 60)
    print("DATABASE PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Database: {get_db_path()}")
    print(f"Report time: {datetime.now().isoformat()}")

    # Row counts
    print("\n--- Table Row Counts ---")
    total_rows = 0
    for table, count in sorted(row_counts.items()):
        status = "" if count >= 0 else " (error)"
        print(f"  {table}: {count:,}{status}")
        if count > 0:
            total_rows += count
    print(f"  TOTAL: {total_rows:,}")

    # Load times
    print("\n--- Load Performance ---")
    if "error" not in load_results:
        print(f"  Iterations: {load_results['iterations']}")
        print(f"  Avg load time: {load_results.get('avg_load_time_ms', 0):.2f} ms")
        print(f"  Min load time: {load_results.get('min_load_time_ms', 0):.2f} ms")
        print(f"  Max load time: {load_results.get('max_load_time_ms', 0):.2f} ms")
        print(f"  Total rows loaded: {load_results['total_rows_loaded']:,}")
    else:
        print(f"  Error: {load_results['error']}")

    # Pivot times
    print("\n--- Pivot Performance ---")
    if "error" not in pivot_results:
        print(f"  Raw load time: {pivot_results['raw_load_time_ms']:.2f} ms")
        print(f"  Pivot time: {pivot_results['pivot_time_ms']:.2f} ms")
        print(f"  Total time: {pivot_results['total_time_ms']:.2f} ms")
        if pivot_results['resulting_shape']:
            print(f"  Result shape: {pivot_results['resulting_shape']}")
    else:
        print(f"  Error: {pivot_results.get('error', 'Unknown')}")

    # Index health
    print("\n--- Index Health ---")
    print(f"  Status: {index_health['status'].upper()}")
    print(f"  Total indexes: {len(index_health['indexes'])}")
    if index_health['indexes']:
        for idx in index_health['indexes'][:5]:  # Show first 5
            print(f"    - {idx['name']} on {idx['table']}")
        if len(index_health['indexes']) > 5:
            print(f"    ... and {len(index_health['indexes']) - 5} more")
    if index_health['missing_recommended_indexes']:
        print("  Missing recommended indexes:")
        for missing in index_health['missing_recommended_indexes']:
            print(f"    - {missing['table']}: {missing['recommended_columns']}")

    # Fragmentation
    print("\n--- Fragmentation Analysis ---")
    print(f"  Status: {fragmentation['status'].upper()}")
    print(f"  Page count: {fragmentation['page_count']:,}")
    print(f"  Page size: {fragmentation['page_size']:,} bytes")
    print(f"  Total size: {fragmentation['total_size_bytes'] / (1024*1024):.2f} MB")
    print(f"  Free pages: {fragmentation['freelist_count']:,}")
    print(f"  Fragmentation ratio: {fragmentation['fragmentation_ratio']:.2%}")

    if fragmentation['status'] == 'fragmented':
        print("  Recommendation: Consider running VACUUM")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for database performance check.

    Returns:
        0 on success, 1 on error
    """
    logger.info("Starting database performance check...")

    try:
        # Run all checks
        row_counts = get_table_row_counts()
        load_results = time_repeated_loads(iterations=5)
        pivot_results = time_pivot_operation()
        index_health = check_index_health()
        fragmentation = detect_fragmentation()

        # Print report
        print_report(row_counts, load_results, pivot_results, index_health, fragmentation)

        logger.info("Database performance check completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in database performance check: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
