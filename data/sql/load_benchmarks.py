"""
PRISM Benchmark Dataset Loader
==============================

Loads benchmark CSV datasets into the PRISM SQLite database for engine-validation testing.

This module:
  - Reads benchmark CSV files from data/benchmark/
  - Creates indicator entries with system='benchmark'
  - Inserts time series data into timeseries table
  - Does NOT modify existing FRED or other indicators

Usage:
    from data.sql.load_benchmarks import load_all_benchmarks
    load_all_benchmarks()

Or via CLI:
    python prism_run.py --load-benchmarks
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

from .db_connector import get_connection


# =============================================================================
# CONFIGURATION
# =============================================================================

# Mapping from benchmark filename to shortname
BENCHMARK_FILES = {
    "benchmark_01_clear_leader.csv": "clear_leader",
    "benchmark_02_two_regimes.csv": "two_regimes",
    "benchmark_03_clusters.csv": "clusters",
    "benchmark_04_periodic.csv": "periodic",
    "benchmark_05_anomalies.csv": "anomalies",
    "benchmark_06_pure_noise.csv": "pure_noise",
}


def _get_benchmark_dir() -> Path:
    """Return the default benchmark directory path."""
    return Path(__file__).parent.parent / "benchmark"


# =============================================================================
# INDICATOR CREATION
# =============================================================================

def _create_benchmark_indicator(
    conn: sqlite3.Connection,
    indicator_name: str,
    file_shortname: str,
    column_name: str
) -> str:
    """
    Create an indicator entry for a benchmark column.

    Uses Schema v2 - indicator_values table.

    Args:
        conn: Database connection
        indicator_name: Unique indicator name (e.g., "clear_leader_A")
        file_shortname: Benchmark file shortname (e.g., "clear_leader")
        column_name: Column name from CSV (e.g., "A")

    Returns:
        The indicator name for indicator_values insertion
    """
    description = f"Benchmark synthetic dataset - known structure ({file_shortname})"
    metadata = f'{{"benchmark_group": "{file_shortname}", "column": "{column_name}"}}'

    conn.execute(
        """
        INSERT INTO indicators (name, system, frequency, source, description, metadata)
        VALUES (?, 'benchmark', 'daily', 'BENCHMARK', ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            system = 'benchmark',
            frequency = 'daily',
            source = 'BENCHMARK',
            description = excluded.description,
            metadata = excluded.metadata,
            updated_at = CURRENT_TIMESTAMP
        """,
        (indicator_name, description, metadata)
    )

    return indicator_name


# =============================================================================
# INDICATOR_VALUES INSERTION (Schema v2)
# =============================================================================

def _insert_benchmark_values(
    conn: sqlite3.Connection,
    indicator_name: str,
    dates: List[str],
    values: List[float]
) -> int:
    """
    Insert time series data for a benchmark indicator into indicator_values.

    Uses Schema v2 - indicator_values table.

    Args:
        conn: Database connection
        indicator_name: Indicator name from indicators table
        dates: List of dates (YYYY-MM-DD format)
        values: List of values

    Returns:
        Number of rows inserted
    """
    rows_inserted = 0

    for date, value in zip(dates, values):
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO indicator_values
                    (indicator_name, date, value, provenance, quality_flag)
                VALUES (?, ?, ?, 'BENCHMARK', 'synthetic')
                """,
                (indicator_name, date, value)
            )
            rows_inserted += 1
        except sqlite3.IntegrityError:
            pass  # Skip duplicates

    return rows_inserted


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_benchmark_file(
    conn: sqlite3.Connection,
    filepath: Path,
    file_shortname: str,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Load a single benchmark CSV file into the database.

    Uses Schema v2 - indicator_values table.

    Args:
        conn: Database connection
        filepath: Path to the CSV file
        file_shortname: Shortname for the benchmark (e.g., "clear_leader")
        verbose: Print progress messages

    Returns:
        Tuple of (indicators_created, rows_inserted)
    """
    if verbose:
        print(f"  Loading {filepath.name}...")

    # Read CSV file
    df = pd.read_csv(filepath, index_col=0)

    # Ensure index is parsed as dates and formatted as YYYY-MM-DD
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    dates = df.index.tolist()

    indicators_created = 0
    total_rows = 0

    # Process each column (A, B, C, D, E, F, etc.)
    for column in df.columns:
        # Create indicator name: "{file_shortname}_{column}"
        indicator_name = f"{file_shortname}_{column}"

        # Create indicator entry in indicators table
        _create_benchmark_indicator(conn, indicator_name, file_shortname, column)
        indicators_created += 1

        # Insert time series data into indicator_values
        values = df[column].tolist()
        rows = _insert_benchmark_values(conn, indicator_name, dates, values)
        total_rows += rows

        if verbose:
            print(f"    + {indicator_name}: {rows} rows")

    return indicators_created, total_rows


def load_all_benchmarks(
    benchmark_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Loads all 6 benchmark CSV files into the PRISM SQLite database.

    Overwrites no existing FRED indicators.
    Creates indicators under system='benchmark'.

    Args:
        benchmark_dir: Path to benchmark directory (default: data/benchmark)
        verbose: Print progress messages

    Returns:
        Dictionary with loading statistics
    """
    if benchmark_dir is None:
        benchmark_path = _get_benchmark_dir()
    else:
        benchmark_path = Path(benchmark_dir)

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_path}")

    if verbose:
        print("=" * 60)
        print("PRISM Benchmark Loader")
        print("=" * 60)
        print(f"Source: {benchmark_path}")
        print()

    # Ensure 'benchmark' system exists (Schema v2: systems.system is primary key)
    conn = get_connection()
    conn.execute("INSERT OR IGNORE INTO systems(system) VALUES ('benchmark')")

    stats = {
        "files_loaded": 0,
        "indicators_created": 0,
        "rows_inserted": 0,
        "errors": [],
    }

    # Load each benchmark file in a single transaction per file
    for filename, shortname in BENCHMARK_FILES.items():
        filepath = benchmark_path / filename

        if not filepath.exists():
            error_msg = f"Missing file: {filename}"
            stats["errors"].append(error_msg)
            if verbose:
                print(f"  [SKIP] {error_msg}")
            continue

        try:
            # Start transaction for this file
            indicators, rows = load_benchmark_file(conn, filepath, shortname, verbose)
            conn.commit()

            stats["files_loaded"] += 1
            stats["indicators_created"] += indicators
            stats["rows_inserted"] += rows

        except Exception as e:
            conn.rollback()
            error_msg = f"Error loading {filename}: {e}"
            stats["errors"].append(error_msg)
            if verbose:
                print(f"  [ERROR] {error_msg}")

    conn.close()

    if verbose:
        print()
        print("=" * 60)
        print("BENCHMARK LOADING COMPLETE")
        print("=" * 60)
        print(f"  Files loaded:      {stats['files_loaded']}")
        print(f"  Indicators created: {stats['indicators_created']}")
        print(f"  Rows inserted:     {stats['rows_inserted']}")
        if stats["errors"]:
            print(f"  Errors:            {len(stats['errors'])}")
            for err in stats["errors"]:
                print(f"    - {err}")
        print("=" * 60)

    return stats


def clear_benchmarks(verbose: bool = True) -> Dict[str, int]:
    """
    Remove all benchmark indicators and their data from the database.

    This is useful for resetting benchmark data before re-loading.
    Uses Schema v2 - indicator_values table.

    Args:
        verbose: Print progress messages

    Returns:
        Dictionary with deletion statistics
    """
    conn = get_connection()

    stats = {
        "indicators_deleted": 0,
        "rows_deleted": 0,
    }

    # Get list of benchmark indicators (Schema v2: 'name' column, not 'indicator_name')
    rows = conn.execute(
        "SELECT name FROM indicators WHERE system = 'benchmark'"
    ).fetchall()

    if not rows:
        if verbose:
            print("No benchmark indicators found to delete.")
        conn.close()
        return stats

    indicator_names = [r[0] for r in rows]

    if verbose:
        print(f"Deleting {len(indicator_names)} benchmark indicators...")

    # Delete time series data first from indicator_values (Schema v2)
    for name in indicator_names:
        result = conn.execute(
            "DELETE FROM indicator_values WHERE indicator_name = ?",
            (name,)
        )
        stats["rows_deleted"] += result.rowcount

    # Delete indicators
    result = conn.execute(
        "DELETE FROM indicators WHERE system = 'benchmark'"
    )
    stats["indicators_deleted"] = result.rowcount

    conn.commit()
    conn.close()

    if verbose:
        print(f"  Indicators deleted: {stats['indicators_deleted']}")
        print(f"  Rows deleted:       {stats['rows_deleted']}")

    return stats


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--clear" in sys.argv:
        clear_benchmarks()
    else:
        load_all_benchmarks()
