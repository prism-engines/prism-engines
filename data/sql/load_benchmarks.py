"""
PRISM Benchmark Dataset Loader
==============================

Loads benchmark CSV datasets into the PRISM SQLite database for engine-validation testing.

Works with Schema v2 (indicator_id + indicator_values table).
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

from .db_connector import get_connection

# =============================================================================
# CONFIGURATION
# =============================================================================

BENCHMARK_FILES = {
    "benchmark_01_clear_leader.csv": "clear_leader",
    "benchmark_02_two_regimes.csv": "two_regimes",
    "benchmark_03_clusters.csv": "clusters",
    "benchmark_04_periodic.csv": "periodic",
    "benchmark_05_anomalies.csv": "anomalies",
    "benchmark_06_pure_noise.csv": "pure_noise",
}

def _get_benchmark_dir() -> Path:
    return Path(__file__).parent.parent / "benchmark"

# =============================================================================
# INDICATOR CREATION
# =============================================================================

def _create_benchmark_indicator(conn, indicator_name, file_shortname, column_name) -> int:
    """
    Create or update an indicator for a benchmark column.
    Returns indicator_id (int).
    """

    description = f"Benchmark synthetic dataset - known structure ({file_shortname})"
    metadata = f'{{"benchmark_group": "{file_shortname}", "column": "{column_name}"}}'

    conn.execute(
        """
        INSERT INTO indicators (name, system, frequency, source, description, metadata)
        VALUES (?, 'benchmark', 'daily', 'BENCHMARK', ?, ?)
        ON CONFLICT(name, system) DO UPDATE SET
            frequency='daily',
            source='BENCHMARK',
            description=excluded.description,
            metadata=excluded.metadata
        """,
        (indicator_name, description, metadata),
    )

    row = conn.execute(
        "SELECT id FROM indicators WHERE name=? AND system='benchmark'",
        (indicator_name,),
    ).fetchone()

    return row["id"]

# =============================================================================
# VALUE INSERTION (Schema v2)
# =============================================================================

def _insert_benchmark_values(conn, indicator_id: int, dates: List[str], values: List[float]) -> int:
    rows_inserted = 0

    for date, value in zip(dates, values):
        conn.execute(
            """
            INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
            VALUES (?, ?, ?)
            """,
            (indicator_id, date, value),
        )
        rows_inserted += 1

    return rows_inserted

# =============================================================================
# MAIN LOADER
# =============================================================================

def load_benchmark_file(conn, filepath: Path, file_shortname: str, verbose=True) -> Tuple[int, int]:

    if verbose:
        print(f"  Loading {filepath.name}...")

    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    dates = df.index.tolist()

    indicators_created = 0
    total_rows = 0

    for column in df.columns:
        indicator_name = f"{file_shortname}_{column}"

        indicator_id = _create_benchmark_indicator(conn, indicator_name, file_shortname, column)
        indicators_created += 1

        values = df[column].tolist()
        rows = _insert_benchmark_values(conn, indicator_id, dates, values)
        total_rows += rows

        if verbose:
            print(f"    + {indicator_name}: {rows} rows")

    return indicators_created, total_rows

def load_all_benchmarks(benchmark_dir=None, verbose=True) -> Dict[str, any]:

    benchmark_path = Path(benchmark_dir) if benchmark_dir else _get_benchmark_dir()

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_path}")

    if verbose:
        print("=" * 60)
        print("PRISM Benchmark Loader")
        print("=" * 60)
        print(f"Source: {benchmark_path}\n")

    conn = get_connection()
    conn.execute("INSERT OR IGNORE INTO systems(system) VALUES ('benchmark')")

    stats = {"files_loaded": 0, "indicators_created": 0, "rows_inserted": 0, "errors": []}

    for filename, shortname in BENCHMARK_FILES.items():
        filepath = benchmark_path / filename

        if not filepath.exists():
            msg = f"Missing file: {filename}"
            stats["errors"].append(msg)
            if verbose:
                print(f"  [SKIP] {msg}")
            continue

        try:
            indicators, rows = load_benchmark_file(conn, filepath, shortname, verbose)
            conn.commit()

            stats["files_loaded"] += 1
            stats["indicators_created"] += indicators
            stats["rows_inserted"] += rows

        except Exception as e:
            conn.rollback()
            msg = f"Error loading {filename}: {e}"
            stats["errors"].append(msg)
            if verbose:
                print(f"  [ERROR] {msg}")

    conn.close()

    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK LOADING COMPLETE")
        print("=" * 60)
        print(f"  Files loaded:      {stats['files_loaded']}")
        print(f"  Indicators created: {stats['indicators_created']}")
        print(f"  Rows inserted:     {stats['rows_inserted']}")
        if stats["errors"]:
            print(f"  Errors: {len(stats['errors'])}")
            for e in stats["errors"]:
                print(f"    - {e}")
        print("=" * 60)

    return stats

# =============================================================================
# CLEAR BENCHMARKS
# =============================================================================

def clear_benchmarks(verbose=True) -> Dict[str, int]:
    conn = get_connection()

    rows = conn.execute("SELECT id FROM indicators WHERE system='benchmark'").fetchall()
    indicator_ids = [r["id"] for r in rows]

    stats = {"indicators_deleted": len(indicator_ids), "rows_deleted": 0}

    for ind_id in indicator_ids:
        result = conn.execute("DELETE FROM indicator_values WHERE indicator_id=?", (ind_id,))
        stats["rows_deleted"] += result.rowcount

    conn.execute("DELETE FROM indicators WHERE system='benchmark'")
    conn.commit()
    conn.close()

    if verbose:
        print(f"Deleted {stats['indicators_deleted']} indicators and {stats['rows_deleted']} rows.")

    return stats


if __name__ == "__main__":
    import sys
    if "--clear" in sys.argv:
        clear_benchmarks()
    else:
        load_all_benchmarks()
