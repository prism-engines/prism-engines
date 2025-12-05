"""
PRISM Engine - SQLite Database Layer
------------------------------------
This module provides a clean, modern API for storing and retrieving
indicator time-series across multiple systems (finance, economic, climate, etc.).

The DB path can be overridden using the PRISM_DB environment variable.

Tables are created automatically from prism_schema.sql.
"""

import os
import sqlite3
from pathlib import Path
import pandas as pd


# ============================================================
# DB PATH RESOLUTION
# ============================================================

def get_db_path() -> str:
    """
    Determine where the SQLite database should live.

    Priority:
    1. Environment variable: PRISM_DB
    2. Default location inside project: data/sql/prism.db
    """
    env_path = os.getenv("PRISM_DB")
    if env_path:
        return env_path

    return str(Path(__file__).parent / "prism.db")


# ============================================================
# CONNECTION
# ============================================================

def connect(db_path: str | None = None) -> sqlite3.Connection:
    """
    Connect to the SQLite database, creating directories if needed.
    """
    if db_path is None:
        db_path = get_db_path()

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database(db_path: str | None = None) -> None:
    """
    Create all tables if they don't already exist.
    Loads prism_schema.sql from the same directory.
    """
    if db_path is None:
        db_path = get_db_path()

    schema_path = Path(__file__).parent / "prism_schema.sql"

    if not schema_path.exists():
        print("ERROR: prism_schema.sql not found next to prism_db.py")
        return

    conn = connect(db_path)
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
        print("Database initialized.")
    finally:
        conn.close()


# ============================================================
# GENERIC QUERY WRAPPER
# ============================================================

def query(sql: str, params=None, conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """
    Run a SQL query and return a DataFrame.
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    df = pd.read_sql_query(sql, conn, params=params or {})

    if close_after:
        conn.close()

    return df


# ============================================================
# INDICATOR API
# ============================================================

def add_indicator(name: str, system: str, frequency: str = "daily", conn=None) -> None:
    """
    Register a new indicator in the `indicators` table.
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    conn.execute(
        """
        INSERT OR IGNORE INTO indicators (name, system, frequency)
        VALUES (?, ?, ?)
        """,
        (name, system, frequency),
    )
    conn.commit()

    if close_after:
        conn.close()


def write_dataframe(df: pd.DataFrame, indicator: str, system: str, conn=None) -> None:
    """
    Write full time-series values for an indicator.
    df must contain: [date, value]
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    records = df[["date", "value"]].to_records(index=False)

    conn.executemany(
        """
        INSERT OR REPLACE INTO indicator_values (indicator, system, date, value)
        VALUES (?, ?, ?, ?)
        """,
        [(indicator, system, str(d), float(v)) for d, v in records],
    )
    conn.commit()

    if close_after:
        conn.close()


def load_indicator(indicator: str, system: str | None = None, conn=None) -> pd.DataFrame:
    """
    Load full time series for an indicator.
    """
    sql = """
        SELECT date, value
        FROM indicator_values
        WHERE indicator = ?
    """

    params = [indicator]

    if system:
        sql += " AND system = ?"
        params.append(system)

    df = query(sql + " ORDER BY date ASC", params=params, conn=conn)
    return df


# ============================================================
# EXPORT UTILITIES
# ============================================================

def export_to_csv(table_name: str, output_path: str, conn=None) -> None:
    """
    Export any table to CSV for debugging.
    """
    df = query(f"SELECT * FROM {table_name}", conn=conn)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} rows → {output_path}")


# ============================================================
# MAIN (For Manual Testing Only)
# ============================================================

if __name__ == "__main__":
    print("PRISM Database Module")
    print("=" * 50)
    print(f"Database path: {get_db_path()}\n")

    try:
        conn = connect()
        print("Connection: OK")
        conn.close()
    except Exception as e:
        print(f"Connection: FAILED – {e}")

    print("\nQuick usage example:")
    print("  from data.sql.prism_db import init_database, add_indicator, write_dataframe, load_indicator")
    print("  init_database()")
    print("  add_indicator('SPY', system='finance')")
    print("  write_dataframe(df, 'SPY', system='finance')")
    print("  data = load_indicator('SPY')")

