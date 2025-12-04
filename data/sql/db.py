"""
PRISM Engine - SQLite Database Layer
------------------------------------
Clean, modernized database API for storing and retrieving indicator
time-series across all PRISM systems (market, economic, climate, etc.).

Key features:
- Automatic DB path resolution
- Automatic schema initialization
- Safe pandas DataFrame handling
- Simple read/write API
- Fetch logging for auditing
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd


# ============================================================
# PATH RESOLUTION
# ============================================================

def _get_project_root() -> Path:
    """Return repository root (`prism-engine/`)."""
    return Path(__file__).parent.parent.parent


def get_db_path() -> str:
    """
    Determine the SQLite database file path.

    Order of precedence:
    1. PRISM_DB environment variable
    2. system_registry.json (paths.database.active_db_path)
    3. Default: data/sql/engine.db
    """
    # 1. Environment override
    env_path = os.getenv("PRISM_DB")
    if env_path:
        return os.path.expanduser(env_path)

    # 2. Registry configuration
    root = _get_project_root()
    registry_path = root / "data" / "registry" / "system_registry.json"

    if registry_path.exists():
        try:
            registry = json.load(open(registry_path, "r"))
            paths = registry.get("paths", {})
            db_paths = paths.get("database", {})
            active = paths.get("active_db_path", "default")

            if active in db_paths:
                return os.path.expanduser(db_paths[active])
        except Exception:
            pass

    # 3. Default
    return str(root / "data" / "sql" / "engine.db")


# ============================================================
# CONNECTION
# ============================================================

def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Connect to SQLite database. Ensures directory exists.

    Returns:
        sqlite3.Connection (row_factory = sqlite3.Row)
    """
    if db_path is None:
        db_path = get_db_path()

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database(db_path: Optional[str] = None) -> None:
    """
    Initialize (or migrate) database using schema.sql.
    """
    if db_path is None:
        db_path = get_db_path()

    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema.sql at: {schema_path}")

    conn = connect(db_path)
    try:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        conn.commit()
        print(f"Database initialized at: {db_path}")
    finally:
        conn.close()


init_db = init_database  # backward compatibility alias


# ============================================================
# GENERIC QUERY WRAPPER
# ============================================================

def query(sql: str, params=None, conn=None) -> pd.DataFrame:
    """Execute SQL and return DataFrame."""
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    finally:
        if close_after:
            conn.close()


# ============================================================
# INDICATOR METADATA API
# ============================================================

def add_indicator(
    name: str,
    system: str,
    frequency: str = "daily",
    source: Optional[str] = None,
    units: Optional[str] = None,
    description: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Insert or ignore indicator metadata. Returns the indicator ID.
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO indicators
                (name, system, frequency, source, units, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, system, frequency, source, units, description),
        )
        conn.commit()

        row = conn.execute(
            "SELECT id FROM indicators WHERE name = ?",
            (name,)
        ).fetchone()

        return int(row["id"]) if row else None
    finally:
        if close_after:
            conn.close()


def get_indicator(name: str, conn=None) -> Optional[Dict[str, Any]]:
    """Return metadata dict for a single indicator."""
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        row = conn.execute(
            "SELECT * FROM indicators WHERE name = ?",
            (name,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def list_indicators(system: Optional[str] = None, conn=None) -> List[Dict[str, Any]]:
    """List indicators, optionally filtered by system."""
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        if system:
            cursor = conn.execute(
                "SELECT * FROM indicators WHERE system = ? ORDER BY name",
                (system,),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM indicators ORDER BY system, name"
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if close_after:
            conn.close()


# ============================================================
# DATA WRITE API (fixed for pandas)
# ============================================================

def write_dataframe(
    df: pd.DataFrame,
    indicator: str,
    system: str,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """
    Write rows into indicator_values.

    SAFE FIX APPLIED:
    - Fully avoids ambiguous pandas boolean checks
    - Correctly skips null values
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        # Safe pandas validation
        if df is None or df.empty:
            return 0

        # Required columns
        if "date" not in df.columns:
            raise ValueError("DataFrame must contain 'date'")
        if "value" not in df.columns:
            raise ValueError("DataFrame must contain 'value'")

        rows = 0

        for _, row in df.iterrows():
            date_val = str(row["date"])[:10]
            value = row["value"]

            if pd.isna(value):
                continue

            value_2 = row["value_2"] if "value_2" in row and pd.notna(row["value_2"]) else None
            adjusted = (
                row["adjusted_value"]
                if "adjusted_value" in row and pd.notna(row["adjusted_value"])
                else None
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO indicator_values
                    (indicator, system, date, value, value_2, adjusted_value)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (indicator, system, date_val, float(value),
                 float(value_2) if value_2 else None,
                 float(adjusted) if adjusted else None),
            )
            rows += 1

        conn.commit()
        return rows

    finally:
        if close_after:
            conn.close()


# ============================================================
# DATA READ API
# ============================================================

def load_indicator(
    indicator: str,
    system: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn=None,
) -> pd.DataFrame:
    """Load time series for a single indicator."""
    sql = """
        SELECT date, value, value_2, adjusted_value
        FROM indicator_values
        WHERE indicator = ?
    """
    params = [indicator]

    if system:
        sql += " AND system = ?"
        params.append(system)

    if start_date:
        sql += " AND date >= ?"
        params.append(start_date)

    if end_date:
        sql += " AND date <= ?"
        params.append(end_date)

    sql += " ORDER BY date"

    df = query(sql, tuple(params), conn)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def load_multiple_indicators(indicators: List[str], system=None,
                             start_date=None, end_date=None, conn=None) -> pd.DataFrame:
    """Load several indicators and pivot them wide."""
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        frames = []

        for name in indicators:
            df = load_indicator(name, system, start_date, end_date, conn)
            if not df.empty:
                df = df[["date", "value"]].rename(columns={"value": name})
                frames.append(df.set_index("date"))

        if not frames:
            return pd.DataFrame()

        out = frames[0]
        for f in frames[1:]:
            out = out.join(f, how="outer")

        return out.sort_index()

    finally:
        if close_after:
            conn.close()


# ============================================================
# FETCH LOGGING
# ============================================================

def log_fetch(indicator: str, system: str, source: str,
              rows_fetched: int, status="success",
              error_message=None, conn=None) -> None:
    """Insert fetch audit entry."""
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        conn.execute(
            """
            INSERT INTO fetch_log
                (indicator, system, source, fetch_date,
                 rows_fetched, status, error_message)
            VALUES (?, ?, ?, date('now'), ?, ?, ?)
            """,
            (indicator, system, source, rows_fetched, status, error_message),
        )
        conn.commit()
    finally:
        if close_after:
            conn.close()


# ============================================================
# UTILITIES
# ============================================================

def get_date_range(indicator: str, system=None, conn=None):
    """Return (min_date, max_date) for an indicator."""
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        sql = "SELECT MIN(date), MAX(date) FROM indicator_values WHERE indicator = ?"
        params = [indicator]

        if system:
            sql += " AND system = ?"
            params.append(system)

        row = conn.execute(sql, params).fetchone()
        return (row[0], row[1]) if row else (None, None)

    finally:
        if close_after:
            conn.close()


def export_to_csv(table: str, path: str, conn=None) -> int:
    """Export table to CSV."""
    df = query(f"SELECT * FROM {table}", conn=conn)
    df.to_csv(path, index=False)
    print(f"Exported {len(df)} rows → {path}")
    return len(df)


def get_table_stats(conn=None) -> Dict[str, int]:
    """Return {table_name: row_count}."""
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        stats = {}
        for t in tables:
            stats[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]

        return stats

    finally:
        if close_after:
            conn.close()


# ============================================================
# MAIN (manual debug)
# ============================================================

if __name__ == "__main__":
    print("PRISM Database Module")
    print("=" * 60)
    print(f"DB Path: {get_db_path()}\n")

    conn = connect()
    print("Connection OK\n")

    stats = get_table_stats(conn)
    print("Tables:")
    for t, c in stats.items():
        print(f"  {t}: {c}")

    conn.close()
