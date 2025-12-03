"""

Migration runner and data helpers for the PRISM database.

This module provides:
    1. Automatic migration management (run pending migrations on startup)
    2. Helper functions for inserting/updating market and economic data
    3. Integration with the core prism_db module

Usage:
    from utils.db_connector import (
        run_pending_migrations,
        insert_market_price,
        insert_market_dividend,
        insert_econ_value,
        upsert_market_prices,
        upsert_econ_values,
    )

    # Run migrations on startup
    run_pending_migrations()

    # Insert single records
    insert_market_price("SPY", "2024-01-15", 450.25, ret=0.005)
    insert_market_dividend("SPY", "2024-03-15", 1.50)
    insert_econ_value("GDP", "2024-01-01", "2024-03-15", 25000.0)

    # Bulk upsert from DataFrame
    upsert_market_prices(df, ticker="SPY")
    upsert_econ_values(df, code="GDP")
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

_logger = logging.getLogger(__name__)

# Default paths
_default_db_path = Path(__file__).parent.parent / "data" / "sql" / "prism.db"
_migrations_dir = Path(__file__).parent.parent / "data" / "sql" / "migrations"

# ISO date format regex
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_path() -> Path:
    """
    Get the database path from environment variable or use default.

    Returns:
        Path to the SQLite database file
    """
    env_path = os.environ.get("PRISM_DB")
    if env_path:
        return Path(env_path)
    return _default_db_path


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """
    Context manager for database connections.

    Args:
        db_path: Optional path to database. If None, uses get_db_path().

    Yields:
        sqlite3.Connection with row factory set to sqlite3.Row
    """
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# DATE VALIDATION
# =============================================================================

def validate_date(date_str: str, field_name: str = "date") -> str:
    """
    Validate and normalize a date string to ISO format (YYYY-MM-DD).

    Args:
        date_str: Date string to validate
        field_name: Name of the field (for error messages)

    Returns:
        Validated date string in ISO format

    Raises:
        ValueError: If date is not valid ISO format
    """
    if not isinstance(date_str, str):
        raise ValueError(f"{field_name} must be a string, got {type(date_str)}")

    if not _ISO_DATE_PATTERN.match(date_str):
        raise ValueError(
            f"{field_name} must be in ISO format (YYYY-MM-DD), got: {date_str}"
        )

    # Validate it's a real date
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid {field_name}: {date_str} - {e}") from e

    return date_str


# =============================================================================
# MIGRATION MANAGEMENT
# =============================================================================

def get_migration_files() -> list[tuple[str, Path]]:
    """
    Get all migration files sorted by version.

    Returns:
        List of (version, path) tuples sorted by version
    """
    if not _migrations_dir.exists():
        return []

    migrations = []
    for f in _migrations_dir.glob("*.sql"):
        # Extract version from filename (e.g., "002" from "002_create_market_tables.sql")
        match = re.match(r"^(\d+)_", f.name)
        if match:
            version = match.group(1)
            migrations.append((version, f))

    return sorted(migrations, key=lambda x: int(x[0]))


def get_applied_migrations(db_path: Optional[Path] = None) -> set[str]:
    """
    Get the set of already-applied migration versions.

    Args:
        db_path: Optional path to database

    Returns:
        Set of applied migration version strings
    """
    with get_connection(db_path) as conn:
        # Check if schema_migrations table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        )
        if cursor.fetchone() is None:
            return set()

        cursor = conn.execute("SELECT version FROM schema_migrations")
        return {row["version"] for row in cursor.fetchall()}


def compute_file_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 checksum
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_migration(
    migration_path: Path,
    db_path: Optional[Path] = None,
    record: bool = True,
) -> None:
    """
    Run a single migration file.

    Args:
        migration_path: Path to the migration SQL file
        db_path: Optional path to database
        record: If True, record the migration in schema_migrations
    """
    version_match = re.match(r"^(\d+)_", migration_path.name)
    version = version_match.group(1) if version_match else migration_path.stem

    _logger.info(f"Running migration {migration_path.name}")

    with get_connection(db_path) as conn:
        with open(migration_path) as f:
            conn.executescript(f.read())

        if record:
            checksum = compute_file_checksum(migration_path)
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_migrations (version, filename, checksum)
                VALUES (?, ?, ?)
                """,
                (version, migration_path.name, checksum),
            )

        conn.commit()

    _logger.info(f"Migration {migration_path.name} completed")


def run_pending_migrations(db_path: Optional[Path] = None) -> list[str]:
    """
    Run all pending migrations.

    This function:
        1. Ensures the schema_migrations table exists
        2. Finds all migration files not yet applied
        3. Runs them in version order

    Args:
        db_path: Optional path to database

    Returns:
        List of migration filenames that were run
    """
    # Ensure migrations directory exists
    if not _migrations_dir.exists():
        _logger.warning(f"Migrations directory not found: {_migrations_dir}")
        return []

    # Ensure database directory exists
    if db_path is None:
        db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # First, run migration 006 to ensure schema_migrations table exists
    # We need to handle this specially since it creates the tracking table
    migration_006 = _migrations_dir / "006_add_metadata_tables.sql"
    if migration_006.exists():
        with get_connection(db_path) as conn:
            # Check if schema_migrations exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            )
            if cursor.fetchone() is None:
                # Run migration 006 first without recording
                _logger.info("Creating schema_migrations table...")
                with open(migration_006) as f:
                    conn.executescript(f.read())
                conn.commit()

    # Get applied migrations
    applied = get_applied_migrations(db_path)

    # Get all migration files
    all_migrations = get_migration_files()

    # Check if this is a fresh database (no indicators table)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='indicators'"
        )
        has_indicators_table = cursor.fetchone() is not None

    # Run pending migrations
    run_migrations = []
    for version, path in all_migrations:
        if version not in applied:
            # Skip migration 001 (panel->system rename) if indicators table doesn't exist
            # This migration is only for upgrading existing databases
            if version == "001" and not has_indicators_table:
                _logger.debug(f"Skipping {path.name} - indicators table does not exist")
                # Mark as applied so it won't run again
                with get_connection(db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO schema_migrations (version, filename)
                        VALUES (?, ?)
                        """,
                        (version, path.name),
                    )
                    conn.commit()
                continue

            run_migration(path, db_path, record=True)
            run_migrations.append(path.name)

    if run_migrations:
        _logger.info(f"Applied {len(run_migrations)} migration(s)")
    else:
        _logger.debug("No pending migrations")

    return run_migrations


def init_database(db_path: Optional[Path] = None) -> None:
    """
    Initialize the database by running all migrations.

    This is a convenience function that ensures:
        1. The database file exists
        2. All migrations are applied

    Args:
        db_path: Optional path to database
    """
    run_pending_migrations(db_path)


# =============================================================================
# MARKET DATA HELPERS
# =============================================================================

def insert_market_price(
    ticker: str,
    date: str,
    price: float,
    *,
    ret: Optional[float] = None,
    price_z: Optional[float] = None,
    price_log: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Insert a single market price record.

    Args:
        ticker: Stock ticker symbol
        date: Date in ISO format (YYYY-MM-DD)
        price: Price value
        ret: Daily return (optional)
        price_z: Normalized price (optional)
        price_log: Log price (optional)
        db_path: Optional path to database

    Returns:
        Row ID of the inserted record

    Raises:
        ValueError: If date is not valid ISO format
    """
    date = validate_date(date, "date")

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO market_prices
                (ticker, date, price, ret, price_z, price_log)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker, date, price, ret, price_z, price_log),
        )
        conn.commit()
        return cursor.lastrowid


def insert_market_dividend(
    ticker: str,
    date: str,
    dividend: float,
    *,
    db_path: Optional[Path] = None,
) -> int:
    """
    Insert a single market dividend record.

    Args:
        ticker: Stock ticker symbol
        date: Date in ISO format (YYYY-MM-DD)
        dividend: Dividend amount
        db_path: Optional path to database

    Returns:
        Row ID of the inserted record
    """
    date = validate_date(date, "date")

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO market_dividends (ticker, date, dividend)
            VALUES (?, ?, ?)
            """,
            (ticker, date, dividend),
        )
        conn.commit()
        return cursor.lastrowid


def insert_market_tri(
    ticker: str,
    date: str,
    tri_value: float,
    *,
    tri_z: Optional[float] = None,
    tri_log: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Insert a single Total Return Index (TRI) record.

    Args:
        ticker: Stock ticker symbol
        date: Date in ISO format (YYYY-MM-DD)
        tri_value: TRI value
        tri_z: Normalized TRI (optional)
        tri_log: Log TRI (optional)
        db_path: Optional path to database

    Returns:
        Row ID of the inserted record
    """
    date = validate_date(date, "date")

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO market_tri
                (ticker, date, tri_value, tri_z, tri_log)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticker, date, tri_value, tri_z, tri_log),
        )
        conn.commit()
        return cursor.lastrowid


def upsert_market_meta(
    ticker: str,
    *,
    first_date: Optional[str] = None,
    last_date: Optional[str] = None,
    source: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """
    Insert or update market metadata for a ticker.

    Args:
        ticker: Stock ticker symbol
        first_date: First available date (optional)
        last_date: Last available date (optional)
        source: Data source (optional)
        notes: Additional notes (optional)
        db_path: Optional path to database
    """
    if first_date:
        first_date = validate_date(first_date, "first_date")
    if last_date:
        last_date = validate_date(last_date, "last_date")

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO market_meta (ticker, first_date, last_date, source, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                first_date = COALESCE(excluded.first_date, first_date),
                last_date = COALESCE(excluded.last_date, last_date),
                source = COALESCE(excluded.source, source),
                notes = COALESCE(excluded.notes, notes),
                updated_at = CURRENT_TIMESTAMP
            """,
            (ticker, first_date, last_date, source, notes),
        )
        conn.commit()


def upsert_market_prices(
    df: pd.DataFrame,
    ticker: str,
    *,
    date_column: str = "date",
    price_column: str = "price",
    ret_column: Optional[str] = "ret",
    price_z_column: Optional[str] = "price_z",
    price_log_column: Optional[str] = "price_log",
    db_path: Optional[Path] = None,
) -> int:
    """
    Bulk upsert market prices from a DataFrame.

    Args:
        df: DataFrame with price data
        ticker: Stock ticker symbol
        date_column: Name of date column
        price_column: Name of price column
        ret_column: Name of return column (optional, None to skip)
        price_z_column: Name of normalized price column (optional)
        price_log_column: Name of log price column (optional)
        db_path: Optional path to database

    Returns:
        Number of rows inserted/updated
    """
    if df.empty:
        return 0

    rows_written = 0
    with get_connection(db_path) as conn:
        for _, row in df.iterrows():
            date_val = pd.to_datetime(row[date_column]).strftime("%Y-%m-%d")
            price_val = float(row[price_column])

            ret_val = float(row[ret_column]) if ret_column and ret_column in df.columns and pd.notna(row.get(ret_column)) else None
            price_z_val = float(row[price_z_column]) if price_z_column and price_z_column in df.columns and pd.notna(row.get(price_z_column)) else None
            price_log_val = float(row[price_log_column]) if price_log_column and price_log_column in df.columns and pd.notna(row.get(price_log_column)) else None

            conn.execute(
                """
                INSERT OR REPLACE INTO market_prices
                    (ticker, date, price, ret, price_z, price_log)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, date_val, price_val, ret_val, price_z_val, price_log_val),
            )
            rows_written += 1

        conn.commit()

    return rows_written


# =============================================================================
# ECONOMIC DATA HELPERS
# =============================================================================

def insert_econ_series(
    code: str,
    *,
    human_name: Optional[str] = None,
    frequency: Optional[str] = None,
    source: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """
    Insert or update an economic series definition.

    Args:
        code: Series code (e.g., "GDP", "UNRATE")
        human_name: Human-readable name
        frequency: Data frequency (daily, monthly, etc.)
        source: Data source (FRED, BLS, etc.)
        notes: Additional notes
        db_path: Optional path to database
    """
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO econ_series (code, human_name, frequency, source, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                human_name = COALESCE(excluded.human_name, human_name),
                frequency = COALESCE(excluded.frequency, frequency),
                source = COALESCE(excluded.source, source),
                notes = COALESCE(excluded.notes, notes),
                updated_at = CURRENT_TIMESTAMP
            """,
            (code, human_name, frequency, source, notes),
        )
        conn.commit()


def insert_econ_value(
    code: str,
    date: str,
    revision_asof: str,
    value_raw: float,
    *,
    value_yoy: Optional[float] = None,
    value_mom: Optional[float] = None,
    value_z: Optional[float] = None,
    value_log: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Insert a single economic value record.

    Args:
        code: Series code
        date: Observation date in ISO format
        revision_asof: Publication/revision date in ISO format
        value_raw: Raw value as published
        value_yoy: Year-over-year change (optional)
        value_mom: Month-over-month change (optional)
        value_z: Normalized value (optional)
        value_log: Log value (optional)
        db_path: Optional path to database

    Returns:
        Row ID of the inserted record
    """
    date = validate_date(date, "date")
    revision_asof = validate_date(revision_asof, "revision_asof")

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO econ_values
                (code, date, revision_asof, value_raw, value_yoy, value_mom, value_z, value_log)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (code, date, revision_asof, value_raw, value_yoy, value_mom, value_z, value_log),
        )
        conn.commit()
        return cursor.lastrowid


def upsert_econ_meta(
    code: str,
    *,
    last_fetched: Optional[str] = None,
    last_revision_asof: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """
    Insert or update economic series metadata.

    Args:
        code: Series code
        last_fetched: Last fetch date
        last_revision_asof: Most recent revision date
        db_path: Optional path to database
    """
    if last_fetched:
        last_fetched = validate_date(last_fetched, "last_fetched")
    if last_revision_asof:
        last_revision_asof = validate_date(last_revision_asof, "last_revision_asof")

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO econ_meta (code, last_fetched, last_revision_asof)
            VALUES (?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                last_fetched = COALESCE(excluded.last_fetched, last_fetched),
                last_revision_asof = COALESCE(excluded.last_revision_asof, last_revision_asof),
                updated_at = CURRENT_TIMESTAMP
            """,
            (code, last_fetched, last_revision_asof),
        )
        conn.commit()


def upsert_econ_values(
    df: pd.DataFrame,
    code: str,
    revision_asof: str,
    *,
    date_column: str = "date",
    value_column: str = "value",
    value_yoy_column: Optional[str] = "value_yoy",
    value_mom_column: Optional[str] = "value_mom",
    value_z_column: Optional[str] = "value_z",
    value_log_column: Optional[str] = "value_log",
    db_path: Optional[Path] = None,
) -> int:
    """
    Bulk upsert economic values from a DataFrame.

    Args:
        df: DataFrame with economic data
        code: Series code
        revision_asof: Publication/revision date for all values
        date_column: Name of date column
        value_column: Name of value column
        value_yoy_column: Name of YoY column (optional)
        value_mom_column: Name of MoM column (optional)
        value_z_column: Name of normalized column (optional)
        value_log_column: Name of log column (optional)
        db_path: Optional path to database

    Returns:
        Number of rows inserted/updated
    """
    if df.empty:
        return 0

    revision_asof = validate_date(revision_asof, "revision_asof")

    rows_written = 0
    with get_connection(db_path) as conn:
        for _, row in df.iterrows():
            date_val = pd.to_datetime(row[date_column]).strftime("%Y-%m-%d")
            value_raw = float(row[value_column])

            value_yoy = float(row[value_yoy_column]) if value_yoy_column and value_yoy_column in df.columns and pd.notna(row.get(value_yoy_column)) else None
            value_mom = float(row[value_mom_column]) if value_mom_column and value_mom_column in df.columns and pd.notna(row.get(value_mom_column)) else None
            value_z = float(row[value_z_column]) if value_z_column and value_z_column in df.columns and pd.notna(row.get(value_z_column)) else None
            value_log = float(row[value_log_column]) if value_log_column and value_log_column in df.columns and pd.notna(row.get(value_log_column)) else None

            conn.execute(
                """
                INSERT OR REPLACE INTO econ_values
                    (code, date, revision_asof, value_raw, value_yoy, value_mom, value_z, value_log)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (code, date_val, revision_asof, value_raw, value_yoy, value_mom, value_z, value_log),
            )
            rows_written += 1

        conn.commit()

    return rows_written


# =============================================================================
# QUERY HELPERS
# =============================================================================

def load_market_prices(
    ticker: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load market prices for a ticker.

    Args:
        ticker: Stock ticker symbol
        start_date: Optional start date (inclusive)
        end_date: Optional end date (inclusive)
        db_path: Optional path to database

    Returns:
        DataFrame with date index and price columns
    """
    with get_connection(db_path) as conn:
        query = """
            SELECT date, price, ret, price_z, price_log
            FROM market_prices
            WHERE ticker = ?
        """
        params = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


def load_econ_values(
    code: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    revision_asof: Optional[str] = None,
    latest_revision: bool = True,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load economic values for a series.

    Args:
        code: Series code
        start_date: Optional start date (inclusive)
        end_date: Optional end date (inclusive)
        revision_asof: Filter to specific revision date
        latest_revision: If True and revision_asof not specified, get latest revision for each date
        db_path: Optional path to database

    Returns:
        DataFrame with date index and value columns
    """
    with get_connection(db_path) as conn:
        if revision_asof:
            query = """
                SELECT date, value_raw, value_yoy, value_mom, value_z, value_log, revision_asof
                FROM econ_values
                WHERE code = ? AND revision_asof = ?
            """
            params = [code, revision_asof]
            use_alias = False
        elif latest_revision:
            # Get the latest revision for each date
            query = """
                SELECT e.date, e.value_raw, e.value_yoy, e.value_mom, e.value_z, e.value_log, e.revision_asof
                FROM econ_values e
                INNER JOIN (
                    SELECT date, MAX(revision_asof) as max_revision
                    FROM econ_values
                    WHERE code = ?
                    GROUP BY date
                ) latest ON e.date = latest.date AND e.revision_asof = latest.max_revision
                WHERE e.code = ?
            """
            params = [code, code]
            use_alias = True
        else:
            query = """
                SELECT date, value_raw, value_yoy, value_mom, value_z, value_log, revision_asof
                FROM econ_values
                WHERE code = ?
            """
            params = [code]
            use_alias = False

        if start_date:
            col_ref = "e.date" if use_alias else "date"
            query += f" AND {col_ref} >= ?"
            params.append(start_date)
        if end_date:
            col_ref = "e.date" if use_alias else "date"
            query += f" AND {col_ref} <= ?"
            params.append(end_date)

        # Use explicit table alias for ORDER BY to avoid ambiguity in joins
        col_ref = "e.date" if use_alias else "date"
        query += f" ORDER BY {col_ref}"

        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


# =============================================================================
# FETCH LOGGING
# =============================================================================

def log_fetch(
    source: str,
    entity: str,
    operation: str,
    status: str,
    *,
    rows_fetched: int = 0,
    rows_inserted: int = 0,
    rows_updated: int = 0,
    error_message: Optional[str] = None,
    started_at: Optional[str] = None,
    duration_ms: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Log a fetch operation.

    Args:
        source: Data source (yahoo, fred, etc.)
        entity: Ticker or series code fetched
        operation: Operation type (fetch, update, backfill)
        status: Status (success, error, partial)
        rows_fetched: Number of rows retrieved
        rows_inserted: Number of rows inserted
        rows_updated: Number of rows updated
        error_message: Error details if status=error
        started_at: Start time (defaults to now)
        duration_ms: Duration in milliseconds
        db_path: Optional path to database

    Returns:
        Row ID of the log entry
    """
    if started_at is None:
        started_at = datetime.now().isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO fetch_log
                (source, entity, operation, status, rows_fetched, rows_inserted,
                 rows_updated, error_message, started_at, completed_at, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (source, entity, operation, status, rows_fetched, rows_inserted,
             rows_updated, error_message, started_at, duration_ms),
        )
        conn.commit()
        return cursor.lastrowid


# =============================================================================
# DATABASE STATISTICS
# =============================================================================

def get_market_stats(db_path: Optional[Path] = None) -> dict:
    """
    Get statistics about market data in the database.

    Returns:
        Dictionary with counts and date ranges
    """
    with get_connection(db_path) as conn:
        stats = {}

        # Check if tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='market_prices'"
        )
        if cursor.fetchone() is None:
            return {"error": "market_prices table not found"}

        # Price stats
        cursor = conn.execute("""
            SELECT
                COUNT(DISTINCT ticker) as tickers,
                COUNT(*) as total_rows,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM market_prices
        """)
        row = cursor.fetchone()
        stats["prices"] = dict(row)

        # Dividend stats
        cursor = conn.execute("""
            SELECT
                COUNT(DISTINCT ticker) as tickers,
                COUNT(*) as total_rows
            FROM market_dividends
        """)
        row = cursor.fetchone()
        stats["dividends"] = dict(row)

        # TRI stats
        cursor = conn.execute("""
            SELECT
                COUNT(DISTINCT ticker) as tickers,
                COUNT(*) as total_rows
            FROM market_tri
        """)
        row = cursor.fetchone()
        stats["tri"] = dict(row)

        return stats


def get_econ_stats(db_path: Optional[Path] = None) -> dict:
    """
    Get statistics about economic data in the database.

    Returns:
        Dictionary with counts and series information
    """
    with get_connection(db_path) as conn:
        stats = {}

        # Check if tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='econ_series'"
        )
        if cursor.fetchone() is None:
            return {"error": "econ_series table not found"}

        # Series count
        cursor = conn.execute("SELECT COUNT(*) as count FROM econ_series")
        stats["series_count"] = cursor.fetchone()["count"]

        # Values stats
        cursor = conn.execute("""
            SELECT
                COUNT(DISTINCT code) as codes,
                COUNT(*) as total_rows,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM econ_values
        """)
        row = cursor.fetchone()
        stats["values"] = dict(row)

        return stats


# =============================================================================
# MODULE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        print("Running migrations...")
        migrations = run_pending_migrations()
        if migrations:
            print(f"Applied: {migrations}")
        else:
            print("No pending migrations")
    else:
        print("PRISM Database Connector")
        print("=" * 50)
        print()
        print("Usage:")
        print("  python -m utils.db_connector migrate  # Run migrations")
        print()
        print("API:")
        print("  from utils.db_connector import (")
        print("      run_pending_migrations,")
        print("      insert_market_price,")
        print("      insert_econ_value,")
        print("      load_market_prices,")
        print("      load_econ_values,")
        print("  )")
