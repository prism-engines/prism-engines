"""
SQL Data Manager - Core SQLite interface for PRISM data operations
===================================================================

Provides SQLite storage for all data ingestion, cleaning, and output operations.
Replaces CSV-based file I/O with SQL-based storage for improved querying,
data integrity, and workflow management.

Schema Overview:
    - raw_data: Original fetched data from sources (Yahoo, FRED, etc.)
    - cleaned_data: Processed data after cleaning pipeline
    - metadata: Source info, fetch timestamps, data quality metrics
    - panels: Consolidated panel data for analysis

Usage:
    from data.sql.sql_data_manager import SQLDataManager

    db = SQLDataManager('data/sql/prism.db')
    db.init_schema()

    # Store raw Yahoo data
    db.store_raw_data(df, source='yahoo', ticker='SPY')

    # Store cleaned data
    db.store_cleaned_data(df, source='yahoo', ticker='SPY')

    # Load all cleaned data as panel
    panel = db.load_cleaned_panel()

    # Query specific indicator
    spy = db.query_indicator('spy')
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

# Get directory where this script lives
_SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DB_PATH = _SCRIPT_DIR / 'prism.db'


# =============================================================================
# SCHEMA DEFINITION
# =============================================================================

SCHEMA = """
-- ============================================================================
-- RAW DATA TABLES
-- ============================================================================

-- Raw time series data as fetched from sources
CREATE TABLE IF NOT EXISTS raw_data (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'yahoo', 'fred', 'climate', 'custom'
    value REAL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, ticker, source)
);

-- Metadata about data sources and fetch operations
CREATE TABLE IF NOT EXISTS data_sources (
    id INTEGER PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    name TEXT,
    category TEXT,  -- 'equity', 'macro', 'rates', 'commodity', etc.
    description TEXT,
    first_date TEXT,
    last_date TEXT,
    row_count INTEGER,
    last_fetched TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CLEANED DATA TABLES
-- ============================================================================

-- Cleaned time series data after processing
CREATE TABLE IF NOT EXISTS cleaned_data (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    value REAL NOT NULL,
    cleaning_method TEXT,  -- 'ffill', 'interpolate', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, ticker)
);

-- Data quality metrics after cleaning
CREATE TABLE IF NOT EXISTS data_quality (
    id INTEGER PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    source_rows INTEGER,
    cleaned_rows INTEGER,
    nan_count_before INTEGER,
    nan_count_after INTEGER,
    nan_pct_before REAL,
    nan_pct_after REAL,
    outliers_detected INTEGER,
    outliers_handled INTEGER,
    cleaning_method TEXT,
    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PANEL DATA TABLES
-- ============================================================================

-- Consolidated panel metadata
CREATE TABLE IF NOT EXISTS panels (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    start_date TEXT,
    end_date TEXT,
    n_indicators INTEGER,
    n_rows INTEGER,
    indicators TEXT,  -- JSON array of indicator names
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Panel data in long format (efficient for querying)
CREATE TABLE IF NOT EXISTS panel_data (
    id INTEGER PRIMARY KEY,
    panel_id INTEGER REFERENCES panels(id),
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    value REAL NOT NULL,
    UNIQUE(panel_id, date, ticker)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_raw_data_ticker ON raw_data(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_data_date ON raw_data(date);
CREATE INDEX IF NOT EXISTS idx_raw_data_source ON raw_data(source);
CREATE INDEX IF NOT EXISTS idx_cleaned_data_ticker ON cleaned_data(ticker);
CREATE INDEX IF NOT EXISTS idx_cleaned_data_date ON cleaned_data(date);
CREATE INDEX IF NOT EXISTS idx_panel_data_panel ON panel_data(panel_id);
CREATE INDEX IF NOT EXISTS idx_panel_data_date ON panel_data(date);
"""


# =============================================================================
# DATABASE MANAGER CLASS
# =============================================================================

class SQLDataManager:
    """
    SQLite database manager for PRISM data operations.

    Provides CRUD operations for raw, cleaned, and panel data.
    """

    def __init__(self, db_path: Union[str, Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to data/sql/prism.db
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        with self.connection() as conn:
            conn.executescript(SCHEMA)
        logger.info(f"Database initialized: {self.db_path}")

    # =========================================================================
    # RAW DATA OPERATIONS
    # =========================================================================

    def store_raw_data(
        self,
        df: pd.DataFrame,
        source: str,
        ticker: str,
        name: str = None,
        category: str = None,
        description: str = None
    ) -> int:
        """
        Store raw fetched data.

        Args:
            df: DataFrame with 'date' column and data columns
            source: Data source ('yahoo', 'fred', 'climate', 'custom')
            ticker: Ticker symbol or series ID
            name: Human-readable name
            category: Category for grouping
            description: Optional description

        Returns:
            Number of rows inserted
        """
        ticker_lower = ticker.lower()
        count = 0

        with self.connection() as conn:
            # Ensure date is in the DataFrame
            if 'date' not in df.columns and df.index.name == 'date':
                df = df.reset_index()

            # Convert date to string format
            df = df.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Determine which columns to use
            value_col = None
            for col in [ticker_lower, 'value', 'close', ticker]:
                if col in df.columns:
                    value_col = col
                    break

            for _, row in df.iterrows():
                date_val = row.get('date', row.name if isinstance(row.name, str) else None)
                if date_val is None:
                    continue

                # Extract values
                value = row.get(value_col) if value_col else None
                open_val = row.get(f'{ticker_lower}_open') or row.get('open')
                high_val = row.get(f'{ticker_lower}_high') or row.get('high')
                low_val = row.get(f'{ticker_lower}_low') or row.get('low')
                close_val = row.get(f'{ticker_lower}_close') or row.get('close') or value
                volume_val = row.get(f'{ticker_lower}_volume') or row.get('volume')

                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO raw_data
                        (date, ticker, source, value, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(date_val),
                        ticker_lower,
                        source,
                        float(value) if value is not None and not pd.isna(value) else None,
                        float(open_val) if open_val is not None and not pd.isna(open_val) else None,
                        float(high_val) if high_val is not None and not pd.isna(high_val) else None,
                        float(low_val) if low_val is not None and not pd.isna(low_val) else None,
                        float(close_val) if close_val is not None and not pd.isna(close_val) else None,
                        float(volume_val) if volume_val is not None and not pd.isna(volume_val) else None,
                    ))
                    count += 1
                except (sqlite3.Error, ValueError, TypeError) as e:
                    logger.warning(f"Error inserting row for {ticker}: {e}")

            # Update data source metadata
            cursor = conn.execute(
                "SELECT MIN(date), MAX(date) FROM raw_data WHERE ticker = ?",
                (ticker_lower,)
            )
            row = cursor.fetchone()
            first_date, last_date = row[0], row[1] if row else (None, None)

            conn.execute("""
                INSERT OR REPLACE INTO data_sources
                (ticker, source, name, category, description, first_date, last_date, row_count, last_fetched)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker_lower,
                source,
                name or ticker,
                category,
                description,
                first_date,
                last_date,
                count,
                datetime.now().isoformat()
            ))

        logger.info(f"Stored {count} rows for {ticker} from {source}")
        return count

    def load_raw_data(
        self,
        ticker: str = None,
        source: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load raw data from database.

        Args:
            ticker: Filter by ticker (optional)
            source: Filter by source (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            DataFrame with raw data
        """
        query = "SELECT * FROM raw_data WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.lower())
        if source:
            query += " AND source = ?"
            params.append(source)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with self.connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    # =========================================================================
    # CLEANED DATA OPERATIONS
    # =========================================================================

    def store_cleaned_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        cleaning_method: str = 'ffill',
        source_rows: int = None,
        nan_before: int = None,
        outliers_detected: int = 0
    ) -> int:
        """
        Store cleaned data.

        Args:
            df: DataFrame with 'date' column and value column
            ticker: Ticker symbol
            cleaning_method: Method used for cleaning
            source_rows: Original row count before cleaning
            nan_before: NaN count before cleaning
            outliers_detected: Number of outliers detected

        Returns:
            Number of rows inserted
        """
        ticker_lower = ticker.lower()
        count = 0

        with self.connection() as conn:
            # Prepare data
            df = df.copy()
            if 'date' not in df.columns and df.index.name == 'date':
                df = df.reset_index()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Find value column
            value_col = None
            for col in [ticker_lower, 'value', ticker]:
                if col in df.columns:
                    value_col = col
                    break

            if value_col is None:
                # Use first numeric column - log which column was selected
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    value_col = numeric_cols[0]
                    logger.debug(f"Auto-selected column '{value_col}' for ticker '{ticker}'")
                else:
                    logger.warning(f"No numeric columns found for ticker '{ticker}'")

            for _, row in df.iterrows():
                date_val = row.get('date', row.name)
                value = row.get(value_col) if value_col else None

                if date_val is None or value is None or pd.isna(value):
                    continue

                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO cleaned_data
                        (date, ticker, value, cleaning_method)
                        VALUES (?, ?, ?, ?)
                    """, (
                        str(date_val),
                        ticker_lower,
                        float(value),
                        cleaning_method
                    ))
                    count += 1
                except (sqlite3.Error, ValueError, TypeError) as e:
                    logger.warning(f"Error inserting cleaned data for {ticker}: {e}")

            # Update quality metrics
            conn.execute("""
                INSERT OR REPLACE INTO data_quality
                (ticker, source_rows, cleaned_rows, nan_count_before, nan_count_after,
                 nan_pct_before, nan_pct_after, outliers_detected, outliers_handled, cleaning_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker_lower,
                source_rows or count,
                count,
                nan_before or 0,
                0,
                (nan_before / source_rows * 100) if source_rows and nan_before else 0,
                0,
                outliers_detected,
                outliers_detected,
                cleaning_method
            ))

        logger.info(f"Stored {count} cleaned rows for {ticker}")
        return count

    def load_cleaned_data(
        self,
        ticker: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load cleaned data from database.

        Args:
            ticker: Filter by ticker (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            DataFrame with cleaned data
        """
        query = "SELECT date, ticker, value FROM cleaned_data WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.lower())
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with self.connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def load_cleaned_panel(
        self,
        tickers: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load cleaned data as a wide-format panel (indicators as columns).

        Args:
            tickers: List of tickers to include (optional, defaults to all)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            DataFrame with date index and ticker columns
        """
        query = "SELECT date, ticker, value FROM cleaned_data WHERE 1=1"
        params = []

        if tickers:
            placeholders = ','.join('?' for _ in tickers)
            query += f" AND ticker IN ({placeholders})"
            params.extend([t.lower() for t in tickers])
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        with self.connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        # Pivot to wide format
        df['date'] = pd.to_datetime(df['date'])
        panel = df.pivot(index='date', columns='ticker', values='value')

        return panel

    # =========================================================================
    # PANEL OPERATIONS
    # =========================================================================

    def store_panel(
        self,
        df: pd.DataFrame,
        name: str,
        description: str = None
    ) -> int:
        """
        Store a consolidated panel DataFrame.

        Args:
            df: Panel DataFrame with date index and ticker columns
            name: Panel name
            description: Optional description

        Returns:
            Panel ID
        """
        # Ensure index is datetime
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Drop rows with NaT index
        df = df[df.index.notna()]
        
        with self.connection() as conn:
            # Create panel metadata
            indicators = list(df.columns)
            start_date = df.index.min().strftime('%Y-%m-%d') if len(df) > 0 else None
            end_date = df.index.max().strftime('%Y-%m-%d') if len(df) > 0 else None

            cursor = conn.execute("""
                INSERT OR REPLACE INTO panels
                (name, description, start_date, end_date, n_indicators, n_rows, indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                description,
                start_date,
                end_date,
                len(indicators),
                len(df),
                json.dumps(indicators)
            ))
            panel_id = cursor.lastrowid

            # Store panel data in long format
            for date_val in df.index:
                if pd.isna(date_val):
                    continue
                date_str = date_val.strftime('%Y-%m-%d')
                for ticker in df.columns:
                    value = df.loc[date_val, ticker]
                    if pd.notna(value):
                        try:
                            float_val = float(value)
                            conn.execute("""
                                INSERT OR REPLACE INTO panel_data
                                (panel_id, date, ticker, value)
                                VALUES (?, ?, ?, ?)
                            """, (panel_id, date_str, ticker, float_val))
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            pass

        logger.info(f"Stored panel '{name}' with {len(indicators)} indicators")
        return panel_id

    def load_panel(self, name: str) -> pd.DataFrame:
        """
        Load a stored panel by name.

        Args:
            name: Panel name

        Returns:
            DataFrame with date index and ticker columns
        """
        with self.connection() as conn:
            # Get panel ID
            cursor = conn.execute(
                "SELECT id FROM panels WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            if not row:
                return pd.DataFrame()

            panel_id = row['id']

            # Load panel data
            df = pd.read_sql_query(
                "SELECT date, ticker, value FROM panel_data WHERE panel_id = ?",
                conn,
                params=[panel_id]
            )

        if df.empty:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        panel = df.pivot(index='date', columns='ticker', values='value')

        return panel

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def query_indicator(
        self,
        ticker: str,
        table: str = 'cleaned_data'
    ) -> pd.DataFrame:
        """
        Query data for a specific indicator.

        Args:
            ticker: Ticker symbol
            table: Table to query ('raw_data' or 'cleaned_data')

        Returns:
            DataFrame with indicator data
        """
        # Whitelist validation to prevent SQL injection
        valid_tables = ['raw_data', 'cleaned_data']
        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}. Valid: {valid_tables}")

        # Safe to use table name in query since it's validated against whitelist
        with self.connection() as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE ticker = ? ORDER BY date",
                conn,
                params=[ticker.lower()]
            )

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def list_indicators(self, table: str = 'cleaned_data') -> List[str]:
        """
        List all available indicators.

        Args:
            table: Table to query ('raw_data', 'cleaned_data', 'data_sources')

        Returns:
            List of ticker symbols
        """
        # Whitelist validation to prevent SQL injection
        valid_tables = ['raw_data', 'cleaned_data', 'data_sources']
        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}. Valid: {valid_tables}")

        # Safe to use table name in query since it's validated against whitelist
        with self.connection() as conn:
            cursor = conn.execute(f"SELECT DISTINCT ticker FROM {table} ORDER BY ticker")
            return [row['ticker'] for row in cursor.fetchall()]

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            return [row['name'] for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        # Hardcoded whitelist of known tables - safe from SQL injection
        known_tables = ['raw_data', 'cleaned_data', 'data_sources', 'data_quality', 'panels', 'panel_data']

        with self.connection() as conn:
            for table in known_tables:
                try:
                    # Safe: table name comes from hardcoded whitelist above
                    cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                    stats[table] = cursor.fetchone()['count']
                except sqlite3.OperationalError:
                    stats[table] = 0

        return stats

    def get_data_quality_report(self) -> pd.DataFrame:
        """Get data quality report for all indicators."""
        with self.connection() as conn:
            return pd.read_sql_query("SELECT * FROM data_quality ORDER BY ticker", conn)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_db: Optional[SQLDataManager] = None


def get_default_db() -> SQLDataManager:
    """Get or create the default database manager."""
    global _default_db
    if _default_db is None:
        _default_db = SQLDataManager()
        _default_db.init_schema()
    return _default_db


def store_dataframe(
    df: pd.DataFrame,
    source: str,
    ticker: str,
    cleaned: bool = False,
    **kwargs
) -> int:
    """
    Convenience function to store a DataFrame.

    Args:
        df: DataFrame to store
        source: Data source
        ticker: Ticker symbol
        cleaned: Whether this is cleaned data
        **kwargs: Additional arguments passed to store method

    Returns:
        Number of rows inserted
    """
    db = get_default_db()
    if cleaned:
        return db.store_cleaned_data(df, ticker, **kwargs)
    else:
        return db.store_raw_data(df, source, ticker, **kwargs)


def load_dataframe(
    ticker: str = None,
    cleaned: bool = True,
    as_panel: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to load data.

    Args:
        ticker: Ticker symbol (optional)
        cleaned: Whether to load cleaned data
        as_panel: Whether to return as wide-format panel
        **kwargs: Additional arguments passed to load method

    Returns:
        DataFrame with data
    """
    db = get_default_db()
    if as_panel:
        tickers = [ticker] if ticker else None
        return db.load_cleaned_panel(tickers=tickers, **kwargs)
    elif cleaned:
        return db.load_cleaned_data(ticker=ticker, **kwargs)
    else:
        return db.load_raw_data(ticker=ticker, **kwargs)


def query_indicator(ticker: str, cleaned: bool = True) -> pd.DataFrame:
    """
    Convenience function to query a specific indicator.

    Args:
        ticker: Ticker symbol
        cleaned: Whether to query cleaned data

    Returns:
        DataFrame with indicator data
    """
    db = get_default_db()
    table = 'cleaned_data' if cleaned else 'raw_data'
    return db.query_indicator(ticker, table)


def list_tables() -> List[str]:
    """List all tables in the default database."""
    return get_default_db().list_tables()
