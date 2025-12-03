"""
Database Manager - SQLite interface for temporal analysis
================================================================

Provides SQLite storage for temporal results, replacing CSV as the
primary output format. Enables efficient querying of indicator histories,
window comparisons, and regime stability analysis.

Usage:
    from utils.db_manager import TemporalDB

    db = TemporalDB('output/temporal/temporal.db')
    db.init_schema()

    # Insert results
    window_id = db.insert_window(2005, 2010, 5, n_days=1260)
    db.insert_consensus(window_id, consensus_df)
    db.insert_lens_results(window_id, lens_results_dict)

    # Query
    history = db.query_indicator_history('SPY')
    top_10 = db.query_window_top_n(window_id, n=10)
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from contextlib import contextmanager


# =============================================================================
# SCHEMA DEFINITION
# =============================================================================

SCHEMA = """
-- Reference tables
CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    category TEXT  -- 'rates', 'inflation', 'employment', 'liquidity', 'equity', 'commodity', etc.
);

CREATE TABLE IF NOT EXISTS windows (
    id INTEGER PRIMARY KEY,
    start_year INTEGER NOT NULL,
    end_year INTEGER NOT NULL,
    increment INTEGER NOT NULL,
    n_days INTEGER,  -- actual trading days in window
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(start_year, end_year, increment)
);

CREATE TABLE IF NOT EXISTS lenses (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT
);

-- Results tables
CREATE TABLE IF NOT EXISTS lens_results (
    id INTEGER PRIMARY KEY,
    window_id INTEGER REFERENCES windows(id),
    indicator_id INTEGER REFERENCES indicators(id),
    lens_id INTEGER REFERENCES lenses(id),
    rank REAL,
    raw_score REAL,
    UNIQUE(window_id, indicator_id, lens_id)
);

CREATE TABLE IF NOT EXISTS consensus (
    id INTEGER PRIMARY KEY,
    window_id INTEGER REFERENCES windows(id),
    indicator_id INTEGER REFERENCES indicators(id),
    consensus_rank REAL,
    consensus_score REAL,
    n_lenses INTEGER,
    UNIQUE(window_id, indicator_id)
);

CREATE TABLE IF NOT EXISTS regime_stability (
    id INTEGER PRIMARY KEY,
    transition_year INTEGER UNIQUE,
    spearman_corr REAL,
    p_value REAL,
    n_indicators INTEGER,
    window_before_id INTEGER REFERENCES windows(id),
    window_after_id INTEGER REFERENCES windows(id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_lens_results_window ON lens_results(window_id);
CREATE INDEX IF NOT EXISTS idx_lens_results_indicator ON lens_results(indicator_id);
CREATE INDEX IF NOT EXISTS idx_consensus_window ON consensus(window_id);
CREATE INDEX IF NOT EXISTS idx_consensus_indicator ON consensus(indicator_id);
"""

# Indicator category mappings
INDICATOR_CATEGORIES = {
    # Rates
    'dgs10': 'rates', 'dgs2': 'rates', 'dgs3mo': 'rates',
    't10y2y': 'rates', 't10y3m': 'rates',
    # Inflation
    'cpiaucsl': 'inflation', 'cpilfesl': 'inflation', 'ppiaco': 'inflation',
    # Employment
    'unrate': 'employment', 'payems': 'employment',
    # Production
    'indpro': 'production', 'houst': 'production', 'permit': 'production',
    # Liquidity
    'm2sl': 'liquidity', 'walcl': 'liquidity',
    # Financial conditions
    'anfci': 'financial_conditions', 'nfci': 'financial_conditions',
    # Equity
    'spy': 'equity', 'qqq': 'equity', 'iwm': 'equity', 'xlu': 'equity',
    # Bonds
    'bnd': 'bonds', 'tlt': 'bonds', 'shy': 'bonds', 'ief': 'bonds',
    'tip': 'bonds', 'lqd': 'bonds', 'hyg': 'bonds',
    # Commodities
    'gld': 'commodity', 'slv': 'commodity', 'uso': 'commodity', 'bcom': 'commodity',
    # Currency
    'dxy': 'currency',
    # Volatility
    'vix': 'volatility',
}

# Lens descriptions
LENS_DESCRIPTIONS = {
    'magnitude': 'Vector magnitude analysis - measures overall movement energy',
    'pca': 'Principal Component Analysis - variance explained',
    'influence': 'Rolling volatility × deviation from mean',
    'clustering': 'Correlation-based clustering - identifies representatives',
    'decomposition': 'Trend/seasonal/residual decomposition',
    'granger': 'Granger causality - predictive relationships',
    'dmd': 'Dynamic Mode Decomposition - system dynamics',
    'mutual_info': 'Mutual information - nonlinear dependencies',
    'network': 'Network centrality metrics',
    'regime_switching': 'Hidden Markov Model regime detection',
    'anomaly': 'Anomaly/outlier detection',
    'transfer_entropy': 'Information flow between indicators',
    'tda': 'Topological Data Analysis - persistent homology',
    'wavelet': 'Multi-scale wavelet decomposition',
}


# =============================================================================
# DATABASE CLASS
# =============================================================================

class TemporalDB:
    """
    SQLite database manager for temporal analysis.

    Provides CRUD operations and analysis queries for temporal results.
    """

    def __init__(self, db_path: str):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
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

        # Populate lens reference table
        self._populate_lenses()

        print(f"Database initialized: {self.db_path}")

    def _populate_lenses(self) -> None:
        """Populate lenses reference table."""
        with self.connection() as conn:
            for name, description in LENS_DESCRIPTIONS.items():
                conn.execute(
                    "INSERT OR IGNORE INTO lenses (name, description) VALUES (?, ?)",
                    (name, description)
                )

    # =========================================================================
    # INSERT OPERATIONS
    # =========================================================================

    def insert_window(
        self,
        start_year: int,
        end_year: int,
        increment: int,
        n_days: int = None
    ) -> int:
        """
        Insert a time window and return its ID.

        Returns existing ID if window already exists.
        """
        with self.connection() as conn:
            # Check if exists
            cursor = conn.execute(
                "SELECT id FROM windows WHERE start_year=? AND end_year=? AND increment=?",
                (start_year, end_year, increment)
            )
            row = cursor.fetchone()
            if row:
                return row['id']

            # Insert new
            cursor = conn.execute(
                "INSERT INTO windows (start_year, end_year, increment, n_days) VALUES (?, ?, ?, ?)",
                (start_year, end_year, increment, n_days)
            )
            return cursor.lastrowid

    def _get_or_create_indicator(self, conn, name: str) -> int:
        """Get indicator ID, creating if necessary."""
        # Clean name
        name_clean = name.lower().replace('_spy', '').replace('_qqq', '').replace('_iwm', '')
        name_clean = name_clean.replace('_gld', '').replace('_slv', '').replace('_uso', '')
        name_clean = name_clean.replace('_close', '')

        cursor = conn.execute(
            "SELECT id FROM indicators WHERE name=?",
            (name,)
        )
        row = cursor.fetchone()
        if row:
            return row['id']

        # Determine category
        category = INDICATOR_CATEGORIES.get(name_clean, 'other')

        cursor = conn.execute(
            "INSERT INTO indicators (name, category) VALUES (?, ?)",
            (name, category)
        )
        return cursor.lastrowid

    def _get_lens_id(self, conn, name: str) -> Optional[int]:
        """Get lens ID by name."""
        cursor = conn.execute("SELECT id FROM lenses WHERE name=?", (name,))
        row = cursor.fetchone()
        return row['id'] if row else None

    def insert_consensus(
        self,
        window_id: int,
        consensus_df: pd.DataFrame
    ) -> int:
        """
        Insert consensus rankings for a window.

        Args:
            window_id: Window ID
            consensus_df: DataFrame with columns: indicator, consensus_rank, consensus_score, n_lenses

        Returns:
            Number of rows inserted
        """
        count = 0
        with self.connection() as conn:
            for _, row in consensus_df.iterrows():
                indicator_id = self._get_or_create_indicator(conn, row['indicator'])

                conn.execute("""
                    INSERT OR REPLACE INTO consensus
                    (window_id, indicator_id, consensus_rank, consensus_score, n_lenses)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    window_id,
                    indicator_id,
                    row['consensus_rank'],
                    row.get('consensus_score', 0),
                    row.get('n_lenses', 0)
                ))
                count += 1

        return count

    def insert_lens_results(
        self,
        window_id: int,
        lens_results: Dict[str, pd.DataFrame]
    ) -> int:
        """
        Insert individual lens results for a window.

        Args:
            window_id: Window ID
            lens_results: Dict mapping lens_name -> ranking DataFrame

        Returns:
            Number of rows inserted
        """
        count = 0
        with self.connection() as conn:
            for lens_name, ranking_df in lens_results.items():
                lens_id = self._get_lens_id(conn, lens_name)
                if lens_id is None:
                    continue

                for _, row in ranking_df.iterrows():
                    if 'indicator' not in row:
                        continue

                    indicator_id = self._get_or_create_indicator(conn, row['indicator'])

                    conn.execute("""
                        INSERT OR REPLACE INTO lens_results
                        (window_id, indicator_id, lens_id, rank, raw_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        window_id,
                        indicator_id,
                        lens_id,
                        row.get('rank', 0),
                        row.get('score', 0)
                    ))
                    count += 1

        return count

    def insert_regime_stability(
        self,
        stability_df: pd.DataFrame,
        window_ids: Dict[str, int]
    ) -> int:
        """
        Insert regime stability correlations.

        Args:
            stability_df: DataFrame from compute_regime_stability()
            window_ids: Dict mapping window_label -> window_id

        Returns:
            Number of rows inserted
        """
        count = 0
        with self.connection() as conn:
            for _, row in stability_df.iterrows():
                before_id = window_ids.get(row['window_from'])
                after_id = window_ids.get(row['window_to'])

                conn.execute("""
                    INSERT OR REPLACE INTO regime_stability
                    (transition_year, spearman_corr, p_value, n_indicators, window_before_id, window_after_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    row['transition_year'],
                    row['spearman_corr'],
                    row.get('p_value', None),
                    row.get('n_indicators', None),
                    before_id,
                    after_id
                ))
                count += 1

        return count

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def query_indicator_history(
        self,
        indicator_name: str
    ) -> pd.DataFrame:
        """
        Get rank trajectory for an indicator across all windows.

        Args:
            indicator_name: Name of indicator

        Returns:
            DataFrame with columns: window, start_year, end_year, consensus_rank
        """
        with self.connection() as conn:
            cursor = conn.execute("""
                SELECT
                    w.start_year || '-' || w.end_year as window,
                    w.start_year,
                    w.end_year,
                    c.consensus_rank,
                    c.consensus_score
                FROM consensus c
                JOIN windows w ON c.window_id = w.id
                JOIN indicators i ON c.indicator_id = i.id
                WHERE i.name = ?
                ORDER BY w.start_year
            """, (indicator_name,))

            rows = cursor.fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    def query_window_top_n(
        self,
        window_id: int,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N indicators for a specific window.

        Args:
            window_id: Window ID
            n: Number of top indicators

        Returns:
            DataFrame with columns: indicator, consensus_rank, consensus_score, category
        """
        with self.connection() as conn:
            cursor = conn.execute("""
                SELECT
                    i.name as indicator,
                    i.category,
                    c.consensus_rank,
                    c.consensus_score,
                    c.n_lenses
                FROM consensus c
                JOIN indicators i ON c.indicator_id = i.id
                WHERE c.window_id = ?
                ORDER BY c.consensus_rank ASC
                LIMIT ?
            """, (window_id, n))

            rows = cursor.fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    def query_all_windows(self) -> pd.DataFrame:
        """Get all windows in database."""
        with self.connection() as conn:
            cursor = conn.execute("""
                SELECT id, start_year, end_year, increment, n_days
                FROM windows
                ORDER BY start_year
            """)
            rows = cursor.fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    def query_regime_stability(self) -> pd.DataFrame:
        """Get all regime stability measurements."""
        with self.connection() as conn:
            cursor = conn.execute("""
                SELECT
                    transition_year,
                    spearman_corr,
                    p_value,
                    n_indicators
                FROM regime_stability
                ORDER BY transition_year
            """)
            rows = cursor.fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    def query_lens_agreement(
        self,
        window_id: int,
        indicator_name: str
    ) -> pd.DataFrame:
        """
        Get how each lens ranked a specific indicator in a window.

        Args:
            window_id: Window ID
            indicator_name: Indicator name

        Returns:
            DataFrame with columns: lens, rank, score
        """
        with self.connection() as conn:
            cursor = conn.execute("""
                SELECT
                    l.name as lens,
                    lr.rank,
                    lr.raw_score
                FROM lens_results lr
                JOIN lenses l ON lr.lens_id = l.id
                JOIN indicators i ON lr.indicator_id = i.id
                WHERE lr.window_id = ? AND i.name = ?
                ORDER BY lr.rank
            """, (window_id, indicator_name))

            rows = cursor.fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    def query_category_evolution(
        self,
        category: str
    ) -> pd.DataFrame:
        """
        Get average rank evolution for all indicators in a category.

        Args:
            category: Indicator category (e.g., 'equity', 'rates')

        Returns:
            DataFrame with average ranks per window
        """
        with self.connection() as conn:
            cursor = conn.execute("""
                SELECT
                    w.start_year || '-' || w.end_year as window,
                    w.start_year,
                    AVG(c.consensus_rank) as avg_rank,
                    COUNT(*) as n_indicators
                FROM consensus c
                JOIN windows w ON c.window_id = w.id
                JOIN indicators i ON c.indicator_id = i.id
                WHERE i.category = ?
                GROUP BY w.id
                ORDER BY w.start_year
            """, (category,))

            rows = cursor.fetchall()

        return pd.DataFrame([dict(row) for row in rows])

    # =========================================================================
    # EXPORT OPERATIONS
    # =========================================================================

    def export_to_csv(
        self,
        table_name: str,
        output_path: str
    ) -> str:
        """
        Export a table to CSV.

        Args:
            table_name: Name of table to export
            output_path: Path for CSV output

        Returns:
            Path to saved file
        """
        valid_tables = ['indicators', 'windows', 'lenses', 'lens_results',
                        'consensus', 'regime_stability']

        if table_name not in valid_tables:
            raise ValueError(f"Invalid table: {table_name}. Valid: {valid_tables}")

        with self.connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        return str(output_path)

    def export_rank_evolution(
        self,
        output_path: str
    ) -> str:
        """
        Export rank evolution pivot table (indicators × windows) to CSV.

        This replicates the old rank_evolution_{n}yr.csv format.
        """
        with self.connection() as conn:
            # Get all consensus data
            df = pd.read_sql_query("""
                SELECT
                    i.name as indicator,
                    w.start_year || '-' || w.end_year as window,
                    c.consensus_rank
                FROM consensus c
                JOIN indicators i ON c.indicator_id = i.id
                JOIN windows w ON c.window_id = w.id
            """, conn)

        if df.empty:
            return None

        # Pivot to indicator × window
        pivot = df.pivot(index='indicator', columns='window', values='consensus_rank')
        pivot['avg_rank'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg_rank')

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(output_path)

        return str(output_path)

    def export_temporal_results(
        self,
        output_path: str
    ) -> str:
        """
        Export long-format temporal results to CSV.

        This replicates the old temporal_results_{n}yr.csv format.
        """
        with self.connection() as conn:
            df = pd.read_sql_query("""
                SELECT
                    w.start_year || '-' || w.end_year as window,
                    w.start_year,
                    w.end_year,
                    i.name as indicator,
                    c.consensus_rank,
                    c.consensus_score,
                    c.n_lenses
                FROM consensus c
                JOIN indicators i ON c.indicator_id = i.id
                JOIN windows w ON c.window_id = w.id
                ORDER BY w.start_year, c.consensus_rank
            """, conn)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        return str(output_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.connection() as conn:
            stats = {}

            for table in ['indicators', 'windows', 'lenses', 'lens_results',
                          'consensus', 'regime_stability']:
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[table] = cursor.fetchone()['count']

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def init_db(path: str) -> TemporalDB:
    """
    Initialize database and return manager.

    Args:
        path: Path to database file

    Returns:
        TemporalDB instance
    """
    db = TemporalDB(path)
    db.init_schema()
    return db


def query_indicator_history(db_path: str, indicator_name: str) -> pd.DataFrame:
    """Convenience function to query indicator history."""
    db = TemporalDB(db_path)
    return db.query_indicator_history(indicator_name)


def query_window_top_n(db_path: str, window_id: int, n: int = 10) -> pd.DataFrame:
    """Convenience function to query top N indicators for a window."""
    db = TemporalDB(db_path)
    return db.query_window_top_n(window_id, n)


def export_to_csv(db_path: str, table_name: str, output_path: str) -> str:
    """Convenience function to export table to CSV."""
    db = TemporalDB(db_path)
    return db.export_to_csv(table_name, output_path)
