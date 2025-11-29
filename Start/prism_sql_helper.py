"""
PRISM Engine - SQL Helper
========================
Drop this in your Start/ folder.
Import functions or run standalone to explore the database.

Usage:
    from prism_sql_helper import connect, top_indicators, indicator_history, regime_shifts
    
    conn = connect()
    df = top_indicators(conn, window_start=2005)
"""

import sqlite3
import pandas as pd
from pathlib import Path


# =============================================================================
# CONNECTION
# =============================================================================

def connect(db_path=None):
    """
    Connect to the PRISM temporal database.
    If no path provided, looks in default location.
    """
    if db_path is None:
        # Try common locations
        possible_paths = [
            Path('../06_output/temporal/prism_temporal.db'),
            Path('06_output/temporal/prism_temporal.db'),
            Path('./prism_temporal.db'),
        ]
        for p in possible_paths:
            if p.exists():
                db_path = p
                break
    
    if db_path is None or not Path(db_path).exists():
        print("Database not found. Specify path: connect('/path/to/prism_temporal.db')")
        return None
    
    conn = sqlite3.connect(db_path)
    print(f"Connected to: {db_path}")
    return conn


# =============================================================================
# QUICK QUERIES
# =============================================================================

def list_windows(conn):
    """Show all time windows in the database."""
    return pd.read_sql_query("""
        SELECT id, start_year, end_year, increment, n_days
        FROM windows
        ORDER BY start_year
    """, conn)


def list_indicators(conn):
    """Show all indicators and their categories."""
    return pd.read_sql_query("""
        SELECT id, name, category
        FROM indicators
        ORDER BY category, name
    """, conn)


def top_indicators(conn, window_start=None, n=10):
    """
    Get top N indicators by consensus rank.
    If window_start specified, returns top for that window.
    Otherwise returns overall average across all windows.
    """
    if window_start:
        return pd.read_sql_query("""
            SELECT i.name, i.category, c.avg_rank, c.std_rank, c.agreement
            FROM consensus c
            JOIN indicators i ON c.indicator_id = i.id
            JOIN windows w ON c.window_id = w.id
            WHERE w.start_year = ?
            ORDER BY c.avg_rank
            LIMIT ?
        """, conn, params=[window_start, n])
    else:
        return pd.read_sql_query("""
            SELECT i.name, i.category, 
                   AVG(c.avg_rank) as mean_rank,
                   AVG(c.std_rank) as mean_std,
                   AVG(c.agreement) as mean_agreement
            FROM consensus c
            JOIN indicators i ON c.indicator_id = i.id
            GROUP BY i.id
            ORDER BY mean_rank
            LIMIT ?
        """, conn, params=[n])


def indicator_history(conn, indicator_name):
    """
    Get full rank trajectory for one indicator across all windows.
    """
    return pd.read_sql_query("""
        SELECT w.start_year, w.end_year, c.avg_rank, c.std_rank, c.agreement
        FROM consensus c
        JOIN indicators i ON c.indicator_id = i.id
        JOIN windows w ON c.window_id = w.id
        WHERE LOWER(i.name) = LOWER(?)
        ORDER BY w.start_year
    """, conn, params=[indicator_name])


def compare_indicators(conn, indicator_list):
    """
    Compare rank trajectories for multiple indicators.
    Returns pivoted DataFrame with years as rows, indicators as columns.
    """
    placeholders = ','.join(['?' for _ in indicator_list])
    df = pd.read_sql_query(f"""
        SELECT w.start_year, i.name, c.avg_rank
        FROM consensus c
        JOIN indicators i ON c.indicator_id = i.id
        JOIN windows w ON c.window_id = w.id
        WHERE LOWER(i.name) IN ({placeholders})
        ORDER BY w.start_year
    """, conn, params=[x.lower() for x in indicator_list])
    
    return df.pivot(index='start_year', columns='name', values='avg_rank')


def regime_shifts(conn):
    """
    Get regime stability scores (Spearman correlation between adjacent windows).
    Low values indicate structural change in market dynamics.
    """
    return pd.read_sql_query("""
        SELECT transition_year, spearman_corr
        FROM regime_stability
        ORDER BY transition_year
    """, conn)


def lens_breakdown(conn, indicator_name, window_start):
    """
    See how each lens ranked a specific indicator in a specific window.
    """
    return pd.read_sql_query("""
        SELECT l.name as lens, lr.rank, lr.raw_score
        FROM lens_results lr
        JOIN lenses l ON lr.lens_id = l.id
        JOIN indicators i ON lr.indicator_id = i.id
        JOIN windows w ON lr.window_id = w.id
        WHERE LOWER(i.name) = LOWER(?)
          AND w.start_year = ?
        ORDER BY lr.rank
    """, conn, params=[indicator_name, window_start])


def window_consensus(conn, window_start):
    """
    Get full consensus table for a specific window.
    """
    return pd.read_sql_query("""
        SELECT i.name, i.category, c.avg_rank, c.std_rank, c.agreement
        FROM consensus c
        JOIN indicators i ON c.indicator_id = i.id
        JOIN windows w ON c.window_id = w.id
        WHERE w.start_year = ?
        ORDER BY c.avg_rank
    """, conn, params=[window_start])


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def export_table(conn, table_name, output_path=None):
    """
    Export any table to CSV.
    """
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Exported to: {output_path}")
    return df


def export_full_analysis(conn, output_dir='.'):
    """
    Export all key tables to CSVs for sharing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    tables = ['windows', 'indicators', 'consensus', 'regime_stability']
    for table in tables:
        export_table(conn, table, output_dir / f'{table}.csv')
    
    print(f"All tables exported to: {output_dir}")


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              PRISM Engine - SQL Helper                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Quick Start:                                                 ║
    ║                                                               ║
    ║    conn = connect()              # Connect to database        ║
    ║    list_windows(conn)            # See all time windows       ║
    ║    list_indicators(conn)         # See all indicators         ║
    ║    top_indicators(conn, 2005)    # Top 10 for 2005-2010       ║
    ║    indicator_history(conn, 'walcl')  # WALCL over time        ║
    ║    regime_shifts(conn)           # Stability scores           ║
    ║                                                               ║
    ║  Compare multiple indicators:                                 ║
    ║    compare_indicators(conn, ['walcl', 'm2sl', 'payems'])      ║
    ║                                                               ║
    ║  Deep dive on one indicator/window:                           ║
    ║    lens_breakdown(conn, 'walcl', 2008)                        ║
    ║                                                               ║
    ║  Export:                                                      ║
    ║    export_full_analysis(conn, './exports')                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Auto-connect if database exists
    conn = connect()
    
    if conn:
        print("\nDatabase contents:")
        print(f"  Windows: {pd.read_sql_query('SELECT COUNT(*) FROM windows', conn).iloc[0,0]}")
        print(f"  Indicators: {pd.read_sql_query('SELECT COUNT(*) FROM indicators', conn).iloc[0,0]}")
        print(f"  Consensus records: {pd.read_sql_query('SELECT COUNT(*) FROM consensus', conn).iloc[0,0]}")
        print("\nReady. Use the functions above to explore.")
