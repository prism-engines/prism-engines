"""
DuckDB connector for PRISM Engine - drop-in replacement for data/sql/db_connector.py
"""
import duckdb
import json
import os
from typing import Optional, List, Dict, Any
import pandas as pd

_db_path_override: Optional[str] = None

def get_db_path() -> str:
    if _db_path_override:
        return _db_path_override
    env_path = os.getenv("PRISM_DUCKDB")
    if env_path:
        return os.path.expanduser(env_path)
    return os.path.expanduser("~/prism_data/prism.duckdb")

def set_db_path(path: str) -> None:
    global _db_path_override
    _db_path_override = os.path.expanduser(path)

def get_connection() -> duckdb.DuckDBPyConnection:
    path = get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return duckdb.connect(path)

def connect() -> duckdb.DuckDBPyConnection:
    return get_connection()

def _next_id(conn, table: str) -> int:
    """Get next ID for a table (DuckDB doesn't auto-increment)"""
    result = conn.execute('SELECT COALESCE(MAX(id), 0) + 1 FROM ' + table).fetchone()
    return result[0]

def load_all_indicators_wide(start_date: Optional[str] = None, end_date: Optional[str] = None,
                             indicators: Optional[List[str]] = None, ffill: bool = True,
                             conn: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    """Load indicators in wide format with optional forward-fill for monthly data."""
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    
    query = """SELECT iv.date, i.name as indicator, iv.value
        FROM indicator_values iv JOIN indicators i ON iv.indicator_id = i.id WHERE 1=1"""
    params = []
    
    if start_date:
        query += " AND iv.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND iv.date <= ?"
        params.append(end_date)
    if indicators:
        placeholders = ','.join(['?' for _ in indicators])
        query += f" AND i.name IN ({placeholders})"
        params.extend(indicators)
    
    query += " ORDER BY iv.date, i.name"
    df = conn.execute(query, params).fetchdf()
    
    if close_conn:
        conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    df['date'] = pd.to_datetime(df['date'])
    df_wide = df.pivot(index='date', columns='indicator', values='value')
    df_wide.columns.name = None
    
    # Forward-fill monthly/quarterly data to daily
    if ffill:
        df_wide = df_wide.ffill()
    
    return df_wide

def _convert_numpy_types(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj

def save_analysis_run(
    start_date: str,
    end_date: Optional[str],
    n_indicators: int,
    n_rows: int,
    n_lenses: int,
    n_errors: int,
    config: Dict,
    lens_results: Dict[str, Any],
    lens_errors: Dict[str, str],
    rankings: pd.DataFrame,
    consensus: pd.DataFrame,
    normalize_methods: Dict[str, str],
    conn: Optional[duckdb.DuckDBPyConnection] = None
) -> int:
    """Save analysis run - matches signature expected by analyze.py"""
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    
    # Get next ID
    run_id = _next_id(conn, 'analysis_runs')
    
    # Insert run record
    conn.execute("""INSERT INTO analysis_runs (id, run_time, config, n_indicators, n_lenses, date_range_start, date_range_end)
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)""",
        [run_id, json.dumps(_convert_numpy_types(config)), n_indicators, n_lenses, start_date, end_date])
    
    # Save lens results
    for lens_name, results in lens_results.items():
        lens_id = _next_id(conn, 'lens_results')
        clean_results = _convert_numpy_types(results)
        conn.execute("INSERT INTO lens_results (id, run_id, lens_name, results) VALUES (?, ?, ?, ?)",
                     [lens_id, run_id, lens_name, json.dumps(clean_results)])
    
    # Save per-lens rankings
    if isinstance(rankings, pd.DataFrame) and not rankings.empty:
        for lens_name in rankings.columns:
            if lens_name == 'indicator':
                continue
            for idx, row in rankings.iterrows():
                indicator = row.get('indicator', idx) if 'indicator' in rankings.columns else idx
                score = row.get(lens_name) if isinstance(row, pd.Series) else row[lens_name] if lens_name in row else None
                if pd.notna(score):
                    rank_id = _next_id(conn, 'indicator_rankings')
                    conn.execute("INSERT INTO indicator_rankings (id, run_id, lens_name, indicator, score) VALUES (?, ?, ?, ?, ?)",
                                 [rank_id, run_id, lens_name, str(indicator), float(score)])
    
    # Save consensus rankings
    if isinstance(consensus, pd.DataFrame) and not consensus.empty:
        for idx, row in consensus.iterrows():
            indicator = row.get('indicator', idx) if 'indicator' in consensus.columns else idx
            rank_val = row.get('rank', 0) if 'rank' in row else 0
            score_val = row.get('score', row.get('consensus_score', row.get('mean_score', 0)))
            n_lenses_val = row.get('n_lenses', n_lenses)
            cons_id = _next_id(conn, 'consensus_rankings')
            conn.execute("""INSERT INTO consensus_rankings (id, run_id, indicator, consensus_rank, consensus_score, n_lenses)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [cons_id, run_id, str(indicator), int(rank_val), float(score_val), int(n_lenses_val)])
    
    if close_conn:
        conn.close()
    return run_id

def get_latest_run_id(conn: Optional[duckdb.DuckDBPyConnection] = None) -> Optional[int]:
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    result = conn.execute("SELECT MAX(id) FROM analysis_runs").fetchone()
    if close_conn:
        conn.close()
    return result[0] if result and result[0] else None

def load_lens_weights(method: str = 'combined', run_id: Optional[int] = None,
                      conn: Optional[duckdb.DuckDBPyConnection] = None) -> Optional[Dict[str, float]]:
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    if run_id is None:
        run_id = get_latest_run_id(conn)
    if run_id is None:
        if close_conn:
            conn.close()
        return None
    result = conn.execute("SELECT weights FROM lens_weights WHERE run_id = ? AND method = ? ORDER BY id DESC LIMIT 1",
                          [run_id, method]).fetchone()
    if close_conn:
        conn.close()
    return json.loads(result[0]) if result else None

def save_lens_weights(run_id: int, method: str, weights: Dict[str, float],
                      conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    wid = _next_id(conn, 'lens_weights')
    conn.execute("INSERT INTO lens_weights (id, run_id, method, weights) VALUES (?, ?, ?, ?)",
                 [wid, run_id, method, json.dumps(_convert_numpy_types(weights))])
    if close_conn:
        conn.close()

def save_lens_geometry(run_id: int, correlation_matrix: Dict, clusters: Dict, redundant_pairs: List,
                       orthogonal_pairs: List, embedding: Optional[Dict] = None,
                       conn: Optional[duckdb.DuckDBPyConnection] = None) -> None:
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    gid = _next_id(conn, 'lens_geometry')
    conn.execute("""INSERT INTO lens_geometry (id, run_id, correlation_matrix, clusters, redundant_pairs, orthogonal_pairs, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [gid, run_id, json.dumps(_convert_numpy_types(correlation_matrix)), json.dumps(_convert_numpy_types(clusters)),
         json.dumps(_convert_numpy_types(redundant_pairs)), json.dumps(_convert_numpy_types(orthogonal_pairs)),
         json.dumps(_convert_numpy_types(embedding)) if embedding else None])
    if close_conn:
        conn.close()

def get_db_stats(conn: Optional[duckdb.DuckDBPyConnection] = None) -> Dict[str, Any]:
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    stats = {}
    stats['n_indicators'] = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()[0] or 0
    stats['n_data_points'] = conn.execute("SELECT COUNT(*) FROM indicator_values").fetchone()[0] or 0
    result = conn.execute("SELECT MIN(date), MAX(date) FROM indicator_values").fetchone()
    stats['date_range'] = {'start': str(result[0]) if result[0] else None, 'end': str(result[1]) if result[1] else None}
    stats['n_analysis_runs'] = conn.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] or 0
    if close_conn:
        conn.close()
    return stats

# =============================================================================
# ADDITIONAL QUERY FUNCTIONS (for query_results.py, update_all.py, etc.)
# =============================================================================

def query(sql: str, params: list = None, conn=None) -> pd.DataFrame:
    """Execute raw SQL and return DataFrame."""
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    result = conn.execute(sql, params or []).fetchdf()
    if close_conn:
        conn.close()
    return result

def get_analysis_runs(limit: int = 10, conn=None) -> pd.DataFrame:
    """Get recent analysis runs."""
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    df = conn.execute(f"""
        SELECT id as run_id, run_time as run_date, n_indicators, n_lenses, 
               date_range_start as start_date, date_range_end as end_date,
               0 as n_errors
        FROM analysis_runs ORDER BY id DESC LIMIT {limit}
    """).fetchdf()
    if close_conn:
        conn.close()
    return df

def get_run_results(run_id: int, conn=None) -> Dict[str, Any]:
    """Get results for a specific run."""
    return load_run_results(run_id, conn)

def get_indicator_rankings(run_id: int, lens_name: str = None, conn=None) -> pd.DataFrame:
    """Get indicator rankings for a run."""
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    
    sql = "SELECT * FROM indicator_rankings WHERE run_id = ?"
    params = [run_id]
    if lens_name:
        sql += " AND lens_name = ?"
        params.append(lens_name)
    sql += " ORDER BY score DESC"
    
    df = conn.execute(sql, params).fetchdf()
    if close_conn:
        conn.close()
    return df

def add_indicator(name: str, category: str = None, source: str = None, 
                  frequency: str = 'D', conn=None) -> int:
    """Add or update an indicator. Alias for register_indicator."""
    return register_indicator(name, category=category, source=source, 
                              frequency=frequency, conn=conn)

def write_dataframe(name: str, df: pd.DataFrame, conn=None) -> int:
    """Write indicator data. Alias for write_indicator_data."""
    return write_indicator_data(name, df, conn)

def init_database(conn=None) -> None:
    """Initialize database schema. Alias for init_schema."""
    init_schema(conn)
