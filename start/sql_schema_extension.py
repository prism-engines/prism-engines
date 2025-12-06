#!/usr/bin/env python3
"""
PRISM SQL Schema Extension
==========================

Adds tables for:
1. Calibration storage (lens weights, indicator tiers, windows)
2. Analysis results (rankings, signals over time)
3. Consensus events (detected regime breaks)

This creates a complete data layer so all scripts use the same
calibrated configuration and store results consistently.

Usage:
    python sql_schema_extension.py
    
Then in any script:
    from prism_db import PrismDB
    db = PrismDB()
    config = db.get_calibration()
    db.save_rankings(rankings_df)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    import os
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

import json
import sqlite3
from datetime import datetime
import pandas as pd

from output_config import OUTPUT_DIR, DATA_DIR

DB_PATH = DATA_DIR / "prism.db"
if not DB_PATH.exists():
    DB_PATH = Path.home() / "prism_data" / "prism.db"


# =============================================================================
# SCHEMA EXTENSION
# =============================================================================

SCHEMA_EXTENSION = """
-- Calibration: Lens Weights
CREATE TABLE IF NOT EXISTS calibration_lenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lens_name TEXT UNIQUE,
    weight REAL,
    hit_rate REAL,
    avg_lead_time REAL,
    tier INTEGER,
    use_lens INTEGER,  -- 1 = active, 0 = inactive
    updated_at TEXT
);

-- Calibration: Indicator Configuration  
CREATE TABLE IF NOT EXISTS calibration_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator TEXT UNIQUE,
    tier INTEGER,
    score REAL,
    optimal_window INTEGER,
    indicator_type TEXT,  -- fast, medium, slow
    use_indicator INTEGER,  -- 1 = active, 0 = inactive
    redundant_to TEXT,  -- NULL or name of better indicator
    updated_at TEXT
);

-- Calibration: Thresholds and Config
CREATE TABLE IF NOT EXISTS calibration_config (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

-- Results: Daily Rankings
CREATE TABLE IF NOT EXISTS analysis_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT,
    analysis_type TEXT,  -- 'tuned', 'overnight', 'calibrated'
    indicator TEXT,
    consensus_rank REAL,
    tier INTEGER,
    window_used INTEGER,
    created_at TEXT
);

-- Results: Daily Signals
CREATE TABLE IF NOT EXISTS analysis_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT,
    indicator TEXT,
    signal_name TEXT,
    signal_meaning TEXT,
    rank REAL,
    status TEXT,  -- 'DANGER', 'WARNING', 'NORMAL'
    created_at TEXT
);

-- Results: Consensus Events (Regime Breaks)
CREATE TABLE IF NOT EXISTS consensus_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date TEXT,
    peak_consensus REAL,
    duration_days INTEGER,
    detected_at TEXT,
    notes TEXT
);

-- Results: Historical Consensus Timeseries
CREATE TABLE IF NOT EXISTS consensus_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    consensus_value REAL,
    window_size INTEGER,
    created_at TEXT,
    UNIQUE(date, window_size)
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_rankings_date ON analysis_rankings(run_date);
CREATE INDEX IF NOT EXISTS idx_rankings_indicator ON analysis_rankings(indicator);
CREATE INDEX IF NOT EXISTS idx_signals_date ON analysis_signals(run_date);
CREATE INDEX IF NOT EXISTS idx_signals_status ON analysis_signals(status);
CREATE INDEX IF NOT EXISTS idx_consensus_date ON consensus_history(date);
"""


def extend_schema():
    """Add new tables to existing database."""
    
    print("=" * 60)
    print("🔧 PRISM SQL SCHEMA EXTENSION")
    print("=" * 60)
    
    print(f"\n📁 Database: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Execute schema extension
    print("\n📝 Creating new tables...")
    
    for statement in SCHEMA_EXTENSION.split(';'):
        statement = statement.strip()
        if statement:
            try:
                cursor.execute(statement)
                # Extract table name for logging
                if 'CREATE TABLE' in statement:
                    table_name = statement.split('EXISTS')[1].split('(')[0].strip()
                    print(f"   ✅ {table_name}")
                elif 'CREATE INDEX' in statement:
                    idx_name = statement.split('EXISTS')[1].split('ON')[0].strip()
                    print(f"   ✅ Index: {idx_name}")
            except sqlite3.Error as e:
                print(f"   ⚠️ {e}")
    
    conn.commit()
    conn.close()
    
    print("\n✅ Schema extension complete")


# =============================================================================
# PRISM DATABASE CLASS
# =============================================================================

class PrismDB:
    """
    Unified database interface for all PRISM scripts.
    
    Usage:
        db = PrismDB()
        
        # Get calibration
        config = db.get_calibration()
        lens_weights = db.get_lens_weights()
        active_indicators = db.get_active_indicators()
        
        # Save results
        db.save_rankings(rankings_df, analysis_type='tuned')
        db.save_signals(signals_df)
        
        # Query history
        history = db.get_ranking_history(indicator='VIXCLS', days=30)
    """
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        
    def _connect(self):
        return sqlite3.connect(self.db_path)
    
    # =========================================================================
    # CALIBRATION: SAVE
    # =========================================================================
    
    def save_lens_weights(self, weights: dict, scores: pd.DataFrame = None):
        """Save lens weights from calibration."""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        for lens_name, weight in weights.items():
            hit_rate = None
            avg_lead = None
            
            if scores is not None and lens_name in scores.index:
                hit_rate = float(scores.loc[lens_name, 'hit_rate'])
                avg_lead = float(scores.loc[lens_name, 'avg_lead_time'])
            
            tier = 1 if weight >= 1.0 else 2 if weight >= 0.7 else 3
            use_lens = 1 if weight >= 0.5 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO calibration_lenses 
                (lens_name, weight, hit_rate, avg_lead_time, tier, use_lens, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (lens_name, weight, hit_rate, avg_lead, tier, use_lens, now))
        
        conn.commit()
        conn.close()
    
    def save_indicator_config(self, indicators: dict):
        """Save indicator configuration from calibration."""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        for indicator, info in indicators.items():
            cursor.execute("""
                INSERT OR REPLACE INTO calibration_indicators
                (indicator, tier, score, optimal_window, indicator_type, 
                 use_indicator, redundant_to, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                indicator,
                info.get('tier'),
                info.get('score'),
                info.get('optimal_window'),
                info.get('type'),
                1 if info.get('use') else 0,
                info.get('redundant_to'),
                now
            ))
        
        conn.commit()
        conn.close()
    
    def save_config(self, key: str, value):
        """Save a configuration value."""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        value_str = json.dumps(value) if not isinstance(value, str) else value
        
        cursor.execute("""
            INSERT OR REPLACE INTO calibration_config (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, value_str, now))
        
        conn.commit()
        conn.close()
    
    def save_full_calibration(self, master_config: dict):
        """Save the complete master calibration config."""
        
        # Save lenses
        if 'lenses' in master_config:
            weights = {l: info['weight'] for l, info in master_config['lenses'].items()}
            self.save_lens_weights(weights)
        
        # Save indicators
        if 'indicators' in master_config:
            self.save_indicator_config(master_config['indicators'])
        
        # Save thresholds
        if 'thresholds' in master_config:
            self.save_config('thresholds', master_config['thresholds'])
        
        # Save windows
        if 'windows' in master_config:
            self.save_config('windows', master_config['windows'])
        
        # Save version info
        self.save_config('calibration_version', master_config.get('version', '1.0'))
        self.save_config('calibration_date', master_config.get('generated', datetime.now().isoformat()))
    
    # =========================================================================
    # CALIBRATION: LOAD
    # =========================================================================
    
    def get_lens_weights(self) -> dict:
        """Get lens weights from database."""
        conn = self._connect()
        
        try:
            df = pd.read_sql("""
                SELECT lens_name, weight, use_lens 
                FROM calibration_lenses 
                WHERE use_lens = 1
            """, conn)
            conn.close()
            
            if df.empty:
                return None
            
            return dict(zip(df['lens_name'], df['weight']))
        except:
            conn.close()
            return None
    
    def get_active_indicators(self) -> list:
        """Get list of active indicators."""
        conn = self._connect()
        
        try:
            df = pd.read_sql("""
                SELECT indicator 
                FROM calibration_indicators 
                WHERE use_indicator = 1
            """, conn)
            conn.close()
            
            return df['indicator'].tolist()
        except:
            conn.close()
            return None
    
    def get_indicator_config(self, indicator: str) -> dict:
        """Get configuration for a specific indicator."""
        conn = self._connect()
        
        try:
            df = pd.read_sql(f"""
                SELECT * FROM calibration_indicators 
                WHERE indicator = '{indicator}'
            """, conn)
            conn.close()
            
            if df.empty:
                return None
            
            return df.iloc[0].to_dict()
        except:
            conn.close()
            return None
    
    def get_config(self, key: str):
        """Get a configuration value."""
        conn = self._connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT value FROM calibration_config WHERE key = ?", (key,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                try:
                    return json.loads(result[0])
                except:
                    return result[0]
            return None
        except:
            conn.close()
            return None
    
    def get_calibration(self) -> dict:
        """Get full calibration config."""
        return {
            'lens_weights': self.get_lens_weights(),
            'active_indicators': self.get_active_indicators(),
            'thresholds': self.get_config('thresholds'),
            'windows': self.get_config('windows'),
            'version': self.get_config('calibration_version'),
            'date': self.get_config('calibration_date'),
        }
    
    # =========================================================================
    # RESULTS: SAVE
    # =========================================================================
    
    def save_rankings(self, rankings_df: pd.DataFrame, analysis_type: str = 'tuned'):
        """Save analysis rankings to database."""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = datetime.now()
        run_date = now.strftime('%Y-%m-%d')
        created_at = now.isoformat()
        
        for indicator, row in rankings_df.iterrows():
            cursor.execute("""
                INSERT INTO analysis_rankings 
                (run_date, analysis_type, indicator, consensus_rank, tier, window_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_date,
                analysis_type,
                indicator,
                row.get('consensus_rank'),
                row.get('tier'),
                row.get('optimal_window'),
                created_at
            ))
        
        conn.commit()
        conn.close()
    
    def save_signals(self, signals_df: pd.DataFrame):
        """Save analysis signals to database."""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = datetime.now()
        run_date = now.strftime('%Y-%m-%d')
        created_at = now.isoformat()
        
        for _, row in signals_df.iterrows():
            cursor.execute("""
                INSERT INTO analysis_signals
                (run_date, indicator, signal_name, signal_meaning, rank, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_date,
                row.get('indicator'),
                row.get('name'),
                row.get('meaning'),
                row.get('rank'),
                row.get('status'),
                created_at
            ))
        
        conn.commit()
        conn.close()
    
    def save_consensus_event(self, event_date, peak_consensus, duration_days, notes=None):
        """Save a detected consensus event."""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO consensus_events
            (event_date, peak_consensus, duration_days, detected_at, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event_date,
            peak_consensus,
            duration_days,
            datetime.now().isoformat(),
            notes
        ))
        
        conn.commit()
        conn.close()
    
    def save_consensus_history(self, consensus_series: pd.Series, window_size: int):
        """Save consensus timeseries."""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        for date, value in consensus_series.items():
            if pd.notna(value):
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO consensus_history
                        (date, consensus_value, window_size, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (
                        str(date.date()) if hasattr(date, 'date') else str(date),
                        float(value),
                        window_size,
                        now
                    ))
                except:
                    pass
        
        conn.commit()
        conn.close()
    
    # =========================================================================
    # RESULTS: QUERY
    # =========================================================================
    
    def get_ranking_history(self, indicator: str = None, days: int = 30) -> pd.DataFrame:
        """Get historical rankings for an indicator or all."""
        conn = self._connect()
        
        if indicator:
            query = f"""
                SELECT run_date, indicator, consensus_rank, status
                FROM analysis_rankings
                WHERE indicator = '{indicator}'
                AND run_date >= date('now', '-{days} days')
                ORDER BY run_date DESC
            """
        else:
            query = f"""
                SELECT run_date, indicator, consensus_rank
                FROM analysis_rankings
                WHERE run_date >= date('now', '-{days} days')
                ORDER BY run_date DESC, consensus_rank ASC
            """
        
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def get_signal_history(self, days: int = 30, status: str = None) -> pd.DataFrame:
        """Get historical signals."""
        conn = self._connect()
        
        query = f"""
            SELECT run_date, indicator, signal_name, rank, status
            FROM analysis_signals
            WHERE run_date >= date('now', '-{days} days')
        """
        
        if status:
            query += f" AND status = '{status}'"
        
        query += " ORDER BY run_date DESC, rank ASC"
        
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def get_latest_signals(self) -> pd.DataFrame:
        """Get most recent signals."""
        conn = self._connect()
        
        df = pd.read_sql("""
            SELECT indicator, signal_name, rank, status
            FROM analysis_signals
            WHERE run_date = (SELECT MAX(run_date) FROM analysis_signals)
            ORDER BY rank ASC
        """, conn)
        
        conn.close()
        return df
    
    def get_danger_history(self, days: int = 90) -> pd.DataFrame:
        """Get history of DANGER signals."""
        return self.get_signal_history(days=days, status='DANGER')
    
    def get_consensus_trend(self, days: int = 365, window_size: int = 63) -> pd.DataFrame:
        """Get consensus trend over time."""
        conn = self._connect()
        
        df = pd.read_sql(f"""
            SELECT date, consensus_value
            FROM consensus_history
            WHERE window_size = {window_size}
            AND date >= date('now', '-{days} days')
            ORDER BY date ASC
        """, conn)
        
        conn.close()
        return df


# =============================================================================
# SYNC EXISTING CALIBRATION TO SQL
# =============================================================================

def sync_calibration_to_sql():
    """Load existing JSON calibration and save to SQL."""
    
    print("\n📥 Syncing calibration files to SQL...")
    
    config_path = OUTPUT_DIR / "calibration" / "master_config.json"
    
    if not config_path.exists():
        print("   ⚠️ No master_config.json found - run calibration first")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    db = PrismDB()
    db.save_full_calibration(config)
    
    print("   ✅ Calibration synced to SQL")
    
    # Verify
    cal = db.get_calibration()
    print(f"\n   Verified:")
    print(f"   - Lenses: {len(cal['lens_weights'] or [])}")
    print(f"   - Indicators: {len(cal['active_indicators'] or [])}")
    print(f"   - Version: {cal['version']}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    extend_schema()
    sync_calibration_to_sql()
    
    print("\n" + "=" * 60)
    print("✅ SQL EXTENSION COMPLETE")
    print("=" * 60)
    print("\nNow all scripts can use:")
    print("   from sql_schema_extension import PrismDB")
    print("   db = PrismDB()")
    print("   config = db.get_calibration()")
