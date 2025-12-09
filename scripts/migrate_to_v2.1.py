#!/usr/bin/env python3
"""
PRISM Schema Migration to v2.1
==============================

Safe Python migration runner with validation.

IMPORTANT: SQLite ALTER TABLE limitations
  - Cannot add UNIQUE constraint directly via ALTER TABLE
  - Must use separate CREATE UNIQUE INDEX statement
  - Cannot add NOT NULL without DEFAULT

Changes applied:
  1. Add fred_code, category columns to indicators (UNIQUE via index)
  2. Add system_name column to systems
  3. Create calibration tables if missing
  4. Create analysis tables if missing
  5. Update schema version metadata

Usage:
    python migrate_to_v2.1.py --dry-run           # Preview changes
    python migrate_to_v2.1.py                     # Apply to default DB
    python migrate_to_v2.1.py --db /path/to.db    # Apply to specific DB
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data.sql.db_path import get_db_path
except ImportError:
    def get_db_path():
        return str(PROJECT_ROOT / "prism_data" / "prism.db")


# ============================================================================
# MIGRATION DEFINITIONS
# ============================================================================

MIGRATION_STEPS = [
    # Step 1: Add fred_code to indicators (no UNIQUE in ALTER - SQLite limitation)
    # Uniqueness enforced via CREATE UNIQUE INDEX
    {
        "name": "Add indicators.fred_code",
        "check": "SELECT 1 FROM pragma_table_info('indicators') WHERE name='fred_code'",
        "apply": "ALTER TABLE indicators ADD COLUMN fred_code TEXT",
        "post": "CREATE UNIQUE INDEX IF NOT EXISTS idx_indicators_fred_code ON indicators(fred_code)",
    },
    # Step 2: Add category to indicators
    {
        "name": "Add indicators.category",
        "check": "SELECT 1 FROM pragma_table_info('indicators') WHERE name='category'",
        "apply": "ALTER TABLE indicators ADD COLUMN category TEXT",
        "post": "CREATE INDEX IF NOT EXISTS idx_indicators_category ON indicators(category); CREATE INDEX IF NOT EXISTS idx_indicators_source ON indicators(source)",
    },
    # Step 3: Add system_name to systems
    {
        "name": "Add systems.system_name",
        "check": "SELECT 1 FROM pragma_table_info('systems') WHERE name='system_name'",
        "apply": "ALTER TABLE systems ADD COLUMN system_name TEXT DEFAULT ''",
        "post": """
            UPDATE systems SET system_name = 'Financial Markets' WHERE system = 'finance' AND (system_name IS NULL OR system_name = '');
            UPDATE systems SET system_name = 'Market Data' WHERE system = 'market' AND (system_name IS NULL OR system_name = '');
            UPDATE systems SET system_name = 'Economic Indicators' WHERE system = 'economic' AND (system_name IS NULL OR system_name = '');
            UPDATE systems SET system_name = 'Climate/Weather' WHERE system = 'climate' AND (system_name IS NULL OR system_name = '');
            UPDATE systems SET system_name = 'Biological Systems' WHERE system = 'biology' AND (system_name IS NULL OR system_name = '');
            UPDATE systems SET system_name = 'Chemical Systems' WHERE system = 'chemistry' AND (system_name IS NULL OR system_name = '');
            UPDATE systems SET system_name = 'Social/Demographic' WHERE system = 'anthropology' AND (system_name IS NULL OR system_name = '');
            UPDATE systems SET system_name = 'Physical Systems' WHERE system = 'physics' AND (system_name IS NULL OR system_name = '');
        """,
    },
    # Step 4: Create calibration_lenses
    {
        "name": "Create calibration_lenses table",
        "check": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='calibration_lenses'",
        "apply": """
            CREATE TABLE calibration_lenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lens_name TEXT UNIQUE,
                weight REAL,
                hit_rate REAL,
                avg_lead_time REAL,
                tier INTEGER,
                use_lens INTEGER DEFAULT 1,
                updated_at TEXT
            )
        """,
    },
    # Step 5: Create calibration_indicators
    {
        "name": "Create calibration_indicators table",
        "check": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='calibration_indicators'",
        "apply": """
            CREATE TABLE calibration_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator TEXT UNIQUE,
                tier INTEGER,
                score REAL,
                optimal_window INTEGER,
                indicator_type TEXT,
                use_indicator INTEGER DEFAULT 1,
                redundant_to TEXT,
                updated_at TEXT
            )
        """,
    },
    # Step 6: Create calibration_config
    {
        "name": "Create calibration_config table",
        "check": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='calibration_config'",
        "apply": """
            CREATE TABLE calibration_config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """,
    },
    # Step 7: Create analysis_rankings
    {
        "name": "Create analysis_rankings table",
        "check": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='analysis_rankings'",
        "apply": """
            CREATE TABLE analysis_rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT,
                analysis_type TEXT,
                indicator TEXT,
                consensus_rank REAL,
                tier INTEGER,
                window_used INTEGER,
                created_at TEXT
            )
        """,
        "post": """
            CREATE INDEX IF NOT EXISTS idx_rankings_date ON analysis_rankings(run_date);
            CREATE INDEX IF NOT EXISTS idx_rankings_indicator ON analysis_rankings(indicator);
        """,
    },
    # Step 8: Create analysis_signals
    {
        "name": "Create analysis_signals table",
        "check": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='analysis_signals'",
        "apply": """
            CREATE TABLE analysis_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT,
                indicator TEXT,
                signal_name TEXT,
                signal_meaning TEXT,
                rank REAL,
                status TEXT,
                created_at TEXT
            )
        """,
        "post": """
            CREATE INDEX IF NOT EXISTS idx_signals_date ON analysis_signals(run_date);
            CREATE INDEX IF NOT EXISTS idx_signals_status ON analysis_signals(status);
        """,
    },
    # Step 9: Create consensus_events
    {
        "name": "Create consensus_events table",
        "check": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='consensus_events'",
        "apply": """
            CREATE TABLE consensus_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT,
                peak_consensus REAL,
                duration_days INTEGER,
                detected_at TEXT,
                notes TEXT
            )
        """,
    },
    # Step 10: Create consensus_history
    {
        "name": "Create consensus_history table",
        "check": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='consensus_history'",
        "apply": """
            CREATE TABLE consensus_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                consensus_value REAL,
                window_size INTEGER,
                created_at TEXT,
                UNIQUE(date, window_size)
            )
        """,
        "post": "CREATE INDEX IF NOT EXISTS idx_consensus_date ON consensus_history(date)",
    },
]


# ============================================================================
# MIGRATION FUNCTIONS
# ============================================================================

def check_step_needed(conn: sqlite3.Connection, step: dict) -> bool:
    """Check if a migration step needs to be applied."""
    try:
        result = conn.execute(step["check"]).fetchone()
        return result is None  # If no result, step is needed
    except sqlite3.Error:
        return True  # If error, assume step is needed


def apply_step(conn: sqlite3.Connection, step: dict, dry_run: bool = False) -> bool:
    """Apply a single migration step."""
    step_name = step["name"]

    if not check_step_needed(conn, step):
        print(f"  [SKIP] {step_name} (already applied)")
        return True

    if dry_run:
        print(f"  [WOULD APPLY] {step_name}")
        return True

    try:
        # Apply main SQL
        for statement in step["apply"].split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(statement)

        # Apply post-SQL if exists
        if "post" in step:
            for statement in step["post"].split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(statement)

        print(f"  [OK] {step_name}")
        return True

    except sqlite3.Error as e:
        print(f"  [ERROR] {step_name}: {e}")
        return False


def update_schema_version(conn: sqlite3.Connection, dry_run: bool = False):
    """Update schema version metadata."""
    if dry_run:
        print("  [WOULD UPDATE] schema_version to 2.1")
        return

    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', '2.1')")
    conn.execute(f"INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_migration', '{datetime.now().isoformat()}')")
    print("  [OK] Updated schema_version to 2.1")


def validate_migration(conn: sqlite3.Connection) -> List[str]:
    """Validate that migration was successful."""
    issues = []

    # Check indicators columns
    cols = [r[1] for r in conn.execute("PRAGMA table_info(indicators)").fetchall()]
    if "fred_code" not in cols:
        issues.append("indicators.fred_code column missing")
    if "category" not in cols:
        issues.append("indicators.category column missing")

    # Check systems columns
    cols = [r[1] for r in conn.execute("PRAGMA table_info(systems)").fetchall()]
    if "system_name" not in cols:
        issues.append("systems.system_name column missing")

    # Check calibration tables
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    required_tables = [
        "calibration_lenses",
        "calibration_indicators",
        "calibration_config",
        "analysis_rankings",
        "analysis_signals",
        "consensus_events",
        "consensus_history",
    ]

    for table in required_tables:
        if table not in tables:
            issues.append(f"{table} table missing")

    # Check schema version
    try:
        version = conn.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if not version or version[0] != "2.1":
            issues.append(f"schema_version is {version[0] if version else 'not set'}, expected 2.1")
    except sqlite3.Error:
        issues.append("metadata table missing or unreadable")

    return issues


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Migrate PRISM database to v2.1")
    parser.add_argument("--db", help="Path to database (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    # Get database path
    db_path = args.db or get_db_path()

    print("=" * 60)
    print("PRISM Schema Migration to v2.1")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print()

    # Check database exists
    if not Path(db_path).exists():
        print("ERROR: Database not found!")
        print("For new databases, use: sqlite3 prism.db < prism_schema_v2.1.sql")
        return 1

    # Connect
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        print("Applying migration steps...")
        print()

        success = True
        for step in MIGRATION_STEPS:
            if not apply_step(conn, step, args.dry_run):
                success = False

        print()

        # Update schema version
        update_schema_version(conn, args.dry_run)

        if not args.dry_run:
            conn.commit()

        print()
        print("-" * 60)

        # Validate
        if not args.dry_run:
            print("Validating migration...")
            issues = validate_migration(conn)
            if issues:
                print("  VALIDATION FAILED:")
                for issue in issues:
                    print(f"    - {issue}")
                success = False
            else:
                print("  ✅ All validations passed")

        print()
        print("=" * 60)
        if args.dry_run:
            print("DRY RUN COMPLETE - No changes made")
        elif success:
            print("MIGRATION COMPLETE - Schema is now v2.1")
        else:
            print("MIGRATION COMPLETED WITH ISSUES")
        print("=" * 60)

        return 0 if success else 1

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
