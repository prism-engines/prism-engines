"""
Database Health Diagnostics

Checks for database connectivity, integrity, and table structure.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from diagnostics.core.base import (
    BaseDiagnostic,
    DiagnosticCategory,
    DiagnosticResult,
)


def get_db_path() -> str:
    """Get the database path using the standard PRISM method."""
    try:
        from data.sql.db_path import get_db_path as prism_get_db_path
        return prism_get_db_path()
    except ImportError:
        # Fallback to default path
        return os.path.expanduser("~/prism_data/prism.db")


class DatabaseConnectionDiagnostic(BaseDiagnostic):
    """Check database connectivity."""

    name = "database_connection"
    description = "Check SQLite database connection"
    category = DiagnosticCategory.HEALTH
    is_quick = True

    def check(self) -> DiagnosticResult:
        db_path = get_db_path()
        details = {"db_path": db_path}

        # Check if file exists
        if not os.path.exists(db_path):
            return self.fail_result(
                f"Database file not found: {db_path}",
                details=details,
                suggestions=[
                    "Initialize the database by running data fetch",
                    "Check DB_PATH environment variable or config"
                ]
            )

        # Check file size
        file_size = os.path.getsize(db_path)
        details["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        if file_size == 0:
            return self.fail_result(
                "Database file is empty",
                details=details,
                suggestions=["Re-initialize the database"]
            )

        # Try to connect
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            conn.execute("SELECT 1")
            conn.close()

            return self.pass_result(
                f"Database connection OK ({details['file_size_mb']}MB)",
                details=details
            )
        except sqlite3.Error as e:
            return self.fail_result(
                f"Database connection failed: {e}",
                details=details,
                suggestions=["Check database file permissions", "Database may be corrupted"]
            )


class DatabaseIntegrityDiagnostic(BaseDiagnostic):
    """Check database integrity."""

    name = "database_integrity"
    description = "Run SQLite integrity check"
    category = DiagnosticCategory.HEALTH
    is_quick = False  # Can take time on large databases
    dependencies = ["database_connection"]

    def check(self) -> DiagnosticResult:
        db_path = get_db_path()

        try:
            conn = sqlite3.connect(db_path, timeout=30)

            # Run integrity check
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]

            # Check WAL mode
            cursor = conn.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]

            conn.close()

            details = {
                "integrity_result": result,
                "journal_mode": journal_mode,
            }

            if result == "ok":
                return self.pass_result(
                    f"Database integrity OK (mode: {journal_mode})",
                    details=details
                )
            else:
                return self.fail_result(
                    f"Database integrity check failed: {result}",
                    details=details,
                    suggestions=[
                        "Backup and restore the database",
                        "Run: sqlite3 prism.db 'REINDEX'"
                    ]
                )

        except sqlite3.Error as e:
            return self.error_result(f"Integrity check failed: {e}")


class DatabaseTablesDiagnostic(BaseDiagnostic):
    """Check expected database tables exist."""

    name = "database_tables"
    description = "Verify required database tables exist"
    category = DiagnosticCategory.HEALTH
    is_quick = True
    dependencies = ["database_connection"]

    # Tables that should exist in a properly initialized database
    EXPECTED_TABLES = [
        'indicators',
        'indicator_data',
        'indicator_families',
    ]

    def check(self) -> DiagnosticResult:
        db_path = get_db_path()

        try:
            conn = sqlite3.connect(db_path, timeout=10)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            # Get row counts for each table
            table_counts = {}
            for table in tables:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    table_counts[table] = cursor.fetchone()[0]
                except sqlite3.Error:
                    table_counts[table] = -1

            conn.close()

            details = {
                "tables": tables,
                "table_counts": table_counts,
                "total_tables": len(tables),
            }

            # Check for expected tables
            missing = [t for t in self.EXPECTED_TABLES if t not in tables]

            if missing:
                return self.fail_result(
                    f"Missing tables: {', '.join(missing)}",
                    details=details,
                    suggestions=[
                        "Run database initialization/migration",
                        "Check if data fetch has been run"
                    ]
                )

            # Check for empty important tables
            empty_tables = [t for t in self.EXPECTED_TABLES if table_counts.get(t, 0) == 0]
            if empty_tables:
                return self.warn_result(
                    f"Empty tables: {', '.join(empty_tables)}",
                    details=details,
                    suggestions=["Run data fetch to populate tables"]
                )

            return self.pass_result(
                f"All {len(tables)} tables present, data populated",
                details=details
            )

        except sqlite3.Error as e:
            return self.error_result(f"Table check failed: {e}")
