"""
Database Connector - SQLite interface using registry paths
===========================================================

Provides SQLite connection utilities that read configuration from
the system_registry.json file, ensuring centralized path management.

Usage:
    from utils.db_connector import get_connection, get_db_path

    # Get database path from registry
    db_path = get_db_path()

    # Use context manager for connections
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM table")
        rows = cursor.fetchall()
"""

import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator


# =============================================================================
# REGISTRY LOADING
# =============================================================================

def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_system_registry() -> dict:
    """
    Load the system registry configuration.

    Returns:
        dict: System registry configuration

    Raises:
        FileNotFoundError: If system_registry.json doesn't exist
        json.JSONDecodeError: If registry is invalid JSON
    """
    registry_path = _get_project_root() / "data_fetch" / "system_registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(
            f"System registry not found at {registry_path}. "
            "Please ensure data_fetch/system_registry.json exists."
        )

    with open(registry_path, "r") as f:
        return json.load(f)


def get_db_path() -> Path:
    """
    Get the database path from system registry.

    Returns:
        Path: Absolute path to the database file

    Raises:
        KeyError: If db_path not configured in registry
    """
    registry = load_system_registry()

    if "paths" not in registry or "db_path" not in registry["paths"]:
        raise KeyError(
            "Database path not configured. "
            "Ensure 'paths.db_path' exists in system_registry.json"
        )

    db_path = _get_project_root() / registry["paths"]["db_path"]
    return db_path


def get_path(key: str) -> Path:
    """
    Get any path from system registry.

    Args:
        key: Path key (e.g., 'data_raw', 'data_clean', 'logs_dir')

    Returns:
        Path: Absolute path

    Raises:
        KeyError: If path key not found in registry
    """
    registry = load_system_registry()

    if "paths" not in registry or key not in registry["paths"]:
        raise KeyError(
            f"Path '{key}' not configured in system_registry.json"
        )

    return _get_project_root() / registry["paths"][key]


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================

@contextmanager
def get_connection(
    db_path: Optional[Path] = None,
    row_factory: bool = True
) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.

    Args:
        db_path: Optional custom database path. If None, uses registry path.
        row_factory: If True, enables sqlite3.Row for dict-like access.

    Yields:
        sqlite3.Connection: Active database connection

    Example:
        with get_connection() as conn:
            cursor = conn.execute("SELECT * FROM indicators")
            for row in cursor:
                print(row['name'])
    """
    if db_path is None:
        db_path = get_db_path()

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    if row_factory:
        conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_query(query: str, params: tuple = ()) -> list:
    """
    Execute a query and return results.

    Args:
        query: SQL query string
        params: Query parameters

    Returns:
        list: List of row dictionaries
    """
    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def execute_script(script: str) -> None:
    """
    Execute a SQL script (multiple statements).

    Args:
        script: SQL script with multiple statements
    """
    with get_connection() as conn:
        conn.executescript(script)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database.

    Args:
        table_name: Name of the table to check

    Returns:
        bool: True if table exists
    """
    query = """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """
    with get_connection() as conn:
        cursor = conn.execute(query, (table_name,))
        return cursor.fetchone() is not None


def get_table_info(table_name: str) -> list:
    """
    Get column information for a table.

    Args:
        table_name: Name of the table

    Returns:
        list: List of column info dictionaries
    """
    with get_connection() as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [dict(row) for row in cursor.fetchall()]


def get_all_tables() -> list:
    """
    Get list of all tables in the database.

    Returns:
        list: List of table names
    """
    query = """
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """
    with get_connection() as conn:
        cursor = conn.execute(query)
        return [row['name'] for row in cursor.fetchall()]
