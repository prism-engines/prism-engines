"""
PRISM Schema Guard

Runtime schema validation. No schema creation at runtime.
All tables must be created via migrations.

Usage:
    from prism.db.schema_guard import assert_table_exists, assert_schema_exists

    assert_schema_exists(conn, "phase1")
    assert_table_exists(conn, "phase1.engine_validation")
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class SchemaMissingError(Exception):
    """Raised when a required schema does not exist."""
    pass


class TableMissingError(Exception):
    """Raised when a required table does not exist."""
    pass


def assert_schema_exists(conn, schema_name: str) -> None:
    """
    Assert that a schema exists. Raise if missing.

    Args:
        conn: DuckDB connection
        schema_name: Name of schema to check

    Raises:
        SchemaMissingError: If schema does not exist
    """
    result = conn.execute("""
        SELECT COUNT(*) FROM information_schema.schemata
        WHERE schema_name = ?
    """, [schema_name]).fetchone()

    if result[0] == 0:
        raise SchemaMissingError(
            f"Required schema '{schema_name}' does not exist. "
            f"Run database migrations: python scripts/build_prism_db.py"
        )


def assert_table_exists(conn, table_path: str) -> None:
    """
    Assert that a table exists. Raise if missing.

    Args:
        conn: DuckDB connection
        table_path: Full table path (schema.table_name)

    Raises:
        TableMissingError: If table does not exist
    """
    parts = table_path.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid table path: {table_path}. Expected format: schema.table_name")

    schema_name, table_name = parts

    result = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
    """, [schema_name, table_name]).fetchone()

    if result[0] == 0:
        raise TableMissingError(
            f"Required table '{table_path}' does not exist. "
            f"Run database migrations: python scripts/build_prism_db.py"
        )


def assert_tables_exist(conn, table_paths: List[str]) -> None:
    """
    Assert that multiple tables exist. Raise if any missing.

    Args:
        conn: DuckDB connection
        table_paths: List of full table paths (schema.table_name)

    Raises:
        TableMissingError: If any table does not exist
    """
    missing = []
    for table_path in table_paths:
        parts = table_path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid table path: {table_path}")

        schema_name, table_name = parts
        result = conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
        """, [schema_name, table_name]).fetchone()

        if result[0] == 0:
            missing.append(table_path)

    if missing:
        raise TableMissingError(
            f"Required tables missing: {', '.join(missing)}. "
            f"Run database migrations: python scripts/build_prism_db.py"
        )
