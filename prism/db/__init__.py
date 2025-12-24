"""
PRISM Database Layer

Provides DuckDB connection management and schema definitions.

All DuckDB access MUST go through this module.
"""

from .connection import DatabaseConnection, get_connection, get_db_path, DB_PATH

__all__ = ["DatabaseConnection", "get_connection", "get_db_path", "DB_PATH"]
