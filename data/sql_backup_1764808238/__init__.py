"""
SQL package initializer for PRISM Engine.

This simply exposes the modern database API from prism_db.py.
"""

from .prism_db import (
    connect,
    init_database,
    add_indicator,
    write_dataframe,
    load_indicator,
    query,
    export_to_csv,
)

__all__ = [
    "connect",
    "init_database",
    "add_indicator",
    "write_dataframe",
    "load_indicator",
    "query",
    "export_to_csv",
]
