"""
SQLite-based storage for indicators and time series data.

Two modules are available:

1. data.sql.db - Modern, unified API (recommended)
    from data.sql.db import init_db, add_indicator, write_dataframe, load_indicator

2. data.sql.prism_db - Legacy API with basic IO support
    from data.sql.prism_db import get_connection, write_dataframe, load_indicator
"""

# Import from the unified db module (includes db_connector functions)
from .db import (
    # Core connection
    get_connection,
    get_db_path,
    init_db,
    initialize_db,
    # Indicator operations (from db_connector)
    add_indicator,
    get_indicator,
    list_indicators,
    update_indicator,
    delete_indicator,
    load_multiple_indicators,
    load_system_indicators,
    get_db_stats,
    # Data IO
    write_dataframe,
    load_indicator,
    # Fetch logging
    log_fetch,
    get_fetch_history,
    # Utils
    run_migration,
    # Convenience
    quick_write,
)

# Also expose prism_db for backward compatibility
from . import prism_db

__all__ = [
    # Core connection
    "get_connection",
    "get_db_path",
    "init_db",
    "initialize_db",
    # Indicator operations
    "add_indicator",
    "get_indicator",
    "list_indicators",
    "update_indicator",
    "delete_indicator",
    "load_multiple_indicators",
    "load_system_indicators",
    "get_db_stats",
    # Data IO
    "write_dataframe",
    "load_indicator",
    # Fetch logging
    "log_fetch",
    "get_fetch_history",
    # Utils
    "run_migration",
    # Convenience
    "quick_write",
    # Legacy module
    "prism_db",
]
