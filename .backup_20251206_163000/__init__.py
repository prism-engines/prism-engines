"""
data.sql - SQLite-based storage for indicators and time series data.

Two modules are available:

1. data.sql.db - Modern, unified API (recommended)
    from data.sql.db import init_database, add_indicator, write_dataframe, load_indicator

2. data.sql.prism_db - Legacy API with basic IO support
    from data.sql.prism_db import get_connection, write_dataframe, load_indicator

Usage:
    import data.sql
    data.sql.init_database()
    data.sql.add_indicator("SPY", system="market")
    data.sql.write_dataframe(df, "market_prices")
    df = data.sql.load_indicator("SPY")
"""

# Import from the unified db module (includes db_connector functions)
from .db import (
    # Core connection
    get_connection,
    connect,
    get_db_path,

    # Initialization
    init_database,
    init_db,
    initialize_db,

    # Indicator operations
    add_indicator,
    get_indicator,
    list_indicators,

    # Data IO
    write_dataframe,
    load_indicator,
    load_multiple_indicators,
    query,
    export_to_csv,

    # Fetch logging
    log_fetch,
    get_fetch_history,

    # Statistics
    database_stats,
    get_table_stats,
    get_date_range,

    # Convenience
    quick_write,
)

# Also expose prism_db for backward compatibility (as a module reference)
from . import prism_db

__all__ = [
    # Core connection
    "get_connection",
    "connect",
    "get_db_path",

    # Initialization
    "init_database",
    "init_db",
    "initialize_db",

    # Indicator operations
    "add_indicator",
    "get_indicator",
    "list_indicators",

    # Data IO
    "write_dataframe",
    "load_indicator",
    "load_multiple_indicators",
    "query",
    "export_to_csv",

    # Fetch logging
    "log_fetch",
    "get_fetch_history",

    # Statistics
    "database_stats",
    "get_table_stats",
    "get_date_range",

    # Convenience
    "quick_write",

    # Legacy module
    "prism_db",
]
