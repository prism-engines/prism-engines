"""
PRISM Engine - SQL Database Module

This exposes the database API from db.py.

Usage:
    from data.sql import db
    
    db.init_database()
    db.add_indicator('SPY', system='market')
    db.write_dataframe(df, indicator='SPY', system='market')
    data = db.load_indicator('SPY')
"""

from .db import (
    # Path resolution
    get_db_path,
    
    # Connection
    connect,
    
    # Initialization
    init_database,
    init_db,  # alias
    
    # Query
    query,
    
    # Indicator API
    add_indicator,
    get_indicator,
    list_indicators,
    
    # Data API
    write_dataframe,
    load_indicator,
    load_multiple_indicators,
    
    # Logging
    log_fetch,
    
    # Utilities
    get_date_range,
    export_to_csv,
    get_table_stats,
)

__all__ = [
    'get_db_path',
    'connect',
    'init_database',
    'init_db',
    'query',
    'add_indicator',
    'get_indicator',
    'list_indicators',
    'write_dataframe',
    'load_indicator',
    'load_multiple_indicators',
    'log_fetch',
    'get_date_range',
    'export_to_csv',
    'get_table_stats',
]
