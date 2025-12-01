"""
SQL Data Layer for PRISM Engine
================================

This module provides SQLite-based data storage and retrieval for all
PRISM data ingestion, cleaning, and output operations.

Usage:
    from data.sql import SQLDataManager

    # Initialize with default database
    db = SQLDataManager()
    db.init_schema()

    # Store raw data
    db.store_raw_data(df, source='yahoo', ticker='SPY')

    # Load cleaned data
    panel = db.load_cleaned_panel()

    # Query data
    spy_data = db.query_indicator('spy')
"""

from .sql_data_manager import (
    SQLDataManager,
    get_default_db,
    store_dataframe,
    load_dataframe,
    query_indicator,
    list_tables,
)

__all__ = [
    'SQLDataManager',
    'get_default_db',
    'store_dataframe',
    'load_dataframe',
    'query_indicator',
    'list_tables',
]
