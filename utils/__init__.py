"""
PRISM Engine - Utilities
"""
from .logging_config import setup_logging
from .checkpoint_manager import CheckpointManager
from .db_manager import TemporalDB, init_db, query_indicator_history, query_window_top_n, export_to_csv

# Database connector (migration runner, data insert/load)
from .db_connector import (
    run_pending_migrations,
    init_database,
    insert_market_price,
    insert_market_dividend,
    insert_market_tri,
    upsert_market_prices,
    insert_econ_value,
    upsert_econ_values,
    load_market_prices,
    load_econ_values,
    log_fetch,
    get_market_stats,
    get_econ_stats,
)

# Database registry (registry-based connection, query utils)
from .db_registry import (
    load_system_registry,
    get_db_path,
    get_path,
    get_connection,
    execute_query,
    execute_script,
    table_exists,
    get_table_info,
    get_all_tables,
)

# Date cleaning utilities
from .date_cleaner import (
    parse_date_strict,
    fix_two_digit_year,
    strip_time_and_timezone,
    validate_date_range,
    clean_date_column,
    to_iso_date,
)

# Number cleaning utilities
from .number_cleaner import (
    parse_numeric,
    is_numeric_value,
    handle_percentage,
    clean_numeric_column,
    coerce_to_float_series,
    detect_numeric_garbage,
    summarize_numeric_issues,
)

# Fetch validation (DataFrame validation)
from .fetch_validator import (
    ValidationError,
    ValidationWarning,
    validate_dataframe_shape,
    detect_footer_garbage,
    remove_footer_garbage,
    validate_no_duplicate_dates,
    remove_duplicate_dates,
    validate_date_sequence,
    sort_by_date,
    validate_frequency,
    validate_numeric_columns,
    validate_no_future_dates,
    comprehensive_validate,
)

# Registry validation
from .registry_validator import (
    validate_all_registries,
    validate_system_registry,
    validate_market_registry,
    validate_economic_registry,
    load_validated_registry,
    get_enabled_tickers,
    registries_are_valid,
)

__all__ = [
    # Logging
    'setup_logging',
    # Checkpoint
    'CheckpointManager',
    # DB Manager (temporal)
    'TemporalDB',
    'init_db',
    'query_indicator_history',
    'query_window_top_n',
    'export_to_csv',
    # DB Connector (data operations)
    'run_pending_migrations',
    'init_database',
    'insert_market_price',
    'insert_market_dividend',
    'insert_market_tri',
    'upsert_market_prices',
    'insert_econ_value',
    'upsert_econ_values',
    'load_market_prices',
    'load_econ_values',
    'log_fetch',
    'get_market_stats',
    'get_econ_stats',
    # DB Registry (connection/query)
    'load_system_registry',
    'get_db_path',
    'get_path',
    'get_connection',
    'execute_query',
    'execute_script',
    'table_exists',
    'get_table_info',
    'get_all_tables',
    # Date cleaning
    'parse_date_strict',
    'fix_two_digit_year',
    'strip_time_and_timezone',
    'validate_date_range',
    'clean_date_column',
    'to_iso_date',
    # Number cleaning
    'parse_numeric',
    'is_numeric_value',
    'handle_percentage',
    'clean_numeric_column',
    'coerce_to_float_series',
    'detect_numeric_garbage',
    'summarize_numeric_issues',
    # Fetch validation (DataFrame)
    'ValidationError',
    'ValidationWarning',
    'validate_dataframe_shape',
    'detect_footer_garbage',
    'remove_footer_garbage',
    'validate_no_duplicate_dates',
    'remove_duplicate_dates',
    'validate_date_sequence',
    'sort_by_date',
    'validate_frequency',
    'validate_numeric_columns',
    'validate_no_future_dates',
    'comprehensive_validate',
    # Registry validation
    'validate_all_registries',
    'validate_system_registry',
    'validate_market_registry',
    'validate_economic_registry',
    'load_validated_registry',
    'get_enabled_tickers',
    'registries_are_valid',
]
