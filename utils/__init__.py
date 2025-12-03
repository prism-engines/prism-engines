"""
PRISM Engine - Utilities
"""

from .logging_config import setup_logging
from .checkpoint_manager import CheckpointManager
from .db_manager import TemporalDB, init_db, query_indicator_history, query_window_top_n, export_to_csv

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
    # DB Registry
    'load_system_registry',
    'get_db_path',
    'get_path',
    'get_connection',
    'execute_query',
    'execute_script',
    'table_exists',
    'get_table_info',
    'get_all_tables',
    # Registry validation
    'validate_all_registries',
    'validate_system_registry',
    'validate_market_registry',
    'validate_economic_registry',
    'load_validated_registry',
    'get_enabled_tickers',
    'registries_are_valid',
]
