"""
Data Fetcher (start/ entry point)
=======================================

This is a convenience wrapper that imports from the main fetcher module.
All functionality is in the root fetcher.py file.

Usage:
    from start.fetcher import fetch_all, fetch_equities, quick_fetch

    panel = fetch_all()
    equities = fetch_equities()
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_start_dir = Path(__file__).parent.resolve()
_project_root = _start_dir.parent
sys.path.insert(0, str(_project_root))

# Re-export everything from root fetcher
from fetcher import (
    UNIVERSE,
    FRED_SERIES,
    fetch_yahoo,
    fetch_fred,
    fetch_equities,
    fetch_fixed_income,
    fetch_commodities,
    fetch_currencies,
    fetch_volatility,
    fetch_macro,
    fetch_all,
    fetch_custom,
    list_universe,
    add_ticker,
    quick_fetch,
)

__all__ = [
    'UNIVERSE',
    'FRED_SERIES',
    'fetch_yahoo',
    'fetch_fred',
    'fetch_equities',
    'fetch_fixed_income',
    'fetch_commodities',
    'fetch_currencies',
    'fetch_volatility',
    'fetch_macro',
    'fetch_all',
    'fetch_custom',
    'list_universe',
    'add_ticker',
    'quick_fetch',
]
