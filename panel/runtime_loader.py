"""
Runtime Panel Loader (Domain-Agnostic)
======================================

Panels are defined at runtime by the UI, not in files.
This module provides a simple interface to load any combination
of indicators as a panel.

Usage:
    from panel.runtime_loader import load_panel

    # UI sends a list of indicator names
    selected = ["sp500", "vix", "t10y2y"]
    panel = load_panel(selected)

    # Panel is a wide-format DataFrame
    # Index: date
    # Columns: indicator names
"""

from typing import List, Optional

import pandas as pd


def load_panel(
    indicator_names: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    panel_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a panel of indicators at runtime.

    This is the domain-agnostic panel loader. Panels are defined
    by the UI at runtime as simple lists of indicator names.

    NO domain logic. NO file-based panel definitions.

    Args:
        indicator_names: List of indicator names to include
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        panel_name: Optional panel name for validation

    Returns:
        Wide-format DataFrame with:
            - Index: datetime
            - Columns: indicator names
            - Values: indicator values
    """
    # Never load climate indicators until project is active
    if panel_name and panel_name.lower().startswith("climate"):
        raise RuntimeError("Climate panels are currently frozen and inactive.")

    from data.sql.db_connector import load_all_indicators_wide

    return load_all_indicators_wide(
        indicators=indicator_names,
        start_date=start_date,
        end_date=end_date,
    )


def list_available_indicators() -> List[str]:
    """
    List all available indicators in the database.

    Returns:
        List of indicator names
    """
    from data.sql.db_connector import list_indicators

    return list_indicators()


def get_indicator_info(indicator_name: str) -> Optional[dict]:
    """
    Get metadata for a specific indicator.

    Args:
        indicator_name: Name of the indicator

    Returns:
        Dict with indicator metadata, or None if not found
    """
    from data.sql.db_connector import get_indicator

    return get_indicator(indicator_name)
