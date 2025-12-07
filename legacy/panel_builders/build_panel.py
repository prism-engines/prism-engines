"""
PRISM Panel Builder - Builds unified panel from database indicators.

DEPRECATED: This module uses domain-specific logic and will be removed.
Use panel.runtime_loader.load_panel() instead.

Schema v2 Compatible:
  - Reads from indicator_values table (universal data table)
  - Falls back to deprecated tables for backward compatibility

Usage:
    from panel.build_panel import build_panel
    df = build_panel()
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

# Deprecation warning
warnings.warn(
    "panel.build_panel is DEPRECATED and will be removed. "
    "Use panel.runtime_loader.load_panel(indicator_names) instead. "
    "Panels should be defined at runtime, not in files.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


def load_registry(path: Path) -> dict:
    """Load a JSON registry file."""
    with open(path) as f:
        return json.load(f)


def load_market_registry() -> dict:
    return load_registry(PROJECT_ROOT / "data" / "registry" / "market_registry.json")


def load_economic_registry() -> dict:
    return load_registry(PROJECT_ROOT / "data" / "registry" / "economic_registry.json")


def get_enabled_keys(registry: dict, key_field: str) -> List[str]:
    """Get enabled keys from a registry."""
    items = registry.get(key_field, [])
    return [item["key"] for item in items if item.get("enabled", True)]


def query_indicators(
    keys: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Query multiple indicators from indicator_values table (Schema v2).

    Falls back to deprecated tables (market_prices, econ_values) if
    indicator not found in indicator_values.

    Args:
        keys: List of indicator names to load
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Dict mapping indicator name to DataFrame with date index
    """
    from data.sql.db_connector import load_indicator, get_connection

    results = {}

    for key in keys:
        try:
            # Load from indicator_values (Schema v2)
            df = load_indicator(key)

            if not df.empty:
                # Apply date filters if provided
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    if start_date:
                        df = df[df['date'] >= start_date]
                    if end_date:
                        df = df[df['date'] <= end_date]
                    df = df.set_index('date')

                # Rename value column to indicator name
                if 'value' in df.columns:
                    df = df.rename(columns={'value': key})

                results[key] = df[[key]] if key in df.columns else df
                logger.debug(f"Loaded {key}: {len(df)} rows")

        except Exception as e:
            logger.warning(f"Could not load {key}: {e}")

    return results


def query_indicators_direct(
    keys: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Query multiple indicators directly with a single SQL query (faster).

    Returns wide-format DataFrame with date as index and indicators as columns.

    Args:
        keys: List of indicator names to load
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Wide-format DataFrame with date index
    """
    from data.sql.db_connector import get_connection
    import sqlite3

    if not keys:
        return pd.DataFrame()

    conn = get_connection()

    # Build query with date filters
    placeholders = ','.join('?' for _ in keys)
    query = f"""
        SELECT indicator_name, date, value
        FROM indicator_values
        WHERE indicator_name IN ({placeholders})
    """
    params = list(keys)

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date, indicator_name"

    try:
        df = pd.read_sql(query, conn, params=params)
    except sqlite3.OperationalError as e:
        logger.warning(f"Query failed: {e}")
        conn.close()
        return pd.DataFrame()

    conn.close()

    if df.empty:
        return pd.DataFrame()

    # Pivot to wide format
    df['date'] = pd.to_datetime(df['date'])
    df_wide = df.pivot(index='date', columns='indicator_name', values='value')
    df_wide = df_wide.sort_index()

    return df_wide


def combine_series(series_dict: Dict[str, pd.DataFrame], fill_method: str = "ffill") -> pd.DataFrame:
    """Combine multiple series into panel."""
    if not series_dict:
        return pd.DataFrame()

    combined = list(series_dict.values())[0]
    for df in list(series_dict.values())[1:]:
        combined = combined.join(df, how="outer")

    combined = combined.sort_index()
    if fill_method == "ffill":
        combined = combined.ffill()
    return combined


def build_panel(
    output_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fill_method: str = "ffill",
    save: bool = True,
    use_direct_query: bool = True,
) -> pd.DataFrame:
    """
    Build master panel from database and registry configuration.

    Uses indicator_values table (Schema v2) as the primary data source.

    Args:
        output_path: Path to save CSV (default: data/panels/master_panel.csv)
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        fill_method: Gap filling method ('ffill', 'bfill', 'none')
        save: Whether to save to disk
        use_direct_query: Use single SQL query (faster) vs individual loads

    Returns:
        Combined panel DataFrame
    """
    logger.info("Building panel from indicator_values (Schema v2)...")

    # Get enabled indicators from registries
    try:
        market_keys = get_enabled_keys(load_market_registry(), "instruments")
    except FileNotFoundError:
        market_keys = []
        logger.warning("market_registry.json not found")

    try:
        economic_keys = get_enabled_keys(load_economic_registry(), "series")
    except FileNotFoundError:
        economic_keys = []
        logger.warning("economic_registry.json not found")

    all_keys = market_keys + economic_keys
    logger.info(f"Registry: {len(market_keys)} market, {len(economic_keys)} economic")

    if not all_keys:
        logger.warning("No indicators configured in registries")
        return pd.DataFrame()

    # Query database
    if use_direct_query:
        panel = query_indicators_direct(all_keys, start_date, end_date)
        logger.info(f"Retrieved: {len(panel.columns)} indicators via direct query")
    else:
        all_series = query_indicators(all_keys, start_date, end_date)
        logger.info(f"Retrieved: {len(all_series)} indicators")
        panel = combine_series(all_series, fill_method)

    if panel.empty:
        logger.warning("Panel is empty - no data found in indicator_values")
        return panel

    # Apply fill method for direct query results
    if use_direct_query and fill_method == "ffill":
        panel = panel.ffill()

    # Save
    if save:
        if output_path is None:
            output_path = "data/panels/master_panel.csv"
        output_file = PROJECT_ROOT / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        panel.index.name = "date"
        panel.to_csv(output_file)
        logger.info(f"Saved: {output_file}")

    logger.info(f"Panel: {panel.shape[0]} rows x {panel.shape[1]} columns")
    return panel


def build_panel_from_list(
    indicator_names: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Build panel from explicit list of indicator names.

    Bypasses registry and loads directly from indicator_values.

    Args:
        indicator_names: List of indicator names to include
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        fill_method: Gap filling method ('ffill', 'bfill', 'none')

    Returns:
        Panel DataFrame
    """
    logger.info(f"Building panel from {len(indicator_names)} indicators...")

    panel = query_indicators_direct(indicator_names, start_date, end_date)

    if not panel.empty and fill_method == "ffill":
        panel = panel.ffill()

    return panel


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build PRISM panel from indicator_values")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--start-date", "-s", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-save", action="store_true", help="Don't save")
    parser.add_argument("--indicators", "-i", nargs="+", help="Specific indicators to load")
    args = parser.parse_args()

    if args.indicators:
        panel = build_panel_from_list(
            indicator_names=args.indicators,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        panel = build_panel(
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            save=not args.no_save,
        )

    print(f"\nPanel shape: {panel.shape}")
    if not panel.empty:
        print(f"Columns: {list(panel.columns)}")
        print(f"Date range: {panel.index.min()} to {panel.index.max()}")
