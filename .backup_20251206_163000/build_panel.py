"""
PRISM Panel Builder - Builds unified panel from database indicators.

Usage:
    from panel.build_panel import build_panel
    df = build_panel()
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

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


def query_indicators(keys: List[str], system: str, start_date=None, end_date=None) -> Dict[str, pd.DataFrame]:
    """Query multiple indicators from database."""
    from data.sql.db import load_indicator
    
    results = {}
    for key in keys:
        try:
            df = load_indicator(key, system, start_date, end_date)
            if not df.empty:
                if "date" in df.columns:
                    df = df.set_index("date")
                if "value" in df.columns:
                    df = df.rename(columns={"value": key})
                results[key] = df[[key]] if key in df.columns else df
        except Exception as e:
            logger.warning(f"Could not load {key}: {e}")
    return results


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
) -> pd.DataFrame:
    """
    Build master panel from database and registry configuration.
    
    Args:
        output_path: Path to save CSV (default: data/panels/master_panel.csv)
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        fill_method: Gap filling method ('ffill', 'bfill', 'none')
        save: Whether to save to disk
    
    Returns:
        Combined panel DataFrame
    """
    logger.info("Building panel...")
    
    # Get enabled indicators from registries
    market_keys = get_enabled_keys(load_market_registry(), "instruments")
    economic_keys = get_enabled_keys(load_economic_registry(), "series")
    
    logger.info(f"Registry: {len(market_keys)} market, {len(economic_keys)} economic")
    
    # Query database
    market_series = query_indicators(market_keys, "market", start_date, end_date)
    economic_series = query_indicators(economic_keys, "economic", start_date, end_date)
    
    logger.info(f"Retrieved: {len(market_series)} market, {len(economic_series)} economic")
    
    # Combine
    all_series = {**market_series, **economic_series}
    panel = combine_series(all_series, fill_method)
    
    if panel.empty:
        logger.warning("Panel is empty")
        return panel
    
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


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Build PRISM panel")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--start-date", "-s", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-save", action="store_true", help="Don't save")
    args = parser.parse_args()
    
    panel = build_panel(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        save=not args.no_save,
    )
    print(f"\nPanel shape: {panel.shape}")
    print(f"Columns: {list(panel.columns)}")
