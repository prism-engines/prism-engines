#!/usr/bin/env python3
"""
PRISM Climate Panel Builder
===========================

Builds analysis panel from climate database.
Same format as market panel for drop-in compatibility with PRISM engines.

Usage:
    python build_climate_panel.py
    
Output:
    ~/prism_data/climate_panel.csv
"""

import pandas as pd
import sqlite3
from pathlib import Path
import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Legacy paths (kept for backward compatibility)
LEGACY_DB_PATH = Path.home() / "prism_data" / "climate.db"
OUTPUT_PATH = Path.home() / "prism_data" / "climate_panel.csv"


def build_panel():
    """Build climate panel from unified indicator_values table."""
    print("=" * 60)
    print("🌍 BUILDING CLIMATE PANEL")
    print("=" * 60)

    # Try unified indicator_values first
    try:
        from data.sql.db_connector import get_connection, list_indicators

        # Get climate indicators from unified table
        climate_indicators = list_indicators(system='climate')

        if climate_indicators:
            print("   Using unified indicator_values table (Schema v2)")
            conn = get_connection()
            placeholders = ",".join("?" for _ in climate_indicators)
            df = pd.read_sql(
                f"SELECT indicator_name AS indicator, date, value FROM indicator_values "
                f"WHERE indicator_name IN ({placeholders}) ORDER BY date",
                conn, params=climate_indicators
            )
            conn.close()
        else:
            raise ValueError("No climate indicators found in unified table")

    except Exception as e:
        # Fallback to legacy climate.db
        warnings.warn(
            f"Could not load from unified indicator_values: {e}. "
            "Falling back to legacy climate.db.",
            DeprecationWarning
        )

        if not LEGACY_DB_PATH.exists():
            print(f"❌ Database not found: {LEGACY_DB_PATH}")
            print("   Run: python fetch_climate.py")
            return None

        conn = sqlite3.connect(LEGACY_DB_PATH)
        df = pd.read_sql("SELECT indicator, date, value FROM climate_values ORDER BY date", conn)
        conn.close()
    
    print(f"   Loaded {len(df)} records")
    print(f"   Indicators: {df['indicator'].unique().tolist()}")
    
    # Pivot to wide format
    panel = df.pivot(index='date', columns='indicator', values='value')
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    
    print(f"   Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"   Shape: {panel.shape}")
    
    # Handle missing data
    # Forward fill then backward fill (climate data is monthly, some gaps OK)
    panel = panel.ffill().bfill()
    
    # Drop rows with any remaining NaN
    before = len(panel)
    panel = panel.dropna()
    after = len(panel)
    
    if before > after:
        print(f"   Dropped {before - after} rows with missing data")
    
    # Reset index for CSV
    panel = panel.reset_index()
    panel = panel.rename(columns={'index': 'date'})
    
    # Save
    panel.to_csv(OUTPUT_PATH, index=False)
    print(f"\n   ✅ Saved: {OUTPUT_PATH}")
    print(f"   Shape: {panel.shape[0]} rows × {panel.shape[1]} columns")
    
    # Preview
    print("\n   Preview (last 5 rows):")
    print(panel.tail().to_string())
    
    return panel


def show_correlations():
    """Show correlations between climate indicators."""
    print("\n" + "=" * 60)
    print("📊 CLIMATE INDICATOR CORRELATIONS")
    print("=" * 60)
    
    panel = pd.read_csv(OUTPUT_PATH, parse_dates=['date'], index_col='date')
    
    corr = panel.corr()
    print(corr.round(2).to_string())
    
    # Highlight strong correlations
    print("\n   Strong correlations (|r| > 0.7):")
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"   {corr.columns[i]} ↔ {corr.columns[j]}: {r:.2f}")


if __name__ == "__main__":
    panel = build_panel()
    if panel is not None:
        show_correlations()
