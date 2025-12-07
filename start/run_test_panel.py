#!/usr/bin/env python3
"""
PRISM Test Panel Runner
=======================

Temporary test script for verifying the domain-agnostic panel loader.
This will be replaced by the HTML UI in production.

Usage:
    # Load specific indicators via CLI
    python start/run_test_panel.py sp500 t10y2y vix

    # Load from a YAML file
    python start/run_test_panel.py --file tests/panels/basic.yaml

    # Load all indicators from registry
    python start/run_test_panel.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_registry() -> dict:
    """Load indicators from Registry v2."""
    registry_path = PROJECT_ROOT / "data" / "registry" / "indicators.yaml"

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with open(registry_path) as f:
        indicators = yaml.safe_load(f)

    return {k: v for k, v in indicators.items() if isinstance(v, dict)}


def load_panel_from_file(file_path: str) -> list:
    """Load indicator list from a YAML file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Panel file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Support both list format and dict with 'indicators' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "indicators" in data:
        return data["indicators"]
    else:
        raise ValueError(f"Invalid panel file format: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the domain-agnostic panel loader"
    )
    parser.add_argument(
        "indicators",
        nargs="*",
        help="Indicator names to load (e.g., sp500 t10y2y vix)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Load indicators from a YAML file"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Load all indicators from registry"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("PRISM TEST PANEL RUNNER")
    print("=" * 60)
    print()

    try:
        # Load registry for validation
        registry = load_registry()
        print(f"Registry loaded: {len(registry)} indicators available")

        # Determine which indicators to load
        if args.file:
            indicator_names = load_panel_from_file(args.file)
            print(f"Loaded from file: {args.file}")
        elif args.all:
            indicator_names = list(registry.keys())
            print("Loading ALL indicators from registry")
        elif args.indicators:
            indicator_names = args.indicators
        else:
            print("ERROR: No indicators specified")
            print("Usage: python run_test_panel.py sp500 vix t10y2y")
            print("       python run_test_panel.py --file panels/test.yaml")
            print("       python run_test_panel.py --all")
            return 1

        print(f"Indicators requested: {indicator_names}")
        print()

        # Validate indicators exist in registry
        missing = [x for x in indicator_names if x not in registry]
        if missing:
            print(f"WARNING: Indicators not in registry: {missing}")
            print("These may still exist in database from previous fetches.")
            print()

        # Load panel using domain-agnostic loader
        from panel.runtime_loader import load_panel

        print("Loading panel...")
        panel = load_panel(
            indicator_names,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if panel.empty:
            print("WARNING: Panel is empty (no data in database)")
            print("Run 'python start/update_all.py' to fetch data first.")
            return 1

        # Print stats
        print()
        print("=" * 60)
        print("PANEL STATISTICS")
        print("=" * 60)
        print(f"Shape:        {panel.shape[0]} rows × {panel.shape[1]} columns")
        print(f"Date range:   {panel.index.min().date()} to {panel.index.max().date()}")
        print(f"Columns:      {list(panel.columns)}")
        print()

        # Data coverage
        print("Data coverage:")
        for col in panel.columns:
            non_null = panel[col].notna().sum()
            pct = non_null / len(panel) * 100
            print(f"  {col:20s}: {non_null:5d} rows ({pct:5.1f}%)")

        print()

        # Basic statistics
        print("Summary statistics:")
        print(panel.describe().round(2).to_string())
        print()

        # Preview
        print("Last 5 rows:")
        print(panel.tail().to_string())
        print()

        print("=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
