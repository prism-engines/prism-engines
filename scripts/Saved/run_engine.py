#!/usr/bin/env python
"""
PRISM Engine Runner

Run engines from command line.

Usage:
    # Run single engine
    python scripts/run_engine.py --engine pca
    
    # Run with specific indicators
    python scripts/run_engine.py --engine pca --indicators SPY AGG XLU
    
    # Run with date range
    python scripts/run_engine.py --engine granger --start 2020-01-01 --end 2024-01-01
    
    # Run multiple engines
    python scripts/run_engine.py --engines pca clustering hurst
    
    # Run all Phase 1 engines
    python scripts/run_engine.py --priority 1
"""

import argparse
import sys
from pathlib import Path
from datetime import date
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prism.engines import (
    get_engine,
    list_engines,
    get_engines_by_priority,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PRISM analysis engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Engine selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--engine", "-e",
        help="Single engine to run"
    )
    group.add_argument(
        "--engines",
        nargs="+",
        help="Multiple engines to run"
    )
    group.add_argument(
        "--priority", "-p",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run all engines up to priority level (1=core, 4=all)"
    )
    group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available engines"
    )
    
    # Data selection
    parser.add_argument(
        "--indicators", "-i",
        nargs="+",
        help="Specific indicators to analyze"
    )
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        help="End date (YYYY-MM-DD)"
    )
    
    # Engine parameters
    parser.add_argument(
        "--normalization", "-n",
        choices=["zscore", "minmax", "returns", "rank", "diff", "none"],
        help="Override default normalization"
    )
    parser.add_argument(
        "--params",
        nargs="*",
        help="Engine parameters as key=value pairs"
    )
    
    # Options
    parser.add_argument(
        "--db",
        type=Path,
        help="Path to DuckDB database"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def parse_params(params_list):
    """Parse key=value parameter pairs."""
    if not params_list:
        return {}
    
    params = {}
    for p in params_list:
        if "=" in p:
            key, value = p.split("=", 1)
            # Try to convert to appropriate type
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
            params[key] = value
    
    return params


def run_engine(engine_name, args, extra_params):
    """Run a single engine."""
    print(f"\n{'='*60}")
    print(f"Running: {engine_name}")
    print(f"{'='*60}")
    
    try:
        engine = get_engine(engine_name, db_path=args.db)
        
        norm = args.normalization
        if norm == "none":
            norm = None
        
        result = engine.execute(
            indicator_ids=args.indicators,
            start_date=args.start,
            end_date=args.end,
            normalization=norm,
            **extra_params
        )
        
        print(result.summary())
        
        if result.success:
            print("✓ Success")
        else:
            print(f"✗ Failed: {result.error}")
        
        return result.success
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    args = parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # List engines
    if args.list:
        print("\nAvailable Engines:")
        print("=" * 70)
        engines = list_engines()
        
        for name, info in sorted(engines.items(), key=lambda x: (x[1]["priority"], x[0])):
            print(f"  {name:20} [{info['phase']:10}] P{info['priority']} - {info['description']}")
        
        print("\nCategories: structure, correlation, causality, information, frequency, complexity, volatility, regime, dynamics")
        print("Phases: unbound, structure, bounded")
        print("Priority: 1=core, 2=info/causality, 3=freq/complexity, 4=advanced")
        return
    
    # Determine engines to run
    if args.engine:
        engines_to_run = [args.engine]
    elif args.engines:
        engines_to_run = args.engines
    else:  # priority
        engines_to_run = get_engines_by_priority(args.priority)
    
    # Parse extra parameters
    extra_params = parse_params(args.params)
    
    # Run engines
    print(f"\nRunning {len(engines_to_run)} engine(s)...")
    
    successes = 0
    failures = 0
    
    for engine_name in engines_to_run:
        if run_engine(engine_name, args, extra_params):
            successes += 1
        else:
            failures += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Complete: {successes} succeeded, {failures} failed")
    print(f"{'='*60}")
    
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
