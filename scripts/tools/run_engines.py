#!/usr/bin/env python3
"""
PRISM Engine Runner with Data Phase Preflight Gate

This script executes engines with MANDATORY Data Phase verification.
Engines may NOT run without a locked Data Phase run.

Usage:
    python scripts/run_engines.py --data-run-id <RUN_ID> --engine pca
    python scripts/run_engines.py --data-run-id <RUN_ID> --engine hmm --indicator SPY

================================================================================
ENGINE PREFLIGHT GATE (HARD BLOCK)
================================================================================
Before ANY engine execution, this script:
    1. Queries: SELECT 1 FROM meta.data_run_lock WHERE run_id = :run_id
    2. If zero rows → ABORT (engines may not run)
    3. If one row → proceed with engine execution

No fallback behavior.
No warnings-only behavior.
Engines may not run without a locked Data Phase run. Period.

================================================================================
DESIGN INTENT
================================================================================
Data Phase is an atomic operation.
Analysis without persistence is incomplete.
Engines must never infer geometry.
This is not stylistic. It is architectural.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.db.open import open_prism_db
from scripts.tools.data_integrity import (
    require_phase1_locked,
    get_phase1_contract,
    Phase1NotLockedError,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Engine Runner (with Data Phase preflight gate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_engines.py --data-run-id data_economic_20251221_123456_abc123 --engine pca
    python scripts/run_engines.py --data-run-id data_economic_20251221_123456_abc123 --engine hmm --indicator SPY
    python scripts/run_engines.py --data-run-id data_economic_20251221_123456_abc123 --list-eligible
        """
    )

    parser.add_argument(
        "--data-run-id",
        type=str,
        required=True,
        help="Data Phase run ID (REQUIRED - engines cannot run without a locked Data Phase run)"
    )

    parser.add_argument(
        "--engine",
        type=str,
        help="Engine to run (e.g., pca, hmm, garch)"
    )

    parser.add_argument(
        "--indicator",
        type=str,
        help="Specific indicator to process (default: all eligible)"
    )

    parser.add_argument(
        "--list-eligible",
        action="store_true",
        help="List eligible indicators from Data Phase contract and exit"
    )

    args = parser.parse_args()

    # =========================================================================
    # ENGINE PREFLIGHT GATE (HARD BLOCK)
    # =========================================================================
    print("=" * 70)
    print("PRISM ENGINE RUNNER - PREFLIGHT CHECK")
    print("=" * 70)

    conn = open_prism_db()

    try:
        # This will raise Phase1NotLockedError if not locked
        require_phase1_locked(conn, args.data_run_id)
        logger.info(f"✓ Data Phase preflight PASSED: {args.data_run_id} is LOCKED")

    except Phase1NotLockedError as e:
        print("\n" + "=" * 70)
        print("ENGINE PREFLIGHT FAILED")
        print("=" * 70)
        print(str(e))
        print("=" * 70)
        sys.exit(1)

    # =========================================================================
    # LOAD DATA PHASE CONTRACT
    # =========================================================================
    try:
        contract = get_phase1_contract(conn, args.data_run_id)
        logger.info(f"  Loaded Data Phase contract: {len(contract['indicators'])} indicators")

    except Exception as e:
        logger.error(f"Failed to load Data Phase contract: {e}")
        sys.exit(1)

    # =========================================================================
    # LIST ELIGIBLE (if requested)
    # =========================================================================
    if args.list_eligible:
        print("\n" + "-" * 70)
        print("ELIGIBLE INDICATORS (from Data Phase contract)")
        print("-" * 70)
        print(f"{'Indicator':<20} {'Window':>8} {'Geometry':<22} {'Eligibility':<12}")
        print("-" * 70)

        for ind in contract['indicators']:
            print(f"{ind['indicator_id']:<20} {ind['optimal_window']:>6.1f}y {ind['geom_class']:<22} {ind['eligibility']:<12}")

        # Summary
        by_elig = {}
        for ind in contract['indicators']:
            e = ind['eligibility']
            by_elig[e] = by_elig.get(e, 0) + 1

        print("-" * 70)
        for e, count in sorted(by_elig.items()):
            print(f"  {e}: {count}")
        print("-" * 70)
        sys.exit(0)

    # =========================================================================
    # ENGINE EXECUTION
    # =========================================================================
    if not args.engine:
        parser.error("--engine is required unless using --list-eligible")

    print("\n" + "-" * 70)
    print(f"ENGINE: {args.engine}")
    print("-" * 70)

    # Filter indicators if specified
    if args.indicator:
        indicators = [i for i in contract['indicators'] if i['indicator_id'] == args.indicator]
        if not indicators:
            logger.error(f"Indicator {args.indicator} not found in Data Phase contract")
            sys.exit(1)
    else:
        # Only run on eligible/conditional
        indicators = [i for i in contract['indicators'] if i['eligibility'] in ['eligible', 'conditional']]

    logger.info(f"  Indicators to process: {len(indicators)}")

    # Try to import and run the engine
    try:
        import importlib
        engine_module = importlib.import_module(f"prism.engines.{args.engine}")

        # Find the engine class
        class_name = args.engine.title().replace('_', '') + 'Engine'
        engine_class = getattr(engine_module, class_name, None)

        if engine_class is None:
            engine_class = getattr(engine_module, "Engine", None)

        if engine_class is None:
            logger.error(f"No engine class found in prism.engines.{args.engine}")
            sys.exit(1)

        engine = engine_class()

        # Execute
        logger.info(f"  Running {args.engine} on {len(indicators)} indicators...")
        started = datetime.now()

        results = []
        for ind in indicators:
            try:
                result = engine.execute(
                    indicator_ids=[ind['indicator_id']],
                    data_run_id=args.data_run_id,  # Passes preflight
                )
                results.append(result)
                status = "✓" if result.success else "✗"
                logger.info(f"    {status} {ind['indicator_id']}: {result.runtime_seconds:.2f}s")
            except Exception as e:
                logger.warning(f"    ✗ {ind['indicator_id']}: {e}")

        elapsed = (datetime.now() - started).total_seconds()

        print("\n" + "-" * 70)
        print(f"ENGINE COMPLETE: {args.engine}")
        print("-" * 70)
        print(f"  Data Phase run: {args.data_run_id}")
        print(f"  Indicators: {len(indicators)}")
        print(f"  Successful: {sum(1 for r in results if r.success)}")
        print(f"  Failed: {sum(1 for r in results if not r.success)}")
        print(f"  Duration: {elapsed:.1f}s")
        print("-" * 70)

    except ImportError as e:
        logger.error(f"Engine '{args.engine}' not found: {e}")
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
