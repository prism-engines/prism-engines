#!/usr/bin/env python3
"""
PRISM Engine Benchmark Harness - Main Entry Point

Run comprehensive benchmarks for the PRISM Engine and generate
Markdown reports.

Usage:
    python benchmarks/run_benchmarks.py --all
    python benchmarks/run_benchmarks.py --suite correctness
    python benchmarks/run_benchmarks.py --suite performance
    python benchmarks/run_benchmarks.py --suite overnight
    python benchmarks/run_benchmarks.py --suite ml
    python benchmarks/run_benchmarks.py --suite correctness --suite performance

Options:
    --all           Run all benchmark suites
    --suite NAME    Run specific suite(s): correctness, performance, overnight, ml
    --verbose       Show detailed progress
    --output DIR    Output directory for reports
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Ensure we can import from the project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Handle --db argument for database path override
from data.sql.db_path import parse_db_arg
parse_db_arg()


def print_banner():
    """Print the benchmark harness banner."""
    print()
    print("=" * 70)
    print("   PRISM ENGINE BENCHMARK HARNESS")
    print("=" * 70)
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()


def run_all_benchmarks(
    suites: list[str],
    verbose: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """
    Run selected benchmark suites and generate reports.

    Parameters
    ----------
    suites : list[str]
        Which suites to run: correctness, performance, overnight, ml
    verbose : bool
        Whether to print detailed progress.
    output_dir : Path, optional
        Output directory for reports.

    Returns
    -------
    dict
        Results from each suite.
    """
    from benchmarks.engine_correctness import run_engine_correctness_benchmarks
    from benchmarks.engine_performance import run_engine_performance_benchmarks
    from benchmarks.overnight_benchmarks import run_overnight_benchmarks
    from benchmarks.ml_seismometer_benchmarks import run_ml_seismometer_benchmarks
    from benchmarks.report_builder import build_all_reports

    results = {
        "correctness": None,
        "performance": None,
        "overnight": None,
        "ml": None,
    }

    total_start = time.time()

    # Run correctness benchmarks
    if "correctness" in suites:
        if verbose:
            print("\n[1/4] Running Correctness Benchmarks...")
        results["correctness"] = run_engine_correctness_benchmarks(
            verbose=verbose,
            use_extended_lenses=False,  # Keep it fast
        )

    # Run performance benchmarks
    if "performance" in suites:
        if verbose:
            print("\n[2/4] Running Performance Benchmarks...")
        results["performance"] = run_engine_performance_benchmarks(
            profiles=["minimal", "standard"],  # Skip powerful for normal runs
            verbose=verbose,
        )

    # Run overnight benchmarks
    if "overnight" in suites:
        if verbose:
            print("\n[3/4] Running Overnight Benchmarks...")
        results["overnight"] = run_overnight_benchmarks(
            modes=["lite", "full"],
            verbose=verbose,
        )

    # Run ML/Seismometer benchmarks
    if "ml" in suites:
        if verbose:
            print("\n[4/4] Running ML/Seismometer Benchmarks...")
        results["ml"] = run_ml_seismometer_benchmarks(
            verbose=verbose,
        )

    total_time = time.time() - total_start

    # Generate reports
    if verbose:
        print("\n" + "=" * 70)
        print("GENERATING REPORTS")
        print("=" * 70)

    report_paths = build_all_reports(
        correctness=results["correctness"],
        performance=results["performance"],
        overnight=results["overnight"],
        ml=results["ml"],
        output_dir=output_dir,
    )

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"\nTotal runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"\nGenerated reports:")
        for path in report_paths:
            print(f"  - {path}")

    return {
        "results": results,
        "report_paths": report_paths,
        "total_time": total_time,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PRISM Engine Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/run_benchmarks.py --all
  python benchmarks/run_benchmarks.py --suite correctness
  python benchmarks/run_benchmarks.py --suite performance --suite ml
  python benchmarks/run_benchmarks.py --all --verbose
  python benchmarks/run_benchmarks.py --all --db ~/my_data/prism.db
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark suites",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=["correctness", "performance", "overnight", "ml"],
        help="Run specific benchmark suite(s)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for reports (default: reports/benchmarks/)",
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to database (overrides auto-detection)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version (minimal + correctness only)",
    )

    args = parser.parse_args()

    # Handle --db argument (already parsed by parse_db_arg at import)
    if args.db:
        from data.sql.db_path import set_db_path
        set_db_path(args.db)

    # Determine which suites to run
    if args.all:
        suites = ["correctness", "performance", "overnight", "ml"]
    elif args.quick:
        suites = ["correctness"]
    elif args.suite:
        suites = args.suite
    else:
        # Default: run all
        print("No suite specified. Use --all to run all suites, or --suite to select specific ones.")
        print("Running all suites by default...\n")
        suites = ["correctness", "performance", "overnight", "ml"]

    # Print banner
    print_banner()

    print(f"Suites to run: {', '.join(suites)}")

    # Show database info
    from data.sql.db_path import get_db_info
    db_info = get_db_info()
    print(f"Database: {db_info['path']}")
    print(f"  Size: {db_info['size_mb']:.1f} MB")
    print(f"  Valid: {db_info['is_valid']}")

    # Run benchmarks
    try:
        output = run_all_benchmarks(
            suites=suites,
            verbose=args.verbose or True,  # Default to verbose
            output_dir=args.output,
        )

        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        results = output["results"]

        for suite_name, suite_results in results.items():
            if suite_results is None:
                continue

            n_total = len(suite_results)
            n_passed = sum(1 for r in suite_results if r.success)

            status = "PASS" if n_passed == n_total else f"PARTIAL ({n_passed}/{n_total})"
            print(f"  {suite_name}: {status}")

        print(f"\nTotal time: {output['total_time']:.1f}s")
        print(f"\nMaster report: {output['report_paths'][0]}")

        # Return exit code based on results
        all_passed = all(
            all(r.success for r in suite_results)
            for suite_results in results.values()
            if suite_results is not None
        )

        return 0 if all_passed else 1

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nError running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
