#!/usr/bin/env python3
"""
PRISM Performance Diagnostics Runner
=====================================

Runs all performance diagnostic scripts in sequence and prints a PASS/FAIL summary.

This is the master script for the performance diagnostics suite.

Usage:
    python start/performance/run_all_diagnostics.py
    python start/performance/run_all_diagnostics.py --quick  # Skip slow tests
    python start/performance/run_all_diagnostics.py --verbose

Exit Codes:
    0 - All diagnostics passed
    1 - One or more diagnostics failed
    2 - Fatal error during execution

Notes:
    - Read-only: never writes to the database
    - All diagnostics run in isolation
    - Failures in one diagnostic do not prevent others from running
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Diagnostic scripts in execution order
DIAGNOSTIC_SCRIPTS = [
    {
        "name": "Fetch Performance",
        "script": "check_fetch_performance.py",
        "description": "Benchmark fetch speed and consistency",
        "quick": True,  # Runs in quick mode
    },
    {
        "name": "DB Performance",
        "script": "check_db_performance.py",
        "description": "Measure DB read/write performance & schema health",
        "quick": True,
    },
    {
        "name": "Geometry Engine",
        "script": "check_geometry_engine.py",
        "description": "Validate geometry input before coherence/lenses",
        "quick": True,
    },
    {
        "name": "Coherence Engine",
        "script": "check_coherence_engine.py",
        "description": "Validate wavelet coherence numerics",
        "quick": True,
    },
    {
        "name": "PCA Stability",
        "script": "check_pca_stability.py",
        "description": "Check PCA eigenvector drift",
        "quick": False,  # Can be slow with large data
    },
    {
        "name": "Hidden Variation",
        "script": "hidden_variation_sanity_check.py",
        "description": "Run HVD on family clusters",
        "quick": False,
    },
    {
        "name": "Lens Performance",
        "script": "check_lens_performance.py",
        "description": "Ensure each lens produces sane outputs",
        "quick": False,
    },
    {
        "name": "Runtime Costs",
        "script": "check_runtime_costs.py",
        "description": "Measure runtime of key engine segments",
        "quick": True,
    },
]


def run_diagnostic(script_name: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run a single diagnostic script.

    Args:
        script_name: Name of the script to run
        verbose: Whether to show script output

    Returns:
        Dict with execution results
    """
    script_path = SCRIPT_DIR / script_name

    result = {
        "script": script_name,
        "status": "unknown",
        "exit_code": -1,
        "duration_s": 0,
        "output": "",
        "error": None,
    }

    if not script_path.exists():
        result["status"] = "missing"
        result["error"] = f"Script not found: {script_path}"
        return result

    try:
        start_time = time.perf_counter()

        # Run script as subprocess
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )

        end_time = time.perf_counter()

        result["exit_code"] = process.returncode
        result["duration_s"] = end_time - start_time
        result["output"] = process.stdout
        result["error"] = process.stderr if process.stderr else None

        if process.returncode == 0:
            result["status"] = "pass"
        else:
            result["status"] = "fail"

        if verbose and process.stdout:
            print(process.stdout)
        if verbose and process.stderr:
            print(process.stderr, file=sys.stderr)

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Script timed out after 5 minutes"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def print_header():
    """Print the diagnostic suite header."""
    print("\n" + "=" * 70)
    print(" PRISM PERFORMANCE DIAGNOSTICS SUITE")
    print("=" * 70)
    print(f" Started: {datetime.now().isoformat()}")
    print(f" Scripts: {len(DIAGNOSTIC_SCRIPTS)}")
    print("=" * 70)


def print_progress(name: str, index: int, total: int):
    """Print progress indicator."""
    print(f"\n[{index}/{total}] Running: {name}...")


def print_summary(results: List[Dict[str, Any]], total_time: float):
    """Print the summary report."""
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    errors = sum(1 for r in results if r["status"] in ("error", "timeout", "missing"))
    skipped = sum(1 for r in results if r["status"] == "skipped")

    print("\n" + "=" * 70)
    print(" DIAGNOSTICS SUMMARY")
    print("=" * 70)

    # Status symbols
    def status_symbol(status: str) -> str:
        symbols = {
            "pass": "[PASS]",
            "fail": "[FAIL]",
            "error": "[ERROR]",
            "timeout": "[TIMEOUT]",
            "missing": "[MISSING]",
            "skipped": "[SKIP]",
        }
        return symbols.get(status, "[?]")

    # Individual results
    print("\n{:<6} {:<25} {:>12} {}".format("Status", "Diagnostic", "Time", "Notes"))
    print("-" * 70)

    for result in results:
        # Find the diagnostic info
        diag_info = next(
            (d for d in DIAGNOSTIC_SCRIPTS if d["script"] == result["script"]),
            {"name": result["script"]}
        )
        name = diag_info.get("name", result["script"])

        status = status_symbol(result["status"])
        time_str = f"{result['duration_s']:.2f}s" if result["duration_s"] > 0 else "N/A"

        notes = ""
        if result.get("error"):
            notes = result["error"][:30] + "..." if len(result.get("error", "")) > 30 else result.get("error", "")

        print(f"{status:<6} {name:<25} {:>12} {notes}")

    print("-" * 70)

    # Totals
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    if errors > 0:
        print(f"Errors: {errors}/{len(results)}")
    if skipped > 0:
        print(f"Skipped: {skipped}/{len(results)}")

    # Overall status
    print("\n" + "=" * 70)
    if failed == 0 and errors == 0:
        print(" OVERALL STATUS: PASS")
        print("=" * 70)
        return 0
    elif errors > 0:
        print(f" OVERALL STATUS: ERROR ({errors} diagnostic(s) could not run)")
        print("=" * 70)
        return 2
    else:
        print(f" OVERALL STATUS: FAIL ({failed} diagnostic(s) failed)")
        print("=" * 70)
        return 1


def main() -> int:
    """
    Main entry point for running all diagnostics.

    Returns:
        0 - All passed
        1 - Some failed
        2 - Errors occurred
    """
    parser = argparse.ArgumentParser(
        description="Run PRISM performance diagnostics suite"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick diagnostics (skip slow tests)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full output from each diagnostic"
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Run only the specified diagnostic (by script name)"
    )

    args = parser.parse_args()

    # Filter diagnostics
    diagnostics = DIAGNOSTIC_SCRIPTS.copy()

    if args.quick:
        diagnostics = [d for d in diagnostics if d.get("quick", False)]

    if args.only:
        diagnostics = [d for d in diagnostics if args.only in d["script"]]

    if not diagnostics:
        print("No diagnostics match the specified criteria.")
        return 1

    print_header()

    if args.quick:
        print(f" Mode: Quick (running {len(diagnostics)}/{len(DIAGNOSTIC_SCRIPTS)} tests)")
    if args.only:
        print(f" Filter: {args.only}")

    results = []
    total_start = time.perf_counter()

    for i, diag in enumerate(diagnostics, 1):
        print_progress(diag["name"], i, len(diagnostics))
        result = run_diagnostic(diag["script"], verbose=args.verbose)
        results.append(result)

        # Print quick status
        if result["status"] == "pass":
            print(f"    Status: PASS ({result['duration_s']:.2f}s)")
        elif result["status"] == "fail":
            print(f"    Status: FAIL ({result['duration_s']:.2f}s)")
            if result.get("error"):
                print(f"    Error: {result['error'][:60]}")
        else:
            print(f"    Status: {result['status'].upper()}")
            if result.get("error"):
                print(f"    Error: {result['error'][:60]}")

    total_time = time.perf_counter() - total_start

    return print_summary(results, total_time)


if __name__ == "__main__":
    sys.exit(main())
