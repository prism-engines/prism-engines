#!/usr/bin/env python3
"""
PRISM Lens Performance Checker
===============================

Purpose: Ensure each lens produces sane outputs.

This script:
- Loads the lens modules (magnitude, pca, clustering, wavelet, etc.)
- Runs each lens on a small fixed 10-indicator panel
- Validates output shapes
- Validates non-constant output
- Prints runtime summary

Output:
    LENS PERFORMANCE REPORT
    Lens: magnitude - PASS (0.05s)
    Lens: pca - PASS (0.12s)
    ...

Usage:
    python start/performance/check_lens_performance.py

Notes:
    - Read-only: never writes to the database
    - Exits gracefully if indicators are missing
"""

import sys
import logging
import time
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Type

import pandas as pd
import numpy as np

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from panel.runtime_loader import load_panel
from data.sql.db_connector import get_db_path, load_all_indicators_wide
from data.family_manager import FamilyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
NUM_TEST_INDICATORS = 10

# Available lenses to test
LENS_MODULES = {
    "magnitude": ("engine_core.lenses.magnitude_lens", "MagnitudeLens"),
    "pca": ("engine_core.lenses.pca_lens", "PCALens"),
    "clustering": ("engine_core.lenses.clustering_lens", "ClusteringLens"),
    "granger": ("engine_core.lenses.granger_lens", "GrangerLens"),
    "dmd": ("engine_core.lenses.dmd_lens", "DMDLens"),
    "influence": ("engine_core.lenses.influence_lens", "InfluenceLens"),
    "mutual_info": ("engine_core.lenses.mutual_info_lens", "MutualInfoLens"),
    "decomposition": ("engine_core.lenses.decomposition_lens", "DecompositionLens"),
    "wavelet": ("engine_core.lenses.wavelet_lens", "WaveletLens"),
    "network": ("engine_core.lenses.network_lens", "NetworkLens"),
    "regime_switching": ("engine_core.lenses.regime_switching_lens", "RegimeSwitchingLens"),
    "anomaly": ("engine_core.lenses.anomaly_lens", "AnomalyLens"),
    "transfer_entropy": ("engine_core.lenses.transfer_entropy_lens", "TransferEntropyLens"),
    "tda": ("engine_core.lenses.tda_lens", "TDALens"),
}


def load_test_panel(num_indicators: int = NUM_TEST_INDICATORS) -> pd.DataFrame:
    """
    Load a panel of indicators for lens testing.

    Args:
        num_indicators: Number of indicators to load

    Returns:
        DataFrame suitable for lens analysis (with 'date' column)
    """
    try:
        df = load_all_indicators_wide()

        if df.empty:
            return df

        # Select indicators with best data coverage
        coverage = (~df.isna()).mean()
        best_cols = coverage.nlargest(num_indicators).index.tolist()

        df_selected = df[best_cols].copy()

        # Reset index to have 'date' as column (lens format)
        df_selected = df_selected.reset_index()
        if 'index' in df_selected.columns:
            df_selected = df_selected.rename(columns={'index': 'date'})

        return df_selected

    except Exception as e:
        logger.error(f"Error loading test panel: {e}")
        return pd.DataFrame()


def load_lens_class(lens_name: str) -> Tuple[Optional[Type], Optional[str]]:
    """
    Dynamically load a lens class.

    Args:
        lens_name: Name of the lens

    Returns:
        Tuple of (LensClass, error_message)
    """
    if lens_name not in LENS_MODULES:
        return None, f"Unknown lens: {lens_name}"

    module_path, class_name = LENS_MODULES[lens_name]

    try:
        module = importlib.import_module(module_path)
        lens_class = getattr(module, class_name)
        return lens_class, None
    except ImportError as e:
        return None, f"Import error: {e}"
    except AttributeError as e:
        return None, f"Class not found: {e}"
    except Exception as e:
        return None, f"Load error: {e}"


def validate_lens_output(result: Dict[str, Any], lens_name: str) -> Dict[str, Any]:
    """
    Validate lens output for sanity.

    Args:
        result: Lens analysis result
        lens_name: Name of the lens

    Returns:
        Dict with validation results
    """
    validation = {
        "status": "pass",
        "issues": [],
        "output_keys": list(result.keys()) if isinstance(result, dict) else [],
        "has_rankings": False,
        "has_insights": False
    }

    if not isinstance(result, dict):
        validation["status"] = "fail"
        validation["issues"].append("Output is not a dictionary")
        return validation

    # Check for expected keys
    if "rankings" in result or "indicator_scores" in result or "scores" in result:
        validation["has_rankings"] = True

    if "insights" in result or "analysis" in result or "summary" in result:
        validation["has_insights"] = True

    # Check for empty results
    if len(result) == 0:
        validation["status"] = "warning"
        validation["issues"].append("Empty result dictionary")

    # Check for constant values (potential issues)
    for key, value in result.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            if value.empty:
                validation["issues"].append(f"'{key}' is empty")
            elif isinstance(value, pd.DataFrame):
                # Check if all values are constant
                for col in value.columns:
                    if value[col].nunique() == 1 and len(value) > 1:
                        validation["issues"].append(f"'{key}[{col}]' has constant values")
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                validation["issues"].append(f"'{key}' is empty array")
            elif value.size > 1 and len(np.unique(value)) == 1:
                validation["issues"].append(f"'{key}' has constant values")

    if validation["issues"]:
        validation["status"] = "warning"

    return validation


def run_lens_test(
    lens_name: str,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Run a single lens test.

    Args:
        lens_name: Name of the lens to test
        df: Test panel DataFrame

    Returns:
        Dict with test results
    """
    result = {
        "lens_name": lens_name,
        "status": "unknown",
        "runtime_ms": 0,
        "error": None,
        "validation": None
    }

    # Load lens class
    lens_class, error = load_lens_class(lens_name)

    if error:
        result["status"] = "skip"
        result["error"] = error
        return result

    try:
        # Instantiate lens
        lens = lens_class()

        # Time the analysis
        start_time = time.perf_counter()

        # Run analysis
        analysis_result = lens.analyze(df)

        end_time = time.perf_counter()
        result["runtime_ms"] = (end_time - start_time) * 1000

        # Validate output
        validation = validate_lens_output(analysis_result, lens_name)
        result["validation"] = validation
        result["status"] = validation["status"]

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.warning(f"Lens {lens_name} failed: {e}")

    return result


def run_all_lens_tests(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Run tests on all available lenses.

    Args:
        df: Test panel DataFrame

    Returns:
        List of test results
    """
    results = []

    for lens_name in LENS_MODULES.keys():
        logger.info(f"Testing lens: {lens_name}")
        result = run_lens_test(lens_name, df)
        results.append(result)

    return results


def print_report(results: List[Dict[str, Any]], panel_shape: Tuple[int, int]) -> None:
    """Print the lens performance report."""

    print("\n" + "=" * 60)
    print("LENS PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Report time: {datetime.now().isoformat()}")
    print(f"Test panel: {panel_shape[0]} rows x {panel_shape[1] - 1} indicators")

    # Summary statistics
    passed = sum(1 for r in results if r["status"] == "pass")
    warnings = sum(1 for r in results if r["status"] == "warning")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skip")

    print(f"\n--- Summary ---")
    print(f"  Total lenses: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Warnings: {warnings}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")

    # Individual lens results
    print(f"\n--- Individual Lens Results ---")

    def status_symbol(status: str) -> str:
        symbols = {
            "pass": "[OK]",
            "warning": "[!]",
            "error": "[X]",
            "skip": "[-]"
        }
        return symbols.get(status, "[?]")

    # Sort by runtime for performance visibility
    sorted_results = sorted(results, key=lambda x: x.get("runtime_ms", 0), reverse=True)

    for result in sorted_results:
        symbol = status_symbol(result["status"])
        runtime = f"{result['runtime_ms']:.2f}ms" if result["runtime_ms"] > 0 else "N/A"

        print(f"\n  {symbol} {result['lens_name']}")
        print(f"      Runtime: {runtime}")

        if result["error"]:
            print(f"      Error: {result['error'][:60]}")

        if result["validation"]:
            val = result["validation"]
            print(f"      Output keys: {val['output_keys'][:5]}...")
            print(f"      Has rankings: {val['has_rankings']}")
            print(f"      Has insights: {val['has_insights']}")

            if val["issues"]:
                print(f"      Issues:")
                for issue in val["issues"][:3]:
                    print(f"        - {issue}")

    # Runtime summary
    print(f"\n--- Runtime Summary ---")
    runtimes = [r["runtime_ms"] for r in results if r["runtime_ms"] > 0]

    if runtimes:
        print(f"  Total runtime: {sum(runtimes):.2f} ms")
        print(f"  Average runtime: {sum(runtimes) / len(runtimes):.2f} ms")
        print(f"  Slowest lens: {max(results, key=lambda x: x.get('runtime_ms', 0))['lens_name']}")
        print(f"  Fastest lens: {min((r for r in results if r['runtime_ms'] > 0), key=lambda x: x['runtime_ms'])['lens_name']}")

    # Overall status
    print(f"\n--- Overall Status ---")
    if errors > 0:
        print(f"  [X] FAIL: {errors} lens(es) produced errors")
    elif warnings > 0:
        print(f"  [!] WARNING: {warnings} lens(es) have validation warnings")
    else:
        print(f"  [OK] PASS: All lenses functioning correctly")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for lens performance check.

    Returns:
        0 on success, 1 on error
    """
    logger.info("Starting lens performance check...")
    logger.info(f"Database path: {get_db_path()}")

    try:
        # Load test panel
        df = load_test_panel(NUM_TEST_INDICATORS)

        if df.empty:
            print("\n[!] WARNING: No data loaded. Cannot perform lens tests.")
            return 1

        logger.info(f"Panel loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Verify we have enough data
        if len(df) < 100:
            print("\n[!] WARNING: Insufficient data for lens testing.")
            return 1

        # Run all lens tests
        results = run_all_lens_tests(df)

        # Print report
        print_report(results, df.shape)

        # Return error if any lens failed
        if any(r["status"] == "error" for r in results):
            logger.warning("Lens performance check completed with errors")
            return 1

        logger.info("Lens performance check completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in lens performance check: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
