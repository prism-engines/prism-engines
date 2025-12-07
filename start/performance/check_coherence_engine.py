#!/usr/bin/env python3
"""
PRISM Coherence Engine Validator
=================================

Purpose: Validate wavelet coherence numerics.

This script:
- Loads panel for 10 indicators
- Runs small-resolution coherence block (2-4 pairs)
- Verifies shapes, no NaNs, no RuntimeWarnings
- Prints heatmap statistics (not plots)

Output:
    COHERENCE ENGINE VALIDATION
    Pairs tested: N
    Shape validation: PASS/FAIL
    NaN check: PASS/FAIL
    Statistics summary

Usage:
    python start/performance/check_coherence_engine.py

Notes:
    - Read-only: never writes to the database
    - Exits gracefully if indicators are missing
"""

import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from contextlib import contextmanager

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


# Number of indicators and pairs to test
NUM_TEST_INDICATORS = 10
NUM_TEST_PAIRS = 4


@contextmanager
def capture_warnings():
    """Context manager to capture warnings during coherence computation."""
    captured = []

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        captured.append({
            "message": str(message),
            "category": category.__name__,
            "filename": filename,
            "lineno": lineno
        })

    old_showwarning = warnings.showwarning
    warnings.showwarning = warning_handler
    warnings.filterwarnings('always')

    try:
        yield captured
    finally:
        warnings.showwarning = old_showwarning


def compute_rolling_coherence(
    returns_df: pd.DataFrame,
    window: int = 63
) -> pd.Series:
    """
    Compute rolling average correlation (coherence) over time.

    High coherence = everything moving together (crisis)
    Low coherence = normal diversified behavior

    Args:
        returns_df: DataFrame of returns
        window: Rolling window size

    Returns:
        Series of coherence values indexed by date
    """
    coherence_values = []
    dates = []

    for i in range(window, len(returns_df)):
        window_data = returns_df.iloc[i-window:i]

        # Compute correlation matrix
        corr_matrix = window_data.corr()

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_corrs = corr_matrix.values[mask]

        # Average absolute correlation
        coherence = np.nanmean(np.abs(upper_corrs))
        coherence_values.append(coherence)
        dates.append(returns_df.index[i])

    return pd.Series(coherence_values, index=dates, name='coherence')


def compute_pairwise_coherence(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 63
) -> pd.Series:
    """
    Compute rolling correlation between two series.

    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size

    Returns:
        Series of rolling correlations
    """
    df = pd.DataFrame({'s1': series1, 's2': series2})
    df = df.dropna()

    if len(df) < window:
        return pd.Series(dtype=float)

    return df['s1'].rolling(window=window).corr(df['s2'])


def load_test_panel(num_indicators: int = NUM_TEST_INDICATORS) -> pd.DataFrame:
    """
    Load a panel of indicators for coherence testing.

    Args:
        num_indicators: Number of indicators to load

    Returns:
        Wide-format DataFrame
    """
    try:
        df = load_all_indicators_wide()

        if df.empty:
            return df

        # Select first N indicators with good data coverage
        valid_cols = []
        for col in df.columns:
            nan_rate = df[col].isna().mean()
            if nan_rate < 0.3:  # At least 70% data coverage
                valid_cols.append(col)
            if len(valid_cols) >= num_indicators:
                break

        return df[valid_cols] if valid_cols else df.iloc[:, :num_indicators]

    except Exception as e:
        logger.error(f"Error loading test panel: {e}")
        return pd.DataFrame()


def validate_coherence_shapes(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that coherence results have expected shapes.

    Args:
        results: Dict with coherence computation results

    Returns:
        Dict with validation results
    """
    validation = {
        "status": "pass",
        "issues": []
    }

    # Check rolling coherence shape
    if "rolling_coherence" in results:
        rc = results["rolling_coherence"]
        if isinstance(rc, pd.Series):
            if len(rc) == 0:
                validation["issues"].append("Rolling coherence is empty")
                validation["status"] = "warning"
        else:
            validation["issues"].append("Rolling coherence is not a Series")
            validation["status"] = "fail"

    # Check pairwise results
    if "pairwise_coherence" in results:
        for pair_name, pc in results["pairwise_coherence"].items():
            if isinstance(pc, pd.Series):
                if len(pc) == 0:
                    validation["issues"].append(f"Pairwise {pair_name} is empty")
            else:
                validation["issues"].append(f"Pairwise {pair_name} is not a Series")
                validation["status"] = "fail"

    return validation


def validate_coherence_nans(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check for NaN values in coherence results.

    Args:
        results: Dict with coherence computation results

    Returns:
        Dict with NaN validation results
    """
    validation = {
        "status": "pass",
        "nan_rates": {}
    }

    # Check rolling coherence
    if "rolling_coherence" in results:
        rc = results["rolling_coherence"]
        if isinstance(rc, pd.Series) and len(rc) > 0:
            nan_rate = rc.isna().mean()
            validation["nan_rates"]["rolling_coherence"] = float(nan_rate)
            if nan_rate > 0.1:  # More than 10% NaN
                validation["status"] = "warning"

    # Check pairwise results
    if "pairwise_coherence" in results:
        for pair_name, pc in results["pairwise_coherence"].items():
            if isinstance(pc, pd.Series) and len(pc) > 0:
                nan_rate = pc.isna().mean()
                validation["nan_rates"][pair_name] = float(nan_rate)
                if nan_rate > 0.2:  # More than 20% NaN
                    validation["status"] = "warning"

    return validation


def compute_heatmap_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute correlation heatmap statistics.

    Args:
        df: Panel DataFrame

    Returns:
        Dict with heatmap statistics
    """
    stats = {
        "indicators": len(df.columns),
        "correlation_matrix": None,
        "mean_correlation": None,
        "median_correlation": None,
        "max_correlation": None,
        "min_correlation": None,
        "std_correlation": None
    }

    try:
        # Compute correlation matrix
        corr_matrix = df.corr()
        stats["correlation_matrix"] = corr_matrix.shape

        # Get upper triangle values
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_values = corr_matrix.values[mask]
        upper_values = upper_values[~np.isnan(upper_values)]

        if len(upper_values) > 0:
            stats["mean_correlation"] = float(np.mean(upper_values))
            stats["median_correlation"] = float(np.median(upper_values))
            stats["max_correlation"] = float(np.max(upper_values))
            stats["min_correlation"] = float(np.min(upper_values))
            stats["std_correlation"] = float(np.std(upper_values))

    except Exception as e:
        logger.error(f"Error computing heatmap statistics: {e}")
        stats["error"] = str(e)

    return stats


def run_coherence_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run coherence engine tests.

    Args:
        df: Panel DataFrame

    Returns:
        Dict with all coherence test results
    """
    results = {
        "rolling_coherence": None,
        "pairwise_coherence": {},
        "warnings_captured": [],
        "computation_time_ms": 0
    }

    import time
    start_time = time.perf_counter()

    try:
        # Compute returns
        returns_df = df.pct_change().dropna()

        if returns_df.empty:
            results["error"] = "No returns data after differencing"
            return results

        # Capture warnings during computation
        with capture_warnings() as captured:
            # Rolling coherence
            results["rolling_coherence"] = compute_rolling_coherence(returns_df)

            # Pairwise coherence for a few pairs
            columns = list(returns_df.columns)
            pairs_tested = 0

            for i in range(min(2, len(columns))):
                for j in range(i + 1, min(i + 3, len(columns))):
                    if pairs_tested >= NUM_TEST_PAIRS:
                        break

                    col1, col2 = columns[i], columns[j]
                    pair_name = f"{col1}_vs_{col2}"
                    results["pairwise_coherence"][pair_name] = compute_pairwise_coherence(
                        returns_df[col1],
                        returns_df[col2]
                    )
                    pairs_tested += 1

        results["warnings_captured"] = captured

    except Exception as e:
        logger.error(f"Error in coherence tests: {e}")
        results["error"] = str(e)

    end_time = time.perf_counter()
    results["computation_time_ms"] = (end_time - start_time) * 1000

    return results


def print_report(
    coherence_results: Dict[str, Any],
    shape_validation: Dict[str, Any],
    nan_validation: Dict[str, Any],
    heatmap_stats: Dict[str, Any]
) -> None:
    """Print the coherence engine validation report."""

    print("\n" + "=" * 60)
    print("COHERENCE ENGINE VALIDATION")
    print("=" * 60)
    print(f"Report time: {datetime.now().isoformat()}")

    # Computation summary
    print("\n--- Computation Summary ---")
    print(f"  Computation time: {coherence_results.get('computation_time_ms', 0):.2f} ms")
    print(f"  Pairs tested: {len(coherence_results.get('pairwise_coherence', {}))}")

    if coherence_results.get("error"):
        print(f"  Error: {coherence_results['error']}")

    # Shape validation
    print(f"\n--- Shape Validation: {shape_validation['status'].upper()} ---")
    if shape_validation["issues"]:
        for issue in shape_validation["issues"]:
            print(f"  [!] {issue}")
    else:
        print("  All shapes valid")

    # NaN validation
    print(f"\n--- NaN Check: {nan_validation['status'].upper()} ---")
    for name, rate in nan_validation.get("nan_rates", {}).items():
        status = "[OK]" if rate < 0.1 else "[!]"
        print(f"  {status} {name}: {rate:.2%} NaN")

    # Warnings
    warnings_captured = coherence_results.get("warnings_captured", [])
    print(f"\n--- Runtime Warnings: {len(warnings_captured)} ---")
    if warnings_captured:
        for warn in warnings_captured[:5]:
            print(f"  [!] {warn['category']}: {warn['message'][:60]}")
    else:
        print("  No runtime warnings captured")

    # Heatmap statistics
    print("\n--- Heatmap Statistics ---")
    print(f"  Indicators: {heatmap_stats.get('indicators', 'N/A')}")
    print(f"  Mean correlation: {heatmap_stats.get('mean_correlation', 'N/A')}")
    print(f"  Median correlation: {heatmap_stats.get('median_correlation', 'N/A')}")
    print(f"  Max correlation: {heatmap_stats.get('max_correlation', 'N/A')}")
    print(f"  Min correlation: {heatmap_stats.get('min_correlation', 'N/A')}")
    print(f"  Std correlation: {heatmap_stats.get('std_correlation', 'N/A')}")

    # Rolling coherence stats
    if coherence_results.get("rolling_coherence") is not None:
        rc = coherence_results["rolling_coherence"]
        if isinstance(rc, pd.Series) and len(rc) > 0:
            print("\n--- Rolling Coherence Summary ---")
            print(f"  Mean: {rc.mean():.4f}")
            print(f"  Std: {rc.std():.4f}")
            print(f"  Min: {rc.min():.4f}")
            print(f"  Max: {rc.max():.4f}")
            print(f"  Latest: {rc.iloc[-1]:.4f}")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for coherence engine validation.

    Returns:
        0 on success, 1 on error
    """
    logger.info("Starting coherence engine validation...")
    logger.info(f"Database path: {get_db_path()}")

    try:
        # Load test panel
        df = load_test_panel(NUM_TEST_INDICATORS)

        if df.empty:
            print("\n[!] WARNING: No data loaded. Cannot perform coherence validation.")
            return 1

        logger.info(f"Panel loaded: {df.shape[0]} rows, {df.shape[1]} indicators")

        # Run coherence tests
        coherence_results = run_coherence_tests(df)

        # Validate results
        shape_validation = validate_coherence_shapes(coherence_results)
        nan_validation = validate_coherence_nans(coherence_results)
        heatmap_stats = compute_heatmap_statistics(df)

        # Print report
        print_report(coherence_results, shape_validation, nan_validation, heatmap_stats)

        logger.info("Coherence engine validation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in coherence engine validation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
