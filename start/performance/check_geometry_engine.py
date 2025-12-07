#!/usr/bin/env python3
"""
PRISM Geometry Engine Health Checker
=====================================

Purpose: Ensure geometry input is valid before running coherence/lenses.

This script checks:
- Missing dates (gaps in time series)
- NaN rates > 1%
- Monotonic date index
- Families do not double-load members

Output:
    GEOMETRY ENGINE HEALTH
    [check] No gaps
    [check] No NaN inflation
    [warn] Warning: indicator X drifting

Usage:
    python start/performance/check_geometry_engine.py

Notes:
    - Read-only: never writes to the database
    - Exits gracefully if indicators are missing
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

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


# Thresholds
NAN_RATE_THRESHOLD = 0.01  # 1%
MAX_GAP_DAYS = 5  # Maximum allowed gap in trading days


def load_geometry_panel() -> pd.DataFrame:
    """
    Load the full panel for geometry analysis.

    Returns:
        Wide-format DataFrame with all indicators
    """
    try:
        df = load_all_indicators_wide()
        if df.empty:
            logger.warning("Empty panel loaded")
        return df
    except Exception as e:
        logger.error(f"Error loading panel: {e}")
        return pd.DataFrame()


def check_missing_dates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for gaps in the date index.

    Args:
        df: Panel DataFrame with date index

    Returns:
        Dict with gap analysis results
    """
    results = {
        "status": "pass",
        "gaps_found": 0,
        "gap_details": [],
        "max_gap_days": 0
    }

    if df.empty or df.index is None:
        results["status"] = "skip"
        results["note"] = "Empty DataFrame"
        return results

    try:
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort index
        df = df.sort_index()

        # Calculate date differences
        date_diffs = df.index.to_series().diff()

        # Find gaps (excluding weekends - allow up to 3 days for regular weekends)
        # For trading data, gaps > 5 calendar days are suspicious
        gaps = date_diffs[date_diffs > pd.Timedelta(days=MAX_GAP_DAYS)]

        results["gaps_found"] = len(gaps)

        if len(gaps) > 0:
            results["status"] = "warning"
            results["max_gap_days"] = gaps.max().days

            # Record first 10 gaps
            for idx, gap in gaps.head(10).items():
                results["gap_details"].append({
                    "date": str(idx.date()),
                    "gap_days": gap.days
                })

    except Exception as e:
        logger.error(f"Error checking missing dates: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def check_nan_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check NaN rates for all indicators.

    Args:
        df: Panel DataFrame

    Returns:
        Dict with NaN rate analysis
    """
    results = {
        "status": "pass",
        "indicators_checked": 0,
        "high_nan_indicators": [],
        "overall_nan_rate": 0.0
    }

    if df.empty:
        results["status"] = "skip"
        results["note"] = "Empty DataFrame"
        return results

    try:
        results["indicators_checked"] = len(df.columns)

        # Calculate NaN rates per column
        nan_rates = df.isna().mean()

        # Overall NaN rate
        results["overall_nan_rate"] = df.isna().sum().sum() / (df.shape[0] * df.shape[1])

        # Find indicators with high NaN rates
        high_nan = nan_rates[nan_rates > NAN_RATE_THRESHOLD]

        if len(high_nan) > 0:
            results["status"] = "warning"
            for indicator, rate in high_nan.items():
                results["high_nan_indicators"].append({
                    "indicator": indicator,
                    "nan_rate": float(rate),
                    "nan_percentage": f"{rate * 100:.2f}%"
                })

    except Exception as e:
        logger.error(f"Error checking NaN rates: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def check_monotonic_index(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ensure the date index is monotonically increasing.

    Args:
        df: Panel DataFrame

    Returns:
        Dict with monotonicity check results
    """
    results = {
        "status": "pass",
        "is_monotonic": True,
        "duplicate_dates": 0,
        "out_of_order_dates": 0
    }

    if df.empty or df.index is None:
        results["status"] = "skip"
        results["note"] = "Empty DataFrame"
        return results

    try:
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        results["duplicate_dates"] = int(duplicates)

        # Check monotonicity
        is_monotonic = df.index.is_monotonic_increasing
        results["is_monotonic"] = bool(is_monotonic)

        if not is_monotonic:
            # Count out-of-order entries
            df_sorted = df.sort_index()
            out_of_order = (df.index != df_sorted.index).sum()
            results["out_of_order_dates"] = int(out_of_order)

        if duplicates > 0 or not is_monotonic:
            results["status"] = "warning"

    except Exception as e:
        logger.error(f"Error checking monotonic index: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def check_family_double_loading(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if families accidentally load multiple members.

    Args:
        df: Panel DataFrame

    Returns:
        Dict with family overlap analysis
    """
    results = {
        "status": "pass",
        "families_checked": 0,
        "double_loaded_families": [],
        "correlation_warnings": []
    }

    if df.empty:
        results["status"] = "skip"
        results["note"] = "Empty DataFrame"
        return results

    try:
        # Load family manager
        fm = FamilyManager()
        families = fm.list_families()
        results["families_checked"] = len(families)

        loaded_indicators = set(df.columns)

        for family in families:
            family_id = family.get("id", "")
            members = family.get("members", [])

            if isinstance(members, dict):
                member_names = list(members.keys())
            elif isinstance(members, list):
                member_names = members
            else:
                continue

            # Find which members are loaded
            loaded_members = [m for m in member_names if m in loaded_indicators]

            if len(loaded_members) > 1:
                results["double_loaded_families"].append({
                    "family_id": family_id,
                    "members_loaded": loaded_members
                })

                # Check correlation between loaded members
                try:
                    for i in range(len(loaded_members)):
                        for j in range(i + 1, len(loaded_members)):
                            m1, m2 = loaded_members[i], loaded_members[j]
                            if m1 in df.columns and m2 in df.columns:
                                corr = df[m1].corr(df[m2])
                                if not np.isnan(corr) and abs(corr) > 0.9:
                                    results["correlation_warnings"].append({
                                        "family_id": family_id,
                                        "member_1": m1,
                                        "member_2": m2,
                                        "correlation": float(corr)
                                    })
                except Exception as e:
                    logger.debug(f"Could not compute correlation: {e}")

        if results["double_loaded_families"]:
            results["status"] = "warning"

    except Exception as e:
        logger.error(f"Error checking family double-loading: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def check_indicator_drift(df: pd.DataFrame, window: int = 252) -> Dict[str, Any]:
    """
    Check for indicators that may be drifting (unusual recent behavior).

    Args:
        df: Panel DataFrame
        window: Rolling window size

    Returns:
        Dict with drift analysis
    """
    results = {
        "status": "pass",
        "indicators_checked": 0,
        "drifting_indicators": []
    }

    if df.empty or len(df) < window * 2:
        results["status"] = "skip"
        results["note"] = "Insufficient data for drift analysis"
        return results

    try:
        results["indicators_checked"] = len(df.columns)

        for col in df.columns:
            try:
                series = df[col].dropna()
                if len(series) < window * 2:
                    continue

                # Compare recent window to historical
                recent = series.iloc[-window:]
                historical = series.iloc[:-window]

                # Z-score of recent mean relative to historical
                hist_mean = historical.mean()
                hist_std = historical.std()

                if hist_std > 0:
                    recent_mean = recent.mean()
                    z_score = (recent_mean - hist_mean) / hist_std

                    if abs(z_score) > 3:
                        results["drifting_indicators"].append({
                            "indicator": col,
                            "z_score": float(z_score),
                            "recent_mean": float(recent_mean),
                            "historical_mean": float(hist_mean)
                        })

            except Exception as e:
                logger.debug(f"Could not check drift for {col}: {e}")

        if results["drifting_indicators"]:
            results["status"] = "warning"

    except Exception as e:
        logger.error(f"Error checking indicator drift: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


def print_report(
    gap_results: Dict[str, Any],
    nan_results: Dict[str, Any],
    monotonic_results: Dict[str, Any],
    family_results: Dict[str, Any],
    drift_results: Dict[str, Any]
) -> None:
    """Print the geometry engine health report."""

    def status_symbol(status: str) -> str:
        if status == "pass":
            return "[OK]"
        elif status == "warning":
            return "[!]"
        elif status == "error":
            return "[X]"
        else:
            return "[-]"

    print("\n" + "=" * 60)
    print("GEOMETRY ENGINE HEALTH")
    print("=" * 60)
    print(f"Report time: {datetime.now().isoformat()}")

    # Gap check
    print(f"\n{status_symbol(gap_results['status'])} Date Gaps Check")
    if gap_results["status"] == "pass":
        print("    No significant gaps detected")
    elif gap_results["status"] == "warning":
        print(f"    Gaps found: {gap_results['gaps_found']}")
        print(f"    Max gap: {gap_results['max_gap_days']} days")
        for gap in gap_results["gap_details"][:3]:
            print(f"      - {gap['date']}: {gap['gap_days']} days")

    # NaN check
    print(f"\n{status_symbol(nan_results['status'])} NaN Rate Check")
    if nan_results["status"] == "pass":
        print(f"    No NaN inflation (overall rate: {nan_results['overall_nan_rate']:.4%})")
    elif nan_results["status"] == "warning":
        print(f"    High NaN indicators: {len(nan_results['high_nan_indicators'])}")
        for ind in nan_results["high_nan_indicators"][:5]:
            print(f"      - {ind['indicator']}: {ind['nan_percentage']}")

    # Monotonic check
    print(f"\n{status_symbol(monotonic_results['status'])} Monotonic Index Check")
    if monotonic_results["status"] == "pass":
        print("    Index is monotonically increasing")
    elif monotonic_results["status"] == "warning":
        print(f"    Duplicate dates: {monotonic_results['duplicate_dates']}")
        print(f"    Out of order: {monotonic_results['out_of_order_dates']}")

    # Family check
    print(f"\n{status_symbol(family_results['status'])} Family Double-Loading Check")
    if family_results["status"] == "pass":
        print(f"    No double-loaded families ({family_results['families_checked']} families checked)")
    elif family_results["status"] == "warning":
        print(f"    Double-loaded families: {len(family_results['double_loaded_families'])}")
        for fam in family_results["double_loaded_families"][:3]:
            print(f"      - {fam['family_id']}: {fam['members_loaded']}")
        if family_results["correlation_warnings"]:
            print("    High correlation warnings:")
            for warn in family_results["correlation_warnings"][:3]:
                print(f"      - {warn['member_1']} <-> {warn['member_2']}: {warn['correlation']:.3f}")

    # Drift check
    print(f"\n{status_symbol(drift_results['status'])} Indicator Drift Check")
    if drift_results["status"] == "pass":
        print(f"    No drifting indicators ({drift_results['indicators_checked']} checked)")
    elif drift_results["status"] == "warning":
        print(f"    Warning: {len(drift_results['drifting_indicators'])} indicator(s) drifting")
        for ind in drift_results["drifting_indicators"][:5]:
            print(f"      - {ind['indicator']}: z-score = {ind['z_score']:.2f}")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for geometry engine health check.

    Returns:
        0 on success, 1 on error
    """
    logger.info("Starting geometry engine health check...")
    logger.info(f"Database path: {get_db_path()}")

    try:
        # Load panel
        df = load_geometry_panel()

        if df.empty:
            print("\n[!] WARNING: No data loaded. Cannot perform geometry checks.")
            return 1

        logger.info(f"Panel loaded: {df.shape[0]} rows, {df.shape[1]} indicators")

        # Run all checks
        gap_results = check_missing_dates(df)
        nan_results = check_nan_rates(df)
        monotonic_results = check_monotonic_index(df)
        family_results = check_family_double_loading(df)
        drift_results = check_indicator_drift(df)

        # Print report
        print_report(gap_results, nan_results, monotonic_results, family_results, drift_results)

        logger.info("Geometry engine health check completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in geometry engine health check: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
