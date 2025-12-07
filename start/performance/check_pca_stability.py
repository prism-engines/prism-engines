#!/usr/bin/env python3
"""
PRISM PCA Stability Checker
============================

Purpose: Check PCA eigenvector drift over time.

This script:
- Runs PCA on rolling 3-year window of 20 indicators
- Measures angle change of PC1/PC2
- Marks anomaly if drift > 30 degrees

Output:
    PCA Stability Report
    Eigenvector drift median: X degrees
    Max drift: Y degrees

Usage:
    python start/performance/check_pca_stability.py

Notes:
    - Read-only: never writes to the database
    - Exits gracefully if indicators are missing
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

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
NUM_INDICATORS = 20
WINDOW_YEARS = 3
DRIFT_THRESHOLD_DEGREES = 30
STEP_SIZE_DAYS = 63  # Quarterly steps


def compute_eigenvector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two eigenvectors in degrees.

    Note: Eigenvectors can flip sign, so we check both orientations.

    Args:
        v1: First eigenvector
        v2: Second eigenvector

    Returns:
        Angle in degrees (0 to 90)
    """
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Compute dot product (cosine of angle)
    cos_angle = np.abs(np.dot(v1_norm, v2_norm))  # abs handles sign flips

    # Clamp to valid range for arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def run_pca(df: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
    """
    Run PCA on the panel data.

    Args:
        df: Panel DataFrame (indicators as columns)
        n_components: Number of principal components

    Returns:
        Dict with PCA results
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    results = {
        "components": None,
        "explained_variance_ratio": None,
        "n_samples": 0,
        "n_features": 0,
        "error": None
    }

    try:
        # Drop rows with any NaN
        df_clean = df.dropna()

        if len(df_clean) < 10:
            results["error"] = "Insufficient data after dropping NaN"
            return results

        results["n_samples"] = len(df_clean)
        results["n_features"] = len(df_clean.columns)

        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_clean)

        # Run PCA
        pca = PCA(n_components=min(n_components, len(df_clean.columns)))
        pca.fit(data_scaled)

        results["components"] = pca.components_
        results["explained_variance_ratio"] = pca.explained_variance_ratio_

    except Exception as e:
        results["error"] = str(e)

    return results


def compute_rolling_pca_drift(
    df: pd.DataFrame,
    window_days: int = 756,  # ~3 years
    step_days: int = STEP_SIZE_DAYS
) -> Dict[str, Any]:
    """
    Compute PCA drift over rolling windows.

    Args:
        df: Panel DataFrame
        window_days: Window size in trading days
        step_days: Step size between windows

    Returns:
        Dict with drift analysis results
    """
    results = {
        "pc1_drifts": [],
        "pc2_drifts": [],
        "dates": [],
        "windows_computed": 0,
        "errors": []
    }

    if len(df) < window_days + step_days:
        results["error"] = "Insufficient data for rolling PCA"
        return results

    prev_pca = None

    for i in range(0, len(df) - window_days, step_days):
        window_df = df.iloc[i:i + window_days]
        end_date = df.index[i + window_days - 1]

        pca_result = run_pca(window_df)

        if pca_result["error"]:
            results["errors"].append({
                "date": str(end_date),
                "error": pca_result["error"]
            })
            continue

        results["windows_computed"] += 1
        results["dates"].append(end_date)

        if prev_pca is not None and pca_result["components"] is not None:
            # Compute drift angles
            try:
                pc1_drift = compute_eigenvector_angle(
                    prev_pca["components"][0],
                    pca_result["components"][0]
                )
                results["pc1_drifts"].append(pc1_drift)

                if len(pca_result["components"]) > 1 and len(prev_pca["components"]) > 1:
                    pc2_drift = compute_eigenvector_angle(
                        prev_pca["components"][1],
                        pca_result["components"][1]
                    )
                    results["pc2_drifts"].append(pc2_drift)

            except Exception as e:
                results["errors"].append({
                    "date": str(end_date),
                    "error": f"Drift computation error: {e}"
                })

        prev_pca = pca_result

    return results


def analyze_drift(drift_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze drift patterns and detect anomalies.

    Args:
        drift_results: Results from compute_rolling_pca_drift

    Returns:
        Dict with drift analysis
    """
    analysis = {
        "pc1_median_drift": None,
        "pc1_max_drift": None,
        "pc1_anomalies": [],
        "pc2_median_drift": None,
        "pc2_max_drift": None,
        "pc2_anomalies": [],
        "status": "pass"
    }

    # Analyze PC1 drift
    if drift_results["pc1_drifts"]:
        drifts = np.array(drift_results["pc1_drifts"])
        analysis["pc1_median_drift"] = float(np.median(drifts))
        analysis["pc1_max_drift"] = float(np.max(drifts))

        # Find anomalies (drift > threshold)
        anomaly_mask = drifts > DRIFT_THRESHOLD_DEGREES
        anomaly_indices = np.where(anomaly_mask)[0]

        for idx in anomaly_indices:
            # Account for offset (drifts start at second window)
            date_idx = idx + 1 if idx + 1 < len(drift_results["dates"]) else idx
            analysis["pc1_anomalies"].append({
                "date": str(drift_results["dates"][date_idx]),
                "drift_degrees": float(drifts[idx])
            })

    # Analyze PC2 drift
    if drift_results["pc2_drifts"]:
        drifts = np.array(drift_results["pc2_drifts"])
        analysis["pc2_median_drift"] = float(np.median(drifts))
        analysis["pc2_max_drift"] = float(np.max(drifts))

        anomaly_mask = drifts > DRIFT_THRESHOLD_DEGREES
        anomaly_indices = np.where(anomaly_mask)[0]

        for idx in anomaly_indices:
            date_idx = idx + 1 if idx + 1 < len(drift_results["dates"]) else idx
            analysis["pc2_anomalies"].append({
                "date": str(drift_results["dates"][date_idx]),
                "drift_degrees": float(drifts[idx])
            })

    # Determine overall status
    if analysis["pc1_anomalies"] or analysis["pc2_anomalies"]:
        analysis["status"] = "warning"

    return analysis


def load_pca_panel(num_indicators: int = NUM_INDICATORS) -> pd.DataFrame:
    """
    Load a panel of indicators suitable for PCA analysis.

    Args:
        num_indicators: Number of indicators to load

    Returns:
        Wide-format DataFrame
    """
    try:
        df = load_all_indicators_wide()

        if df.empty:
            return df

        # Select indicators with best data coverage
        coverage = (~df.isna()).mean()
        best_cols = coverage.nlargest(num_indicators).index.tolist()

        return df[best_cols]

    except Exception as e:
        logger.error(f"Error loading PCA panel: {e}")
        return pd.DataFrame()


def print_report(
    drift_results: Dict[str, Any],
    drift_analysis: Dict[str, Any],
    panel_shape: Tuple[int, int]
) -> None:
    """Print the PCA stability report."""

    print("\n" + "=" * 60)
    print("PCA STABILITY REPORT")
    print("=" * 60)
    print(f"Report time: {datetime.now().isoformat()}")

    # Panel info
    print(f"\n--- Data Summary ---")
    print(f"  Panel shape: {panel_shape[0]} rows x {panel_shape[1]} indicators")
    print(f"  Window size: {WINDOW_YEARS} years (~756 trading days)")
    print(f"  Step size: {STEP_SIZE_DAYS} days (quarterly)")
    print(f"  Windows computed: {drift_results.get('windows_computed', 0)}")

    # PC1 results
    print(f"\n--- PC1 Drift Analysis ---")
    if drift_analysis["pc1_median_drift"] is not None:
        print(f"  Median drift: {drift_analysis['pc1_median_drift']:.2f} degrees")
        print(f"  Max drift: {drift_analysis['pc1_max_drift']:.2f} degrees")

        if drift_analysis["pc1_anomalies"]:
            print(f"  Anomalies (>{DRIFT_THRESHOLD_DEGREES} deg): {len(drift_analysis['pc1_anomalies'])}")
            for anom in drift_analysis["pc1_anomalies"][:5]:
                print(f"    - {anom['date']}: {anom['drift_degrees']:.2f} degrees")
        else:
            print(f"  No anomalies detected (threshold: {DRIFT_THRESHOLD_DEGREES} degrees)")
    else:
        print("  Insufficient data for PC1 analysis")

    # PC2 results
    print(f"\n--- PC2 Drift Analysis ---")
    if drift_analysis["pc2_median_drift"] is not None:
        print(f"  Median drift: {drift_analysis['pc2_median_drift']:.2f} degrees")
        print(f"  Max drift: {drift_analysis['pc2_max_drift']:.2f} degrees")

        if drift_analysis["pc2_anomalies"]:
            print(f"  Anomalies (>{DRIFT_THRESHOLD_DEGREES} deg): {len(drift_analysis['pc2_anomalies'])}")
            for anom in drift_analysis["pc2_anomalies"][:5]:
                print(f"    - {anom['date']}: {anom['drift_degrees']:.2f} degrees")
        else:
            print(f"  No anomalies detected (threshold: {DRIFT_THRESHOLD_DEGREES} degrees)")
    else:
        print("  Insufficient data for PC2 analysis")

    # Errors
    if drift_results.get("errors"):
        print(f"\n--- Errors ({len(drift_results['errors'])}) ---")
        for err in drift_results["errors"][:3]:
            print(f"  - {err['date']}: {err['error']}")

    # Overall status
    print(f"\n--- Overall Status: {drift_analysis['status'].upper()} ---")
    if drift_analysis["status"] == "warning":
        print("  Significant eigenvector drift detected")
        print("  This may indicate regime changes in the market")
    else:
        print("  PCA structure is stable")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for PCA stability check.

    Returns:
        0 on success, 1 on error
    """
    logger.info("Starting PCA stability check...")
    logger.info(f"Database path: {get_db_path()}")

    try:
        # Load panel
        df = load_pca_panel(NUM_INDICATORS)

        if df.empty:
            print("\n[!] WARNING: No data loaded. Cannot perform PCA stability check.")
            return 1

        logger.info(f"Panel loaded: {df.shape[0]} rows, {df.shape[1]} indicators")

        # Check minimum data requirement
        min_required = 756 + STEP_SIZE_DAYS  # 3 years + one step
        if len(df) < min_required:
            print(f"\n[!] WARNING: Insufficient data ({len(df)} rows). Need at least {min_required} for rolling PCA.")
            return 1

        # Compute rolling PCA drift
        drift_results = compute_rolling_pca_drift(df)

        # Analyze drift
        drift_analysis = analyze_drift(drift_results)

        # Print report
        print_report(drift_results, drift_analysis, df.shape)

        logger.info("PCA stability check completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in PCA stability check: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
