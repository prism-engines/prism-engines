#!/usr/bin/env python3
"""
PRISM Hidden Variation Sanity Check
====================================

Purpose: Integrate HiddenVariationDetector for family cluster analysis.

This script:
- Runs HVD detector on family clusters
- Verifies latent-factor stability
- Prints top 3 hidden components per family
- Warns if multi-resolution signals collapse to same vector

Output:
    HIDDEN VARIATION SANITY CHECK
    Family: SPX
      Top components: [...]
    Warning: Collapsed signals detected

Usage:
    python start/performance/hidden_variation_sanity_check.py

Notes:
    - Read-only: never writes to the database
    - HVD is advisory-only and never modifies data
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
from analysis.hidden_variation_detector import (
    build_hvd_report,
    HVDReport,
    ClusterResult,
    FamilyDivergence
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
TOP_COMPONENTS_PER_FAMILY = 3
COLLAPSE_CORRELATION_THRESHOLD = 0.95


def load_full_panel() -> pd.DataFrame:
    """
    Load the full panel for HVD analysis.

    Returns:
        Wide-format DataFrame
    """
    try:
        df = load_all_indicators_wide()
        return df
    except Exception as e:
        logger.error(f"Error loading panel: {e}")
        return pd.DataFrame()


def get_family_indicators() -> Dict[str, List[str]]:
    """
    Get all families and their member indicators.

    Returns:
        Dict mapping family_id to list of member indicator names
    """
    family_indicators = {}

    try:
        fm = FamilyManager()
        families = fm.list_families()

        for family in families:
            family_id = family.get("id", "")
            members = family.get("members", {})

            if isinstance(members, dict):
                member_names = list(members.keys())
            elif isinstance(members, list):
                member_names = members
            else:
                member_names = []

            if member_names:
                family_indicators[family_id] = member_names

    except Exception as e:
        logger.error(f"Error loading families: {e}")

    return family_indicators


def run_hvd_on_panel(df: pd.DataFrame) -> Optional[HVDReport]:
    """
    Run HVD analysis on the full panel.

    Args:
        df: Panel DataFrame

    Returns:
        HVDReport or None on error
    """
    try:
        fm = FamilyManager()
        report = build_hvd_report(
            panel=df,
            family_manager=fm,
            similarity_threshold=0.90,
            cluster_method='auto',
            n_clusters=4,
            min_overlap=252
        )
        return report
    except Exception as e:
        logger.error(f"Error running HVD: {e}")
        return None


def compute_family_pca_components(
    df: pd.DataFrame,
    family_indicators: Dict[str, List[str]],
    n_components: int = TOP_COMPONENTS_PER_FAMILY
) -> Dict[str, Any]:
    """
    Compute PCA components for each family cluster.

    Args:
        df: Panel DataFrame
        family_indicators: Dict of family_id -> member indicators
        n_components: Number of components to extract

    Returns:
        Dict with family component analysis
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    results = {}

    for family_id, members in family_indicators.items():
        # Get available members in panel
        available_members = [m for m in members if m in df.columns]

        if len(available_members) < 2:
            results[family_id] = {
                "status": "skip",
                "note": f"Only {len(available_members)} member(s) available"
            }
            continue

        try:
            # Extract family data
            family_df = df[available_members].dropna()

            if len(family_df) < 50:
                results[family_id] = {
                    "status": "skip",
                    "note": "Insufficient overlapping data"
                }
                continue

            # Standardize
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(family_df)

            # Run PCA
            n_comp = min(n_components, len(available_members))
            pca = PCA(n_components=n_comp)
            pca.fit(data_scaled)

            # Get loadings (which indicators contribute to each component)
            loadings = pd.DataFrame(
                pca.components_.T,
                index=available_members,
                columns=[f"PC{i+1}" for i in range(n_comp)]
            )

            # Top contributors per component
            top_contributors = {}
            for pc in loadings.columns:
                abs_loadings = loadings[pc].abs().sort_values(ascending=False)
                top_contributors[pc] = [
                    {"indicator": idx, "loading": float(loadings.loc[idx, pc])}
                    for idx in abs_loadings.head(3).index
                ]

            results[family_id] = {
                "status": "ok",
                "members_analyzed": available_members,
                "variance_explained": [float(v) for v in pca.explained_variance_ratio_],
                "top_contributors": top_contributors
            }

        except Exception as e:
            results[family_id] = {
                "status": "error",
                "error": str(e)
            }

    return results


def detect_signal_collapse(
    df: pd.DataFrame,
    family_indicators: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Detect multi-resolution signals that collapse to the same vector.

    This happens when different resolution members (e.g., daily vs monthly)
    are so highly correlated they provide no additional information.

    Args:
        df: Panel DataFrame
        family_indicators: Dict of family_id -> member indicators

    Returns:
        List of collapse warnings
    """
    warnings = []

    for family_id, members in family_indicators.items():
        available_members = [m for m in members if m in df.columns]

        if len(available_members) < 2:
            continue

        try:
            # Check pairwise correlations
            for i, m1 in enumerate(available_members):
                for m2 in available_members[i+1:]:
                    # Get overlapping data
                    mask = df[m1].notna() & df[m2].notna()
                    if mask.sum() < 100:
                        continue

                    corr = df.loc[mask, m1].corr(df.loc[mask, m2])

                    if abs(corr) > COLLAPSE_CORRELATION_THRESHOLD:
                        warnings.append({
                            "family_id": family_id,
                            "member_1": m1,
                            "member_2": m2,
                            "correlation": float(corr),
                            "message": f"Signals may collapse: {m1} and {m2} have {abs(corr):.3f} correlation"
                        })

        except Exception as e:
            logger.debug(f"Error checking collapse for {family_id}: {e}")

    return warnings


def analyze_hvd_clusters(hvd_report: Optional[HVDReport]) -> Dict[str, Any]:
    """
    Analyze HVD cluster results.

    Args:
        hvd_report: HVDReport from build_hvd_report

    Returns:
        Dict with cluster analysis
    """
    analysis = {
        "total_clusters": 0,
        "cluster_summary": [],
        "high_similarity_pairs": 0,
        "family_divergences": 0,
        "warnings": []
    }

    if hvd_report is None:
        analysis["error"] = "No HVD report available"
        return analysis

    # Cluster summary
    analysis["total_clusters"] = len(hvd_report.clusters)

    for cluster in hvd_report.clusters:
        analysis["cluster_summary"].append({
            "cluster_id": cluster.cluster_id,
            "members": cluster.members[:5],  # Truncate for display
            "size": len(cluster.members),
            "method": cluster.method,
            "intra_correlation": cluster.intra_cluster_correlation
        })

    # High similarity pairs
    analysis["high_similarity_pairs"] = len(hvd_report.high_similarity_pairs)

    # Family divergences
    analysis["family_divergences"] = len(hvd_report.family_divergences)

    # Warnings
    analysis["warnings"] = hvd_report.warnings[:10]  # First 10 warnings

    return analysis


def print_report(
    family_components: Dict[str, Any],
    collapse_warnings: List[Dict[str, Any]],
    hvd_analysis: Dict[str, Any],
    panel_shape: Tuple[int, int]
) -> None:
    """Print the hidden variation sanity check report."""

    print("\n" + "=" * 60)
    print("HIDDEN VARIATION SANITY CHECK")
    print("=" * 60)
    print(f"Report time: {datetime.now().isoformat()}")
    print(f"Panel: {panel_shape[0]} rows x {panel_shape[1]} indicators")

    # Family component analysis
    print(f"\n--- Family Component Analysis ({len(family_components)} families) ---")

    for family_id, result in list(family_components.items())[:10]:
        status = result.get("status", "unknown")

        if status == "ok":
            print(f"\n  Family: {family_id}")
            print(f"    Members analyzed: {len(result['members_analyzed'])}")
            print(f"    Variance explained: {[f'{v:.1%}' for v in result['variance_explained']]}")

            # Top 3 components
            for pc, contributors in list(result['top_contributors'].items())[:TOP_COMPONENTS_PER_FAMILY]:
                top_inds = [c['indicator'] for c in contributors[:2]]
                print(f"    {pc} top contributors: {top_inds}")

        elif status == "skip":
            print(f"\n  Family: {family_id} - SKIPPED ({result.get('note', 'N/A')})")
        else:
            print(f"\n  Family: {family_id} - ERROR ({result.get('error', 'Unknown')})")

    # Signal collapse warnings
    print(f"\n--- Signal Collapse Detection ---")
    if collapse_warnings:
        print(f"  [!] {len(collapse_warnings)} potential signal collapse(s) detected:")
        for warn in collapse_warnings[:5]:
            print(f"      {warn['family_id']}: {warn['member_1']} <-> {warn['member_2']}")
            print(f"        Correlation: {warn['correlation']:.3f}")
    else:
        print("  No signal collapse detected")

    # HVD cluster analysis
    print(f"\n--- HVD Cluster Analysis ---")
    print(f"  Total clusters: {hvd_analysis.get('total_clusters', 'N/A')}")
    print(f"  High similarity pairs: {hvd_analysis.get('high_similarity_pairs', 'N/A')}")
    print(f"  Family divergences: {hvd_analysis.get('family_divergences', 'N/A')}")

    if hvd_analysis.get("cluster_summary"):
        print("  Cluster summary:")
        for cluster in hvd_analysis["cluster_summary"][:5]:
            print(f"    - Cluster {cluster['cluster_id']}: {cluster['size']} members")
            if cluster.get('intra_correlation'):
                print(f"      Intra-cluster correlation: {cluster['intra_correlation']:.3f}")

    # HVD warnings
    if hvd_analysis.get("warnings"):
        print(f"\n--- HVD Warnings ({len(hvd_analysis['warnings'])}) ---")
        for warn in hvd_analysis["warnings"][:5]:
            print(f"  [!] {warn[:80]}...")

    # Overall status
    print(f"\n--- Overall Status ---")
    issues = len(collapse_warnings) + hvd_analysis.get("family_divergences", 0)
    if issues > 0:
        print(f"  [!] WARNING: {issues} issue(s) detected")
        print("  Review family configurations and multi-resolution signals")
    else:
        print("  [OK] No significant hidden variation issues detected")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for hidden variation sanity check.

    Returns:
        0 on success, 1 on error
    """
    logger.info("Starting hidden variation sanity check...")
    logger.info(f"Database path: {get_db_path()}")

    try:
        # Load panel
        df = load_full_panel()

        if df.empty:
            print("\n[!] WARNING: No data loaded. Cannot perform HVD sanity check.")
            return 1

        logger.info(f"Panel loaded: {df.shape[0]} rows, {df.shape[1]} indicators")

        # Get family indicators
        family_indicators = get_family_indicators()
        logger.info(f"Found {len(family_indicators)} families")

        # Run HVD analysis
        logger.info("Running HVD analysis...")
        hvd_report = run_hvd_on_panel(df)

        # Compute family PCA components
        logger.info("Computing family PCA components...")
        family_components = compute_family_pca_components(df, family_indicators)

        # Detect signal collapse
        logger.info("Detecting signal collapse...")
        collapse_warnings = detect_signal_collapse(df, family_indicators)

        # Analyze HVD clusters
        hvd_analysis = analyze_hvd_clusters(hvd_report)

        # Print report
        print_report(family_components, collapse_warnings, hvd_analysis, df.shape)

        logger.info("Hidden variation sanity check completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in hidden variation sanity check: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
