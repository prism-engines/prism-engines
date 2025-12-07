"""
Runtime Panel Loader (Domain-Agnostic)
======================================

Panels are defined at runtime by the UI, not in files.
This module provides a simple interface to load any combination
of indicators as a panel.

Includes HVD (Hidden Variation Detector) integration to warn
about family duplicates. HVD is advisory-only and never modifies data.

Usage:
    from panel.runtime_loader import load_panel

    # UI sends a list of indicator names
    selected = ["sp500", "vix", "t10y2y"]
    panel = load_panel(selected)

    # Panel is a wide-format DataFrame
    # Index: date
    # Columns: indicator names
"""

from typing import List, Optional, Dict, Any
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _check_family_duplicates(
    indicator_names: List[str],
    panel_df: pd.DataFrame
) -> None:
    """
    Run HVD family divergence check and log warnings.

    This is advisory-only and never modifies data.

    Args:
        indicator_names: List of indicator names loaded
        panel_df: The loaded panel DataFrame
    """
    try:
        from analysis.hidden_variation_detector import detect_family_divergence
        from pathlib import Path
        import yaml

        # Load families registry
        project_root = Path(__file__).parent.parent
        families_path = project_root / "data" / "registry" / "families.yaml"

        if not families_path.exists():
            return

        with open(families_path, "r") as f:
            families_data = yaml.safe_load(f)

        families = families_data.get("families", {})

        # For each family, check if multiple members are in the panel
        for family_id, family_config in families.items():
            members_config = family_config.get("members", {})
            member_names = list(members_config.keys())

            # Find which members are in the loaded indicators
            members_in_panel = [m for m in member_names if m in indicator_names]

            if len(members_in_panel) >= 2:
                # Build df_dict for these members
                df_dict = {}
                for member in members_in_panel:
                    if member in panel_df.columns:
                        df_dict[member] = pd.DataFrame({
                            'date': panel_df.index,
                            'value': panel_df[member]
                        }).reset_index(drop=True)

                # Run HVD check
                warnings = detect_family_divergence(
                    indicator_family=family_id,
                    df_dict=df_dict,
                    threshold=family_config.get("rules", {}).get(
                        "correlation_warning_threshold", 0.70
                    )
                )

                # Log warnings (advisory only)
                for warning in warnings:
                    logger.warning(warning)

    except Exception as e:
        # HVD is advisory - never let it break panel loading
        logger.debug(f"HVD check skipped: {e}")


def load_panel(
    indicator_names: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    panel_name: Optional[str] = None,
    skip_hvd_check: bool = False,
) -> pd.DataFrame:
    """
    Load a panel of indicators at runtime.

    This is the domain-agnostic panel loader. Panels are defined
    by the UI at runtime as simple lists of indicator names.

    NO domain logic. NO file-based panel definitions.

    Includes HVD (Hidden Variation Detector) integration to log
    warnings about family duplicates. HVD is advisory-only.

    Args:
        indicator_names: List of indicator names to include
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        panel_name: Optional panel name for validation
        skip_hvd_check: Set to True to skip HVD family divergence check

    Returns:
        Wide-format DataFrame with:
            - Index: datetime
            - Columns: indicator names
            - Values: indicator values
    """
    # Never load climate indicators until project is active
    if panel_name and panel_name.lower().startswith("climate"):
        raise RuntimeError("Climate panels are currently frozen and inactive.")

    from data.sql.db_connector import load_all_indicators_wide

    panel_df = load_all_indicators_wide(
        indicators=indicator_names,
        start_date=start_date,
        end_date=end_date,
    )

    # Run HVD family divergence check (advisory only)
    if not skip_hvd_check and not panel_df.empty:
        _check_family_duplicates(indicator_names, panel_df)

    return panel_df


def list_available_indicators() -> List[str]:
    """
    List all available indicators in the database.

    Returns:
        List of indicator names
    """
    from data.sql.db_connector import list_indicators

    return list_indicators()


def get_indicator_info(indicator_name: str) -> Optional[dict]:
    """
    Get metadata for a specific indicator.

    Args:
        indicator_name: Name of the indicator

    Returns:
        Dict with indicator metadata, or None if not found
    """
    from data.sql.db_connector import get_indicator

    return get_indicator(indicator_name)


def load_calibrated_indicators_from_registry() -> List[str]:
    """
    Load indicator names from registry that have calibration enabled.

    Filters indicators by the 'calibration' tag or 'purpose' containing 'calibration'.
    Falls back to ALL indicators if none have calibration enabled.

    Returns:
        List of indicator names for calibration
    """
    from pathlib import Path
    import yaml

    project_root = Path(__file__).parent.parent
    registry_path = project_root / "data" / "registry" / "indicators.yaml"

    if not registry_path.exists():
        logger.warning(f"Registry not found at {registry_path}")
        return []

    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)

    if not registry:
        logger.warning("Registry is empty")
        return []

    # Filter indicators with calibration: true or purpose containing 'calibration'
    selected = []
    for k, v in registry.items():
        if isinstance(v, dict):
            # Check explicit calibration flag
            if v.get("calibration", False) is True:
                selected.append(k)
            # Also check purpose list for 'calibration'
            elif "calibration" in v.get("purpose", []):
                selected.append(k)

    # Safety fallback: if no indicators matched, use ALL indicators
    if len(selected) == 0:
        logger.warning("No indicators matched calibration filters — using ALL indicators.")
        selected = [k for k, v in registry.items() if isinstance(v, dict)]

    logger.info(f"Using {len(selected)} indicators for calibration")

    return selected


def load_calibrated_panel(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    skip_hvd_check: bool = False,
) -> pd.DataFrame:
    """
    Load a panel of calibration-enabled indicators.

    Convenience function that combines load_calibrated_indicators_from_registry()
    with load_panel().

    Args:
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        skip_hvd_check: Set to True to skip HVD family divergence check

    Returns:
        Wide-format DataFrame with calibrated indicators

    Raises:
        RuntimeError: If the resulting panel is empty
    """
    indicator_names = load_calibrated_indicators_from_registry()

    if not indicator_names:
        raise RuntimeError("No indicators available for calibration")

    panel_df = load_panel(
        indicator_names=indicator_names,
        start_date=start_date,
        end_date=end_date,
        skip_hvd_check=skip_hvd_check,
    )

    # Validate panel contains data
    if panel_df.empty:
        raise RuntimeError(
            "Calibration panel is empty — registry filters or DB paths likely incorrect."
        )

    return panel_df
