"""
PRISM Engine - Stage 04: Clean Data (Engine-Ready)

This directory stores cleaned, aligned data ready for analysis.

Structure:
    financial/
        panel_daily.parquet    - Daily frequency panel
        panel_weekly.parquet   - Weekly resampled
        metadata.json          - Column info, date ranges

    climate/
        panel_monthly.parquet  - Monthly frequency panel
        metadata.json          - Column info, date ranges

    checkpoints/
        correlation_matrix.html
        time_coverage.html
        data_quality_score.json
"""

from pathlib import Path
import pandas as pd
import json
from typing import Optional, Dict

# Define paths
DATA_CLEAN_ROOT = Path(__file__).parent

FINANCIAL_DIR = DATA_CLEAN_ROOT / "financial"
CLIMATE_DIR = DATA_CLEAN_ROOT / "climate"
CHECKPOINT_DIR = DATA_CLEAN_ROOT / "checkpoints"


def load_financial_panel(frequency: str = "daily") -> Optional[pd.DataFrame]:
    """
    Load cleaned financial panel.

    Args:
        frequency: 'daily' or 'weekly'

    Returns:
        DataFrame or None if not found
    """
    path = FINANCIAL_DIR / f"panel_{frequency}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_climate_panel() -> Optional[pd.DataFrame]:
    """
    Load cleaned climate panel.

    Returns:
        DataFrame or None if not found
    """
    path = CLIMATE_DIR / "panel_monthly.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def get_metadata(domain: str = "financial") -> Optional[Dict]:
    """
    Load metadata for a domain.

    Args:
        domain: 'financial' or 'climate'

    Returns:
        Metadata dictionary or None
    """
    if domain == "financial":
        path = FINANCIAL_DIR / "metadata.json"
    else:
        path = CLIMATE_DIR / "metadata.json"

    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_panel(
    df: pd.DataFrame,
    domain: str,
    frequency: str,
    metadata: Optional[Dict] = None
) -> Path:
    """
    Save a cleaned panel.

    Args:
        df: DataFrame to save
        domain: 'financial' or 'climate'
        frequency: 'daily', 'weekly', 'monthly'
        metadata: Optional metadata dict

    Returns:
        Path to saved file
    """
    if domain == "financial":
        directory = FINANCIAL_DIR
    else:
        directory = CLIMATE_DIR

    directory.mkdir(parents=True, exist_ok=True)

    # Save panel
    panel_path = directory / f"panel_{frequency}.parquet"
    df.to_parquet(panel_path)

    # Save metadata
    if metadata is None:
        metadata = {}

    metadata.update({
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "date_range": {
            "start": str(df["date"].min()) if "date" in df.columns else None,
            "end": str(df["date"].max()) if "date" in df.columns else None,
        }
    })

    meta_path = directory / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return panel_path
