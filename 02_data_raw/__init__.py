"""
PRISM Engine - Stage 02: Raw Data Storage

This directory stores raw data as fetched from sources.
No transformations applied - exact API responses preserved.

Structure:
    financial/
        equities/     - Stock/ETF data
        bonds/        - Fixed income data
        commodities/  - Commodity futures
        macro/        - FRED economic indicators

    climate/
        temperature/  - Temperature anomaly data
        co2/          - Atmospheric CO2 levels
        sea_level/    - Sea level measurements
        ice/          - Ice extent data

    checkpoints/
        data_inventory.csv  - What we have
        raw_summary.html    - Visual overview
"""

from pathlib import Path

# Define data paths
DATA_RAW_ROOT = Path(__file__).parent

FINANCIAL_PATHS = {
    "equities": DATA_RAW_ROOT / "financial" / "equities",
    "bonds": DATA_RAW_ROOT / "financial" / "bonds",
    "commodities": DATA_RAW_ROOT / "financial" / "commodities",
    "macro": DATA_RAW_ROOT / "financial" / "macro",
}

CLIMATE_PATHS = {
    "temperature": DATA_RAW_ROOT / "climate" / "temperature",
    "co2": DATA_RAW_ROOT / "climate" / "co2",
    "sea_level": DATA_RAW_ROOT / "climate" / "sea_level",
    "ice": DATA_RAW_ROOT / "climate" / "ice",
}
