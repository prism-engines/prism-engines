"""
PRISM Registry Module
=====================

Loads and validates the metric registry for the PRISM Engine.

The registry defines all indicators, their sources, frequencies, and metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

REGISTRY_DIR = Path(__file__).parent


def load_metric_registry() -> Dict[str, Any]:
    """
    Load the complete metric registry from JSON files.

    Returns:
        Dictionary containing all registry data with keys:
            - version: Registry version string
            - market: List of market indicator definitions
            - economic: List of economic indicator definitions
            - synthetic: List of synthetic indicator definitions
            - technical: List of technical indicator definitions
    """
    registry: Dict[str, Any] = {
        "version": "1.0.0",
        "market": [],
        "economic": [],
        "synthetic": [],
        "technical": [],
    }

    # Load market registry
    market_path = REGISTRY_DIR / "market_registry.json"
    if market_path.exists():
        with open(market_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                registry["market"] = data
            elif isinstance(data, dict):
                registry["market"] = data.get("indicators", data.get("market", []))
                if "version" in data:
                    registry["version"] = data["version"]

    # Load economic registry
    economic_path = REGISTRY_DIR / "economic_registry.json"
    if economic_path.exists():
        with open(economic_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                registry["economic"] = data
            elif isinstance(data, dict):
                registry["economic"] = data.get("indicators", data.get("economic", []))

    # Load metric registry (may contain synthetic/technical)
    metric_path = REGISTRY_DIR / "metric_registry.json"
    if metric_path.exists():
        with open(metric_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                registry["synthetic"] = data.get("synthetic", [])
                registry["technical"] = data.get("technical", [])
                if "version" in data:
                    registry["version"] = data["version"]

    # Load system registry for additional metadata
    system_path = REGISTRY_DIR / "system_registry.json"
    if system_path.exists():
        with open(system_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                registry["system"] = data

    logger.info(
        f"Loaded registry v{registry['version']}: "
        f"{len(registry['market'])} market, "
        f"{len(registry['economic'])} economic, "
        f"{len(registry['synthetic'])} synthetic, "
        f"{len(registry['technical'])} technical"
    )

    return registry


def validate_registry(registry: Dict[str, Any]) -> bool:
    """
    Validate the metric registry structure and contents.

    Args:
        registry: The registry dictionary to validate

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails with details about the issue
    """
    required_keys = ["version", "market", "economic"]

    for key in required_keys:
        if key not in registry:
            raise ValueError(f"Registry missing required key: {key}")

    # Validate market indicators
    for i, indicator in enumerate(registry.get("market", [])):
        if not isinstance(indicator, dict):
            raise ValueError(f"Market indicator {i} is not a dictionary")
        if "name" not in indicator and "ticker" not in indicator:
            raise ValueError(f"Market indicator {i} missing 'name' or 'ticker'")

    # Validate economic indicators
    for i, indicator in enumerate(registry.get("economic", [])):
        if not isinstance(indicator, dict):
            raise ValueError(f"Economic indicator {i} is not a dictionary")
        if "name" not in indicator and "series_id" not in indicator:
            raise ValueError(f"Economic indicator {i} missing 'name' or 'series_id'")

    logger.info("Registry validation passed")
    return True


def get_market_tickers(registry: Dict[str, Any]) -> List[str]:
    """
    Extract Yahoo Finance tickers from the market registry.

    Args:
        registry: The loaded registry dictionary

    Returns:
        List of ticker symbols
    """
    tickers = []
    for indicator in registry.get("market", []):
        ticker = indicator.get("ticker") or indicator.get("name")
        if ticker:
            tickers.append(ticker)
    return tickers


def get_fred_series(registry: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract FRED series information from the economic registry.

    Args:
        registry: The loaded registry dictionary

    Returns:
        List of dictionaries with 'name' and 'series_id' keys
    """
    series = []
    for indicator in registry.get("economic", []):
        name = indicator.get("name", "").lower()
        series_id = indicator.get("series_id") or indicator.get("fred_id")
        if name and series_id:
            series.append({"name": name, "series_id": series_id})
    return series
