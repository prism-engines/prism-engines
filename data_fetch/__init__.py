"""
PRISM Engine - Data Fetch Registries
=====================================

This package contains the registry files that centralize all
data paths, dataset configurations, and series-level metadata.

Registries:
    - system_registry.json: Core paths and settings
    - market_registry.json: Yahoo Finance tickers
    - economic_registry.json: FRED economic indicators

Usage:
    from utils.fetch_validator import load_validated_registry

    system = load_validated_registry('system')
    market = load_validated_registry('market')
    economic = load_validated_registry('economic')
"""

import json
from pathlib import Path
from typing import Dict, Any


def _get_registry_path(name: str) -> Path:
    """Get path to a registry file."""
    return Path(__file__).parent / f"{name}_registry.json"


def load_registry(name: str) -> Dict[str, Any]:
    """
    Load a registry by name without validation.

    For production use, prefer utils.fetch_validator.load_validated_registry()

    Args:
        name: Registry name ('system', 'market', or 'economic')

    Returns:
        Registry data as dictionary
    """
    path = _get_registry_path(name)
    with open(path, 'r') as f:
        return json.load(f)


__all__ = ['load_registry']
