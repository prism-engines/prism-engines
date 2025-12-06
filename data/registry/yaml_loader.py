"""
YAML Registry Loader for PRISM Engine
======================================

Loads and merges all YAML registry files from the registry/yaml/ directory.
Provides utilities for accessing indicators by source, group, or category.

This module supports the Full Institutional Pack (250+ indicators).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)

REGISTRY_DIR = Path(__file__).parent
YAML_DIR = REGISTRY_DIR / "yaml"

# Valid indicator sources
VALID_SOURCES = {"yahoo", "fred", "synthetic", "technical", "stooq"}

# Registry categories (file stems)
REGISTRY_FILES = [
    "market_global",
    "rates_us",
    "rates_global",
    "credit",
    "liquidity",
    "fx",
    "commodities",
    "volatility",
    "economics_us",
    "economics_global",
    "synthetic",
    "technical",
]


class RegistryLoadError(Exception):
    """Raised when registry loading fails."""
    pass


class DuplicateIndicatorError(Exception):
    """Raised when duplicate indicator IDs are found."""
    pass


def load_yaml_file(filepath: Path) -> Dict[str, Any]:
    """
    Load a single YAML file.

    Args:
        filepath: Path to the YAML file

    Returns:
        Parsed YAML contents as dictionary

    Raises:
        RegistryLoadError: If file cannot be loaded
    """
    try:
        with open(filepath) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise RegistryLoadError(f"Invalid YAML in {filepath}: {e}")
    except FileNotFoundError:
        raise RegistryLoadError(f"Registry file not found: {filepath}")
    except Exception as e:
        raise RegistryLoadError(f"Error loading {filepath}: {e}")


def load_all_yaml_registries() -> Dict[str, Any]:
    """
    Load all YAML registry files and merge into a unified registry.

    Returns:
        Merged registry dictionary with structure:
        {
            "version": str,
            "indicators": List[Dict],  # All indicators flattened
            "by_source": Dict[str, List[Dict]],  # Grouped by source
            "by_group": Dict[str, List[Dict]],  # Grouped by group
            "by_category": Dict[str, List[Dict]],  # Original file category
            "technical_rules": Dict,  # Auto-apply rules for technicals
            "metadata": Dict,  # Registry metadata
        }

    Raises:
        RegistryLoadError: If loading fails
        DuplicateIndicatorError: If duplicate IDs are found
    """
    if not YAML_DIR.exists():
        raise RegistryLoadError(f"YAML registry directory not found: {YAML_DIR}")

    registry: Dict[str, Any] = {
        "version": "2.0.0",
        "indicators": [],
        "by_source": {},
        "by_group": {},
        "by_category": {},
        "technical_rules": {},
        "metadata": {
            "files_loaded": [],
            "total_indicators": 0,
        },
    }

    seen_ids: Set[str] = set()
    all_indicators: List[Dict[str, Any]] = []

    # Load each YAML file
    for file_stem in REGISTRY_FILES:
        filepath = YAML_DIR / f"{file_stem}.yaml"

        if not filepath.exists():
            logger.warning(f"Registry file not found: {filepath}")
            continue

        try:
            data = load_yaml_file(filepath)
        except RegistryLoadError as e:
            logger.error(str(e))
            continue

        registry["metadata"]["files_loaded"].append(file_stem)

        # Store technical auto-apply rules if present
        if "auto_apply_rules" in data:
            registry["technical_rules"] = data["auto_apply_rules"]

        # Process indicators
        indicators = data.get("indicators", [])
        category_indicators = []

        for ind in indicators:
            ind_id = ind.get("id")

            if not ind_id:
                logger.warning(f"Indicator without ID in {file_stem}: {ind}")
                continue

            # Check for duplicates
            if ind_id in seen_ids:
                raise DuplicateIndicatorError(
                    f"Duplicate indicator ID '{ind_id}' found in {file_stem}"
                )
            seen_ids.add(ind_id)

            # Add category metadata
            ind["_category"] = file_stem

            # Add to collections
            all_indicators.append(ind)
            category_indicators.append(ind)

            # Group by source
            source = ind.get("source", "unknown")
            if source not in registry["by_source"]:
                registry["by_source"][source] = []
            registry["by_source"][source].append(ind)

            # Group by group
            group = ind.get("group", "unknown")
            if group not in registry["by_group"]:
                registry["by_group"][group] = []
            registry["by_group"][group].append(ind)

        registry["by_category"][file_stem] = category_indicators
        logger.debug(f"Loaded {len(category_indicators)} indicators from {file_stem}")

    registry["indicators"] = all_indicators
    registry["metadata"]["total_indicators"] = len(all_indicators)

    logger.info(
        f"Loaded {len(all_indicators)} indicators from "
        f"{len(registry['metadata']['files_loaded'])} files"
    )

    return registry


def get_indicators_by_source(registry: Dict[str, Any], source: str) -> List[Dict]:
    """
    Get all indicators for a specific source.

    Args:
        registry: The loaded registry
        source: Source name (yahoo, fred, synthetic, technical)

    Returns:
        List of indicator definitions
    """
    return registry.get("by_source", {}).get(source, [])


def get_indicators_by_group(registry: Dict[str, Any], group: str) -> List[Dict]:
    """
    Get all indicators for a specific group.

    Args:
        registry: The loaded registry
        group: Group name (e.g., us_equity_index, fx_major)

    Returns:
        List of indicator definitions
    """
    return registry.get("by_group", {}).get(group, [])


def get_yahoo_indicators(registry: Dict[str, Any]) -> List[Dict]:
    """Get all Yahoo Finance indicators with ticker info."""
    indicators = get_indicators_by_source(registry, "yahoo")
    result = []
    for ind in indicators:
        ticker = ind.get("params", {}).get("ticker")
        if ticker:
            result.append({
                "id": ind["id"],
                "name": ind["name"],
                "ticker": ticker,
                "group": ind.get("group"),
                "asset_class": ind.get("asset_class"),
            })
    return result


def get_fred_indicators(registry: Dict[str, Any]) -> List[Dict]:
    """Get all FRED indicators with series_id info."""
    indicators = get_indicators_by_source(registry, "fred")
    result = []
    for ind in indicators:
        series_id = ind.get("params", {}).get("series_id")
        if series_id:
            result.append({
                "id": ind["id"],
                "name": ind["name"],
                "series_id": series_id,
                "group": ind.get("group"),
                "frequency": ind.get("frequency", "daily"),
            })
    return result


def get_synthetic_indicators(registry: Dict[str, Any]) -> List[Dict]:
    """Get all synthetic indicator definitions."""
    indicators = get_indicators_by_source(registry, "synthetic")
    result = []
    for ind in indicators:
        params = ind.get("params", {})
        result.append({
            "id": ind["id"],
            "name": ind["name"],
            "formula": params.get("formula"),
            "inputs": params.get("inputs", []),
            "group": ind.get("group"),
        })
    return result


def get_technical_indicators(registry: Dict[str, Any]) -> List[Dict]:
    """Get all technical indicator definitions."""
    indicators = get_indicators_by_source(registry, "technical")
    result = []
    for ind in indicators:
        params = ind.get("params", {})
        result.append({
            "id": ind["id"],
            "name": ind["name"],
            "base": params.get("base"),
            "type": params.get("type"),
            "params": {k: v for k, v in params.items() if k not in ["base", "type"]},
            "group": ind.get("group"),
        })
    return result


def get_technical_auto_apply_rules(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Get the technical indicator auto-apply rules."""
    return registry.get("technical_rules", {})


def validate_yaml_registry(registry: Dict[str, Any]) -> bool:
    """
    Validate the loaded YAML registry.

    Checks:
    - All indicators have required fields (id, name, source)
    - All IDs are lowercase
    - Synthetic indicators have valid inputs
    - Technical indicators have valid base references

    Args:
        registry: The loaded registry dictionary

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    errors = []

    # Build set of all indicator IDs for dependency checking
    all_ids = {ind["id"] for ind in registry.get("indicators", [])}

    for ind in registry.get("indicators", []):
        ind_id = ind.get("id", "unknown")

        # Check required fields
        if not ind.get("id"):
            errors.append(f"Indicator missing 'id' field: {ind}")
        if not ind.get("name"):
            errors.append(f"Indicator {ind_id} missing 'name' field")
        if not ind.get("source"):
            errors.append(f"Indicator {ind_id} missing 'source' field")

        # Check ID is lowercase
        if ind_id != ind_id.lower():
            errors.append(f"Indicator ID '{ind_id}' must be lowercase")

        # Validate synthetic dependencies
        if ind.get("source") == "synthetic":
            inputs = ind.get("params", {}).get("inputs", [])
            for input_id in inputs:
                if input_id not in all_ids:
                    errors.append(
                        f"Synthetic indicator '{ind_id}' depends on "
                        f"unknown indicator '{input_id}'"
                    )

        # Validate technical base reference
        if ind.get("source") == "technical":
            base = ind.get("params", {}).get("base")
            if base and base not in all_ids:
                errors.append(
                    f"Technical indicator '{ind_id}' references "
                    f"unknown base '{base}'"
                )

    if errors:
        error_msg = "Registry validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    logger.info(f"Registry validation passed: {len(all_ids)} indicators")
    return True


def convert_to_legacy_format(registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert YAML registry to legacy JSON format for backward compatibility.

    This allows the new YAML-based registry to work with existing code
    that expects the old JSON structure.

    Args:
        registry: The loaded YAML registry

    Returns:
        Dictionary in legacy format with market, economic, synthetic, technical keys
    """
    legacy = {
        "version": registry.get("version", "2.0.0"),
        "market": [],
        "economic": [],
        "synthetic": [],
        "technical": [],
    }

    for ind in registry.get("indicators", []):
        source = ind.get("source")
        ind_id = ind.get("id")
        params = ind.get("params", {})

        # Both yahoo and stooq are market data sources
        if source in ("yahoo", "stooq"):
            legacy["market"].append({
                "name": ind_id,
                "ticker": params.get("ticker", ind_id.upper()),
                "source": source,  # CRITICAL: preserve source for routing
                "type": "market_price",
                "frequency": ind.get("frequency", "daily"),
                "group": ind.get("group"),
                "asset_class": ind.get("asset_class"),
                "params": params,  # Keep full params for ticker access
            })

        elif source == "fred":
            legacy["economic"].append({
                "name": ind_id,
                "series_id": params.get("series_id"),
                "type": "econ_series",
                "frequency": ind.get("frequency", "daily"),
                "group": ind.get("group"),
            })

        elif source == "synthetic":
            legacy["synthetic"].append({
                "name": ind_id,
                "formula": params.get("formula"),
                "inputs": params.get("inputs", []),
                "depends_on": params.get("inputs", []),
                "group": ind.get("group"),
            })

        elif source == "technical":
            legacy["technical"].append({
                "name": ind_id,
                "base": params.get("base"),
                "type": params.get("type"),
                "params": {k: v for k, v in params.items() if k not in ["base", "type"]},
                "depends_on": [params.get("base")] if params.get("base") else [],
                "group": ind.get("group"),
            })

    # Store technical rules
    legacy["technical_rules"] = registry.get("technical_rules", {})

    return legacy


def print_registry_summary(registry: Dict[str, Any]) -> None:
    """Print a summary of the loaded registry."""
    print("\n" + "=" * 60)
    print("PRISM REGISTRY SUMMARY")
    print("=" * 60)
    print(f"Version: {registry.get('version', 'unknown')}")
    print(f"Total indicators: {registry['metadata']['total_indicators']}")
    print()

    # By source
    print("By Source:")
    for source, indicators in sorted(registry.get("by_source", {}).items()):
        print(f"  {source:15s}: {len(indicators):4d}")
    print()

    # By category
    print("By Category:")
    for category, indicators in sorted(registry.get("by_category", {}).items()):
        print(f"  {category:20s}: {len(indicators):4d}")
    print()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        registry = load_all_yaml_registries()
        validate_yaml_registry(registry)
        print("Validation: PASSED")
        print_registry_summary(registry)

        # Test legacy conversion
        legacy = convert_to_legacy_format(registry)
        print(f"Legacy format: {len(legacy['market'])} market, "
              f"{len(legacy['economic'])} economic, "
              f"{len(legacy['synthetic'])} synthetic, "
              f"{len(legacy['technical'])} technical")

        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
