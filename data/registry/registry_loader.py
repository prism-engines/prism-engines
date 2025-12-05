"""
Registry Loader for PRISM Engine.

Loads and validates the metric registry JSON, providing utilities for
querying metric names and dependencies.

CLI Usage:
    python -m data.registry.registry_loader

This will load the registry, validate it, and print summary counts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Registry sections that contain metrics
METRIC_SECTIONS = ["market", "economic", "synthetic", "technical", "geometry", "model_outputs"]

# Required top-level keys in the registry
REQUIRED_KEYS = ["version", "market", "economic"]


def _get_default_registry_path() -> Path:
    """Get default path to metric_registry.json relative to repo root."""
    return Path(__file__).parent / "metric_registry.json"


def load_metric_registry(path: Optional[str | Path] = None) -> dict:
    """
    Load the metric registry from JSON file.

    Args:
        path: Optional path to the registry file. If None, uses the default
              path relative to this module (data/registry/metric_registry.json).

    Returns:
        Dictionary containing the parsed registry data.

    Raises:
        FileNotFoundError: If the registry file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if path is None:
        path = _get_default_registry_path()
    else:
        path = Path(path)

    logger.info(f"Loading metric registry from {path}")

    with open(path) as f:
        registry = json.load(f)

    return registry


def validate_registry(reg: dict) -> None:
    """
    Validate the metric registry for correctness.

    Checks:
    - Required keys are present
    - All metric names are lowercase
    - No duplicate names across sections
    - All depends_on references point to existing metrics in market/economic

    Args:
        reg: The registry dictionary to validate.

    Raises:
        ValueError: If validation fails with description of the issue.
    """
    # Check required keys
    for key in REQUIRED_KEYS:
        if key not in reg:
            raise ValueError(f"Registry missing required key: '{key}'")

    # Collect all metric names and check for duplicates
    all_names: dict[str, str] = {}  # name -> section it came from
    base_metrics: set[str] = set()  # metrics from market + economic (can be depended on)

    for section in METRIC_SECTIONS:
        if section not in reg:
            continue

        for item in reg[section]:
            name = item.get("name")
            if name is None:
                raise ValueError(f"Metric in section '{section}' missing 'name' field")

            # Check lowercase
            if name != name.lower():
                raise ValueError(
                    f"Metric name '{name}' in section '{section}' must be lowercase"
                )

            # Check for duplicates
            if name in all_names:
                raise ValueError(
                    f"Duplicate metric name '{name}' found in sections "
                    f"'{all_names[name]}' and '{section}'"
                )

            all_names[name] = section

            # Track base metrics that can be dependencies
            if section in ("market", "economic"):
                base_metrics.add(name)

    # Validate depends_on references
    for section in ["synthetic", "technical"]:
        if section not in reg:
            continue

        for item in reg[section]:
            depends_on = item.get("depends_on", [])
            name = item.get("name", "unknown")

            for dep in depends_on:
                if dep not in base_metrics:
                    raise ValueError(
                        f"Metric '{name}' in section '{section}' depends on '{dep}' "
                        f"which is not found in market or economic sections"
                    )

    logger.info(f"Registry validation passed: {len(all_names)} metrics")


def get_all_metric_names(reg: dict) -> set[str]:
    """
    Get all metric names from the registry.

    Args:
        reg: The registry dictionary.

    Returns:
        Set of all metric names across all sections.
    """
    names: set[str] = set()

    for section in METRIC_SECTIONS:
        if section not in reg:
            continue

        for item in reg[section]:
            name = item.get("name")
            if name:
                names.add(name)

    return names


def get_dependencies(reg: dict, name: str) -> list[str] | None:
    """
    Get the dependencies for a specific metric.

    Args:
        reg: The registry dictionary.
        name: The metric name to look up.

    Returns:
        List of dependency names if the metric has depends_on field,
        None if the metric has no dependencies or is not found.
    """
    for section in METRIC_SECTIONS:
        if section not in reg:
            continue

        for item in reg[section]:
            if item.get("name") == name:
                deps = item.get("depends_on")
                return deps if deps else None

    return None


def get_metrics_by_section(reg: dict, section: str) -> list[dict]:
    """
    Get all metrics from a specific section.

    Args:
        reg: The registry dictionary.
        section: The section name (e.g., 'market', 'economic', 'synthetic').

    Returns:
        List of metric dictionaries from that section.
    """
    return reg.get(section, [])


def get_section_names(reg: dict, section: str) -> list[str]:
    """
    Get metric names from a specific section.

    Args:
        reg: The registry dictionary.
        section: The section name.

    Returns:
        List of metric names from that section.
    """
    return [item["name"] for item in reg.get(section, []) if "name" in item]


def print_summary(reg: dict) -> None:
    """Print a summary of the registry contents."""
    print("=" * 50)
    print("PRISM Metric Registry Summary")
    print("=" * 50)
    print(f"Version: {reg.get('version', 'unknown')}")
    print(f"Generated: {reg.get('generated', 'unknown')}")
    print()

    total = 0
    for section in METRIC_SECTIONS:
        if section not in reg:
            continue
        count = len(reg[section])
        total += count
        print(f"  {section:15s}: {count:3d} metrics")

    print("-" * 50)
    print(f"  {'TOTAL':15s}: {total:3d} metrics")
    print()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        # Load registry
        registry = load_metric_registry()

        # Validate
        validate_registry(registry)
        print("Validation: PASSED")
        print()

        # Print summary
        print_summary(registry)

        # Show sample metrics from each section
        print("Sample metrics by section:")
        print("-" * 50)
        for section in METRIC_SECTIONS:
            if section in registry and registry[section]:
                names = get_section_names(registry, section)[:3]
                print(f"  {section}: {', '.join(names)}...")

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"ERROR: Registry file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"VALIDATION ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
