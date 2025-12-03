"""
Registry Validator - Validation utilities for registry entries

Provides validation functions for registry entries to ensure
data integrity before any fetch operations occur.

Usage:
    from utils.registry_validator import (
        validate_system_registry,
        validate_market_registry,
        validate_economic_registry,
        validate_all_registries
    )

    # Validate all registries
    errors = validate_all_registries()
    if errors:
        for error in errors:
            print(f"Validation error: {error}")
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Required keys for system registry
SYSTEM_REGISTRY_SCHEMA = {
    "required_keys": ["paths", "registries"],
    "paths_required": ["data_raw", "data_clean", "db_path"],
    "registries_required": ["market", "economic"]
}

# Required keys for market entries
MARKET_ENTRY_SCHEMA = {
    "required_keys": ["fetch_type", "enabled", "frequency", "tables", "use_column"],
    "valid_fetch_types": ["yahoo", "custom"],
    "valid_frequencies": ["daily", "weekly", "monthly"],
    "valid_table_types": ["prices", "dividends", "tri", "splits"]
}

# Required keys for economic entries
ECONOMIC_ENTRY_SCHEMA = {
    "required_keys": ["fetch_type", "enabled", "frequency", "table", "code", "use_column"],
    "valid_fetch_types": ["fred", "custom"],
    "valid_frequencies": ["daily", "weekly", "monthly", "quarterly", "annual"]
}

# Valid SQL identifier pattern (table/column names)
SQL_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def _load_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    """
    Load a JSON file safely.

    Returns:
        Tuple of (data, error_message)
    """
    if not path.exists():
        return None, f"File not found: {path}"

    try:
        with open(path, "r") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in {path}: {e}"


def _is_valid_sql_identifier(name: str) -> bool:
    """Check if a name is a valid SQL identifier."""
    return bool(SQL_IDENTIFIER_PATTERN.match(name))


# =============================================================================
# SYSTEM REGISTRY VALIDATION
# =============================================================================

def validate_system_registry(registry_path: Optional[Path] = None) -> List[str]:
    """
    Validate the system registry structure.

    Args:
        registry_path: Optional custom path. Uses default if None.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    path = registry_path or (_get_project_root() / "data_fetch" / "system_registry.json")
    data, load_error = _load_json(path)

    if load_error:
        return [load_error]

    # Check required top-level keys
    for key in SYSTEM_REGISTRY_SCHEMA["required_keys"]:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    # Check required paths
    if "paths" in data:
        for path_key in SYSTEM_REGISTRY_SCHEMA["paths_required"]:
            if path_key not in data["paths"]:
                errors.append(f"Missing required path: paths.{path_key}")

    # Check required registries
    if "registries" in data:
        for reg_key in SYSTEM_REGISTRY_SCHEMA["registries_required"]:
            if reg_key not in data["registries"]:
                errors.append(f"Missing required registry: registries.{reg_key}")

    return errors


# =============================================================================
# MARKET REGISTRY VALIDATION
# =============================================================================

def validate_market_entry(ticker: str, entry: Dict[str, Any]) -> List[str]:
    """Validate a single market registry entry."""
    errors = []

    # Check required keys
    for key in MARKET_ENTRY_SCHEMA["required_keys"]:
        if key not in entry:
            errors.append(f"{ticker}: Missing required key '{key}'")

    # Validate fetch_type
    if "fetch_type" in entry:
        if entry["fetch_type"] not in MARKET_ENTRY_SCHEMA["valid_fetch_types"]:
            errors.append(
                f"{ticker}: Invalid fetch_type '{entry['fetch_type']}'. "
                f"Valid: {MARKET_ENTRY_SCHEMA['valid_fetch_types']}"
            )

    # Validate frequency
    if "frequency" in entry:
        if entry["frequency"] not in MARKET_ENTRY_SCHEMA["valid_frequencies"]:
            errors.append(
                f"{ticker}: Invalid frequency '{entry['frequency']}'. "
                f"Valid: {MARKET_ENTRY_SCHEMA['valid_frequencies']}"
            )

    # Validate tables
    if "tables" in entry:
        for table_type in entry["tables"]:
            if table_type not in MARKET_ENTRY_SCHEMA["valid_table_types"]:
                errors.append(
                    f"{ticker}: Invalid table type '{table_type}'. "
                    f"Valid: {MARKET_ENTRY_SCHEMA['valid_table_types']}"
                )

    # Validate use_column is valid SQL identifier
    if "use_column" in entry:
        if not _is_valid_sql_identifier(entry["use_column"]):
            errors.append(
                f"{ticker}: Invalid use_column '{entry['use_column']}'. "
                "Must be valid SQL identifier."
            )

    return errors


def validate_market_registry(registry_path: Optional[Path] = None) -> List[str]:
    """Validate the entire market registry."""
    errors = []

    path = registry_path or (_get_project_root() / "data_fetch" / "market_registry.json")
    data, load_error = _load_json(path)

    if load_error:
        return [load_error]

    for ticker, entry in data.items():
        errors.extend(validate_market_entry(ticker, entry))

    return errors


# =============================================================================
# ECONOMIC REGISTRY VALIDATION
# =============================================================================

def validate_economic_entry(series_id: str, entry: Dict[str, Any]) -> List[str]:
    """Validate a single economic registry entry."""
    errors = []

    # Check required keys
    for key in ECONOMIC_ENTRY_SCHEMA["required_keys"]:
        if key not in entry:
            errors.append(f"{series_id}: Missing required key '{key}'")

    # Validate fetch_type
    if "fetch_type" in entry:
        if entry["fetch_type"] not in ECONOMIC_ENTRY_SCHEMA["valid_fetch_types"]:
            errors.append(
                f"{series_id}: Invalid fetch_type '{entry['fetch_type']}'. "
                f"Valid: {ECONOMIC_ENTRY_SCHEMA['valid_fetch_types']}"
            )

    # Validate frequency
    if "frequency" in entry:
        if entry["frequency"] not in ECONOMIC_ENTRY_SCHEMA["valid_frequencies"]:
            errors.append(
                f"{series_id}: Invalid frequency '{entry['frequency']}'. "
                f"Valid: {ECONOMIC_ENTRY_SCHEMA['valid_frequencies']}"
            )

    # Validate table is valid SQL identifier
    if "table" in entry:
        if not _is_valid_sql_identifier(entry["table"]):
            errors.append(
                f"{series_id}: Invalid table '{entry['table']}'. "
                "Must be valid SQL identifier."
            )

    # Validate use_column is valid SQL identifier
    if "use_column" in entry:
        if not _is_valid_sql_identifier(entry["use_column"]):
            errors.append(
                f"{series_id}: Invalid use_column '{entry['use_column']}'. "
                "Must be valid SQL identifier."
            )

    return errors


def validate_economic_registry(registry_path: Optional[Path] = None) -> List[str]:
    """Validate the entire economic registry."""
    errors = []

    path = registry_path or (_get_project_root() / "data_fetch" / "economic_registry.json")
    data, load_error = _load_json(path)

    if load_error:
        return [load_error]

    for series_id, entry in data.items():
        errors.extend(validate_economic_entry(series_id, entry))

    return errors


# =============================================================================
# COMBINED VALIDATION
# =============================================================================

def validate_all_registries() -> Dict[str, List[str]]:
    """
    Validate all registries.

    Returns:
        Dict mapping registry name to list of errors
    """
    return {
        "system": validate_system_registry(),
        "market": validate_market_registry(),
        "economic": validate_economic_registry()
    }


def registries_are_valid() -> Tuple[bool, Dict[str, List[str]]]:
    """
    Check if all registries are valid.

    Returns:
        Tuple of (all_valid, error_dict)
    """
    errors = validate_all_registries()
    all_valid = all(len(e) == 0 for e in errors.values())
    return all_valid, errors


def load_validated_registry(registry_type: str) -> dict:
    """
    Load a registry after validating it.

    Args:
        registry_type: One of 'system', 'market', 'economic'

    Returns:
        The validated registry data

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If registry file does not exist
    """
    registry_paths = {
        "system": _get_project_root() / "data_fetch" / "system_registry.json",
        "market": _get_project_root() / "data_fetch" / "market_registry.json",
        "economic": _get_project_root() / "data_fetch" / "economic_registry.json"
    }

    if registry_type not in registry_paths:
        raise ValueError(
            f"Invalid registry type: {registry_type}. "
            f"Valid: {list(registry_paths.keys())}"
        )

    path = registry_paths[registry_type]

    # Validate
    validators = {
        "system": validate_system_registry,
        "market": validate_market_registry,
        "economic": validate_economic_registry
    }

    errors = validators[registry_type](path)
    if errors:
        raise ValueError(
            f"Registry validation failed for {registry_type}:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    # Load and return
    with open(path, "r") as f:
        return json.load(f)


def get_enabled_tickers(registry_type: str) -> List[str]:
    """
    Get list of enabled tickers/series from a registry.

    Args:
        registry_type: 'market' or 'economic'

    Returns:
        List of enabled ticker/series IDs
    """
    if registry_type not in ["market", "economic"]:
        raise ValueError(f"Invalid registry_type: {registry_type}")

    registry = load_validated_registry(registry_type)

    return [
        key for key, entry in registry.items()
        if entry.get("enabled", False)
    ]
