"""
Fetch Validator - Registry validation utilities
================================================

Provides validation functions for registry entries to ensure
data integrity before any fetch operations occur.

Usage:
    from utils.fetch_validator import (
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
        List of validation error messages. Empty if valid.
    """
    errors = []

    if registry_path is None:
        registry_path = _get_project_root() / "data_fetch" / "system_registry.json"

    data, error = _load_json(registry_path)
    if error:
        return [error]

    # Check required top-level keys
    for key in SYSTEM_REGISTRY_SCHEMA["required_keys"]:
        if key not in data:
            errors.append(f"System registry missing required key: '{key}'")

    # Check paths section
    if "paths" in data:
        for path_key in SYSTEM_REGISTRY_SCHEMA["paths_required"]:
            if path_key not in data["paths"]:
                errors.append(f"System registry paths missing: '{path_key}'")

    # Check registries section
    if "registries" in data:
        for reg_key in SYSTEM_REGISTRY_SCHEMA["registries_required"]:
            if reg_key not in data["registries"]:
                errors.append(f"System registry registries missing: '{reg_key}'")
            else:
                # Verify the referenced registry file exists
                ref_path = _get_project_root() / data["registries"][reg_key]
                if not ref_path.exists():
                    errors.append(
                        f"Referenced registry not found: {data['registries'][reg_key]}"
                    )

    return errors


# =============================================================================
# MARKET REGISTRY VALIDATION
# =============================================================================

def validate_market_entry(ticker: str, entry: Dict[str, Any]) -> List[str]:
    """
    Validate a single market registry entry.

    Args:
        ticker: The ticker symbol
        entry: The entry configuration dict

    Returns:
        List of validation error messages
    """
    errors = []
    prefix = f"Market '{ticker}'"

    # Check required keys
    for key in MARKET_ENTRY_SCHEMA["required_keys"]:
        if key not in entry:
            errors.append(f"{prefix}: missing required key '{key}'")

    # Validate fetch_type
    if "fetch_type" in entry:
        if entry["fetch_type"] not in MARKET_ENTRY_SCHEMA["valid_fetch_types"]:
            errors.append(
                f"{prefix}: invalid fetch_type '{entry['fetch_type']}'. "
                f"Valid: {MARKET_ENTRY_SCHEMA['valid_fetch_types']}"
            )

    # Validate frequency
    if "frequency" in entry:
        if entry["frequency"] not in MARKET_ENTRY_SCHEMA["valid_frequencies"]:
            errors.append(
                f"{prefix}: invalid frequency '{entry['frequency']}'. "
                f"Valid: {MARKET_ENTRY_SCHEMA['valid_frequencies']}"
            )

    # Validate enabled is boolean
    if "enabled" in entry and not isinstance(entry["enabled"], bool):
        errors.append(f"{prefix}: 'enabled' must be boolean")

    # Validate tables structure
    if "tables" in entry:
        if not isinstance(entry["tables"], dict):
            errors.append(f"{prefix}: 'tables' must be a dictionary")
        else:
            for table_type, table_name in entry["tables"].items():
                if table_type not in MARKET_ENTRY_SCHEMA["valid_table_types"]:
                    errors.append(
                        f"{prefix}: invalid table type '{table_type}'. "
                        f"Valid: {MARKET_ENTRY_SCHEMA['valid_table_types']}"
                    )
                if not _is_valid_sql_identifier(table_name):
                    errors.append(
                        f"{prefix}: invalid table name '{table_name}'. "
                        "Must be valid SQL identifier."
                    )

    # Validate use_column
    if "use_column" in entry:
        if not _is_valid_sql_identifier(entry["use_column"]):
            errors.append(
                f"{prefix}: invalid use_column '{entry['use_column']}'. "
                "Must be valid SQL identifier."
            )

    return errors


def validate_market_registry(registry_path: Optional[Path] = None) -> List[str]:
    """
    Validate the complete market registry.

    Args:
        registry_path: Optional custom path. Uses default if None.

    Returns:
        List of validation error messages. Empty if valid.
    """
    errors = []

    if registry_path is None:
        registry_path = _get_project_root() / "data_fetch" / "market_registry.json"

    data, error = _load_json(registry_path)
    if error:
        return [error]

    if not isinstance(data, dict):
        return ["Market registry must be a JSON object"]

    for ticker, entry in data.items():
        entry_errors = validate_market_entry(ticker, entry)
        errors.extend(entry_errors)

    return errors


# =============================================================================
# ECONOMIC REGISTRY VALIDATION
# =============================================================================

def validate_economic_entry(series_id: str, entry: Dict[str, Any]) -> List[str]:
    """
    Validate a single economic registry entry.

    Args:
        series_id: The economic series identifier
        entry: The entry configuration dict

    Returns:
        List of validation error messages
    """
    errors = []
    prefix = f"Economic '{series_id}'"

    # Check required keys
    for key in ECONOMIC_ENTRY_SCHEMA["required_keys"]:
        if key not in entry:
            errors.append(f"{prefix}: missing required key '{key}'")

    # Validate fetch_type
    if "fetch_type" in entry:
        if entry["fetch_type"] not in ECONOMIC_ENTRY_SCHEMA["valid_fetch_types"]:
            errors.append(
                f"{prefix}: invalid fetch_type '{entry['fetch_type']}'. "
                f"Valid: {ECONOMIC_ENTRY_SCHEMA['valid_fetch_types']}"
            )

    # Validate frequency
    if "frequency" in entry:
        if entry["frequency"] not in ECONOMIC_ENTRY_SCHEMA["valid_frequencies"]:
            errors.append(
                f"{prefix}: invalid frequency '{entry['frequency']}'. "
                f"Valid: {ECONOMIC_ENTRY_SCHEMA['valid_frequencies']}"
            )

    # Validate enabled is boolean
    if "enabled" in entry and not isinstance(entry["enabled"], bool):
        errors.append(f"{prefix}: 'enabled' must be boolean")

    # Validate table name
    if "table" in entry:
        if not _is_valid_sql_identifier(entry["table"]):
            errors.append(
                f"{prefix}: invalid table name '{entry['table']}'. "
                "Must be valid SQL identifier."
            )

    # Validate use_column
    if "use_column" in entry:
        if not _is_valid_sql_identifier(entry["use_column"]):
            errors.append(
                f"{prefix}: invalid use_column '{entry['use_column']}'. "
                "Must be valid SQL identifier."
            )

    # Validate code matches series_id (for FRED)
    if "code" in entry and entry.get("fetch_type") == "fred":
        if entry["code"] != series_id:
            errors.append(
                f"{prefix}: FRED code '{entry['code']}' should match series_id"
            )

    # Validate transformations is a list
    if "transformations" in entry:
        if not isinstance(entry["transformations"], list):
            errors.append(f"{prefix}: 'transformations' must be a list")

    # Validate revision_tracking is boolean
    if "revision_tracking" in entry:
        if not isinstance(entry["revision_tracking"], bool):
            errors.append(f"{prefix}: 'revision_tracking' must be boolean")

    return errors


def validate_economic_registry(registry_path: Optional[Path] = None) -> List[str]:
    """
    Validate the complete economic registry.

    Args:
        registry_path: Optional custom path. Uses default if None.

    Returns:
        List of validation error messages. Empty if valid.
    """
    errors = []

    if registry_path is None:
        registry_path = _get_project_root() / "data_fetch" / "economic_registry.json"

    data, error = _load_json(registry_path)
    if error:
        return [error]

    if not isinstance(data, dict):
        return ["Economic registry must be a JSON object"]

    for series_id, entry in data.items():
        entry_errors = validate_economic_entry(series_id, entry)
        errors.extend(entry_errors)

    return errors


# =============================================================================
# COMBINED VALIDATION
# =============================================================================

def validate_all_registries() -> Dict[str, List[str]]:
    """
    Validate all registries and return results.

    Returns:
        Dict mapping registry name to list of errors.
        Empty lists indicate valid registries.
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
        Tuple of (is_valid, errors_dict)
    """
    errors = validate_all_registries()
    is_valid = all(len(errs) == 0 for errs in errors.values())
    return is_valid, errors


# =============================================================================
# REGISTRY LOADING WITH VALIDATION
# =============================================================================

def load_validated_registry(registry_type: str) -> dict:
    """
    Load a registry after validation.

    Args:
        registry_type: One of 'system', 'market', 'economic'

    Returns:
        The validated registry data

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If registry file doesn't exist
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
        List of enabled ticker/series identifiers
    """
    if registry_type not in ("market", "economic"):
        raise ValueError("registry_type must be 'market' or 'economic'")

    registry = load_validated_registry(registry_type)

    return [
        key for key, entry in registry.items()
        if entry.get("enabled", False)
    ]
