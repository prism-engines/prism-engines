"""
PRISM Database Path Resolution
==============================

Single source of truth for database path across all PRISM modules.

Resolution order:
1. Explicit override via set_db_path() or --db CLI argument
2. PRISM_DB environment variable
3. ~/prism_data/prism.db (if exists and non-empty)
4. data/sql/prism.db (if exists and non-empty)
5. ~/prism_data/prism.db (default, even if doesn't exist yet)

Usage:
    from data.sql.db_path import get_db_path, set_db_path

    # Normal usage - auto-detects correct database
    db_path = get_db_path()

    # Override for CLI --db argument
    set_db_path("/path/to/custom.db")
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Global override (set via CLI --db argument or set_db_path())
_db_path_override: Optional[str] = None

# Cache the resolved path to avoid repeated filesystem checks
_cached_db_path: Optional[str] = None

# Track if we've already printed warnings this session
_warned_about_path: bool = False


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def _is_valid_db(path: str) -> bool:
    """
    Check if a database file exists and is non-empty.

    Returns True if file exists and has size > 0 bytes.
    """
    try:
        p = Path(path)
        return p.exists() and p.stat().st_size > 0
    except (OSError, IOError):
        return False


def _get_db_size_mb(path: str) -> float:
    """Get database file size in MB."""
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except (OSError, IOError):
        return 0.0


def set_db_path(path: str) -> None:
    """
    Override the database path for this session.

    Use this for CLI --db arguments or testing.

    Args:
        path: Absolute or relative path to database file
    """
    global _db_path_override, _cached_db_path
    _db_path_override = os.path.expanduser(path)
    _cached_db_path = None  # Clear cache to force re-resolution


def clear_db_path_override() -> None:
    """Clear any database path override."""
    global _db_path_override, _cached_db_path
    _db_path_override = None
    _cached_db_path = None


def get_db_path(verbose: bool = False) -> str:
    """
    Get the path to the PRISM database.

    Resolution order:
    1. Explicit override via set_db_path()
    2. PRISM_DB environment variable
    3. ~/prism_data/prism.db (if exists and non-empty)
    4. data/sql/prism.db (if exists and non-empty)
    5. ~/prism_data/prism.db (default)

    Args:
        verbose: If True, always print which path is being used

    Returns:
        Absolute path to the database file
    """
    global _cached_db_path, _warned_about_path

    # Return cached path if available and no override
    if _cached_db_path is not None and _db_path_override is None:
        return _cached_db_path

    # 1. Check for explicit override
    if _db_path_override is not None:
        if verbose:
            print(f"[DB] Using override path: {_db_path_override}")
        return _db_path_override

    # 2. Check environment variable
    env_path = os.getenv("PRISM_DB")
    if env_path:
        resolved = os.path.expanduser(env_path)
        if verbose:
            print(f"[DB] Using PRISM_DB env var: {resolved}")
        _cached_db_path = resolved
        return resolved

    # Define candidate paths
    home_db = os.path.expanduser("~/prism_data/prism.db")
    project_root = _get_project_root()
    local_db = str(project_root / "data" / "sql" / "prism.db")

    # 3. Check ~/prism_data/prism.db (preferred location)
    if _is_valid_db(home_db):
        if verbose:
            size_mb = _get_db_size_mb(home_db)
            print(f"[DB] Using ~/prism_data/prism.db ({size_mb:.1f} MB)")
        _cached_db_path = home_db
        return home_db

    # 4. Check data/sql/prism.db
    if _is_valid_db(local_db):
        # Warn about using local DB if home DB exists but is empty
        if not _warned_about_path and os.path.exists(home_db):
            size_local = _get_db_size_mb(local_db)
            print(f"[WARNING] ~/prism_data/prism.db exists but is empty (0 bytes).")
            print(f"[WARNING] Using data/sql/prism.db instead ({size_local:.1f} MB).")
            _warned_about_path = True
        elif verbose:
            size_mb = _get_db_size_mb(local_db)
            print(f"[DB] Using data/sql/prism.db ({size_mb:.1f} MB)")
        _cached_db_path = local_db
        return local_db

    # 5. Default to home location (even if doesn't exist yet)
    # This is where new databases should be created
    if not _warned_about_path:
        if os.path.exists(local_db) and not _is_valid_db(local_db):
            print(f"[WARNING] data/sql/prism.db is empty (0 bytes).")
        if not os.path.exists(home_db):
            print(f"[INFO] Database not found. Will create at: {home_db}")
        _warned_about_path = True

    _cached_db_path = home_db
    return home_db


def parse_db_arg() -> None:
    """
    Parse --db argument from sys.argv if present.

    Call this early in scripts that support --db override:

        from data.sql.db_path import parse_db_arg
        parse_db_arg()  # Must be called before get_db_path()
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--db" and i + 1 < len(sys.argv):
            set_db_path(sys.argv[i + 1])
            return
        elif arg.startswith("--db="):
            set_db_path(arg[5:])
            return


def get_db_info() -> dict:
    """
    Get information about the current database configuration.

    Returns:
        Dictionary with path, size, exists, and resolution method
    """
    path = get_db_path()
    exists = os.path.exists(path)
    size_mb = _get_db_size_mb(path) if exists else 0.0

    # Determine how path was resolved
    if _db_path_override:
        method = "override"
    elif os.getenv("PRISM_DB"):
        method = "environment"
    else:
        method = "auto-detected"

    return {
        "path": path,
        "exists": exists,
        "size_mb": round(size_mb, 2),
        "resolution_method": method,
        "is_valid": _is_valid_db(path)
    }


# Auto-parse --db argument on import (if present)
# This allows any script to support --db without explicit code
if "--db" in sys.argv or any(arg.startswith("--db=") for arg in sys.argv):
    parse_db_arg()
