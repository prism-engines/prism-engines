"""
Data Phase DB Integrity Guards

Semantic Phase: DATA
- Admissible indicator data: ingestion, cleaning, normalization, suitability, cohorts
- Anything before engines create new measurements

IMMUTABILITY CONTRACT:
    - run_id is locked at start of Data Phase execution
    - Once Data Phase completes, no writes are allowed to that run_id
    - Any attempt to write to a locked run is a SYSTEM ERROR

Lock Table:
    - meta.data_run_lock
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import logging
import duckdb

logger = logging.getLogger(__name__)


@dataclass
class RunLockResult:
    created: bool
    message: str


def assert_run_id_unused(conn: duckdb.DuckDBPyConnection, run_id: str, *, force: bool = False) -> RunLockResult:
    """
    Assert that a run_id has not been used before (check only, no insert).

    Called at the START of Data Phase execution to verify the run_id is available.
    Does NOT insert the lock - that happens on successful completion.

    Args:
        conn: Database connection
        run_id: The run ID to check
        force: If True, bypass check (NOT RECOMMENDED - violates canon)

    Returns:
        RunLockResult with status

    Raises:
        RuntimeError: If run_id is already locked and force=False
    """
    existing = conn.execute(
        "SELECT run_id FROM meta.data_run_lock WHERE run_id = ?",
        [run_id]
    ).fetchone()

    if existing and not force:
        raise RuntimeError(
            f"DATA PHASE INTEGRITY ERROR: run_id already locked: {run_id}. "
            f"Do not overwrite evidence. Generate a new run_id."
        )

    if existing and force:
        return RunLockResult(created=False, message=f"run_id already locked (force=True): {run_id}")

    # Do NOT insert lock here - lock is inserted on successful completion
    return RunLockResult(created=False, message=f"run_id available: {run_id}")


def finalize_run_lock(conn: duckdb.DuckDBPyConnection, run_id: str, domain: str, mode: str) -> RunLockResult:
    """
    Finalize the run by inserting the lock record.

    Called ONLY on successful Data Phase completion.
    After this, the run_id is IMMUTABLE - no further writes allowed.

    Args:
        conn: Database connection
        run_id: The run ID to lock
        domain: The domain (e.g., 'economic')
        mode: The mode (e.g., 'short', 'full')

    Returns:
        RunLockResult with status
    """
    existing = conn.execute(
        "SELECT run_id FROM meta.data_run_lock WHERE run_id = ?",
        [run_id]
    ).fetchone()

    if existing:
        return RunLockResult(created=False, message=f"run_id already locked: {run_id}")

    conn.execute(
        "INSERT INTO meta.data_run_lock(run_id, started_at, domain, mode, locked_by) VALUES (?, now(), ?, ?, ?)",
        [run_id, domain, mode, "data_phase_orchestrator"]
    )
    return RunLockResult(created=True, message=f"run_id LOCKED (immutable): {run_id}")


def is_run_locked(conn: duckdb.DuckDBPyConnection, run_id: str) -> bool:
    """
    Check if a run_id is locked.

    Args:
        conn: Database connection
        run_id: The run ID to check

    Returns:
        True if locked, False otherwise
    """
    result = conn.execute(
        "SELECT run_id FROM meta.data_run_lock WHERE run_id = ?",
        [run_id]
    ).fetchone()
    return result is not None


def guard_data_write(conn: duckdb.DuckDBPyConnection, run_id: str, table_name: str) -> None:
    """
    Guard a write operation to a Data Phase table.

    Must be called BEFORE any write to Data Phase protected tables.
    Raises an error if the run is already locked.

    Args:
        conn: Database connection
        run_id: The run ID being written
        table_name: The table being written to (for error message)

    Raises:
        RuntimeError: If run_id is locked
    """
    if is_run_locked(conn, run_id):
        raise RuntimeError(
            f"DATA PHASE INTEGRITY VIOLATION: Cannot write to {table_name} for locked run_id: {run_id}. "
            f"Data Phase artifacts are IMMUTABLE after completion."
        )


# =============================================================================
# ENGINE PREFLIGHT GATE (HARD BLOCK)
# =============================================================================

class DataPhaseNotLockedError(RuntimeError):
    """Raised when an engine attempts to run without a locked Data Phase run."""
    pass


# Alias for backwards compatibility
Phase1NotLockedError = DataPhaseNotLockedError


def require_data_phase_locked(conn: duckdb.DuckDBPyConnection, run_id: str) -> None:
    """
    ENGINE PREFLIGHT GATE: Block execution unless Data Phase is locked.

    This function MUST be called by all engines before execution.
    Engines may NOT run on unpersisted Data Phase output.

    Behavior:
        - If Data Phase is locked: returns normally
        - If Data Phase is NOT locked: raises DataPhaseNotLockedError

    No fallback behavior.
    No warnings-only behavior.
    Engines may not run without a locked Data Phase run. Period.

    Args:
        conn: Database connection
        run_id: The Data Phase run ID to verify

    Raises:
        DataPhaseNotLockedError: If run_id is not locked in meta.data_run_lock
    """
    result = conn.execute(
        "SELECT 1 FROM meta.data_run_lock WHERE run_id = ?",
        [run_id]
    ).fetchone()

    if result is None:
        raise DataPhaseNotLockedError(
            f"ENGINE PREFLIGHT FAILED: Data Phase run '{run_id}' is NOT locked.\n"
            f"Engines may NOT run on unpersisted Data Phase output.\n"
            f"Run Data Phase to completion before executing engines.\n"
            f"This is not optional. This is architectural."
        )


# Alias for backwards compatibility
require_phase1_locked = require_data_phase_locked


def get_data_phase_contract(conn: duckdb.DuckDBPyConnection, run_id: str) -> dict:
    """
    Load the Data Phase engine contract for a run.

    Args:
        conn: Database connection
        run_id: The Data Phase run ID

    Returns:
        Dict with:
            - run_id: The run ID
            - locked: Whether the run is locked
            - indicators: List of indicator contracts (empty for now)

    Raises:
        DataPhaseNotLockedError: If run_id is not locked
    """
    # First verify Data Phase is locked
    require_data_phase_locked(conn, run_id)

    # For now, return minimal contract
    # Full contract loading will be added when data phase tables are populated
    return {
        'run_id': run_id,
        'locked': True,
        'indicators': [],
    }


# Alias for backwards compatibility
get_phase1_contract = get_data_phase_contract
