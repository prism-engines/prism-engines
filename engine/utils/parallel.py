"""
engine/utils/parallel.py

Safe, auto-tuning parallelization utility for PRISM engine.

This module provides:
- CPU detection with fallbacks
- RAM heuristics to disable parallel on low-memory machines
- Dynamic worker scaling based on system resources
- Automatic fallback to serial execution for small workloads
- Unified run_parallel() API

Usage:
    from engine.utils.parallel import run_parallel

    results = run_parallel(process_fn, items)  # Auto-tunes workers
    results = run_parallel(process_fn, items, max_workers=4)  # Manual override
"""

from __future__ import annotations

import os
import multiprocessing as mp
from typing import Callable, Sequence, TypeVar, List, Optional

T = TypeVar("T")
R = TypeVar("R")

# Minimum RAM (in GB) required for parallel execution
MIN_RAM_GB_FOR_PARALLEL = 4.0

# Minimum items required to justify parallelization (per worker)
MIN_ITEMS_PER_WORKER = 2


def get_cpu_count() -> int:
    """
    Get the number of available CPUs.

    Uses psutil if available for more accurate detection,
    falls back to os.cpu_count().

    Returns:
        Number of CPUs, minimum 1.
    """
    try:
        import psutil
        return psutil.cpu_count(logical=True) or 1
    except ImportError:
        return os.cpu_count() or 1


def get_system_memory_gb() -> float:
    """
    Get total system memory in gigabytes.

    Uses psutil if available, returns conservative default if not.

    Returns:
        System memory in GB, or 8.0 if detection fails.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3)
    except ImportError:
        # Conservative default - assume enough RAM
        return 8.0
    except Exception:
        return 8.0


def get_optimal_workers(max_workers: Optional[int] = None) -> int:
    """
    Calculate optimal number of workers based on system resources.

    Heuristics:
    - 1-2 CPUs: 1 worker (serial execution)
    - 3-4 CPUs: 2 workers
    - 5+ CPUs: cpu_count - 2 (leave headroom for system)

    Args:
        max_workers: Optional maximum to cap the result.

    Returns:
        Optimal number of workers (>= 1).
    """
    cpu = get_cpu_count()

    if cpu <= 2:
        optimal = 1
    elif cpu <= 4:
        optimal = 2
    else:
        optimal = max(1, cpu - 2)

    if max_workers is not None:
        optimal = min(optimal, max_workers)

    return max(1, optimal)


def should_use_parallel(
    n_items: int,
    max_workers: Optional[int] = None,
    min_ram_gb: float = MIN_RAM_GB_FOR_PARALLEL,
) -> bool:
    """
    Determine if parallel execution should be used.

    Returns False if:
    - System has less than min_ram_gb RAM
    - Number of items is too small to benefit from parallelization
    - Optimal workers is 1

    Args:
        n_items: Number of items to process.
        max_workers: Optional maximum workers.
        min_ram_gb: Minimum RAM required for parallel execution.

    Returns:
        True if parallel execution is beneficial.
    """
    # RAM guard
    ram_gb = get_system_memory_gb()
    if ram_gb < min_ram_gb:
        return False

    # Get optimal workers
    workers = get_optimal_workers(max_workers)

    # No benefit from parallel with 1 worker
    if workers <= 1:
        return False

    # Small job guard - need enough items to justify overhead
    if n_items < workers * MIN_ITEMS_PER_WORKER:
        return False

    return True


def run_parallel(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: Optional[int] = None,
    force_serial: bool = False,
) -> List[R]:
    """
    Execute a function over items in parallel with automatic tuning.

    Features:
    - Automatically determines optimal worker count
    - Falls back to serial execution on low-RAM systems
    - Falls back to serial for small workloads
    - Maintains result order (same as input order)

    Args:
        fn: Function to apply to each item. Must be picklable.
        items: Sequence of items to process.
        max_workers: Optional cap on number of workers.
        force_serial: If True, always use serial execution.

    Returns:
        List of results in same order as input items.

    Note:
        The function `fn` must be defined at module level (not a lambda
        or nested function) for multiprocessing to work correctly.
    """
    items_list = list(items)
    n_items = len(items_list)

    # Empty case
    if n_items == 0:
        return []

    # Decide whether to use parallel execution
    use_parallel = (
        not force_serial
        and should_use_parallel(n_items, max_workers)
    )

    if not use_parallel:
        # Serial execution
        return [fn(item) for item in items_list]

    # Parallel execution
    workers = get_optimal_workers(max_workers)

    try:
        with mp.Pool(processes=workers) as pool:
            results = pool.map(fn, items_list)
        return list(results)
    except Exception:
        # Fallback to serial on any multiprocessing error
        return [fn(item) for item in items_list]


def run_parallel_with_progress(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: Optional[int] = None,
    force_serial: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[R]:
    """
    Execute a function over items in parallel with progress tracking.

    Similar to run_parallel but supports progress callbacks.
    Uses imap_unordered internally for progress tracking, then reorders results.

    Args:
        fn: Function to apply to each item.
        items: Sequence of items to process.
        max_workers: Optional cap on number of workers.
        force_serial: If True, always use serial execution.
        progress_callback: Optional callback(completed, total) for progress updates.

    Returns:
        List of results in same order as input items.
    """
    items_list = list(items)
    n_items = len(items_list)

    if n_items == 0:
        return []

    use_parallel = (
        not force_serial
        and should_use_parallel(n_items, max_workers)
    )

    if not use_parallel:
        # Serial with progress
        results = []
        for i, item in enumerate(items_list):
            results.append(fn(item))
            if progress_callback:
                progress_callback(i + 1, n_items)
        return results

    # Parallel with progress - need to track order
    workers = get_optimal_workers(max_workers)

    # Create indexed items to preserve order
    indexed_items = list(enumerate(items_list))

    def indexed_fn(indexed_item):
        idx, item = indexed_item
        return idx, fn(item)

    try:
        with mp.Pool(processes=workers) as pool:
            indexed_results = []
            for i, result in enumerate(pool.imap_unordered(indexed_fn, indexed_items)):
                indexed_results.append(result)
                if progress_callback:
                    progress_callback(i + 1, n_items)

            # Reorder by original index
            indexed_results.sort(key=lambda x: x[0])
            return [r[1] for r in indexed_results]
    except Exception:
        # Fallback to serial
        results = []
        for i, item in enumerate(items_list):
            results.append(fn(item))
            if progress_callback:
                progress_callback(i + 1, n_items)
        return results


__all__ = [
    "get_cpu_count",
    "get_system_memory_gb",
    "get_optimal_workers",
    "should_use_parallel",
    "run_parallel",
    "run_parallel_with_progress",
    "MIN_RAM_GB_FOR_PARALLEL",
    "MIN_ITEMS_PER_WORKER",
]
