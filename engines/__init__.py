"""
PRISM Engines Package
======================

Auto-discoverable analysis engines loaded from registry.yaml.

Usage:
    from engines import EngineLoader, list_engines, load_engine

    # List all engines
    engines = list_engines()

    # Load a specific engine
    pca = load_engine("pca")

    # Run analysis
    results = pca.rank_indicators(df)
"""

from .engine_loader import (
    EngineLoader,
    EngineLoadError,
    get_loader,
    list_engines,
    list_lenses,
    load_engine,
    run_engine,
)

__all__ = [
    "EngineLoader",
    "EngineLoadError",
    "get_loader",
    "list_engines",
    "list_lenses",
    "load_engine",
    "run_engine",
]
