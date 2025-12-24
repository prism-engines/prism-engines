"""
PRISM Engines Registry.

Available engines:
    - correlation: Pairwise correlation analysis
    - pca: Principal component analysis
    - hurst: Long-term memory/persistence
"""

from .base import BaseEngine, EngineResult
from .correlation import CorrelationEngine
from .pca import PCAEngine
from .hurst import HurstEngine

# Engine registry
ENGINES = {
    "correlation": CorrelationEngine,
    "pca": PCAEngine,
    "hurst": HurstEngine,
}

DEFAULT_ENGINES = ["correlation", "pca", "hurst"]


def get_engine(name: str) -> BaseEngine:
    """Get an engine instance by name."""
    if name not in ENGINES:
        raise ValueError(f"Unknown engine: {name}. Available: {list(ENGINES.keys())}")
    return ENGINES[name]()


def list_engines() -> list:
    """List all available engine names."""
    return list(ENGINES.keys())


__all__ = [
    "BaseEngine",
    "EngineResult",
    "CorrelationEngine",
    "PCAEngine",
    "HurstEngine",
    "ENGINES",
    "DEFAULT_ENGINES",
    "get_engine",
    "list_engines",
]
