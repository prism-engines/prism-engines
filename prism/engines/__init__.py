"""
PRISM Engines Module

Registry of all analysis engines.

Usage:
    from prism.engines import get_engine, list_engines
    
    # Get specific engine
    pca = get_engine("pca")
    result = pca.execute(indicators=["SPY", "AGG"])
    
    # List available engines
    for name, info in list_engines().items():
        print(f"{name}: {info['description']}")
"""

from typing import Dict, Any, Optional, Type
from pathlib import Path

from .base import BaseEngine, EngineResult

# Import all engines
from .pca import PCAEngine
from .cross_correlation import CrossCorrelationEngine
from .rolling_beta import RollingBetaEngine
from .clustering import ClusteringEngine
from .hurst import HurstEngine
from .granger import GrangerEngine
from .mutual_information import MutualInformationEngine
from .transfer_entropy import TransferEntropyEngine
from .entropy import EntropyEngine
from .hmm import HMMEngine
from .dmd import DMDEngine
from .wavelet import WaveletEngine
from .spectral import SpectralEngine
from .garch import GARCHEngine
from .dtw import DTWEngine
from .copula import CopulaEngine
from .rqa import RQAEngine
from .cointegration import CointegrationEngine
from .lyapunov import LyapunovEngine
from .distance import DistanceEngine


# Engine registry
ENGINE_REGISTRY: Dict[str, Type[BaseEngine]] = {
    "pca": PCAEngine,
    "cross_correlation": CrossCorrelationEngine,
    "rolling_beta": RollingBetaEngine,
    "clustering": ClusteringEngine,
    "hurst": HurstEngine,
    "granger": GrangerEngine,
    "mutual_information": MutualInformationEngine,
    "transfer_entropy": TransferEntropyEngine,
    "wavelet": WaveletEngine,
    "entropy": EntropyEngine,
    "spectral": SpectralEngine,
    "garch": GARCHEngine,
    "hmm": HMMEngine,
    "dmd": DMDEngine,
    "dtw": DTWEngine,
    "copula": CopulaEngine,
    "rqa": RQAEngine,
    "cointegration": CointegrationEngine,
    "lyapunov": LyapunovEngine,
    "distance": DistanceEngine,
}


# Engine metadata
ENGINE_INFO: Dict[str, Dict[str, Any]] = {
    "pca": {
        "phase": "derived",
        "category": "structure",
        "description": "Principal Component Analysis - variance structure",
        "normalization": "zscore",
        "priority": 1,
    },
    "cross_correlation": {
        "phase": "derived",
        "category": "correlation",
        "description": "Lead/lag relationships between indicators",
        "normalization": "zscore",
        "priority": 1,
    },
    "rolling_beta": {
        "phase": "derived",
        "category": "regression",
        "description": "Time-varying sensitivity to reference",
        "normalization": "returns",
        "priority": 1,
    },
    "clustering": {
        "phase": "structure",
        "category": "structure",
        "description": "Behavioral grouping of indicators",
        "normalization": "zscore",
        "priority": 1,
    },
    "hurst": {
        "phase": "derived",
        "category": "complexity",
        "description": "Long-term memory / persistence",
        "normalization": None,
        "priority": 1,
    },
    "granger": {
        "phase": "derived",
        "category": "causality",
        "description": "Directional predictive relationships",
        "normalization": None,
        "priority": 2,
    },
    "mutual_information": {
        "phase": "derived",
        "category": "information",
        "description": "Non-linear dependence",
        "normalization": "discretize",
        "priority": 2,
    },
    "transfer_entropy": {
        "phase": "derived",
        "category": "information",
        "description": "Directional information flow",
        "normalization": "discretize",
        "priority": 2,
    },
    "wavelet": {
        "phase": "derived",
        "category": "frequency",
        "description": "Time-frequency co-movement",
        "normalization": None,
        "priority": 3,
    },
    "entropy": {
        "phase": "derived",
        "category": "complexity",
        "description": "Complexity and predictability",
        "normalization": "varies",
        "priority": 3,
    },
    "spectral": {
        "phase": "derived",
        "category": "frequency",
        "description": "Dominant cycles",
        "normalization": "diff",
        "priority": 3,
    },
    "garch": {
        "phase": "derived",
        "category": "volatility",
        "description": "Conditional variance dynamics",
        "normalization": "returns",
        "priority": 4,
    },
    "hmm": {
        "phase": "structure",
        "category": "latent_state",
        "description": "Hidden Markov Model latent state identification",
        "normalization": "zscore",
        "priority": 4,
    },
    "dmd": {
        "phase": "structure",
        "category": "dynamics",
        "description": "Dynamic Mode Decomposition",
        "normalization": "zscore",
        "priority": 4,
    },
    "dtw": {
        "phase": "derived",
        "category": "similarity",
        "description": "Dynamic Time Warping - shape similarity",
        "normalization": "zscore",
        "priority": 3,
    },
    "copula": {
        "phase": "derived",
        "category": "dependence",
        "description": "Copula - tail dependence structure",
        "normalization": "rank",
        "priority": 4,
    },
    "rqa": {
        "phase": "derived",
        "category": "complexity",
        "description": "Recurrence Quantification Analysis",
        "normalization": "zscore",
        "priority": 4,
    },
    "cointegration": {
        "phase": "derived",
        "category": "causality",
        "description": "Long-run equilibrium relationships",
        "normalization": None,
        "priority": 3,
    },
    "lyapunov": {
        "phase": "derived",
        "category": "complexity",
        "description": "Lyapunov exponent - chaos detection",
        "normalization": "zscore",
        "priority": 4,
    },
    "distance": {
        "phase": "derived",
        "category": "geometry",
        "description": "Geometric distances (Euclidean, Mahalanobis)",
        "normalization": "zscore",
        "priority": 2,
    },
}


def get_engine(name: str) -> BaseEngine:
    """
    Get an engine instance by name.
    """
    if name not in ENGINE_REGISTRY:
        available = list(ENGINE_REGISTRY.keys())
        raise ValueError(f"Unknown engine: {name}. Available: {available}")

    engine_class = ENGINE_REGISTRY[name]
    return engine_class()


def list_engines() -> Dict[str, Dict[str, Any]]:
    """List all available engines with metadata."""
    return ENGINE_INFO.copy()


def get_engines_by_phase(phase: str) -> list:
    """Get all engines for a specific phase."""
    return [name for name, info in ENGINE_INFO.items() if info["phase"] == phase]


def get_engines_by_priority(max_priority: int) -> list:
    """Get engines up to a priority level."""
    return [name for name, info in ENGINE_INFO.items() if info["priority"] <= max_priority]


__all__ = [
    "BaseEngine",
    "EngineResult",
    "get_engine",
    "list_engines",
    "get_engines_by_phase",
    "get_engines_by_priority",
    "ENGINE_REGISTRY",
    "ENGINE_INFO",
]
