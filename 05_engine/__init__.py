"""
PRISM Engine - Stage 05: Core Engine (The Math)

The heart of PRISM - multiple mathematical "lenses" for analyzing data.

Each lens provides a different perspective:
    - magnitude_lens: Simple vector magnitude analysis
    - pca_lens: Principal Component Analysis
    - granger_lens: Granger causality testing
    - dmd_lens: Dynamic Mode Decomposition
    - influence_lens: Influence/importance scoring
    - mutual_info_lens: Mutual information analysis
    - clustering_lens: Hierarchical clustering
    - decomposition_lens: Time series decomposition

Advanced lenses:
    - wavelet_lens: Wavelet transform analysis
    - network_lens: Network graph analysis
    - regime_switching_lens: Regime detection
    - anomaly_lens: Anomaly detection
    - transfer_entropy_lens: Information flow
    - tda_lens: Topological data analysis
"""

from .lenses import (
    BaseLens,
    MagnitudeLens,
    PCALens,
    GrangerLens,
    DMDLens,
    InfluenceLens,
    MutualInfoLens,
    ClusteringLens,
    DecompositionLens,
    WaveletLens,
    NetworkLens,
    RegimeSwitchingLens,
    AnomalyLens,
    TransferEntropyLens,
    TDALens,
)

from .orchestration import (
    LensComparator,
    ConsensusEngine,
    IndicatorEngine,
)

__all__ = [
    # Base
    'BaseLens',
    # Basic lenses
    'MagnitudeLens',
    'PCALens',
    'GrangerLens',
    'DMDLens',
    'InfluenceLens',
    'MutualInfoLens',
    'ClusteringLens',
    'DecompositionLens',
    # Advanced lenses
    'WaveletLens',
    'NetworkLens',
    'RegimeSwitchingLens',
    'AnomalyLens',
    'TransferEntropyLens',
    'TDALens',
    # Orchestration
    'LensComparator',
    'ConsensusEngine',
    'IndicatorEngine',
]

# Lens registry for dynamic loading
LENS_REGISTRY = {
    'magnitude': MagnitudeLens,
    'pca': PCALens,
    'granger': GrangerLens,
    'dmd': DMDLens,
    'influence': InfluenceLens,
    'mutual_info': MutualInfoLens,
    'clustering': ClusteringLens,
    'decomposition': DecompositionLens,
    'wavelet': WaveletLens,
    'network': NetworkLens,
    'regime': RegimeSwitchingLens,
    'anomaly': AnomalyLens,
    'transfer_entropy': TransferEntropyLens,
    'tda': TDALens,
}


def get_lens(name: str) -> 'BaseLens':
    """
    Get a lens instance by name.

    Args:
        name: Lens name (e.g., 'pca', 'granger')

    Returns:
        Lens instance
    """
    if name not in LENS_REGISTRY:
        raise ValueError(f"Unknown lens: {name}. Available: {list(LENS_REGISTRY.keys())}")
    return LENS_REGISTRY[name]()
