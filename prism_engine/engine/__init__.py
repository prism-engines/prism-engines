"""
Engine submodule - re-exports from 05_engine
"""
import sys
from pathlib import Path

# Ensure parent is in path
_pkg_root = Path(__file__).parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# Re-export everything from 05_engine
from importlib import import_module as _import

_engine = _import("05_engine")

# Lenses
BaseLens = _engine.BaseLens
MagnitudeLens = _engine.MagnitudeLens
PCALens = _engine.PCALens
GrangerLens = _engine.GrangerLens
DMDLens = _engine.DMDLens
InfluenceLens = _engine.InfluenceLens
MutualInfoLens = _engine.MutualInfoLens
ClusteringLens = _engine.ClusteringLens
DecompositionLens = _engine.DecompositionLens
WaveletLens = _engine.WaveletLens
NetworkLens = _engine.NetworkLens
RegimeSwitchingLens = _engine.RegimeSwitchingLens
AnomalyLens = _engine.AnomalyLens
TransferEntropyLens = _engine.TransferEntropyLens
TDALens = _engine.TDALens

# Orchestration
LensComparator = _engine.LensComparator
ConsensusEngine = _engine.ConsensusEngine
IndicatorEngine = _engine.IndicatorEngine

# Registry
LENS_REGISTRY = _engine.LENS_REGISTRY
get_lens = _engine.get_lens

__all__ = [
    # Lenses
    'BaseLens',
    'MagnitudeLens',
    'PCALens',
    'GrangerLens',
    'DMDLens',
    'InfluenceLens',
    'MutualInfoLens',
    'ClusteringLens',
    'DecompositionLens',
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
    # Registry
    'LENS_REGISTRY',
    'get_lens',
]
