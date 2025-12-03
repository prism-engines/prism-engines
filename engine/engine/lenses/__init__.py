"""
Lenses submodule - re-exports from engine_core/lenses
"""
import sys
from pathlib import Path

# Ensure parent is in path
_pkg_root = Path(__file__).parent.parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# Re-export from engine_core.lenses
from importlib import import_module as _import

_lenses = _import("engine_core.lenses")

BaseLens = _lenses.BaseLens
MagnitudeLens = _lenses.MagnitudeLens
PCALens = _lenses.PCALens
GrangerLens = _lenses.GrangerLens
DMDLens = _lenses.DMDLens
InfluenceLens = _lenses.InfluenceLens
MutualInfoLens = _lenses.MutualInfoLens
ClusteringLens = _lenses.ClusteringLens
DecompositionLens = _lenses.DecompositionLens
WaveletLens = _lenses.WaveletLens
NetworkLens = _lenses.NetworkLens
RegimeSwitchingLens = _lenses.RegimeSwitchingLens
AnomalyLens = _lenses.AnomalyLens
TransferEntropyLens = _lenses.TransferEntropyLens
TDALens = _lenses.TDALens

__all__ = [
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
]
