"""
Orchestration submodule - re-exports from 05_engine/orchestration
"""
import sys
from pathlib import Path

# Ensure parent is in path
_pkg_root = Path(__file__).parent.parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# Re-export from 05_engine.orchestration
from importlib import import_module as _import

_orch = _import("05_engine.orchestration")

LensComparator = _orch.LensComparator
ConsensusEngine = _orch.ConsensusEngine
IndicatorEngine = _orch.IndicatorEngine

__all__ = [
    'LensComparator',
    'ConsensusEngine',
    'IndicatorEngine',
]
