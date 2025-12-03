"""
Orchestration submodule - re-exports from engine_core/orchestration
"""
import sys
from pathlib import Path

# Ensure parent is in path
_pkg_root = Path(__file__).parent.parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# Re-export from engine_core.orchestration
from importlib import import_module as _import

_orch = _import("engine_core.orchestration")

LensComparator = _orch.LensComparator
ConsensusEngine = _orch.ConsensusEngine
IndicatorEngine = _orch.IndicatorEngine

__all__ = [
    'LensComparator',
    'ConsensusEngine',
    'IndicatorEngine',
]
