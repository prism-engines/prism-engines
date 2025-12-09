"""
PRISM Engine - Stage 05: Core Engine (The Math)

The heart of PRISM - multiple mathematical "lenses" for analyzing data.

Lenses are auto-discovered from engine_core/lenses/*_lens.py files.
Convention: file {name}_lens.py contains class {Name}Lens

Available lenses (auto-discovered):
    - magnitude: Simple vector magnitude analysis
    - pca: Principal Component Analysis
    - granger: Granger causality testing
    - dmd: Dynamic Mode Decomposition
    - influence: Influence/importance scoring
    - mutual_info: Mutual information analysis
    - clustering: Hierarchical clustering
    - decomposition: Time series decomposition
    - wavelet: Wavelet transform analysis
    - network: Network graph analysis
    - regime_switching: Regime detection
    - anomaly: Anomaly detection
    - transfer_entropy: Information flow
    - tda: Topological data analysis
"""

# Auto-discovered lens registry
from .lenses import (
    BaseLens,
    LENS_CLASSES,
    get_lens,
    list_lenses,
)

# Orchestration components
from .orchestration import (
    LensComparator,
    ConsensusEngine,
    IndicatorEngine,
)

# Re-export LENS_CLASSES as LENS_REGISTRY for backward compatibility
LENS_REGISTRY = LENS_CLASSES

# Dynamically expose lens classes at module level for backward compatibility
# e.g., `from engine_core import PCALens` still works
_module = __import__(__name__)
for _name, _cls in LENS_CLASSES.items():
    setattr(_module, _cls.__name__, _cls)

__all__ = [
    # Registry
    'LENS_CLASSES',
    'LENS_REGISTRY',
    'get_lens',
    'list_lenses',
    # Base
    'BaseLens',
    # Orchestration
    'LensComparator',
    'ConsensusEngine',
    'IndicatorEngine',
] + [cls.__name__ for cls in LENS_CLASSES.values()]
