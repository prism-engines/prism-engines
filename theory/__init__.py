"""
Theory Module
=============

Physics formalism and theoretical framework for PRISM.
Provides rigorous mathematical foundations for a skeptical physicist.

Modules:
- geometry: Metric tensor and distance definitions
- hamiltonian: Energy-based dynamics (if applicable)
- symmetries: Invariance properties
- dimensions: Dimensional analysis
- ergodicity: Ergodicity assumptions and tests
"""

from .geometry import (
    MetricTensor,
    compute_distance,
    compute_geodesic,
    GeometryReport
)

from .symmetries import (
    test_scale_invariance,
    test_translation_invariance,
    SymmetryReport
)

from .dimensions import (
    DimensionalAnalyzer,
    DimensionalReport
)

from .ergodicity import (
    ErgodicityAnalyzer,
    ErgodicityReport
)

__all__ = [
    'MetricTensor',
    'compute_distance',
    'compute_geodesic',
    'GeometryReport',
    'test_scale_invariance',
    'test_translation_invariance',
    'SymmetryReport',
    'DimensionalAnalyzer',
    'DimensionalReport',
    'ErgodicityAnalyzer',
    'ErgodicityReport'
]
