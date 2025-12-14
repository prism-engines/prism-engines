# prism/schema/__init__.py
"""
Geometry artifact schemas.

These are the canonical output types for PRISM observations.
Immutable, serializable, no interpretation.
"""

from prism.schema.observation import (
    LensObservation,
    IndicatorGeometry,
    SystemSnapshot,
)

__all__ = [
    "LensObservation",
    "IndicatorGeometry",
    "SystemSnapshot",
]
