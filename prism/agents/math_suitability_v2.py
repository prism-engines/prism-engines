"""PRISM Math Suitability Agent v2 - Import Alias

This module exists for backwards compatibility with scripts that import from:
    from prism.agents.math_suitability_v2 import ...

The actual implementation lives in agent_math_suitability.py
"""

from prism.agents.agent_math_suitability import (
    EligibilityStatus,
    WindowGeometryResult,
    WindowEligibility,
    WindowSuitabilityPolicy,
    WindowSuitabilityAgent,
)

__all__ = [
    "EligibilityStatus",
    "WindowGeometryResult",
    "WindowEligibility",
    "WindowSuitabilityPolicy",
    "WindowSuitabilityAgent",
]
