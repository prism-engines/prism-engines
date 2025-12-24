"""
PRISM Core - Geometric measurement without regime semantics.

Components:
    - Semantic Guard: Runtime enforcement against regime semantics
    - Geometry Reference: Homeostatic band definition and distance computation
    - Excursion Metrics: Time outside homeostatic band tracking
    - Trajectory Dynamics: Velocity, acceleration, mean-reversion, hazard estimation
    - Conditional Attribution: Which indicators lead excursions
"""

from .semantic_guard import (
    check_output_semantics,
    validate_hmm_output,
    validate_agent_output,
    validate_engine_output,
    FORBIDDEN_TOKENS,
    ALLOWED_TOKENS,
    SemanticViolationError,
    is_allowed_token,
    is_forbidden_token,
)

from .geometry_reference import (
    HomeostasisBand,
    DistanceResult,
    GeometryReference,
    fit_homeostasis_band,
    compute_deformation,
)

from .excursion_metrics import (
    ExcursionSummary,
    ExcursionEvent,
    ExcursionTracker,
    compute_excursion_rate,
    count_boundary_crossings,
)

from .trajectory_dynamics import (
    TrajectoryState,
    MeanReversionEstimate,
    HazardEstimates,
    PullStrength,
    TrajectoryAnalyzer,
    compute_velocity,
    classify_trajectory,
)

from .conditional_attribution import (
    IndicatorAttribution,
    AttributionReport,
    ConditionalAttribution,
    compute_indicator_lead,
    rank_indicators_by_effect,
)

__all__ = [
    # Semantic Guard
    "check_output_semantics",
    "validate_hmm_output",
    "validate_agent_output",
    "validate_engine_output",
    "FORBIDDEN_TOKENS",
    "ALLOWED_TOKENS",
    "SemanticViolationError",
    "is_allowed_token",
    "is_forbidden_token",
    # Geometry Reference (HSB)
    "HomeostasisBand",
    "DistanceResult",
    "GeometryReference",
    "fit_homeostasis_band",
    "compute_deformation",
    # Excursion Metrics
    "ExcursionSummary",
    "ExcursionEvent",
    "ExcursionTracker",
    "compute_excursion_rate",
    "count_boundary_crossings",
    # Trajectory Dynamics
    "TrajectoryState",
    "MeanReversionEstimate",
    "HazardEstimates",
    "PullStrength",
    "TrajectoryAnalyzer",
    "compute_velocity",
    "classify_trajectory",
    # Conditional Attribution
    "IndicatorAttribution",
    "AttributionReport",
    "ConditionalAttribution",
    "compute_indicator_lead",
    "rank_indicators_by_effect",
]
