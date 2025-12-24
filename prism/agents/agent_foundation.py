"""
PRISM Agent Foundation - Engine Admission Control Spec Implementation

Shared infrastructure for PRISM's agent layer implementing the Engine Admission
Control Spec. Provides base classes, result dataclasses, diagnostic registry,
and orchestration scaffolding with proper null calibration, stability scoring,
weight capping, and consensus reporting.

Spec Sections Implemented:
    A: Data Quality Validation
    B: Null Calibration (surrogate testing)
    C: Stability Scoring (bootstrap/subsample/temporal)
    D: Weight Computation and Capping
    H: Consensus Reporting

Architecture:
    Agent1 (Data Quality) -> Agent2 (Geometry) -> Agent3 (Routing)
    -> [engines] -> Agent4 (Stability) -> Agent5 (Blind Spot)
    -> ConsensusReport
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging
import yaml

# HSB imports
from prism.core import (
    DistanceResult,
    ExcursionSummary,
    TrajectoryState,
    HazardEstimates,
    MeanReversionEstimate,
    GeometryReference,
    ExcursionTracker,
    TrajectoryAnalyzer,
    check_output_semantics,
    validate_agent_output,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Audit Entry
# =============================================================================

@dataclass
class AuditEntry:
    """Single logged decision from an agent."""
    timestamp: str
    agent_name: str
    decision: str
    reason: str
    window_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Spec Section B2: Null Calibration Result
# =============================================================================

@dataclass
class NullCalibrationResult:
    """
    Result of null/surrogate calibration test (Spec Section B2).

    Determines whether an engine's output is distinguishable from what
    would be produced by structureless data (phase-randomized surrogates).

    Pass criteria (Spec B3):
        - p_value <= 0.05, OR
        - effect_size >= 2.0
    """
    null_mean: float              # Mean of null distribution
    null_std: float               # Std of null distribution
    observed_value: float         # Engine's actual output metric
    empirical_p_value: float      # (count(null >= observed) + 1) / (N_null + 1)
    effect_size: float            # (observed - null_mean) / (null_std + eps)
    passed: bool                  # p <= 0.05 OR effect_size >= 2.0
    n_surrogates: int = 100       # Number of surrogates used

    def __post_init__(self):
        """Compute pass status if not provided."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            # Recompute passed based on criteria
            self.passed = (self.empirical_p_value <= 0.05) or (self.effect_size >= 2.0)


# =============================================================================
# Spec Section C1-C2: Stability Score
# =============================================================================

@dataclass
class StabilityScore:
    """
    Stability assessment for an engine's output (Spec Section C1-C2).

    Measures how stable the engine's key metric is under resampling.
    Consensus requires score >= 0.6 (Spec C3).

    Methods:
        - "bootstrap": Resample with replacement
        - "subsample": Random 80% subsets
        - "temporal": Rolling window blocks
    """
    score: float                  # S in [0,1], require >= 0.6 for consensus
    method: str                   # "bootstrap", "subsample", "temporal"
    cv: float = 0.0               # Coefficient of variation
    iqr_ratio: float = 0.0        # IQR / median ratio
    n_resamples: int = 50         # Number of resamples used
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passes_threshold(self) -> bool:
        """Check if stability passes consensus threshold (0.6)."""
        return self.score >= 0.6


# =============================================================================
# Spec Section D1: Engine Contribution
# =============================================================================

@dataclass
class EngineContribution:
    """
    Per-engine per-window contribution record (Spec Section D1).

    Captures all factors that determine an engine's weight in consensus:
    - Gate pass (did it pass admission gates?)
    - Null pass (did it beat surrogate baseline?)
    - Stability score (is it robust to resampling?)
    - Benchmark score (global validation performance)
    - Domain fit (is this domain appropriate?)

    Weight computation (Spec D1):
        raw_weight = gate_pass * null_pass * stability_score * benchmark_score * domain_fit
        capped_weight = min(raw_weight, 0.25)  # Spec D2
    """
    engine_name: str
    gate_pass: bool               # 0 or 1 - passed admission gates
    null_pass: bool               # 0 or 1 - passed null calibration
    stability_score: float        # [0,1] - stability under resampling
    benchmark_score: float        # [0,1] - global benchmark performance
    domain_fit: float             # [0,1] - domain appropriateness
    raw_weight: float             # Product of above factors
    capped_weight: float          # min(raw_weight, 0.25) per Spec D2
    gated_reason: str = ""        # Why excluded, if applicable
    null_calibration: Optional[NullCalibrationResult] = None
    stability_details: Optional[StabilityScore] = None

    def __post_init__(self):
        """Compute weights if not provided."""
        if self.raw_weight == 0.0 and self.gate_pass and self.null_pass:
            self.raw_weight = (
                float(self.gate_pass) *
                float(self.null_pass) *
                self.stability_score *
                self.benchmark_score *
                self.domain_fit
            )
        if self.capped_weight == 0.0:
            self.capped_weight = min(self.raw_weight, 0.25)

    @property
    def is_contributing(self) -> bool:
        """Check if engine contributes to consensus."""
        return self.gate_pass and self.null_pass and self.capped_weight > 0


# =============================================================================
# Spec Section H: Consensus Report
# =============================================================================

@dataclass
class ConsensusReport:
    """
    Complete consensus report for a window (Spec Section H).

    Required output containing:
    - All contributing engines with their weights
    - All excluded engines with reasons
    - Normalized weights (sum to 1.0)
    - Sensitivity analysis (leave-one-out)
    """
    window_id: str
    contributing_engines: List[EngineContribution]
    excluded_engines: List[EngineContribution]
    normalized_weights: Dict[str, float]  # Sum to 1.0
    sensitivity_check: Dict[str, Any]     # Does consensus change if one engine removed?
    total_raw_weight: float = 0.0
    n_contributing: int = 0
    n_excluded: int = 0
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute derived fields."""
        self.n_contributing = len(self.contributing_engines)
        self.n_excluded = len(self.excluded_engines)
        self.total_raw_weight = sum(e.raw_weight for e in self.contributing_engines)

    @property
    def is_valid(self) -> bool:
        """Check if consensus is valid (at least 2 contributing engines)."""
        return self.n_contributing >= 2

    def get_top_engines(self, n: int = 3) -> List[str]:
        """Get top N engines by normalized weight."""
        sorted_weights = sorted(
            self.normalized_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [name for name, _ in sorted_weights[:n]]


# =============================================================================
# Agent Result Dataclasses
# =============================================================================

@dataclass
class DataQualityResult:
    """
    Result from the Data Quality Agent (Spec Section A).

    Validates data preprocessing and checks for:
    - Frequency alignment
    - Missing data handling
    - Outlier treatment
    - Scaling methods
    """
    frequency_aligned: bool = True
    missing_rate: float = 0.0
    missing_policy_applied: str = "none"
    outlier_handling: str = "none"
    scaling_method: str = "none"
    safety_flag: str = "safe"     # "safe" / "questionable" / "unsafe"
    reasons: List[str] = field(default_factory=list)
    transformation_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.safety_flag == "safe"


@dataclass
class RoutingResult:
    """Result from the Routing Agent."""
    engine_eligibility: Dict[str, str]    # engine -> "allowed"/"downweighted"/"suppressed"
    weights: Dict[str, float]
    suppression_reasons: Dict[str, str] = field(default_factory=dict)
    min_window_violations: List[str] = field(default_factory=list)


@dataclass
class StabilityResult:
    """
    Result from the Stability Agent (Spec Sections B, C, D).

    Contains null calibration results, stability scores, and
    computed engine contributions with capped weights.

    Note: Legacy fields (trustworthiness_scores, artifact_flags, adjusted_weights)
    are maintained for backward compatibility with existing agents.
    """
    # Legacy fields (required for backward compatibility)
    trustworthiness_scores: Dict[str, float]
    artifact_flags: List[str]
    adjusted_weights: Dict[str, float] = field(default_factory=dict)
    # Spec-compliant fields (optional for backward compatibility)
    null_results: Dict[str, NullCalibrationResult] = field(default_factory=dict)
    stability_scores: Dict[str, StabilityScore] = field(default_factory=dict)
    contributions: List[EngineContribution] = field(default_factory=list)
    consensus_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class BlindSpotResult:
    """Result from the Blind Spot Agent."""
    blind_spots: List[str]
    escalation_recommendations: List[str] = field(default_factory=list)
    sensitivity_report: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HSB: Geometry Measurement
# =============================================================================

@dataclass
class GeometryMeasurement:
    """
    Complete geometry measurement for a time point.

    Combines all HSB outputs into a single measurement object that
    captures the current state relative to homeostasis.
    """
    timestamp: str

    # Distance from homeostasis
    distance: DistanceResult

    # Trajectory dynamics
    trajectory: Optional[TrajectoryState] = None

    # Excursion metrics
    excursion: Optional[ExcursionSummary] = None

    # Hazard estimates
    hazards: Optional[HazardEstimates] = None

    # Mean reversion estimate
    mean_reversion: Optional[MeanReversionEstimate] = None

    @property
    def is_inside_band(self) -> bool:
        """True if currently inside homeostatic band."""
        return self.distance.inside_band

    @property
    def is_escalating(self) -> bool:
        """True if trajectory is escalating away from homeostasis."""
        if self.trajectory is None:
            return False
        return self.trajectory.trajectory_type == "escalating"

    @property
    def is_reverting(self) -> bool:
        """True if trajectory is reverting toward homeostasis."""
        if self.trajectory is None:
            return False
        return self.trajectory.trajectory_type == "reverting"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "timestamp": self.timestamp,
            "distance": self.distance.distance,
            "inside_band": self.distance.inside_band,
            "percentile": self.distance.percentile,
        }
        if self.trajectory:
            result["trajectory_type"] = self.trajectory.trajectory_type
            result["velocity"] = self.trajectory.velocity
        if self.excursion:
            result["current_streak"] = self.excursion.current_streak_length
            result["pct_time_outside"] = self.excursion.pct_time_outside
        if self.hazards:
            result["p_reentry_4wk"] = self.hazards.p_reentry_4wk
            result["p_escalate_4wk"] = self.hazards.p_escalate_4wk
        if self.mean_reversion:
            result["beta"] = self.mean_reversion.beta
            result["is_mean_reverting"] = self.mean_reversion.is_mean_reverting
        return result


@dataclass
class PipelineResult:
    """
    Complete result from the agent pipeline.

    Wraps all agent outputs and provides the final ConsensusReport
    as required by Spec Section H.
    """
    quality: Optional[DataQualityResult] = None
    geometry: Optional[Any] = None  # GeometryProfile
    routing: Optional[RoutingResult] = None
    engine_results: Optional[Dict[str, Any]] = None
    stability: Optional[StabilityResult] = None
    blindspots: Optional[BlindSpotResult] = None
    consensus: Optional[ConsensusReport] = None  # Spec Section H
    audit_trail: List[AuditEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """Check if the pipeline ran without major issues."""
        if self.quality and self.quality.safety_flag == "unsafe":
            return False
        if self.blindspots and len(self.blindspots.blind_spots) > 5:
            return False
        if self.consensus and not self.consensus.is_valid:
            return False
        return True

    @property
    def normalized_weights(self) -> Dict[str, float]:
        """Get normalized consensus weights."""
        if self.consensus:
            return self.consensus.normalized_weights
        return {}


# =============================================================================
# Diagnostic Registry
# =============================================================================

class DiagnosticRegistry:
    """
    Central store for all computed metrics across agents.

    Supports window-based storage for multi-window analysis
    and provides ConsensusReport generation per Spec Section H.

    Usage:
        registry = DiagnosticRegistry()
        registry.store("routing", "pca_weight", 0.35, window_id="w001")
        weight = registry.get("routing", "pca_weight", window_id="w001")
        report = registry.get_engine_report("w001")
    """

    def __init__(self):
        self._metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}  # window -> agent -> metric -> value
        self._global_metrics: Dict[str, Dict[str, Any]] = {}      # agent -> metric -> value (no window)
        self._audit_trail: List[AuditEntry] = []
        self._contributions: Dict[str, List[EngineContribution]] = {}  # window -> contributions
        self._current_window: Optional[str] = None

    def set_current_window(self, window_id: str) -> None:
        """Set the current window for subsequent operations."""
        self._current_window = window_id

    def store(
        self,
        agent: str,
        metric_name: str,
        value: Any,
        window_id: Optional[str] = None
    ) -> None:
        """
        Store a metric value (Spec Section H compliant).

        Args:
            agent: Agent name (e.g., "routing", "stability")
            metric_name: Name of the metric
            value: Metric value
            window_id: Optional window identifier
        """
        wid = window_id or self._current_window or "_global"

        if wid == "_global":
            if agent not in self._global_metrics:
                self._global_metrics[agent] = {}
            self._global_metrics[agent][metric_name] = value
        else:
            if wid not in self._metrics:
                self._metrics[wid] = {}
            if agent not in self._metrics[wid]:
                self._metrics[wid][agent] = {}
            self._metrics[wid][agent][metric_name] = value

        logger.debug(f"Registry: [{wid}] {agent}.{metric_name} = {value}")

    # Alias for backward compatibility
    def set(self, agent: str, metric_name: str, value: Any) -> None:
        """Store a metric value (backward compatible)."""
        self.store(agent, metric_name, value)

    def get(
        self,
        agent: str,
        metric_name: str,
        window_id: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        Retrieve a metric value.

        Args:
            agent: Agent name
            metric_name: Name of the metric
            window_id: Optional window identifier
            default: Value to return if metric not found

        Returns:
            Metric value or default
        """
        wid = window_id or self._current_window or "_global"

        if wid == "_global":
            return self._global_metrics.get(agent, {}).get(metric_name, default)

        return self._metrics.get(wid, {}).get(agent, {}).get(metric_name, default)

    def get_agent_metrics(self, agent: str, window_id: Optional[str] = None) -> Dict[str, Any]:
        """Get all metrics for a specific agent."""
        wid = window_id or self._current_window or "_global"

        if wid == "_global":
            return self._global_metrics.get(agent, {}).copy()

        return self._metrics.get(wid, {}).get(agent, {}).copy()

    def get_all(self, window_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get all metrics from all agents."""
        wid = window_id or self._current_window

        if wid is None:
            return {k: v.copy() for k, v in self._global_metrics.items()}

        return {k: v.copy() for k, v in self._metrics.get(wid, {}).items()}

    def store_contribution(
        self,
        contribution: EngineContribution,
        window_id: Optional[str] = None
    ) -> None:
        """Store an engine contribution for consensus reporting."""
        wid = window_id or self._current_window or "_global"
        if wid not in self._contributions:
            self._contributions[wid] = []
        self._contributions[wid].append(contribution)

    def get_engine_report(
        self,
        window_id: Optional[str] = None,
        weight_cap: float = 0.25
    ) -> ConsensusReport:
        """
        Generate ConsensusReport for a window (Spec Section H).

        Args:
            window_id: Window identifier
            weight_cap: Maximum weight per engine (Spec D2, default 0.25)

        Returns:
            ConsensusReport with all required fields
        """
        wid = window_id or self._current_window or "_global"
        contributions = self._contributions.get(wid, [])

        # Separate contributing and excluded engines
        contributing = [c for c in contributions if c.is_contributing]
        excluded = [c for c in contributions if not c.is_contributing]

        # Compute normalized weights
        total_capped = sum(c.capped_weight for c in contributing)
        normalized = {}
        if total_capped > 0:
            for c in contributing:
                normalized[c.engine_name] = c.capped_weight / total_capped

        # Sensitivity check: leave-one-out analysis
        sensitivity = self._compute_sensitivity(contributing, total_capped, weight_cap)

        return ConsensusReport(
            window_id=wid,
            contributing_engines=contributing,
            excluded_engines=excluded,
            normalized_weights=normalized,
            sensitivity_check=sensitivity,
        )

    def _compute_sensitivity(
        self,
        contributing: List[EngineContribution],
        total_weight: float,
        weight_cap: float
    ) -> Dict[str, Any]:
        """Compute leave-one-out sensitivity analysis."""
        sensitivity = {
            "stable": True,
            "max_weight_shift": 0.0,
            "vulnerable_to": [],
        }

        if len(contributing) < 2:
            sensitivity["stable"] = False
            return sensitivity

        for left_out in contributing:
            remaining = [c for c in contributing if c.engine_name != left_out.engine_name]
            remaining_total = sum(c.capped_weight for c in remaining)

            if remaining_total > 0:
                # Compute how much weights would shift
                for c in remaining:
                    new_norm = c.capped_weight / remaining_total
                    old_norm = c.capped_weight / total_weight if total_weight > 0 else 0
                    shift = abs(new_norm - old_norm)
                    sensitivity["max_weight_shift"] = max(
                        sensitivity["max_weight_shift"],
                        shift
                    )

                    # If removing one engine shifts any other by > 20%, flag it
                    if shift > 0.2:
                        sensitivity["stable"] = False
                        if left_out.engine_name not in sensitivity["vulnerable_to"]:
                            sensitivity["vulnerable_to"].append(left_out.engine_name)

        return sensitivity

    def add_audit_entry(self, entry: AuditEntry) -> None:
        """Add an entry to the audit trail."""
        self._audit_trail.append(entry)
        logger.debug(f"Audit: [{entry.agent_name}] {entry.decision}")

    def export_audit_trail(self) -> List[AuditEntry]:
        """Export the full audit trail."""
        return self._audit_trail.copy()

    def clear(self, window_id: Optional[str] = None) -> None:
        """Clear metrics and audit trail."""
        if window_id:
            self._metrics.pop(window_id, None)
            self._contributions.pop(window_id, None)
        else:
            self._metrics.clear()
            self._global_metrics.clear()
            self._audit_trail.clear()
            self._contributions.clear()
            self._current_window = None


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all PRISM agents.

    All agents must implement:
        - name: Agent identifier
        - run(): Execute agent logic
        - explain(): Human-readable explanation of last run

    Provides:
        - log(): Structured logging to shared registry
        - Access to DiagnosticRegistry
    """

    def __init__(self, registry: DiagnosticRegistry, config: Optional[Dict] = None):
        """
        Initialize agent.

        Args:
            registry: Shared DiagnosticRegistry instance
            config: Optional agent-specific configuration
        """
        self._registry = registry
        self._config = config or {}
        self._last_result: Optional[Any] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier."""
        ...

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute agent logic.

        Args:
            inputs: Input data for the agent

        Returns:
            Agent-specific result dataclass
        """
        ...

    @abstractmethod
    def explain(self) -> str:
        """
        Human-readable explanation of the last run.

        Returns:
            Explanation string
        """
        ...

    def log(
        self,
        decision: str,
        reason: str,
        metrics: Optional[Dict] = None,
        window_id: Optional[str] = None
    ) -> None:
        """
        Log a decision to the shared registry.

        Args:
            decision: What was decided
            reason: Why it was decided
            metrics: Optional associated metrics
            window_id: Optional window identifier
        """
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            agent_name=self.name,
            decision=decision,
            reason=reason,
            window_id=window_id,
            metrics=metrics or {},
        )
        self._registry.add_audit_entry(entry)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)


# =============================================================================
# Agent Orchestrator
# =============================================================================

class AgentOrchestrator:
    """
    Executes agents in order and passes data between them.

    Implements Engine Admission Control Spec with:
    - Weight capping (Spec D2: max 0.25 per engine)
    - ConsensusReport generation (Spec Section H)
    - Null calibration integration (Spec Section B)
    - Stability scoring integration (Spec Section C)

    Pipeline:
        Agent1 (Data Quality) -> Agent2 (Geometry) -> Agent3 (Routing)
        -> [engines] -> Agent4 (Stability) -> Agent5 (Blind Spot)
        -> ConsensusReport

    Usage:
        orchestrator = AgentOrchestrator()
        orchestrator.register_agents()
        result = orchestrator.run_pipeline(raw_data, cleaned_data)
        consensus = result.consensus  # ConsensusReport per Spec H
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize orchestrator.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.registry = DiagnosticRegistry()
        self._agents: Dict[str, BaseAgent] = {}
        self._enabled_agents: List[str] = []

        # Spec parameters
        self.w_cap = self.config.get('weight_cap', 0.25)  # Spec D2
        self.stability_threshold = self.config.get('stability_threshold', 0.6)  # Spec C3
        self.null_p_threshold = self.config.get('null_p_threshold', 0.05)  # Spec B3
        self.null_effect_threshold = self.config.get('null_effect_size_threshold', 2.0)  # Spec B3

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent instance to register
        """
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def set_pipeline(self, agent_names: List[str]) -> None:
        """
        Set the execution order for agents.

        Args:
            agent_names: Ordered list of agent names
        """
        for name in agent_names:
            if name not in self._agents:
                raise ValueError(f"Unknown agent: {name}")
        self._enabled_agents = agent_names
        logger.info(f"Pipeline set: {' -> '.join(agent_names)}")

    def run_pipeline(
        self,
        raw_data: Any,
        cleaned_data: Any,
        run_engines_fn: Optional[Callable] = None,
        window_size: Optional[int] = None,
        sample_count: Optional[int] = None,
        window_id: Optional[str] = None,
        domain: str = "finance",
    ) -> PipelineResult:
        """
        Execute the full agent pipeline.

        Returns ConsensusReport per Spec Section H.

        Args:
            raw_data: Original time series (numpy array or dict of arrays)
            cleaned_data: Preprocessed time series
            run_engines_fn: Callable that takes (data, routing_result) and returns
                           engine_results dict. If None, skip engine execution.
            window_size: Optional window size override.
            sample_count: Optional sample count override.
            window_id: Optional window identifier for multi-window analysis.
            domain: Data domain ("finance", "climate", "epidemiology").

        Returns:
            PipelineResult with ConsensusReport per Spec Section H
        """
        import numpy as np

        # Set up window context
        wid = window_id or f"w_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.registry.clear()
        self.registry.set_current_window(wid)
        warnings: List[str] = []

        # Determine window/sample size from data
        if isinstance(cleaned_data, np.ndarray):
            data_len = len(cleaned_data)
        elif isinstance(cleaned_data, dict):
            data_len = len(next(iter(cleaned_data.values())))
        else:
            data_len = 100

        window_size = window_size or data_len
        sample_count = sample_count or data_len

        # Store input metadata
        self.registry.store("input", "raw_data", raw_data, wid)
        self.registry.store("input", "cleaned_data", cleaned_data, wid)
        self.registry.store("input", "window_size", window_size, wid)
        self.registry.store("input", "sample_count", sample_count, wid)
        self.registry.store("input", "domain", domain, wid)

        # =====================================================================
        # Agent 1: Data Quality (Spec Section A)
        # =====================================================================
        quality_result = None
        if "data_quality" in self._agents:
            logger.info("Running Agent 1: Data Quality")
            try:
                quality_result = self._agents["data_quality"].run({
                    "raw_data": raw_data,
                    "cleaned_data": cleaned_data,
                })
                if quality_result.safety_flag == "unsafe":
                    warnings.append(
                        f"WARNING: Data quality flagged as UNSAFE: {quality_result.reasons}"
                    )
                    logger.warning(f"Data quality unsafe: {quality_result.reasons}")
            except Exception as e:
                logger.error(f"Data quality agent failed: {e}")
                warnings.append(f"Data quality agent failed: {e}")

        # =====================================================================
        # Agent 2: Geometry Detection
        # =====================================================================
        geometry_result = None
        if "geometry_diagnostic" in self._agents:
            logger.info("Running Agent 2: Geometry Detection")
            try:
                geometry_result = self._agents["geometry_diagnostic"].run({
                    "cleaned_data": cleaned_data,
                })
            except Exception as e:
                logger.error(f"Geometry agent failed: {e}")
                warnings.append(f"Geometry agent failed: {e}")

        # =====================================================================
        # Agent 3: Engine Routing
        # =====================================================================
        routing_result = None
        if "engine_routing" in self._agents:
            logger.info("Running Agent 3: Engine Routing")
            try:
                routing_result = self._agents["engine_routing"].run({
                    "geometry_diagnostic_result": geometry_result,
                    "window_size": window_size,
                    "sample_count": sample_count,
                    "domain": domain,
                })
            except Exception as e:
                logger.error(f"Routing agent failed: {e}")
                warnings.append(f"Routing agent failed: {e}")

        # =====================================================================
        # Run Engines (external)
        # =====================================================================
        engine_results = {}
        if run_engines_fn is not None:
            logger.info("Running engines...")
            try:
                engine_results = run_engines_fn(cleaned_data, routing_result)
            except Exception as e:
                logger.error(f"Engine execution failed: {e}")
                warnings.append(f"Engine execution failed: {e}")

        # =====================================================================
        # Agent 4: Stability Audit (Spec Sections B, C, D)
        # =====================================================================
        stability_result = None
        if "engine_stability" in self._agents:
            logger.info("Running Agent 4: Stability Audit")
            try:
                stability_result = self._agents["engine_stability"].run({
                    "engine_results": engine_results,
                    "routing_result": routing_result,
                    "original_data": cleaned_data,
                    "window_id": wid,
                    "weight_cap": self.w_cap,
                })
            except Exception as e:
                logger.error(f"Stability agent failed: {e}")
                warnings.append(f"Stability agent failed: {e}")

        # =====================================================================
        # Agent 5: Blind Spot Detection
        # =====================================================================
        blindspot_result = None
        if "blind_spot_detection" in self._agents:
            logger.info("Running Agent 5: Blind Spot Detection")
            try:
                blindspot_result = self._agents["blind_spot_detection"].run({
                    "geometry_diagnostic_result": geometry_result,
                    "engine_results": engine_results,
                    "engine_stability_result": stability_result,
                    "cleaned_data": cleaned_data,
                })
            except Exception as e:
                logger.error(f"Blind spot agent failed: {e}")
                warnings.append(f"Blind spot agent failed: {e}")

        # =====================================================================
        # Generate ConsensusReport (Spec Section H)
        # =====================================================================
        consensus = self.registry.get_engine_report(wid, self.w_cap)

        # Add any warnings to consensus
        consensus.warnings.extend(warnings)

        # =====================================================================
        # Build Result
        # =====================================================================
        return PipelineResult(
            quality=quality_result,
            geometry=geometry_result,
            routing=routing_result,
            engine_results=engine_results,
            stability=stability_result,
            blindspots=blindspot_result,
            consensus=consensus,
            audit_trail=self.registry.export_audit_trail(),
            warnings=warnings,
        )

    def run_pipeline_simple(
        self,
        raw_data: Any,
        cleaned_data: Any,
    ) -> Dict[str, Any]:
        """
        Execute a simple pipeline using registered agents in order.

        This is a simpler version for backward compatibility.
        """
        self.registry.clear()
        results: Dict[str, Any] = {}

        self.registry.set("input", "raw_data", raw_data)
        self.registry.set("input", "cleaned_data", cleaned_data)

        current_inputs = {
            "raw_data": raw_data,
            "cleaned_data": cleaned_data,
        }

        for agent_name in self._enabled_agents:
            agent = self._agents[agent_name]
            logger.info(f"Running agent: {agent_name}")

            try:
                result = agent.run(current_inputs)
                results[agent_name] = result
                current_inputs[f"{agent_name}_result"] = result

            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                agent.log(
                    decision="agent_failure",
                    reason=str(e),
                    metrics={"exception_type": type(e).__name__},
                )
                results[agent_name] = None

        return {
            "results": results,
            "audit_trail": self.registry.export_audit_trail(),
            "metrics": self.registry.get_all(),
        }

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a registered agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())


# =============================================================================
# Config Loader
# =============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent / "agent_config.yaml"


def load_agent_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load agent configuration from YAML.

    DEPRECATED: Use prism.config module instead.

    Args:
        config_path: Path to config file. Uses default if not provided.

    Returns:
        Configuration dictionary
    """
    # Try to load from new config system first
    try:
        from prism.config import get_stability_config, get_data_quality_config
        stability_cfg = get_stability_config()
        quality_cfg = get_data_quality_config()

        # Merge configs into expected format
        config = get_default_config()
        config.update({
            "weight_cap": stability_cfg.get("w_cap", 0.25),
            "stability_threshold": stability_cfg.get("stability_threshold", 0.6),
            "n_null_surrogates": stability_cfg.get("n_null", 100),
            "n_stability_resamples": stability_cfg.get("n_subsamples", 50),
            "missing_rate_multivariate": quality_cfg.get("missing_rate_multivariate", 0.05),
            "missing_rate_univariate": quality_cfg.get("missing_rate_univariate", 0.10),
        })
        return config
    except (FileNotFoundError, ImportError):
        pass

    # Fallback to legacy config file
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning(f"Config not found: {path}, using defaults")
        return get_default_config()

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    logger.info(f"Loaded agent config from {path}")
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values per spec.

    Returns:
        Default configuration dictionary
    """
    return {
        "weight_cap": 0.25,                    # Spec D2
        "stability_threshold": 0.6,            # Spec C3
        "null_p_threshold": 0.05,              # Spec B3
        "null_effect_size_threshold": 2.0,     # Spec B3
        "missing_rate_multivariate": 0.05,     # Spec A3
        "missing_rate_univariate": 0.10,       # Spec A3
        "min_contributing_engines": 2,         # Consensus validity
        "n_null_surrogates": 100,              # Null calibration
        "n_stability_resamples": 50,           # Stability scoring
    }
