"""
PRISM Math Suitability Agent (Window-Aware)

Eligibility is PER-INDICATOR-PER-WINDOW.

An indicator may be:
- Eligible at 3-year window (sufficient structure)
- Ineligible at 1-year window (too noisy)
- Conditional at 5-year window (over-smoothed)

Flow:
    1. Temporal Geometry Scan -> WindowGeometryResult per (indicator, window)
    2. Math Suitability Agent -> WindowEligibility per (indicator, window)
    3. Engine Runner -> queries eligible (indicator, window, engine) triples

Usage:
    from prism.agents.agent_math_suitability import (
        WindowSuitabilityAgent,
        WindowGeometryResult,
        WindowEligibility,
    )
    
    agent = WindowSuitabilityAgent()
    eligibility = agent.evaluate(window_geometry_result)

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json
import logging

from prism.config import get_windows_config

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class EligibilityStatus(Enum):
    ELIGIBLE = "eligible"
    CONDITIONAL = "conditional"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WindowGeometryResult:
    """Geometry result for a specific (indicator, window) pair."""
    indicator_id: str
    window_years: float
    window_days: int

    dominant_geometry: str
    geometry_pct: float

    avg_confidence: float
    avg_disagreement: float
    confidence_std: float

    n_observations: int
    n_transitions: int
    stability: float

    quality_score: float
    is_optimal: bool = False


@dataclass
class WindowEligibility:
    """Eligibility decision for a specific (indicator, window) pair."""
    indicator_id: str
    window_years: float

    geometry: str
    confidence: float
    disagreement: float
    stability: float

    status: EligibilityStatus

    allowed_engines: List[str]
    prohibited_engines: List[str]
    conditional_engines: List[Dict]

    rationale: List[str]
    policy_version: str

    created_at: datetime = field(default_factory=datetime.now)

    def to_db_record(self, run_id: str) -> Dict[str, Any]:
        """Convert to dictionary for database persistence."""
        return {
            "run_id": run_id,
            "indicator_id": self.indicator_id,
            "window_years": self.window_years,
            "geometry": self.geometry,
            "confidence": self.confidence,
            "disagreement": self.disagreement,
            "stability": self.stability,
            "status": self.status.value,
            "allowed_engines": json.dumps(self.allowed_engines),
            "prohibited_engines": json.dumps(self.prohibited_engines),
            "conditional_engines": json.dumps(self.conditional_engines),
            "rationale": json.dumps(self.rationale),
            "policy_version": self.policy_version,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# POLICY CONFIGURATION
# =============================================================================

@dataclass
class WindowSuitabilityPolicy:
    """
    Centralized policy for math suitability decisions.

    All thresholds live here - nowhere else.

    Parameters can be loaded from config/windows.yaml via from_config().
    Default values are preserved for backward compatibility.
    """
    version: str = "1.0.0"

    # Base thresholds
    min_quality_score: float = 0.35
    min_confidence: float = 0.30
    max_disagreement: float = 0.65
    min_stability: float = 0.50
    min_observations: int = 10

    # Pure noise handling
    pure_noise_blocks_inference: bool = True
    pure_noise_threshold: float = 0.70

    # Window-specific adjustments
    short_window_threshold: float = 1.5
    long_window_threshold: float = 5.0

    # Short windows: more lenient on confidence, stricter on stability
    short_window_confidence_adj: float = -0.05
    short_window_stability_min: float = 0.60

    # Long windows: check for over-smoothing
    long_window_max_geometry_pct: float = 0.95

    # Engine categories by geometry
    universal_engines: List[str] = field(default_factory=lambda: [
        "pca", "correlation", "entropy", "descriptive_stats"
    ])

    latent_flow_engines: List[str] = field(default_factory=lambda: [
        "pca", "correlation", "entropy", "wavelets", "trend", "drift"
    ])

    reflexive_engines: List[str] = field(default_factory=lambda: [
        "pca", "correlation", "entropy", "hmm", "regime_detection",
        "volatility_clustering", "copula", "garch"
    ])

    oscillator_engines: List[str] = field(default_factory=lambda: [
        "pca", "correlation", "entropy", "wavelets", "spectral",
        "phase_space", "dmd"
    ])

    noise_allowed_engines: List[str] = field(default_factory=lambda: [
        "descriptive_stats", "null_check", "data_quality"
    ])

    @classmethod
    def from_config(cls) -> "WindowSuitabilityPolicy":
        """
        Create policy from config/windows.yaml.

        Falls back to defaults if config not found.
        """
        try:
            cfg = get_windows_config()
        except FileNotFoundError:
            logger.warning("windows.yaml not found, using defaults")
            return cls()

        return cls(
            version=cfg.get("version", "1.0.0"),
            min_quality_score=cfg.get("min_quality_score", 0.35),
            min_confidence=cfg.get("min_confidence", 0.30),
            max_disagreement=cfg.get("max_disagreement", 0.65),
            min_stability=cfg.get("min_stability", 0.50),
            min_observations=cfg.get("min_observations", 10),
            pure_noise_blocks_inference=cfg.get("pure_noise_blocks_inference", True),
            pure_noise_threshold=cfg.get("pure_noise_threshold", 0.70),
            short_window_threshold=cfg.get("short_window_threshold", 1.5),
            long_window_threshold=cfg.get("long_window_threshold", 5.0),
            short_window_confidence_adj=cfg.get("short_window_confidence_adj", -0.05),
            short_window_stability_min=cfg.get("short_window_stability_min", 0.60),
            long_window_max_geometry_pct=cfg.get("long_window_max_geometry_pct", 0.95),
            universal_engines=cfg.get("universal_engines", [
                "pca", "correlation", "entropy", "descriptive_stats"
            ]),
            latent_flow_engines=cfg.get("latent_flow_engines", [
                "pca", "correlation", "entropy", "wavelets", "trend", "drift"
            ]),
            reflexive_engines=cfg.get("reflexive_engines", [
                "pca", "correlation", "entropy", "hmm", "regime_detection",
                "volatility_clustering", "copula", "garch"
            ]),
            oscillator_engines=cfg.get("oscillator_engines", [
                "pca", "correlation", "entropy", "wavelets", "spectral",
                "phase_space", "dmd"
            ]),
            noise_allowed_engines=cfg.get("noise_allowed_engines", [
                "descriptive_stats", "null_check", "data_quality"
            ]),
        )


# =============================================================================
# WINDOW SUITABILITY AGENT
# =============================================================================

class WindowSuitabilityAgent:
    """
    Evaluates math suitability for each (indicator, window) pair.
    
    This is the gatekeeper - no engine runs without eligibility.
    
    Usage:
        agent = WindowSuitabilityAgent()
        
        # Single evaluation
        eligibility = agent.evaluate(window_geometry_result)
        
        # Batch by indicator
        by_window = agent.evaluate_indicator(list_of_results)
        
        # Batch all
        all_eligibility = agent.evaluate_all(results_dict)
    """
    
    def __init__(self, policy: Optional[WindowSuitabilityPolicy] = None):
        self.policy = policy or WindowSuitabilityPolicy()
        logger.info(f"WindowSuitabilityAgent initialized (policy v{self.policy.version})")

    def evaluate(self, result: WindowGeometryResult) -> WindowEligibility:
        """
        Evaluate eligibility for a single (indicator, window) pair.
        
        Args:
            result: WindowGeometryResult with geometry analysis
            
        Returns:
            WindowEligibility with status, allowed engines, rationale
        """
        rationale: List[str] = []
        allowed = set()
        prohibited = set()
        conditional: List[Dict] = []
        status = EligibilityStatus.ELIGIBLE

        p = self.policy

        # Adjust thresholds by window
        conf_threshold = p.min_confidence
        stab_threshold = p.min_stability

        if result.window_years < p.short_window_threshold:
            conf_threshold += p.short_window_confidence_adj
            stab_threshold = p.short_window_stability_min
            rationale.append(f"short window ({result.window_years}y): adjusted thresholds")

        # -----------------------------------------------------------------
        # CHECK 1: Minimum observations
        # -----------------------------------------------------------------
        if result.n_observations < p.min_observations:
            return WindowEligibility(
                indicator_id=result.indicator_id,
                window_years=result.window_years,
                geometry=result.dominant_geometry,
                confidence=result.avg_confidence,
                disagreement=result.avg_disagreement,
                stability=result.stability,
                status=EligibilityStatus.INELIGIBLE,
                allowed_engines=list(p.noise_allowed_engines),
                prohibited_engines=[],
                conditional_engines=[],
                rationale=[f"insufficient observations ({result.n_observations} < {p.min_observations})"],
                policy_version=p.version,
            )

        # -----------------------------------------------------------------
        # CHECK 2: Pure noise blocks inference
        # -----------------------------------------------------------------
        if result.dominant_geometry == "pure_noise" and p.pure_noise_blocks_inference:
            if result.geometry_pct > p.pure_noise_threshold:
                return WindowEligibility(
                    indicator_id=result.indicator_id,
                    window_years=result.window_years,
                    geometry=result.dominant_geometry,
                    confidence=result.avg_confidence,
                    disagreement=result.avg_disagreement,
                    stability=result.stability,
                    status=EligibilityStatus.INELIGIBLE,
                    allowed_engines=list(p.noise_allowed_engines),
                    prohibited_engines=[],
                    conditional_engines=[],
                    rationale=[f"pure_noise dominant ({result.geometry_pct*100:.0f}%) at {result.window_years}y window"],
                    policy_version=p.version,
                )

        # -----------------------------------------------------------------
        # CHECK 3: Quality score
        # -----------------------------------------------------------------
        if result.quality_score < p.min_quality_score:
            status = EligibilityStatus.CONDITIONAL
            rationale.append(f"low quality score ({result.quality_score:.2f} < {p.min_quality_score})")

        # -----------------------------------------------------------------
        # CHECK 4: Confidence
        # -----------------------------------------------------------------
        if result.avg_confidence < conf_threshold:
            status = EligibilityStatus.CONDITIONAL
            rationale.append(f"low confidence ({result.avg_confidence:.2f} < {conf_threshold:.2f})")

        # -----------------------------------------------------------------
        # CHECK 5: Disagreement
        # -----------------------------------------------------------------
        if result.avg_disagreement > p.max_disagreement:
            status = EligibilityStatus.CONDITIONAL
            rationale.append(f"high disagreement ({result.avg_disagreement:.2f} > {p.max_disagreement})")

        # -----------------------------------------------------------------
        # CHECK 6: Stability
        # -----------------------------------------------------------------
        if result.stability < stab_threshold:
            status = EligibilityStatus.CONDITIONAL
            rationale.append(f"low stability ({result.stability:.2f} < {stab_threshold})")

        # -----------------------------------------------------------------
        # CHECK 7: Over-smoothing (long windows)
        # -----------------------------------------------------------------
        if result.window_years >= p.long_window_threshold:
            if result.geometry_pct > p.long_window_max_geometry_pct:
                status = EligibilityStatus.CONDITIONAL
                rationale.append(f"possible over-smoothing: {result.geometry_pct*100:.0f}% single geometry at {result.window_years}y")

        # -----------------------------------------------------------------
        # DETERMINE ALLOWED ENGINES BY GEOMETRY
        # -----------------------------------------------------------------
        allowed = set(p.universal_engines)

        if result.dominant_geometry == "latent_flow":
            allowed.update(p.latent_flow_engines)
            rationale.append("geometry=latent_flow: trend/drift engines enabled")
        elif result.dominant_geometry == "reflexive_stochastic":
            allowed.update(p.reflexive_engines)
            rationale.append("geometry=reflexive_stochastic: regime/feedback engines enabled")
        elif result.dominant_geometry == "coupled_oscillator":
            allowed.update(p.oscillator_engines)
            rationale.append("geometry=coupled_oscillator: spectral engines enabled")
        elif result.dominant_geometry == "pure_noise":
            allowed = set(p.universal_engines)
            rationale.append("geometry=pure_noise: limited to universal engines")

        if status == EligibilityStatus.ELIGIBLE:
            rationale.append(f"passed all checks at {result.window_years}y window")

        return WindowEligibility(
            indicator_id=result.indicator_id,
            window_years=result.window_years,
            geometry=result.dominant_geometry,
            confidence=result.avg_confidence,
            disagreement=result.avg_disagreement,
            stability=result.stability,
            status=status,
            allowed_engines=sorted(allowed),
            prohibited_engines=sorted(prohibited),
            conditional_engines=conditional,
            rationale=rationale,
            policy_version=p.version,
        )

    def evaluate_indicator(self, results: List[WindowGeometryResult]) -> Dict[float, WindowEligibility]:
        """Evaluate all windows for a single indicator."""
        return {r.window_years: self.evaluate(r) for r in results}

    def evaluate_all(
        self, 
        all_results: Dict[str, List[WindowGeometryResult]]
    ) -> Dict[str, Dict[float, WindowEligibility]]:
        """Evaluate all indicators across all windows."""
        return {ind: self.evaluate_indicator(results) for ind, results in all_results.items()}
