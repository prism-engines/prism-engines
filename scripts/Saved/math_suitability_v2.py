"""
PRISM Math Suitability Agent v2 (Window-Aware)

Eligibility is PER-INDICATOR-PER-WINDOW.

An indicator may be:
- Eligible at 3-year window (sufficient structure)
- Ineligible at 1-year window (too noisy)
- Conditional at 5-year window (over-smoothed)

The agent evaluates each (indicator, window) pair independently.

Flow:
    1. Temporal Geometry Scan → WindowGeometryResult per (indicator, window)
    2. Math Suitability Agent → EligibilityDecision per (indicator, window)
    3. Engine Runner → queries eligible (indicator, window, engine) triples

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ELIGIBILITY STATUS
# =============================================================================

class EligibilityStatus(Enum):
    ELIGIBLE = "eligible"           # Full engine access at this window
    CONDITIONAL = "conditional"     # With constraints
    INELIGIBLE = "ineligible"       # No inference engines at this window
    UNKNOWN = "unknown"             # Cannot determine


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WindowGeometryResult:
    """
    Geometry result for a specific (indicator, window) pair.
    Output of Temporal Geometry Scan.
    """
    indicator_id: str
    window_years: float
    window_days: int
    
    # Dominant geometry at this window
    dominant_geometry: str
    geometry_pct: float
    
    # Quality metrics
    avg_confidence: float
    avg_disagreement: float
    confidence_std: float
    
    # Stability
    n_observations: int
    n_transitions: int
    stability: float
    
    # Combined score
    quality_score: float
    
    # Is this the optimal window?
    is_optimal: bool = False


@dataclass
class WindowEligibility:
    """
    Eligibility decision for a specific (indicator, window) pair.
    """
    indicator_id: str
    window_years: float
    
    # Geometry at this window
    geometry: str
    confidence: float
    disagreement: float
    stability: float
    
    # Decision
    status: EligibilityStatus
    
    # Engine routing for THIS window
    allowed_engines: List[str]
    prohibited_engines: List[str]
    conditional_engines: List[Dict]
    
    # Trace
    rationale: List[str]
    policy_version: str
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_db_record(self, run_id: str) -> Dict[str, Any]:
        """Convert to database record."""
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
# WINDOW-AWARE SUITABILITY POLICY
# =============================================================================

@dataclass
class WindowSuitabilityPolicy:
    """
    Policy for window-aware eligibility.
    
    Thresholds may vary by window scale:
    - Short windows (< 2y): Higher noise tolerance, stricter stability
    - Long windows (> 5y): Lower noise tolerance, may over-smooth
    """
    version: str = "2.0.0"
    
    # Minimum quality score to be eligible
    min_quality_score: float = 0.35
    
    # Minimum confidence to authorize
    min_confidence: float = 0.30
    
    # Maximum disagreement to authorize
    max_disagreement: float = 0.65
    
    # Minimum stability (1.0 = no transitions)
    min_stability: float = 0.50
    
    # Minimum observations required
    min_observations: int = 10
    
    # Pure noise blocks inference
    pure_noise_blocks_inference: bool = True
    
    # Window-specific adjustments
    short_window_threshold: float = 1.5   # years
    long_window_threshold: float = 5.0    # years
    
    # Short window: more lenient on noise, stricter on stability
    short_window_confidence_adj: float = -0.05
    short_window_stability_min: float = 0.60
    
    # Long window: may over-smooth, check for lost structure
    long_window_max_geometry_pct: float = 0.95  # Too dominant = over-smoothed
    
    # Engine categories
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


# =============================================================================
# WINDOW-AWARE SUITABILITY AGENT
# =============================================================================

class WindowSuitabilityAgent:
    """
    Evaluates eligibility for each (indicator, window) pair.
    
    Usage:
        agent = WindowSuitabilityAgent()
        
        # Evaluate single (indicator, window)
        eligibility = agent.evaluate(window_geometry_result)
        
        # Evaluate all windows for an indicator
        eligibilities = agent.evaluate_indicator(list_of_window_results)
    """
    
    def __init__(self, policy: Optional[WindowSuitabilityPolicy] = None):
        self.policy = policy or WindowSuitabilityPolicy()
    
    def evaluate(self, result: WindowGeometryResult) -> WindowEligibility:
        """
        Evaluate eligibility for a single (indicator, window) pair.
        """
        rationale = []
        allowed = set()
        prohibited = set()
        conditional = []
        status = EligibilityStatus.ELIGIBLE
        
        p = self.policy
        
        # Adjust thresholds by window scale
        conf_threshold = p.min_confidence
        stab_threshold = p.min_stability
        
        if result.window_years < p.short_window_threshold:
            conf_threshold += p.short_window_confidence_adj
            stab_threshold = p.short_window_stability_min
            rationale.append(f"short window ({result.window_years}y): adjusted thresholds")
        
        # =================================================================
        # CHECK 1: Minimum observations
        # =================================================================
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
        
        # =================================================================
        # CHECK 2: Pure noise blocks inference
        # =================================================================
        if result.dominant_geometry == "pure_noise" and p.pure_noise_blocks_inference:
            if result.geometry_pct > 0.7:  # Strongly noise
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
        
        # =================================================================
        # CHECK 3: Quality score
        # =================================================================
        if result.quality_score < p.min_quality_score:
            status = EligibilityStatus.CONDITIONAL
            rationale.append(f"low quality score ({result.quality_score:.2f} < {p.min_quality_score})")
        
        # =================================================================
        # CHECK 4: Confidence
        # =================================================================
        if result.avg_confidence < conf_threshold:
            status = EligibilityStatus.CONDITIONAL
            rationale.append(f"low confidence ({result.avg_confidence:.2f} < {conf_threshold:.2f})")
        
        # =================================================================
        # CHECK 5: Disagreement
        # =================================================================
        if result.avg_disagreement > p.max_disagreement:
            status = EligibilityStatus.CONDITIONAL
            rationale.append(f"high disagreement ({result.avg_disagreement:.2f} > {p.max_disagreement})")
        
        # =================================================================
        # CHECK 6: Stability
        # =================================================================
        if result.stability < stab_threshold:
            status = EligibilityStatus.CONDITIONAL
            rationale.append(f"low stability ({result.stability:.2f} < {stab_threshold})")
        
        # =================================================================
        # CHECK 7: Over-smoothing (long windows)
        # =================================================================
        if result.window_years >= p.long_window_threshold:
            if result.geometry_pct > p.long_window_max_geometry_pct:
                status = EligibilityStatus.CONDITIONAL
                rationale.append(f"possible over-smoothing: {result.geometry_pct*100:.0f}% single geometry at {result.window_years}y")
        
        # =================================================================
        # DETERMINE ALLOWED ENGINES BY GEOMETRY
        # =================================================================
        allowed = set(p.universal_engines)
        
        if result.dominant_geometry == "latent_flow":
            allowed.update(p.latent_flow_engines)
            rationale.append(f"geometry=latent_flow: trend/drift engines enabled")
            
        elif result.dominant_geometry == "reflexive_stochastic":
            allowed.update(p.reflexive_engines)
            rationale.append(f"geometry=reflexive_stochastic: regime/feedback engines enabled")
            
        elif result.dominant_geometry == "coupled_oscillator":
            allowed.update(p.oscillator_engines)
            rationale.append(f"geometry=coupled_oscillator: spectral engines enabled")
            
        elif result.dominant_geometry == "pure_noise":
            # Already handled above, but partial noise still gets limited engines
            allowed = set(p.universal_engines)
            rationale.append(f"geometry=pure_noise: limited to universal engines")
        
        # If still eligible after all checks
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
    
    def evaluate_indicator(
        self, 
        results: List[WindowGeometryResult]
    ) -> Dict[float, WindowEligibility]:
        """
        Evaluate all windows for a single indicator.
        
        Returns:
            Dict mapping window_years -> WindowEligibility
        """
        return {r.window_years: self.evaluate(r) for r in results}
    
    def evaluate_all(
        self,
        all_results: Dict[str, List[WindowGeometryResult]]
    ) -> Dict[str, Dict[float, WindowEligibility]]:
        """
        Evaluate all (indicator, window) pairs.
        
        Args:
            all_results: Dict mapping indicator_id -> list of WindowGeometryResult
            
        Returns:
            Nested dict: indicator_id -> window_years -> WindowEligibility
        """
        return {
            ind: self.evaluate_indicator(results)
            for ind, results in all_results.items()
        }
    
    def persist(
        self,
        run_id: str,
        eligibilities: Dict[str, Dict[float, WindowEligibility]],
        conn=None
    ) -> None:
        """Persist all eligibility decisions to database."""
        if conn is None:
            from prism.db.connection import get_connection
            conn = get_connection()
            should_close = True
        else:
            should_close = False
        
        try:
            for ind_id, windows in eligibilities.items():
                for window_years, elig in windows.items():
                    record = elig.to_db_record(run_id)
                    conn.execute("""
                        INSERT OR REPLACE INTO meta.engine_eligibility
                        (run_id, indicator_id, window_years, geometry, confidence,
                         disagreement, stability, status, allowed_engines,
                         prohibited_engines, conditional_engines, rationale,
                         policy_version, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        record["run_id"],
                        record["indicator_id"],
                        record["window_years"],
                        record["geometry"],
                        record["confidence"],
                        record["disagreement"],
                        record["stability"],
                        record["status"],
                        record["allowed_engines"],
                        record["prohibited_engines"],
                        record["conditional_engines"],
                        record["rationale"],
                        record["policy_version"],
                        record["created_at"],
                    ])
            
            logger.info(f"Persisted eligibility for {len(eligibilities)} indicators")
        finally:
            if should_close:
                conn.close()


# =============================================================================
# SUMMARY HELPER
# =============================================================================

def summarize_eligibility(
    eligibilities: Dict[str, Dict[float, WindowEligibility]]
) -> None:
    """Print summary of eligibility decisions."""
    
    total = sum(len(windows) for windows in eligibilities.values())
    by_status = {}
    by_window = {}
    
    for ind_id, windows in eligibilities.items():
        for window_years, elig in windows.items():
            status = elig.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            w_key = f"{window_years}y"
            if w_key not in by_window:
                by_window[w_key] = {"eligible": 0, "conditional": 0, "ineligible": 0, "unknown": 0}
            by_window[w_key][status] += 1
    
    print(f"\nTotal (indicator, window) pairs: {total}")
    print(f"\nBy Status:")
    for status, count in sorted(by_status.items()):
        pct = 100 * count / total
        print(f"  {status:15} {count:4} ({pct:5.1f}%)")
    
    print(f"\nBy Window:")
    for w in sorted(by_window.keys(), key=lambda x: float(x.rstrip('y'))):
        counts = by_window[w]
        elig = counts.get("eligible", 0)
        cond = counts.get("conditional", 0)
        inelig = counts.get("ineligible", 0)
        total_w = elig + cond + inelig + counts.get("unknown", 0)
        print(f"  {w:5} eligible={elig:3}, conditional={cond:3}, ineligible={inelig:3}")
