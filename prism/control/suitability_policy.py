"""
PRISM Math Suitability Policy

Centralized control-plane configuration for engine eligibility decisions.

THIS IS THE ONLY PLACE WHERE SUITABILITY THRESHOLDS LIVE.
No scripts, no agents, no runners may embed routing thresholds.

The policy converts geometry measurements into routing constraints.
It does NOT interpret geometry - only converts to actionable decisions.

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum
import json


class EligibilityStatus(Enum):
    """Possible outcomes of suitability assessment."""
    ELIGIBLE = "eligible"           # Engines may run freely
    CONDITIONAL = "conditional"     # Engines may run if conditions met
    INELIGIBLE = "ineligible"       # No inference engines permitted
    UNKNOWN = "unknown"             # Cannot determine - need more data


@dataclass
class SuitabilityPolicy:
    """
    Central policy configuration for engine eligibility.
    
    All threshold logic lives HERE and ONLY here.
    
    The policy is domain-agnostic. It operates only on:
    - Geometry class per view
    - Confidence per view
    - Disagreement across views
    - Hybrid flags
    - Pure noise outcomes
    """
    
    # Version for reproducibility
    version: str = "1.0.0"
    
    # ==========================================================================
    # CONFIDENCE THRESHOLDS
    # ==========================================================================
    
    # Minimum confidence to authorize inference engines
    min_confidence_to_authorize: float = 0.40
    
    # Minimum confidence for "high confidence" authorization
    high_confidence_threshold: float = 0.70
    
    # ==========================================================================
    # DISAGREEMENT THRESHOLDS
    # ==========================================================================
    
    # Maximum disagreement to allow inference
    max_disagreement_to_authorize: float = 0.60
    
    # Disagreement above this triggers "unknown" status
    unknown_disagreement_threshold: float = 0.70
    
    # ==========================================================================
    # PURE NOISE HANDLING
    # ==========================================================================
    
    # If consensus is pure_noise, block all inference engines
    pure_noise_blocks_inference: bool = True
    
    # Engines allowed even when pure_noise (descriptive only)
    noise_allowed_engines: List[str] = field(default_factory=lambda: [
        "null_check",
        "descriptive_stats",
        "data_quality",
    ])
    
    # ==========================================================================
    # COHORT-LEVEL GUARDRAILS
    # ==========================================================================
    
    # Minimum indicators required for inference
    min_indicators_for_inference: int = 5
    
    # If cohort too small, degrade to these engines only
    degraded_mode_engines: List[str] = field(default_factory=lambda: [
        "descriptive_stats",
        "correlation",
        "pca",
    ])
    
    # ==========================================================================
    # ENGINE CATEGORIES BY GEOMETRY
    # ==========================================================================
    
    # Engines appropriate for latent_flow geometry
    latent_flow_engines: List[str] = field(default_factory=lambda: [
        "pca",
        "correlation", 
        "entropy",
        "wavelets",
        "trend",
        "drift",
        "saturation",
    ])
    
    # Engines appropriate for reflexive_stochastic geometry
    reflexive_engines: List[str] = field(default_factory=lambda: [
        "pca",
        "correlation",
        "entropy",
        "hmm",
        "regime_detection",
        "volatility_clustering",
        "copula",
        "tail_dependence",
    ])
    
    # Engines appropriate for coupled_oscillator geometry
    oscillator_engines: List[str] = field(default_factory=lambda: [
        "pca",
        "correlation",
        "entropy",
        "wavelets",
        "spectral",
        "phase_space",
        "coherence",
        "dmd",
    ])
    
    # Universal engines (always allowed unless pure_noise)
    universal_engines: List[str] = field(default_factory=lambda: [
        "pca",
        "correlation",
        "entropy",
    ])
    
    # ==========================================================================
    # ENGINE PROHIBITIONS BY GEOMETRY
    # ==========================================================================
    
    # Engines to prohibit for latent_flow (feedback-dependent engines)
    latent_flow_prohibited: List[str] = field(default_factory=lambda: [
        "regime_detection",
        "volatility_clustering",
    ])
    
    # Engines to prohibit for reflexive (trend-dependent engines)  
    reflexive_prohibited: List[str] = field(default_factory=lambda: [
        "saturation",
        "drift",
    ])
    
    # Engines to prohibit for oscillator (chaos-sensitive engines)
    oscillator_prohibited: List[str] = field(default_factory=lambda: [
        "lyapunov",
        "rqa",
    ])
    
    # ==========================================================================
    # CONDITIONAL ENGINE RULES
    # ==========================================================================
    
    # Engines that require specific view evidence
    conditional_rules: Dict[str, Dict] = field(default_factory=lambda: {
        "hmm": {
            "requires": "volatility_reflexive_or_level_reflexive",
            "min_confidence": 0.50,
            "description": "HMM requires reflexive evidence in volatility or level view",
        },
        "copula": {
            "requires": "reflexive_consensus_or_volatility_reflexive",
            "min_confidence": 0.60,
            "description": "Copula requires strong reflexive evidence for tail modeling",
        },
        "wavelets": {
            "requires": "oscillator_in_any_view_or_high_confidence_level",
            "min_confidence": 0.50,
            "description": "Wavelets need oscillator evidence or very confident level geometry",
        },
        "granger": {
            "requires": "not_pure_noise_and_min_confidence",
            "min_confidence": 0.50,
            "description": "Granger causality needs sufficient structure",
        },
    })
    
    def to_dict(self) -> Dict:
        """Serialize policy for storage."""
        return {
            "version": self.version,
            "min_confidence_to_authorize": self.min_confidence_to_authorize,
            "high_confidence_threshold": self.high_confidence_threshold,
            "max_disagreement_to_authorize": self.max_disagreement_to_authorize,
            "unknown_disagreement_threshold": self.unknown_disagreement_threshold,
            "pure_noise_blocks_inference": self.pure_noise_blocks_inference,
            "min_indicators_for_inference": self.min_indicators_for_inference,
        }
    
    def to_json(self) -> str:
        """Serialize policy to JSON."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# POLICY PRESETS
# =============================================================================

def get_permissive_policy() -> SuitabilityPolicy:
    """Permissive policy for exploratory analysis."""
    return SuitabilityPolicy(
        version="1.0.0-permissive",
        min_confidence_to_authorize=0.30,
        max_disagreement_to_authorize=0.70,
        unknown_disagreement_threshold=0.80,
        min_indicators_for_inference=3,
    )


def get_strict_policy() -> SuitabilityPolicy:
    """Strict policy for production/publication."""
    return SuitabilityPolicy(
        version="1.0.0-strict",
        min_confidence_to_authorize=0.50,
        max_disagreement_to_authorize=0.45,
        unknown_disagreement_threshold=0.55,
        min_indicators_for_inference=8,
    )


def get_default_policy() -> SuitabilityPolicy:
    """Default balanced policy."""
    return SuitabilityPolicy()


# =============================================================================
# GLOBAL POLICY INSTANCE
# =============================================================================

_current_policy: Optional[SuitabilityPolicy] = None


def set_suitability_policy(policy: SuitabilityPolicy) -> None:
    """Set the active suitability policy."""
    global _current_policy
    _current_policy = policy


def get_suitability_policy() -> SuitabilityPolicy:
    """Get the active suitability policy."""
    global _current_policy
    if _current_policy is None:
        _current_policy = get_default_policy()
    return _current_policy
