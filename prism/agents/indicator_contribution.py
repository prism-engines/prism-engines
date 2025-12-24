"""
Indicator Contribution Agent

Computes contribution scores per indicator. Does NOT make exclusion decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class DecisionHint(Enum):
    KEEP = "keep"
    REVIEW = "review"
    EXCLUDE_CANDIDATE = "exclude_candidate"


@dataclass
class IndicatorContribution:
    """Contribution metrics for a single indicator."""
    indicator_id: str

    # Input metrics (normalized 0-1, higher = better)
    geometric_influence: float = 0.0      # How much it affects geometry
    marginal_contribution: float = 0.0    # Leave-one-out delta
    stability: float = 0.0                # Consistency across windows
    noise_pressure: float = 0.0           # Inverse of noise contamination
    confidence_effect: float = 0.0        # Impact on downstream certainty

    # Computed outputs
    ics: float = 0.0                      # Indicator Contribution Score [0-1]
    failure_ratio: float = 0.0            # 1 - ics
    decision_hint: DecisionHint = DecisionHint.KEEP
    rationale: str = ""

    # Weights used (for audit trail)
    weights_used: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContributionReport:
    """Output from IndicatorContributionAgent."""
    contributions: List[IndicatorContribution]
    timestamp: str
    cohort_id: Optional[str] = None

    def to_dataframe(self):
        """Convert to pandas DataFrame for easy inspection."""
        import pandas as pd
        return pd.DataFrame([
            {
                "indicator_id": c.indicator_id,
                "ics": c.ics,
                "failure_ratio": c.failure_ratio,
                "decision_hint": c.decision_hint.value,
                "rationale": c.rationale,
                "geometric_influence": c.geometric_influence,
                "stability": c.stability,
                "noise_pressure": c.noise_pressure,
            }
            for c in self.contributions
        ])


class IndicatorContributionAgent:
    """
    Computes Indicator Contribution Scores from upstream metrics.

    RULES:
    - Does NOT exclude indicators
    - Does NOT apply thresholds
    - Only computes and reports
    """

    # Default weights (can be overridden via config)
    DEFAULT_WEIGHTS = {
        "geometric_influence": 0.30,
        "marginal_contribution": 0.20,
        "stability": 0.25,
        "noise_pressure": 0.15,
        "confidence_effect": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def compute_contribution(
        self,
        indicator_id: str,
        geometric_influence: float,
        marginal_contribution: float,
        stability: float,
        noise_pressure: float,
        confidence_effect: float,
    ) -> IndicatorContribution:
        """Compute ICS for a single indicator."""

        # Clamp inputs to [0, 1]
        inputs = {
            "geometric_influence": np.clip(geometric_influence, 0, 1),
            "marginal_contribution": np.clip(marginal_contribution, 0, 1),
            "stability": np.clip(stability, 0, 1),
            "noise_pressure": np.clip(noise_pressure, 0, 1),
            "confidence_effect": np.clip(confidence_effect, 0, 1),
        }

        # Weighted sum
        ics = sum(inputs[k] * self.weights[k] for k in self.weights)
        failure_ratio = 1 - ics

        # Advisory hint (NOT a decision)
        if failure_ratio > 0.6:
            hint = DecisionHint.EXCLUDE_CANDIDATE
            rationale = f"High failure ratio ({failure_ratio:.2f})"
        elif failure_ratio > 0.4:
            hint = DecisionHint.REVIEW
            rationale = f"Moderate failure ratio ({failure_ratio:.2f})"
        else:
            hint = DecisionHint.KEEP
            rationale = f"Good contribution (ICS={ics:.2f})"

        return IndicatorContribution(
            indicator_id=indicator_id,
            geometric_influence=inputs["geometric_influence"],
            marginal_contribution=inputs["marginal_contribution"],
            stability=inputs["stability"],
            noise_pressure=inputs["noise_pressure"],
            confidence_effect=inputs["confidence_effect"],
            ics=ics,
            failure_ratio=failure_ratio,
            decision_hint=hint,
            rationale=rationale,
            weights_used=self.weights.copy(),
        )

    def analyze_cohort(
        self,
        indicator_metrics: Dict[str, Dict[str, float]],
        cohort_id: Optional[str] = None,
    ) -> ContributionReport:
        """
        Analyze all indicators in a cohort.

        Args:
            indicator_metrics: {
                "SPY": {"geometric_influence": 0.8, "stability": 0.7, ...},
                "VIX": {"geometric_influence": 0.3, "stability": 0.9, ...},
            }
        """
        from datetime import datetime

        contributions = []
        for ind_id, metrics in indicator_metrics.items():
            contrib = self.compute_contribution(
                indicator_id=ind_id,
                geometric_influence=metrics.get("geometric_influence", 0.5),
                marginal_contribution=metrics.get("marginal_contribution", 0.5),
                stability=metrics.get("stability", 0.5),
                noise_pressure=metrics.get("noise_pressure", 0.5),
                confidence_effect=metrics.get("confidence_effect", 0.5),
            )
            contributions.append(contrib)

        return ContributionReport(
            contributions=contributions,
            timestamp=datetime.now().isoformat(),
            cohort_id=cohort_id,
        )
