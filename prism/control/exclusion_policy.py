"""
Exclusion Policy Control Plane

THIS IS THE ONLY PLACE WHERE THRESHOLDS LIVE.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from enum import Enum
import logging

from prism.agents.indicator_contribution import ContributionReport, IndicatorContribution


class PolicyMode(Enum):
    OBSERVE = "observe"   # Log only, exclude nothing
    WARN = "warn"         # Log warnings, exclude nothing
    EXCLUDE = "exclude"   # Actually exclude indicators


class StrictnessProfile(Enum):
    PERMISSIVE = "permissive"  # failure_threshold = 0.70
    BALANCED = "balanced"      # failure_threshold = 0.50
    STRICT = "strict"          # failure_threshold = 0.35


@dataclass
class ExclusionPolicy:
    """
    Central policy configuration.

    RULE: All threshold logic lives HERE and ONLY here.
    """
    mode: PolicyMode = PolicyMode.OBSERVE
    failure_threshold: float = 0.50
    min_indicators_required: int = 5
    strictness_profile: StrictnessProfile = StrictnessProfile.BALANCED

    def __post_init__(self):
        # Apply strictness profile defaults if threshold not explicitly set
        profile_thresholds = {
            StrictnessProfile.PERMISSIVE: 0.70,
            StrictnessProfile.BALANCED: 0.50,
            StrictnessProfile.STRICT: 0.35,
        }
        if self.failure_threshold == 0.50:  # default
            self.failure_threshold = profile_thresholds[self.strictness_profile]


@dataclass
class ExclusionDecision:
    """Result of applying policy to a contribution report."""
    active_indicators: Set[str]
    excluded_indicators: Set[str]
    exclusion_reasons: Dict[str, str]  # indicator_id -> reason
    confidence_impact: float  # Estimated impact on analysis confidence
    policy_used: ExclusionPolicy

    def summary(self) -> str:
        return (
            f"Active: {len(self.active_indicators)} | "
            f"Excluded: {len(self.excluded_indicators)} | "
            f"Confidence impact: {self.confidence_impact:.2f}"
        )


class ExclusionController:
    """
    Applies exclusion policy to contribution reports.

    RULES:
    - This is the ONLY class that applies thresholds
    - All exclusions are logged with full rationale
    - Exclusions can be overridden without touching other code
    """

    def __init__(self, policy: Optional[ExclusionPolicy] = None):
        self.policy = policy or ExclusionPolicy()
        self.logger = logging.getLogger("prism.control.exclusion")

    def apply(self, report: ContributionReport) -> ExclusionDecision:
        """
        Apply exclusion policy to a ContributionReport.

        Args:
            report: ContributionReport from IndicatorContributionAgent

        Returns:
            ExclusionDecision with active/excluded sets
        """
        active = set()
        excluded = set()
        reasons = {}

        for contrib in report.contributions:
            should_exclude = contrib.failure_ratio > self.policy.failure_threshold

            if should_exclude and self.policy.mode == PolicyMode.EXCLUDE:
                excluded.add(contrib.indicator_id)
                reasons[contrib.indicator_id] = (
                    f"failure_ratio={contrib.failure_ratio:.3f} > "
                    f"threshold={self.policy.failure_threshold:.3f}"
                )
                self.logger.warning(
                    f"EXCLUDED {contrib.indicator_id}: {reasons[contrib.indicator_id]}"
                )
            elif should_exclude and self.policy.mode == PolicyMode.WARN:
                active.add(contrib.indicator_id)
                self.logger.warning(
                    f"WOULD EXCLUDE {contrib.indicator_id}: "
                    f"failure_ratio={contrib.failure_ratio:.3f}"
                )
            else:
                active.add(contrib.indicator_id)

        # Safety check: don't exclude too many
        if len(active) < self.policy.min_indicators_required:
            self.logger.error(
                f"Too few indicators remaining ({len(active)}). "
                f"Reverting exclusions to meet minimum of {self.policy.min_indicators_required}"
            )
            # Restore excluded indicators by lowest failure_ratio first
            sorted_excluded = sorted(
                [(c.indicator_id, c.failure_ratio) for c in report.contributions
                 if c.indicator_id in excluded],
                key=lambda x: x[1]
            )
            while len(active) < self.policy.min_indicators_required and sorted_excluded:
                restore_id, _ = sorted_excluded.pop(0)
                active.add(restore_id)
                excluded.remove(restore_id)
                del reasons[restore_id]

        # Estimate confidence impact
        if excluded:
            excluded_ics = [c.ics for c in report.contributions if c.indicator_id in excluded]
            confidence_impact = sum(excluded_ics) / len(report.contributions)
        else:
            confidence_impact = 0.0

        return ExclusionDecision(
            active_indicators=active,
            excluded_indicators=excluded,
            exclusion_reasons=reasons,
            confidence_impact=confidence_impact,
            policy_used=self.policy,
        )

    def update_policy(self, **kwargs):
        """Update policy parameters at runtime."""
        for key, value in kwargs.items():
            if hasattr(self.policy, key):
                setattr(self.policy, key, value)
                self.logger.info(f"Policy updated: {key}={value}")
