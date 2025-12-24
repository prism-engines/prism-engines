"""
Tests for Indicator Contribution Controls (ICC) System

Tests verify:
1. Contribution score calculation
2. Exclusion policy modes (observe, warn, exclude)
3. Allowlist registry functionality
4. Minimum indicator safety checks
"""

import pytest

from prism.agents.indicator_contribution import (
    IndicatorContributionAgent,
    IndicatorContribution,
    ContributionReport,
    DecisionHint,
)
from prism.control.exclusion_policy import (
    ExclusionController,
    ExclusionPolicy,
    PolicyMode,
    StrictnessProfile,
)
from prism.control.indicator_allowlist import (
    IndicatorAllowlist,
    AllowlistRegistry,
    set_current_allowlist,
    get_current_allowlist,
    indicator_is_active,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Fresh IndicatorContributionAgent for each test."""
    return IndicatorContributionAgent()


@pytest.fixture
def sample_metrics():
    """Sample indicator metrics for testing."""
    return {
        "SPY": {
            "geometric_influence": 0.8,
            "marginal_contribution": 0.7,
            "stability": 0.9,
            "noise_pressure": 0.6,
            "confidence_effect": 0.8,
        },
        "VIX": {
            "geometric_influence": 0.3,
            "marginal_contribution": 0.4,
            "stability": 0.5,
            "noise_pressure": 0.3,
            "confidence_effect": 0.4,
        },
        "GOLD": {
            "geometric_influence": 0.6,
            "marginal_contribution": 0.5,
            "stability": 0.7,
            "noise_pressure": 0.5,
            "confidence_effect": 0.6,
        },
    }


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear allowlist registry before each test."""
    AllowlistRegistry().clear()
    yield
    AllowlistRegistry().clear()


# =============================================================================
# Tests: Contribution Score Calculation
# =============================================================================

class TestContributionScore:
    """Test ICS computation."""

    def test_contribution_score_calculation(self, agent):
        """Basic ICS calculation should work correctly."""
        contrib = agent.compute_contribution(
            indicator_id="SPY",
            geometric_influence=0.8,
            marginal_contribution=0.7,
            stability=0.9,
            noise_pressure=0.6,
            confidence_effect=0.8,
        )

        assert 0 <= contrib.ics <= 1
        assert contrib.failure_ratio == 1 - contrib.ics
        assert contrib.decision_hint == DecisionHint.KEEP

    def test_high_failure_ratio_hints_exclude(self, agent):
        """High failure ratio should hint exclusion."""
        contrib = agent.compute_contribution(
            indicator_id="BAD",
            geometric_influence=0.1,
            marginal_contribution=0.1,
            stability=0.1,
            noise_pressure=0.1,
            confidence_effect=0.1,
        )

        assert contrib.failure_ratio > 0.6
        assert contrib.decision_hint == DecisionHint.EXCLUDE_CANDIDATE

    def test_moderate_failure_ratio_hints_review(self, agent):
        """Moderate failure ratio should hint review."""
        contrib = agent.compute_contribution(
            indicator_id="MEDIUM",
            geometric_influence=0.5,
            marginal_contribution=0.5,
            stability=0.5,
            noise_pressure=0.5,
            confidence_effect=0.5,
        )

        # ICS = 0.5, failure_ratio = 0.5
        assert 0.4 <= contrib.failure_ratio <= 0.6
        assert contrib.decision_hint == DecisionHint.REVIEW

    def test_weights_are_normalized(self, agent):
        """Weights should sum to 1."""
        total = sum(agent.weights.values())
        assert abs(total - 1.0) < 0.001

    def test_inputs_are_clamped(self, agent):
        """Input values should be clamped to [0, 1]."""
        contrib = agent.compute_contribution(
            indicator_id="OUTLIER",
            geometric_influence=1.5,  # > 1
            marginal_contribution=-0.2,  # < 0
            stability=0.5,
            noise_pressure=0.5,
            confidence_effect=0.5,
        )

        assert contrib.geometric_influence == 1.0
        assert contrib.marginal_contribution == 0.0

    def test_custom_weights(self):
        """Custom weights should be applied correctly."""
        custom_weights = {
            "geometric_influence": 1.0,
            "marginal_contribution": 0.0,
            "stability": 0.0,
            "noise_pressure": 0.0,
            "confidence_effect": 0.0,
        }
        agent = IndicatorContributionAgent(weights=custom_weights)

        contrib = agent.compute_contribution(
            indicator_id="TEST",
            geometric_influence=0.8,
            marginal_contribution=0.1,
            stability=0.1,
            noise_pressure=0.1,
            confidence_effect=0.1,
        )

        # With only geometric_influence weighted, ICS should equal it
        assert abs(contrib.ics - 0.8) < 0.001


# =============================================================================
# Tests: Cohort Analysis
# =============================================================================

class TestCohortAnalysis:
    """Test analyze_cohort method."""

    def test_analyze_cohort_returns_report(self, agent, sample_metrics):
        """Cohort analysis should return ContributionReport."""
        report = agent.analyze_cohort(sample_metrics, cohort_id="test_cohort")

        assert isinstance(report, ContributionReport)
        assert len(report.contributions) == 3
        assert report.cohort_id == "test_cohort"

    def test_report_to_dataframe(self, agent, sample_metrics):
        """Report should convert to DataFrame."""
        report = agent.analyze_cohort(sample_metrics)
        df = report.to_dataframe()

        assert len(df) == 3
        assert "indicator_id" in df.columns
        assert "ics" in df.columns
        assert "failure_ratio" in df.columns

    def test_default_metrics(self, agent):
        """Missing metrics should default to 0.5."""
        metrics = {"PARTIAL": {"geometric_influence": 0.8}}
        report = agent.analyze_cohort(metrics)

        contrib = report.contributions[0]
        assert contrib.stability == 0.5  # default


# =============================================================================
# Tests: Exclusion Policy - Observe Mode
# =============================================================================

class TestExclusionPolicyObserve:
    """Test ExclusionPolicy in observe mode."""

    def test_observe_mode_excludes_nothing(self, agent, sample_metrics):
        """In observe mode, nothing is excluded."""
        report = agent.analyze_cohort(sample_metrics)

        policy = ExclusionPolicy(mode=PolicyMode.OBSERVE, failure_threshold=0.3)
        controller = ExclusionController(policy)
        decision = controller.apply(report)

        # All indicators should be active
        assert len(decision.active_indicators) == 3
        assert len(decision.excluded_indicators) == 0


# =============================================================================
# Tests: Exclusion Policy - Warn Mode
# =============================================================================

class TestExclusionPolicyWarn:
    """Test ExclusionPolicy in warn mode."""

    def test_warn_mode_excludes_nothing(self, agent, sample_metrics):
        """In warn mode, nothing is excluded (just logged)."""
        report = agent.analyze_cohort(sample_metrics)

        policy = ExclusionPolicy(mode=PolicyMode.WARN, failure_threshold=0.3)
        controller = ExclusionController(policy)
        decision = controller.apply(report)

        # All indicators should still be active
        assert len(decision.active_indicators) == 3
        assert len(decision.excluded_indicators) == 0


# =============================================================================
# Tests: Exclusion Policy - Exclude Mode
# =============================================================================

class TestExclusionPolicyExclude:
    """Test ExclusionPolicy in exclude mode."""

    def test_exclude_mode_removes_bad_indicators(self):
        """In exclude mode, bad indicators are removed."""
        # Create mock report with high failure indicator
        report = ContributionReport(
            contributions=[
                IndicatorContribution(indicator_id="BAD", ics=0.2, failure_ratio=0.8),
                IndicatorContribution(indicator_id="GOOD", ics=0.9, failure_ratio=0.1),
            ],
            timestamp="2024-01-01",
        )

        # Use min_indicators_required=1 to allow exclusion with only 2 indicators
        policy = ExclusionPolicy(
            mode=PolicyMode.EXCLUDE,
            failure_threshold=0.5,
            min_indicators_required=1
        )
        controller = ExclusionController(policy)
        decision = controller.apply(report)

        assert "BAD" in decision.excluded_indicators
        assert "GOOD" in decision.active_indicators
        assert "BAD" in decision.exclusion_reasons

    def test_exclusion_reason_provided(self):
        """Exclusion reasons should be documented."""
        report = ContributionReport(
            contributions=[
                IndicatorContribution(indicator_id="BAD", ics=0.2, failure_ratio=0.8),
                IndicatorContribution(indicator_id="GOOD", ics=0.9, failure_ratio=0.1),
            ],
            timestamp="2024-01-01",
        )

        # Use min_indicators_required=1 to allow exclusion
        policy = ExclusionPolicy(
            mode=PolicyMode.EXCLUDE,
            failure_threshold=0.5,
            min_indicators_required=1
        )
        controller = ExclusionController(policy)
        decision = controller.apply(report)

        reason = decision.exclusion_reasons.get("BAD", "")
        assert "failure_ratio" in reason
        assert "threshold" in reason


# =============================================================================
# Tests: Minimum Indicator Safety
# =============================================================================

class TestMinimumIndicatorSafety:
    """Test minimum indicator requirement."""

    def test_minimum_indicators_enforced(self):
        """Cannot exclude below minimum threshold."""
        # Create report where all would be excluded
        report = ContributionReport(
            contributions=[
                IndicatorContribution(indicator_id="A", ics=0.2, failure_ratio=0.8),
                IndicatorContribution(indicator_id="B", ics=0.3, failure_ratio=0.7),
                IndicatorContribution(indicator_id="C", ics=0.25, failure_ratio=0.75),
                IndicatorContribution(indicator_id="D", ics=0.35, failure_ratio=0.65),
                IndicatorContribution(indicator_id="E", ics=0.4, failure_ratio=0.6),
                IndicatorContribution(indicator_id="F", ics=0.15, failure_ratio=0.85),
            ],
            timestamp="2024-01-01",
        )

        policy = ExclusionPolicy(
            mode=PolicyMode.EXCLUDE,
            failure_threshold=0.5,
            min_indicators_required=5
        )
        controller = ExclusionController(policy)
        decision = controller.apply(report)

        # At least 5 should remain active
        assert len(decision.active_indicators) >= 5


# =============================================================================
# Tests: Strictness Profiles
# =============================================================================

class TestStrictnessProfiles:
    """Test strictness profile presets."""

    def test_permissive_threshold(self):
        """Permissive profile should have high threshold."""
        policy = ExclusionPolicy(strictness_profile=StrictnessProfile.PERMISSIVE)
        assert policy.failure_threshold == 0.70

    def test_balanced_threshold(self):
        """Balanced profile should have medium threshold."""
        policy = ExclusionPolicy(strictness_profile=StrictnessProfile.BALANCED)
        assert policy.failure_threshold == 0.50

    def test_strict_threshold(self):
        """Strict profile should have low threshold."""
        policy = ExclusionPolicy(strictness_profile=StrictnessProfile.STRICT)
        assert policy.failure_threshold == 0.35


# =============================================================================
# Tests: Allowlist Registry
# =============================================================================

class TestAllowlistRegistry:
    """Test allowlist registry functionality."""

    def test_set_and_get_allowlist(self):
        """Should be able to set and retrieve allowlist."""
        allowlist = IndicatorAllowlist(
            active_indicators=frozenset(["SPY", "GOLD"]),
            excluded_indicators=frozenset(["VIX"]),
            policy_mode="exclude",
        )
        set_current_allowlist(allowlist)

        retrieved = get_current_allowlist()
        assert retrieved is not None
        assert "SPY" in retrieved
        assert "VIX" not in retrieved

    def test_indicator_is_active(self):
        """indicator_is_active should check allowlist."""
        allowlist = IndicatorAllowlist(
            active_indicators=frozenset(["SPY"]),
            excluded_indicators=frozenset(["VIX"]),
            policy_mode="exclude",
        )
        set_current_allowlist(allowlist)

        assert indicator_is_active("SPY") is True
        assert indicator_is_active("VIX") is False

    def test_no_allowlist_allows_all(self):
        """With no allowlist, all indicators are active."""
        # Registry is cleared by fixture
        assert indicator_is_active("ANYTHING") is True

    def test_allowlist_iteration(self):
        """Allowlist should be iterable."""
        allowlist = IndicatorAllowlist(
            active_indicators=frozenset(["SPY", "GOLD"]),
            excluded_indicators=frozenset(),
            policy_mode="observe",
        )

        indicators = list(allowlist)
        assert len(indicators) == 2
        assert "SPY" in indicators
        assert "GOLD" in indicators

    def test_allowlist_length(self):
        """Allowlist should report length."""
        allowlist = IndicatorAllowlist(
            active_indicators=frozenset(["A", "B", "C"]),
            excluded_indicators=frozenset(),
            policy_mode="observe",
        )

        assert len(allowlist) == 3


# =============================================================================
# Tests: Decision Summary
# =============================================================================

class TestDecisionSummary:
    """Test decision summary output."""

    def test_summary_format(self):
        """Decision summary should be human-readable."""
        report = ContributionReport(
            contributions=[
                IndicatorContribution(indicator_id="A", ics=0.8, failure_ratio=0.2),
                IndicatorContribution(indicator_id="B", ics=0.3, failure_ratio=0.7),
            ],
            timestamp="2024-01-01",
        )

        policy = ExclusionPolicy(mode=PolicyMode.EXCLUDE, failure_threshold=0.5)
        controller = ExclusionController(policy)
        decision = controller.apply(report)

        summary = decision.summary()
        assert "Active" in summary
        assert "Excluded" in summary
        assert "Confidence impact" in summary


# =============================================================================
# Tests: Policy Updates
# =============================================================================

class TestPolicyUpdates:
    """Test runtime policy updates."""

    def test_update_policy_threshold(self):
        """Should be able to update threshold at runtime."""
        policy = ExclusionPolicy(mode=PolicyMode.EXCLUDE, failure_threshold=0.5)
        controller = ExclusionController(policy)

        controller.update_policy(failure_threshold=0.7)
        assert controller.policy.failure_threshold == 0.7

    def test_update_policy_mode(self):
        """Should be able to update mode at runtime."""
        policy = ExclusionPolicy(mode=PolicyMode.OBSERVE)
        controller = ExclusionController(policy)

        controller.update_policy(mode=PolicyMode.EXCLUDE)
        assert controller.policy.mode == PolicyMode.EXCLUDE


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
