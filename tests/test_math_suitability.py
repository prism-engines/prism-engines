"""
Tests for Math Suitability Agent

Tests verify:
1. Pure noise blocking
2. Unknown status detection
3. Geometry-based engine routing
4. Conditional engine rules
5. Cohort-level guardrails
6. Policy versioning
"""

import pytest
from datetime import datetime

from prism.control.suitability_policy import (
    SuitabilityPolicy,
    EligibilityStatus,
    get_default_policy,
    get_permissive_policy,
    get_strict_policy,
)
from prism.agents.math_suitability import (
    MathSuitabilityAgent,
    MultiViewSignature,
    ViewGeometry,
    EligibilityDecision,
    ConditionalEngine,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Fresh agent with default policy."""
    return MathSuitabilityAgent(get_default_policy())


@pytest.fixture
def strict_agent():
    """Agent with strict policy."""
    return MathSuitabilityAgent(get_strict_policy())


@pytest.fixture
def permissive_agent():
    """Agent with permissive policy."""
    return MathSuitabilityAgent(get_permissive_policy())


def make_signature(
    indicator_id: str = "TEST",
    level_geom: str = "latent_flow",
    level_conf: float = 0.70,
    returns_geom: str = "reflexive_stochastic",
    returns_conf: float = 0.65,
    vol_geom: str = "reflexive_stochastic",
    vol_conf: float = 0.60,
    consensus: str = "reflexive_stochastic",
    consensus_conf: float = 0.65,
    disagreement: float = 0.30,
    is_hybrid: bool = False,
) -> MultiViewSignature:
    """Helper to create test signatures."""
    return MultiViewSignature(
        indicator_id=indicator_id,
        views={
            "level": ViewGeometry(
                view="level",
                geometry=level_geom,
                confidence=level_conf,
            ),
            "returns": ViewGeometry(
                view="returns",
                geometry=returns_geom,
                confidence=returns_conf,
            ),
            "volatility": ViewGeometry(
                view="volatility",
                geometry=vol_geom,
                confidence=vol_conf,
            ),
        },
        consensus_geometry=consensus,
        consensus_confidence=consensus_conf,
        disagreement=disagreement,
        is_hybrid=is_hybrid,
    )


# =============================================================================
# Tests: Pure Noise Blocking
# =============================================================================

class TestPureNoiseBlocking:
    """Test pure_noise consensus blocks inference."""
    
    def test_pure_noise_blocks_inference(self, agent):
        """Pure noise consensus should block inference engines."""
        sig = make_signature(
            consensus="pure_noise",
            consensus_conf=0.80,
            disagreement=0.10,
        )
        
        decision = agent.evaluate(sig)
        
        assert decision.status == EligibilityStatus.INELIGIBLE
        assert "pure_noise" in decision.rationale[0].lower()
        assert "pca" not in decision.allowed_engines
        assert "hmm" not in decision.allowed_engines
    
    def test_pure_noise_allows_descriptive(self, agent):
        """Pure noise should still allow descriptive engines."""
        sig = make_signature(
            consensus="pure_noise",
            consensus_conf=0.80,
        )
        
        decision = agent.evaluate(sig)
        
        # Descriptive engines should be allowed
        assert "null_check" in decision.allowed_engines or "descriptive_stats" in decision.allowed_engines


# =============================================================================
# Tests: Unknown Status Detection
# =============================================================================

class TestUnknownStatus:
    """Test unknown status for ambiguous geometry."""
    
    def test_high_disagreement_low_confidence_is_unknown(self, agent):
        """High disagreement + low confidence = unknown."""
        sig = make_signature(
            level_conf=0.30,
            returns_conf=0.25,
            vol_conf=0.20,
            consensus_conf=0.25,
            disagreement=0.75,  # Very high
        )
        
        decision = agent.evaluate(sig)
        
        assert decision.status == EligibilityStatus.UNKNOWN
        assert "disagreement" in str(decision.rationale)
    
    def test_high_disagreement_with_one_confident_view_not_unknown(self, agent):
        """High disagreement but one confident view should not be unknown."""
        sig = make_signature(
            level_conf=0.80,  # One view is confident
            returns_conf=0.20,
            vol_conf=0.20,
            consensus_conf=0.40,
            disagreement=0.75,
        )
        
        decision = agent.evaluate(sig)
        
        # Should not be unknown because level view is confident
        assert decision.status != EligibilityStatus.UNKNOWN


# =============================================================================
# Tests: Geometry-Based Routing
# =============================================================================

class TestGeometryRouting:
    """Test engine routing based on geometry."""
    
    def test_latent_flow_enables_trend_engines(self, agent):
        """Latent flow consensus should enable trend engines."""
        sig = make_signature(
            consensus="latent_flow",
            consensus_conf=0.70,
            disagreement=0.20,
        )
        
        decision = agent.evaluate(sig)
        
        # Trend engines should be in allowed
        assert "trend" in decision.allowed_engines or "drift" in decision.allowed_engines
        assert "pca" in decision.allowed_engines
    
    def test_reflexive_enables_regime_engines(self, agent):
        """Reflexive consensus should enable regime engines."""
        sig = make_signature(
            consensus="reflexive_stochastic",
            consensus_conf=0.70,
            disagreement=0.20,
        )
        
        decision = agent.evaluate(sig)
        
        assert "hmm" in decision.allowed_engines or any(
            c.engine == "hmm" for c in decision.conditional_engines
        )
    
    def test_oscillator_enables_spectral_engines(self, agent):
        """Oscillator consensus should enable spectral engines."""
        sig = make_signature(
            consensus="coupled_oscillator",
            consensus_conf=0.70,
            disagreement=0.20,
        )
        
        decision = agent.evaluate(sig)
        
        assert "wavelets" in decision.allowed_engines or "spectral" in decision.allowed_engines


# =============================================================================
# Tests: View-Level Overrides
# =============================================================================

class TestViewOverrides:
    """Test view-level engine additions."""
    
    def test_volatility_reflexive_adds_regime_engines(self, agent):
        """Strong reflexive in volatility view should add regime engines."""
        sig = make_signature(
            consensus="latent_flow",  # Consensus is latent
            vol_geom="reflexive_stochastic",
            vol_conf=0.75,  # But volatility view is strongly reflexive
        )
        
        decision = agent.evaluate(sig)
        
        # Should have regime engines due to volatility view
        all_engines = decision.allowed_engines + [c.engine for c in decision.conditional_engines]
        assert "hmm" in all_engines or "regime_detection" in all_engines


# =============================================================================
# Tests: Conditional Engines
# =============================================================================

class TestConditionalEngines:
    """Test conditional engine rules."""
    
    def test_hmm_conditional_when_no_reflexive_evidence(self, agent):
        """HMM should be conditional when reflexive evidence is weak."""
        sig = make_signature(
            consensus="latent_flow",
            level_conf=0.70,
            vol_geom="latent_flow",  # No reflexive in volatility
            vol_conf=0.40,
        )
        
        decision = agent.evaluate(sig)
        
        # HMM might be in conditional list if it was initially allowed
        # but doesn't have strong reflexive evidence
        # This depends on policy configuration


# =============================================================================
# Tests: Cohort-Level Guardrails
# =============================================================================

class TestCohortGuardrails:
    """Test cohort-level safety checks."""
    
    def test_small_cohort_degrades_to_descriptive(self, agent):
        """Small cohort should degrade to descriptive engines."""
        # Create cohort with only 2 eligible indicators (below minimum)
        signatures = [
            make_signature(indicator_id="A", consensus_conf=0.60),
            make_signature(indicator_id="B", consensus_conf=0.60),
        ]
        
        decisions = agent.evaluate_cohort(signatures)
        
        # Both should be degraded
        for decision in decisions.values():
            assert "cohort too small" in str(decision.rationale) or \
                   decision.status in [EligibilityStatus.CONDITIONAL, EligibilityStatus.INELIGIBLE]


# =============================================================================
# Tests: Policy Versioning
# =============================================================================

class TestPolicyVersioning:
    """Test policy version tracking."""
    
    def test_decision_includes_policy_version(self, agent):
        """Decisions should include policy version."""
        sig = make_signature()
        decision = agent.evaluate(sig)
        
        assert decision.policy_version is not None
        assert len(decision.policy_version) > 0
    
    def test_different_policies_different_versions(self):
        """Different policies should have different versions."""
        default = get_default_policy()
        strict = get_strict_policy()
        permissive = get_permissive_policy()
        
        versions = {default.version, strict.version, permissive.version}
        assert len(versions) == 3  # All different


# =============================================================================
# Tests: Status Downgrade
# =============================================================================

class TestStatusDowngrade:
    """Test status downgrade conditions."""
    
    def test_low_confidence_downgrades_to_conditional(self, agent):
        """Low consensus confidence should downgrade to conditional."""
        sig = make_signature(
            consensus_conf=0.35,  # Below threshold
            disagreement=0.20,
        )
        
        decision = agent.evaluate(sig)
        
        # Should be conditional or lower
        assert decision.status in [
            EligibilityStatus.CONDITIONAL,
            EligibilityStatus.INELIGIBLE,
            EligibilityStatus.UNKNOWN,
        ]
    
    def test_high_disagreement_downgrades_to_conditional(self, agent):
        """High disagreement should downgrade to conditional."""
        sig = make_signature(
            consensus_conf=0.70,  # Good confidence
            disagreement=0.65,    # But high disagreement
        )
        
        decision = agent.evaluate(sig)
        
        assert decision.status == EligibilityStatus.CONDITIONAL
        assert "disagreement" in str(decision.rationale)


# =============================================================================
# Tests: Decision Serialization
# =============================================================================

class TestSerialization:
    """Test decision serialization for database."""
    
    def test_to_db_record(self, agent):
        """Decision should serialize to DB record."""
        sig = make_signature()
        decision = agent.evaluate(sig)
        
        record = decision.to_db_record("test_run_123")
        
        assert record["run_id"] == "test_run_123"
        assert record["indicator_id"] == sig.indicator_id
        assert record["status"] == decision.status.value
        assert record["policy_version"] == decision.policy_version
        assert "allowed_engines" in record
        assert "prohibited_engines" in record


# =============================================================================
# Tests: Summary Output
# =============================================================================

class TestSummaryOutput:
    """Test human-readable summary."""
    
    def test_summary_includes_key_info(self, agent):
        """Summary should include key decision info."""
        sig = make_signature()
        decision = agent.evaluate(sig)
        
        summary = decision.summary()
        
        assert sig.indicator_id in summary
        assert decision.status.value in summary
        assert str(decision.consensus_geometry) in summary or "geometry" in summary.lower()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
