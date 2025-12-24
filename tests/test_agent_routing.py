"""
Tests for GeometryDiagnosticAgent and EngineRoutingAgent

Tests verify:
1. GeometryDiagnosticAgent correctly wraps GeometrySignatureAgent
2. EngineRoutingAgent properly routes based on geometry
3. High noise suppresses most engines
4. Strong oscillator geometry enables Granger
5. Strong latent_flow geometry enables SIR
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts to path
_scripts_path = Path(__file__).parent.parent / "scripts"
if str(_scripts_path) not in sys.path:
    sys.path.insert(0, str(_scripts_path))

from prism_geometry_signature_agent import GeometryProfile, GeometryClass
from prism_engine_gates import Domain

from prism.agents.prism_agent_foundation import DiagnosticRegistry, RoutingResult
from prism.agents.prism_agent_routing import (
    GeometryDiagnosticAgent,
    EngineRoutingAgent,
    GEOMETRY_ALIGNMENT,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def registry():
    """Fresh DiagnosticRegistry for each test."""
    return DiagnosticRegistry()


@pytest.fixture
def geometry_agent(registry):
    """GeometryDiagnosticAgent instance."""
    return GeometryDiagnosticAgent(registry)


@pytest.fixture
def routing_agent(registry):
    """EngineRoutingAgent instance."""
    return EngineRoutingAgent(registry, domain=Domain.FINANCE)


@pytest.fixture
def epi_routing_agent(registry):
    """EngineRoutingAgent for epidemiology domain."""
    return EngineRoutingAgent(registry, domain=Domain.EPIDEMIOLOGY)


# =============================================================================
# Helper: Create GeometryProfile with specific scores
# =============================================================================

def make_profile(
    latent_flow: float = 0.3,
    oscillator: float = 0.3,
    reflexive: float = 0.3,
    noise: float = 0.3,
) -> GeometryProfile:
    """Create a GeometryProfile with specified scores."""
    return GeometryProfile(
        latent_flow_score=latent_flow,
        oscillator_score=oscillator,
        reflexive_score=reflexive,
        noise_score=noise,
    )


# =============================================================================
# Tests: GeometryDiagnosticAgent
# =============================================================================

class TestGeometryDiagnosticAgent:
    """Test GeometryDiagnosticAgent wrapper."""

    def test_analyze_sine_wave(self, geometry_agent):
        """Sine wave should detect oscillator geometry."""
        np.random.seed(42)
        sine = np.sin(np.linspace(0, 8 * np.pi, 200)) + np.random.normal(0, 0.1, 200)

        profile = geometry_agent.run({"cleaned_data": sine})

        # Sine wave should have strong oscillator score and periodicity
        assert profile.oscillator_score >= 0.5, "Sine wave should have high oscillator score"
        # May also trigger latent_flow due to asymmetry diagnostics, which is acceptable

    def test_analyze_logistic_growth(self, geometry_agent):
        """Logistic growth should detect latent flow geometry."""
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        logistic = 100 / (1 + np.exp(-1.5 * (t - 5))) + np.random.normal(0, 2, len(t))

        profile = geometry_agent.run({"cleaned_data": logistic})

        # Should detect some latent flow signature
        assert profile.latent_flow_score >= 0.2

    def test_analyze_pure_noise(self, geometry_agent):
        """Pure noise should have high noise score."""
        np.random.seed(42)
        noise = np.random.normal(0, 1, 200)

        profile = geometry_agent.run({"cleaned_data": noise})

        assert profile.noise_score > 0.4

    def test_registry_updated(self, geometry_agent, registry):
        """Registry should contain geometry scores after run."""
        np.random.seed(42)
        data = np.random.randn(100)

        geometry_agent.run({"cleaned_data": data})

        assert registry.get("geometry_diagnostic", "noise_score") is not None
        assert registry.get("geometry_diagnostic", "dominant_geometry") is not None

    def test_explain_method(self, geometry_agent):
        """explain() should return formatted string."""
        np.random.seed(42)
        data = np.random.randn(100)

        geometry_agent.run({"cleaned_data": data})
        explanation = geometry_agent.explain()

        assert "Geometry Diagnostic Results" in explanation
        assert "Dominant Geometry" in explanation

    def test_no_data_returns_high_noise(self, geometry_agent):
        """Missing data should return high noise profile."""
        profile = geometry_agent.run({})

        assert profile.noise_score >= 0.8


# =============================================================================
# Tests: EngineRoutingAgent - Basic Routing
# =============================================================================

class TestEngineRoutingBasic:
    """Test basic routing functionality."""

    def test_core_engines_always_allowed(self, routing_agent):
        """Core engines (PCA, correlation, entropy, wavelets) should be allowed."""
        profile = make_profile(noise=0.2)

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 128,  # wavelets requires MIN_N=128 per Spec E
            "sample_count": 500,
        })

        core_engines = ["pca", "correlation", "entropy", "wavelets"]
        for engine in core_engines:
            assert result.engine_eligibility.get(engine) in ["allowed", "downweighted"], \
                f"{engine} should be allowed/downweighted"
            assert result.weights.get(engine, 0) > 0, f"{engine} should have positive weight"

    def test_excluded_engines_suppressed(self, routing_agent):
        """Excluded engines should always be suppressed."""
        profile = make_profile()

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 1000,
            "sample_count": 10000,
        })

        excluded = ["lyapunov", "rqa", "fft", "dtw"]
        for engine in excluded:
            assert result.engine_eligibility.get(engine) == "suppressed"
            assert result.weights.get(engine, 0) == 0

    def test_window_size_gate(self, routing_agent):
        """Engines should be suppressed if window too small."""
        profile = make_profile(reflexive=0.8)  # Good reflexive score

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 20,  # Very small
            "sample_count": 500,
        })

        # HMM requires min_window=126
        assert result.engine_eligibility.get("hmm") == "suppressed"
        assert "window_size" in result.suppression_reasons.get("hmm", "")

    def test_sample_count_gate(self, routing_agent):
        """Engines should be suppressed if samples too few."""
        profile = make_profile()

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 500,
            "sample_count": 10,  # Very few
        })

        # Mutual information requires min_samples=100
        assert result.engine_eligibility.get("mutual_information") == "suppressed"


# =============================================================================
# Tests: High Noise Suppression
# =============================================================================

class TestHighNoiseSuppression:
    """Test that high noise suppresses most engines."""

    def test_high_noise_reduces_weights(self, routing_agent):
        """High noise should reduce all weights."""
        low_noise_profile = make_profile(reflexive=0.6, noise=0.1)
        high_noise_profile = make_profile(reflexive=0.6, noise=0.8)

        low_noise_result = routing_agent.run({
            "geometry_diagnostic_result": low_noise_profile,
            "window_size": 200,
            "sample_count": 500,
        })

        high_noise_result = routing_agent.run({
            "geometry_diagnostic_result": high_noise_profile,
            "window_size": 200,
            "sample_count": 500,
        })

        # Weights should be lower with high noise
        for engine in ["pca", "correlation", "entropy"]:
            low_weight = low_noise_result.weights.get(engine, 0)
            high_weight = high_noise_result.weights.get(engine, 0)
            assert high_weight < low_weight, \
                f"{engine} weight should be lower with high noise"

    def test_extreme_noise_downweights_all(self, routing_agent):
        """Extreme noise should heavily downweight all engines."""
        profile = make_profile(noise=0.95)

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        # All non-excluded engines should have low weights
        for engine, weight in result.weights.items():
            if result.engine_eligibility.get(engine) != "suppressed":
                assert weight < 0.6, f"{engine} weight {weight} too high for noisy data"


# =============================================================================
# Tests: Oscillator Geometry Enables Granger
# =============================================================================

class TestOscillatorEnablesGranger:
    """Test that strong oscillator geometry enables Granger."""

    def test_strong_oscillator_allows_granger(self, epi_routing_agent):
        """Strong oscillator score should allow Granger in epi domain."""
        profile = make_profile(oscillator=0.7, noise=0.2)

        result = epi_routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
            "domain": Domain.EPIDEMIOLOGY,
        })

        assert result.engine_eligibility.get("granger") in ["allowed", "downweighted"]
        assert result.weights.get("granger", 0) > 0

    def test_weak_oscillator_suppresses_granger(self, epi_routing_agent):
        """Weak oscillator (and latent_flow) should suppress Granger."""
        profile = make_profile(oscillator=0.2, latent_flow=0.2, noise=0.2)

        result = epi_routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
            "domain": Domain.EPIDEMIOLOGY,
        })

        assert result.engine_eligibility.get("granger") == "suppressed"
        assert "Geometry mismatch" in result.suppression_reasons.get("granger", "")

    def test_latent_flow_also_enables_granger(self, epi_routing_agent):
        """Latent flow should also enable Granger (OR condition)."""
        profile = make_profile(oscillator=0.2, latent_flow=0.7, noise=0.2)

        result = epi_routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
            "domain": Domain.EPIDEMIOLOGY,
        })

        # Granger has OR condition: oscillator >= 0.5 OR latent_flow >= 0.5
        assert result.engine_eligibility.get("granger") in ["allowed", "downweighted"]


# =============================================================================
# Tests: Latent Flow Geometry Enables SIR
# =============================================================================

class TestLatentFlowEnablesSIR:
    """Test that strong latent_flow geometry enables SIR."""

    def test_strong_latent_flow_allows_sir(self, epi_routing_agent):
        """Strong latent_flow score should allow SIR in epi domain."""
        profile = make_profile(latent_flow=0.8, noise=0.2)

        result = epi_routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
            "domain": Domain.EPIDEMIOLOGY,
        })

        assert result.engine_eligibility.get("sir") in ["allowed", "downweighted"]
        assert result.weights.get("sir", 0) > 0

    def test_weak_latent_flow_suppresses_sir(self, epi_routing_agent):
        """Weak latent_flow should suppress SIR."""
        profile = make_profile(latent_flow=0.3, noise=0.2)

        result = epi_routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
            "domain": Domain.EPIDEMIOLOGY,
        })

        assert result.engine_eligibility.get("sir") == "suppressed"
        assert "latent_flow" in result.suppression_reasons.get("sir", "")

    def test_sir_requires_epi_domain(self, routing_agent):
        """SIR should be suppressed in finance domain."""
        profile = make_profile(latent_flow=0.9)  # Strong latent flow

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
            "domain": Domain.FINANCE,
        })

        assert result.engine_eligibility.get("sir") == "suppressed"
        assert "domain" in result.suppression_reasons.get("sir", "").lower()


# =============================================================================
# Tests: Reflexive Geometry Enables HMM/DMD/Copula
# =============================================================================

class TestReflexiveEnablesHMM:
    """Test that reflexive geometry enables HMM, DMD, Copula."""

    def test_strong_reflexive_allows_hmm(self, routing_agent):
        """Strong reflexive score should allow HMM."""
        profile = make_profile(reflexive=0.7, noise=0.2)

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        assert result.engine_eligibility.get("hmm") in ["allowed", "downweighted"]

    def test_weak_reflexive_suppresses_hmm(self, routing_agent):
        """Weak reflexive should suppress HMM."""
        profile = make_profile(reflexive=0.2, noise=0.2)

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        assert result.engine_eligibility.get("hmm") == "suppressed"

    def test_strong_reflexive_allows_dmd(self, routing_agent):
        """Strong reflexive should allow DMD."""
        profile = make_profile(reflexive=0.6, noise=0.2)

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        assert result.engine_eligibility.get("dmd") in ["allowed", "downweighted"]

    def test_copula_requires_reflexive_and_finance(self, routing_agent):
        """Copula needs reflexive geometry AND finance domain."""
        profile = make_profile(reflexive=0.7, noise=0.2)

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 300,  # Copula needs 252
            "sample_count": 500,
            "domain": Domain.FINANCE,
        })

        assert result.engine_eligibility.get("copula") in ["allowed", "downweighted"]


# =============================================================================
# Tests: Pending Checks
# =============================================================================

class TestPendingChecks:
    """Test that required validation checks are tracked."""

    def test_hmm_requires_stability_check(self, routing_agent):
        """HMM should require stability check."""
        profile = make_profile(reflexive=0.7)

        routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        pending = routing_agent.get_pending_checks()
        assert "stability_check" in pending.get("hmm", [])

    def test_mutual_info_requires_null_test(self, routing_agent):
        """Mutual information should require null test."""
        profile = make_profile()

        routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        pending = routing_agent.get_pending_checks()
        assert "null_test" in pending.get("mutual_information", [])


# =============================================================================
# Tests: Registry and Audit Trail
# =============================================================================

class TestRegistryIntegration:
    """Test registry and audit trail integration."""

    def test_routing_stored_in_registry(self, routing_agent, registry):
        """Routing results should be stored in registry."""
        profile = make_profile()

        routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        assert registry.get("engine_routing", "eligibility") is not None
        assert registry.get("engine_routing", "weights") is not None

    def test_audit_trail_logged(self, routing_agent, registry):
        """Routing decisions should be in audit trail."""
        profile = make_profile()

        routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        trail = registry.export_audit_trail()
        assert len(trail) >= 1
        assert trail[0].agent_name == "engine_routing"


# =============================================================================
# Tests: explain() Method
# =============================================================================

class TestExplainMethod:
    """Test explain() output formatting."""

    def test_explain_shows_allowed(self, routing_agent):
        """explain() should list allowed engines."""
        profile = make_profile(reflexive=0.7, noise=0.2)

        routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        explanation = routing_agent.explain()
        assert "ALLOWED" in explanation
        assert "pca" in explanation.lower()

    def test_explain_shows_suppressed(self, routing_agent):
        """explain() should list suppressed engines with reasons."""
        profile = make_profile(noise=0.2)

        routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 50,  # Too small for many engines
            "sample_count": 500,
        })

        explanation = routing_agent.explain()
        assert "SUPPRESSED" in explanation


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_profile_returns_empty(self, routing_agent):
        """Missing profile should return empty routing."""
        result = routing_agent.run({
            "window_size": 200,
            "sample_count": 500,
        })

        assert result.engine_eligibility == {}
        assert "No geometry profile" in str(result.suppression_reasons)

    def test_zero_scores_profile(self, routing_agent):
        """All-zero scores should still route core engines."""
        profile = make_profile(
            latent_flow=0.0,
            oscillator=0.0,
            reflexive=0.0,
            noise=0.0,
        )

        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 500,
        })

        # Core engines should still be allowed (no geometry requirement)
        assert result.engine_eligibility.get("pca") in ["allowed", "downweighted"]


# =============================================================================
# Integration Test: End-to-End Pipeline
# =============================================================================

class TestEndToEnd:
    """Test full geometry -> routing pipeline."""

    def test_pipeline_with_synthetic_data(self, geometry_agent, routing_agent):
        """Test full pipeline with synthetic oscillator data."""
        np.random.seed(42)
        sine = np.sin(np.linspace(0, 8 * np.pi, 200)) + np.random.normal(0, 0.1, 200)

        # Run geometry analysis
        profile = geometry_agent.run({"cleaned_data": sine})

        # Run routing
        result = routing_agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 200,
            "sample_count": 200,
        })

        # Should have some allowed engines
        allowed = [e for e, s in result.engine_eligibility.items() if s == "allowed"]
        assert len(allowed) > 0, "Should have at least one allowed engine"

        # Explain should work
        geo_explain = geometry_agent.explain()
        route_explain = routing_agent.explain()

        assert len(geo_explain) > 0
        assert len(route_explain) > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
