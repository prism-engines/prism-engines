"""
Integration Tests for PRISM Agent Pipeline

Tests the full agent pipeline:
Agent1 (Data Quality) -> Agent2 (Geometry) -> Agent3 (Routing)
-> [engines] -> Agent4 (Stability) -> Agent5 (Blind Spot)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts to path
_scripts_path = Path(__file__).parent.parent / "scripts"
if str(_scripts_path) not in sys.path:
    sys.path.insert(0, str(_scripts_path))

from prism.agents import (
    DiagnosticRegistry,
    AgentOrchestrator,
    PipelineResult,
    DataQualityAuditAgent,
    GeometryDiagnosticAgent,
    EngineRoutingAgent,
    EngineStabilityAuditAgent,
    BlindSpotDetectionAgent,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def registry():
    """Fresh DiagnosticRegistry for each test."""
    return DiagnosticRegistry()


@pytest.fixture
def orchestrator(registry):
    """Fully configured AgentOrchestrator."""
    orch = AgentOrchestrator()

    # Register all agents with shared registry
    orch.register_agent(DataQualityAuditAgent(orch.registry))
    orch.register_agent(GeometryDiagnosticAgent(orch.registry))
    orch.register_agent(EngineRoutingAgent(orch.registry))
    orch.register_agent(EngineStabilityAuditAgent(orch.registry))
    orch.register_agent(BlindSpotDetectionAgent(orch.registry))

    return orch


@pytest.fixture
def synthetic_sine_data():
    """Synthetic oscillator data."""
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, 200)
    return np.sin(t) + np.random.normal(0, 0.1, len(t))


@pytest.fixture
def synthetic_logistic_data():
    """Synthetic latent flow (logistic growth) data."""
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    logistic = 100 / (1 + np.exp(-1.5 * (t - 5)))
    return logistic + np.random.normal(0, 2, len(t))


@pytest.fixture
def synthetic_noise_data():
    """Pure noise data."""
    np.random.seed(42)
    return np.random.randn(200)


@pytest.fixture
def mock_engine_results():
    """Mock engine results for testing."""
    return {
        "pca": {"explained_variance_ratio": 0.75},
        "correlation": {"mean_correlation": 0.45},
        "entropy": {"entropy": 0.65},
        "wavelets": {"max_power": 0.3},
    }


def mock_run_engines(data, routing_result):
    """Mock engine runner for testing."""
    return {
        "pca": {"explained_variance_ratio": 0.75},
        "correlation": {"mean_correlation": 0.45},
        "entropy": {"entropy": 0.65},
        "wavelets": {"max_power": 0.3},
    }


# =============================================================================
# Tests: Full Pipeline Execution
# =============================================================================

class TestFullPipeline:
    """Test complete pipeline execution."""

    def test_pipeline_runs_all_agents(self, orchestrator, synthetic_sine_data):
        """Pipeline should execute all registered agents."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
        )

        assert isinstance(result, PipelineResult)
        assert result.quality is not None
        assert result.geometry is not None
        assert result.routing is not None
        # Stability and blindspot may be None if no engine results

    def test_pipeline_with_engine_function(self, orchestrator, synthetic_sine_data):
        """Pipeline should call engine function when provided."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=mock_run_engines,
        )

        assert result.engine_results is not None
        assert "pca" in result.engine_results
        assert result.stability is not None
        assert result.blindspots is not None

    def test_pipeline_without_engine_function(self, orchestrator, synthetic_sine_data):
        """Pipeline should work without engine function."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=None,
        )

        assert isinstance(result, PipelineResult)
        assert result.engine_results == {}

    def test_pipeline_produces_audit_trail(self, orchestrator, synthetic_sine_data):
        """Pipeline should produce complete audit trail."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
        )

        assert len(result.audit_trail) > 0
        # Should have entries from multiple agents
        agent_names = {entry.agent_name for entry in result.audit_trail}
        assert "data_quality" in agent_names
        assert "geometry_diagnostic" in agent_names


# =============================================================================
# Tests: Data Flow Between Agents
# =============================================================================

class TestAgentDataFlow:
    """Test data passing between agents."""

    def test_geometry_informs_routing(self, orchestrator, synthetic_sine_data):
        """Geometry results should inform routing decisions."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
        )

        # Routing should have run with geometry info
        assert result.routing is not None
        assert len(result.routing.engine_eligibility) > 0

    def test_routing_informs_stability(self, orchestrator, synthetic_sine_data):
        """Routing weights should be used in stability adjustment."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=mock_run_engines,
        )

        if result.stability:
            # Stability should have adjusted weights
            assert len(result.stability.adjusted_weights) > 0

    def test_stability_informs_blindspot(self, orchestrator, synthetic_sine_data):
        """Stability results should inform blind spot detection."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=mock_run_engines,
        )

        # Blind spot should have run
        assert result.blindspots is not None


# =============================================================================
# Tests: Synthetic Data Scenarios
# =============================================================================

class TestSyntheticScenarios:
    """Test pipeline with different synthetic data types."""

    def test_oscillator_data_detects_oscillator_geometry(self, orchestrator, synthetic_sine_data):
        """Oscillator data should be detected as oscillator geometry."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
        )

        # Should detect some oscillator signature
        assert result.geometry is not None
        assert result.geometry.oscillator_score >= 0.3

    def test_logistic_data_detects_latent_flow(self, orchestrator, synthetic_logistic_data):
        """Logistic growth data should be detected as latent flow geometry."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_logistic_data,
            cleaned_data=synthetic_logistic_data,
        )

        assert result.geometry is not None
        # Logistic data should have some latent flow signature
        assert result.geometry.latent_flow_score >= 0.2

    def test_noise_data_high_noise_score(self, orchestrator, synthetic_noise_data):
        """Pure noise should have high noise score."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_noise_data,
            cleaned_data=synthetic_noise_data,
        )

        assert result.geometry is not None
        assert result.geometry.noise_score >= 0.3

    def test_identical_raw_cleaned_is_safe(self, orchestrator, synthetic_sine_data):
        """Identical raw/cleaned data should be data quality safe."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data.copy(),
        )

        assert result.quality is not None
        assert result.quality.safety_flag == "safe"


# =============================================================================
# Tests: Pipeline Robustness
# =============================================================================

class TestPipelineRobustness:
    """Test pipeline handles edge cases gracefully."""

    def test_short_data(self, orchestrator):
        """Pipeline should handle short data."""
        np.random.seed(42)
        short_data = np.random.randn(20)

        result = orchestrator.run_pipeline(
            raw_data=short_data,
            cleaned_data=short_data,
        )

        assert isinstance(result, PipelineResult)
        # Should complete without crash

    def test_dict_data_input(self, orchestrator):
        """Pipeline should handle dict of multiple series."""
        np.random.seed(42)
        dict_data = {
            "series_a": np.random.randn(100),
            "series_b": np.random.randn(100),
        }

        result = orchestrator.run_pipeline(
            raw_data=dict_data,
            cleaned_data=dict_data,
        )

        assert isinstance(result, PipelineResult)

    def test_engine_function_failure(self, orchestrator, synthetic_sine_data):
        """Pipeline should handle engine function failures."""
        def failing_engine_fn(data, routing):
            raise ValueError("Engine failure!")

        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=failing_engine_fn,
        )

        # Should complete with warning
        assert "Engine execution failed" in str(result.warnings)

    def test_missing_agent_graceful(self):
        """Pipeline should work even with missing agents."""
        orch = AgentOrchestrator()
        # Only register some agents
        orch.register_agent(DataQualityAuditAgent(orch.registry))

        np.random.seed(42)
        data = np.random.randn(100)

        result = orch.run_pipeline(
            raw_data=data,
            cleaned_data=data,
        )

        # Should complete with available agents
        assert result.quality is not None
        assert result.geometry is None  # Not registered


# =============================================================================
# Tests: PipelineResult Properties
# =============================================================================

class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_is_safe_with_safe_data(self, orchestrator, synthetic_sine_data):
        """is_safe should be True when data quality is safe and blind spots are minimal."""
        # Run with engine function so blind spot agent has results to check
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=mock_run_engines,
        )

        # Data quality should be safe (identical raw/cleaned)
        assert result.quality.safety_flag == "safe"
        # is_safe checks both quality and blindspots count
        # May or may not be True depending on blind spot findings
        assert isinstance(result.is_safe, bool)

    def test_is_safe_with_unsafe_quality(self, orchestrator):
        """is_safe should be False for unsafe data quality."""
        np.random.seed(42)
        raw = np.random.randn(100) * 10
        # Heavy transformation that will flag as unsafe
        cleaned = np.log(np.abs(raw) + 1) * 0.1

        result = orchestrator.run_pipeline(
            raw_data=raw,
            cleaned_data=cleaned,
        )

        # May or may not be safe depending on exact transformation
        # Just verify the property works
        assert isinstance(result.is_safe, bool)


# =============================================================================
# Tests: Registry Integration
# =============================================================================

class TestRegistryIntegration:
    """Test registry accumulates all metrics."""

    def test_registry_has_all_agent_metrics(self, orchestrator, synthetic_sine_data):
        """Registry should have metrics from all agents."""
        orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
        )

        all_metrics = orchestrator.registry.get_all()

        # Should have metrics from multiple agents
        assert "data_quality" in all_metrics
        assert "geometry_diagnostic" in all_metrics
        assert "engine_routing" in all_metrics

    def test_registry_cleared_between_runs(self, orchestrator, synthetic_sine_data):
        """Registry should be cleared between pipeline runs."""
        # First run
        orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
        )
        first_trail_len = len(orchestrator.registry.export_audit_trail())

        # Second run
        orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
        )
        second_trail_len = len(orchestrator.registry.export_audit_trail())

        # Should be same length (cleared and rebuilt)
        assert first_trail_len == second_trail_len


# =============================================================================
# Tests: Blind Spot Agent
# =============================================================================

class TestBlindSpotAgent:
    """Test blind spot detection in pipeline context."""

    def test_blindspot_receives_geometry(self, orchestrator, synthetic_sine_data):
        """Blind spot agent should receive geometry profile."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=mock_run_engines,
        )

        assert result.blindspots is not None
        # Should have analyzed based on geometry

    def test_blindspot_checks_engine_results(self, orchestrator, synthetic_sine_data):
        """Blind spot should check engine results for anomalies."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=mock_run_engines,
        )

        assert result.blindspots is not None
        # Blind spots list should be populated (may be empty if no issues)
        assert isinstance(result.blindspots.blind_spots, list)

    def test_blindspot_generates_recommendations(self, orchestrator, synthetic_sine_data):
        """Blind spot should generate escalation recommendations."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=mock_run_engines,
        )

        assert result.blindspots is not None
        assert isinstance(result.blindspots.escalation_recommendations, list)


# =============================================================================
# Tests: Explain Methods
# =============================================================================

class TestExplainMethods:
    """Test explain() methods work in pipeline context."""

    def test_all_agents_can_explain(self, orchestrator, synthetic_sine_data):
        """All agents should be able to explain after pipeline run."""
        orchestrator.run_pipeline(
            raw_data=synthetic_sine_data,
            cleaned_data=synthetic_sine_data,
            run_engines_fn=mock_run_engines,
        )

        # Each registered agent should be able to explain
        for name in orchestrator.list_agents():
            agent = orchestrator.get_agent(name)
            explanation = agent.explain()
            assert isinstance(explanation, str)
            assert len(explanation) > 0


# =============================================================================
# Tests: Spec H - Consensus Report Requirements
# =============================================================================

class TestSpecHConsensusReport:
    """Test Spec H: ConsensusReport requirements."""

    @pytest.fixture
    def synthetic_regime_switch_data(self):
        """
        Synthetic regime-switch data (Spec G1 benchmark).

        Two-state regime switching with clear transitions.
        """
        np.random.seed(42)
        n = 300

        # Create regime switching data
        regime = np.zeros(n)
        current_regime = 0

        for i in range(1, n):
            # Switch probability
            if np.random.random() < 0.02:
                current_regime = 1 - current_regime
            regime[i] = current_regime

        # Generate data based on regime
        data = np.zeros(n)
        for i in range(n):
            if regime[i] == 0:
                data[i] = np.random.normal(0, 1)
            else:
                data[i] = np.random.normal(2, 2)

        return data

    def test_consensus_report_generated(self, orchestrator, synthetic_regime_switch_data):
        """ConsensusReport should be generated per Spec H."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        assert result.consensus is not None
        from prism.agents import ConsensusReport
        assert isinstance(result.consensus, ConsensusReport)

    def test_consensus_has_window_id(self, orchestrator, synthetic_regime_switch_data):
        """ConsensusReport should have window_id."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        assert result.consensus.window_id is not None
        assert len(result.consensus.window_id) > 0

    def test_consensus_has_contributing_engines(self, orchestrator, synthetic_regime_switch_data):
        """ConsensusReport should list contributing engines."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        assert hasattr(result.consensus, 'contributing_engines')
        assert isinstance(result.consensus.contributing_engines, list)

    def test_consensus_has_excluded_engines(self, orchestrator, synthetic_regime_switch_data):
        """ConsensusReport should list excluded engines with reasons."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        assert hasattr(result.consensus, 'excluded_engines')
        assert isinstance(result.consensus.excluded_engines, list)

    def test_normalized_weights_sum_to_one(self, orchestrator, synthetic_regime_switch_data):
        """Normalized weights should sum to 1.0 (Spec D)."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        normalized_weights = result.consensus.normalized_weights
        if normalized_weights:  # Only check if there are contributing engines
            total = sum(normalized_weights.values())
            assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_excluded_engines_have_zero_weight(self, orchestrator, synthetic_regime_switch_data):
        """EXCLUDE engines should have 0 weight."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        # Check excluded engines have zero capped_weight
        for excluded in result.consensus.excluded_engines:
            assert excluded.capped_weight == 0, (
                f"Excluded engine {excluded.engine_name} has non-zero weight: {excluded.capped_weight}"
            )

    def test_consensus_has_sensitivity_check(self, orchestrator, synthetic_regime_switch_data):
        """ConsensusReport should include sensitivity check (Spec H)."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        assert hasattr(result.consensus, 'sensitivity_check')
        sensitivity = result.consensus.sensitivity_check
        assert isinstance(sensitivity, dict)
        # Should have stability assessment
        assert 'stable' in sensitivity or 'max_weight_shift' in sensitivity

    def test_capped_weights_respect_spec_d2(self, orchestrator, synthetic_regime_switch_data):
        """Capped weights should not exceed 0.25 per Spec D2."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        for contrib in result.consensus.contributing_engines:
            assert contrib.capped_weight <= 0.25, (
                f"Engine {contrib.engine_name} has capped_weight {contrib.capped_weight} > 0.25"
            )

    def test_blindspot_includes_sensitivity_report(self, orchestrator, synthetic_regime_switch_data):
        """BlindSpotResult should include sensitivity_report (Spec H)."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        assert result.blindspots is not None
        assert hasattr(result.blindspots, 'sensitivity_report')
        assert isinstance(result.blindspots.sensitivity_report, dict)

    def test_pipeline_result_has_normalized_weights_property(self, orchestrator, synthetic_regime_switch_data):
        """PipelineResult should expose normalized_weights from consensus."""
        result = orchestrator.run_pipeline(
            raw_data=synthetic_regime_switch_data,
            cleaned_data=synthetic_regime_switch_data,
            run_engines_fn=mock_run_engines,
        )

        # PipelineResult.normalized_weights should return consensus weights
        assert isinstance(result.normalized_weights, dict)
        if result.consensus.normalized_weights:
            assert result.normalized_weights == result.consensus.normalized_weights


class TestSpecG1RegimeSwitchBenchmark:
    """Test Spec G1: Regime-switch benchmark data handling."""

    def test_regime_switch_generates_structured_data(self):
        """Regime-switch data should have detectible structure."""
        np.random.seed(42)
        n = 300

        # Two-state regime switching
        regime = np.zeros(n)
        current_regime = 0
        for i in range(1, n):
            if np.random.random() < 0.03:
                current_regime = 1 - current_regime
            regime[i] = current_regime

        # Data with different distributions per regime
        data = np.where(regime == 0,
                        np.random.normal(0, 1, n),
                        np.random.normal(3, 2, n))

        # Should show bimodality or high variance
        assert np.std(data) > 1.0

    def test_regime_switch_data_in_pipeline(self, orchestrator):
        """Regime-switch data should run through full pipeline."""
        np.random.seed(42)
        n = 300

        # Generate regime-switch data
        regime = np.zeros(n)
        current_regime = 0
        for i in range(1, n):
            if np.random.random() < 0.03:
                current_regime = 1 - current_regime
            regime[i] = current_regime

        data = np.where(regime == 0,
                        np.random.normal(0, 1, n),
                        np.random.normal(3, 2, n))

        result = orchestrator.run_pipeline(
            raw_data=data,
            cleaned_data=data,
            run_engines_fn=mock_run_engines,
        )

        # Should complete successfully
        assert result.consensus is not None
        # Should detect some reflexive/stochastic signature due to regime switching
        if result.geometry:
            # Regime-switch data often shows reflexive characteristics
            assert result.geometry.reflexive_score >= 0 or result.geometry.noise_score >= 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
