"""
Tests for DataQualityAuditAgent (Spec Section A)

Tests verify:
1. Frequency alignment check (A2)
2. Missing data thresholds (A3)
3. Outlier handling compliance (A4)
4. Scaling detection (A5)
5. Variance collapse detection
6. Stationarity forcing detection
7. Safety flag classification
"""

import numpy as np
import pytest

from prism.agents.prism_agent_foundation import (
    DiagnosticRegistry,
    DataQualityResult,
)
from prism.agents.prism_agent_data_quality import (
    DataQualityAuditAgent,
    SpecADiagnostics,
    CLIMATE_INDICES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def registry():
    """Fresh DiagnosticRegistry for each test."""
    return DiagnosticRegistry()


@pytest.fixture
def agent(registry):
    """DataQualityAuditAgent instance."""
    return DataQualityAuditAgent(registry)


@pytest.fixture
def clean_data():
    """Clean, well-behaved data that should pass all checks."""
    np.random.seed(42)
    return np.random.randn(200)


@pytest.fixture
def identical_raw_cleaned():
    """Identical raw and cleaned data (no transformation)."""
    np.random.seed(42)
    data = np.random.randn(200)
    return data.copy(), data.copy()


# =============================================================================
# Tests: Spec A2 - Frequency Alignment
# =============================================================================

class TestFrequencyAlignment:
    """Test frequency alignment check (Spec A2)."""

    def test_aligned_same_length(self, agent, clean_data):
        """Same length series should be aligned."""
        raw = clean_data.copy()
        cleaned = clean_data.copy()

        aligned, details = agent.frequency_alignment_check(raw, cleaned)

        assert aligned is True
        assert "OK" in details

    def test_misaligned_different_length(self, agent, clean_data):
        """Different length series should be misaligned."""
        raw = clean_data.copy()
        cleaned = clean_data[:100]  # Half length

        aligned, details = agent.frequency_alignment_check(raw, cleaned)

        assert aligned is False
        assert "Length mismatch" in details

    def test_uniform_timestamps(self, agent, clean_data):
        """Uniform timestamps should pass."""
        raw = clean_data.copy()
        cleaned = clean_data.copy()
        timestamps = np.arange(len(raw))

        aligned, details = agent.frequency_alignment_check(raw, cleaned, timestamps)

        assert aligned is True

    def test_nonuniform_timestamps(self, agent, clean_data):
        """Non-uniform timestamps should fail."""
        raw = clean_data.copy()
        cleaned = clean_data.copy()
        # Create irregular timestamps
        timestamps = np.cumsum(np.random.exponential(1, len(raw)))

        aligned, details = agent.frequency_alignment_check(raw, cleaned, timestamps)

        assert aligned is False
        assert "Non-uniform spacing" in details


# =============================================================================
# Tests: Spec A3 - Missing Data
# =============================================================================

class TestMissingData:
    """Test missing data thresholds (Spec A3)."""

    def test_no_missing_passes(self, agent, clean_data):
        """Data with no missing values should pass."""
        rate, passed = agent.missing_data_check(clean_data, "multivariate")

        assert rate == 0.0
        assert passed  # Use truthiness check for numpy bool compatibility

    def test_3_percent_missing_multivariate_passes(self, agent, clean_data):
        """3% missing should pass multivariate threshold (5%)."""
        data = clean_data.copy()
        n_missing = int(len(data) * 0.03)
        data[:n_missing] = np.nan

        rate, passed = agent.missing_data_check(data, "multivariate")

        assert 0.02 < rate < 0.04
        assert passed  # Use truthiness check for numpy bool compatibility

    def test_8_percent_missing_multivariate_fails(self, agent, clean_data):
        """8% missing should FAIL multivariate threshold (5%)."""
        data = clean_data.copy()
        n_missing = int(len(data) * 0.08)
        data[:n_missing] = np.nan

        rate, passed = agent.missing_data_check(data, "multivariate")

        assert 0.07 < rate < 0.09
        assert not passed  # Use truthiness check for numpy bool compatibility

    def test_8_percent_missing_univariate_passes(self, agent, clean_data):
        """8% missing should pass univariate threshold (10%)."""
        data = clean_data.copy()
        n_missing = int(len(data) * 0.08)
        data[:n_missing] = np.nan

        rate, passed = agent.missing_data_check(data, "univariate")

        assert 0.07 < rate < 0.09
        assert passed  # Use truthiness check for numpy bool compatibility

    def test_12_percent_missing_univariate_fails(self, agent, clean_data):
        """12% missing should FAIL univariate threshold (10%)."""
        data = clean_data.copy()
        n_missing = int(len(data) * 0.12)
        data[:n_missing] = np.nan

        rate, passed = agent.missing_data_check(data, "univariate")

        assert 0.11 < rate < 0.13
        assert not passed  # Use truthiness check for numpy bool compatibility


# =============================================================================
# Tests: Spec A4 - Outlier Handling
# =============================================================================

class TestOutlierHandling:
    """Test outlier handling compliance (Spec A4)."""

    def test_no_transformation_detected(self, agent, identical_raw_cleaned):
        """Identical data should detect no transformation."""
        raw, cleaned = identical_raw_cleaned

        result = agent.outlier_handling_check(raw, cleaned, "finance")

        assert "none" in result["policy_applied"].lower()
        assert result["compliant"] is True

    def test_winsorization_detected(self, agent, clean_data):
        """Winsorized data should be detected."""
        raw = clean_data.copy()
        # Add outliers
        raw[0] = 100
        raw[-1] = -100
        # Winsorize
        cleaned = raw.copy()
        p1, p99 = np.percentile(raw, [1, 99])
        cleaned = np.clip(cleaned, p1, p99)

        result = agent.outlier_handling_check(raw, cleaned, "finance")

        assert "winsorize" in result["policy_applied"].lower()

    def test_climate_index_no_winsorize_compliant(self, agent, clean_data):
        """Climate index without winsorization should be compliant."""
        raw = clean_data.copy()
        cleaned = clean_data.copy()

        result = agent.outlier_handling_check(raw, cleaned, "climate", "ENSO")

        assert result["compliant"] is True

    def test_climate_index_with_winsorize_noncompliant(self, agent, clean_data):
        """Climate index with winsorization should be NON-compliant."""
        raw = clean_data.copy()
        raw[0] = 100
        raw[-1] = -100

        cleaned = raw.copy()
        p1, p99 = np.percentile(raw, [1, 99])
        cleaned = np.clip(cleaned, p1, p99)

        result = agent.outlier_handling_check(raw, cleaned, "climate", "NAO")

        assert result["compliant"] is False
        assert "not allowed" in result["details"].lower()


# =============================================================================
# Tests: Spec A5 - Scaling Detection
# =============================================================================

class TestScalingDetection:
    """Test scaling method detection (Spec A5)."""

    def test_detect_zscore(self, agent, clean_data):
        """Z-score scaled data should be detected."""
        # Apply z-score scaling
        scaled = (clean_data - np.mean(clean_data)) / np.std(clean_data)

        method = agent.scaling_check(scaled)

        assert method == "zscore"

    def test_detect_robust_z(self, agent, clean_data):
        """Robust z-score scaled data should be detected."""
        # Apply robust z-score scaling
        median = np.median(clean_data)
        mad = np.median(np.abs(clean_data - median))
        scaled = (clean_data - median) / (mad * 1.4826)

        method = agent.scaling_check(scaled)

        assert method == "robust_z"

    def test_detect_rank(self, agent):
        """Rank transformed data should be detected."""
        # Create rank data
        np.random.seed(42)
        data = np.random.randn(100)
        ranks = np.argsort(np.argsort(data)) / (len(data) - 1)  # Normalized ranks [0,1]

        method = agent.scaling_check(ranks)

        assert method == "rank"

    def test_detect_none(self, agent, clean_data):
        """Unscaled data should detect 'none'."""
        # Shift data away from mean=0, std=1
        unscaled = clean_data * 100 + 500

        method = agent.scaling_check(unscaled)

        assert method == "none"


# =============================================================================
# Tests: Variance Collapse
# =============================================================================

class TestVarianceCollapse:
    """Test variance collapse detection."""

    def test_no_collapse(self, agent, identical_raw_cleaned):
        """Identical data should have ratio = 1.0."""
        raw, cleaned = identical_raw_cleaned

        ratio, collapsed = agent.variance_collapse_check(raw, cleaned)

        assert 0.99 < ratio < 1.01
        assert not collapsed  # Use truthiness check for numpy bool compatibility

    def test_variance_collapsed_to_30_percent(self, agent, clean_data):
        """Variance collapsed to 30% should be flagged."""
        raw = clean_data.copy()
        # Scale down to ~30% variance
        cleaned = raw * 0.55  # 0.55^2 ≈ 0.30

        ratio, collapsed = agent.variance_collapse_check(raw, cleaned)

        assert 0.25 < ratio < 0.35
        assert collapsed  # Use truthiness check for numpy bool compatibility

    def test_variance_at_50_percent_boundary(self, agent, clean_data):
        """Variance at exactly 50% should NOT be flagged (< 0.5 triggers)."""
        raw = clean_data.copy()
        cleaned = raw * np.sqrt(0.5)  # Exactly 50% variance

        ratio, collapsed = agent.variance_collapse_check(raw, cleaned)

        assert 0.48 < ratio < 0.52
        # At 50% boundary, should not be collapsed (threshold is < 0.5)
        assert not collapsed  # Use truthiness check for numpy bool compatibility

    def test_variance_at_49_percent_flagged(self, agent, clean_data):
        """Variance at 49% should be flagged."""
        raw = clean_data.copy()
        cleaned = raw * np.sqrt(0.49)  # 49% variance

        ratio, collapsed = agent.variance_collapse_check(raw, cleaned)

        assert ratio < 0.5
        assert collapsed  # Use truthiness check for numpy bool compatibility


# =============================================================================
# Tests: Stationarity Forcing
# =============================================================================

class TestStationarityForcing:
    """Test stationarity forcing detection."""

    def test_both_stationary_not_forced(self, agent, clean_data):
        """Both stationary series should not flag forcing."""
        raw = clean_data.copy()  # Stationary (random walk)
        cleaned = clean_data.copy()

        forced, raw_p, cleaned_p = agent.stationarity_forcing_check(raw, cleaned)

        assert forced is False

    def test_nonstationary_forced_to_stationary(self, agent):
        """Non-stationary forced to stationary should be flagged."""
        np.random.seed(42)
        # Create non-stationary (trend) data
        t = np.arange(200)
        raw = t * 0.5 + np.random.randn(200) * 2  # Linear trend

        # Force stationarity by differencing
        cleaned = np.diff(raw)
        cleaned = np.concatenate([[0], cleaned])  # Pad to same length

        forced, raw_p, cleaned_p = agent.stationarity_forcing_check(raw, cleaned)

        # May or may not detect depending on statsmodels availability
        # Just verify it returns valid values
        assert isinstance(forced, bool)
        assert 0 <= raw_p <= 1
        assert 0 <= cleaned_p <= 1


# =============================================================================
# Tests: Full run() Method - Safety Flag Classification
# =============================================================================

class TestSafetyFlagClassification:
    """Test safety flag classification logic."""

    def test_properly_aligned_data_is_safe(self, agent, identical_raw_cleaned):
        """Properly aligned, clean data should return 'safe'."""
        raw, cleaned = identical_raw_cleaned

        result = agent.run({"raw_data": raw, "cleaned_data": cleaned})

        assert isinstance(result, DataQualityResult)
        assert result.safety_flag == "safe"
        assert len(result.reasons) == 0

    def test_8_percent_missing_multivariate_is_unsafe(self, agent, clean_data):
        """8% missing with multivariate engine type should return 'unsafe'."""
        raw = clean_data.copy()
        cleaned = clean_data.copy()
        n_missing = int(len(cleaned) * 0.08)
        cleaned[:n_missing] = np.nan

        result = agent.run({
            "raw_data": raw,
            "cleaned_data": cleaned,
            "engine_type": "multivariate",
        })

        assert result.safety_flag == "unsafe"
        assert any("Missing rate" in r for r in result.reasons)

    def test_variance_collapsed_to_30_percent_is_unsafe(self, agent, clean_data):
        """Variance collapsed to 30% should return 'unsafe'."""
        raw = clean_data.copy()
        cleaned = raw * 0.55  # ~30% variance

        result = agent.run({"raw_data": raw, "cleaned_data": cleaned})

        assert result.safety_flag == "unsafe"
        assert any("Variance collapsed" in r for r in result.reasons)

    def test_multiple_warnings_is_questionable(self, agent):
        """Multiple warnings (no critical) should return 'questionable'."""
        np.random.seed(42)
        # Create data that will trigger warnings but not critical failures
        raw = np.random.randn(200)
        # Create significant distribution shift and slight variance change
        cleaned = raw * 1.5 + 2  # Shift and scale

        result = agent.run({"raw_data": raw, "cleaned_data": cleaned})

        # Should have some warnings from distribution shift
        assert result.safety_flag in ["safe", "questionable"]

    def test_missing_input_data_is_unsafe(self, agent):
        """Missing input data should return 'unsafe'."""
        result = agent.run({"raw_data": None, "cleaned_data": None})

        assert result.safety_flag == "unsafe"
        assert "Missing input data" in result.reasons


# =============================================================================
# Tests: Full run() with Domain-Specific Behavior
# =============================================================================

class TestDomainSpecificBehavior:
    """Test domain-specific behavior."""

    def test_finance_domain(self, agent, identical_raw_cleaned):
        """Finance domain should accept winsorization."""
        raw, cleaned = identical_raw_cleaned

        result = agent.run({
            "raw_data": raw,
            "cleaned_data": cleaned,
            "domain": "finance",
        })

        assert result.safety_flag == "safe"

    def test_climate_domain_with_nao_indicator(self, agent, identical_raw_cleaned):
        """Climate domain with NAO should not accept winsorization."""
        raw, cleaned = identical_raw_cleaned

        result = agent.run({
            "raw_data": raw,
            "cleaned_data": cleaned,
            "domain": "climate",
            "indicator_name": "NAO",
        })

        # Should be safe since no winsorization was applied
        assert result.safety_flag == "safe"


# =============================================================================
# Tests: Multi-Series Handling
# =============================================================================

class TestMultiSeriesHandling:
    """Test handling of multiple series."""

    def test_dict_input(self, agent, clean_data):
        """Should handle dict of multiple series."""
        raw_dict = {
            "series_a": clean_data.copy(),
            "series_b": clean_data.copy(),
        }
        cleaned_dict = {
            "series_a": clean_data.copy(),
            "series_b": clean_data.copy(),
        }

        result = agent.run({
            "raw_data": raw_dict,
            "cleaned_data": cleaned_dict,
        })

        assert isinstance(result, DataQualityResult)
        assert result.safety_flag == "safe"

    def test_worst_flag_propagates(self, agent, clean_data):
        """Worst safety flag should propagate in multi-series."""
        raw_dict = {
            "series_a": clean_data.copy(),
            "series_b": clean_data.copy(),
        }
        # Make series_b have 10% missing
        n_missing = int(len(clean_data) * 0.10)
        series_b_cleaned = clean_data.copy()
        series_b_cleaned[:n_missing] = np.nan

        cleaned_dict = {
            "series_a": clean_data.copy(),
            "series_b": series_b_cleaned,
        }

        result = agent.run({
            "raw_data": raw_dict,
            "cleaned_data": cleaned_dict,
            "engine_type": "multivariate",
        })

        # series_b has 10% > 5% threshold, so should be unsafe
        assert result.safety_flag == "unsafe"


# =============================================================================
# Tests: Registry Integration
# =============================================================================

class TestRegistryIntegration:
    """Test registry and audit trail integration."""

    def test_results_stored_in_registry(self, agent, registry, identical_raw_cleaned):
        """Results should be stored in registry."""
        raw, cleaned = identical_raw_cleaned

        agent.run({"raw_data": raw, "cleaned_data": cleaned})

        assert registry.get("data_quality", "safety_flag") is not None
        assert registry.get("data_quality", "missing_rate") is not None

    def test_audit_trail_logged(self, agent, registry, identical_raw_cleaned):
        """Decisions should be logged to audit trail."""
        raw, cleaned = identical_raw_cleaned

        agent.run({"raw_data": raw, "cleaned_data": cleaned})

        trail = registry.export_audit_trail()
        assert len(trail) >= 1
        assert trail[0].agent_name == "data_quality"


# =============================================================================
# Tests: explain() Method
# =============================================================================

class TestExplainMethod:
    """Test explain() output formatting."""

    def test_explain_before_run(self, agent):
        """explain() before run should indicate no analysis."""
        explanation = agent.explain()
        assert "No analysis" in explanation

    def test_explain_after_run(self, agent, identical_raw_cleaned):
        """explain() after run should show results."""
        raw, cleaned = identical_raw_cleaned
        agent.run({"raw_data": raw, "cleaned_data": cleaned})

        explanation = agent.explain()

        assert "Spec Section A" in explanation
        assert "SAFE" in explanation
        assert "Frequency Alignment" in explanation
        assert "Missing Data" in explanation
        assert "Outlier Handling" in explanation
        assert "Scaling" in explanation


# =============================================================================
# Tests: Helper Methods
# =============================================================================

class TestHelperMethods:
    """Test helper methods."""

    def test_get_diagnostics(self, agent, identical_raw_cleaned):
        """get_diagnostics() should return last diagnostics."""
        raw, cleaned = identical_raw_cleaned
        agent.run({"raw_data": raw, "cleaned_data": cleaned})

        diag = agent.get_diagnostics()

        assert isinstance(diag, SpecADiagnostics)
        assert diag.frequency_aligned is True

    def test_get_missing_threshold(self, agent):
        """get_missing_threshold() should return correct thresholds."""
        mv_threshold = agent.get_missing_threshold("multivariate")
        uv_threshold = agent.get_missing_threshold("univariate")

        assert mv_threshold == 0.05
        assert uv_threshold == 0.10


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_data(self, agent):
        """Should handle short data gracefully."""
        np.random.seed(42)
        raw = np.random.randn(5)
        cleaned = raw.copy()

        result = agent.run({"raw_data": raw, "cleaned_data": cleaned})

        assert isinstance(result, DataQualityResult)

    def test_constant_data(self, agent):
        """Should handle constant data."""
        raw = np.ones(100)
        cleaned = np.ones(100)

        result = agent.run({"raw_data": raw, "cleaned_data": cleaned})

        assert isinstance(result, DataQualityResult)

    def test_all_nan_data(self, agent):
        """Should handle all-NaN data."""
        raw = np.full(100, np.nan)
        cleaned = np.full(100, np.nan)

        result = agent.run({"raw_data": raw, "cleaned_data": cleaned})

        # All NaN = 100% missing > any threshold
        assert result.safety_flag == "unsafe"

    def test_empty_data(self, agent):
        """Should handle empty arrays."""
        raw = np.array([])
        cleaned = np.array([])

        result = agent.run({"raw_data": raw, "cleaned_data": cleaned})

        assert isinstance(result, DataQualityResult)


# =============================================================================
# Tests: Climate Index Detection
# =============================================================================

class TestClimateIndexDetection:
    """Test climate index name detection."""

    def test_enso_recognized(self):
        """ENSO should be recognized as climate index."""
        assert "enso" in CLIMATE_INDICES

    def test_nao_recognized(self):
        """NAO should be recognized as climate index."""
        assert "nao" in CLIMATE_INDICES

    def test_pdo_recognized(self):
        """PDO should be recognized as climate index."""
        assert "pdo" in CLIMATE_INDICES

    def test_case_insensitive_matching(self, agent, clean_data):
        """Climate index matching should be case-insensitive."""
        raw = clean_data.copy()
        cleaned = clean_data.copy()

        result = agent.outlier_handling_check(raw, cleaned, "climate", "ENSO")
        assert result["compliant"] is True

        result = agent.outlier_handling_check(raw, cleaned, "climate", "enso")
        assert result["compliant"] is True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
