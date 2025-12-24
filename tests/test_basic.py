"""
Basic tests for PRISM Engines.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import prism_engines as prism
from prism_engines import (
    load_csv,
    run_engines,
    Results,
    CorrelationEngine,
    PCAEngine,
    HurstEngine,
    list_engines,
    get_engine,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "A": np.cumsum(np.random.randn(n)),
        "B": np.cumsum(np.random.randn(n)),
        "C": np.cumsum(np.random.randn(n)),
        "D": np.cumsum(np.random.randn(n)),
    })


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    """Create a temporary CSV file."""
    path = tmp_path / "test_data.csv"
    sample_df.to_csv(path)
    return path


# =============================================================================
# Data Loading Tests
# =============================================================================

def test_load_csv(sample_csv):
    """Test CSV loading."""
    df = load_csv(sample_csv)
    assert not df.empty
    assert len(df.columns) == 4


def test_load_csv_not_found():
    """Test error on missing file."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent.csv")


# =============================================================================
# Engine Tests
# =============================================================================

def test_list_engines():
    """Test engine listing."""
    engines = list_engines()
    assert "correlation" in engines
    assert "pca" in engines
    assert "hurst" in engines


def test_get_engine():
    """Test engine retrieval."""
    engine = get_engine("pca")
    assert isinstance(engine, PCAEngine)


def test_correlation_engine(sample_df):
    """Test correlation engine."""
    engine = CorrelationEngine()
    result = engine.run(sample_df)

    assert result.success
    assert "correlation_matrix" in result.metrics
    assert "mean_abs_correlation" in result.metrics
    assert 0 <= result.metrics["mean_abs_correlation"] <= 1


def test_pca_engine(sample_df):
    """Test PCA engine."""
    engine = PCAEngine()
    result = engine.run(sample_df)

    assert result.success
    assert "explained_variance_ratio" in result.metrics
    assert "effective_dimension" in result.metrics
    assert "global_forcing_metric" in result.metrics
    assert result.metrics["effective_dimension"] >= 1


def test_hurst_engine(sample_df):
    """Test Hurst engine."""
    engine = HurstEngine()
    result = engine.run(sample_df)

    assert result.success
    assert "hurst_exponents" in result.metrics
    assert "mean_hurst" in result.metrics


# =============================================================================
# Runner Tests
# =============================================================================

def test_run_engines_df(sample_df):
    """Test running engines on DataFrame."""
    results = run_engines(sample_df)

    assert isinstance(results, Results)
    assert "correlation" in results.engine_results
    assert "pca" in results.engine_results
    assert "hurst" in results.engine_results


def test_run_engines_csv(sample_csv):
    """Test running engines on CSV path."""
    results = run_engines(sample_csv)
    assert len(results.engine_results) == 3


def test_run_specific_engines(sample_df):
    """Test running specific engines only."""
    results = run_engines(sample_df, engines=["pca"])
    assert len(results.engine_results) == 1
    assert "pca" in results.engine_results


def test_results_report(sample_df):
    """Test report generation."""
    results = run_engines(sample_df)
    report = results.report()

    assert "PRISM ENGINES" in report
    assert "CORRELATION" in report
    assert "PCA" in report


# =============================================================================
# Main API Tests
# =============================================================================

def test_prism_run(sample_csv):
    """Test main prism.run() API."""
    results = prism.run(sample_csv)
    assert isinstance(results, Results)
    assert results["pca"].success


def test_version():
    """Test version is set."""
    assert hasattr(prism, "__version__")
    assert prism.__version__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
