"""
Basic tests for prism-engines.

Run with: pytest tests/test_basic.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestDataLoading:
    """Tests for data loading."""
    
    def test_load_csv(self, tmp_path):
        """Test loading a simple CSV."""
        from prism_engines import load_csv
        
        # Create test CSV
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "date,A,B,C\n"
            "2023-01-01,1.0,2.0,3.0\n"
            "2023-01-02,1.1,2.1,3.1\n"
            "2023-01-03,1.2,2.2,3.2\n"
        )
        
        df = load_csv(csv_path)
        
        assert len(df) == 3
        assert list(df.columns) == ["A", "B", "C"]
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_load_csv_missing_file(self):
        """Test error on missing file."""
        from prism_engines import load_csv
        
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")


class TestEngines:
    """Tests for individual engines."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        
        # Create correlated data
        base = np.cumsum(np.random.randn(100))
        
        df = pd.DataFrame({
            "A": base + np.random.randn(100) * 0.5,
            "B": base * 0.8 + np.random.randn(100) * 0.5,
            "C": -base * 0.5 + np.random.randn(100) * 0.5,
            "D": np.random.randn(100),  # Uncorrelated
        }, index=dates)
        
        return df
    
    def test_correlation_engine(self, sample_df):
        """Test correlation engine."""
        from prism_engines import get_engine
        
        engine = get_engine("correlation")
        result = engine.run(sample_df)
        
        assert result.success
        assert "avg_correlation" in result.metrics
        assert "correlation_matrix" in result.details
        
        # A and B should be positively correlated
        corr_matrix = result.details["correlation_matrix"]
        assert corr_matrix.loc["A", "B"] > 0.5
    
    def test_pca_engine(self, sample_df):
        """Test PCA engine."""
        from prism_engines import get_engine
        
        engine = get_engine("pca")
        result = engine.run(sample_df)
        
        assert result.success
        assert "variance_explained_pc1" in result.metrics
        assert "loadings" in result.details
        
        # PC1 should explain significant variance (correlated data)
        assert result.metrics["variance_explained_pc1"] > 0.3
    
    def test_hurst_engine(self, sample_df):
        """Test Hurst engine."""
        from prism_engines import get_engine
        
        engine = get_engine("hurst")
        result = engine.run(sample_df)
        
        assert result.success
        assert "avg_hurst" in result.metrics
        assert "hurst_exponents" in result.details
        
        # Random walk should have H ≈ 0.5
        # Our trending data (cumsum) should have H > 0.5
        h = result.details["hurst_exponents"]
        assert not h.isna().all()


class TestRunner:
    """Tests for the engine runner."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        
        df = pd.DataFrame({
            "X": np.cumsum(np.random.randn(50)),
            "Y": np.cumsum(np.random.randn(50)),
        }, index=dates)
        
        return df
    
    def test_run_all_engines(self, sample_df):
        """Test running all engines."""
        from prism_engines import run_engines
        
        results = run_engines(sample_df)
        
        assert "correlation" in results.engine_results
        assert "pca" in results.engine_results
        assert "hurst" in results.engine_results
    
    def test_run_specific_engines(self, sample_df):
        """Test running specific engines."""
        from prism_engines import run_engines
        
        results = run_engines(sample_df, engines=["pca"])
        
        assert "pca" in results.engine_results
        assert "correlation" not in results.engine_results
    
    def test_results_summary(self, sample_df):
        """Test results summary generation."""
        from prism_engines import run_engines
        
        results = run_engines(sample_df)
        summary = results.summary()
        
        assert "PRISM ENGINES RESULTS" in summary
        assert "CORRELATION" in summary
        assert "PCA" in summary
        assert "HURST" in summary
    
    def test_results_to_csv(self, sample_df, tmp_path):
        """Test exporting results to CSV."""
        from prism_engines import run_engines
        
        results = run_engines(sample_df)
        
        csv_path = tmp_path / "results.csv"
        results.to_csv(str(csv_path))
        
        assert csv_path.exists()
        
        # Load and check
        df = pd.read_csv(csv_path)
        assert "engine" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns


class TestListEngines:
    """Test engine listing."""
    
    def test_list_engines(self):
        """Test listing available engines."""
        from prism_engines import list_engines
        
        engines = list_engines()
        
        assert "correlation" in engines
        assert "pca" in engines
        assert "hurst" in engines
        
        # Check metadata
        assert "description" in engines["correlation"]
        assert "question" in engines["correlation"]
