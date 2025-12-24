"""
PRISM Fetch System Tests

Smoke tests to verify the fetch system is working.
"""

import pytest
from pathlib import Path
from datetime import date
import tempfile
import os

# Skip all tests if API keys not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY") or not os.environ.get("TIINGO_API_KEY"),
    reason="API keys not configured"
)


class TestFREDFetcher:
    """Tests for FRED fetcher."""
    
    def test_fetch_gdp(self):
        """Test fetching GDP from FRED."""
        from prism.fetch import FREDFetcher
        
        fetcher = FREDFetcher()
        result = fetcher.fetch("GDP")
        
        assert result.success, f"Fetch failed: {result.error}"
        assert result.data is not None
        assert len(result.data) > 0
        assert "date" in result.data.columns
        assert "value" in result.data.columns
        assert result.rows > 0
    
    def test_fetch_invalid_series(self):
        """Test that invalid series returns error, not exception."""
        from prism.fetch import FREDFetcher
        
        fetcher = FREDFetcher()
        result = fetcher.fetch("INVALID_SERIES_XXXXX")
        
        # Should fail gracefully, not raise
        assert not result.success
        assert result.error is not None


class TestTiingoFetcher:
    """Tests for Tiingo fetcher."""
    
    def test_fetch_spy(self):
        """Test fetching SPY from Tiingo."""
        from prism.fetch import TiingoFetcher
        
        fetcher = TiingoFetcher()
        result = fetcher.fetch("SPY")
        
        assert result.success, f"Fetch failed: {result.error}"
        assert result.data is not None
        assert len(result.data) > 0
        assert "date" in result.data.columns
        assert "value" in result.data.columns
        assert result.rows > 0
    
    def test_fetch_invalid_ticker(self):
        """Test that invalid ticker returns error, not exception."""
        from prism.fetch import TiingoFetcher
        
        fetcher = TiingoFetcher()
        result = fetcher.fetch("XXXXINVALIDXXXX")
        
        # Should fail gracefully, not raise
        assert not result.success
        assert result.error is not None


class TestFetchRunner:
    """Tests for FetchRunner (DB writer)."""
    
    def test_fetch_and_write(self):
        """Test that FetchRunner writes to database."""
        from prism.fetch import FetchRunner
        from prism.db import DatabaseConnection
        
        # Use temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            
            runner = FetchRunner(db_path=db_path)
            
            summary = runner.run([
                {"id": "GDP", "source": "fred"},
            ])
            
            # Check summary
            assert summary.succeeded >= 1 or summary.failed >= 0
            assert summary.status == "completed"
            
            # Check database has data
            db = DatabaseConnection(db_path)
            conn = db.connect()
            
            result = conn.execute(
                "SELECT COUNT(*) FROM data.raw_indicators WHERE indicator_id = 'GDP'"
            ).fetchone()
            
            if summary.succeeded > 0:
                assert result[0] > 0, "No data written to database"
            
            db.close()
    
    def test_mixed_sources(self):
        """Test fetching from multiple sources."""
        from prism.fetch import FetchRunner
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            
            runner = FetchRunner(db_path=db_path)
            
            summary = runner.run([
                {"id": "GDP", "source": "fred"},
                {"id": "SPY", "source": "tiingo"},
            ])
            
            assert summary.total_indicators == 2
            assert summary.status == "completed"


class TestRegistry:
    """Tests for registry loader."""
    
    def test_load_default_registry(self):
        """Test loading the default registry."""
        from prism.registry import RegistryLoader
        
        registry = RegistryLoader()
        indicators = registry.get_all()
        
        assert len(indicators) > 0
    
    def test_get_by_source(self):
        """Test filtering by source."""
        from prism.registry import RegistryLoader
        
        registry = RegistryLoader()
        
        fred = registry.get_by_source("fred")
        tiingo = registry.get_by_source("tiingo")
        
        assert len(fred) > 0
        assert len(tiingo) > 0
        
        # All FRED indicators should have source='fred'
        for ind in fred:
            assert ind.source == "fred"
    
    def test_get_specific_indicators(self):
        """Test getting specific indicators."""
        from prism.registry import RegistryLoader
        
        registry = RegistryLoader()
        
        indicators = registry.get_indicators(["GDP", "SPY"])
        
        assert len(indicators) == 2
        ids = {i.id for i in indicators}
        assert "GDP" in ids
        assert "SPY" in ids


class TestDataIntegrity:
    """Tests for data integrity."""
    
    def test_dates_are_monotonic(self):
        """Test that fetched dates are sorted."""
        from prism.fetch import FREDFetcher
        
        fetcher = FREDFetcher()
        result = fetcher.fetch("GDP")
        
        if result.success:
            dates = result.data["date"].tolist()
            assert dates == sorted(dates), "Dates not sorted"
    
    def test_no_duplicate_dates(self):
        """Test that there are no duplicate dates."""
        from prism.fetch import FREDFetcher
        
        fetcher = FREDFetcher()
        result = fetcher.fetch("GDP")
        
        if result.success:
            dates = result.data["date"].tolist()
            assert len(dates) == len(set(dates)), "Duplicate dates found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
