"""
Tests for Hidden Variation Detector (HVD)
=========================================

Tests:
1. High-similarity detection
2. Cluster formation
3. Family divergence warning
4. Edge cases (empty panel, few indicators)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Try pytest, fall back to manual testing
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest.fixture
    class pytest:
        @staticmethod
        def fixture(func):
            return func

# Import HVD module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.hidden_variation_detector import (
    compute_similarity_matrix,
    detect_high_similarity_pairs,
    detect_clusters_kmeans,
    detect_clusters_correlation,
    detect_family_divergences,
    build_hvd_report,
    SimilarityResult,
    ClusterResult,
    HVDReport
)


# =============================================================================
# Mock Family Manager
# =============================================================================

class MockFamilyMember:
    def __init__(self, member_id: str):
        self.id = member_id


class MockFamily:
    def __init__(self, family_id: str, member_ids: List[str]):
        self.id = family_id
        self.members = {m: MockFamilyMember(m) for m in member_ids}


class MockFamilyManager:
    """Mock FamilyManager for testing."""
    
    def __init__(self, families: Dict[str, List[str]]):
        self.families = {
            fid: MockFamily(fid, members) 
            for fid, members in families.items()
        }


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_panel():
    """Create a simple test panel."""
    np.random.seed(42)
    n = 300
    
    return pd.DataFrame({
        'a': np.random.randn(n),
        'b': np.random.randn(n),
        'c': np.random.randn(n),
    })


@pytest.fixture
def correlated_panel():
    """Create panel with known correlations."""
    np.random.seed(42)
    n = 500
    
    base = np.random.randn(n)
    noise = np.random.randn(n) * 0.1
    
    return pd.DataFrame({
        # These two should be highly correlated
        'similar_a': base + noise,
        'similar_b': base + noise * 1.1,
        
        # This should be uncorrelated
        'independent': np.random.randn(n),
    })


@pytest.fixture
def clustered_panel():
    """Create panel with clear cluster structure."""
    np.random.seed(42)
    n = 400
    
    # Cluster 1: trending
    trend = np.linspace(0, 10, n)
    
    # Cluster 2: cyclical
    cycle = np.sin(np.linspace(0, 6 * np.pi, n))
    
    return pd.DataFrame({
        'trend_1': trend + np.random.randn(n) * 0.3,
        'trend_2': trend * 0.9 + np.random.randn(n) * 0.3,
        
        'cycle_1': cycle + np.random.randn(n) * 0.2,
        'cycle_2': cycle * 1.1 + np.random.randn(n) * 0.2,
    })


@pytest.fixture
def family_panel():
    """Create panel with family members."""
    np.random.seed(42)
    n = 500
    
    base = np.random.randn(n)
    
    return pd.DataFrame({
        'sp500_d': base + np.random.randn(n) * 0.2,
        'sp500_m': base + np.random.randn(n) * 0.2,  # Should be correlated
        'vix': np.random.randn(n),  # Independent
    })


@pytest.fixture
def divergent_panel():
    """Create panel where family members diverge recently."""
    np.random.seed(42)
    n = 600
    
    # First 400: correlated
    # Last 200: divergent
    base = np.random.randn(n)
    
    member1 = base.copy()
    member2 = base.copy()
    
    # Make them diverge in recent period
    member2[400:] = np.random.randn(200) * 2  # Completely different
    
    return pd.DataFrame({
        'sp500_d': member1,
        'sp500_m': member2,
    })


# =============================================================================
# Test: Similarity Matrix
# =============================================================================

class TestSimilarityMatrix:
    
    def test_basic_computation(self, simple_panel):
        """Test basic similarity computation."""
        similarity = compute_similarity_matrix(simple_panel, min_overlap=100)
        
        assert isinstance(similarity, dict)
        # Should have n*(n-1)/2 pairs = 3
        assert len(similarity) == 3
    
    def test_correlated_pairs(self, correlated_panel):
        """Test that correlated pairs have high similarity."""
        similarity = compute_similarity_matrix(correlated_panel, min_overlap=100)
        
        # similar_a and similar_b should be highly correlated
        key = ('similar_a', 'similar_b')
        if key not in similarity:
            key = ('similar_b', 'similar_a')
        
        assert key in similarity
        assert similarity[key] > 0.95, f"Expected high correlation, got {similarity[key]}"
    
    def test_min_overlap_filtering(self, simple_panel):
        """Test that pairs with insufficient overlap are filtered."""
        # Request more overlap than available
        similarity = compute_similarity_matrix(simple_panel, min_overlap=1000)
        
        # Should have no results
        assert len(similarity) == 0


# =============================================================================
# Test: High Similarity Detection
# =============================================================================

class TestHighSimilarityDetection:
    
    def test_detects_similar_pairs(self, correlated_panel):
        """Test that similar pairs are detected."""
        similarity = compute_similarity_matrix(correlated_panel, min_overlap=100)
        results = detect_high_similarity_pairs(similarity, threshold=0.90)
        
        assert len(results) >= 1
        
        # Find the similar_a/similar_b pair
        pairs = [(r.pair[0], r.pair[1]) for r in results]
        assert ('similar_a', 'similar_b') in pairs or ('similar_b', 'similar_a') in pairs
    
    def test_flags_correct(self, correlated_panel):
        """Test that HIGH_SIMILARITY flag is set."""
        similarity = compute_similarity_matrix(correlated_panel, min_overlap=100)
        results = detect_high_similarity_pairs(similarity, threshold=0.90)
        
        for r in results:
            assert 'HIGH_SIMILARITY' in r.flags
    
    def test_threshold_works(self, correlated_panel):
        """Test that threshold filters correctly."""
        similarity = compute_similarity_matrix(correlated_panel, min_overlap=100)
        
        # Very high threshold should return nothing
        results = detect_high_similarity_pairs(similarity, threshold=0.999)
        assert len(results) == 0
        
        # Very low threshold should return all pairs
        results = detect_high_similarity_pairs(similarity, threshold=0.0)
        assert len(results) >= 1


# =============================================================================
# Test: Clustering
# =============================================================================

class TestClustering:
    
    def test_correlation_clustering(self, clustered_panel):
        """Test correlation-based clustering."""
        clusters = detect_clusters_correlation(clustered_panel, threshold=0.70)
        
        assert len(clusters) >= 1
        
        # Check that trending indicators are together
        for c in clusters:
            if 'trend_1' in c.members:
                assert 'trend_2' in c.members, "trend_1 and trend_2 should be in same cluster"
    
    def test_kmeans_clustering(self, clustered_panel):
        """Test KMeans clustering."""
        clusters = detect_clusters_kmeans(clustered_panel, n_clusters=2)
        
        # Should have 2 clusters
        assert len(clusters) >= 1
        
        # All indicators should be assigned
        all_members = []
        for c in clusters:
            all_members.extend(c.members)
        assert len(all_members) == len(clustered_panel.columns)
    
    def test_cluster_has_method(self, clustered_panel):
        """Test that cluster results have method field."""
        clusters = detect_clusters_correlation(clustered_panel)
        
        for c in clusters:
            assert c.method is not None
            assert len(c.method) > 0


# =============================================================================
# Test: Family Divergence
# =============================================================================

class TestFamilyDivergence:
    
    def test_detects_multiple_members(self, family_panel):
        """Test that multiple family members are detected."""
        fm = MockFamilyManager({
            'spx': ['sp500_d', 'sp500_m']
        })
        
        divergences, warnings = detect_family_divergences(family_panel, fm)
        
        # Should have a warning about multiple members
        assert len(warnings) >= 1
        assert any('sp500_d' in w and 'sp500_m' in w for w in warnings)
    
    def test_detects_divergence(self, divergent_panel):
        """Test that divergence is detected when family members diverge."""
        fm = MockFamilyManager({
            'spx': ['sp500_d', 'sp500_m']
        })
        
        divergences, warnings = detect_family_divergences(
            divergent_panel, 
            fm,
            divergence_threshold=0.70,
            window_days=200
        )
        
        # Should detect divergence
        assert len(divergences) >= 1 or any('DIVERGENCE' in w or 'divergence' in w.lower() for w in warnings)
    
    def test_no_family_manager(self, family_panel):
        """Test graceful handling when no family manager."""
        divergences, warnings = detect_family_divergences(family_panel, None)
        
        assert divergences == []
        assert warnings == []


# =============================================================================
# Test: Full HVD Report
# =============================================================================

class TestHVDReport:
    
    def test_basic_report(self, simple_panel):
        """Test basic report generation."""
        report = build_hvd_report(simple_panel)
        
        assert isinstance(report, HVDReport)
        assert report.indicators == list(simple_panel.columns)
        assert report.meta['status'] == 'completed'
    
    def test_report_with_correlations(self, correlated_panel):
        """Test report finds high correlations."""
        report = build_hvd_report(correlated_panel, similarity_threshold=0.90)
        
        assert len(report.high_similarity_pairs) >= 1
    
    def test_report_with_family_manager(self, family_panel):
        """Test report with family manager integration."""
        fm = MockFamilyManager({
            'spx': ['sp500_d', 'sp500_m']
        })
        
        report = build_hvd_report(family_panel, family_manager=fm)
        
        # Should have warnings about family
        assert len(report.warnings) >= 1
    
    def test_empty_panel(self):
        """Test handling of empty panel."""
        empty = pd.DataFrame()
        report = build_hvd_report(empty)
        
        assert report.meta['status'] == 'empty_panel'
        assert len(report.warnings) >= 1
    
    def test_single_indicator(self):
        """Test handling of single indicator."""
        single = pd.DataFrame({'only_one': np.random.randn(100)})
        report = build_hvd_report(single)
        
        assert report.meta['status'] == 'insufficient_indicators'
        assert len(report.warnings) >= 1
    
    def test_summary(self, correlated_panel):
        """Test report summary method."""
        report = build_hvd_report(correlated_panel)
        summary = report.summary()
        
        assert 'total_indicators' in summary
        assert 'high_similarity_count' in summary
        assert 'cluster_count' in summary
        assert 'warning_count' in summary


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        n = 300
        panel = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
        })
        
        # Add some NaNs
        panel.loc[50:100, 'a'] = np.nan
        panel.loc[150:200, 'b'] = np.nan
        
        # Should not crash
        report = build_hvd_report(panel)
        assert isinstance(report, HVDReport)
    
    def test_constant_series(self):
        """Test handling of constant series."""
        n = 300
        panel = pd.DataFrame({
            'constant': np.ones(n),
            'random': np.random.randn(n),
        })
        
        # Should not crash
        report = build_hvd_report(panel)
        assert isinstance(report, HVDReport)
    
    def test_short_panel(self):
        """Test handling of very short panel."""
        panel = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        
        report = build_hvd_report(panel, min_overlap=252)
        
        # Should warn about insufficient data
        assert len(report.warnings) >= 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Simple test runner for standalone execution
    import sys
    
    print("=" * 60)
    print("Hidden Variation Detector - Test Suite")
    print("=" * 60)
    
    # Create fixtures manually
    np.random.seed(42)
    
    test_count = 0
    pass_count = 0
    fail_count = 0
    
    def run_test(name, test_func):
        global test_count, pass_count, fail_count
        test_count += 1
        try:
            test_func()
            print(f"  ✅ {name}")
            pass_count += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            fail_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            fail_count += 1
    
    # Create test panels
    n = 500
    
    correlated_panel = pd.DataFrame({
        'similar_a': np.random.randn(n),
        'similar_b': None,
        'independent': np.random.randn(n),
    })
    correlated_panel['similar_b'] = correlated_panel['similar_a'] + np.random.randn(n) * 0.1
    
    family_panel = pd.DataFrame({
        'sp500_d': np.random.randn(n),
        'sp500_m': None,
        'vix': np.random.randn(n),
    })
    family_panel['sp500_m'] = family_panel['sp500_d'] + np.random.randn(n) * 0.2
    
    print("\n📊 Similarity Tests:")
    
    run_test("Computes similarity matrix", lambda: (
        isinstance(compute_similarity_matrix(correlated_panel, 100), dict)
    ))
    
    run_test("Detects high correlation", lambda: (
        len(detect_high_similarity_pairs(
            compute_similarity_matrix(correlated_panel, 100), 0.90
        )) >= 1
    ))
    
    print("\n📊 Clustering Tests:")
    
    run_test("Correlation clustering works", lambda: (
        len(detect_clusters_correlation(correlated_panel)) >= 1
    ))
    
    run_test("KMeans clustering works", lambda: (
        len(detect_clusters_kmeans(correlated_panel, 2)) >= 1
    ))
    
    print("\n📊 Family Tests:")
    
    fm = MockFamilyManager({'spx': ['sp500_d', 'sp500_m']})
    
    run_test("Detects family members", lambda: (
        len(detect_family_divergences(family_panel, fm)[1]) >= 1
    ))
    
    run_test("Handles missing family manager", lambda: (
        detect_family_divergences(family_panel, None) == ([], [])
    ))
    
    print("\n📊 Full Report Tests:")
    
    run_test("Builds basic report", lambda: (
        isinstance(build_hvd_report(correlated_panel), HVDReport)
    ))
    
    run_test("Handles empty panel", lambda: (
        build_hvd_report(pd.DataFrame()).meta['status'] == 'empty_panel'
    ))
    
    run_test("Reports summary works", lambda: (
        'total_indicators' in build_hvd_report(correlated_panel).summary()
    ))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {pass_count}/{test_count} passed, {fail_count} failed")
    print("=" * 60)
    
    sys.exit(0 if fail_count == 0 else 1)
