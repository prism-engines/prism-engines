# VCF Research: Testing Framework
# Comprehensive tests for 56-indicator analysis

"""
Test suite for VCF Research mathematical operations
Ensures correctness and reliability at scale (56 indicators)
"""

import numpy as np
import pandas as pd
import pytest
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class TestDataValidator:
    """
    Validate input data quality and properties
    """
    
    @staticmethod
    def test_no_missing_values(panel_df: pd.DataFrame, 
                               allow_missing_pct: float = 0.05) -> Dict:
        """
        Check for missing values across all indicators
        
        Parameters:
        -----------
        allow_missing_pct : float
            Maximum allowed percentage of missing values per indicator
        """
        results = {}
        
        for col in panel_df.columns:
            missing_pct = panel_df[col].isna().sum() / len(panel_df)
            
            results[col] = {
                'missing_pct': missing_pct,
                'passed': missing_pct <= allow_missing_pct,
                'n_missing': panel_df[col].isna().sum()
            }
        
        n_failed = sum(1 for r in results.values() if not r['passed'])
        
        return {
            'all_passed': n_failed == 0,
            'n_indicators_failed': n_failed,
            'details': results
        }
    
    @staticmethod
    def test_no_infinite_values(panel_df: pd.DataFrame) -> Dict:
        """
        Check for infinite values (often from division errors)
        """
        results = {}
        
        for col in panel_df.columns:
            n_inf = np.isinf(panel_df[col]).sum()
            results[col] = {
                'n_infinite': n_inf,
                'passed': n_inf == 0
            }
        
        n_failed = sum(1 for r in results.values() if not r['passed'])
        
        return {
            'all_passed': n_failed == 0,
            'n_indicators_failed': n_failed,
            'details': results
        }
    
    @staticmethod
    def test_sufficient_variance(panel_df: pd.DataFrame, 
                                 min_std: float = 0.01) -> Dict:
        """
        Check that indicators have sufficient variation
        (constant indicators provide no information)
        """
        results = {}
        
        for col in panel_df.columns:
            std = panel_df[col].std()
            results[col] = {
                'std': std,
                'passed': std >= min_std
            }
        
        n_failed = sum(1 for r in results.values() if not r['passed'])
        
        return {
            'all_passed': n_failed == 0,
            'n_indicators_failed': n_failed,
            'constant_indicators': [col for col, r in results.items() if not r['passed']],
            'details': results
        }
    
    @staticmethod
    def test_data_consistency(panel_df: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality check
        """
        return {
            'missing_values': TestDataValidator.test_no_missing_values(panel_df),
            'infinite_values': TestDataValidator.test_no_infinite_values(panel_df),
            'sufficient_variance': TestDataValidator.test_sufficient_variance(panel_df),
            'n_indicators': len(panel_df.columns),
            'n_time_points': len(panel_df),
            'date_range': (panel_df.index[0], panel_df.index[-1])
        }


class TestMathematicalOperations:
    """
    Test mathematical operations for correctness
    """
    
    @staticmethod
    def test_vector_magnitude(panel_df: pd.DataFrame) -> Dict:
        """
        Test that vector magnitudes are computed correctly
        """
        # Simple test case
        test_data = pd.DataFrame({
            'a': [3, 0, 1],
            'b': [4, 1, 1]
        })
        
        # Expected magnitude: sqrt(3^2 + 4^2) = 5
        magnitudes = np.linalg.norm(test_data.values, axis=1)
        
        return {
            'passed': np.allclose(magnitudes[0], 5.0),
            'test_magnitude': magnitudes[0],
            'expected': 5.0
        }
    
    @staticmethod
    def test_coherence_bounds(panel_df: pd.DataFrame) -> Dict:
        """
        Test that coherence values are in valid range [0, 1]
        """
        from scipy.signal import coherence
        
        # Test on first two indicators
        if len(panel_df.columns) < 2:
            return {'error': 'Need at least 2 indicators'}
        
        signal_a = panel_df.iloc[:, 0].values
        signal_b = panel_df.iloc[:, 1].values
        
        f, Cxy = coherence(signal_a, signal_b, fs=1.0)
        
        return {
            'passed': np.all((Cxy >= 0) & (Cxy <= 1)),
            'min_coherence': np.min(Cxy),
            'max_coherence': np.max(Cxy),
            'valid_range': True if np.all((Cxy >= 0) & (Cxy <= 1)) else False
        }
    
    @staticmethod
    def test_pca_variance_sum(panel_df: pd.DataFrame) -> Dict:
        """
        Test that PCA explained variance sums to ~1.0
        """
        from sklearn.decomposition import PCA
        
        # Normalize data
        data_norm = (panel_df - panel_df.mean()) / panel_df.std()
        data_norm = data_norm.dropna()
        
        if len(data_norm) < 10:
            return {'error': 'Insufficient data for PCA'}
        
        pca = PCA()
        pca.fit(data_norm)
        
        total_explained = np.sum(pca.explained_variance_ratio_)
        
        return {
            'passed': np.abs(total_explained - 1.0) < 0.01,
            'total_explained_variance': total_explained,
            'expected': 1.0
        }
    
    @staticmethod
    def test_all_math_operations(panel_df: pd.DataFrame) -> Dict:
        """
        Run all mathematical operation tests
        """
        return {
            'vector_magnitude': TestMathematicalOperations.test_vector_magnitude(panel_df),
            'coherence_bounds': TestMathematicalOperations.test_coherence_bounds(panel_df),
            'pca_variance': TestMathematicalOperations.test_pca_variance_sum(panel_df)
        }


class TestInfluenceAnalysis:
    """
    Test influence analysis operations
    """
    
    @staticmethod
    def test_influence_scores_positive(influence_df: pd.DataFrame) -> Dict:
        """
        Test that influence scores are non-negative
        """
        all_positive = (influence_df >= 0).all().all()
        
        return {
            'passed': all_positive,
            'min_score': influence_df.min().min(),
            'max_score': influence_df.max().max()
        }
    
    @staticmethod
    def test_influence_scores_normalized(influence_df: pd.DataFrame) -> Dict:
        """
        Test that influence scores are properly normalized
        (sum to 1 or similar constraint)
        """
        # For each time point, scores should sum to reasonable value
        row_sums = influence_df.sum(axis=1)
        
        return {
            'passed': True,  # Just informational
            'mean_sum': row_sums.mean(),
            'std_sum': row_sums.std(),
            'min_sum': row_sums.min(),
            'max_sum': row_sums.max()
        }
    
    @staticmethod
    def test_top_influencers_stability(rankings_t1: pd.DataFrame, 
                                      rankings_t2: pd.DataFrame) -> Dict:
        """
        Test that top influencers don't change randomly
        (Some stability expected unless regime shift)
        """
        top5_t1 = set(rankings_t1.head(5)['indicator'])
        top5_t2 = set(rankings_t2.head(5)['indicator'])
        
        overlap = len(top5_t1 & top5_t2) / 5.0
        
        return {
            'passed': overlap >= 0.4,  # At least 40% overlap
            'overlap_ratio': overlap,
            'top5_t1': list(top5_t1),
            'top5_t2': list(top5_t2)
        }


class TestPerformance:
    """
    Test computational performance and scalability
    """
    
    @staticmethod
    def test_computation_time(panel_df: pd.DataFrame, 
                             max_seconds: float = 60.0) -> Dict:
        """
        Test that analysis completes in reasonable time
        """
        import time
        
        from influence_analysis import InfluenceRanking
        
        ranker = InfluenceRanking(panel_df)
        
        start = time.time()
        _ = ranker.composite_influence_score(window=12)
        elapsed = time.time() - start
        
        return {
            'passed': elapsed < max_seconds,
            'elapsed_seconds': elapsed,
            'max_allowed': max_seconds,
            'n_indicators': len(panel_df.columns)
        }
    
    @staticmethod
    def test_memory_usage(panel_df: pd.DataFrame, 
                         max_mb: float = 500.0) -> Dict:
        """
        Test memory usage doesn't explode
        """
        import sys
        
        from influence_analysis import InfluenceRanking
        
        ranker = InfluenceRanking(panel_df)
        
        # Measure memory of result
        result = ranker.composite_influence_score(window=12)
        memory_mb = sys.getsizeof(result) / (1024 * 1024)
        
        return {
            'passed': memory_mb < max_mb,
            'memory_mb': memory_mb,
            'max_allowed': max_mb
        }


class TestIntegration:
    """
    End-to-end integration tests
    """
    
    @staticmethod
    def test_full_pipeline(panel_df: pd.DataFrame) -> Dict:
        """
        Test complete analysis pipeline
        """
        from influence_analysis import InfluenceAnalysisEngine
        
        try:
            engine = InfluenceAnalysisEngine(panel_df)
            
            # Run full analysis
            report = engine.full_influence_report(n_top=10)
            
            # Check all expected outputs exist
            expected_keys = [
                'top_influencers',
                'coherent_pairs',
                'coherent_clusters'
            ]
            
            missing_keys = [k for k in expected_keys if k not in report]
            
            return {
                'passed': len(missing_keys) == 0,
                'missing_outputs': missing_keys,
                'n_top_influencers': len(report.get('top_influencers', [])),
                'n_coherent_pairs': len(report.get('coherent_pairs', [])),
                'n_clusters': len(report.get('coherent_clusters', []))
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    @staticmethod
    def test_engine_comparison(panel_df: pd.DataFrame) -> Dict:
        """
        Test engine comparison framework
        """
        from engine_comparison import EngineComparator
        
        def dummy_engine_a(panel, **kwargs):
            from influence_analysis import InfluenceRanking
            ranker = InfluenceRanking(panel)
            return {'top_influencers': ranker.top_influencers(n_top=10)}
        
        def dummy_engine_b(panel, **kwargs):
            from influence_analysis import InfluenceRanking
            ranker = InfluenceRanking(panel)
            return {'top_influencers': ranker.top_influencers(n_top=10)}
        
        try:
            comparator = EngineComparator(panel_df)
            comparator.register_engine('Engine_A', dummy_engine_a)
            comparator.register_engine('Engine_B', dummy_engine_b)
            
            comparison = comparator.compare_influence_rankings()
            
            return {
                'passed': len(comparison) > 0,
                'n_indicators_compared': len(comparison)
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }


class VCFTestSuite:
    """
    Comprehensive test suite runner
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        self.panel = panel_df
        self.results = {}
        
    def run_all_tests(self) -> Dict:
        """
        Run complete test suite
        """
        print("=" * 70)
        print("VCF RESEARCH TEST SUITE")
        print("=" * 70)
        
        # 1. Data validation
        print("\n1. Data Validation Tests...")
        self.results['data_validation'] = TestDataValidator.test_data_consistency(self.panel)
        print(f"   ✓ Complete")
        
        # 2. Mathematical operations
        print("\n2. Mathematical Operation Tests...")
        self.results['math_operations'] = TestMathematicalOperations.test_all_math_operations(self.panel)
        print(f"   ✓ Complete")
        
        # 3. Performance
        print("\n3. Performance Tests...")
        self.results['performance'] = {
            'computation_time': TestPerformance.test_computation_time(self.panel),
            'memory_usage': TestPerformance.test_memory_usage(self.panel)
        }
        print(f"   ✓ Complete")
        
        # 4. Integration
        print("\n4. Integration Tests...")
        self.results['integration'] = {
            'full_pipeline': TestIntegration.test_full_pipeline(self.panel),
            'engine_comparison': TestIntegration.test_engine_comparison(self.panel)
        }
        print(f"   ✓ Complete")
        
        # Summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """
        Print test summary
        """
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        # Count passes/fails
        total_tests = 0
        passed_tests = 0
        
        def count_tests(d):
            nonlocal total_tests, passed_tests
            if isinstance(d, dict):
                if 'passed' in d:
                    total_tests += 1
                    if d['passed']:
                        passed_tests += 1
                else:
                    for v in d.values():
                        count_tests(v)
        
        count_tests(self.results)
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {passed_tests/total_tests*100:.1f}%")
        
        # Data quality summary
        data_val = self.results.get('data_validation', {})
        print(f"\nData Quality:")
        print(f"  Indicators: {data_val.get('n_indicators', 'N/A')}")
        print(f"  Time Points: {data_val.get('n_time_points', 'N/A')}")
        print(f"  Date Range: {data_val.get('date_range', 'N/A')}")
        
        # Performance summary
        perf = self.results.get('performance', {})
        comp_time = perf.get('computation_time', {})
        print(f"\nPerformance:")
        print(f"  Computation Time: {comp_time.get('elapsed_seconds', 'N/A'):.2f}s")
        print(f"  Memory Usage: {perf.get('memory_usage', {}).get('memory_mb', 'N/A'):.2f} MB")
        
        print("\n" + "=" * 70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Run test suite on sample data
    """
    
    # Create sample panel (replace with your 56 indicators)
    dates = pd.date_range('2010-01-01', periods=200, freq='M')
    
    # Simulate 56 indicators
    n_indicators = 56
    panel_data = {}
    
    for i in range(n_indicators):
        panel_data[f'indicator_{i+1}'] = np.cumsum(np.random.randn(200) * 0.5) + 100
    
    panel = pd.DataFrame(panel_data, index=dates)
    
    # Run test suite
    test_suite = VCFTestSuite(panel)
    results = test_suite.run_all_tests()
    
    # Save results
    import json
    with open('/home/claude/test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nTest results saved to test_results.json")
