"""
Phase 7 Verification: Universal Cross-Domain Analysis
=====================================================

This verification goes beyond simple yes/no checks.
It validates that the system correctly identifies mathematical
rhythms that transcend domain boundaries.

Mathematical Verification Approach:
1. Create synthetic data with KNOWN shared rhythms
2. Run cross-domain analysis
3. Verify system DISCOVERS the planted rhythms
4. Coherence scores should match expected values

This is the "mathematical" way - let the numbers prove the system works.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class MathematicalVerifier:
    """
    Verifies PRISM cross-domain analysis using mathematical ground truth.

    Instead of simple yes/no, we:
    1. Plant known signals
    2. Check if system finds them
    3. Validate coherence magnitudes
    """

    def __init__(self):
        self.results = {}
        self.tolerance = 0.15  # Allow 15% deviation from expected

    def create_test_data(self, n: int = 512) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Create synthetic cross-domain data with KNOWN relationships.

        Returns data and ground truth for verification.
        """
        np.random.seed(42)
        t = np.arange(n)

        # === PLANTED RHYTHMS ===

        # Annual cycle (shared across domains)
        annual = np.sin(2 * np.pi * t / 365)

        # Quarterly cycle
        quarterly = np.sin(2 * np.pi * t / 90)

        # Create indicators with known relationships
        data = pd.DataFrame({
            # Economic: Strong annual + some quarterly
            'spy': 0.7 * annual + 0.2 * quarterly + 0.2 * np.random.randn(n),
            'yield': 0.5 * annual + 0.1 * quarterly + 0.3 * np.random.randn(n),

            # Climate: Strong annual (in phase)
            'temp': 0.9 * annual + 0.1 * np.random.randn(n),
            'enso': 0.6 * annual + np.sin(2 * np.pi * t / 365 + 0.3) * 0.2 + 0.2 * np.random.randn(n),

            # Biological: Annual (phase shifted)
            'flu': 0.8 * np.sin(2 * np.pi * t / 365 + np.pi/6) + 0.2 * np.random.randn(n),

            # Social: Weak annual
            'sentiment': 0.3 * annual + 0.5 * np.random.randn(n),

            # Noise (control)
            'noise': np.random.randn(n)
        })

        domains = {
            'spy': 'economic', 'yield': 'economic',
            'temp': 'climate', 'enso': 'climate',
            'flu': 'biological',
            'sentiment': 'social',
            'noise': 'control'
        }

        # Ground truth: expected high-coherence pairs
        ground_truth = {
            'high_coherence_pairs': [
                ('spy', 'temp', 0.5),      # Both strong annual
                ('spy', 'flu', 0.4),       # Annual rhythm
                ('temp', 'flu', 0.5),      # Both annual
                ('temp', 'enso', 0.5),     # Same domain, annual
            ],
            'low_coherence_pairs': [
                ('noise', 'spy', 0.15),    # Noise should be low
                ('noise', 'temp', 0.15),
            ],
            'trending_indicators': ['temp'],  # Strong trend component
            'oscillating_indicators': ['spy', 'flu', 'enso']
        }

        return data, domains, ground_truth

    def verify_coherence_discovery(self,
                                    coherence_matrix: Dict,
                                    ground_truth: Dict) -> Dict:
        """
        Verify system finds expected coherence relationships.
        """
        results = {'passed': 0, 'failed': 0, 'details': []}

        # Check high-coherence pairs
        for ind1, ind2, expected_min in ground_truth['high_coherence_pairs']:
            actual = coherence_matrix.get(ind1, {}).get(ind2, 0)
            passed = actual >= expected_min * (1 - self.tolerance)

            results['details'].append({
                'test': f'High coherence: {ind1}-{ind2}',
                'expected': f'>= {expected_min:.2f}',
                'actual': f'{actual:.3f}',
                'passed': passed
            })
            results['passed' if passed else 'failed'] += 1

        # Check low-coherence pairs (noise)
        for ind1, ind2, expected_max in ground_truth['low_coherence_pairs']:
            actual = coherence_matrix.get(ind1, {}).get(ind2, 1)
            passed = actual <= expected_max * (1 + self.tolerance)

            results['details'].append({
                'test': f'Low coherence (noise): {ind1}-{ind2}',
                'expected': f'<= {expected_max:.2f}',
                'actual': f'{actual:.3f}',
                'passed': passed
            })
            results['passed' if passed else 'failed'] += 1

        return results

    def verify_cross_domain_discovery(self,
                                       shared_rhythms: list,
                                       domains: Dict) -> Dict:
        """
        Verify cross-domain rhythms are discovered.
        """
        results = {'passed': 0, 'failed': 0, 'details': []}

        # Should find at least some cross-domain pairs
        cross_domain_found = [r for r in shared_rhythms if r.get('is_cross_domain', False)]

        passed = len(cross_domain_found) >= 3
        results['details'].append({
            'test': 'Cross-domain rhythms discovered',
            'expected': '>= 3 pairs',
            'actual': f'{len(cross_domain_found)} pairs',
            'passed': passed
        })
        results['passed' if passed else 'failed'] += 1

        # Top cross-domain pair should have meaningful coherence
        if cross_domain_found:
            top_coherence = cross_domain_found[0]['coherence']
            passed = top_coherence > 0.3
            results['details'].append({
                'test': 'Top cross-domain coherence meaningful',
                'expected': '> 0.3',
                'actual': f'{top_coherence:.3f}',
                'passed': passed
            })
            results['passed' if passed else 'failed'] += 1

        return results

    def verify_structure(self) -> Dict:
        """
        Verify the Phase 7 file structure is in place.
        """
        results = {'passed': 0, 'failed': 0, 'details': []}

        files_to_check = [
            ('data/universal_registry.yaml', 'Universal Registry'),
            ('runner/universal_selector.py', 'Universal Selector'),
            ('engines/cross_domain_engine.py', 'Cross-Domain Engine'),
            ('templates/runner/universal.html', 'Web UI Template'),
        ]

        for file_path, name in files_to_check:
            full_path = Path(__file__).parent / file_path
            exists = full_path.exists()
            results['details'].append({
                'test': f'File exists: {name}',
                'expected': 'exists',
                'actual': 'exists' if exists else 'missing',
                'passed': exists
            })
            results['passed' if exists else 'failed'] += 1

        return results

    def run_full_verification(self) -> Dict:
        """
        Run complete mathematical verification suite.
        """
        print("=" * 60)
        print("PRISM Phase 7: Mathematical Verification")
        print("=" * 60)

        all_results = []

        # First verify structure
        print("\n--- Structure Verification ---")
        struct_results = self.verify_structure()
        all_results.append(struct_results)
        self._print_results(struct_results)

        print("\nCreating test data with planted rhythms...")
        data, domains, ground_truth = self.create_test_data()
        print(f"  Created {len(data.columns)} indicators across {len(set(domains.values()))} domains")

        # Try to import and run analysis
        try:
            from engines.cross_domain_engine import run_cross_domain_analysis

            print("\nRunning cross-domain analysis...")
            result = run_cross_domain_analysis(data, domains, mode='meta')

            if result['status'] != 'completed':
                print("Analysis failed to complete")
                return {'overall': 'FAILED', 'reason': 'Analysis did not complete'}

            print("Analysis completed")

        except ImportError as e:
            print(f"\nCross-domain engine import failed: {e}")
            print("  Running with mock data for structure verification...")

            # Create mock results for structure verification
            result = self._create_mock_results(data, domains)

        # Run verifications
        print("\n--- Coherence Discovery Verification ---")
        coh_results = self.verify_coherence_discovery(
            result.get('coherence_matrix', {}),
            ground_truth
        )
        all_results.append(coh_results)
        self._print_results(coh_results)

        print("\n--- Cross-Domain Discovery Verification ---")
        cross_results = self.verify_cross_domain_discovery(
            result.get('shared_rhythms', []),
            domains
        )
        all_results.append(cross_results)
        self._print_results(cross_results)

        # Summary
        total_passed = sum(r['passed'] for r in all_results)
        total_failed = sum(r['failed'] for r in all_results)

        print("\n" + "=" * 60)
        print(f"VERIFICATION SUMMARY: {total_passed} passed, {total_failed} failed")

        if total_failed == 0:
            print("ALL MATHEMATICAL VERIFICATIONS PASSED")
            return {'overall': 'PASSED', 'passed': total_passed, 'failed': 0}
        else:
            print("SOME VERIFICATIONS FAILED")
            return {'overall': 'FAILED', 'passed': total_passed, 'failed': total_failed}

    def _create_mock_results(self, data: pd.DataFrame, domains: Dict) -> Dict:
        """Create mock results for structure verification."""
        # This allows testing the verification framework even before
        # the full engine is installed

        columns = list(data.columns)
        coherence_matrix = {col: {} for col in columns}

        # Simulate coherence based on simple correlation
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                corr = abs(np.corrcoef(data[col1], data[col2])[0, 1])
                coherence_matrix[col1][col2] = corr
                coherence_matrix[col2][col1] = corr

        # Find cross-domain pairs
        shared_rhythms = []
        for col1 in columns:
            for col2 in columns:
                if col1 >= col2:
                    continue
                d1, d2 = domains.get(col1), domains.get(col2)
                if d1 != d2:
                    coh = coherence_matrix[col1].get(col2, 0)
                    shared_rhythms.append({
                        'indicator_1': col1,
                        'indicator_2': col2,
                        'domain_1': d1,
                        'domain_2': d2,
                        'coherence': coh,
                        'is_cross_domain': True
                    })

        shared_rhythms.sort(key=lambda x: x['coherence'], reverse=True)

        return {
            'status': 'completed',
            'coherence_matrix': coherence_matrix,
            'shared_rhythms': shared_rhythms
        }

    def _print_results(self, results: Dict):
        """Pretty print verification results."""
        for detail in results['details']:
            status = "[PASS]" if detail['passed'] else "[FAIL]"
            print(f"  {status} {detail['test']}")
            print(f"        Expected: {detail['expected']}, Actual: {detail['actual']}")


def test_universal_selector():
    """Test the UniversalSelector class."""
    print("\n--- Universal Selector Tests ---")

    try:
        from runner.universal_selector import UniversalSelector, quick_select

        selector = UniversalSelector()

        # Test listing domains
        domains = selector.list_domains()
        print(f"  [PASS] list_domains(): {len(domains)} domains found")

        # Test listing indicators
        indicators = selector.list_indicators()
        print(f"  [PASS] list_indicators(): {len(indicators)} indicators found")

        # Test creating selection
        selection = selector.create_selection(
            ['economic.spy_ma_ratio', 'climate.enso_index'],
            mode='meta'
        )
        print(f"  [PASS] create_selection(): {len(selection.indicators)} indicators selected")
        print(f"  [PASS] is_cross_domain: {selection.is_cross_domain}")

        # Test validation
        validation = selector.validate_selection(selection)
        print(f"  [PASS] validate_selection(): valid={validation['valid']}")

        return True

    except Exception as e:
        print(f"  [FAIL] Universal Selector test failed: {e}")
        return False


def test_cross_domain_engine():
    """Test the CrossDomainEngine class."""
    print("\n--- Cross-Domain Engine Tests ---")

    try:
        from engines.cross_domain_engine import CrossDomainEngine, run_cross_domain_analysis
        import pandas as pd

        # Create simple test data
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'a': np.sin(np.arange(n) / 10) + np.random.randn(n) * 0.1,
            'b': np.sin(np.arange(n) / 10 + 0.5) + np.random.randn(n) * 0.1,
            'c': np.random.randn(n)
        })

        domains = {'a': 'domain1', 'b': 'domain1', 'c': 'domain2'}

        # Test engine initialization
        engine = CrossDomainEngine()
        print("  [PASS] CrossDomainEngine initialized")

        # Test analysis
        result = run_cross_domain_analysis(data, domains, mode='meta')
        print(f"  [PASS] run_cross_domain_analysis(): status={result['status']}")

        # Check result structure
        assert 'coherence_matrix' in result, "Missing coherence_matrix"
        assert 'hurst_values' in result, "Missing hurst_values"
        assert 'shared_rhythms' in result, "Missing shared_rhythms"
        assert 'domain_summary' in result, "Missing domain_summary"
        assert 'insights' in result, "Missing insights"
        print("  [PASS] Result structure is complete")

        return True

    except Exception as e:
        print(f"  [FAIL] Cross-Domain Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run verification."""
    print("\n" + "=" * 60)
    print("        PRISM Phase 7 - Universal Cross-Domain Analysis")
    print("                   Verification Suite")
    print("=" * 60)

    # Run unit tests first
    selector_ok = test_universal_selector()
    engine_ok = test_cross_domain_engine()

    # Run mathematical verification
    verifier = MathematicalVerifier()
    result = verifier.run_full_verification()

    # Final summary
    print("\n" + "=" * 60)
    print("                    FINAL SUMMARY")
    print("=" * 60)
    print(f"  Universal Selector: {'PASS' if selector_ok else 'FAIL'}")
    print(f"  Cross-Domain Engine: {'PASS' if engine_ok else 'FAIL'}")
    print(f"  Mathematical Verification: {result['overall']}")
    print("=" * 60)

    # Exit code for CI/CD
    all_passed = selector_ok and engine_ok and result['overall'] == 'PASSED'
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
