"""
Lens Validator - Scores lenses against synthetic ground truth
==============================================================

Tests each lens against data where we KNOW the correct answer,
producing accuracy scores and diagnostic information.

Usage:
    from lens_validator import LensValidator
    from synthetic_data_generator import SyntheticDataGenerator
    
    validator = LensValidator()
    results = validator.validate_all_lenses(lenses_dict, generator)
    validator.print_report(results)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import warnings

# Import ground truth container
from synthetic_data_generator import SyntheticDataGenerator, GroundTruth


@dataclass
class ValidationResult:
    """Result of validating a single lens against ground truth."""
    lens_name: str
    test_name: str
    score: float  # 0-1, higher is better
    passed: bool
    details: Dict
    diagnostics: str


class LensValidator:
    """
    Validates mathematical lenses against synthetic data with known ground truth.
    """
    
    def __init__(self, tolerance: float = 0.2):
        """
        Args:
            tolerance: Tolerance for matching (e.g., regime boundary within N points)
        """
        self.tolerance = tolerance
        self.results = []
    
    # =========================================================================
    # REGIME DETECTION VALIDATION
    # =========================================================================
    
    def validate_regime_lens(
        self, 
        lens, 
        data: pd.DataFrame, 
        truth: GroundTruth,
        boundary_tolerance: int = 10
    ) -> ValidationResult:
        """
        Validate regime detection lens.
        
        Metrics:
        - Boundary detection: Did it find regime transitions near true boundaries?
        - Regime count: Did it find approximately the right number of regimes?
        - Label consistency: Are points in same true regime assigned same label?
        """
        try:
            result = lens.analyze(data)
        except Exception as e:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='regime_detection',
                score=0.0,
                passed=False,
                details={'error': str(e)},
                diagnostics=f"Lens failed to run: {e}"
            )
        
        true_boundaries = truth.expected_results['boundaries']
        true_labels = np.array(truth.expected_results['regime_labels'])
        n_true_regimes = truth.parameters['n_regimes']
        
        # Extract detected regimes
        if 'regime_labels' in result:
            detected_labels = np.array(result['regime_labels'])
        elif 'states' in result:
            detected_labels = np.array(result['states'])
        else:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='regime_detection',
                score=0.0,
                passed=False,
                details={'available_keys': list(result.keys())},
                diagnostics="No regime labels found in output"
            )
        
        # Metric 1: Boundary detection
        detected_boundaries = self._find_transitions(detected_labels)
        boundary_matches = 0
        for true_b in true_boundaries[1:]:  # Skip first (always 0)
            for det_b in detected_boundaries:
                if abs(true_b - det_b) <= boundary_tolerance:
                    boundary_matches += 1
                    break
        
        boundary_score = boundary_matches / max(len(true_boundaries) - 1, 1)
        
        # Metric 2: Number of regimes
        n_detected = len(np.unique(detected_labels))
        regime_count_score = 1.0 - abs(n_detected - n_true_regimes) / max(n_true_regimes, n_detected)
        regime_count_score = max(0, regime_count_score)
        
        # Metric 3: Adjusted Rand Index (label consistency)
        ari = self._adjusted_rand_index(true_labels, detected_labels)
        
        # Combined score
        score = 0.4 * boundary_score + 0.3 * regime_count_score + 0.3 * max(0, ari)
        
        return ValidationResult(
            lens_name=type(lens).__name__,
            test_name='regime_detection',
            score=score,
            passed=score > 0.5,
            details={
                'boundary_score': boundary_score,
                'regime_count_score': regime_count_score,
                'ari': ari,
                'true_regimes': n_true_regimes,
                'detected_regimes': n_detected,
                'boundaries_matched': f"{boundary_matches}/{len(true_boundaries)-1}"
            },
            diagnostics=f"Found {n_detected} regimes (true: {n_true_regimes}), matched {boundary_matches}/{len(true_boundaries)-1} boundaries"
        )
    
    def _find_transitions(self, labels: np.ndarray) -> List[int]:
        """Find indices where regime changes."""
        transitions = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transitions.append(i)
        return transitions
    
    def _adjusted_rand_index(self, labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        """Compute Adjusted Rand Index for clustering comparison."""
        # Simplified implementation
        n = len(labels_true)
        if n != len(labels_pred):
            return 0.0
        
        # Contingency table
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        
        contingency = np.zeros((len(classes), len(clusters)))
        for i, c in enumerate(classes):
            for j, k in enumerate(clusters):
                contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
        
        # Compute ARI
        sum_comb_c = sum(self._comb2(contingency[:, j].sum()) for j in range(len(clusters)))
        sum_comb_k = sum(self._comb2(contingency[i, :].sum()) for i in range(len(classes)))
        sum_comb = sum(self._comb2(contingency[i, j]) for i in range(len(classes)) for j in range(len(clusters)))
        
        total_comb = self._comb2(n)
        
        expected = sum_comb_c * sum_comb_k / total_comb if total_comb > 0 else 0
        max_index = (sum_comb_c + sum_comb_k) / 2
        
        if max_index - expected == 0:
            return 1.0 if sum_comb == expected else 0.0
        
        return (sum_comb - expected) / (max_index - expected)
    
    def _comb2(self, n: float) -> float:
        """Compute n choose 2."""
        return n * (n - 1) / 2 if n > 1 else 0
    
    # =========================================================================
    # CAUSALITY VALIDATION
    # =========================================================================
    
    def validate_causality_lens(
        self,
        lens,
        data: pd.DataFrame,
        truth: GroundTruth
    ) -> ValidationResult:
        """
        Validate causality detection (Granger, Transfer Entropy).
        
        Metrics:
        - True positive rate: Did it find the planted causal relationships?
        - False positive rate: Did it avoid false causality claims?
        - Direction accuracy: Did it get the direction right?
        """
        try:
            result = lens.analyze(data)
        except Exception as e:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='causality_detection',
                score=0.0,
                passed=False,
                details={'error': str(e)},
                diagnostics=f"Lens failed: {e}"
            )
        
        true_pairs = truth.expected_results['causal_pairs']  # (source, target, lag, strength)
        true_matrix = np.array(truth.expected_results['causality_matrix'])
        
        # Extract detected causality
        if 'te_matrix' in result:
            detected_matrix = np.array(result['te_matrix'])
        elif 'causality_matrix' in result:
            detected_matrix = np.array(result['causality_matrix'])
        elif 'granger_matrix' in result:
            detected_matrix = np.array(result['granger_matrix'])
        else:
            # Try to reconstruct from importance
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='causality_detection',
                score=0.0,
                passed=False,
                details={'available_keys': list(result.keys())},
                diagnostics="No causality matrix found"
            )
        
        # Threshold detected matrix
        threshold = np.percentile(detected_matrix[detected_matrix > 0], 50) if np.any(detected_matrix > 0) else 0
        detected_binary = (detected_matrix > threshold).astype(int)
        true_binary = (true_matrix > 0).astype(int)
        
        # True positives, false positives, false negatives
        tp = np.sum((detected_binary == 1) & (true_binary == 1))
        fp = np.sum((detected_binary == 1) & (true_binary == 0))
        fn = np.sum((detected_binary == 0) & (true_binary == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Direction accuracy: for detected pairs, is direction correct?
        direction_correct = 0
        direction_total = 0
        for s, t, lag, strength in true_pairs:
            if detected_matrix[s, t] > threshold:
                direction_total += 1
                if detected_matrix[s, t] > detected_matrix[t, s]:
                    direction_correct += 1
        
        direction_score = direction_correct / direction_total if direction_total > 0 else 0
        
        score = 0.4 * f1 + 0.3 * recall + 0.3 * direction_score
        
        return ValidationResult(
            lens_name=type(lens).__name__,
            test_name='causality_detection',
            score=score,
            passed=score > 0.4,
            details={
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'direction_accuracy': direction_score,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            },
            diagnostics=f"Precision: {precision:.2f}, Recall: {recall:.2f}, Direction: {direction_score:.2f}"
        )
    
    # =========================================================================
    # PERIODICITY VALIDATION
    # =========================================================================
    
    def validate_periodicity_lens(
        self,
        lens,
        data: pd.DataFrame,
        truth: GroundTruth,
        period_tolerance: float = 0.15  # 15% tolerance
    ) -> ValidationResult:
        """
        Validate periodicity detection (Wavelet).
        
        Metrics:
        - Period accuracy: Did it find the correct dominant periods?
        """
        try:
            result = lens.analyze(data)
        except Exception as e:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='periodicity_detection',
                score=0.0,
                passed=False,
                details={'error': str(e)},
                diagnostics=f"Lens failed: {e}"
            )
        
        true_periods = truth.expected_results['dominant_periods']
        
        # Extract detected periods
        detected_periods = []
        if 'dominant_period' in result:
            if isinstance(result['dominant_period'], dict):
                detected_periods = list(result['dominant_period'].values())
            elif isinstance(result['dominant_period'], pd.Series):
                detected_periods = result['dominant_period'].tolist()
            else:
                detected_periods = [result['dominant_period']]
        
        if not detected_periods:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='periodicity_detection',
                score=0.0,
                passed=False,
                details={'available_keys': list(result.keys())},
                diagnostics="No periods found in output"
            )
        
        # Match detected to true periods
        matches = 0
        for i, true_p in enumerate(true_periods[:len(detected_periods)]):
            if i < len(detected_periods):
                detected_p = detected_periods[i]
                relative_error = abs(detected_p - true_p) / true_p
                if relative_error <= period_tolerance:
                    matches += 1
        
        score = matches / len(true_periods)
        
        return ValidationResult(
            lens_name=type(lens).__name__,
            test_name='periodicity_detection',
            score=score,
            passed=score > 0.5,
            details={
                'true_periods': true_periods,
                'detected_periods': detected_periods,
                'matches': matches
            },
            diagnostics=f"Matched {matches}/{len(true_periods)} periods"
        )
    
    # =========================================================================
    # ANOMALY VALIDATION
    # =========================================================================
    
    def validate_anomaly_lens(
        self,
        lens,
        data: pd.DataFrame,
        truth: GroundTruth,
        time_tolerance: int = 2
    ) -> ValidationResult:
        """
        Validate anomaly detection.
        
        Metrics:
        - Detection rate: Did it find the planted anomalies?
        - False positive rate: Did it avoid flagging normal points?
        """
        try:
            result = lens.analyze(data)
        except Exception as e:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='anomaly_detection',
                score=0.0,
                passed=False,
                details={'error': str(e)},
                diagnostics=f"Lens failed: {e}"
            )
        
        true_anomalies = set(truth.expected_results['all_anomaly_times'])
        
        # Extract detected anomalies
        detected = set()
        if 'anomaly_indices' in result:
            for indices in result['anomaly_indices'].values():
                detected.update(indices)
        elif 'multivariate_anomaly_dates' in result:
            # Convert dates to indices
            dates = result['multivariate_anomaly_dates']
            for d in dates:
                if d in data.index:
                    detected.add(data.index.get_loc(d))
        elif 'anomalies' in result:
            detected = set(result['anomalies'])
        
        if not detected:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='anomaly_detection',
                score=0.0,
                passed=False,
                details={'available_keys': list(result.keys())},
                diagnostics="No anomalies found in output"
            )
        
        # Calculate with tolerance
        tp = 0
        for true_idx in true_anomalies:
            for det_idx in detected:
                if abs(true_idx - det_idx) <= time_tolerance:
                    tp += 1
                    break
        
        fp = len(detected) - tp
        fn = len(true_anomalies) - tp
        
        precision = tp / len(detected) if detected else 0
        recall = tp / len(true_anomalies) if true_anomalies else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return ValidationResult(
            lens_name=type(lens).__name__,
            test_name='anomaly_detection',
            score=f1,
            passed=f1 > 0.3,
            details={
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_anomalies': len(true_anomalies),
                'detected_anomalies': len(detected),
                'true_positives': tp
            },
            diagnostics=f"Precision: {precision:.2f}, Recall: {recall:.2f}"
        )
    
    # =========================================================================
    # NETWORK VALIDATION
    # =========================================================================
    
    def validate_network_lens(
        self,
        lens,
        data: pd.DataFrame,
        truth: GroundTruth
    ) -> ValidationResult:
        """
        Validate network structure detection.
        
        Metrics:
        - Hub identification: Did it find the most central node?
        - Edge detection: Did it find the correct edges?
        """
        try:
            result = lens.analyze(data)
        except Exception as e:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='network_detection',
                score=0.0,
                passed=False,
                details={'error': str(e)},
                diagnostics=f"Lens failed: {e}"
            )
        
        true_hub = truth.expected_results['hub_indicator']
        true_centrality = np.array(truth.expected_results['degree_centrality'])
        
        # Extract detected centrality
        if 'degree_centrality' in result:
            detected_centrality = result['degree_centrality']
            if isinstance(detected_centrality, dict):
                detected_centrality = np.array([detected_centrality.get(c, 0) for c in data.columns])
            elif isinstance(detected_centrality, pd.Series):
                detected_centrality = detected_centrality.values
            else:
                detected_centrality = np.array(detected_centrality)
        elif 'importance' in result:
            if isinstance(result['importance'], pd.Series):
                detected_centrality = result['importance'].values
            else:
                detected_centrality = np.array(result['importance'])
        else:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='network_detection',
                score=0.0,
                passed=False,
                details={'available_keys': list(result.keys())},
                diagnostics="No centrality measure found"
            )
        
        # Hub identification
        detected_hub = int(np.argmax(detected_centrality))
        hub_correct = detected_hub == true_hub
        
        # Rank correlation of centralities
        from scipy.stats import spearmanr
        try:
            rank_corr, _ = spearmanr(true_centrality, detected_centrality)
        except:
            rank_corr = 0
        
        score = 0.5 * float(hub_correct) + 0.5 * max(0, rank_corr)
        
        return ValidationResult(
            lens_name=type(lens).__name__,
            test_name='network_detection',
            score=score,
            passed=score > 0.5,
            details={
                'hub_correct': hub_correct,
                'true_hub': true_hub,
                'detected_hub': detected_hub,
                'rank_correlation': rank_corr
            },
            diagnostics=f"Hub {'correct' if hub_correct else 'incorrect'}, rank corr: {rank_corr:.2f}"
        )
    
    # =========================================================================
    # TOPOLOGY VALIDATION
    # =========================================================================
    
    def validate_topology_lens(
        self,
        lens,
        data: pd.DataFrame,
        truth: GroundTruth
    ) -> ValidationResult:
        """
        Validate topological structure detection.
        
        Metrics:
        - Distinguishes structured (attractor) from noise
        - Relative persistence ordering correct
        """
        try:
            result = lens.analyze(data)
        except Exception as e:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='topology_detection',
                score=0.0,
                passed=False,
                details={'error': str(e)},
                diagnostics=f"Lens failed: {e}"
            )
        
        expected_persistence = truth.expected_results['expected_persistence']
        is_deterministic = truth.expected_results['is_deterministic']
        
        # Extract persistence measure
        if 'total_persistence' in result:
            persistence = result['total_persistence']
            if isinstance(persistence, dict):
                persistence = np.mean(list(persistence.values()))
            elif isinstance(persistence, pd.Series):
                persistence = persistence.mean()
        elif 'persistence' in result:
            persistence = result['persistence']
        else:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name='topology_detection',
                score=0.0,
                passed=False,
                details={'available_keys': list(result.keys())},
                diagnostics="No persistence measure found"
            )
        
        # For this test, we just check if deterministic systems have higher persistence
        # This is a weak test - ideally we'd run multiple and compare
        if is_deterministic:
            score = 1.0 if persistence > 0.1 else 0.5
            diagnostics = f"Deterministic system, persistence: {persistence:.3f}"
        else:
            score = 1.0 if persistence < 0.5 else 0.5
            diagnostics = f"Noise system, persistence: {persistence:.3f}"
        
        return ValidationResult(
            lens_name=type(lens).__name__,
            test_name='topology_detection',
            score=score,
            passed=score > 0.5,
            details={
                'persistence': persistence,
                'expected': expected_persistence,
                'is_deterministic': is_deterministic
            },
            diagnostics=diagnostics
        )
    
    # =========================================================================
    # MAIN VALIDATION RUNNER
    # =========================================================================
    
    def validate_lens(
        self,
        lens,
        test_type: str,
        data: pd.DataFrame,
        truth: GroundTruth
    ) -> ValidationResult:
        """Route to appropriate validation method."""
        validators = {
            'regime': self.validate_regime_lens,
            'causality': self.validate_causality_lens,
            'periodicity': self.validate_periodicity_lens,
            'anomaly': self.validate_anomaly_lens,
            'network': self.validate_network_lens,
            'topology': self.validate_topology_lens,
        }
        
        if test_type not in validators:
            return ValidationResult(
                lens_name=type(lens).__name__,
                test_name=test_type,
                score=0.0,
                passed=False,
                details={},
                diagnostics=f"Unknown test type: {test_type}"
            )
        
        return validators[test_type](lens, data, truth)
    
    def validate_lens_suite(
        self,
        lens,
        generator: SyntheticDataGenerator
    ) -> List[ValidationResult]:
        """
        Run all applicable tests for a lens.
        
        Args:
            lens: Lens instance to test
            generator: SyntheticDataGenerator instance
            
        Returns:
            List of ValidationResults
        """
        results = []
        lens_name = type(lens).__name__.lower()
        
        # Determine which tests apply to this lens
        test_mapping = {
            'regime': ['regime'],
            'switch': ['regime'],
            'hmm': ['regime'],
            'causal': ['causality'],
            'granger': ['causality'],
            'transfer': ['causality'],
            'entropy': ['causality'],
            'wavelet': ['periodicity'],
            'spectral': ['periodicity'],
            'fourier': ['periodicity'],
            'anomaly': ['anomaly'],
            'outlier': ['anomaly'],
            'network': ['network'],
            'graph': ['network'],
            'central': ['network'],
            'topolog': ['topology'],
            'tda': ['topology'],
            'persist': ['topology'],
        }
        
        applicable_tests = []
        for keyword, tests in test_mapping.items():
            if keyword in lens_name:
                applicable_tests.extend(tests)
        
        # If no specific tests identified, run importance test
        if not applicable_tests:
            applicable_tests = ['importance']
        
        # Remove duplicates
        applicable_tests = list(set(applicable_tests))
        
        # Run applicable tests
        for test_type in applicable_tests:
            if test_type == 'regime':
                data, truth = generator.regime_data(n_regimes=3)
                results.append(self.validate_regime_lens(lens, data, truth))
            elif test_type == 'causality':
                data, truth = generator.causal_data()
                results.append(self.validate_causality_lens(lens, data, truth))
            elif test_type == 'periodicity':
                data, truth = generator.periodic_data()
                results.append(self.validate_periodicity_lens(lens, data, truth))
            elif test_type == 'anomaly':
                data, truth = generator.anomaly_data()
                results.append(self.validate_anomaly_lens(lens, data, truth))
            elif test_type == 'network':
                data, truth = generator.network_data()
                results.append(self.validate_network_lens(lens, data, truth))
            elif test_type == 'topology':
                data, truth = generator.topological_data(attractor_type='lorenz')
                results.append(self.validate_topology_lens(lens, data, truth))
        
        return results
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_report(self, results: List[ValidationResult]):
        """Print a formatted validation report."""
        print("\n" + "="*70)
        print("LENS VALIDATION REPORT")
        print("="*70 + "\n")
        
        # Group by lens
        by_lens = {}
        for r in results:
            if r.lens_name not in by_lens:
                by_lens[r.lens_name] = []
            by_lens[r.lens_name].append(r)
        
        for lens_name, lens_results in by_lens.items():
            avg_score = np.mean([r.score for r in lens_results])
            status = "✓ PASS" if all(r.passed for r in lens_results) else "✗ FAIL"
            
            print(f"{lens_name}")
            print("-" * 40)
            print(f"  Overall: {avg_score:.2f} {status}")
            
            for r in lens_results:
                status_char = "✓" if r.passed else "✗"
                print(f"  {status_char} {r.test_name}: {r.score:.2f}")
                print(f"      {r.diagnostics}")
            print()
        
        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        print("="*70)
        print(f"SUMMARY: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
        print("="*70)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def quick_validate(lenses: List, n_points: int = 500, seed: int = 42) -> Dict[str, float]:
    """
    Quick validation of a list of lenses.
    
    Args:
        lenses: List of lens instances
        n_points: Number of data points for synthetic data
        seed: Random seed
        
    Returns:
        Dict mapping lens name to average score
    """
    generator = SyntheticDataGenerator(n_points=n_points, seed=seed)
    validator = LensValidator()
    
    scores = {}
    for lens in lenses:
        results = validator.validate_lens_suite(lens, generator)
        if results:
            scores[type(lens).__name__] = np.mean([r.score for r in results])
        else:
            scores[type(lens).__name__] = None
    
    return scores


if __name__ == '__main__':
    print("Lens Validator - Run with actual lens instances")
    print("Example usage:")
    print("  from lens_validator import LensValidator, quick_validate")
    print("  from your_lenses import RegimeSwitchingLens, WaveletLens")
    print("  ")
    print("  scores = quick_validate([RegimeSwitchingLens(), WaveletLens()])")
    print("  print(scores)")
