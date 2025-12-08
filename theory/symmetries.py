"""
Symmetry Analysis Module
========================

Document invariance properties of the indicator space analysis.
Important for understanding what transformations preserve results.

Key symmetries to test:
- Scale invariance: f(λx) = λ^α f(x)
- Translation invariance: f(x + c) = f(x)
- Rotation invariance: f(Rx) = f(x)
- Time translation: f(t + τ) = f(t)

DISCLOSURE: Document which symmetries are BROKEN as well as preserved.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class SymmetryTestResult:
    """Result of a single symmetry test."""
    symmetry_name: str
    is_invariant: bool
    invariance_score: float  # 0-1, how well invariance holds
    violation_measure: float  # Quantifies degree of violation
    transformation_tested: str
    n_tests: int


@dataclass
class SymmetryReport:
    """Complete symmetry analysis report."""
    n_dimensions: int
    test_results: List[SymmetryTestResult]
    preserved_symmetries: List[str]
    broken_symmetries: List[str]
    implications: List[str]


class SymmetryAnalyzer:
    """
    Analyze symmetry properties of analysis functions.
    """

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize analyzer.

        Args:
            tolerance: Relative tolerance for invariance
        """
        self.tolerance = tolerance

    def test_scale_invariance(self,
                               func: Callable,
                               data: np.ndarray,
                               scales: List[float] = [0.5, 2.0, 5.0]) -> SymmetryTestResult:
        """
        Test scale invariance: f(λx) = λ^α f(x) for some α.

        Many physical laws have power-law scaling.
        Financial returns, Hurst exponents, etc. may or may not scale.

        Args:
            func: Function to test
            data: Input data
            scales: Scale factors to test

        Returns:
            SymmetryTestResult
        """
        base_result = func(data)
        if isinstance(base_result, dict):
            base_result = list(base_result.values())[0]
        base_result = float(base_result)

        violations = []
        scaling_exponents = []

        for scale in scales:
            scaled_data = data * scale
            scaled_result = func(scaled_data)

            if isinstance(scaled_result, dict):
                scaled_result = list(scaled_result.values())[0]
            scaled_result = float(scaled_result)

            # Check for power-law scaling
            if base_result != 0 and scaled_result != 0:
                # f(λx) = λ^α f(x) => α = log(f(λx)/f(x)) / log(λ)
                ratio = scaled_result / base_result
                if ratio > 0:
                    alpha = np.log(ratio) / np.log(scale)
                    scaling_exponents.append(alpha)

                    # Violation is deviation from consistent α
                    violations.append(alpha)

        if len(scaling_exponents) > 1:
            # Check if scaling exponents are consistent
            alpha_std = np.std(scaling_exponents)
            alpha_mean = np.mean(scaling_exponents)

            is_invariant = alpha_std < self.tolerance * abs(alpha_mean + 1e-10)
            invariance_score = max(0, 1 - alpha_std / (abs(alpha_mean) + 1e-10))
            violation = alpha_std
        else:
            is_invariant = False
            invariance_score = 0
            violation = np.inf

        return SymmetryTestResult(
            symmetry_name="Scale Invariance",
            is_invariant=is_invariant,
            invariance_score=invariance_score,
            violation_measure=violation,
            transformation_tested=f"x -> λx for λ ∈ {scales}",
            n_tests=len(scales)
        )

    def test_translation_invariance(self,
                                     func: Callable,
                                     data: np.ndarray,
                                     shifts: List[float] = None) -> SymmetryTestResult:
        """
        Test translation invariance: f(x + c) = f(x).

        Mean-adjusted metrics should be translation invariant.
        Absolute level metrics (like mean) are not.

        Args:
            func: Function to test
            data: Input data
            shifts: Shift values to test

        Returns:
            SymmetryTestResult
        """
        if shifts is None:
            data_std = np.std(data)
            shifts = [-data_std, data_std, 2 * data_std]

        base_result = func(data)
        if isinstance(base_result, dict):
            base_result = list(base_result.values())[0]
        base_result = float(base_result)

        violations = []

        for shift in shifts:
            shifted_data = data + shift
            shifted_result = func(shifted_data)

            if isinstance(shifted_result, dict):
                shifted_result = list(shifted_result.values())[0]
            shifted_result = float(shifted_result)

            if base_result != 0:
                relative_change = abs(shifted_result - base_result) / abs(base_result)
            else:
                relative_change = abs(shifted_result - base_result)

            violations.append(relative_change)

        max_violation = max(violations)
        is_invariant = max_violation < self.tolerance

        return SymmetryTestResult(
            symmetry_name="Translation Invariance",
            is_invariant=is_invariant,
            invariance_score=max(0, 1 - max_violation),
            violation_measure=max_violation,
            transformation_tested=f"x -> x + c for c ∈ {shifts}",
            n_tests=len(shifts)
        )

    def test_time_translation_invariance(self,
                                          func: Callable,
                                          data: np.ndarray,
                                          window_size: int = None) -> SymmetryTestResult:
        """
        Test time translation invariance: same result for different time windows.

        Stationary processes should give similar results across time.

        Args:
            func: Function to test
            data: Time series data
            window_size: Size of windows to compare

        Returns:
            SymmetryTestResult
        """
        n = len(data)
        if window_size is None:
            window_size = n // 4

        if window_size > n // 2:
            window_size = n // 4

        # Compute function on different time windows
        results = []
        n_windows = max(2, n // window_size)

        for i in range(n_windows):
            start = i * window_size
            end = min(start + window_size, n)
            if end - start < window_size // 2:
                continue

            window_data = data[start:end]
            result = func(window_data)

            if isinstance(result, dict):
                result = list(result.values())[0]
            results.append(float(result))

        if len(results) < 2:
            return SymmetryTestResult(
                symmetry_name="Time Translation Invariance",
                is_invariant=False,
                invariance_score=0,
                violation_measure=np.inf,
                transformation_tested="t -> t + τ",
                n_tests=len(results)
            )

        # Check consistency across windows
        result_std = np.std(results)
        result_mean = np.mean(results)

        if result_mean != 0:
            cv = result_std / abs(result_mean)
        else:
            cv = result_std

        is_invariant = cv < self.tolerance
        invariance_score = max(0, 1 - cv)

        return SymmetryTestResult(
            symmetry_name="Time Translation Invariance",
            is_invariant=is_invariant,
            invariance_score=invariance_score,
            violation_measure=cv,
            transformation_tested=f"Windows of size {window_size}",
            n_tests=len(results)
        )

    def test_permutation_invariance(self,
                                     func: Callable,
                                     data: np.ndarray,
                                     n_permutations: int = 10) -> SymmetryTestResult:
        """
        Test permutation invariance: f(permute(x)) = f(x).

        Order-independent statistics (mean, variance) are permutation invariant.
        Time series statistics (autocorrelation, Hurst) are not.

        Args:
            func: Function to test
            data: Input data
            n_permutations: Number of permutations to test

        Returns:
            SymmetryTestResult
        """
        base_result = func(data)
        if isinstance(base_result, dict):
            base_result = list(base_result.values())[0]
        base_result = float(base_result)

        violations = []
        rng = np.random.RandomState(42)

        for _ in range(n_permutations):
            perm_data = rng.permutation(data)
            perm_result = func(perm_data)

            if isinstance(perm_result, dict):
                perm_result = list(perm_result.values())[0]
            perm_result = float(perm_result)

            if base_result != 0:
                violation = abs(perm_result - base_result) / abs(base_result)
            else:
                violation = abs(perm_result - base_result)

            violations.append(violation)

        max_violation = max(violations)
        is_invariant = max_violation < self.tolerance

        return SymmetryTestResult(
            symmetry_name="Permutation Invariance",
            is_invariant=is_invariant,
            invariance_score=max(0, 1 - max_violation),
            violation_measure=max_violation,
            transformation_tested=f"{n_permutations} random permutations",
            n_tests=n_permutations
        )

    def analyze(self,
                func: Callable,
                data: np.ndarray) -> SymmetryReport:
        """
        Complete symmetry analysis.

        Args:
            func: Analysis function to test
            data: Input data

        Returns:
            SymmetryReport
        """
        results = []

        # Run all tests
        results.append(self.test_scale_invariance(func, data))
        results.append(self.test_translation_invariance(func, data))
        results.append(self.test_time_translation_invariance(func, data))
        results.append(self.test_permutation_invariance(func, data))

        # Categorize
        preserved = [r.symmetry_name for r in results if r.is_invariant]
        broken = [r.symmetry_name for r in results if not r.is_invariant]

        # Generate implications
        implications = []

        if "Scale Invariance" in preserved:
            implications.append("Results may follow power-law scaling")
        if "Scale Invariance" in broken:
            implications.append("Results are scale-dependent - interpret with caution")

        if "Translation Invariance" in preserved:
            implications.append("Results independent of data level")
        if "Translation Invariance" in broken:
            implications.append("Results depend on absolute levels - normalization recommended")

        if "Time Translation Invariance" in preserved:
            implications.append("Results stable across time - process may be stationary")
        if "Time Translation Invariance" in broken:
            implications.append("Results vary over time - possible non-stationarity")

        if "Permutation Invariance" in preserved:
            implications.append("Results are order-independent")
        if "Permutation Invariance" in broken:
            implications.append("Temporal structure matters for this metric")

        n_dims = len(data.shape)

        return SymmetryReport(
            n_dimensions=n_dims,
            test_results=results,
            preserved_symmetries=preserved,
            broken_symmetries=broken,
            implications=implications
        )


# Convenience functions
def test_scale_invariance(func: Callable, data: np.ndarray) -> SymmetryTestResult:
    """Quick scale invariance test."""
    analyzer = SymmetryAnalyzer()
    return analyzer.test_scale_invariance(func, data)


def test_translation_invariance(func: Callable, data: np.ndarray) -> SymmetryTestResult:
    """Quick translation invariance test."""
    analyzer = SymmetryAnalyzer()
    return analyzer.test_translation_invariance(func, data)


def print_symmetry_report(report: SymmetryReport):
    """Pretty-print symmetry report."""
    print("=" * 60)
    print("SYMMETRY ANALYSIS")
    print("=" * 60)
    print()

    print("TEST RESULTS:")
    print("-" * 40)
    for result in report.test_results:
        status = "PRESERVED" if result.is_invariant else "BROKEN"
        print(f"\n{result.symmetry_name}: [{status}]")
        print(f"  Score: {result.invariance_score:.3f}")
        print(f"  Violation: {result.violation_measure:.4f}")
        print(f"  Tests: {result.n_tests}")

    print()
    print("SUMMARY:")
    print("-" * 40)
    print(f"  Preserved: {', '.join(report.preserved_symmetries) or 'None'}")
    print(f"  Broken: {', '.join(report.broken_symmetries) or 'None'}")

    print()
    print("IMPLICATIONS:")
    for imp in report.implications:
        print(f"  - {imp}")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Symmetry Analysis - Demo")
    print("=" * 60)

    np.random.seed(42)

    # Test data
    data = np.random.randn(500)

    # Test functions
    print("\nTest 1: Mean (should be translation variant, scale invariant)")
    print("-" * 40)
    analyzer = SymmetryAnalyzer()
    report = analyzer.analyze(np.mean, data)
    print_symmetry_report(report)

    print("\nTest 2: Standard Deviation (should be translation invariant)")
    print("-" * 40)
    report = analyzer.analyze(np.std, data)
    print_symmetry_report(report)

    print("\nTest 3: Autocorrelation (should be permutation variant)")
    print("-" * 40)
    def autocorr(x):
        x = x - np.mean(x)
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    report = analyzer.analyze(autocorr, data)
    print_symmetry_report(report)

    print("\nTest completed!")
