"""
Ergodicity Analysis Module
==========================

Analyze ergodicity assumptions in financial analysis.
CRITICAL for honest interpretation of results.

Key Issue:
---------
Most financial analysis assumes ergodicity: that time averages equal
ensemble averages. For non-ergodic systems, this is FALSE.

"We have a single realization of market history. Our time averages are
used as estimators of ensemble properties. This assumes ergodicity,
which may not hold for economic systems."

This module tests and documents ergodicity assumptions.

References:
- Ole Peters, "The ergodicity problem in economics" (2019)
- Peters & Gell-Mann, "Evaluating gambles using dynamics" (2016)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ErgodicityTestResult:
    """Result of ergodicity test."""
    test_name: str
    is_ergodic: bool
    time_average: float
    ensemble_estimate: float
    discrepancy: float  # |time - ensemble| / |ensemble|
    mixing_time: Optional[int] = None
    conclusion: str = ""


@dataclass
class ErgodicityReport:
    """Complete ergodicity analysis report."""
    n_observations: int
    test_results: List[ErgodicityTestResult]
    overall_ergodic: bool
    key_findings: List[str]
    caveats: List[str]
    recommendations: List[str]


class ErgodicityAnalyzer:
    """
    Analyze ergodicity of time series processes.

    Ergodicity = time average converges to ensemble average as T → ∞

    For financial data:
    - We only have ONE realization (history)
    - We treat subsample statistics as "ensemble" proxies
    - This may be misleading if process is non-ergodic
    """

    def __init__(self, min_subsample_size: int = 50):
        """
        Initialize analyzer.

        Args:
            min_subsample_size: Minimum observations per subsample
        """
        self.min_subsample = min_subsample_size

    def test_mean_ergodicity(self, data: np.ndarray) -> ErgodicityTestResult:
        """
        Test if sample mean is stable across time.

        For ergodic processes, subsample means should converge to same value.

        Args:
            data: Time series data

        Returns:
            ErgodicityTestResult
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = len(data)

        # Full sample time average
        time_average = np.mean(data)

        # Split into subsamples
        n_subsamples = max(2, n // self.min_subsample)
        subsample_size = n // n_subsamples

        subsample_means = []
        for i in range(n_subsamples):
            start = i * subsample_size
            end = start + subsample_size
            subsample_means.append(np.mean(data[start:end]))

        # "Ensemble" estimate from subsample distribution
        ensemble_estimate = np.mean(subsample_means)
        subsample_std = np.std(subsample_means)

        # Discrepancy
        if ensemble_estimate != 0:
            discrepancy = abs(time_average - ensemble_estimate) / abs(ensemble_estimate)
        else:
            discrepancy = abs(time_average - ensemble_estimate)

        # Check if discrepancy is small relative to variability
        is_ergodic = discrepancy < 0.1 and subsample_std / (abs(ensemble_estimate) + 1e-10) < 0.3

        conclusion = (
            "Mean appears ergodic: time average stable across subsamples"
            if is_ergodic else
            "Mean may be non-ergodic: subsample means vary significantly"
        )

        return ErgodicityTestResult(
            test_name="Mean Ergodicity",
            is_ergodic=is_ergodic,
            time_average=time_average,
            ensemble_estimate=ensemble_estimate,
            discrepancy=discrepancy,
            conclusion=conclusion
        )

    def test_variance_ergodicity(self, data: np.ndarray) -> ErgodicityTestResult:
        """
        Test if sample variance is stable across time.

        Variance non-ergodicity is common (volatility clustering).

        Args:
            data: Time series data

        Returns:
            ErgodicityTestResult
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = len(data)

        # Full sample variance
        time_average = np.var(data)

        # Subsample variances
        n_subsamples = max(2, n // self.min_subsample)
        subsample_size = n // n_subsamples

        subsample_vars = []
        for i in range(n_subsamples):
            start = i * subsample_size
            end = start + subsample_size
            subsample_vars.append(np.var(data[start:end]))

        ensemble_estimate = np.mean(subsample_vars)
        subsample_std = np.std(subsample_vars)

        # Discrepancy
        if ensemble_estimate != 0:
            discrepancy = abs(time_average - ensemble_estimate) / ensemble_estimate
        else:
            discrepancy = abs(time_average - ensemble_estimate)

        # Variance is often non-ergodic due to volatility clustering
        # More lenient threshold
        is_ergodic = discrepancy < 0.3 and subsample_std / (ensemble_estimate + 1e-10) < 0.5

        conclusion = (
            "Variance appears ergodic"
            if is_ergodic else
            "Variance is non-ergodic: volatility varies significantly over time"
        )

        return ErgodicityTestResult(
            test_name="Variance Ergodicity",
            is_ergodic=is_ergodic,
            time_average=time_average,
            ensemble_estimate=ensemble_estimate,
            discrepancy=discrepancy,
            conclusion=conclusion
        )

    def test_growth_rate_ergodicity(self, returns: np.ndarray) -> ErgodicityTestResult:
        """
        Test ergodicity of growth rates (Ole Peters' key example).

        The time-average growth rate can differ dramatically from
        the ensemble average return. This is the core ergodicity issue
        in finance.

        Time average: <log(1+r)>_T (geometric mean)
        Ensemble average: <r> (arithmetic mean)

        For multiplicative processes, these can have OPPOSITE signs!

        Args:
            returns: Return series

        Returns:
            ErgodicityTestResult
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        # Arithmetic mean (ensemble average under ergodicity assumption)
        arithmetic_mean = np.mean(returns)

        # Geometric mean (true time-average growth rate)
        # log(geometric_mean) = mean(log(1 + r))
        log_returns = np.log(1 + returns)
        geometric_mean = np.exp(np.mean(log_returns)) - 1

        # Time average of growth
        time_average = geometric_mean

        # "Ensemble" estimate (arithmetic mean)
        ensemble_estimate = arithmetic_mean

        # Discrepancy
        if ensemble_estimate != 0:
            discrepancy = abs(time_average - ensemble_estimate) / abs(ensemble_estimate)
        else:
            discrepancy = abs(time_average - ensemble_estimate)

        # Check ergodicity gap
        # For ergodic processes, arithmetic ~ geometric
        is_ergodic = discrepancy < 0.2

        if arithmetic_mean > 0 and geometric_mean < 0:
            conclusion = "CRITICAL: Arithmetic mean positive but geometric mean negative!"
        elif discrepancy > 0.5:
            conclusion = f"Large ergodicity gap: arithmetic={arithmetic_mean:.4f}, geometric={geometric_mean:.4f}"
        else:
            conclusion = "Growth rate appears approximately ergodic"

        return ErgodicityTestResult(
            test_name="Growth Rate Ergodicity",
            is_ergodic=is_ergodic,
            time_average=time_average,
            ensemble_estimate=ensemble_estimate,
            discrepancy=discrepancy,
            conclusion=conclusion
        )

    def estimate_mixing_time(self, data: np.ndarray) -> int:
        """
        Estimate mixing time (how long until samples become independent).

        For ergodic processes, correlations decay. Mixing time estimates
        how many observations until dependence is negligible.

        Args:
            data: Time series data

        Returns:
            Estimated mixing time (in observations)
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = len(data)

        # Compute autocorrelation function
        data_centered = data - np.mean(data)
        var = np.var(data)

        if var == 0:
            return 1

        # Find lag where autocorrelation drops below threshold
        threshold = 0.05
        max_lag = min(n // 4, 500)

        for lag in range(1, max_lag):
            acf = np.mean(data_centered[lag:] * data_centered[:-lag]) / var
            if abs(acf) < threshold:
                return lag

        return max_lag  # Didn't decay within tested range

    def compute_time_vs_ensemble(self,
                                  data: np.ndarray,
                                  statistic: callable) -> Tuple[float, float, float]:
        """
        Compare time average vs ensemble-style estimate.

        Args:
            data: Time series data
            statistic: Function to compute statistic

        Returns:
            Tuple of (full_sample, subsample_mean, subsample_std)
        """
        data = np.asarray(data)
        n = len(data)

        # Full sample
        full_sample = statistic(data)

        # Subsamples
        n_subsamples = max(5, n // self.min_subsample)
        subsample_size = n // n_subsamples

        subsample_stats = []
        for i in range(n_subsamples):
            start = i * subsample_size
            end = start + subsample_size
            subsample_stats.append(statistic(data[start:end]))

        return full_sample, np.mean(subsample_stats), np.std(subsample_stats)

    def analyze(self,
                data: np.ndarray,
                is_returns: bool = True) -> ErgodicityReport:
        """
        Complete ergodicity analysis.

        Args:
            data: Time series data
            is_returns: Whether data represents returns (for growth rate test)

        Returns:
            ErgodicityReport
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = len(data)

        results = []

        # Mean ergodicity
        results.append(self.test_mean_ergodicity(data))

        # Variance ergodicity
        results.append(self.test_variance_ergodicity(data))

        # Growth rate ergodicity (if returns)
        if is_returns and np.all(data > -1):
            results.append(self.test_growth_rate_ergodicity(data))

        # Mixing time
        mixing_time = self.estimate_mixing_time(data)

        # Overall assessment
        overall_ergodic = all(r.is_ergodic for r in results)

        # Key findings
        key_findings = []
        for r in results:
            if not r.is_ergodic:
                key_findings.append(f"{r.test_name}: {r.conclusion}")

        key_findings.append(f"Estimated mixing time: {mixing_time} observations")

        # Caveats
        caveats = [
            "This analysis assumes the process is stationary, which may not hold.",
            "We only have one historical realization - 'ensemble' is estimated from subsamples.",
            "Non-ergodicity implies time averages do not represent 'expected' outcomes.",
            "Individual outcomes can differ dramatically from time averages.",
        ]

        # Recommendations
        recommendations = []
        if not overall_ergodic:
            recommendations.append("Use geometric/time averages instead of arithmetic means for growth")
            recommendations.append("Consider expected utility frameworks (Kelly criterion)")
            recommendations.append("Be cautious about ensemble-average interpretations")
            recommendations.append("Report both arithmetic and geometric averages")

        recommendations.append(f"Use at least {mixing_time * 10} observations for reliable statistics")

        return ErgodicityReport(
            n_observations=n,
            test_results=results,
            overall_ergodic=overall_ergodic,
            key_findings=key_findings,
            caveats=caveats,
            recommendations=recommendations
        )


def print_ergodicity_report(report: ErgodicityReport):
    """Pretty-print ergodicity report."""
    print("=" * 70)
    print("ERGODICITY ANALYSIS")
    print("=" * 70)
    print(f"Observations: {report.n_observations}")
    print(f"Overall Ergodic: {report.overall_ergodic}")
    print()

    print("TEST RESULTS:")
    print("-" * 50)
    for result in report.test_results:
        status = "ERGODIC" if result.is_ergodic else "NON-ERGODIC"
        print(f"\n{result.test_name}: [{status}]")
        print(f"  Time average: {result.time_average:.6f}")
        print(f"  Ensemble est: {result.ensemble_estimate:.6f}")
        print(f"  Discrepancy:  {result.discrepancy:.4f}")
        print(f"  {result.conclusion}")

    print()
    print("KEY FINDINGS:")
    print("-" * 50)
    for finding in report.key_findings:
        print(f"  - {finding}")

    print()
    print("CAVEATS:")
    print("-" * 50)
    for caveat in report.caveats:
        print(f"  ! {caveat}")

    print()
    print("RECOMMENDATIONS:")
    print("-" * 50)
    for rec in report.recommendations:
        print(f"  > {rec}")

    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("Ergodicity Analysis - Demo")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Stationary ergodic process (simple random returns)
    print("\nTest 1: Stationary Returns (should be ergodic)")
    print("-" * 50)

    returns_ergodic = np.random.randn(2000) * 0.01 + 0.0005

    analyzer = ErgodicityAnalyzer()
    report = analyzer.analyze(returns_ergodic, is_returns=True)
    print_ergodicity_report(report)

    # Test 2: Non-stationary process (trending volatility)
    print("\nTest 2: Non-stationary Process (likely non-ergodic)")
    print("-" * 50)

    n = 2000
    volatility = np.linspace(0.01, 0.03, n)  # Trending volatility
    returns_nonstat = np.random.randn(n) * volatility

    report = analyzer.analyze(returns_nonstat, is_returns=True)
    print_ergodicity_report(report)

    # Test 3: Multiplicative dynamics (the classic case)
    print("\nTest 3: Multiplicative Dynamics (Ole Peters example)")
    print("-" * 50)

    # Coin flip: +50% or -40% (positive expected value, negative growth)
    outcomes = np.random.choice([0.5, -0.4], size=1000)
    print(f"Arithmetic mean return: {np.mean(outcomes):.4f}")
    print(f"Geometric mean return:  {np.exp(np.mean(np.log(1 + outcomes))) - 1:.4f}")

    report = analyzer.analyze(outcomes, is_returns=True)
    # Just show growth rate result
    for r in report.test_results:
        if r.test_name == "Growth Rate Ergodicity":
            print(f"\n{r.conclusion}")

    print("\nTest completed!")
