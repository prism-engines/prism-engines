"""
Power Analysis Module
=====================

Sample size requirements and statistical power calculations.
Critical for knowing whether analyses have sufficient data.

Functions:
- Minimum samples for reliable Hurst estimation
- Frequency resolution for coherence analysis
- Minimum regime transitions for change detection
- Empirical power via simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class PowerResult:
    """Result from power analysis."""
    analysis_type: str
    current_n: int
    required_n: int
    achieved_power: float
    target_power: float
    is_sufficient: bool
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerReport:
    """Comprehensive power analysis report."""
    n_observations: int
    analyses: Dict[str, PowerResult] = field(default_factory=dict)
    overall_sufficient: bool = True
    recommendations: List[str] = field(default_factory=list)


class PowerAnalyzer:
    """
    Power analysis for time series metrics.

    Helps determine if sample size is adequate for reliable inference.
    """

    def __init__(self, target_power: float = 0.80, alpha: float = 0.05):
        """
        Initialize analyzer.

        Args:
            target_power: Desired statistical power (default 80%)
            alpha: Significance level
        """
        self.target_power = target_power
        self.alpha = alpha

    def hurst_min_samples(self,
                          target_precision: float = 0.05,
                          method: str = 'rs') -> Dict[str, Any]:
        """
        Minimum samples for reliable Hurst estimation.

        Based on Monte Carlo studies of Hurst estimator variance.

        Args:
            target_precision: Desired standard error of Hurst estimate
            method: 'rs' (R/S), 'dfa', or 'wavelet'

        Returns:
            Dictionary with sample requirements
        """
        # Empirical relationships from simulation studies
        # SE(H) ~ c / sqrt(N) where c depends on method

        method_coefficients = {
            'rs': 1.5,      # R/S has higher variance
            'dfa': 1.2,     # DFA more efficient
            'wavelet': 1.0  # Wavelet most efficient
        }

        c = method_coefficients.get(method, 1.5)

        # Required N for target precision: N = (c / target_precision)^2
        required_n = int((c / target_precision) ** 2)

        # Absolute minimums from literature
        absolute_mins = {
            'rs': 256,      # Need enough windows for R/S
            'dfa': 500,     # DFA needs longer series
            'wavelet': 128  # Wavelet can work with less
        }

        min_n = max(required_n, absolute_mins.get(method, 256))

        return {
            'method': method,
            'target_precision': target_precision,
            'required_n': min_n,
            'absolute_minimum': absolute_mins.get(method, 256),
            'expected_se': round(c / np.sqrt(min_n), 4),
            'recommendations': [
                f"Use at least {min_n} observations for {method} Hurst estimation",
                f"Expected SE: {c / np.sqrt(min_n):.4f}",
                "Consider DFA or wavelet for shorter series"
            ]
        }

    def coherence_min_samples(self,
                              freq_of_interest: float,
                              sampling_rate: float = 1.0,
                              n_segments: int = 8) -> Dict[str, Any]:
        """
        Minimum samples for coherence at a target frequency.

        Coherence estimation requires sufficient frequency resolution.

        Args:
            freq_of_interest: Target frequency (cycles per unit time)
            sampling_rate: Samples per unit time
            n_segments: Number of segments for Welch method

        Returns:
            Dictionary with sample requirements
        """
        # Frequency resolution = sampling_rate / segment_length
        # To resolve freq_of_interest, need segment_length > sampling_rate / freq_of_interest

        min_segment_length = int(sampling_rate / freq_of_interest * 2)  # Nyquist factor
        required_n = min_segment_length * n_segments

        # For statistical stability, need even more
        # Degrees of freedom ~ 2 * n_segments
        dof = 2 * n_segments

        # For reliable coherence, need DOF > 20
        if dof < 20:
            n_segments_needed = 10
            required_n = min_segment_length * n_segments_needed
            dof = 2 * n_segments_needed

        return {
            'freq_of_interest': freq_of_interest,
            'sampling_rate': sampling_rate,
            'min_segment_length': min_segment_length,
            'n_segments_used': n_segments,
            'degrees_of_freedom': dof,
            'required_n': required_n,
            'frequency_resolution': round(sampling_rate / min_segment_length, 6),
            'recommendations': [
                f"Need at least {required_n} samples for coherence at {freq_of_interest} Hz",
                f"Frequency resolution: {sampling_rate / min_segment_length:.6f}",
                f"Degrees of freedom: {dof}"
            ]
        }

    def regime_min_transitions(self,
                               n_regimes: int = 3,
                               min_per_regime: int = 30,
                               expected_dwell: int = 50) -> Dict[str, Any]:
        """
        Minimum data requirements for regime detection.

        Need enough transitions to reliably estimate regime parameters.

        Args:
            n_regimes: Number of regimes to detect
            min_per_regime: Minimum observations per regime
            expected_dwell: Expected average regime duration

        Returns:
            Dictionary with requirements
        """
        # Need enough observations in each regime
        min_total_per_regime = min_per_regime * n_regimes

        # Need enough transitions
        min_transitions = 2 * n_regimes  # At least 2 visits to each regime
        required_n = expected_dwell * (min_transitions + 1)

        # Take maximum
        required_n = max(required_n, min_total_per_regime)

        return {
            'n_regimes': n_regimes,
            'min_per_regime': min_per_regime,
            'expected_dwell': expected_dwell,
            'min_transitions': min_transitions,
            'required_n': required_n,
            'recommendations': [
                f"Need at least {required_n} observations for {n_regimes}-regime model",
                f"Expect ~{required_n // expected_dwell} regime transitions",
                "Increase sample if regimes are imbalanced"
            ]
        }

    def correlation_power(self,
                          n: int,
                          r_expected: float) -> Dict[str, Any]:
        """
        Power for correlation test.

        H0: rho = 0 vs H1: rho = r_expected

        Args:
            n: Sample size
            r_expected: Expected correlation under alternative

        Returns:
            Dictionary with power analysis
        """
        try:
            from scipy import stats

            # Fisher z-transform
            z = 0.5 * np.log((1 + r_expected) / (1 - r_expected + 1e-10))
            se = 1 / np.sqrt(n - 3)

            # Non-centrality parameter
            ncp = abs(z) / se

            # Critical value
            z_crit = stats.norm.ppf(1 - self.alpha / 2)

            # Power
            power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

            # Required N for target power
            z_beta = stats.norm.ppf(self.target_power)
            required_n = int(((z_crit + z_beta) / z) ** 2) + 3

        except ImportError:
            # Fallback approximation
            power = 1 - 0.5 ** (n * r_expected ** 2 / 10)
            required_n = int(10 / r_expected ** 2) if r_expected != 0 else 1000

        return {
            'test': 'correlation',
            'current_n': n,
            'expected_effect': r_expected,
            'achieved_power': round(power, 4),
            'target_power': self.target_power,
            'is_sufficient': power >= self.target_power,
            'required_n': required_n,
            'recommendations': [
                f"Power: {power:.1%}" + (" (sufficient)" if power >= self.target_power else " (insufficient)"),
                f"Need N={required_n} for {self.target_power:.0%} power"
            ]
        }

    def mean_difference_power(self,
                              n: int,
                              d_expected: float,
                              sigma: float = 1.0) -> Dict[str, Any]:
        """
        Power for detecting mean difference (e.g., regime shift).

        Args:
            n: Sample size per group
            d_expected: Expected difference in means
            sigma: Standard deviation

        Returns:
            Dictionary with power analysis
        """
        try:
            from scipy import stats

            # Effect size (Cohen's d)
            effect_size = d_expected / sigma

            # Non-centrality parameter for t-test
            ncp = effect_size * np.sqrt(n / 2)

            # Degrees of freedom
            df = 2 * n - 2

            # Critical value
            t_crit = stats.t.ppf(1 - self.alpha / 2, df)

            # Power using non-central t
            power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

            # Required N per group
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            z_beta = stats.norm.ppf(self.target_power)
            required_n = int(2 * ((z_alpha + z_beta) / effect_size) ** 2) + 1

        except ImportError:
            effect_size = d_expected / sigma
            power = 1 - 0.5 ** (n * effect_size ** 2 / 4)
            required_n = int(4 / effect_size ** 2) if effect_size != 0 else 1000

        return {
            'test': 'mean_difference',
            'current_n': n,
            'expected_difference': d_expected,
            'effect_size': round(effect_size, 4),
            'achieved_power': round(power, 4),
            'target_power': self.target_power,
            'is_sufficient': power >= self.target_power,
            'required_n_per_group': required_n,
            'recommendations': [
                f"Effect size (Cohen's d): {effect_size:.2f}",
                f"Power: {power:.1%}",
                f"Need N={required_n} per group for {self.target_power:.0%} power"
            ]
        }

    def bootstrap_power(self,
                        data: np.ndarray,
                        statistic: Callable,
                        null_value: float = 0,
                        alternative: str = 'two-sided',
                        n_simulations: int = 1000,
                        n_bootstrap: int = 500) -> Dict[str, Any]:
        """
        Empirical power estimation via simulation.

        Uses bootstrap to estimate the distribution of the test statistic
        and compute power.

        Args:
            data: Observed data
            statistic: Function computing test statistic
            null_value: Value under null hypothesis
            alternative: 'two-sided', 'greater', or 'less'
            n_simulations: Number of simulations
            n_bootstrap: Bootstrap samples per simulation

        Returns:
            Dictionary with empirical power estimate
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = len(data)

        observed_stat = statistic(data)

        # Bootstrap to get confidence interval
        rng = np.random.RandomState(42)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            sample = data[rng.randint(0, n, size=n)]
            bootstrap_stats.append(statistic(sample))

        bootstrap_stats = np.array(bootstrap_stats)
        se = np.std(bootstrap_stats)

        # Simulate power
        # Under alternative, data comes from observed distribution
        # Check how often we reject null

        rejections = 0
        for _ in range(n_simulations):
            # Resample from data (alternative hypothesis)
            sample = data[rng.randint(0, n, size=n)]
            sample_stat = statistic(sample)

            # Bootstrap CI for this sample
            boot_ci = []
            for _ in range(100):  # Fewer for speed
                boot_sample = sample[rng.randint(0, n, size=n)]
                boot_ci.append(statistic(boot_sample))

            ci_lower = np.percentile(boot_ci, 2.5)
            ci_upper = np.percentile(boot_ci, 97.5)

            # Reject if null_value outside CI
            if alternative == 'two-sided':
                reject = null_value < ci_lower or null_value > ci_upper
            elif alternative == 'greater':
                reject = null_value < ci_lower
            else:
                reject = null_value > ci_upper

            if reject:
                rejections += 1

        empirical_power = rejections / n_simulations

        return {
            'test': 'bootstrap',
            'observed_statistic': round(observed_stat, 4),
            'null_value': null_value,
            'standard_error': round(se, 4),
            'achieved_power': round(empirical_power, 4),
            'target_power': self.target_power,
            'is_sufficient': empirical_power >= self.target_power,
            'n_simulations': n_simulations,
            'recommendations': [
                f"Observed statistic: {observed_stat:.4f} (SE: {se:.4f})",
                f"Empirical power: {empirical_power:.1%}",
                "Increase N if power insufficient"
            ]
        }

    def analyze(self,
                data: pd.DataFrame,
                analyses: List[str] = None) -> PowerReport:
        """
        Comprehensive power analysis for a dataset.

        Args:
            data: Time series data
            analyses: List of analyses to check ('hurst', 'coherence', 'regime', 'correlation')

        Returns:
            PowerReport
        """
        if analyses is None:
            analyses = ['hurst', 'coherence', 'regime', 'correlation']

        n = len(data)
        results = {}
        recommendations = []
        overall_sufficient = True

        if 'hurst' in analyses:
            hurst_req = self.hurst_min_samples(target_precision=0.05)
            is_sufficient = n >= hurst_req['required_n']
            results['hurst'] = PowerResult(
                analysis_type='hurst',
                current_n=n,
                required_n=hurst_req['required_n'],
                achieved_power=0.8 if is_sufficient else n / hurst_req['required_n'],
                target_power=self.target_power,
                is_sufficient=is_sufficient,
                recommendations=hurst_req['recommendations'],
                details=hurst_req
            )
            if not is_sufficient:
                overall_sufficient = False
                recommendations.append(f"Hurst: Need {hurst_req['required_n']} samples (have {n})")

        if 'coherence' in analyses:
            # Check for annual cycle (common in financial data)
            coh_req = self.coherence_min_samples(freq_of_interest=1/252)  # Annual
            is_sufficient = n >= coh_req['required_n']
            results['coherence'] = PowerResult(
                analysis_type='coherence',
                current_n=n,
                required_n=coh_req['required_n'],
                achieved_power=0.8 if is_sufficient else n / coh_req['required_n'],
                target_power=self.target_power,
                is_sufficient=is_sufficient,
                recommendations=coh_req['recommendations'],
                details=coh_req
            )
            if not is_sufficient:
                overall_sufficient = False
                recommendations.append(f"Coherence: Need {coh_req['required_n']} samples")

        if 'regime' in analyses:
            reg_req = self.regime_min_transitions(n_regimes=3)
            is_sufficient = n >= reg_req['required_n']
            results['regime'] = PowerResult(
                analysis_type='regime',
                current_n=n,
                required_n=reg_req['required_n'],
                achieved_power=0.8 if is_sufficient else n / reg_req['required_n'],
                target_power=self.target_power,
                is_sufficient=is_sufficient,
                recommendations=reg_req['recommendations'],
                details=reg_req
            )
            if not is_sufficient:
                overall_sufficient = False
                recommendations.append(f"Regime: Need {reg_req['required_n']} samples")

        if 'correlation' in analyses:
            corr_power = self.correlation_power(n, r_expected=0.3)
            results['correlation'] = PowerResult(
                analysis_type='correlation',
                current_n=n,
                required_n=corr_power['required_n'],
                achieved_power=corr_power['achieved_power'],
                target_power=self.target_power,
                is_sufficient=corr_power['is_sufficient'],
                recommendations=corr_power['recommendations'],
                details=corr_power
            )
            if not corr_power['is_sufficient']:
                overall_sufficient = False
                recommendations.append(f"Correlation: Need {corr_power['required_n']} samples")

        if overall_sufficient:
            recommendations.insert(0, "Sample size is adequate for all requested analyses")

        return PowerReport(
            n_observations=n,
            analyses=results,
            overall_sufficient=overall_sufficient,
            recommendations=recommendations
        )


def print_power_report(report: PowerReport):
    """Pretty-print power report."""
    print("=" * 60)
    print("Power Analysis Report")
    print("=" * 60)
    print(f"Sample Size: {report.n_observations}")
    print(f"Overall Sufficient: {report.overall_sufficient}")
    print()

    for name, result in report.analyses.items():
        status = "OK" if result.is_sufficient else "INSUFFICIENT"
        print(f"{name.upper()}: [{status}]")
        print(f"  Current N: {result.current_n}")
        print(f"  Required N: {result.required_n}")
        print(f"  Power: {result.achieved_power:.1%}")
        for rec in result.recommendations[:2]:
            print(f"  - {rec}")
        print()

    print("Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Power Analysis Module - Test")
    print("=" * 60)

    np.random.seed(42)

    # Create test data
    n = 500
    data = pd.DataFrame({
        'series1': np.random.randn(n),
        'series2': np.random.randn(n)
    })

    analyzer = PowerAnalyzer(target_power=0.80)

    print("\nHurst minimum samples:")
    hurst_req = analyzer.hurst_min_samples(target_precision=0.05)
    print(f"  Required: {hurst_req['required_n']}")
    print(f"  Have: {n}")
    print(f"  Sufficient: {n >= hurst_req['required_n']}")

    print("\nCoherence requirements:")
    coh_req = analyzer.coherence_min_samples(freq_of_interest=0.01)
    print(f"  Required: {coh_req['required_n']}")

    print("\nCorrelation power:")
    corr_power = analyzer.correlation_power(n, r_expected=0.2)
    print(f"  Power: {corr_power['achieved_power']:.1%}")
    print(f"  Sufficient: {corr_power['is_sufficient']}")

    print("\nFull report:")
    report = analyzer.analyze(data)
    print_power_report(report)

    print("\nTest completed!")
