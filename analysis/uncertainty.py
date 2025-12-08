"""
Uncertainty Quantification Module
=================================

Confidence intervals and standard errors for all estimates.
Critical for publication-ready analysis.

Methods:
- Bootstrap (non-parametric): General purpose
- Block Bootstrap: Preserves time series autocorrelation
- Jackknife: Leave-one-out estimation
- Parametric: When distribution is known

For every metric, we need: estimate, SE, CI_lower, CI_upper, method
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class UncertaintyResult:
    """Complete uncertainty quantification for an estimate."""
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    ci_level: float  # e.g., 0.95
    method: str
    n_samples: int  # Original sample size
    n_resamples: int  # Bootstrap iterations (if applicable)
    bias: Optional[float] = None

    def __repr__(self):
        return (f"UncertaintyResult(estimate={self.estimate:.4f}, "
                f"SE={self.standard_error:.4f}, "
                f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
                f"method={self.method})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'estimate': round(self.estimate, 6),
            'standard_error': round(self.standard_error, 6),
            'ci_lower': round(self.ci_lower, 6),
            'ci_upper': round(self.ci_upper, 6),
            'ci_level': self.ci_level,
            'method': self.method,
            'n_samples': self.n_samples,
            'n_resamples': self.n_resamples,
            'bias': round(self.bias, 6) if self.bias else None
        }


class UncertaintyEstimator:
    """
    Comprehensive uncertainty estimation toolkit.

    Wraps any statistic with proper confidence intervals.
    """

    def __init__(self,
                 ci_level: float = 0.95,
                 n_bootstrap: int = 1000,
                 random_seed: Optional[int] = None):
        """
        Initialize estimator.

        Args:
            ci_level: Confidence level (default 95%)
            n_bootstrap: Number of bootstrap iterations
            random_seed: For reproducibility
        """
        self.ci_level = ci_level
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_seed)

    def bootstrap_ci(self,
                     data: np.ndarray,
                     statistic: Callable[[np.ndarray], float],
                     method: str = 'percentile') -> UncertaintyResult:
        """
        Non-parametric bootstrap confidence interval.

        Args:
            data: 1D array of observations
            statistic: Function that computes statistic from data
            method: 'percentile', 'basic', or 'bca'

        Returns:
            UncertaintyResult with CI
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = len(data)

        # Original estimate
        theta_hat = statistic(data)

        # Bootstrap distribution
        bootstrap_estimates = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            # Resample with replacement
            indices = self.rng.randint(0, n, size=n)
            bootstrap_sample = data[indices]
            bootstrap_estimates[i] = statistic(bootstrap_sample)

        # Standard error
        se = np.std(bootstrap_estimates, ddof=1)

        # Bias
        bias = np.mean(bootstrap_estimates) - theta_hat

        # Confidence interval
        alpha = 1 - self.ci_level

        if method == 'percentile':
            ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

        elif method == 'basic':
            # Basic bootstrap (reflects around estimate)
            q_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
            q_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
            ci_lower = 2 * theta_hat - q_upper
            ci_upper = 2 * theta_hat - q_lower

        elif method == 'bca':
            # BCa (Bias-Corrected and Accelerated)
            ci_lower, ci_upper = self._bca_interval(
                data, statistic, theta_hat, bootstrap_estimates, alpha
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return UncertaintyResult(
            estimate=theta_hat,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method=f'bootstrap_{method}',
            n_samples=n,
            n_resamples=self.n_bootstrap,
            bias=bias
        )

    def block_bootstrap_ci(self,
                           data: np.ndarray,
                           statistic: Callable[[np.ndarray], float],
                           block_size: Optional[int] = None) -> UncertaintyResult:
        """
        Block bootstrap for time series data.

        Preserves autocorrelation structure by resampling blocks.

        Args:
            data: Time series data
            statistic: Function to compute statistic
            block_size: Size of blocks (auto if None)

        Returns:
            UncertaintyResult with CI
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = len(data)

        # Auto block size (Politis & White suggestion)
        if block_size is None:
            block_size = max(1, int(n ** (1/3)))

        # Original estimate
        theta_hat = statistic(data)

        # Number of blocks needed
        n_blocks = int(np.ceil(n / block_size))

        # Bootstrap
        bootstrap_estimates = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            # Sample block start indices
            block_starts = self.rng.randint(0, n - block_size + 1, size=n_blocks)

            # Build resampled series
            resampled = []
            for start in block_starts:
                resampled.extend(data[start:start + block_size])
            resampled = np.array(resampled[:n])  # Trim to original length

            bootstrap_estimates[i] = statistic(resampled)

        # Standard error and CI
        se = np.std(bootstrap_estimates, ddof=1)
        bias = np.mean(bootstrap_estimates) - theta_hat

        alpha = 1 - self.ci_level
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

        return UncertaintyResult(
            estimate=theta_hat,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method=f'block_bootstrap_b{block_size}',
            n_samples=n,
            n_resamples=self.n_bootstrap,
            bias=bias
        )

    def jackknife_ci(self,
                     data: np.ndarray,
                     statistic: Callable[[np.ndarray], float]) -> UncertaintyResult:
        """
        Jackknife (leave-one-out) confidence interval.

        Good for estimating bias and variance with fewer resamples.

        Args:
            data: 1D array
            statistic: Function to compute statistic

        Returns:
            UncertaintyResult with CI
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        n = len(data)

        # Original estimate
        theta_hat = statistic(data)

        # Leave-one-out estimates
        jackknife_estimates = np.zeros(n)
        for i in range(n):
            loo_data = np.delete(data, i)
            jackknife_estimates[i] = statistic(loo_data)

        # Jackknife estimate of statistic
        theta_jack = np.mean(jackknife_estimates)

        # Bias estimate
        bias = (n - 1) * (theta_jack - theta_hat)

        # Bias-corrected estimate
        theta_corrected = theta_hat - bias

        # Variance estimate
        var_jack = ((n - 1) / n) * np.sum((jackknife_estimates - theta_jack) ** 2)
        se = np.sqrt(var_jack)

        # CI using normal approximation
        try:
            from scipy import stats
            z = stats.norm.ppf(1 - (1 - self.ci_level) / 2)
        except ImportError:
            z = 1.96  # Fallback for 95% CI

        ci_lower = theta_corrected - z * se
        ci_upper = theta_corrected + z * se

        return UncertaintyResult(
            estimate=theta_corrected,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method='jackknife',
            n_samples=n,
            n_resamples=n,
            bias=bias
        )

    def parametric_ci(self,
                      estimate: float,
                      standard_error: float,
                      n: int,
                      distribution: str = 'normal') -> UncertaintyResult:
        """
        Parametric confidence interval.

        Use when distribution of estimator is known.

        Args:
            estimate: Point estimate
            standard_error: Standard error of estimate
            n: Sample size
            distribution: 'normal' or 't'

        Returns:
            UncertaintyResult with CI
        """
        try:
            from scipy import stats
            alpha = 1 - self.ci_level

            if distribution == 'normal':
                z = stats.norm.ppf(1 - alpha / 2)
                ci_lower = estimate - z * standard_error
                ci_upper = estimate + z * standard_error
            elif distribution == 't':
                t = stats.t.ppf(1 - alpha / 2, df=n - 1)
                ci_lower = estimate - t * standard_error
                ci_upper = estimate + t * standard_error
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
        except ImportError:
            # Fallback
            z = 1.96
            ci_lower = estimate - z * standard_error
            ci_upper = estimate + z * standard_error
            distribution = 'normal_fallback'

        return UncertaintyResult(
            estimate=estimate,
            standard_error=standard_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method=f'parametric_{distribution}',
            n_samples=n,
            n_resamples=0
        )

    def _bca_interval(self,
                      data: np.ndarray,
                      statistic: Callable,
                      theta_hat: float,
                      bootstrap_estimates: np.ndarray,
                      alpha: float) -> Tuple[float, float]:
        """
        BCa (Bias-Corrected and Accelerated) confidence interval.

        More accurate than percentile for skewed distributions.
        """
        try:
            from scipy import stats
        except ImportError:
            # Fallback to percentile
            ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
            return ci_lower, ci_upper

        n = len(data)

        # Bias correction factor
        z0 = stats.norm.ppf(np.mean(bootstrap_estimates <= theta_hat))

        # Acceleration factor (from jackknife)
        jackknife_estimates = np.zeros(n)
        for i in range(n):
            loo_data = np.delete(data, i)
            jackknife_estimates[i] = statistic(loo_data)

        theta_dot = np.mean(jackknife_estimates)
        num = np.sum((theta_dot - jackknife_estimates) ** 3)
        den = 6 * (np.sum((theta_dot - jackknife_estimates) ** 2) ** 1.5)
        a = num / den if den != 0 else 0

        # Adjusted percentiles
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1_alpha = stats.norm.ppf(1 - alpha / 2)

        def adjust_percentile(z):
            return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

        p_lower = adjust_percentile(z_alpha)
        p_upper = adjust_percentile(z_1_alpha)

        # Clamp to valid range
        p_lower = np.clip(p_lower, 0.001, 0.999)
        p_upper = np.clip(p_upper, 0.001, 0.999)

        ci_lower = np.percentile(bootstrap_estimates, 100 * p_lower)
        ci_upper = np.percentile(bootstrap_estimates, 100 * p_upper)

        return ci_lower, ci_upper

    def ci_for_correlation(self,
                           r: float,
                           n: int) -> UncertaintyResult:
        """
        Confidence interval for correlation coefficient.

        Uses Fisher z-transformation.

        Args:
            r: Sample correlation
            n: Sample size

        Returns:
            UncertaintyResult
        """
        try:
            from scipy import stats
            z_crit = stats.norm.ppf(1 - (1 - self.ci_level) / 2)
        except ImportError:
            z_crit = 1.96

        # Fisher z-transform
        z = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
        se_z = 1 / np.sqrt(n - 3)

        # CI in z-space
        z_lower = z - z_crit * se_z
        z_upper = z + z_crit * se_z

        # Transform back
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        # SE in original scale (delta method approximation)
        se_r = (1 - r**2) / np.sqrt(n - 3)

        return UncertaintyResult(
            estimate=r,
            standard_error=se_r,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method='fisher_z',
            n_samples=n,
            n_resamples=0
        )

    def ci_for_proportion(self,
                          p: float,
                          n: int,
                          method: str = 'wilson') -> UncertaintyResult:
        """
        Confidence interval for proportion.

        Args:
            p: Sample proportion
            n: Sample size
            method: 'wilson' (recommended), 'wald', or 'clopper_pearson'

        Returns:
            UncertaintyResult
        """
        try:
            from scipy import stats
            alpha = 1 - self.ci_level
            z = stats.norm.ppf(1 - alpha / 2)
        except ImportError:
            alpha = 0.05
            z = 1.96

        if method == 'wald':
            se = np.sqrt(p * (1 - p) / n)
            ci_lower = p - z * se
            ci_upper = p + z * se

        elif method == 'wilson':
            # Wilson score interval (better coverage)
            denom = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denom
            margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denom
            ci_lower = center - margin
            ci_upper = center + margin
            se = margin / z

        elif method == 'clopper_pearson':
            # Exact binomial (conservative)
            try:
                k = int(p * n)
                if k == 0:
                    ci_lower = 0
                else:
                    ci_lower = stats.beta.ppf(alpha / 2, k, n - k + 1)
                if k == n:
                    ci_upper = 1
                else:
                    ci_upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
            except:
                ci_lower = p - z * np.sqrt(p * (1 - p) / n)
                ci_upper = p + z * np.sqrt(p * (1 - p) / n)
            se = np.sqrt(p * (1 - p) / n)
        else:
            raise ValueError(f"Unknown method: {method}")

        ci_lower = np.clip(ci_lower, 0, 1)
        ci_upper = np.clip(ci_upper, 0, 1)

        return UncertaintyResult(
            estimate=p,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method=f'proportion_{method}',
            n_samples=n,
            n_resamples=0
        )


# =============================================================================
# Wrapper functions for common statistics
# =============================================================================

def mean_with_ci(data: np.ndarray,
                 ci_level: float = 0.95,
                 method: str = 'parametric') -> UncertaintyResult:
    """Compute mean with confidence interval."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    estimate = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(len(data))

    estimator = UncertaintyEstimator(ci_level=ci_level)

    if method == 'parametric':
        return estimator.parametric_ci(estimate, se, len(data), distribution='t')
    else:
        return estimator.bootstrap_ci(data, np.mean)


def correlation_with_ci(x: np.ndarray,
                        y: np.ndarray,
                        ci_level: float = 0.95) -> UncertaintyResult:
    """Compute correlation with confidence interval."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    r = np.corrcoef(x, y)[0, 1]

    estimator = UncertaintyEstimator(ci_level=ci_level)
    return estimator.ci_for_correlation(r, len(x))


def hurst_with_ci(data: np.ndarray,
                  ci_level: float = 0.95,
                  n_bootstrap: int = 500) -> UncertaintyResult:
    """
    Compute Hurst exponent with bootstrap confidence interval.

    Uses block bootstrap to preserve autocorrelation.
    """
    # Import or define Hurst computation
    def compute_hurst_rs(series):
        """Simple R/S Hurst computation."""
        n = len(series)
        if n < 100:
            return 0.5

        max_k = int(np.log2(n / 4))
        min_k = 2

        if max_k <= min_k:
            return 0.5

        window_sizes = [2**k for k in range(min_k, max_k + 1)]
        rs_values = []

        for window in window_sizes:
            if window > n // 2:
                continue

            n_windows = n // window
            rs_list = []

            for i in range(n_windows):
                start = i * window
                end = start + window
                subseries = series[start:end]

                mean_val = np.mean(subseries)
                adjusted = subseries - mean_val
                cumdev = np.cumsum(adjusted)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(subseries, ddof=1)

                if S > 0:
                    rs_list.append(R / S)

            if rs_list:
                rs_values.append((window, np.mean(rs_list)))

        if len(rs_values) < 3:
            return 0.5

        log_n = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])

        slope, _ = np.polyfit(log_n, log_rs, 1)
        return np.clip(slope, 0, 1)

    estimator = UncertaintyEstimator(ci_level=ci_level, n_bootstrap=n_bootstrap)
    return estimator.block_bootstrap_ci(data, compute_hurst_rs)


# =============================================================================
# Batch processing
# =============================================================================

def add_uncertainty_to_results(results: Dict[str, float],
                               data: pd.DataFrame,
                               statistic_funcs: Dict[str, Callable],
                               method: str = 'block_bootstrap') -> Dict[str, UncertaintyResult]:
    """
    Add confidence intervals to a dictionary of results.

    Args:
        results: Dictionary of {metric_name: value}
        data: Original data used
        statistic_funcs: Dictionary of {metric_name: function}
        method: Bootstrap method

    Returns:
        Dictionary of {metric_name: UncertaintyResult}
    """
    estimator = UncertaintyEstimator()
    output = {}

    for name, value in results.items():
        if name in statistic_funcs:
            func = statistic_funcs[name]
            if method == 'block_bootstrap':
                output[name] = estimator.block_bootstrap_ci(
                    data.values.flatten() if isinstance(data, pd.DataFrame) else data,
                    func
                )
            else:
                output[name] = estimator.bootstrap_ci(
                    data.values.flatten() if isinstance(data, pd.DataFrame) else data,
                    func
                )
        else:
            # No function provided, create placeholder
            output[name] = UncertaintyResult(
                estimate=value,
                standard_error=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                ci_level=0.95,
                method='none',
                n_samples=len(data),
                n_resamples=0
            )

    return output


if __name__ == "__main__":
    print("=" * 60)
    print("Uncertainty Quantification Module - Test")
    print("=" * 60)

    np.random.seed(42)

    # Test data
    n = 200
    normal_data = np.random.randn(n) * 2 + 5  # mean=5, std=2

    # AR(1) process (autocorrelated)
    ar1_data = np.zeros(n)
    for i in range(1, n):
        ar1_data[i] = 0.7 * ar1_data[i-1] + np.random.randn()

    estimator = UncertaintyEstimator(n_bootstrap=1000, random_seed=42)

    print("\nTest 1: Mean of Normal Data (true mean = 5)")
    print("-" * 40)

    # Parametric
    result = mean_with_ci(normal_data, method='parametric')
    print(f"Parametric: {result}")

    # Bootstrap
    result = estimator.bootstrap_ci(normal_data, np.mean, method='percentile')
    print(f"Bootstrap (percentile): {result}")

    print("\nTest 2: Mean of AR(1) Data (autocorrelated)")
    print("-" * 40)

    # Regular bootstrap (biased SE)
    result_wrong = estimator.bootstrap_ci(ar1_data, np.mean)
    print(f"Regular bootstrap: {result_wrong}")

    # Block bootstrap (correct SE)
    result_block = estimator.block_bootstrap_ci(ar1_data, np.mean)
    print(f"Block bootstrap: {result_block}")

    print("\nTest 3: Correlation")
    print("-" * 40)

    x = np.random.randn(100)
    y = 0.7 * x + 0.3 * np.random.randn(100)

    result = correlation_with_ci(x, y)
    print(f"Correlation: {result}")

    print("\nTest completed!")
