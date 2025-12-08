"""
Stationarity Testing Module
===========================

Comprehensive stationarity analysis for time series.
Critical for establishing valid statistical inference.

Tests Included:
- ADF (Augmented Dickey-Fuller): Null = unit root
- KPSS: Null = stationary (opposite of ADF)
- Phillips-Perron: Robust to heteroskedasticity
- Variance Ratio: Tests random walk hypothesis

Integration Order Determination:
- I(0): Stationary
- I(1): First difference stationary
- I(2): Second difference stationary

For publication, every series must have documented stationarity status.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Statistical imports
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.stattools import acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available - using fallback implementations")


class IntegrationOrder(Enum):
    """Integration order classification."""
    I0 = 0  # Stationary
    I1 = 1  # Unit root, first difference stationary
    I2 = 2  # Needs second differencing
    UNKNOWN = -1


@dataclass
class StationarityTestResult:
    """Result from a single stationarity test."""
    test_name: str
    statistic: float
    p_value: float
    critical_values: Dict[str, float]
    null_hypothesis: str
    reject_null: bool
    conclusion: str
    lags_used: Optional[int] = None


@dataclass
class StationarityReport:
    """Comprehensive stationarity analysis for one series."""
    series_name: str
    n_observations: int

    # Test results
    adf_result: Optional[StationarityTestResult] = None
    kpss_result: Optional[StationarityTestResult] = None
    pp_result: Optional[StationarityTestResult] = None

    # Conclusions
    integration_order: IntegrationOrder = IntegrationOrder.UNKNOWN
    is_stationary: bool = False
    recommended_transform: str = "none"
    confidence: str = "low"  # low, medium, high

    # Diagnostics
    has_trend: bool = False
    has_seasonality: bool = False
    structural_breaks: List[int] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)


class StationarityAnalyzer:
    """
    Comprehensive stationarity testing suite.

    For academic rigor, we run multiple tests with different null hypotheses
    and synthesize results.
    """

    def __init__(self,
                 significance_level: float = 0.05,
                 max_lags: Optional[int] = None):
        """
        Initialize analyzer.

        Args:
            significance_level: Alpha for hypothesis tests
            max_lags: Maximum lags for ADF (None = auto)
        """
        self.alpha = significance_level
        self.max_lags = max_lags

        if not STATSMODELS_AVAILABLE:
            print("Warning: Full functionality requires statsmodels")

    def adf_test(self,
                 series: np.ndarray,
                 regression: str = 'c') -> StationarityTestResult:
        """
        Augmented Dickey-Fuller test.

        Null hypothesis: Series has a unit root (non-stationary)

        Args:
            series: Time series data
            regression: 'c' (constant), 'ct' (constant + trend), 'n' (none)

        Returns:
            StationarityTestResult
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]

        if not STATSMODELS_AVAILABLE:
            return self._fallback_adf(series)

        result = adfuller(
            series,
            maxlag=self.max_lags,
            regression=regression,
            autolag='AIC'
        )

        statistic = result[0]
        p_value = result[1]
        lags = result[2]
        critical_values = {
            '1%': result[4]['1%'],
            '5%': result[4]['5%'],
            '10%': result[4]['10%']
        }

        reject_null = p_value < self.alpha

        if reject_null:
            conclusion = "Stationary (reject unit root)"
        else:
            conclusion = "Non-stationary (cannot reject unit root)"

        return StationarityTestResult(
            test_name="Augmented Dickey-Fuller",
            statistic=round(statistic, 4),
            p_value=round(p_value, 4),
            critical_values={k: round(v, 4) for k, v in critical_values.items()},
            null_hypothesis="Series has unit root",
            reject_null=reject_null,
            conclusion=conclusion,
            lags_used=lags
        )

    def kpss_test(self,
                  series: np.ndarray,
                  regression: str = 'c') -> StationarityTestResult:
        """
        KPSS test (Kwiatkowski-Phillips-Schmidt-Shin).

        Null hypothesis: Series is stationary
        (Opposite of ADF - useful for confirmation)

        Args:
            series: Time series data
            regression: 'c' (level stationary), 'ct' (trend stationary)

        Returns:
            StationarityTestResult
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]

        if not STATSMODELS_AVAILABLE:
            return self._fallback_kpss(series)

        # KPSS can raise warnings about p-value bounds
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(series, regression=regression, nlags='auto')

        statistic = result[0]
        p_value = result[1]
        lags = result[2]
        critical_values = {
            '1%': result[3]['1%'],
            '5%': result[3]['5%'],
            '10%': result[3]['10%']
        }

        reject_null = p_value < self.alpha

        if reject_null:
            conclusion = "Non-stationary (reject stationarity)"
        else:
            conclusion = "Stationary (cannot reject stationarity)"

        return StationarityTestResult(
            test_name="KPSS",
            statistic=round(statistic, 4),
            p_value=round(p_value, 4),
            critical_values={k: round(v, 4) for k, v in critical_values.items()},
            null_hypothesis="Series is stationary",
            reject_null=reject_null,
            conclusion=conclusion,
            lags_used=lags
        )

    def phillips_perron_test(self, series: np.ndarray) -> StationarityTestResult:
        """
        Phillips-Perron test.

        Similar to ADF but robust to heteroskedasticity and
        serial correlation without adding lagged differences.

        Note: Uses ADF as approximation if PP not available.
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]

        # statsmodels doesn't have PP directly, use ADF with different lag structure
        # For full PP, would need arch package or manual implementation

        if STATSMODELS_AVAILABLE:
            # Use ADF with Newey-West style lag selection as PP approximation
            result = adfuller(series, maxlag=int(4 * (len(series)/100)**0.25),
                            regression='c', autolag=None)

            statistic = result[0]
            p_value = result[1]
            critical_values = {
                '1%': result[4]['1%'],
                '5%': result[4]['5%'],
                '10%': result[4]['10%']
            }
        else:
            return self._fallback_pp(series)

        reject_null = p_value < self.alpha
        conclusion = "Stationary" if reject_null else "Non-stationary"

        return StationarityTestResult(
            test_name="Phillips-Perron (ADF approx)",
            statistic=round(statistic, 4),
            p_value=round(p_value, 4),
            critical_values={k: round(v, 4) for k, v in critical_values.items()},
            null_hypothesis="Series has unit root",
            reject_null=reject_null,
            conclusion=conclusion
        )

    def variance_ratio_test(self,
                            series: np.ndarray,
                            periods: List[int] = [2, 4, 8, 16]) -> Dict[str, Any]:
        """
        Variance ratio test for random walk hypothesis.

        Under random walk, Var(k-period return) = k * Var(1-period return)
        Ratio should be 1.0 for pure random walk.

        Args:
            series: Price or level series
            periods: Holding periods to test

        Returns:
            Dictionary with VR statistics for each period
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]

        # Log returns
        if np.all(series > 0):
            returns = np.diff(np.log(series))
        else:
            returns = np.diff(series)

        n = len(returns)
        var_1 = np.var(returns, ddof=1)

        results = {}
        for k in periods:
            if k >= n:
                continue

            # k-period returns
            k_returns = np.array([
                np.sum(returns[i:i+k])
                for i in range(0, n - k + 1, k)
            ])

            var_k = np.var(k_returns, ddof=1)

            # Variance ratio
            vr = var_k / (k * var_1) if var_1 > 0 else 1.0

            # Under null (random walk), VR ~ 1
            # Z-statistic (Lo-MacKinlay)
            # Simplified - full version needs heteroskedasticity correction
            se = np.sqrt(2 * (2*k - 1) * (k - 1) / (3 * k * n))
            z_stat = (vr - 1) / se if se > 0 else 0

            results[f'VR({k})'] = {
                'ratio': round(vr, 4),
                'z_statistic': round(z_stat, 4),
                'interpretation': 'mean_reverting' if vr < 0.8 else
                                 'random_walk' if 0.8 <= vr <= 1.2 else 'trending'
            }

        return results

    def determine_integration_order(self,
                                    series: np.ndarray,
                                    max_order: int = 2) -> Tuple[IntegrationOrder, List[StationarityTestResult]]:
        """
        Determine integration order I(d) of series.

        Tests series, then first difference, then second difference
        until stationarity is achieved.

        Args:
            series: Original series
            max_order: Maximum differencing to try

        Returns:
            Tuple of (IntegrationOrder, list of test results)
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]

        test_results = []
        current = series.copy()

        for d in range(max_order + 1):
            if d > 0:
                current = np.diff(current)

            if len(current) < 50:
                break

            adf = self.adf_test(current)
            kpss = self.kpss_test(current)

            test_results.append({
                'difference_order': d,
                'adf': adf,
                'kpss': kpss
            })

            # Both tests agree on stationarity
            if adf.reject_null and not kpss.reject_null:
                return IntegrationOrder(d), test_results

            # ADF says stationary (conservative)
            if adf.reject_null:
                return IntegrationOrder(d), test_results

        return IntegrationOrder.UNKNOWN, test_results

    def analyze(self,
                series: Union[np.ndarray, pd.Series],
                series_name: str = "series") -> StationarityReport:
        """
        Comprehensive stationarity analysis.

        Args:
            series: Time series data
            series_name: Name for reporting

        Returns:
            StationarityReport with all test results and recommendations
        """
        if isinstance(series, pd.Series):
            series_name = series.name or series_name
            series = series.values

        series = np.asarray(series)
        series = series[~np.isnan(series)]
        n = len(series)

        warnings = []

        # Check sample size
        if n < 50:
            warnings.append(f"Small sample size ({n}). Results may be unreliable.")

        # Run tests on original series
        adf_result = self.adf_test(series)
        kpss_result = self.kpss_test(series)
        pp_result = self.phillips_perron_test(series)

        # Determine integration order
        integration_order, _ = self.determine_integration_order(series)

        # Synthesize results
        # Conservative: require multiple tests to agree
        adf_stationary = adf_result.reject_null
        kpss_stationary = not kpss_result.reject_null

        if adf_stationary and kpss_stationary:
            is_stationary = True
            confidence = "high"
        elif adf_stationary or kpss_stationary:
            is_stationary = adf_stationary  # Favor ADF
            confidence = "medium"
            warnings.append("ADF and KPSS disagree. Using ADF result.")
        else:
            is_stationary = False
            confidence = "high"

        # Determine recommended transform
        if is_stationary:
            recommended_transform = "none"
        elif integration_order == IntegrationOrder.I1:
            recommended_transform = "first_difference"
        elif integration_order == IntegrationOrder.I2:
            recommended_transform = "second_difference"
            warnings.append("I(2) series are rare in economics. Check for structural breaks.")
        else:
            recommended_transform = "first_difference"  # Default
            warnings.append("Integration order unclear. Defaulting to first difference.")

        # Check for trend
        if len(series) > 20:
            trend_slope = np.polyfit(np.arange(len(series)), series, 1)[0]
            has_trend = abs(trend_slope) > np.std(series) / len(series)
        else:
            has_trend = False

        return StationarityReport(
            series_name=series_name,
            n_observations=n,
            adf_result=adf_result,
            kpss_result=kpss_result,
            pp_result=pp_result,
            integration_order=integration_order,
            is_stationary=is_stationary,
            recommended_transform=recommended_transform,
            confidence=confidence,
            has_trend=has_trend,
            warnings=warnings
        )

    def analyze_panel(self,
                      data: pd.DataFrame) -> Dict[str, StationarityReport]:
        """
        Analyze stationarity for all columns in a panel.

        Args:
            data: DataFrame with indicators as columns

        Returns:
            Dictionary mapping column names to StationarityReports
        """
        results = {}
        for col in data.columns:
            results[col] = self.analyze(data[col], series_name=col)
        return results

    def generate_manifest(self,
                          data: pd.DataFrame,
                          output_format: str = 'dict') -> Union[Dict, str]:
        """
        Generate stationarity manifest for all indicators.

        Args:
            data: Panel DataFrame
            output_format: 'dict' or 'yaml'

        Returns:
            Manifest as dictionary or YAML string
        """
        reports = self.analyze_panel(data)

        manifest = {'indicators': {}}

        for name, report in reports.items():
            manifest['indicators'][name] = {
                'n_observations': report.n_observations,
                'adf_statistic': report.adf_result.statistic if report.adf_result else None,
                'adf_pvalue': report.adf_result.p_value if report.adf_result else None,
                'kpss_statistic': report.kpss_result.statistic if report.kpss_result else None,
                'kpss_pvalue': report.kpss_result.p_value if report.kpss_result else None,
                'integration_order': report.integration_order.value,
                'is_stationary': report.is_stationary,
                'recommended_transform': report.recommended_transform,
                'confidence': report.confidence,
                'has_trend': report.has_trend,
                'warnings': report.warnings
            }

        if output_format == 'yaml':
            import yaml
            return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

        return manifest

    # =========================================================================
    # Fallback implementations (when statsmodels unavailable)
    # =========================================================================

    def _fallback_adf(self, series: np.ndarray) -> StationarityTestResult:
        """Simple ADF approximation without statsmodels."""
        n = len(series)

        # First difference
        diff = np.diff(series)

        # Regress diff on lagged level
        y = diff[1:]
        x = series[1:-1]

        # OLS
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)

        # t-statistic (simplified)
        residuals = y - (y_mean + beta * (x - x_mean))
        se = np.sqrt(np.sum(residuals**2) / (n - 3)) / np.sqrt(np.sum((x - x_mean)**2))
        t_stat = beta / se if se > 0 else 0

        # Approximate critical values (from Fuller tables)
        critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}

        reject = t_stat < critical_values['5%']

        return StationarityTestResult(
            test_name="ADF (fallback)",
            statistic=round(t_stat, 4),
            p_value=0.05 if reject else 0.10,  # Approximate
            critical_values=critical_values,
            null_hypothesis="Series has unit root",
            reject_null=reject,
            conclusion="Stationary" if reject else "Non-stationary"
        )

    def _fallback_kpss(self, series: np.ndarray) -> StationarityTestResult:
        """Simple KPSS approximation without statsmodels."""
        n = len(series)

        # Detrend
        mean = np.mean(series)
        residuals = series - mean

        # Cumulative sum of residuals
        S = np.cumsum(residuals)

        # KPSS statistic
        sigma2 = np.sum(residuals**2) / n
        eta = np.sum(S**2) / (n**2 * sigma2) if sigma2 > 0 else 0

        # Approximate critical values
        critical_values = {'1%': 0.739, '5%': 0.463, '10%': 0.347}

        reject = eta > critical_values['5%']

        return StationarityTestResult(
            test_name="KPSS (fallback)",
            statistic=round(eta, 4),
            p_value=0.05 if reject else 0.10,
            critical_values=critical_values,
            null_hypothesis="Series is stationary",
            reject_null=reject,
            conclusion="Non-stationary" if reject else "Stationary"
        )

    def _fallback_pp(self, series: np.ndarray) -> StationarityTestResult:
        """Use ADF fallback for PP."""
        result = self._fallback_adf(series)
        result.test_name = "Phillips-Perron (fallback)"
        return result


# =============================================================================
# Convenience functions
# =============================================================================

def test_stationarity(series: np.ndarray) -> bool:
    """Quick stationarity check."""
    analyzer = StationarityAnalyzer()
    report = analyzer.analyze(series)
    return report.is_stationary


def get_integration_order(series: np.ndarray) -> int:
    """Get integration order as integer."""
    analyzer = StationarityAnalyzer()
    order, _ = analyzer.determine_integration_order(series)
    return order.value if order != IntegrationOrder.UNKNOWN else -1


# =============================================================================
# Formatting / Printing
# =============================================================================

def print_stationarity_report(report: StationarityReport):
    """Pretty-print a stationarity report."""
    print("=" * 60)
    print(f"Stationarity Report: {report.series_name}")
    print("=" * 60)
    print(f"Observations: {report.n_observations}")
    print()

    if report.adf_result:
        adf = report.adf_result
        print(f"ADF Test:")
        print(f"  Statistic: {adf.statistic}")
        print(f"  P-value: {adf.p_value}")
        print(f"  Conclusion: {adf.conclusion}")
        print()

    if report.kpss_result:
        kpss = report.kpss_result
        print(f"KPSS Test:")
        print(f"  Statistic: {kpss.statistic}")
        print(f"  P-value: {kpss.p_value}")
        print(f"  Conclusion: {kpss.conclusion}")
        print()

    print(f"Integration Order: I({report.integration_order.value})")
    print(f"Is Stationary: {report.is_stationary}")
    print(f"Confidence: {report.confidence}")
    print(f"Recommended Transform: {report.recommended_transform}")

    if report.warnings:
        print()
        print("Warnings:")
        for w in report.warnings:
            print(f"  - {w}")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Stationarity Testing Module - Test")
    print("=" * 60)

    np.random.seed(42)
    n = 500

    # Create test series
    # 1. Stationary (AR1)
    stationary = np.zeros(n)
    for i in range(1, n):
        stationary[i] = 0.5 * stationary[i-1] + np.random.randn()

    # 2. Random walk (I(1))
    random_walk = np.cumsum(np.random.randn(n))

    # 3. Trend stationary
    trend_stationary = 0.05 * np.arange(n) + np.random.randn(n)

    panel = pd.DataFrame({
        'stationary_ar1': stationary,
        'random_walk': random_walk,
        'trend_stationary': trend_stationary,
    })

    print(f"\nTest panel: {panel.shape[0]} rows, {panel.shape[1]} columns")
    print(f"statsmodels available: {STATSMODELS_AVAILABLE}")

    # Run analysis
    analyzer = StationarityAnalyzer()

    for col in panel.columns:
        report = analyzer.analyze(panel[col], col)
        print()
        print_stationarity_report(report)

    print("\nTest completed!")
