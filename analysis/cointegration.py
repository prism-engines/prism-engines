"""
Cointegration Testing Module
============================

Tests for long-run equilibrium relationships between non-stationary series.
Critical for any claims about long-term cross-domain relationships.

Methods:
- Engle-Granger: Two-step procedure for pairs
- Johansen: Maximum likelihood for multivariate systems
- Bounds Test: ARDL bounds for mixed I(0)/I(1) series

Key concept: Two I(1) series are cointegrated if their linear combination is I(0).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class CointegrationRank(Enum):
    """Cointegration rank."""
    NONE = 0
    ONE = 1
    TWO = 2
    MULTIPLE = 3
    FULL = -1


@dataclass
class EngleGrangerResult:
    """Result from Engle-Granger cointegration test."""
    test_name: str = "Engle-Granger"
    series_1: str = ""
    series_2: str = ""
    adf_statistic: float = 0.0
    p_value: float = 1.0
    critical_values: Dict[str, float] = field(default_factory=dict)
    cointegrated: bool = False
    cointegrating_vector: List[float] = field(default_factory=list)
    conclusion: str = ""


@dataclass
class JohansenResult:
    """Result from Johansen cointegration test."""
    test_name: str = "Johansen"
    n_series: int = 0
    max_eigenvalue_stats: List[float] = field(default_factory=list)
    trace_stats: List[float] = field(default_factory=list)
    critical_values_max: Dict[int, Dict[str, float]] = field(default_factory=dict)
    critical_values_trace: Dict[int, Dict[str, float]] = field(default_factory=dict)
    rank: int = 0
    cointegrating_vectors: Optional[np.ndarray] = None
    conclusion: str = ""


@dataclass
class CointegrationReport:
    """Comprehensive cointegration analysis."""
    series_names: List[str]
    n_series: int
    n_observations: int

    # Individual series properties
    integration_orders: Dict[str, int] = field(default_factory=dict)

    # Test results
    eg_result: Optional[EngleGrangerResult] = None
    johansen_result: Optional[JohansenResult] = None

    # Conclusions
    has_cointegration: bool = False
    cointegration_rank: int = 0
    equilibrium_relationships: List[Dict] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)


class CointegrationAnalyzer:
    """
    Comprehensive cointegration testing suite.

    Use this to validate any long-run relationship claims between series.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize analyzer.

        Args:
            significance_level: Alpha for hypothesis tests
        """
        self.alpha = significance_level

        if not STATSMODELS_AVAILABLE:
            print("Warning: statsmodels required for full cointegration testing")

    def engle_granger_test(self,
                           y1: np.ndarray,
                           y2: np.ndarray,
                           name1: str = "y1",
                           name2: str = "y2") -> EngleGrangerResult:
        """
        Engle-Granger two-step cointegration test.

        Step 1: Estimate cointegrating regression y1 = a + b*y2 + e
        Step 2: Test residuals for unit root

        Args:
            y1, y2: Two time series
            name1, name2: Series names for reporting

        Returns:
            EngleGrangerResult
        """
        y1 = np.asarray(y1).flatten()
        y2 = np.asarray(y2).flatten()

        # Remove NaN
        mask = ~(np.isnan(y1) | np.isnan(y2))
        y1, y2 = y1[mask], y2[mask]

        if not STATSMODELS_AVAILABLE:
            return self._fallback_eg(y1, y2, name1, name2)

        # Use statsmodels coint function
        stat, pvalue, crit_values = coint(y1, y2, trend='c', autolag='AIC')

        cointegrated = pvalue < self.alpha

        # Estimate cointegrating vector via OLS
        X = np.column_stack([np.ones(len(y2)), y2])
        beta = np.linalg.lstsq(X, y1, rcond=None)[0]

        if cointegrated:
            conclusion = f"Series are cointegrated (p={pvalue:.4f})"
        else:
            conclusion = f"No cointegration found (p={pvalue:.4f})"

        return EngleGrangerResult(
            series_1=name1,
            series_2=name2,
            adf_statistic=round(stat, 4),
            p_value=round(pvalue, 4),
            critical_values={
                '1%': round(crit_values[0], 4),
                '5%': round(crit_values[1], 4),
                '10%': round(crit_values[2], 4)
            },
            cointegrated=cointegrated,
            cointegrating_vector=[round(b, 4) for b in beta],
            conclusion=conclusion
        )

    def johansen_test(self,
                      data: Union[pd.DataFrame, np.ndarray],
                      det_order: int = 0,
                      k_ar_diff: int = 1) -> JohansenResult:
        """
        Johansen cointegration test for multiple series.

        Uses maximum likelihood to determine cointegration rank
        and estimate cointegrating vectors.

        Args:
            data: DataFrame or array with series as columns
            det_order: Deterministic trend order (-1, 0, 1)
            k_ar_diff: Number of lagged differences in VECM

        Returns:
            JohansenResult
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.asarray(data)

        # Remove rows with NaN
        mask = ~np.any(np.isnan(data_array), axis=1)
        data_array = data_array[mask]

        n_series = data_array.shape[1]

        if not STATSMODELS_AVAILABLE:
            return self._fallback_johansen(data_array, n_series)

        # Run Johansen test
        result = coint_johansen(data_array, det_order=det_order, k_ar_diff=k_ar_diff)

        # Extract results
        trace_stats = result.lr1.tolist()  # Trace statistics
        max_eigen_stats = result.lr2.tolist()  # Max eigenvalue statistics

        # Critical values
        trace_cv = {}
        max_cv = {}
        for r in range(n_series):
            trace_cv[r] = {
                '90%': result.cvt[r, 0],
                '95%': result.cvt[r, 1],
                '99%': result.cvt[r, 2]
            }
            max_cv[r] = {
                '90%': result.cvm[r, 0],
                '95%': result.cvm[r, 1],
                '99%': result.cvm[r, 2]
            }

        # Determine rank using trace statistic at 95%
        rank = 0
        for r in range(n_series):
            if trace_stats[r] > trace_cv[r]['95%']:
                rank = r + 1

        # Cointegrating vectors
        coint_vectors = result.evec[:, :rank] if rank > 0 else None

        if rank == 0:
            conclusion = "No cointegration (rank = 0)"
        elif rank == n_series:
            conclusion = f"Full rank ({rank}) - all series stationary"
        else:
            conclusion = f"Cointegration rank = {rank}"

        return JohansenResult(
            n_series=n_series,
            max_eigenvalue_stats=[round(s, 4) for s in max_eigen_stats],
            trace_stats=[round(s, 4) for s in trace_stats],
            critical_values_max=max_cv,
            critical_values_trace=trace_cv,
            rank=rank,
            cointegrating_vectors=coint_vectors,
            conclusion=conclusion
        )

    def bounds_test(self,
                    y: np.ndarray,
                    x: np.ndarray,
                    max_lag: int = 4) -> Dict[str, Any]:
        """
        ARDL Bounds test for cointegration.

        Valid for mixed I(0)/I(1) series.
        Tests H0: No level relationship.

        Args:
            y: Dependent variable
            x: Independent variable(s)
            max_lag: Maximum lag order

        Returns:
            Dictionary with F-statistic and bounds
        """
        y = np.asarray(y).flatten()
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Remove NaN
        mask = ~(np.isnan(y) | np.any(np.isnan(x), axis=1))
        y, x = y[mask], x[mask]

        n = len(y)

        # Estimate ARDL model and compute F-statistic
        # Simplified version - full implementation would use statsmodels ARDL

        # First differences
        dy = np.diff(y)
        dx = np.diff(x, axis=0)

        # Lagged levels
        y_lag = y[:-1]
        x_lag = x[:-1]

        # Regression: dy = c + phi*y_{-1} + theta*x_{-1} + ... + e
        X_reg = np.column_stack([
            np.ones(len(dy)),
            y_lag,
            x_lag,
            dy[:-1] if len(dy) > 1 else [],  # Simplified lag structure
        ])

        if X_reg.shape[0] > X_reg.shape[1]:
            try:
                beta = np.linalg.lstsq(X_reg[1:], dy[1:], rcond=None)[0]
                residuals = dy[1:] - X_reg[1:] @ beta

                # F-statistic (simplified)
                SSR = np.sum(residuals**2)
                SST = np.sum((dy[1:] - np.mean(dy[1:]))**2)
                R2 = 1 - SSR / SST if SST > 0 else 0

                k = X_reg.shape[1]
                n_reg = len(dy) - 1
                F_stat = (R2 / k) / ((1 - R2) / (n_reg - k - 1)) if R2 < 1 else 0
            except:
                F_stat = 0
                R2 = 0
        else:
            F_stat = 0
            R2 = 0

        # Critical values (Pesaran et al. bounds - approximate)
        # These depend on number of regressors k
        k = x.shape[1]
        bounds_critical = {
            'I0_5%': 3.79 + 0.5 * k,  # Lower bound (all I(0))
            'I1_5%': 4.85 + 0.5 * k,  # Upper bound (all I(1))
            'I0_10%': 3.17 + 0.4 * k,
            'I1_10%': 4.14 + 0.4 * k,
        }

        if F_stat > bounds_critical['I1_5%']:
            conclusion = "Cointegration exists (F > upper bound)"
            cointegrated = True
        elif F_stat < bounds_critical['I0_5%']:
            conclusion = "No cointegration (F < lower bound)"
            cointegrated = False
        else:
            conclusion = "Inconclusive (F within bounds)"
            cointegrated = None

        return {
            'test_name': 'ARDL Bounds',
            'F_statistic': round(F_stat, 4),
            'critical_bounds': bounds_critical,
            'cointegrated': cointegrated,
            'conclusion': conclusion
        }

    def analyze(self,
                data: pd.DataFrame,
                method: str = 'auto') -> CointegrationReport:
        """
        Comprehensive cointegration analysis.

        Args:
            data: DataFrame with series as columns
            method: 'engle_granger', 'johansen', or 'auto'

        Returns:
            CointegrationReport
        """
        series_names = list(data.columns)
        n_series = len(series_names)
        n_obs = len(data)

        warnings = []

        # Check integration orders
        from .stationarity import StationarityAnalyzer
        stat_analyzer = StationarityAnalyzer()
        integration_orders = {}

        for col in data.columns:
            report = stat_analyzer.analyze(data[col])
            integration_orders[col] = report.integration_order.value

        # Check if analysis is appropriate
        i1_count = sum(1 for v in integration_orders.values() if v == 1)
        i0_count = sum(1 for v in integration_orders.values() if v == 0)

        if i1_count < 2:
            warnings.append("Less than 2 I(1) series. Cointegration may not be meaningful.")

        # Run appropriate test
        eg_result = None
        joh_result = None

        if method == 'auto':
            method = 'engle_granger' if n_series == 2 else 'johansen'

        if method == 'engle_granger' and n_series == 2:
            eg_result = self.engle_granger_test(
                data.iloc[:, 0].values,
                data.iloc[:, 1].values,
                series_names[0],
                series_names[1]
            )
            has_coint = eg_result.cointegrated
            rank = 1 if has_coint else 0

        elif method == 'johansen' or n_series > 2:
            joh_result = self.johansen_test(data)
            has_coint = joh_result.rank > 0
            rank = joh_result.rank
        else:
            has_coint = False
            rank = 0

        # Build equilibrium relationships
        equilibrium = []
        if eg_result and eg_result.cointegrated:
            equilibrium.append({
                'type': 'pairwise',
                'series': [series_names[0], series_names[1]],
                'vector': eg_result.cointegrating_vector
            })
        elif joh_result and joh_result.rank > 0:
            for i in range(joh_result.rank):
                if joh_result.cointegrating_vectors is not None:
                    vec = joh_result.cointegrating_vectors[:, i].tolist()
                    equilibrium.append({
                        'type': 'multivariate',
                        'series': series_names,
                        'vector': [round(v, 4) for v in vec]
                    })

        return CointegrationReport(
            series_names=series_names,
            n_series=n_series,
            n_observations=n_obs,
            integration_orders=integration_orders,
            eg_result=eg_result,
            johansen_result=joh_result,
            has_cointegration=has_coint,
            cointegration_rank=rank,
            equilibrium_relationships=equilibrium,
            warnings=warnings
        )

    # =========================================================================
    # Fallback implementations
    # =========================================================================

    def _fallback_eg(self, y1, y2, name1, name2) -> EngleGrangerResult:
        """Fallback Engle-Granger without statsmodels."""
        # OLS regression
        X = np.column_stack([np.ones(len(y2)), y2])
        beta = np.linalg.lstsq(X, y1, rcond=None)[0]
        residuals = y1 - X @ beta

        # ADF on residuals (simplified)
        diff_resid = np.diff(residuals)
        lag_resid = residuals[:-1]

        # Regression
        if len(lag_resid) > 1:
            rho = np.sum(diff_resid * lag_resid) / np.sum(lag_resid**2)
            se = np.std(diff_resid) / np.sqrt(np.sum(lag_resid**2))
            t_stat = rho / se if se > 0 else 0
        else:
            t_stat = 0

        # Critical values for cointegration (stricter than regular ADF)
        critical_values = {'1%': -3.96, '5%': -3.37, '10%': -3.07}
        cointegrated = t_stat < critical_values['5%']

        return EngleGrangerResult(
            series_1=name1,
            series_2=name2,
            adf_statistic=round(t_stat, 4),
            p_value=0.05 if cointegrated else 0.10,
            critical_values=critical_values,
            cointegrated=cointegrated,
            cointegrating_vector=[round(b, 4) for b in beta],
            conclusion="Cointegrated" if cointegrated else "Not cointegrated"
        )

    def _fallback_johansen(self, data, n_series) -> JohansenResult:
        """Fallback Johansen without statsmodels."""
        return JohansenResult(
            n_series=n_series,
            max_eigenvalue_stats=[],
            trace_stats=[],
            critical_values_max={},
            critical_values_trace={},
            rank=0,
            cointegrating_vectors=None,
            conclusion="Johansen test requires statsmodels"
        )


def print_cointegration_report(report: CointegrationReport):
    """Pretty-print cointegration report."""
    print("=" * 60)
    print("Cointegration Report")
    print("=" * 60)
    print(f"Series: {', '.join(report.series_names)}")
    print(f"Observations: {report.n_observations}")
    print()

    print("Integration Orders:")
    for name, order in report.integration_orders.items():
        print(f"  {name}: I({order})")
    print()

    if report.eg_result:
        eg = report.eg_result
        print(f"Engle-Granger Test:")
        print(f"  Statistic: {eg.adf_statistic}")
        print(f"  P-value: {eg.p_value}")
        print(f"  {eg.conclusion}")
        print()

    if report.johansen_result:
        joh = report.johansen_result
        print(f"Johansen Test:")
        print(f"  Trace statistics: {joh.trace_stats}")
        print(f"  Rank: {joh.rank}")
        print(f"  {joh.conclusion}")
        print()

    print(f"Has Cointegration: {report.has_cointegration}")
    print(f"Rank: {report.cointegration_rank}")

    if report.equilibrium_relationships:
        print("\nEquilibrium Relationships:")
        for eq in report.equilibrium_relationships:
            print(f"  {eq}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Cointegration Testing Module - Test")
    print("=" * 60)

    np.random.seed(42)
    n = 500

    # Create cointegrated series
    # Common trend
    trend = np.cumsum(np.random.randn(n))

    # Two series sharing the trend
    y1 = trend + np.random.randn(n) * 0.5
    y2 = 0.8 * trend + np.random.randn(n) * 0.5

    # Non-cointegrated series
    y3 = np.cumsum(np.random.randn(n))

    panel = pd.DataFrame({
        'y1': y1,
        'y2': y2,
        'y3': y3
    })

    print(f"\nTest panel: {panel.shape}")
    print(f"statsmodels available: {STATSMODELS_AVAILABLE}")

    analyzer = CointegrationAnalyzer()

    # Test pairwise
    print("\nPairwise test (y1 vs y2 - should be cointegrated):")
    eg_result = analyzer.engle_granger_test(y1, y2, 'y1', 'y2')
    print(f"  {eg_result.conclusion}")

    print("\nPairwise test (y1 vs y3 - should NOT be cointegrated):")
    eg_result2 = analyzer.engle_granger_test(y1, y3, 'y1', 'y3')
    print(f"  {eg_result2.conclusion}")

    # Full analysis
    print("\nFull analysis:")
    report = analyzer.analyze(panel[['y1', 'y2']])
    print_cointegration_report(report)

    print("\nTest completed!")
