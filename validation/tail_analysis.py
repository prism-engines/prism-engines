"""
Tail Risk Analysis Module
=========================

Analyze tail behavior conditional on regime/signal states.
Critical for assessing practical risk management value.

Metrics:
- Conditional VaR (Value at Risk)
- Conditional ES (Expected Shortfall / CVaR)
- Tail dependence (co-crash probability)
- Drawdown distribution by regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class TailMetrics:
    """Tail risk metrics for a single condition."""
    condition_name: str
    n_observations: int
    var_95: float  # 5% VaR (worst 5%)
    var_99: float  # 1% VaR (worst 1%)
    expected_shortfall_95: float  # CVaR at 95%
    expected_shortfall_99: float  # CVaR at 99%
    max_loss: float
    skewness: float
    kurtosis: float


@dataclass
class ConditionalTailResult:
    """Comparison of tail metrics across conditions."""
    metric_name: str
    conditions: Dict[str, TailMetrics]
    var_ratio: float  # Crisis VaR / Normal VaR
    es_ratio: float   # Crisis ES / Normal ES
    significant_difference: bool
    p_value: float


@dataclass
class TailDependenceResult:
    """Tail dependence analysis result."""
    series_1: str
    series_2: str
    lower_tail_dependence: float  # Co-crash probability
    upper_tail_dependence: float  # Co-rally probability
    threshold: float
    n_joint_tail_events: int


@dataclass
class TailAnalysisReport:
    """Complete tail analysis report."""
    returns_series: str
    n_total: int
    unconditional_metrics: TailMetrics
    conditional_results: List[ConditionalTailResult]
    tail_dependence: Optional[TailDependenceResult] = None
    value_at_stake: float = 0.0  # Total risk reduction from regime awareness


class TailAnalyzer:
    """
    Analyze tail risk properties.
    """

    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize analyzer.

        Args:
            confidence_levels: VaR confidence levels
        """
        self.confidence_levels = confidence_levels

    def compute_var(self,
                    returns: np.ndarray,
                    confidence: float = 0.95) -> float:
        """
        Compute Value at Risk (VaR).

        VaR_95 is the 5th percentile of returns (worst 5%).

        Args:
            returns: Array of returns
            confidence: Confidence level

        Returns:
            VaR (as positive number representing loss)
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        alpha = 1 - confidence
        var = -np.percentile(returns, alpha * 100)

        return var

    def compute_expected_shortfall(self,
                                    returns: np.ndarray,
                                    confidence: float = 0.95) -> float:
        """
        Compute Expected Shortfall (ES) / Conditional VaR.

        ES is the average loss given that loss exceeds VaR.

        Args:
            returns: Array of returns
            confidence: Confidence level

        Returns:
            ES (as positive number representing loss)
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        alpha = 1 - confidence
        var = np.percentile(returns, alpha * 100)

        # Average of returns below VaR
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return -var

        es = -np.mean(tail_returns)
        return es

    def compute_tail_metrics(self,
                             returns: np.ndarray,
                             condition_name: str = "all") -> TailMetrics:
        """
        Compute comprehensive tail metrics.

        Args:
            returns: Array of returns
            condition_name: Name of condition

        Returns:
            TailMetrics
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 10:
            return TailMetrics(
                condition_name=condition_name,
                n_observations=len(returns),
                var_95=np.nan,
                var_99=np.nan,
                expected_shortfall_95=np.nan,
                expected_shortfall_99=np.nan,
                max_loss=np.nan,
                skewness=np.nan,
                kurtosis=np.nan
            )

        var_95 = self.compute_var(returns, 0.95)
        var_99 = self.compute_var(returns, 0.99)
        es_95 = self.compute_expected_shortfall(returns, 0.95)
        es_99 = self.compute_expected_shortfall(returns, 0.99)

        mean = np.mean(returns)
        std = np.std(returns)

        if std > 0:
            skewness = np.mean(((returns - mean) / std) ** 3)
            kurtosis = np.mean(((returns - mean) / std) ** 4)
        else:
            skewness = 0
            kurtosis = 0

        return TailMetrics(
            condition_name=condition_name,
            n_observations=len(returns),
            var_95=round(var_95, 6),
            var_99=round(var_99, 6),
            expected_shortfall_95=round(es_95, 6),
            expected_shortfall_99=round(es_99, 6),
            max_loss=round(-np.min(returns), 6),
            skewness=round(skewness, 4),
            kurtosis=round(kurtosis, 4)
        )

    def conditional_tail_analysis(self,
                                   returns: pd.Series,
                                   condition: pd.Series,
                                   condition_name: str = "regime") -> ConditionalTailResult:
        """
        Compare tail metrics across conditions.

        Args:
            returns: Return series
            condition: Binary condition (0/1)
            condition_name: Name for reporting

        Returns:
            ConditionalTailResult
        """
        # Align series
        df = pd.DataFrame({'returns': returns, 'condition': condition}).dropna()

        normal_returns = df[df['condition'] == 0]['returns'].values
        crisis_returns = df[df['condition'] == 1]['returns'].values

        normal_metrics = self.compute_tail_metrics(normal_returns, f"{condition_name}_0")
        crisis_metrics = self.compute_tail_metrics(crisis_returns, f"{condition_name}_1")

        # Compute ratios
        var_ratio = crisis_metrics.var_95 / normal_metrics.var_95 if normal_metrics.var_95 > 0 else np.inf
        es_ratio = crisis_metrics.expected_shortfall_95 / normal_metrics.expected_shortfall_95 if normal_metrics.expected_shortfall_95 > 0 else np.inf

        # Test significance via bootstrap
        p_value = self._bootstrap_tail_test(normal_returns, crisis_returns)

        return ConditionalTailResult(
            metric_name=condition_name,
            conditions={
                'normal': normal_metrics,
                'crisis': crisis_metrics
            },
            var_ratio=round(var_ratio, 3),
            es_ratio=round(es_ratio, 3),
            significant_difference=p_value < 0.05,
            p_value=round(p_value, 4)
        )

    def _bootstrap_tail_test(self,
                             returns_1: np.ndarray,
                             returns_2: np.ndarray,
                             n_bootstrap: int = 1000) -> float:
        """
        Bootstrap test for difference in tail risk.
        """
        rng = np.random.RandomState(42)

        # Observed difference in ES
        es1 = self.compute_expected_shortfall(returns_1)
        es2 = self.compute_expected_shortfall(returns_2)
        observed_diff = es2 - es1

        # Pool data
        pooled = np.concatenate([returns_1, returns_2])
        n1 = len(returns_1)

        # Bootstrap under null
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            rng.shuffle(pooled)
            boot1 = pooled[:n1]
            boot2 = pooled[n1:]

            boot_es1 = self.compute_expected_shortfall(boot1)
            boot_es2 = self.compute_expected_shortfall(boot2)
            bootstrap_diffs.append(boot_es2 - boot_es1)

        # Two-sided p-value
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        return max(p_value, 1 / (n_bootstrap + 1))

    def compute_tail_dependence(self,
                                 returns_1: np.ndarray,
                                 returns_2: np.ndarray,
                                 threshold: float = 0.05,
                                 name_1: str = "series_1",
                                 name_2: str = "series_2") -> TailDependenceResult:
        """
        Compute tail dependence between two series.

        Lower tail dependence = P(Y < VaR | X < VaR)

        Args:
            returns_1, returns_2: Return series
            threshold: Quantile for tail definition
            name_1, name_2: Series names

        Returns:
            TailDependenceResult
        """
        r1 = np.asarray(returns_1)
        r2 = np.asarray(returns_2)

        # Remove NaN
        mask = ~(np.isnan(r1) | np.isnan(r2))
        r1, r2 = r1[mask], r2[mask]
        n = len(r1)

        # Lower tail thresholds
        q1_lower = np.percentile(r1, threshold * 100)
        q2_lower = np.percentile(r2, threshold * 100)

        # Upper tail thresholds
        q1_upper = np.percentile(r1, (1 - threshold) * 100)
        q2_upper = np.percentile(r2, (1 - threshold) * 100)

        # Lower tail dependence
        in_lower_1 = r1 <= q1_lower
        in_lower_2 = r2 <= q2_lower
        joint_lower = in_lower_1 & in_lower_2

        if np.sum(in_lower_1) > 0:
            lower_td = np.sum(joint_lower) / np.sum(in_lower_1)
        else:
            lower_td = 0

        # Upper tail dependence
        in_upper_1 = r1 >= q1_upper
        in_upper_2 = r2 >= q2_upper
        joint_upper = in_upper_1 & in_upper_2

        if np.sum(in_upper_1) > 0:
            upper_td = np.sum(joint_upper) / np.sum(in_upper_1)
        else:
            upper_td = 0

        return TailDependenceResult(
            series_1=name_1,
            series_2=name_2,
            lower_tail_dependence=round(lower_td, 4),
            upper_tail_dependence=round(upper_td, 4),
            threshold=threshold,
            n_joint_tail_events=int(np.sum(joint_lower))
        )

    def drawdown_analysis(self,
                          prices: pd.Series,
                          condition: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze drawdown distribution, optionally by condition.

        Args:
            prices: Price series
            condition: Optional binary condition

        Returns:
            Dictionary with drawdown statistics
        """
        # Compute drawdowns
        rolling_max = prices.expanding().max()
        drawdowns = (rolling_max - prices) / rolling_max

        results = {
            'unconditional': {
                'max_drawdown': round(drawdowns.max(), 4),
                'mean_drawdown': round(drawdowns.mean(), 4),
                'median_drawdown': round(drawdowns.median(), 4),
                'drawdown_90': round(drawdowns.quantile(0.90), 4),
                'drawdown_95': round(drawdowns.quantile(0.95), 4),
            }
        }

        if condition is not None:
            df = pd.DataFrame({'dd': drawdowns, 'cond': condition}).dropna()

            for cond_val in [0, 1]:
                cond_dd = df[df['cond'] == cond_val]['dd']
                cond_name = 'normal' if cond_val == 0 else 'crisis'

                results[cond_name] = {
                    'max_drawdown': round(cond_dd.max(), 4),
                    'mean_drawdown': round(cond_dd.mean(), 4),
                    'median_drawdown': round(cond_dd.median(), 4),
                    'drawdown_90': round(cond_dd.quantile(0.90), 4),
                    'drawdown_95': round(cond_dd.quantile(0.95), 4),
                    'n_observations': len(cond_dd)
                }

        return results

    def analyze(self,
                returns: pd.Series,
                condition: Optional[pd.Series] = None,
                prices: Optional[pd.Series] = None,
                returns_2: Optional[pd.Series] = None) -> TailAnalysisReport:
        """
        Comprehensive tail analysis.

        Args:
            returns: Primary return series
            condition: Binary condition for conditional analysis
            prices: Price series for drawdown analysis
            returns_2: Second return series for tail dependence

        Returns:
            TailAnalysisReport
        """
        returns_arr = returns.dropna().values

        # Unconditional metrics
        unconditional = self.compute_tail_metrics(returns_arr, "unconditional")

        # Conditional analysis
        conditional_results = []
        if condition is not None:
            cond_result = self.conditional_tail_analysis(returns, condition, "regime")
            conditional_results.append(cond_result)

        # Tail dependence
        tail_dep = None
        if returns_2 is not None:
            tail_dep = self.compute_tail_dependence(
                returns_arr,
                returns_2.dropna().values,
                name_1="primary",
                name_2="secondary"
            )

        # Value at stake (risk reduction from regime awareness)
        value_at_stake = 0.0
        if conditional_results:
            cond = conditional_results[0]
            normal_es = cond.conditions['normal'].expected_shortfall_95
            crisis_es = cond.conditions['crisis'].expected_shortfall_95
            value_at_stake = crisis_es - normal_es

        return TailAnalysisReport(
            returns_series=returns.name or "returns",
            n_total=len(returns_arr),
            unconditional_metrics=unconditional,
            conditional_results=conditional_results,
            tail_dependence=tail_dep,
            value_at_stake=value_at_stake
        )


def print_tail_analysis_report(report: TailAnalysisReport):
    """Pretty-print tail analysis report."""
    print("=" * 60)
    print("TAIL RISK ANALYSIS")
    print("=" * 60)
    print(f"Series: {report.returns_series}")
    print(f"Observations: {report.n_total}")
    print()

    print("UNCONDITIONAL TAIL METRICS:")
    print("-" * 40)
    m = report.unconditional_metrics
    print(f"  VaR 95%:    {m.var_95:.4f} ({m.var_95*100:.2f}%)")
    print(f"  VaR 99%:    {m.var_99:.4f} ({m.var_99*100:.2f}%)")
    print(f"  ES 95%:     {m.expected_shortfall_95:.4f} ({m.expected_shortfall_95*100:.2f}%)")
    print(f"  ES 99%:     {m.expected_shortfall_99:.4f} ({m.expected_shortfall_99*100:.2f}%)")
    print(f"  Max Loss:   {m.max_loss:.4f}")
    print(f"  Skewness:   {m.skewness:.3f}")
    print(f"  Kurtosis:   {m.kurtosis:.3f}")

    for cond in report.conditional_results:
        print()
        print(f"CONDITIONAL: {cond.metric_name}")
        print("-" * 40)

        for name, metrics in cond.conditions.items():
            print(f"\n  {name.upper()} (n={metrics.n_observations}):")
            print(f"    VaR 95%: {metrics.var_95:.4f}")
            print(f"    ES 95%:  {metrics.expected_shortfall_95:.4f}")

        print(f"\n  VaR Ratio (crisis/normal): {cond.var_ratio:.2f}x")
        print(f"  ES Ratio (crisis/normal):  {cond.es_ratio:.2f}x")
        print(f"  Significant: {cond.significant_difference} (p={cond.p_value:.4f})")

    if report.tail_dependence:
        td = report.tail_dependence
        print()
        print("TAIL DEPENDENCE:")
        print("-" * 40)
        print(f"  Series: {td.series_1} vs {td.series_2}")
        print(f"  Lower tail (co-crash): {td.lower_tail_dependence:.3f}")
        print(f"  Upper tail (co-rally): {td.upper_tail_dependence:.3f}")
        print(f"  Joint tail events: {td.n_joint_tail_events}")

    print()
    print(f"VALUE AT STAKE: {report.value_at_stake:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Tail Analysis Module - Demo")
    print("=" * 60)

    np.random.seed(42)

    # Create synthetic returns with regime structure
    n = 2000
    dates = pd.date_range('2010-01-01', periods=n, freq='B')

    # Regime (20% crisis)
    regime = np.zeros(n)
    in_crisis = False
    for i in range(n):
        if not in_crisis and np.random.random() < 0.01:
            in_crisis = True
        elif in_crisis and np.random.random() < 0.05:
            in_crisis = False
        regime[i] = 1 if in_crisis else 0

    # Returns (higher vol and negative skew in crisis)
    returns = np.zeros(n)
    for i in range(n):
        if regime[i] == 0:
            returns[i] = np.random.randn() * 0.01 + 0.0003
        else:
            returns[i] = np.random.randn() * 0.025 - 0.002

    returns_series = pd.Series(returns, index=dates, name='returns')
    regime_series = pd.Series(regime, index=dates, name='regime')
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name='price')

    # Second series for tail dependence
    returns_2 = returns * 0.8 + np.random.randn(n) * 0.005
    returns_2_series = pd.Series(returns_2, index=dates, name='returns_2')

    print("\nRunning tail analysis...")
    analyzer = TailAnalyzer()

    report = analyzer.analyze(
        returns=returns_series,
        condition=regime_series,
        prices=prices,
        returns_2=returns_2_series
    )

    print_tail_analysis_report(report)

    print("\nTest completed!")
