"""
Regime Classification Accuracy Module
=====================================

Evaluate accuracy of regime detection against known benchmarks.

Benchmarks:
- NBER recession dates
- Market drawdown thresholds
- VIX regime thresholds
- Trend-following signals

Metrics:
- Confusion matrix
- Precision, Recall, F1
- ROC/AUC
- Lead/lag analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification."""
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total > 0 else 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else 0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else 0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    @property
    def specificity(self) -> float:
        denom = self.true_negative + self.false_positive
        return self.true_negative / denom if denom > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tp': self.true_positive,
            'tn': self.true_negative,
            'fp': self.false_positive,
            'fn': self.false_negative,
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'specificity': round(self.specificity, 4)
        }


@dataclass
class RegimeAccuracyResult:
    """Complete regime accuracy analysis result."""
    benchmark_name: str
    n_observations: int
    confusion_matrix: ConfusionMatrix
    roc_auc: Optional[float] = None
    brier_score: Optional[float] = None
    lead_lag_days: Optional[int] = None  # Positive = predicted leads actual
    regime_durations: Dict[str, List[int]] = field(default_factory=dict)


# NBER Recession dates (official turning points)
NBER_RECESSIONS = [
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01'),
]


class BenchmarkGenerator:
    """
    Generate benchmark regime labels from various sources.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with data.

        Args:
            data: DataFrame with DatetimeIndex
        """
        self.data = data
        self.index = data.index

    def nber_recession(self) -> pd.Series:
        """
        Generate NBER recession indicator.

        Returns:
            Series with 1 = recession, 0 = expansion
        """
        labels = pd.Series(0, index=self.index)

        for start, end in NBER_RECESSIONS:
            mask = (self.index >= start) & (self.index <= end)
            labels[mask] = 1

        return labels

    def drawdown_regime(self,
                        price_col: str = 'price',
                        threshold: float = 0.10) -> pd.Series:
        """
        Generate regime based on drawdown from peak.

        Args:
            price_col: Column name for price data
            threshold: Drawdown threshold for "crisis" regime

        Returns:
            Series with 1 = crisis (drawdown > threshold), 0 = normal
        """
        if price_col not in self.data.columns:
            raise ValueError(f"Column {price_col} not found")

        prices = self.data[price_col]

        # Rolling peak
        rolling_peak = prices.expanding().max()

        # Current drawdown
        drawdown = (rolling_peak - prices) / rolling_peak

        return (drawdown >= threshold).astype(int)

    def vix_regime(self,
                   vix_col: str = 'vix',
                   threshold: float = 25) -> pd.Series:
        """
        Generate regime based on VIX level.

        Args:
            vix_col: Column name for VIX data
            threshold: VIX level for "high volatility" regime

        Returns:
            Series with 1 = high vol, 0 = normal
        """
        if vix_col not in self.data.columns:
            raise ValueError(f"Column {vix_col} not found")

        vix = self.data[vix_col]
        return (vix >= threshold).astype(int)

    def trend_following(self,
                        price_col: str = 'price',
                        fast_window: int = 50,
                        slow_window: int = 200) -> pd.Series:
        """
        Generate regime based on moving average crossover.

        Args:
            price_col: Column name for price data
            fast_window: Fast MA window
            slow_window: Slow MA window

        Returns:
            Series with 1 = downtrend, 0 = uptrend
        """
        if price_col not in self.data.columns:
            raise ValueError(f"Column {price_col} not found")

        prices = self.data[price_col]

        fast_ma = prices.rolling(fast_window).mean()
        slow_ma = prices.rolling(slow_window).mean()

        # Downtrend = fast below slow
        return (fast_ma < slow_ma).astype(int)

    def volatility_regime(self,
                          returns_col: str = 'returns',
                          window: int = 21,
                          threshold_percentile: float = 75) -> pd.Series:
        """
        Generate regime based on realized volatility.

        Args:
            returns_col: Column name for returns
            window: Rolling window for vol calculation
            threshold_percentile: Percentile threshold for "high vol"

        Returns:
            Series with 1 = high vol, 0 = normal
        """
        if returns_col not in self.data.columns:
            raise ValueError(f"Column {returns_col} not found")

        returns = self.data[returns_col]
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)

        threshold = rolling_vol.quantile(threshold_percentile / 100)
        return (rolling_vol >= threshold).astype(int)


class RegimeAccuracyAnalyzer:
    """
    Analyze accuracy of predicted regimes against benchmarks.
    """

    def __init__(self):
        pass

    def compute_confusion_matrix(self,
                                  predicted: pd.Series,
                                  actual: pd.Series) -> ConfusionMatrix:
        """
        Compute confusion matrix.

        Args:
            predicted: Predicted regime labels (0/1)
            actual: Actual regime labels (0/1)

        Returns:
            ConfusionMatrix
        """
        # Align series
        df = pd.DataFrame({'predicted': predicted, 'actual': actual}).dropna()

        pred = df['predicted'].values.astype(int)
        act = df['actual'].values.astype(int)

        tp = np.sum((pred == 1) & (act == 1))
        tn = np.sum((pred == 0) & (act == 0))
        fp = np.sum((pred == 1) & (act == 0))
        fn = np.sum((pred == 0) & (act == 1))

        return ConfusionMatrix(
            true_positive=int(tp),
            true_negative=int(tn),
            false_positive=int(fp),
            false_negative=int(fn)
        )

    def compute_roc_auc(self,
                        predicted_prob: pd.Series,
                        actual: pd.Series) -> float:
        """
        Compute ROC AUC score.

        Args:
            predicted_prob: Predicted probabilities (0-1)
            actual: Actual labels (0/1)

        Returns:
            AUC score
        """
        try:
            from sklearn.metrics import roc_auc_score
            df = pd.DataFrame({'prob': predicted_prob, 'actual': actual}).dropna()
            return roc_auc_score(df['actual'], df['prob'])
        except ImportError:
            # Manual AUC calculation
            df = pd.DataFrame({'prob': predicted_prob, 'actual': actual}).dropna()
            df = df.sort_values('prob', ascending=False)

            n_pos = df['actual'].sum()
            n_neg = len(df) - n_pos

            if n_pos == 0 or n_neg == 0:
                return 0.5

            tpr_sum = 0
            cum_pos = 0

            for _, row in df.iterrows():
                if row['actual'] == 1:
                    cum_pos += 1
                else:
                    tpr_sum += cum_pos

            auc = tpr_sum / (n_pos * n_neg)
            return auc

    def compute_brier_score(self,
                            predicted_prob: pd.Series,
                            actual: pd.Series) -> float:
        """
        Compute Brier score (lower is better).

        Args:
            predicted_prob: Predicted probabilities
            actual: Actual labels

        Returns:
            Brier score
        """
        df = pd.DataFrame({'prob': predicted_prob, 'actual': actual}).dropna()
        return np.mean((df['prob'] - df['actual']) ** 2)

    def compute_lead_lag(self,
                         predicted: pd.Series,
                         actual: pd.Series,
                         max_lag: int = 30) -> int:
        """
        Compute optimal lead/lag between predicted and actual.

        Positive = predicted leads actual (early warning)
        Negative = predicted lags actual (late detection)

        Args:
            predicted: Predicted labels
            actual: Actual labels
            max_lag: Maximum lag to test

        Returns:
            Optimal lag in periods
        """
        best_corr = -1
        best_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                shifted_pred = predicted.shift(lag)
            else:
                shifted_pred = predicted.shift(lag)

            df = pd.DataFrame({'pred': shifted_pred, 'act': actual}).dropna()

            if len(df) > 10:
                corr = np.corrcoef(df['pred'], df['act'])[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

        return best_lag

    def analyze(self,
                predicted: pd.Series,
                actual: pd.Series,
                predicted_prob: Optional[pd.Series] = None,
                benchmark_name: str = "benchmark") -> RegimeAccuracyResult:
        """
        Complete accuracy analysis.

        Args:
            predicted: Predicted regime labels (0/1)
            actual: Actual regime labels (0/1)
            predicted_prob: Predicted probabilities (optional)
            benchmark_name: Name of benchmark

        Returns:
            RegimeAccuracyResult
        """
        # Confusion matrix
        cm = self.compute_confusion_matrix(predicted, actual)

        # ROC AUC
        if predicted_prob is not None:
            roc_auc = self.compute_roc_auc(predicted_prob, actual)
            brier = self.compute_brier_score(predicted_prob, actual)
        else:
            roc_auc = None
            brier = None

        # Lead/lag
        lead_lag = self.compute_lead_lag(predicted, actual)

        # Regime durations
        df = pd.DataFrame({'pred': predicted, 'act': actual}).dropna()

        pred_durations = self._compute_regime_durations(df['pred'])
        act_durations = self._compute_regime_durations(df['act'])

        return RegimeAccuracyResult(
            benchmark_name=benchmark_name,
            n_observations=len(df),
            confusion_matrix=cm,
            roc_auc=roc_auc,
            brier_score=brier,
            lead_lag_days=lead_lag,
            regime_durations={
                'predicted': pred_durations,
                'actual': act_durations
            }
        )

    def _compute_regime_durations(self, labels: pd.Series) -> Dict[int, List[int]]:
        """Compute distribution of regime durations."""
        durations = {0: [], 1: []}

        current_regime = labels.iloc[0]
        current_duration = 1

        for label in labels.iloc[1:]:
            if label == current_regime:
                current_duration += 1
            else:
                durations[int(current_regime)].append(current_duration)
                current_regime = label
                current_duration = 1

        durations[int(current_regime)].append(current_duration)

        return durations

    def compare_to_random(self,
                          actual: pd.Series,
                          n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Compare results to random baseline.

        Args:
            actual: Actual regime labels
            n_simulations: Number of random simulations

        Returns:
            Dictionary with comparison statistics
        """
        actual = actual.dropna()
        n = len(actual)
        base_rate = actual.mean()

        random_accuracies = []
        random_f1s = []

        rng = np.random.RandomState(42)

        for _ in range(n_simulations):
            random_pred = pd.Series(
                rng.binomial(1, base_rate, n),
                index=actual.index
            )
            cm = self.compute_confusion_matrix(random_pred, actual)
            random_accuracies.append(cm.accuracy)
            random_f1s.append(cm.f1_score)

        return {
            'base_rate': base_rate,
            'random_accuracy_mean': np.mean(random_accuracies),
            'random_accuracy_std': np.std(random_accuracies),
            'random_f1_mean': np.mean(random_f1s),
            'random_f1_std': np.std(random_f1s),
            'accuracy_95_threshold': np.percentile(random_accuracies, 95),
            'f1_95_threshold': np.percentile(random_f1s, 95)
        }


def print_regime_accuracy_report(result: RegimeAccuracyResult,
                                  random_baseline: Optional[Dict] = None):
    """Pretty-print regime accuracy report."""
    print("=" * 60)
    print(f"REGIME ACCURACY: {result.benchmark_name}")
    print("=" * 60)
    print(f"Observations: {result.n_observations}")
    print()

    cm = result.confusion_matrix
    print("CONFUSION MATRIX:")
    print("-" * 40)
    print(f"                  Actual")
    print(f"                  0       1")
    print(f"Predicted  0    {cm.true_negative:5d}   {cm.false_negative:5d}")
    print(f"           1    {cm.false_positive:5d}   {cm.true_positive:5d}")
    print()

    print("METRICS:")
    print("-" * 40)
    print(f"  Accuracy:    {cm.accuracy:.3f}")
    print(f"  Precision:   {cm.precision:.3f}")
    print(f"  Recall:      {cm.recall:.3f}")
    print(f"  F1 Score:    {cm.f1_score:.3f}")
    print(f"  Specificity: {cm.specificity:.3f}")

    if result.roc_auc is not None:
        print(f"  ROC AUC:     {result.roc_auc:.3f}")

    if result.lead_lag_days is not None:
        print(f"  Lead/Lag:    {result.lead_lag_days} days")

    if random_baseline:
        print()
        print("vs RANDOM BASELINE:")
        print("-" * 40)
        print(f"  Random accuracy: {random_baseline['random_accuracy_mean']:.3f} ± {random_baseline['random_accuracy_std']:.3f}")
        print(f"  Random F1:       {random_baseline['random_f1_mean']:.3f} ± {random_baseline['random_f1_std']:.3f}")

        if cm.accuracy > random_baseline['accuracy_95_threshold']:
            print(f"  Accuracy SIGNIFICANTLY better than random (p < 0.05)")
        else:
            print(f"  Accuracy NOT significantly better than random")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Regime Accuracy Module - Demo")
    print("=" * 60)

    np.random.seed(42)

    # Create synthetic data
    dates = pd.date_range('2000-01-01', '2020-12-31', freq='B')
    n = len(dates)

    # Actual regimes (with some structure)
    actual = np.zeros(n)
    regime = 0
    for i in range(n):
        if np.random.random() < 0.02:  # 2% chance of regime change
            regime = 1 - regime
        actual[i] = regime

    # Predicted (correlated with actual but with noise)
    predicted = actual.copy()
    noise_idx = np.random.choice(n, size=int(n * 0.2), replace=False)
    predicted[noise_idx] = 1 - predicted[noise_idx]

    # Shift predicted by 5 days (early detection)
    predicted = np.roll(predicted, -5)

    actual_series = pd.Series(actual, index=dates)
    predicted_series = pd.Series(predicted, index=dates)

    analyzer = RegimeAccuracyAnalyzer()

    print("\nAnalyzing regime accuracy...")
    result = analyzer.analyze(predicted_series, actual_series, benchmark_name="Synthetic Test")

    print("\nComparing to random baseline...")
    random_baseline = analyzer.compare_to_random(actual_series, n_simulations=500)

    print_regime_accuracy_report(result, random_baseline)

    print("\nTest completed!")
