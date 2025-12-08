"""
Temporal Validation Module
==========================

Out-of-sample validation protocols for time series.
Critical for preventing look-ahead bias and overfitting.

Methods:
- Expanding window: Train on [0:t], test on [t:t+1]
- Rolling window: Train on [t-w:t], test on [t:t+1]
- Blocked cross-validation: K-fold with temporal blocks
- Purged cross-validation: Gap between train/test (prevent leakage)

Key principle: NEVER train on future data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Generator, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ValidationFold:
    """Definition of a single validation fold."""
    fold_id: int
    train_start: int  # Index
    train_end: int
    test_start: int
    test_end: int
    gap: int = 0  # Gap between train and test


@dataclass
class FoldResult:
    """Result from evaluating one fold."""
    fold_id: int
    train_size: int
    test_size: int
    metric_value: float
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Complete validation result."""
    method: str
    n_folds: int
    fold_results: List[FoldResult]
    mean_metric: float
    std_metric: float
    ci_lower: float
    ci_upper: float
    all_predictions: Optional[np.ndarray] = None
    all_actuals: Optional[np.ndarray] = None


class TemporalValidator:
    """
    Temporal validation for time series models.
    """

    def __init__(self,
                 min_train_size: int = 252,
                 test_size: int = 21,
                 gap: int = 0):
        """
        Initialize validator.

        Args:
            min_train_size: Minimum training observations
            test_size: Test set size
            gap: Gap between train and test (to prevent leakage)
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap

    def expanding_window_split(self,
                                n: int,
                                step: int = None) -> Generator[ValidationFold, None, None]:
        """
        Expanding window cross-validation.

        Training window grows over time, starting from min_train_size.

        Args:
            n: Total number of observations
            step: Step size between folds (defaults to test_size)

        Yields:
            ValidationFold definitions
        """
        if step is None:
            step = self.test_size

        fold_id = 0
        train_end = self.min_train_size

        while train_end + self.gap + self.test_size <= n:
            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            yield ValidationFold(
                fold_id=fold_id,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                gap=self.gap
            )

            fold_id += 1
            train_end += step

    def rolling_window_split(self,
                              n: int,
                              window_size: int = None,
                              step: int = None) -> Generator[ValidationFold, None, None]:
        """
        Rolling window cross-validation.

        Fixed-size training window that rolls forward.

        Args:
            n: Total number of observations
            window_size: Training window size (defaults to min_train_size)
            step: Step size between folds

        Yields:
            ValidationFold definitions
        """
        if window_size is None:
            window_size = self.min_train_size

        if step is None:
            step = self.test_size

        fold_id = 0
        train_start = 0

        while train_start + window_size + self.gap + self.test_size <= n:
            train_end = train_start + window_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            yield ValidationFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                gap=self.gap
            )

            fold_id += 1
            train_start += step

    def blocked_cv_split(self,
                          n: int,
                          n_blocks: int = 5) -> Generator[ValidationFold, None, None]:
        """
        Blocked time series cross-validation.

        Divides data into temporal blocks, uses each as test set once.

        Args:
            n: Total number of observations
            n_blocks: Number of blocks

        Yields:
            ValidationFold definitions
        """
        block_size = n // n_blocks

        for fold_id in range(n_blocks):
            test_start = fold_id * block_size
            test_end = min((fold_id + 1) * block_size, n)

            # Train on all data except test block
            # For temporal validity, only use data before test block
            train_end = test_start - self.gap

            if train_end < self.min_train_size:
                continue  # Skip early folds with insufficient training data

            yield ValidationFold(
                fold_id=fold_id,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                gap=self.gap
            )

    def purged_cv_split(self,
                         n: int,
                         n_folds: int = 5,
                         purge_window: int = 21) -> Generator[ValidationFold, None, None]:
        """
        Purged cross-validation.

        Includes a gap/purge window to prevent information leakage
        from train to test set (important for overlapping features).

        Args:
            n: Total observations
            n_folds: Number of folds
            purge_window: Size of purge window on each side

        Yields:
            ValidationFold definitions
        """
        fold_size = n // n_folds

        for fold_id in range(n_folds):
            test_start = fold_id * fold_size
            test_end = min((fold_id + 1) * fold_size, n)

            # Purge: create gap before and after test set
            train_end = max(0, test_start - purge_window)

            if train_end < self.min_train_size:
                continue

            yield ValidationFold(
                fold_id=fold_id,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                gap=purge_window
            )

    def validate(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 train_func: Callable,
                 predict_func: Callable,
                 metric_func: Callable,
                 method: str = 'expanding',
                 **kwargs) -> ValidationResult:
        """
        Run temporal validation.

        Args:
            data: Full dataset
            train_func: Function(train_data) -> model
            predict_func: Function(model, test_data) -> predictions
            metric_func: Function(predictions, actuals) -> metric
            method: 'expanding', 'rolling', 'blocked', or 'purged'
            **kwargs: Additional arguments for split method

        Returns:
            ValidationResult
        """
        if isinstance(data, pd.DataFrame):
            n = len(data)
        else:
            n = len(data)

        # Get splits
        split_methods = {
            'expanding': self.expanding_window_split,
            'rolling': self.rolling_window_split,
            'blocked': self.blocked_cv_split,
            'purged': self.purged_cv_split
        }

        if method not in split_methods:
            raise ValueError(f"Unknown method: {method}")

        splits = list(split_methods[method](n, **kwargs))

        # Run validation
        fold_results = []
        all_predictions = []
        all_actuals = []

        for fold in splits:
            # Get train/test data
            if isinstance(data, pd.DataFrame):
                train_data = data.iloc[fold.train_start:fold.train_end]
                test_data = data.iloc[fold.test_start:fold.test_end]
            else:
                train_data = data[fold.train_start:fold.train_end]
                test_data = data[fold.test_start:fold.test_end]

            try:
                # Train
                model = train_func(train_data)

                # Predict
                predictions = predict_func(model, test_data)

                # Get actuals (assume last column or single column)
                if isinstance(test_data, pd.DataFrame):
                    actuals = test_data.iloc[:, -1].values
                else:
                    actuals = test_data

                # Compute metric
                metric_value = metric_func(predictions, actuals)

                fold_results.append(FoldResult(
                    fold_id=fold.fold_id,
                    train_size=fold.train_end - fold.train_start,
                    test_size=fold.test_end - fold.test_start,
                    metric_value=metric_value,
                    predictions=predictions,
                    actuals=actuals
                ))

                all_predictions.extend(predictions)
                all_actuals.extend(actuals)

            except Exception as e:
                # Log and skip failed folds
                print(f"Fold {fold.fold_id} failed: {e}")
                continue

        if len(fold_results) == 0:
            raise ValueError("All folds failed")

        # Aggregate results
        metrics = [r.metric_value for r in fold_results]
        mean_metric = np.mean(metrics)
        std_metric = np.std(metrics)

        # 95% CI
        ci_lower = mean_metric - 1.96 * std_metric / np.sqrt(len(metrics))
        ci_upper = mean_metric + 1.96 * std_metric / np.sqrt(len(metrics))

        return ValidationResult(
            method=method,
            n_folds=len(fold_results),
            fold_results=fold_results,
            mean_metric=mean_metric,
            std_metric=std_metric,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            all_predictions=np.array(all_predictions) if all_predictions else None,
            all_actuals=np.array(all_actuals) if all_actuals else None
        )


class WalkForwardValidator:
    """
    Walk-forward optimization and validation.

    Combines parameter optimization with out-of-sample testing.
    """

    def __init__(self,
                 train_period: int = 252,
                 test_period: int = 63,
                 anchored: bool = True):
        """
        Initialize walk-forward validator.

        Args:
            train_period: In-sample training period
            test_period: Out-of-sample test period
            anchored: If True, training starts from beginning (expanding)
                      If False, rolling window
        """
        self.train_period = train_period
        self.test_period = test_period
        self.anchored = anchored

    def walk_forward(self,
                      data: Union[np.ndarray, pd.DataFrame],
                      optimize_func: Callable,
                      evaluate_func: Callable) -> Dict[str, Any]:
        """
        Run walk-forward analysis.

        Args:
            data: Full dataset
            optimize_func: Function(train_data) -> optimal_params
            evaluate_func: Function(test_data, params) -> metric

        Returns:
            Dictionary with walk-forward results
        """
        n = len(data)

        results = []
        current_pos = self.train_period

        while current_pos + self.test_period <= n:
            # Define periods
            if self.anchored:
                train_start = 0
            else:
                train_start = current_pos - self.train_period

            train_end = current_pos
            test_start = current_pos
            test_end = current_pos + self.test_period

            # Get data
            if isinstance(data, pd.DataFrame):
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[test_start:test_end]
            else:
                train_data = data[train_start:train_end]
                test_data = data[test_start:test_end]

            try:
                # Optimize on training data
                optimal_params = optimize_func(train_data)

                # Evaluate on test data
                oos_metric = evaluate_func(test_data, optimal_params)

                results.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'params': optimal_params,
                    'oos_metric': oos_metric
                })
            except Exception as e:
                print(f"Walk-forward step failed: {e}")

            current_pos += self.test_period

        # Aggregate
        if results:
            oos_metrics = [r['oos_metric'] for r in results]
            return {
                'method': 'walk_forward',
                'anchored': self.anchored,
                'n_periods': len(results),
                'mean_oos_metric': np.mean(oos_metrics),
                'std_oos_metric': np.std(oos_metrics),
                'period_results': results
            }
        else:
            return {
                'method': 'walk_forward',
                'n_periods': 0,
                'error': 'No successful periods'
            }


def temporal_train_test_split(data: Union[np.ndarray, pd.DataFrame],
                               train_ratio: float = 0.7,
                               gap: int = 0) -> Tuple[Any, Any]:
    """
    Simple temporal train/test split.

    Args:
        data: Dataset
        train_ratio: Fraction for training
        gap: Gap between train and test

    Returns:
        Tuple of (train_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    test_start = train_end + gap

    if isinstance(data, pd.DataFrame):
        return data.iloc[:train_end], data.iloc[test_start:]
    else:
        return data[:train_end], data[test_start:]


def print_validation_result(result: ValidationResult):
    """Pretty-print validation result."""
    print("=" * 60)
    print(f"TEMPORAL VALIDATION: {result.method.upper()}")
    print("=" * 60)
    print(f"Number of folds: {result.n_folds}")
    print()

    print("FOLD RESULTS:")
    print("-" * 40)
    for fold in result.fold_results:
        print(f"  Fold {fold.fold_id}: metric={fold.metric_value:.4f} "
              f"(train={fold.train_size}, test={fold.test_size})")

    print()
    print("AGGREGATE:")
    print("-" * 40)
    print(f"  Mean metric: {result.mean_metric:.4f}")
    print(f"  Std metric:  {result.std_metric:.4f}")
    print(f"  95% CI:      [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Validation Module - Demo")
    print("=" * 60)

    np.random.seed(42)

    # Create synthetic data
    n = 1000
    X = np.random.randn(n, 3)  # Features
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5  # Target

    data = np.column_stack([X, y])

    # Simple model functions
    def train_func(train_data):
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        # Simple linear regression
        beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        return beta

    def predict_func(model, test_data):
        X_test = test_data[:, :-1]
        return X_test @ model

    def metric_func(predictions, actuals):
        # MSE
        return np.mean((predictions - actuals) ** 2)

    validator = TemporalValidator(min_train_size=200, test_size=50, gap=5)

    print("\nExpanding Window Validation:")
    print("-" * 40)
    result = validator.validate(
        data, train_func, predict_func, metric_func,
        method='expanding'
    )
    print_validation_result(result)

    print("\nRolling Window Validation:")
    print("-" * 40)
    result = validator.validate(
        data, train_func, predict_func, metric_func,
        method='rolling'
    )
    print_validation_result(result)

    print("\nPurged CV Validation:")
    print("-" * 40)
    result = validator.validate(
        data, train_func, predict_func, metric_func,
        method='purged', n_folds=5, purge_window=10
    )
    print_validation_result(result)

    print("\nTest completed!")
