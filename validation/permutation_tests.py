"""
Permutation Tests - Statistical significance testing for lens results
"""

from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)


class PermutationTester:
    """
    Test statistical significance of lens results using permutation tests.

    Permutation tests determine if observed patterns could arise by chance
    by comparing against a null distribution generated from shuffled data.
    """

    def __init__(self, n_permutations: int = 1000, seed: int = 42):
        """
        Initialize permutation tester.

        Args:
            n_permutations: Number of permutations to run
            seed: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.seed = seed
        np.random.seed(seed)

    def test_ranking_significance(
        self,
        df: pd.DataFrame,
        lens_class: type,
        top_n: int = 10,
        **lens_kwargs
    ) -> Dict[str, Any]:
        """
        Test if top-ranked indicators are significantly different from random.

        Args:
            df: Input DataFrame
            lens_class: Lens class to test
            top_n: Number of top indicators to consider
            **lens_kwargs: Arguments passed to lens

        Returns:
            Dictionary with significance results
        """
        # Get actual ranking
        lens = lens_class()
        actual_ranking = lens.rank_indicators(df, **lens_kwargs)
        actual_top = set(actual_ranking.head(top_n)['indicator'].tolist())
        actual_scores = actual_ranking.head(top_n)['score'].tolist()

        # Generate null distribution
        null_overlaps = []
        null_score_means = []

        value_cols = [c for c in df.columns if c != 'date']

        for i in range(self.n_permutations):
            # Shuffle each column independently (breaks relationships)
            shuffled_df = df.copy()
            for col in value_cols:
                shuffled_df[col] = np.random.permutation(shuffled_df[col].values)

            # Get ranking on shuffled data
            try:
                null_ranking = lens.rank_indicators(shuffled_df, **lens_kwargs)
                null_top = set(null_ranking.head(top_n)['indicator'].tolist())
                null_scores = null_ranking.head(top_n)['score'].tolist()

                # Measure overlap with actual top
                overlap = len(actual_top & null_top) / top_n
                null_overlaps.append(overlap)
                null_score_means.append(np.mean(null_scores))
            except Exception:
                continue

        # Compute p-value
        # How often does random shuffling produce similar or better overlap?
        actual_mean_score = np.mean(actual_scores)
        p_value = np.mean([s >= actual_mean_score for s in null_score_means])

        return {
            "lens": lens_class.__name__,
            "n_permutations": len(null_overlaps),
            "actual_top_indicators": list(actual_top),
            "actual_mean_score": actual_mean_score,
            "null_mean_score": np.mean(null_score_means),
            "null_std_score": np.std(null_score_means),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "z_score": (actual_mean_score - np.mean(null_score_means)) / (np.std(null_score_means) + 1e-10)
        }

    def test_correlation_significance(
        self,
        df: pd.DataFrame,
        indicator1: str,
        indicator2: str
    ) -> Dict[str, Any]:
        """
        Test if correlation between two indicators is significant.

        Args:
            df: Input DataFrame
            indicator1: First indicator name
            indicator2: Second indicator name

        Returns:
            Dictionary with significance results
        """
        # Actual correlation
        actual_corr = df[indicator1].corr(df[indicator2])

        # Null distribution
        null_corrs = []
        for _ in range(self.n_permutations):
            shuffled = np.random.permutation(df[indicator2].values)
            null_corr = df[indicator1].corr(pd.Series(shuffled))
            null_corrs.append(null_corr)

        # Two-tailed p-value
        p_value = np.mean([abs(nc) >= abs(actual_corr) for nc in null_corrs])

        return {
            "indicator1": indicator1,
            "indicator2": indicator2,
            "actual_correlation": actual_corr,
            "null_mean": np.mean(null_corrs),
            "null_std": np.std(null_corrs),
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    def test_regime_significance(
        self,
        df: pd.DataFrame,
        regime_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Test if detected regimes are significantly different from random partitions.

        Args:
            df: Input DataFrame
            regime_labels: Regime assignments for each row

        Returns:
            Dictionary with significance results
        """
        value_cols = [c for c in df.columns if c != 'date']

        # Compute actual between-regime variance
        df_with_regime = df.copy()
        df_with_regime['regime'] = regime_labels

        actual_variance = self._between_regime_variance(df_with_regime, value_cols)

        # Null distribution (random regime assignments)
        null_variances = []
        n_regimes = len(set(regime_labels))

        for _ in range(self.n_permutations):
            # Random partition
            random_labels = np.random.randint(0, n_regimes, len(df))
            df_null = df.copy()
            df_null['regime'] = random_labels

            null_var = self._between_regime_variance(df_null, value_cols)
            null_variances.append(null_var)

        p_value = np.mean([nv >= actual_variance for nv in null_variances])

        return {
            "actual_between_variance": actual_variance,
            "null_mean_variance": np.mean(null_variances),
            "null_std_variance": np.std(null_variances),
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    def _between_regime_variance(
        self,
        df: pd.DataFrame,
        value_cols: List[str]
    ) -> float:
        """Compute between-regime variance."""
        regime_means = df.groupby('regime')[value_cols].mean()
        overall_mean = df[value_cols].mean()

        # Weighted sum of squared differences from overall mean
        variance = 0
        for regime in regime_means.index:
            regime_size = (df['regime'] == regime).sum()
            diff = regime_means.loc[regime] - overall_mean
            variance += regime_size * (diff ** 2).sum()

        return variance / len(df)


def run_permutation_tests(
    df: pd.DataFrame,
    lens_classes: List[type],
    n_permutations: int = 500
) -> pd.DataFrame:
    """
    Run permutation tests for multiple lenses.

    Args:
        df: Input DataFrame
        lens_classes: List of lens classes to test
        n_permutations: Number of permutations

    Returns:
        DataFrame with test results
    """
    tester = PermutationTester(n_permutations=n_permutations)
    results = []

    for lens_class in lens_classes:
        logger.info(f"Testing {lens_class.__name__}...")
        try:
            result = tester.test_ranking_significance(df, lens_class)
            results.append(result)
        except Exception as e:
            logger.error(f"Error testing {lens_class.__name__}: {e}")

    return pd.DataFrame(results)
