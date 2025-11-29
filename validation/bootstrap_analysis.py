"""
Bootstrap Analysis - Confidence intervals and stability analysis
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BootstrapAnalyzer:
    """
    Bootstrap analysis for lens result stability and confidence intervals.

    Bootstrap resampling helps assess:
    - Stability of indicator rankings
    - Confidence intervals for scores
    - Robustness to sample variation
    """

    def __init__(self, n_bootstrap: int = 1000, seed: int = 42):
        """
        Initialize bootstrap analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples
            seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        np.random.seed(seed)

    def bootstrap_ranking_stability(
        self,
        df: pd.DataFrame,
        lens_class: type,
        top_n: int = 10,
        **lens_kwargs
    ) -> Dict[str, Any]:
        """
        Assess stability of indicator rankings via bootstrap.

        Args:
            df: Input DataFrame
            lens_class: Lens class to analyze
            top_n: Number of top indicators to track
            **lens_kwargs: Arguments passed to lens

        Returns:
            Dictionary with stability analysis
        """
        n_rows = len(df)
        rank_counts = defaultdict(list)  # indicator -> list of ranks
        top_n_counts = defaultdict(int)  # indicator -> times in top N

        for i in range(self.n_bootstrap):
            # Bootstrap sample (sample with replacement)
            indices = np.random.choice(n_rows, size=n_rows, replace=True)
            bootstrap_df = df.iloc[indices].reset_index(drop=True)

            try:
                lens = lens_class()
                ranking = lens.rank_indicators(bootstrap_df, **lens_kwargs)

                # Track ranks
                for _, row in ranking.iterrows():
                    rank_counts[row['indicator']].append(row['rank'])
                    if row['rank'] <= top_n:
                        top_n_counts[row['indicator']] += 1
            except Exception:
                continue

        # Compute statistics
        stability_scores = {}
        rank_stats = {}

        for indicator, ranks in rank_counts.items():
            if ranks:
                mean_rank = np.mean(ranks)
                std_rank = np.std(ranks)
                stability = 1 / (1 + std_rank)  # Higher = more stable

                stability_scores[indicator] = stability
                rank_stats[indicator] = {
                    'mean_rank': mean_rank,
                    'std_rank': std_rank,
                    'min_rank': min(ranks),
                    'max_rank': max(ranks),
                    'top_n_frequency': top_n_counts[indicator] / self.n_bootstrap
                }

        # Most stable top indicators
        stable_leaders = sorted(
            [(ind, stats) for ind, stats in rank_stats.items()
             if stats['mean_rank'] <= top_n],
            key=lambda x: x[1]['std_rank']
        )[:top_n]

        return {
            'lens': lens_class.__name__,
            'n_bootstrap': self.n_bootstrap,
            'stability_scores': stability_scores,
            'rank_statistics': rank_stats,
            'stable_leaders': [x[0] for x in stable_leaders],
            'mean_stability': np.mean(list(stability_scores.values()))
        }

    def bootstrap_confidence_intervals(
        self,
        df: pd.DataFrame,
        lens_class: type,
        confidence: float = 0.95,
        **lens_kwargs
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Compute confidence intervals for indicator scores.

        Args:
            df: Input DataFrame
            lens_class: Lens class to analyze
            confidence: Confidence level (default 0.95 for 95% CI)
            **lens_kwargs: Arguments passed to lens

        Returns:
            Dictionary mapping indicator -> (lower, upper) CI
        """
        n_rows = len(df)
        score_samples = defaultdict(list)

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n_rows, size=n_rows, replace=True)
            bootstrap_df = df.iloc[indices].reset_index(drop=True)

            try:
                lens = lens_class()
                ranking = lens.rank_indicators(bootstrap_df, **lens_kwargs)

                for _, row in ranking.iterrows():
                    score_samples[row['indicator']].append(row['score'])
            except Exception:
                continue

        # Compute percentile-based confidence intervals
        alpha = 1 - confidence
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        confidence_intervals = {}
        for indicator, scores in score_samples.items():
            if len(scores) >= 10:  # Need enough samples
                lower = np.percentile(scores, lower_pct)
                upper = np.percentile(scores, upper_pct)
                confidence_intervals[indicator] = {
                    'lower': lower,
                    'upper': upper,
                    'mean': np.mean(scores),
                    'median': np.median(scores)
                }

        return confidence_intervals

    def bootstrap_consensus_stability(
        self,
        df: pd.DataFrame,
        lens_classes: List[type],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Assess stability of consensus rankings across bootstrap samples.

        Args:
            df: Input DataFrame
            lens_classes: List of lens classes
            top_n: Number of top indicators

        Returns:
            Dictionary with consensus stability analysis
        """
        n_rows = len(df)
        consensus_counts = defaultdict(int)

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n_rows, size=n_rows, replace=True)
            bootstrap_df = df.iloc[indices].reset_index(drop=True)

            # Get top N from each lens
            all_top = []
            for lens_class in lens_classes:
                try:
                    lens = lens_class()
                    ranking = lens.rank_indicators(bootstrap_df)
                    top = ranking.head(top_n)['indicator'].tolist()
                    all_top.extend(top)
                except Exception:
                    continue

            # Count consensus (appeared in multiple lens tops)
            from collections import Counter
            counts = Counter(all_top)
            for indicator, count in counts.items():
                if count >= len(lens_classes) // 2:  # Majority consensus
                    consensus_counts[indicator] += 1

        # Normalize by number of bootstrap samples
        consensus_frequency = {
            ind: count / self.n_bootstrap
            for ind, count in consensus_counts.items()
        }

        # Stable consensus indicators (appear in >50% of bootstraps)
        stable_consensus = [
            ind for ind, freq in consensus_frequency.items()
            if freq > 0.5
        ]

        return {
            'n_bootstrap': self.n_bootstrap,
            'n_lenses': len(lens_classes),
            'consensus_frequency': consensus_frequency,
            'stable_consensus_indicators': stable_consensus,
            'n_stable': len(stable_consensus)
        }


def run_bootstrap_analysis(
    df: pd.DataFrame,
    lens_classes: List[type],
    n_bootstrap: int = 500
) -> Dict[str, Any]:
    """
    Run comprehensive bootstrap analysis.

    Args:
        df: Input DataFrame
        lens_classes: List of lens classes
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with all bootstrap results
    """
    analyzer = BootstrapAnalyzer(n_bootstrap=n_bootstrap)

    results = {
        'individual_lens_stability': {},
        'confidence_intervals': {},
    }

    for lens_class in lens_classes:
        logger.info(f"Bootstrap analysis for {lens_class.__name__}...")

        try:
            stability = analyzer.bootstrap_ranking_stability(df, lens_class)
            results['individual_lens_stability'][lens_class.__name__] = stability

            ci = analyzer.bootstrap_confidence_intervals(df, lens_class)
            results['confidence_intervals'][lens_class.__name__] = ci
        except Exception as e:
            logger.error(f"Error analyzing {lens_class.__name__}: {e}")

    # Consensus stability
    results['consensus_stability'] = analyzer.bootstrap_consensus_stability(
        df, lens_classes
    )

    return results
