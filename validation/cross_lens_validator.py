"""
Cross-Lens Validator - Validate consistency across multiple lenses
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class CrossLensValidator:
    """
    Validate consistency and agreement across multiple lenses.

    Tests:
    - Ranking correlation between lenses
    - Stable vs unstable indicators
    - Lens clustering (which lenses agree?)
    """

    def __init__(self):
        self.rankings: Dict[str, pd.DataFrame] = {}
        self.agreement_matrix: Optional[pd.DataFrame] = None

    def collect_rankings(
        self,
        df: pd.DataFrame,
        lens_classes: List[type],
        **lens_kwargs
    ) -> None:
        """
        Collect rankings from all lenses.

        Args:
            df: Input DataFrame
            lens_classes: List of lens classes
            **lens_kwargs: Arguments passed to lenses
        """
        self.rankings = {}

        for lens_class in lens_classes:
            try:
                lens = lens_class()
                ranking = lens.rank_indicators(df, **lens_kwargs)
                self.rankings[lens_class.__name__] = ranking
            except Exception as e:
                logger.error(f"Error collecting ranking from {lens_class.__name__}: {e}")

    def compute_rank_correlations(self) -> pd.DataFrame:
        """
        Compute pairwise Spearman rank correlations between lenses.

        Returns:
            Correlation matrix DataFrame
        """
        if not self.rankings:
            raise ValueError("No rankings collected. Call collect_rankings first.")

        lens_names = list(self.rankings.keys())
        n_lenses = len(lens_names)

        corr_matrix = np.zeros((n_lenses, n_lenses))

        for i, lens1 in enumerate(lens_names):
            for j, lens2 in enumerate(lens_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = self._rank_correlation(
                        self.rankings[lens1],
                        self.rankings[lens2]
                    )
                    corr_matrix[i, j] = corr

        self.agreement_matrix = pd.DataFrame(
            corr_matrix,
            index=lens_names,
            columns=lens_names
        )

        return self.agreement_matrix

    def _rank_correlation(
        self,
        ranking1: pd.DataFrame,
        ranking2: pd.DataFrame
    ) -> float:
        """Compute Spearman correlation between two rankings."""
        # Merge on indicator
        merged = ranking1[['indicator', 'rank']].merge(
            ranking2[['indicator', 'rank']],
            on='indicator',
            suffixes=('_1', '_2')
        )

        if len(merged) < 3:
            return 0.0

        corr, _ = stats.spearmanr(merged['rank_1'], merged['rank_2'])
        return corr if not np.isnan(corr) else 0.0

    def identify_stable_indicators(
        self,
        top_n: int = 10,
        min_agreement: float = 0.5
    ) -> Dict[str, Any]:
        """
        Find indicators that are consistently ranked high across lenses.

        Args:
            top_n: Consider top N from each lens
            min_agreement: Minimum fraction of lenses that must agree

        Returns:
            Dictionary with stable indicator analysis
        """
        if not self.rankings:
            raise ValueError("No rankings collected. Call collect_rankings first.")

        # Count appearances in top N
        top_n_counts = {}
        all_indicators = set()

        for lens_name, ranking in self.rankings.items():
            top_indicators = ranking.head(top_n)['indicator'].tolist()
            for ind in top_indicators:
                all_indicators.add(ind)
                top_n_counts[ind] = top_n_counts.get(ind, 0) + 1

        n_lenses = len(self.rankings)
        threshold = int(n_lenses * min_agreement)

        # Categorize indicators
        stable = []  # In top N for most lenses
        variable = []  # In top N for some lenses
        rare = []  # In top N for few lenses

        for ind in all_indicators:
            count = top_n_counts.get(ind, 0)
            if count >= threshold:
                stable.append((ind, count))
            elif count >= 2:
                variable.append((ind, count))
            else:
                rare.append((ind, count))

        # Sort by count
        stable.sort(key=lambda x: -x[1])
        variable.sort(key=lambda x: -x[1])

        # Compute rank statistics for stable indicators
        stable_stats = {}
        for ind, _ in stable:
            ranks = []
            for ranking in self.rankings.values():
                match = ranking[ranking['indicator'] == ind]
                if len(match) > 0:
                    ranks.append(match['rank'].iloc[0])

            stable_stats[ind] = {
                'mean_rank': np.mean(ranks),
                'std_rank': np.std(ranks),
                'min_rank': min(ranks),
                'max_rank': max(ranks),
                'n_lenses': len(ranks)
            }

        return {
            'stable_indicators': [x[0] for x in stable],
            'stable_counts': dict(stable),
            'stable_statistics': stable_stats,
            'variable_indicators': [x[0] for x in variable],
            'rare_indicators': [x[0] for x in rare],
            'n_lenses': n_lenses,
            'threshold': threshold
        }

    def cluster_lenses(self) -> Dict[str, Any]:
        """
        Cluster lenses by their ranking similarity.

        Returns:
            Dictionary with lens clustering results
        """
        if self.agreement_matrix is None:
            self.compute_rank_correlations()

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Convert correlation to distance
        dist_matrix = 1 - self.agreement_matrix.values
        np.fill_diagonal(dist_matrix, 0)

        # Ensure symmetry
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        # Cluster
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method='average')

        # Get 2-3 clusters
        labels_2 = fcluster(Z, 2, criterion='maxclust')
        labels_3 = fcluster(Z, 3, criterion='maxclust')

        lens_names = self.agreement_matrix.index.tolist()

        clusters_2 = {}
        for i, label in enumerate(labels_2):
            if label not in clusters_2:
                clusters_2[label] = []
            clusters_2[label].append(lens_names[i])

        clusters_3 = {}
        for i, label in enumerate(labels_3):
            if label not in clusters_3:
                clusters_3[label] = []
            clusters_3[label].append(lens_names[i])

        return {
            'clusters_2': clusters_2,
            'clusters_3': clusters_3,
            'linkage_matrix': Z.tolist(),
            'lens_names': lens_names
        }

    def lens_diversity_score(self) -> float:
        """
        Compute diversity score for the lens ensemble.

        Higher diversity = lenses capture different aspects.

        Returns:
            Diversity score (0-1)
        """
        if self.agreement_matrix is None:
            self.compute_rank_correlations()

        # Mean off-diagonal correlation
        mask = ~np.eye(len(self.agreement_matrix), dtype=bool)
        mean_corr = self.agreement_matrix.values[mask].mean()

        # Diversity = 1 - mean_correlation
        # High correlation = low diversity
        diversity = 1 - abs(mean_corr)

        return diversity

    def validate_all(
        self,
        df: pd.DataFrame,
        lens_classes: List[type],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run all cross-lens validation tests.

        Args:
            df: Input DataFrame
            lens_classes: List of lens classes

        Returns:
            Complete validation results
        """
        # Collect rankings
        self.collect_rankings(df, lens_classes, **kwargs)

        # Run all analyses
        results = {
            'rank_correlations': self.compute_rank_correlations().to_dict(),
            'stable_indicators': self.identify_stable_indicators(),
            'lens_clusters': self.cluster_lenses(),
            'diversity_score': self.lens_diversity_score(),
            'n_lenses': len(lens_classes),
            'n_indicators': len(df.columns) - 1
        }

        return results
