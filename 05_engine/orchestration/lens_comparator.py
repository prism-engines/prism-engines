"""
Lens Comparator - Run and compare multiple lenses
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class LensComparator:
    """
    Run multiple lenses and compare their results.

    Provides:
    - Parallel lens execution
    - Ranking comparison across lenses
    - Agreement analysis
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize comparator.

        Args:
            checkpoint_dir: Directory for saving results
        """
        self.checkpoint_dir = checkpoint_dir or Path("05_engine/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}
        self.rankings: Dict[str, pd.DataFrame] = {}

    def run_lenses(
        self,
        df: pd.DataFrame,
        lenses: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run multiple lenses on the data.

        Args:
            df: Input DataFrame
            lenses: List of lens names to run (default: all basic lenses)
            **kwargs: Parameters passed to each lens

        Returns:
            Dictionary with results per lens
        """
        from .. import LENS_REGISTRY

        if lenses is None:
            lenses = ['magnitude', 'pca', 'influence', 'clustering', 'decomposition']

        self.results = {}
        self.rankings = {}
        timing = {}

        for lens_name in lenses:
            if lens_name not in LENS_REGISTRY:
                logger.warning(f"Unknown lens: {lens_name}")
                continue

            logger.info(f"Running {lens_name} lens...")
            start = time.time()

            try:
                lens = LENS_REGISTRY[lens_name]()
                self.results[lens_name] = lens.analyze(df, **kwargs)
                self.rankings[lens_name] = lens.rank_indicators(df, **kwargs)
                timing[lens_name] = time.time() - start
                logger.info(f"  Completed in {timing[lens_name]:.2f}s")
            except Exception as e:
                logger.error(f"  Error: {e}")
                self.results[lens_name] = {"error": str(e)}

        return {
            "results": self.results,
            "rankings": {k: v.to_dict() for k, v in self.rankings.items()},
            "timing": timing,
            "n_lenses": len([r for r in self.results.values() if "error" not in r]),
        }

    def compare_rankings(self, top_n: int = 10) -> pd.DataFrame:
        """
        Compare indicator rankings across lenses.

        Args:
            top_n: Number of top indicators to compare

        Returns:
            DataFrame with indicator rankings per lens
        """
        if not self.rankings:
            return pd.DataFrame()

        # Get all indicators
        all_indicators = set()
        for ranking_df in self.rankings.values():
            all_indicators.update(ranking_df['indicator'].tolist())

        # Build comparison table
        comparison = []
        for indicator in all_indicators:
            row = {"indicator": indicator}
            for lens_name, ranking_df in self.rankings.items():
                match = ranking_df[ranking_df['indicator'] == indicator]
                if len(match) > 0:
                    row[f"{lens_name}_rank"] = int(match['rank'].iloc[0])
                    row[f"{lens_name}_score"] = float(match['score'].iloc[0])
                else:
                    row[f"{lens_name}_rank"] = len(ranking_df) + 1
                    row[f"{lens_name}_score"] = 0

            # Average rank
            rank_cols = [c for c in row if c.endswith('_rank')]
            row['avg_rank'] = np.mean([row[c] for c in rank_cols])

            comparison.append(row)

        df = pd.DataFrame(comparison)
        df = df.sort_values('avg_rank')

        return df.head(top_n * 2)  # Return more for analysis

    def get_agreement_matrix(self) -> pd.DataFrame:
        """
        Compute agreement matrix between lenses.

        Shows how often lenses agree on top indicators.
        """
        if len(self.rankings) < 2:
            return pd.DataFrame()

        lens_names = list(self.rankings.keys())
        n_lenses = len(lens_names)

        agreement = np.zeros((n_lenses, n_lenses))

        for i, lens1 in enumerate(lens_names):
            for j, lens2 in enumerate(lens_names):
                if i == j:
                    agreement[i, j] = 1.0
                else:
                    # Compare top 10 indicators
                    top1 = set(self.rankings[lens1].head(10)['indicator'])
                    top2 = set(self.rankings[lens2].head(10)['indicator'])
                    overlap = len(top1 & top2) / 10
                    agreement[i, j] = overlap

        return pd.DataFrame(agreement, index=lens_names, columns=lens_names)

    def get_consistent_leaders(self, min_agreement: float = 0.5) -> List[str]:
        """
        Find indicators consistently ranked high across lenses.

        Args:
            min_agreement: Minimum fraction of lenses that must agree

        Returns:
            List of consistently top-ranked indicators
        """
        if not self.rankings:
            return []

        comparison = self.compare_rankings(top_n=20)
        rank_cols = [c for c in comparison.columns if c.endswith('_rank')]

        n_lenses = len(rank_cols)
        min_count = int(n_lenses * min_agreement)

        consistent = []
        for _, row in comparison.iterrows():
            # Count how many lenses rank this in top 10
            top_10_count = sum(1 for c in rank_cols if row[c] <= 10)
            if top_10_count >= min_count:
                consistent.append(row['indicator'])

        return consistent

    def save_comparison(self, name: Optional[str] = None) -> Path:
        """
        Save comparison results to checkpoint.

        Args:
            name: Optional name for the checkpoint

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or "lens_comparison"

        save_dir = self.checkpoint_dir / f"{name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save rankings
        for lens_name, ranking_df in self.rankings.items():
            ranking_df.to_csv(save_dir / f"{lens_name}_ranking.csv", index=False)

        # Save comparison
        comparison = self.compare_rankings()
        comparison.to_csv(save_dir / "comparison.csv", index=False)

        # Save agreement matrix
        agreement = self.get_agreement_matrix()
        agreement.to_csv(save_dir / "agreement_matrix.csv")

        logger.info(f"Comparison saved to {save_dir}")
        return save_dir
