"""
Consensus Engine - Find agreement across multiple lenses
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """
    Build consensus from multiple lens results.

    Methods:
    - Rank aggregation (Borda count, Kemeny-Young)
    - Score fusion (weighted average, voting)
    - Confidence estimation
    """

    def __init__(self, rankings: Dict[str, pd.DataFrame]):
        """
        Initialize with lens rankings.

        Args:
            rankings: Dictionary mapping lens_name -> ranking DataFrame
        """
        self.rankings = rankings
        self.lens_weights: Dict[str, float] = {}

    def set_lens_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom weights for each lens.

        Args:
            weights: Dictionary mapping lens_name -> weight
        """
        self.lens_weights = weights

    def borda_count(self) -> pd.DataFrame:
        """
        Aggregate rankings using Borda count.

        Each indicator gets points = n - rank, where n is total indicators.

        Returns:
            DataFrame with consensus ranking
        """
        all_indicators = set()
        for ranking_df in self.rankings.values():
            all_indicators.update(ranking_df['indicator'].tolist())

        n = len(all_indicators)
        scores = {indicator: 0 for indicator in all_indicators}

        for lens_name, ranking_df in self.rankings.items():
            weight = self.lens_weights.get(lens_name, 1.0)

            for _, row in ranking_df.iterrows():
                indicator = row['indicator']
                rank = row['rank']
                # Borda score: n - rank
                scores[indicator] += weight * (n - rank)

        # Normalize
        max_score = max(scores.values()) if scores else 1
        scores = {k: v / max_score for k, v in scores.items()}

        result = pd.DataFrame([
            {"indicator": k, "consensus_score": v}
            for k, v in scores.items()
        ])
        result = result.sort_values("consensus_score", ascending=False)
        result["consensus_rank"] = range(1, len(result) + 1)

        return result.reset_index(drop=True)

    def score_fusion(self, method: str = "mean") -> pd.DataFrame:
        """
        Fuse scores across lenses.

        Args:
            method: 'mean', 'median', 'max', or 'geometric'

        Returns:
            DataFrame with fused scores
        """
        all_indicators = set()
        for ranking_df in self.rankings.values():
            all_indicators.update(ranking_df['indicator'].tolist())

        indicator_scores = {ind: [] for ind in all_indicators}

        for lens_name, ranking_df in self.rankings.items():
            weight = self.lens_weights.get(lens_name, 1.0)

            for _, row in ranking_df.iterrows():
                indicator = row['indicator']
                score = row['score'] * weight
                indicator_scores[indicator].append(score)

        # Fuse scores
        fused = {}
        for indicator, scores in indicator_scores.items():
            if not scores:
                fused[indicator] = 0
            elif method == "mean":
                fused[indicator] = np.mean(scores)
            elif method == "median":
                fused[indicator] = np.median(scores)
            elif method == "max":
                fused[indicator] = np.max(scores)
            elif method == "geometric":
                fused[indicator] = stats.gmean([s + 0.01 for s in scores])  # Add small value to avoid zero
            else:
                fused[indicator] = np.mean(scores)

        result = pd.DataFrame([
            {"indicator": k, "fused_score": v}
            for k, v in fused.items()
        ])
        result = result.sort_values("fused_score", ascending=False)
        result["rank"] = range(1, len(result) + 1)

        return result.reset_index(drop=True)

    def voting(self, top_n: int = 10) -> pd.DataFrame:
        """
        Simple voting: count how many lenses rank each indicator in top N.

        Args:
            top_n: Consider top N from each lens

        Returns:
            DataFrame with vote counts
        """
        votes = {}

        for lens_name, ranking_df in self.rankings.items():
            top_indicators = ranking_df.head(top_n)['indicator'].tolist()

            for indicator in top_indicators:
                if indicator not in votes:
                    votes[indicator] = 0
                votes[indicator] += 1

        result = pd.DataFrame([
            {"indicator": k, "votes": v, "vote_pct": v / len(self.rankings) * 100}
            for k, v in votes.items()
        ])
        result = result.sort_values("votes", ascending=False)
        result["rank"] = range(1, len(result) + 1)

        return result.reset_index(drop=True)

    def confidence_scores(self) -> pd.DataFrame:
        """
        Compute confidence in each indicator's ranking.

        Confidence is based on:
        - Agreement across lenses (lower rank variance = higher confidence)
        - Absolute score magnitude

        Returns:
            DataFrame with confidence scores
        """
        all_indicators = set()
        for ranking_df in self.rankings.values():
            all_indicators.update(ranking_df['indicator'].tolist())

        confidence = {}

        for indicator in all_indicators:
            ranks = []
            scores = []

            for ranking_df in self.rankings.values():
                match = ranking_df[ranking_df['indicator'] == indicator]
                if len(match) > 0:
                    ranks.append(match['rank'].iloc[0])
                    scores.append(match['score'].iloc[0])

            if ranks:
                # Lower rank variance = higher agreement = higher confidence
                rank_var = np.var(ranks) if len(ranks) > 1 else 0
                rank_agreement = 1 / (1 + rank_var)

                # Higher mean score = more important
                mean_score = np.mean(scores)

                # Combine
                confidence[indicator] = rank_agreement * (1 + mean_score)
            else:
                confidence[indicator] = 0

        # Normalize
        max_conf = max(confidence.values()) if confidence else 1
        confidence = {k: v / max_conf for k, v in confidence.items()}

        result = pd.DataFrame([
            {"indicator": k, "confidence": v}
            for k, v in confidence.items()
        ])
        result = result.sort_values("confidence", ascending=False)

        return result.reset_index(drop=True)

    def full_consensus(self) -> Dict[str, Any]:
        """
        Run all consensus methods and return combined results.

        Returns:
            Dictionary with all consensus analyses
        """
        borda = self.borda_count()
        fused = self.score_fusion()
        votes = self.voting()
        confidence = self.confidence_scores()

        # Merge all results
        consensus = borda[['indicator', 'consensus_score', 'consensus_rank']]
        consensus = consensus.merge(
            fused[['indicator', 'fused_score']],
            on='indicator', how='left'
        )
        consensus = consensus.merge(
            votes[['indicator', 'votes']],
            on='indicator', how='left'
        )
        consensus = consensus.merge(
            confidence[['indicator', 'confidence']],
            on='indicator', how='left'
        )

        consensus = consensus.fillna(0)
        consensus = consensus.sort_values('consensus_rank')

        # Top consensus indicators (high rank + high confidence)
        top_consensus = consensus[
            (consensus['consensus_rank'] <= 10) &
            (consensus['confidence'] >= 0.5)
        ]['indicator'].tolist()

        return {
            "full_ranking": consensus,
            "top_consensus_indicators": top_consensus,
            "n_high_confidence": int((consensus['confidence'] >= 0.7).sum()),
            "n_unanimous_top10": int((consensus['votes'] == len(self.rankings)).sum()),
        }
