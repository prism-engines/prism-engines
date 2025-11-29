"""
Integration tests for the full PRISM pipeline
"""

import pytest
import pandas as pd
import numpy as np


class TestIndicatorEngine:
    """Test the high-level IndicatorEngine."""

    def test_basic_analysis_runs(self, sample_panel_small):
        from prism_engine.engine.orchestration import IndicatorEngine

        engine = IndicatorEngine()
        results = engine.analyze(sample_panel_small, mode="basic")

        assert "timestamp" in results
        assert "top_indicators" in results
        assert "consensus" in results

    def test_quick_analysis_returns_dataframe(self, sample_panel_small):
        from prism_engine.engine.orchestration import IndicatorEngine

        engine = IndicatorEngine()
        ranking = engine.quick_analysis(sample_panel_small)

        assert isinstance(ranking, pd.DataFrame)
        assert len(ranking) > 0


class TestLensComparator:
    """Test lens comparison functionality."""

    def test_run_multiple_lenses(self, sample_panel_small):
        from prism_engine.engine.orchestration import LensComparator

        comparator = LensComparator()
        results = comparator.run_lenses(
            sample_panel_small,
            lenses=["magnitude", "pca"]
        )

        assert "magnitude" in results["results"]
        assert "pca" in results["results"]

    def test_agreement_matrix_is_symmetric(self, sample_panel_small):
        from prism_engine.engine.orchestration import LensComparator

        comparator = LensComparator()
        comparator.run_lenses(
            sample_panel_small,
            lenses=["magnitude", "pca", "influence"]
        )

        agreement = comparator.get_agreement_matrix()

        # Matrix should be symmetric
        for i in range(len(agreement)):
            for j in range(len(agreement)):
                assert abs(agreement.iloc[i, j] - agreement.iloc[j, i]) < 0.01


class TestConsensusEngine:
    """Test consensus building."""

    def test_borda_count_ranking(self, sample_panel_small):
        from prism_engine.engine.orchestration import LensComparator, ConsensusEngine

        comparator = LensComparator()
        comparator.run_lenses(sample_panel_small, lenses=["magnitude", "pca"])

        consensus = ConsensusEngine(comparator.rankings)
        borda = consensus.borda_count()

        assert "indicator" in borda.columns
        assert "consensus_score" in borda.columns
        assert "consensus_rank" in borda.columns

    def test_voting_counts_correctly(self, sample_panel_small):
        from prism_engine.engine.orchestration import LensComparator, ConsensusEngine

        comparator = LensComparator()
        comparator.run_lenses(sample_panel_small, lenses=["magnitude", "pca", "influence"])

        consensus = ConsensusEngine(comparator.rankings)
        votes = consensus.voting(top_n=5)

        # Vote counts should not exceed number of lenses
        assert votes["votes"].max() <= 3
