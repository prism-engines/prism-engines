"""
Tests for MagnitudeLens
"""

import pytest
import pandas as pd
import numpy as np


class TestMagnitudeLens:
    """Test MagnitudeLens functionality."""

    def test_analyze_returns_required_keys(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        result = lens.analyze(sample_panel_small)

        required_keys = [
            "current_magnitude",
            "mean_magnitude",
            "std_magnitude",
            "indicator_contributions",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_contributions_sum_to_one(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        result = lens.analyze(sample_panel_small)

        contributions = result["indicator_contributions"]
        total = sum(contributions.values())

        assert abs(total - 1.0) < 0.01, f"Contributions sum to {total}, expected 1.0"

    def test_rank_indicators_returns_dataframe(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        ranking = lens.rank_indicators(sample_panel_small)

        assert isinstance(ranking, pd.DataFrame)
        assert "indicator" in ranking.columns
        assert "score" in ranking.columns
        assert "rank" in ranking.columns

    def test_ranking_is_sorted(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        ranking = lens.rank_indicators(sample_panel_small)

        # Check ranks are 1, 2, 3, ...
        expected_ranks = list(range(1, len(ranking) + 1))
        assert ranking["rank"].tolist() == expected_ranks

        # Check sorted by score descending
        scores = ranking["score"].tolist()
        assert scores == sorted(scores, reverse=True)
