"""
Tests for PCALens
"""

import pytest
import pandas as pd
import numpy as np


class TestPCALens:
    """Test PCALens functionality."""

    def test_explained_variance_sums_to_one(self, sample_panel_small):
        from prism_engine.engine.lenses import PCALens

        lens = PCALens()
        result = lens.analyze(sample_panel_small)

        total_var = sum(result["explained_variance_ratio"])
        assert abs(total_var - 1.0) < 0.01

    def test_cumulative_variance_is_monotonic(self, sample_panel_small):
        from prism_engine.engine.lenses import PCALens

        lens = PCALens()
        result = lens.analyze(sample_panel_small)

        cumulative = result["cumulative_variance"]

        # Should be monotonically increasing
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i-1]

    def test_loadings_have_correct_shape(self, sample_panel_small):
        from prism_engine.engine.lenses import PCALens

        lens = PCALens()
        result = lens.analyze(sample_panel_small, n_components=2)

        loadings = result["loadings"]

        # Should have 2 PCs
        assert "PC1" in loadings
        assert "PC2" in loadings

        # Each PC should have loadings for all indicators
        n_indicators = len(sample_panel_small.columns) - 1  # Exclude date
        assert len(loadings["PC1"]) == n_indicators
