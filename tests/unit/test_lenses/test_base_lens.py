"""
Tests for BaseLens functionality
"""

import pytest
import pandas as pd
import numpy as np


class TestBaseLensValidation:
    """Test input validation."""

    def test_validate_rejects_empty_dataframe(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            lens.validate_input(empty_df)

    def test_validate_requires_date_column(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        no_date = sample_panel_small.drop(columns=["date"])

        with pytest.raises(ValueError, match="date"):
            lens.validate_input(no_date)

    def test_validate_requires_multiple_indicators(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        single_col = sample_panel_small[["date", "indicator_a"]]

        with pytest.raises(ValueError, match="at least 2"):
            lens.validate_input(single_col)


class TestBaseLensNormalization:
    """Test data normalization."""

    def test_zscore_normalization(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        normalized = lens.normalize_data(sample_panel_small, method="zscore")

        # Z-score normalized data should have mean ~0 and std ~1
        for col in ["indicator_a", "indicator_b"]:
            assert abs(normalized[col].mean()) < 0.1
            assert abs(normalized[col].std() - 1.0) < 0.1

    def test_minmax_normalization(self, sample_panel_small):
        from prism_engine.engine.lenses import MagnitudeLens

        lens = MagnitudeLens()
        normalized = lens.normalize_data(sample_panel_small, method="minmax")

        # MinMax should be in [0, 1]
        for col in ["indicator_a", "indicator_b"]:
            assert normalized[col].min() >= 0
            assert normalized[col].max() <= 1
