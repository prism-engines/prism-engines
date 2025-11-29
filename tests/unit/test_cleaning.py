"""
Tests for data cleaning module
"""

import pytest
import pandas as pd
import numpy as np


class TestNaNAnalyzer:
    """Test NaN analysis functionality."""

    def test_summary_counts_nan(self, sample_panel_with_nan):
        from prism_engine.cleaning import NaNAnalyzer

        analyzer = NaNAnalyzer(sample_panel_with_nan)
        summary = analyzer.summary()

        # "clean" column should have 0 NaN
        assert summary["clean"]["n_nan"] == 0

        # "sparse" and "gappy" should have NaN
        assert summary["sparse"]["n_nan"] > 0
        assert summary["gappy"]["n_nan"] > 0

    def test_gap_analysis_finds_gaps(self, sample_panel_with_nan):
        from prism_engine.cleaning import NaNAnalyzer

        analyzer = NaNAnalyzer(sample_panel_with_nan)
        gaps = analyzer.gap_analysis()

        # "gappy" has a large gap of 21 values
        assert gaps["gappy"]["max_gap"] >= 20


class TestNaNStrategies:
    """Test NaN filling strategies."""

    def test_ffill_fills_gaps(self, sample_panel_with_nan):
        from prism_engine.cleaning import get_strategy

        strategy = get_strategy("ffill")
        series = sample_panel_with_nan["sparse"].copy()

        filled = strategy.fill(series)

        # Should have fewer NaN
        assert filled.isna().sum() < series.isna().sum()

    def test_linear_interpolation(self, sample_panel_with_nan):
        from prism_engine.cleaning import get_strategy

        strategy = get_strategy("linear")
        series = sample_panel_with_nan["gappy"].copy()

        filled = strategy.fill(series)

        # Should have fewer NaN
        assert filled.isna().sum() < series.isna().sum()

    def test_unknown_strategy_raises(self):
        from prism_engine.cleaning import get_strategy

        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent_strategy")
