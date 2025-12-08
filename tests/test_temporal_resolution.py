"""
Tests for temporal resolution configuration and aggregation.

These tests verify:
- Resolution presets (weekly, monthly, quarterly)
- Aggregation rules for different data types
- TemporalConfig.from_resolution() factory method
- Panel aggregation functions
- Frequency detection
- Backward compatibility with legacy profiles
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from engine.orchestration.temporal_analysis import (
    RESOLUTION_PRESETS,
    AGGREGATION_RULES,
    TEMPORAL_PROFILES,
    TemporalConfig,
    TemporalResult,
    build_temporal_config,
    run_temporal_analysis_from_panel,
    get_indicator_data_type,
    aggregate_panel_to_frequency,
    detect_native_frequency,
    parse_lookback,
)


class TestResolutionPresets:
    """Tests for resolution preset definitions."""

    def test_all_resolutions_defined(self):
        """All expected resolutions should be defined."""
        assert "weekly" in RESOLUTION_PRESETS
        assert "monthly" in RESOLUTION_PRESETS
        assert "quarterly" in RESOLUTION_PRESETS

    def test_resolution_structure(self):
        """Each resolution should have required fields."""
        required_fields = [
            "frequency",
            "window_periods",
            "stride_divisor",
            "lookback_default",
        ]
        for name, preset in RESOLUTION_PRESETS.items():
            for field in required_fields:
                assert field in preset, f"Missing {field} in {name}"

    def test_weekly_resolution(self):
        """Weekly resolution should have correct values."""
        preset = RESOLUTION_PRESETS["weekly"]
        assert preset["frequency"] == "W-FRI"
        assert preset["window_periods"] == 52  # 1 year
        assert preset["stride_divisor"] == 4
        assert preset["lookback_default"] == "2Y"

    def test_monthly_resolution(self):
        """Monthly resolution should have correct values."""
        preset = RESOLUTION_PRESETS["monthly"]
        assert preset["frequency"] == "M"
        assert preset["window_periods"] == 60  # 5 years
        assert preset["stride_divisor"] == 4
        assert preset["lookback_default"] == "10Y"

    def test_quarterly_resolution(self):
        """Quarterly resolution should have correct values."""
        preset = RESOLUTION_PRESETS["quarterly"]
        assert preset["frequency"] == "Q"
        assert preset["window_periods"] == 40  # 10 years
        assert preset["stride_divisor"] == 4
        assert preset["lookback_default"] == "30Y"

    def test_stride_calculation(self):
        """Stride should be window_periods / stride_divisor."""
        for name, preset in RESOLUTION_PRESETS.items():
            expected_stride = preset["window_periods"] // preset["stride_divisor"]
            assert expected_stride > 0, f"Invalid stride for {name}"


class TestAggregationRules:
    """Tests for aggregation rule definitions."""

    def test_all_types_defined(self):
        """All expected data types should have aggregation rules."""
        expected_types = [
            "price",
            "yield",
            "rate",
            "ratio",
            "volume",
            "flow",
            "volatility",
            "index",
            "level",
            "change",
            "default",
        ]
        for data_type in expected_types:
            assert data_type in AGGREGATION_RULES

    def test_price_uses_last(self):
        """Price data should use 'last' aggregation."""
        assert AGGREGATION_RULES["price"] == "last"

    def test_yield_uses_mean(self):
        """Yield data should use 'mean' aggregation."""
        assert AGGREGATION_RULES["yield"] == "mean"

    def test_volume_uses_sum(self):
        """Volume data should use 'sum' aggregation."""
        assert AGGREGATION_RULES["volume"] == "sum"

    def test_volatility_uses_mean(self):
        """Volatility data should use 'mean' aggregation."""
        assert AGGREGATION_RULES["volatility"] == "mean"


class TestGetIndicatorDataType:
    """Tests for indicator data type detection."""

    def test_spy_is_price(self):
        """SPY should be detected as price."""
        assert get_indicator_data_type("spy_close") == "price"
        assert get_indicator_data_type("SPY") == "price"

    def test_dgs10_is_yield(self):
        """DGS10 should be detected as yield."""
        assert get_indicator_data_type("dgs10") == "yield"
        assert get_indicator_data_type("t10y2y") == "yield"

    def test_vix_is_volatility(self):
        """VIX should be detected as volatility."""
        assert get_indicator_data_type("vix") == "volatility"
        assert get_indicator_data_type("realized_vol") == "volatility"

    def test_volume_is_volume(self):
        """Volume indicators should be detected as volume."""
        assert get_indicator_data_type("spy_volume") == "volume"
        assert get_indicator_data_type("turnover") == "volume"

    def test_cpi_is_index(self):
        """CPI should be detected as index."""
        assert get_indicator_data_type("cpi") == "index"
        assert get_indicator_data_type("gdp") == "index"

    def test_unknown_is_default(self):
        """Unknown indicators should return default."""
        assert get_indicator_data_type("unknown_indicator") == "default"

    def test_registry_override(self):
        """Registry should override heuristic."""
        registry = {"custom": {"data_type": "volatility"}}
        assert get_indicator_data_type("custom", registry) == "volatility"


class TestTemporalConfigFromResolution:
    """Tests for TemporalConfig.from_resolution() factory method."""

    def test_default_resolution_is_monthly(self):
        """Default resolution should be monthly."""
        config = TemporalConfig.from_resolution()
        assert config.resolution == "monthly"
        assert config.frequency == "M"
        assert config.window_periods == 60

    def test_weekly_resolution(self):
        """Weekly resolution should configure correctly."""
        config = TemporalConfig.from_resolution("weekly")
        assert config.resolution == "weekly"
        assert config.frequency == "W-FRI"
        assert config.window_periods == 52
        assert config.stride_periods == 13

    def test_quarterly_resolution(self):
        """Quarterly resolution should configure correctly."""
        config = TemporalConfig.from_resolution("quarterly")
        assert config.resolution == "quarterly"
        assert config.frequency == "Q"
        assert config.window_periods == 40
        assert config.stride_periods == 10

    def test_override_window(self):
        """Window periods should be overridable."""
        config = TemporalConfig.from_resolution("monthly", window_periods=24)
        assert config.window_periods == 24

    def test_override_stride(self):
        """Stride periods should be overridable."""
        config = TemporalConfig.from_resolution("monthly", stride_periods=5)
        assert config.stride_periods == 5

    def test_override_lookback(self):
        """Lookback should be overridable."""
        config = TemporalConfig.from_resolution("monthly", lookback="5Y")
        assert config.lookback == "5Y"

    def test_all_lenses_available(self):
        """All 14 lenses should be available by default."""
        config = TemporalConfig.from_resolution()
        assert len(config.lenses) == 14


class TestBuildTemporalConfig:
    """Tests for build_temporal_config function."""

    def test_resolution_based_config(self):
        """Resolution parameter should work."""
        cfg = build_temporal_config(resolution="weekly")
        assert cfg.resolution == "weekly"
        assert cfg.frequency == "W-FRI"

    def test_default_is_monthly(self):
        """Default should be monthly resolution."""
        cfg = build_temporal_config()
        assert cfg.resolution == "monthly"

    def test_invalid_resolution_raises(self):
        """Invalid resolution should raise ValueError."""
        with pytest.raises(ValueError):
            build_temporal_config(resolution="invalid")

    def test_legacy_profile_works(self):
        """Legacy profile parameter should still work."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = build_temporal_config(profile="standard")
            assert cfg.profile_name == "standard"
            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_resolution_overrides_profile(self):
        """Resolution should take precedence over profile."""
        cfg = build_temporal_config(resolution="weekly", profile="standard")
        assert cfg.resolution == "weekly"

    def test_lookback_override(self):
        """Lookback should be overridable."""
        cfg = build_temporal_config(resolution="monthly", lookback="5Y")
        assert cfg.lookback == "5Y"

    def test_window_override(self):
        """Window should be overridable."""
        cfg = build_temporal_config(resolution="monthly", window=24)
        assert cfg.window_periods == 24

    def test_systems_filter(self):
        """Systems filter should be passed through."""
        cfg = build_temporal_config(resolution="monthly", systems=["finance"])
        assert cfg.systems == ("finance",)

    def test_lenses_override(self):
        """Lenses should be overridable."""
        cfg = build_temporal_config(
            resolution="monthly", lenses=["magnitude", "pca"]
        )
        assert cfg.lenses == ("magnitude", "pca")


class TestAggregatePanelToFrequency:
    """Tests for panel frequency aggregation."""

    @pytest.fixture
    def daily_panel(self):
        """Create a sample daily panel."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        np.random.seed(42)
        return pd.DataFrame(
            {
                "spy_close": np.random.randn(len(dates)).cumsum() + 100,
                "dgs10": np.random.randn(len(dates)) * 0.5 + 2.0,
                "spy_volume": np.random.randint(1000, 10000, len(dates)),
                "vix": np.abs(np.random.randn(len(dates))) * 5 + 15,
            },
            index=dates,
        )

    def test_aggregate_to_monthly(self, daily_panel):
        """Aggregation to monthly should work."""
        result = aggregate_panel_to_frequency(daily_panel, "M")
        assert len(result) == 12  # 12 months
        assert all(col in result.columns for col in daily_panel.columns)

    def test_aggregate_to_weekly(self, daily_panel):
        """Aggregation to weekly should work."""
        result = aggregate_panel_to_frequency(daily_panel, "W-FRI")
        assert len(result) >= 50  # ~52 weeks

    def test_aggregate_to_quarterly(self, daily_panel):
        """Aggregation to quarterly should work."""
        result = aggregate_panel_to_frequency(daily_panel, "Q")
        assert len(result) == 4  # 4 quarters

    def test_price_uses_last(self, daily_panel):
        """Price columns should use last value."""
        result = aggregate_panel_to_frequency(daily_panel, "M")
        # Last value of January should approximately match
        jan_data = daily_panel.loc["2020-01":"2020-01", "spy_close"]
        expected_last = jan_data.iloc[-1]
        assert abs(result.loc["2020-01-31", "spy_close"] - expected_last) < 0.01

    def test_volume_uses_sum(self, daily_panel):
        """Volume columns should use sum."""
        result = aggregate_panel_to_frequency(daily_panel, "M")
        # Sum of January should match
        jan_sum = daily_panel.loc["2020-01":"2020-01", "spy_volume"].sum()
        assert result.loc["2020-01-31", "spy_volume"] == jan_sum


class TestDetectNativeFrequency:
    """Tests for native frequency detection."""

    def test_detect_daily(self):
        """Daily frequency should be detected."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        series = pd.Series(range(100), index=dates)
        assert detect_native_frequency(series) == "daily"

    def test_detect_weekly(self):
        """Weekly frequency should be detected."""
        dates = pd.date_range("2020-01-01", periods=52, freq="W")
        series = pd.Series(range(52), index=dates)
        assert detect_native_frequency(series) == "weekly"

    def test_detect_monthly(self):
        """Monthly frequency should be detected."""
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        series = pd.Series(range(24), index=dates)
        assert detect_native_frequency(series) == "monthly"

    def test_detect_quarterly(self):
        """Quarterly frequency should be detected."""
        dates = pd.date_range("2020-01-01", periods=12, freq="Q")
        series = pd.Series(range(12), index=dates)
        assert detect_native_frequency(series) == "quarterly"

    def test_empty_series(self):
        """Empty series should return unknown."""
        series = pd.Series([], dtype=float)
        assert detect_native_frequency(series) == "unknown"


class TestParseLookback:
    """Tests for lookback string parsing."""

    def test_parse_years(self):
        """Year lookback should work."""
        end_date = pd.Timestamp("2020-12-31")
        result = parse_lookback("5Y", end_date)
        assert result.year == 2015

    def test_parse_months(self):
        """Month lookback should work."""
        end_date = pd.Timestamp("2020-12-31")
        result = parse_lookback("6M", end_date)
        assert result.month == 6

    def test_parse_weeks(self):
        """Week lookback should work."""
        end_date = pd.Timestamp("2020-12-31")
        result = parse_lookback("4W", end_date)
        expected = end_date - pd.DateOffset(weeks=4)
        assert result == expected

    def test_parse_date_string(self):
        """Date string should work."""
        result = parse_lookback("2015-01-01")
        assert result == pd.Timestamp("2015-01-01")


class TestRunTemporalAnalysisFromPanel:
    """Tests for run_temporal_analysis_from_panel with resolution config."""

    @pytest.fixture
    def sample_panel(self):
        """Create a sample panel for testing."""
        dates = pd.date_range("2000-01-01", periods=500, freq="D")
        np.random.seed(42)
        data = {
            "indicator_a": np.random.randn(500).cumsum(),
            "indicator_b": np.random.randn(500).cumsum(),
            "indicator_c": np.random.randn(500).cumsum(),
        }
        return pd.DataFrame(data, index=dates)

    def test_run_with_resolution_config(self, sample_panel):
        """Analysis should work with resolution config."""
        cfg = build_temporal_config(resolution="monthly")
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        assert isinstance(result, TemporalResult)
        assert isinstance(result.scores, pd.DataFrame)
        assert result.config.resolution == "monthly"

    def test_run_with_weekly_resolution(self, sample_panel):
        """Analysis should work with weekly resolution."""
        cfg = build_temporal_config(resolution="weekly")
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        assert isinstance(result, TemporalResult)
        assert result.config.frequency == "W-FRI"


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with legacy API."""

    def test_temporal_profiles_still_exist(self):
        """TEMPORAL_PROFILES should still be available."""
        assert "chromebook" in TEMPORAL_PROFILES
        assert "standard" in TEMPORAL_PROFILES
        assert "powerful" in TEMPORAL_PROFILES

    def test_profile_based_config_still_works(self):
        """Profile-based configuration should still work."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = build_temporal_config(profile="chromebook")
            assert cfg is not None
            assert len(cfg.lenses) > 0

    def test_runner_legacy_imports(self):
        """Legacy imports from runner should still work."""
        from start.temporal_runner import (
            PERFORMANCE_PROFILES,
            run_temporal_analysis,
            get_summary,
            find_trending,
        )

        assert "chromebook" in PERFORMANCE_PROFILES
        assert "standard" in PERFORMANCE_PROFILES
