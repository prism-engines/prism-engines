"""
Tests for temporal resolution profiles and configuration.

These tests verify:
- Profile registry contains expected profiles
- build_temporal_config works correctly with defaults and overrides
- TemporalConfig dataclass behaves as expected
"""

import pytest
import pandas as pd
import numpy as np

from engine.orchestration.temporal_analysis import (
    TEMPORAL_PROFILES,
    TemporalProfile,
    TemporalConfig,
    TemporalResult,
    build_temporal_config,
    run_temporal_analysis_from_panel,
)


class TestTemporalProfiles:
    """Tests for temporal profile registry."""

    def test_profiles_registered(self):
        """All expected profiles should be registered."""
        assert "chromebook" in TEMPORAL_PROFILES
        assert "standard" in TEMPORAL_PROFILES
        assert "powerful" in TEMPORAL_PROFILES

    def test_profile_structure(self):
        """Each profile should have required fields."""
        for name, profile in TEMPORAL_PROFILES.items():
            assert isinstance(profile, TemporalProfile)
            assert profile.name == name
            assert len(profile.description) > 0
            assert profile.step_months > 0
            assert len(profile.window_months) >= 2
            assert len(profile.lenses) >= 3
            assert profile.min_history_years >= 5
            assert profile.frequency in ("D", "W", "M")

    def test_chromebook_profile(self):
        """Chromebook profile should be optimized for low-CPU machines."""
        profile = TEMPORAL_PROFILES["chromebook"]
        assert profile.step_months >= 1.0  # Larger steps for speed
        assert len(profile.lenses) <= 4  # Fewer lenses

    def test_powerful_profile(self):
        """Powerful profile should have high resolution."""
        profile = TEMPORAL_PROFILES["powerful"]
        assert profile.step_months <= 1.0  # Finer steps
        assert len(profile.lenses) >= 4  # More lenses
        assert len(profile.window_months) >= 4  # More windows


class TestBuildTemporalConfig:
    """Tests for build_temporal_config function."""

    def test_build_temporal_config_basic(self):
        """Basic config building with defaults."""
        cfg = build_temporal_config(profile="standard")
        assert cfg.profile_name == "standard"
        assert cfg.step_months > 0
        assert len(cfg.window_months) >= 3
        assert len(cfg.lenses) >= 3

    def test_build_temporal_config_overrides(self):
        """Config building with custom overrides."""
        cfg = build_temporal_config(
            profile="standard",
            start_date="2000-01-01",
            end_date="2010-01-01",
            step_months=0.5,
            lenses=["magnitude"],
        )
        assert cfg.start_date == pd.Timestamp("2000-01-01")
        assert cfg.end_date == pd.Timestamp("2010-01-01")
        assert cfg.step_months == 0.5
        assert list(cfg.lenses) == ["magnitude"]

    def test_build_temporal_config_with_systems(self):
        """Config building with system filters."""
        cfg = build_temporal_config(
            profile="standard",
            systems=["finance", "climate"],
        )
        assert cfg.systems == ("finance", "climate")

    def test_build_temporal_config_with_indicators(self):
        """Config building with indicator filters."""
        cfg = build_temporal_config(
            profile="standard",
            indicators=["sp500", "vix", "t10y2y"],
        )
        assert cfg.indicators == ("sp500", "vix", "t10y2y")

    def test_build_temporal_config_invalid_profile(self):
        """Invalid profile should raise ValueError."""
        with pytest.raises(ValueError):
            build_temporal_config(profile="nonexistent")

    def test_build_temporal_config_all_profiles(self):
        """All profiles should build without error."""
        for profile_name in TEMPORAL_PROFILES:
            cfg = build_temporal_config(profile=profile_name)
            assert cfg.profile_name == profile_name

    def test_build_temporal_config_window_months_override(self):
        """Window months can be overridden."""
        cfg = build_temporal_config(
            profile="standard",
            window_months=[6, 12],
        )
        assert list(cfg.window_months) == [6, 12]

    def test_build_temporal_config_frequency_override(self):
        """Frequency can be overridden."""
        cfg = build_temporal_config(
            profile="standard",
            frequency="W",
        )
        assert cfg.frequency == "W"


class TestTemporalConfig:
    """Tests for TemporalConfig dataclass."""

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        cfg = TemporalConfig(profile_name="test")
        assert cfg.step_months == 1.0
        assert cfg.frequency == "M"
        assert cfg.min_history_years == 10
        assert cfg.systems is None
        assert cfg.indicators is None

    def test_config_with_dates(self):
        """Config should accept date parameters."""
        cfg = TemporalConfig(
            profile_name="test",
            start_date=pd.Timestamp("2000-01-01"),
            end_date=pd.Timestamp("2020-12-31"),
        )
        assert cfg.start_date == pd.Timestamp("2000-01-01")
        assert cfg.end_date == pd.Timestamp("2020-12-31")


class TestTemporalResult:
    """Tests for TemporalResult dataclass."""

    def test_result_structure(self):
        """Result should contain required fields."""
        cfg = build_temporal_config(profile="standard")
        scores = pd.DataFrame(
            {"test_score": [0.5, 0.6, 0.7]},
            index=pd.date_range("2020-01-01", periods=3, freq="M"),
        )
        metadata = {"test_key": "test_value"}

        result = TemporalResult(config=cfg, scores=scores, metadata=metadata)

        assert result.config == cfg
        assert isinstance(result.scores, pd.DataFrame)
        assert result.metadata == metadata


class TestRunTemporalAnalysisFromPanel:
    """Tests for run_temporal_analysis_from_panel function."""

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

    def test_run_from_panel_basic(self, sample_panel):
        """Basic panel analysis should work."""
        cfg = build_temporal_config(profile="chromebook")
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        assert isinstance(result, TemporalResult)
        assert isinstance(result.scores, pd.DataFrame)
        assert result.config == cfg
        assert "profile" in result.metadata

    def test_run_from_panel_empty_raises(self):
        """Empty panel should raise ValueError."""
        empty_panel = pd.DataFrame()
        cfg = build_temporal_config(profile="standard")

        with pytest.raises(ValueError):
            run_temporal_analysis_from_panel(empty_panel, cfg)

    def test_run_from_panel_with_date_filter(self, sample_panel):
        """Date filtering should work."""
        cfg = build_temporal_config(
            profile="chromebook",
            start_date="2000-06-01",
            end_date="2001-01-01",
        )
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        assert isinstance(result, TemporalResult)
        # Result should have fewer dates than full panel
        assert result.scores.shape[0] <= sample_panel.shape[0]

    def test_run_from_panel_metadata(self, sample_panel):
        """Result metadata should contain expected fields."""
        cfg = build_temporal_config(profile="chromebook")
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        assert "n_dates" in result.metadata
        assert "n_indicators" in result.metadata
        assert "profile" in result.metadata
        assert "elapsed_seconds" in result.metadata


class TestIntegrationWithRunner:
    """Integration tests with temporal_runner module."""

    def test_import_from_runner(self):
        """Runner module should be importable."""
        from start.temporal_runner import (
            PERFORMANCE_PROFILES,
            run_temporal_analysis,
            get_summary,
            find_trending,
            quick_start,
            get_default_db_path,
        )

        assert "chromebook" in PERFORMANCE_PROFILES
        assert "standard" in PERFORMANCE_PROFILES
        assert "powerful" in PERFORMANCE_PROFILES

    def test_runner_cli_dry_run(self):
        """Runner CLI should work in dry-run mode."""
        from start.temporal_runner import main

        # Dry run should return 0
        result = main(["--profile", "chromebook", "--dry-run"])
        assert result == 0
