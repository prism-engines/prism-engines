"""
Tests for parallel temporal analysis.

These tests verify:
- Serial vs parallel equivalence (results should be identical)
- Worker auto-tuning behavior
- RAM guard behavior
- Small task fallback to serial execution
- Parallel configuration in TemporalConfig
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from engine.utils.parallel import (
    get_cpu_count,
    get_system_memory_gb,
    get_optimal_workers,
    should_use_parallel,
    run_parallel,
    MIN_RAM_GB_FOR_PARALLEL,
    MIN_ITEMS_PER_WORKER,
)
from engine.orchestration.temporal_analysis import (
    TemporalConfig,
    build_temporal_config,
    run_temporal_analysis_from_panel,
)


# ---------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_panel():
    """Create a sample panel for testing."""
    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=1000, freq="D")
    data = {
        "indicator_a": np.random.randn(1000).cumsum(),
        "indicator_b": np.random.randn(1000).cumsum(),
        "indicator_c": np.random.randn(1000).cumsum(),
        "indicator_d": np.random.randn(1000).cumsum(),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def small_panel():
    """Create a small panel that should trigger serial execution."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = {
        "indicator_a": np.random.randn(100).cumsum(),
        "indicator_b": np.random.randn(100).cumsum(),
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------
# Tests for parallel utility functions
# ---------------------------------------------------------------------

class TestGetCpuCount:
    """Tests for get_cpu_count function."""

    def test_returns_positive_integer(self):
        """CPU count should always be positive."""
        count = get_cpu_count()
        assert isinstance(count, int)
        assert count >= 1

    def test_fallback_without_psutil(self):
        """Should work without psutil installed."""
        with patch.dict('sys.modules', {'psutil': None}):
            with patch('os.cpu_count', return_value=4):
                # Re-import to test fallback
                from engine.utils import parallel
                count = parallel.get_cpu_count()
                assert count >= 1


class TestGetSystemMemoryGb:
    """Tests for get_system_memory_gb function."""

    def test_returns_positive_float(self):
        """Memory should always be positive."""
        mem = get_system_memory_gb()
        assert isinstance(mem, float)
        assert mem > 0

    def test_fallback_returns_default(self):
        """Should return default when psutil not available."""
        with patch.dict('sys.modules', {'psutil': None}):
            from engine.utils import parallel
            mem = parallel.get_system_memory_gb()
            assert mem == 8.0  # Default fallback


class TestGetOptimalWorkers:
    """Tests for get_optimal_workers function."""

    def test_single_cpu_returns_one(self):
        """Single CPU should return 1 worker."""
        with patch('engine.utils.parallel.get_cpu_count', return_value=1):
            assert get_optimal_workers() == 1

    def test_two_cpus_returns_one(self):
        """Two CPUs should return 1 worker."""
        with patch('engine.utils.parallel.get_cpu_count', return_value=2):
            assert get_optimal_workers() == 1

    def test_four_cpus_returns_two(self):
        """Four CPUs should return 2 workers."""
        with patch('engine.utils.parallel.get_cpu_count', return_value=4):
            assert get_optimal_workers() == 2

    def test_eight_cpus_returns_six(self):
        """Eight CPUs should return 6 workers (cpu - 2)."""
        with patch('engine.utils.parallel.get_cpu_count', return_value=8):
            assert get_optimal_workers() == 6

    def test_max_workers_cap(self):
        """Max workers should cap the result."""
        with patch('engine.utils.parallel.get_cpu_count', return_value=16):
            assert get_optimal_workers(max_workers=4) == 4

    def test_always_returns_at_least_one(self):
        """Should never return less than 1."""
        with patch('engine.utils.parallel.get_cpu_count', return_value=0):
            assert get_optimal_workers() >= 1


class TestShouldUseParallel:
    """Tests for should_use_parallel function."""

    def test_low_ram_disables_parallel(self):
        """Low RAM should disable parallel execution."""
        with patch('engine.utils.parallel.get_system_memory_gb', return_value=2.0):
            assert should_use_parallel(100) is False

    def test_sufficient_ram_enables_parallel(self):
        """Sufficient RAM should allow parallel execution."""
        with patch('engine.utils.parallel.get_system_memory_gb', return_value=8.0):
            with patch('engine.utils.parallel.get_optimal_workers', return_value=4):
                assert should_use_parallel(100) is True

    def test_small_item_count_disables_parallel(self):
        """Too few items should disable parallel."""
        with patch('engine.utils.parallel.get_system_memory_gb', return_value=8.0):
            with patch('engine.utils.parallel.get_optimal_workers', return_value=4):
                # Need at least 4 * 2 = 8 items
                assert should_use_parallel(4) is False
                assert should_use_parallel(8) is True

    def test_single_worker_disables_parallel(self):
        """Single worker should disable parallel."""
        with patch('engine.utils.parallel.get_system_memory_gb', return_value=8.0):
            with patch('engine.utils.parallel.get_optimal_workers', return_value=1):
                assert should_use_parallel(100) is False

    def test_custom_min_ram(self):
        """Custom min_ram_gb should be respected."""
        with patch('engine.utils.parallel.get_system_memory_gb', return_value=6.0):
            with patch('engine.utils.parallel.get_optimal_workers', return_value=4):
                assert should_use_parallel(100, min_ram_gb=8.0) is False
                assert should_use_parallel(100, min_ram_gb=4.0) is True


class TestRunParallel:
    """Tests for run_parallel function."""

    def test_empty_items_returns_empty(self):
        """Empty items should return empty list."""
        result = run_parallel(lambda x: x * 2, [])
        assert result == []

    def test_serial_execution_on_small_input(self):
        """Small input should use serial execution."""
        items = [1, 2, 3]
        result = run_parallel(lambda x: x * 2, items)
        assert result == [2, 4, 6]

    def test_force_serial_flag(self):
        """force_serial should bypass parallel execution."""
        items = list(range(100))
        result = run_parallel(lambda x: x * 2, items, force_serial=True)
        assert result == [x * 2 for x in items]

    def test_maintains_order(self):
        """Results should maintain input order."""
        items = list(range(20))
        result = run_parallel(lambda x: x * 2, items)
        assert result == [x * 2 for x in items]

    def test_with_max_workers(self):
        """max_workers parameter should be respected."""
        items = list(range(50))
        result = run_parallel(lambda x: x + 1, items, max_workers=2)
        assert result == [x + 1 for x in items]


# ---------------------------------------------------------------------
# Tests for parallel temporal analysis
# ---------------------------------------------------------------------

class TestTemporalConfigParallel:
    """Tests for TemporalConfig parallel settings."""

    def test_default_parallel_enabled(self):
        """Parallel should be enabled by default."""
        cfg = TemporalConfig(profile_name="test")
        assert cfg.use_parallel is True
        assert cfg.max_workers is None

    def test_config_with_parallel_disabled(self):
        """Can explicitly disable parallel."""
        cfg = TemporalConfig(profile_name="test", use_parallel=False)
        assert cfg.use_parallel is False

    def test_config_with_max_workers(self):
        """Can set max_workers."""
        cfg = TemporalConfig(profile_name="test", max_workers=4)
        assert cfg.max_workers == 4

    def test_build_config_parallel_params(self):
        """build_temporal_config should accept parallel params."""
        cfg = build_temporal_config(
            profile="standard",
            use_parallel=False,
            max_workers=2,
        )
        assert cfg.use_parallel is False
        assert cfg.max_workers == 2


class TestSerialParallelEquivalence:
    """Tests verifying serial and parallel produce identical results."""

    def test_results_match(self, sample_panel):
        """Serial and parallel execution should produce identical results."""
        # Run with parallel enabled
        cfg_parallel = build_temporal_config(
            profile="chromebook",
            use_parallel=True,
        )
        result_parallel = run_temporal_analysis_from_panel(sample_panel, cfg_parallel)

        # Run with parallel disabled
        cfg_serial = build_temporal_config(
            profile="chromebook",
            use_parallel=False,
        )
        result_serial = run_temporal_analysis_from_panel(sample_panel, cfg_serial)

        # Compare scores
        pd.testing.assert_frame_equal(
            result_parallel.scores,
            result_serial.scores,
            check_exact=False,
            rtol=1e-10,
        )

        # Compare key metadata
        assert result_parallel.metadata["n_dates"] == result_serial.metadata["n_dates"]
        assert result_parallel.metadata["n_indicators"] == result_serial.metadata["n_indicators"]
        assert result_parallel.metadata["n_windows"] == result_serial.metadata["n_windows"]

    def test_rankings_match(self, sample_panel):
        """Rankings should be identical between serial and parallel."""
        cfg_parallel = build_temporal_config(profile="chromebook", use_parallel=True)
        cfg_serial = build_temporal_config(profile="chromebook", use_parallel=False)

        result_parallel = run_temporal_analysis_from_panel(sample_panel, cfg_parallel)
        result_serial = run_temporal_analysis_from_panel(sample_panel, cfg_serial)

        # All rank columns should match
        rank_cols = [c for c in result_parallel.scores.columns if c.endswith("_rank")]
        for col in rank_cols:
            np.testing.assert_array_almost_equal(
                result_parallel.scores[col].values,
                result_serial.scores[col].values,
            )


class TestSmallTaskFallback:
    """Tests for automatic fallback to serial on small tasks."""

    def test_small_panel_uses_serial(self, small_panel):
        """Small panels should fall back to serial execution."""
        cfg = build_temporal_config(
            profile="chromebook",
            use_parallel=True,
        )
        result = run_temporal_analysis_from_panel(small_panel, cfg)

        # Should complete without error
        assert isinstance(result.scores, pd.DataFrame)
        # Note: We can't directly test that serial was used,
        # but we verify it completes correctly

    def test_parallel_disabled_in_metadata(self, small_panel):
        """Metadata should reflect actual execution mode."""
        cfg = build_temporal_config(
            profile="chromebook",
            use_parallel=False,
        )
        result = run_temporal_analysis_from_panel(small_panel, cfg)

        # When explicitly disabled, metadata should show it
        assert result.metadata.get("parallel_enabled") is False


class TestWorkerAutoTuning:
    """Tests for worker auto-tuning behavior."""

    def test_optimal_workers_used(self, sample_panel):
        """Should use optimal number of workers."""
        cfg = build_temporal_config(
            profile="chromebook",
            use_parallel=True,
            max_workers=None,  # Auto-tune
        )
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        # Result should complete successfully
        assert result.metadata["n_dates"] > 0

    def test_max_workers_respected(self, sample_panel):
        """max_workers should cap the number of workers."""
        cfg = build_temporal_config(
            profile="chromebook",
            use_parallel=True,
            max_workers=2,
        )
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        # Workers in metadata should not exceed max
        assert result.metadata.get("workers", 1) <= 2


class TestRAMGuard:
    """Tests for RAM guard behavior."""

    def test_low_ram_falls_back_to_serial(self, sample_panel):
        """Low RAM should trigger serial fallback."""
        with patch('engine.utils.parallel.get_system_memory_gb', return_value=2.0):
            cfg = build_temporal_config(
                profile="chromebook",
                use_parallel=True,
            )
            result = run_temporal_analysis_from_panel(sample_panel, cfg)

            # Should still complete successfully (via serial fallback)
            assert isinstance(result.scores, pd.DataFrame)
            assert result.metadata["n_dates"] > 0


class TestParallelMetadata:
    """Tests for parallel execution metadata."""

    def test_metadata_includes_parallel_info(self, sample_panel):
        """Result metadata should include parallel execution info."""
        cfg = build_temporal_config(
            profile="chromebook",
            use_parallel=True,
        )
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        assert "parallel_enabled" in result.metadata
        assert "workers" in result.metadata

    def test_serial_metadata(self, sample_panel):
        """Serial execution should report correct metadata."""
        cfg = build_temporal_config(
            profile="chromebook",
            use_parallel=False,
        )
        result = run_temporal_analysis_from_panel(sample_panel, cfg)

        assert result.metadata.get("parallel_enabled") is False
        assert result.metadata.get("workers") == 1
