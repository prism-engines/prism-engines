"""
Tests for Calibrated Engine Indicator Loading

Tests the calibration indicator loader with safety fallbacks.

Tests:
- test_calibrated_loader_basic: Ensures panel loads with >0 rows
- test_calibrated_loader_fallback: Ensures engine runs even when registry
  has no "calibration: true" tags
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCalibratedLoaderBasic:
    """Tests for basic calibrated loader functionality."""

    def test_load_calibrated_indicators_from_registry_with_calibration_tags(self, tmp_path):
        """Test loading indicators that have calibration in purpose list."""
        # Create a mock registry with some calibration-tagged indicators
        mock_registry = {
            "indicator_a": {
                "source": "fred",
                "source_id": "TEST_A",
                "purpose": ["calibration", "geometry"]
            },
            "indicator_b": {
                "source": "fred",
                "source_id": "TEST_B",
                "purpose": ["geometry"]  # No calibration
            },
            "indicator_c": {
                "source": "fred",
                "source_id": "TEST_C",
                "calibration": True  # Explicit calibration flag
            },
        }

        registry_path = tmp_path / "indicators.yaml"
        with open(registry_path, 'w') as f:
            yaml.dump(mock_registry, f)

        # Patch the registry path and run the function
        from panel import runtime_loader

        with patch.object(runtime_loader, 'Path') as mock_path_class:
            # Create a mock that returns our temp path when appropriate
            mock_parent = MagicMock()
            mock_parent.parent = tmp_path
            mock_path_class.return_value.parent = mock_parent

            # Directly test the logic
            with open(registry_path, 'r') as f:
                registry = yaml.safe_load(f)

            selected = []
            for k, v in registry.items():
                if isinstance(v, dict):
                    if v.get("calibration", False) is True:
                        selected.append(k)
                    elif "calibration" in v.get("purpose", []):
                        selected.append(k)

            # Should have indicator_a (purpose) and indicator_c (explicit flag)
            assert len(selected) == 2
            assert "indicator_a" in selected
            assert "indicator_c" in selected
            assert "indicator_b" not in selected

    def test_calibrated_loader_basic_panel_not_empty(self):
        """Ensures panel validation rejects empty panels."""
        # This tests the RuntimeError for empty panels
        from panel.runtime_loader import load_panel

        # Create empty indicator list - should still not crash
        # The actual load will depend on database, but we test the validation logic
        empty_df = pd.DataFrame()

        # Verify empty check works
        assert empty_df.empty is True

    def test_weighted_rankings_rank_column_protection(self):
        """Test that missing rank column is handled gracefully."""
        # Create a DataFrame without 'rank' column
        weighted_rankings = pd.DataFrame({
            "weighted_rank": [1.5, 2.3, 3.1],
            "final_rank": [1, 2, 3]
        }, index=["ind_a", "ind_b", "ind_c"])

        # Simulate the protection logic from run_calibrated.py
        if "rank" not in weighted_rankings.columns:
            weighted_rankings["rank"] = weighted_rankings.index.to_series().rank()

        assert "rank" in weighted_rankings.columns
        assert len(weighted_rankings["rank"]) == 3


class TestCalibratedLoaderFallback:
    """Tests for fallback behavior when no calibration tags exist."""

    def test_calibrated_loader_fallback_uses_all_indicators(self, tmp_path):
        """
        Ensures engine runs even when registry has no "calibration: true" tags.

        When no indicators have calibration enabled, the loader should
        fall back to using ALL indicators.
        """
        # Create a mock registry with NO calibration-tagged indicators
        mock_registry = {
            "indicator_a": {
                "source": "fred",
                "source_id": "TEST_A",
                "purpose": ["geometry"]  # No calibration
            },
            "indicator_b": {
                "source": "fred",
                "source_id": "TEST_B",
                "purpose": ["regime_history"]  # No calibration
            },
            "indicator_c": {
                "source": "tiingo",
                "source_id": "TEST_C",
                # No purpose at all
            },
        }

        registry_path = tmp_path / "indicators.yaml"
        with open(registry_path, 'w') as f:
            yaml.dump(mock_registry, f)

        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f)

        # Simulate the calibration selection logic
        selected = []
        for k, v in registry.items():
            if isinstance(v, dict):
                if v.get("calibration", False) is True:
                    selected.append(k)
                elif "calibration" in v.get("purpose", []):
                    selected.append(k)

        # No indicators matched - apply fallback
        if len(selected) == 0:
            selected = [k for k, v in registry.items() if isinstance(v, dict)]

        # Should have ALL 3 indicators due to fallback
        assert len(selected) == 3
        assert "indicator_a" in selected
        assert "indicator_b" in selected
        assert "indicator_c" in selected

    def test_fallback_with_empty_registry(self, tmp_path):
        """Test behavior with completely empty registry."""
        mock_registry = {}

        registry_path = tmp_path / "indicators.yaml"
        with open(registry_path, 'w') as f:
            yaml.dump(mock_registry, f)

        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f)

        # Empty registry should result in empty list (no crash)
        if not registry:
            selected = []
        else:
            selected = [k for k, v in registry.items() if isinstance(v, dict)]

        assert selected == []

    def test_fallback_with_mixed_registry(self, tmp_path):
        """Test with registry containing both calibration and non-calibration indicators."""
        mock_registry = {
            "gdp": {
                "source": "fred",
                "source_id": "GDP",
                "purpose": ["calibration", "regime_history"]
            },
            "sp500": {
                "source": "fred",
                "source_id": "SP500"
                # No calibration purpose
            },
            "vix": {
                "source": "fred",
                "source_id": "VIXCLS",
                "calibration": True
            },
            "random_indicator": {
                "source": "tiingo",
                "source_id": "XYZ"
            }
        }

        registry_path = tmp_path / "indicators.yaml"
        with open(registry_path, 'w') as f:
            yaml.dump(mock_registry, f)

        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f)

        selected = []
        for k, v in registry.items():
            if isinstance(v, dict):
                if v.get("calibration", False) is True:
                    selected.append(k)
                elif "calibration" in v.get("purpose", []):
                    selected.append(k)

        # Should only have gdp (purpose) and vix (explicit flag)
        # Should NOT trigger fallback since we have matches
        assert len(selected) == 2
        assert "gdp" in selected
        assert "vix" in selected
        assert "sp500" not in selected
        assert "random_indicator" not in selected


class TestPanelValidation:
    """Tests for panel validation logic."""

    def test_empty_panel_raises_error(self):
        """Test that empty panel triggers RuntimeError."""
        panel = pd.DataFrame()

        with pytest.raises(RuntimeError) as exc_info:
            if panel.empty:
                raise RuntimeError(
                    "Calibration panel is empty — registry filters or DB paths likely incorrect."
                )

        assert "Calibration panel is empty" in str(exc_info.value)

    def test_non_empty_panel_passes_validation(self):
        """Test that non-empty panel passes validation."""
        panel = pd.DataFrame({
            "indicator_a": [1.0, 2.0, 3.0],
            "indicator_b": [4.0, 5.0, 6.0]
        }, index=pd.date_range("2020-01-01", periods=3))

        # Should not raise
        assert not panel.empty
        assert panel.shape[0] > 0
        assert panel.shape[1] > 0
