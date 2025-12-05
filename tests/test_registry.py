"""
Tests for the metric registry loader and validator.

Tests:
- load_metric_registry() doesn't raise
- validate_registry() doesn't raise on valid registry
- No duplicate names across sections
- All depends_on references found in market/economic
"""

import pytest
import json
import tempfile
from pathlib import Path

from data.registry.registry_loader import (
    load_metric_registry,
    validate_registry,
    get_all_metric_names,
    get_dependencies,
    get_metrics_by_section,
    get_section_names,
    METRIC_SECTIONS,
)


class TestLoadMetricRegistry:
    """Tests for load_metric_registry function."""

    def test_load_default_registry_does_not_raise(self):
        """Loading the default registry should not raise any exceptions."""
        registry = load_metric_registry()
        assert registry is not None
        assert isinstance(registry, dict)

    def test_load_registry_has_required_keys(self):
        """Registry should have required keys."""
        registry = load_metric_registry()
        assert "version" in registry
        assert "market" in registry
        assert "economic" in registry

    def test_load_registry_has_expected_sections(self):
        """Registry should have all expected sections."""
        registry = load_metric_registry()
        expected_sections = ["market", "economic", "synthetic", "technical", "geometry", "model_outputs"]
        for section in expected_sections:
            assert section in registry, f"Missing section: {section}"

    def test_load_custom_path(self):
        """Should be able to load registry from custom path."""
        # Create a minimal test registry
        test_registry = {
            "version": "test",
            "market": [{"name": "test_ticker", "type": "market_price", "frequency": "daily"}],
            "economic": [{"name": "test_econ", "type": "econ_series", "frequency": "monthly"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_registry, f)
            temp_path = f.name

        try:
            registry = load_metric_registry(temp_path)
            assert registry["version"] == "test"
            assert len(registry["market"]) == 1
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file_raises(self):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_metric_registry("/nonexistent/path/registry.json")


class TestValidateRegistry:
    """Tests for validate_registry function."""

    def test_validate_default_registry_does_not_raise(self):
        """Validating the default registry should not raise."""
        registry = load_metric_registry()
        # Should not raise
        validate_registry(registry)

    def test_validate_missing_required_key_raises(self):
        """Missing required key should raise ValueError."""
        registry = {"version": "1.0", "economic": []}  # Missing "market"
        with pytest.raises(ValueError, match="missing required key"):
            validate_registry(registry)

    def test_validate_uppercase_name_raises(self):
        """Uppercase metric names should raise ValueError."""
        registry = {
            "version": "1.0",
            "market": [{"name": "SPY", "type": "market_price", "frequency": "daily"}],
            "economic": [],
        }
        with pytest.raises(ValueError, match="must be lowercase"):
            validate_registry(registry)

    def test_validate_duplicate_name_raises(self):
        """Duplicate metric names should raise ValueError."""
        registry = {
            "version": "1.0",
            "market": [
                {"name": "spy", "type": "market_price", "frequency": "daily"},
                {"name": "spy", "type": "market_price", "frequency": "daily"},
            ],
            "economic": [],
        }
        with pytest.raises(ValueError, match="Duplicate metric name"):
            validate_registry(registry)

    def test_validate_duplicate_across_sections_raises(self):
        """Duplicate names across sections should raise ValueError."""
        registry = {
            "version": "1.0",
            "market": [{"name": "duplicate", "type": "market_price", "frequency": "daily"}],
            "economic": [{"name": "duplicate", "type": "econ_series", "frequency": "monthly"}],
        }
        with pytest.raises(ValueError, match="Duplicate metric name"):
            validate_registry(registry)

    def test_validate_missing_dependency_raises(self):
        """depends_on reference to nonexistent metric should raise."""
        registry = {
            "version": "1.0",
            "market": [{"name": "spy", "type": "market_price", "frequency": "daily"}],
            "economic": [],
            "synthetic": [{"name": "test_spread", "depends_on": ["spy", "nonexistent"]}],
        }
        with pytest.raises(ValueError, match="not found in market or economic"):
            validate_registry(registry)

    def test_validate_valid_dependencies_pass(self):
        """Valid dependencies should pass validation."""
        registry = {
            "version": "1.0",
            "market": [{"name": "spy", "type": "market_price", "frequency": "daily"}],
            "economic": [{"name": "dgs10", "type": "econ_series", "frequency": "daily"}],
            "synthetic": [{"name": "test_spread", "depends_on": ["spy", "dgs10"]}],
        }
        # Should not raise
        validate_registry(registry)


class TestNoDuplicateNames:
    """Test that the actual registry has no duplicate names."""

    def test_no_duplicate_names_across_all_sections(self):
        """All metric names across all sections should be unique."""
        registry = load_metric_registry()
        all_names = get_all_metric_names(registry)

        # Count names per section
        total_count = 0
        for section in METRIC_SECTIONS:
            if section in registry:
                total_count += len(registry[section])

        # If there are duplicates, all_names (a set) would be smaller
        assert len(all_names) == total_count, "Duplicate names found in registry"


class TestAllDependenciesExist:
    """Test that all depends_on references exist in market/economic."""

    def test_synthetic_dependencies_exist(self):
        """All synthetic depends_on should reference market/economic metrics."""
        registry = load_metric_registry()

        # Get base metrics (market + economic)
        base_metrics = set()
        for section in ["market", "economic"]:
            for item in registry.get(section, []):
                name = item.get("name")
                if name:
                    base_metrics.add(name)

        # Check synthetic dependencies
        for item in registry.get("synthetic", []):
            name = item.get("name", "unknown")
            depends_on = item.get("depends_on", [])

            for dep in depends_on:
                assert dep in base_metrics, (
                    f"Synthetic '{name}' depends on '{dep}' "
                    f"which is not in market/economic"
                )

    def test_technical_dependencies_exist(self):
        """All technical depends_on should reference market/economic metrics."""
        registry = load_metric_registry()

        # Get base metrics (market + economic)
        base_metrics = set()
        for section in ["market", "economic"]:
            for item in registry.get(section, []):
                name = item.get("name")
                if name:
                    base_metrics.add(name)

        # Check technical dependencies
        for item in registry.get("technical", []):
            name = item.get("name", "unknown")
            depends_on = item.get("depends_on", [])

            for dep in depends_on:
                assert dep in base_metrics, (
                    f"Technical '{name}' depends on '{dep}' "
                    f"which is not in market/economic"
                )


class TestGetAllMetricNames:
    """Tests for get_all_metric_names function."""

    def test_returns_set(self):
        """Should return a set of strings."""
        registry = load_metric_registry()
        names = get_all_metric_names(registry)
        assert isinstance(names, set)
        assert all(isinstance(n, str) for n in names)

    def test_includes_all_sections(self):
        """Should include metrics from all sections."""
        registry = load_metric_registry()
        names = get_all_metric_names(registry)

        # Spot check some known metrics
        assert "spy" in names  # market
        assert "cpi" in names  # economic
        assert "t10y2y" in names  # synthetic
        assert "spy_mom_12m" in names  # technical
        assert "theta" in names  # geometry
        assert "mrf_core" in names  # model_outputs


class TestGetDependencies:
    """Tests for get_dependencies function."""

    def test_returns_list_for_dependent_metric(self):
        """Should return list of dependencies for metrics with depends_on."""
        registry = load_metric_registry()

        deps = get_dependencies(registry, "t10y2y")
        assert deps is not None
        assert isinstance(deps, list)
        assert "dgs10" in deps
        assert "dgs2" in deps

    def test_returns_none_for_base_metric(self):
        """Should return None for metrics without depends_on."""
        registry = load_metric_registry()

        deps = get_dependencies(registry, "spy")
        assert deps is None

    def test_returns_none_for_nonexistent_metric(self):
        """Should return None for nonexistent metric."""
        registry = load_metric_registry()

        deps = get_dependencies(registry, "nonexistent_metric")
        assert deps is None


class TestGetMetricsBySection:
    """Tests for get_metrics_by_section function."""

    def test_returns_list(self):
        """Should return a list."""
        registry = load_metric_registry()
        metrics = get_metrics_by_section(registry, "market")
        assert isinstance(metrics, list)

    def test_returns_empty_for_missing_section(self):
        """Should return empty list for missing section."""
        registry = load_metric_registry()
        metrics = get_metrics_by_section(registry, "nonexistent_section")
        assert metrics == []


class TestGetSectionNames:
    """Tests for get_section_names function."""

    def test_returns_list_of_strings(self):
        """Should return list of metric names."""
        registry = load_metric_registry()
        names = get_section_names(registry, "market")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert "spy" in names

    def test_returns_empty_for_missing_section(self):
        """Should return empty list for missing section."""
        registry = load_metric_registry()
        names = get_section_names(registry, "nonexistent_section")
        assert names == []


class TestRegistryContent:
    """Tests for specific registry content requirements."""

    def test_all_names_lowercase(self):
        """All metric names should be lowercase."""
        registry = load_metric_registry()
        names = get_all_metric_names(registry)

        for name in names:
            assert name == name.lower(), f"Name '{name}' is not lowercase"

    def test_market_section_has_spy(self):
        """Market section should include SPY."""
        registry = load_metric_registry()
        market_names = get_section_names(registry, "market")
        assert "spy" in market_names

    def test_economic_section_has_rates(self):
        """Economic section should include treasury rates."""
        registry = load_metric_registry()
        economic_names = get_section_names(registry, "economic")
        assert "dgs10" in economic_names
        assert "dgs2" in economic_names

    def test_synthetic_has_yield_spreads(self):
        """Synthetic section should include yield curve spreads."""
        registry = load_metric_registry()
        synthetic_names = get_section_names(registry, "synthetic")
        assert "t10y2y" in synthetic_names
        assert "t10y3m" in synthetic_names

    def test_technical_has_momentum(self):
        """Technical section should include momentum indicators."""
        registry = load_metric_registry()
        technical_names = get_section_names(registry, "technical")
        assert "spy_mom_12m" in technical_names
        assert "spy_mom_6m" in technical_names

    def test_geometry_has_required_fields(self):
        """Geometry section should have theta, phi, magnitude, etc."""
        registry = load_metric_registry()
        geometry_names = get_section_names(registry, "geometry")
        required = ["theta", "phi", "magnitude", "coherence"]
        for name in required:
            assert name in geometry_names, f"Missing geometry field: {name}"
