"""
Tests for PRISM SQL database module.
"""

import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from data.sql.prism_db import (
    system_types,
    add_indicator,
    delete_indicator,
    get_db_stats,
    get_indicator,
    init_db,
    list_indicators,
    load_indicator,
    load_multiple_indicators,
    load_system_indicators,
    update_indicator,
    write_dataframe,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    init_db(db_path)
    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "value": [100.0, 101.5, 102.0],
    })


class TestSystemTypes:
    """Tests for system_types constant."""

    def test_system_types_is_list(self):
        """system_types should be a list."""
        assert isinstance(system_types, list)

    def test_system_types_contains_expected_values(self):
        """system_types should contain expected domain types."""
        expected = ["finance", "climate", "chemistry"]
        for system in expected:
            assert system in system_types


class TestAddIndicator:
    """Tests for add_indicator function."""

    def test_add_indicator_with_system(self, temp_db):
        """Can add indicator with system parameter."""
        indicator_id = add_indicator(
            name="TEST_IND",
            system="finance",
            frequency="daily",
            db_path=temp_db,
        )
        assert indicator_id is not None
        assert indicator_id > 0

    def test_add_indicator_with_deprecated_panel(self, temp_db):
        """Adding indicator with panel parameter works but warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            indicator_id = add_indicator(
                name="TEST_IND_DEPRECATED",
                panel="finance",
                frequency="daily",
                db_path=temp_db,
            )
            assert indicator_id is not None
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "panel" in str(w[0].message)
            assert "system" in str(w[0].message)

    def test_add_indicator_requires_system(self, temp_db):
        """Adding indicator without system raises ValueError."""
        with pytest.raises(ValueError, match="system.*required"):
            add_indicator(
                name="TEST_IND_NO_SYSTEM",
                frequency="daily",
                db_path=temp_db,
            )

    def test_add_indicator_invalid_system(self, temp_db):
        """Adding indicator with invalid system raises ValueError."""
        with pytest.raises(ValueError, match="Unknown system"):
            add_indicator(
                name="TEST_IND_INVALID",
                system="invalid_system",
                frequency="daily",
                db_path=temp_db,
            )

    def test_add_indicator_all_fields(self, temp_db):
        """Can add indicator with all optional fields."""
        indicator_id = add_indicator(
            name="FULL_IND",
            system="climate",
            frequency="monthly",
            source="NOAA",
            units="Celsius",
            description="Temperature anomaly",
            db_path=temp_db,
        )

        indicator = get_indicator("FULL_IND", db_path=temp_db)
        assert indicator["name"] == "FULL_IND"
        assert indicator["system"] == "climate"
        assert indicator["frequency"] == "monthly"
        assert indicator["source"] == "NOAA"
        assert indicator["units"] == "Celsius"
        assert indicator["description"] == "Temperature anomaly"


class TestGetIndicator:
    """Tests for get_indicator function."""

    def test_get_existing_indicator(self, temp_db):
        """Can retrieve existing indicator."""
        add_indicator(name="GET_TEST", system="finance", db_path=temp_db)
        indicator = get_indicator("GET_TEST", db_path=temp_db)

        assert indicator is not None
        assert indicator["name"] == "GET_TEST"
        assert indicator["system"] == "finance"

    def test_get_nonexistent_indicator(self, temp_db):
        """Returns None for nonexistent indicator."""
        indicator = get_indicator("DOES_NOT_EXIST", db_path=temp_db)
        assert indicator is None


class TestListIndicators:
    """Tests for list_indicators function."""

    def test_list_all_indicators(self, temp_db):
        """Can list all indicators."""
        add_indicator(name="IND1", system="finance", db_path=temp_db)
        add_indicator(name="IND2", system="climate", db_path=temp_db)

        df = list_indicators(db_path=temp_db)
        assert len(df) == 2
        assert "IND1" in df["name"].values
        assert "IND2" in df["name"].values

    def test_list_indicators_by_system(self, temp_db):
        """Can filter indicators by system."""
        add_indicator(name="FIN1", system="finance", db_path=temp_db)
        add_indicator(name="FIN2", system="finance", db_path=temp_db)
        add_indicator(name="CLI1", system="climate", db_path=temp_db)

        df = list_indicators(system="finance", db_path=temp_db)
        assert len(df) == 2
        assert all(df["system"] == "finance")

    def test_list_indicators_deprecated_panel(self, temp_db):
        """Using panel parameter works but warns."""
        add_indicator(name="TEST", system="finance", db_path=temp_db)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = list_indicators(panel="finance", db_path=temp_db)
            assert len(df) == 1
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


class TestUpdateIndicator:
    """Tests for update_indicator function."""

    def test_update_indicator_fields(self, temp_db):
        """Can update indicator fields."""
        add_indicator(name="UPDATE_TEST", system="finance", db_path=temp_db)

        success = update_indicator(
            "UPDATE_TEST",
            description="Updated description",
            db_path=temp_db,
        )
        assert success is True

        indicator = get_indicator("UPDATE_TEST", db_path=temp_db)
        assert indicator["description"] == "Updated description"

    def test_update_nonexistent_indicator(self, temp_db):
        """Returns False for nonexistent indicator."""
        success = update_indicator(
            "NONEXISTENT",
            description="Test",
            db_path=temp_db,
        )
        assert success is False


class TestDeleteIndicator:
    """Tests for delete_indicator function."""

    def test_delete_existing_indicator(self, temp_db):
        """Can delete existing indicator."""
        add_indicator(name="DELETE_TEST", system="finance", db_path=temp_db)
        success = delete_indicator("DELETE_TEST", db_path=temp_db)

        assert success is True
        assert get_indicator("DELETE_TEST", db_path=temp_db) is None

    def test_delete_nonexistent_indicator(self, temp_db):
        """Returns False for nonexistent indicator."""
        success = delete_indicator("NONEXISTENT", db_path=temp_db)
        assert success is False


class TestWriteDataframe:
    """Tests for write_dataframe function."""

    def test_write_dataframe_creates_indicator(self, temp_db, sample_dataframe):
        """write_dataframe creates indicator if missing."""
        rows = write_dataframe(
            sample_dataframe,
            indicator_name="NEW_IND",
            system="finance",
            db_path=temp_db,
        )

        assert rows == 3
        indicator = get_indicator("NEW_IND", db_path=temp_db)
        assert indicator is not None

    def test_write_dataframe_to_existing_indicator(self, temp_db, sample_dataframe):
        """write_dataframe works with existing indicator."""
        add_indicator(name="EXISTING", system="finance", db_path=temp_db)

        rows = write_dataframe(
            sample_dataframe,
            indicator_name="EXISTING",
            db_path=temp_db,
        )
        assert rows == 3

    def test_write_dataframe_deprecated_panel(self, temp_db, sample_dataframe):
        """Using panel parameter works but warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_dataframe(
                sample_dataframe,
                indicator_name="PANEL_TEST",
                panel="finance",
                db_path=temp_db,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


class TestLoadIndicator:
    """Tests for load_indicator function."""

    def test_load_indicator_roundtrip(self, temp_db, sample_dataframe):
        """Data round-trips correctly through database."""
        write_dataframe(
            sample_dataframe,
            indicator_name="ROUNDTRIP",
            system="finance",
            db_path=temp_db,
        )

        loaded = load_indicator("ROUNDTRIP", db_path=temp_db)

        assert len(loaded) == 3
        assert loaded.iloc[0]["value"] == 100.0
        assert loaded.iloc[1]["value"] == 101.5
        assert loaded.iloc[2]["value"] == 102.0

    def test_load_indicator_date_range(self, temp_db, sample_dataframe):
        """Can load data with date range filter."""
        write_dataframe(
            sample_dataframe,
            indicator_name="DATE_RANGE",
            system="finance",
            db_path=temp_db,
        )

        loaded = load_indicator(
            "DATE_RANGE",
            start_date="2024-01-02",
            end_date="2024-01-02",
            db_path=temp_db,
        )

        assert len(loaded) == 1
        assert loaded.iloc[0]["value"] == 101.5

    def test_load_nonexistent_indicator(self, temp_db):
        """Raises ValueError for nonexistent indicator."""
        with pytest.raises(ValueError, match="not found"):
            load_indicator("NONEXISTENT", db_path=temp_db)


class TestLoadMultipleIndicators:
    """Tests for load_multiple_indicators function."""

    def test_load_multiple_indicators(self, temp_db, sample_dataframe):
        """Can load multiple indicators into single DataFrame."""
        for name in ["IND_A", "IND_B", "IND_C"]:
            write_dataframe(
                sample_dataframe,
                indicator_name=name,
                system="finance",
                db_path=temp_db,
            )

        loaded = load_multiple_indicators(
            ["IND_A", "IND_B", "IND_C"],
            db_path=temp_db,
        )

        assert "IND_A" in loaded.columns
        assert "IND_B" in loaded.columns
        assert "IND_C" in loaded.columns
        assert len(loaded) == 3


class TestLoadSystemIndicators:
    """Tests for load_system_indicators function."""

    def test_load_system_indicators(self, temp_db, sample_dataframe):
        """Can load all indicators for a system."""
        for name in ["FIN_1", "FIN_2"]:
            write_dataframe(
                sample_dataframe,
                indicator_name=name,
                system="finance",
                db_path=temp_db,
            )

        # Add climate indicator that should not be loaded
        write_dataframe(
            sample_dataframe,
            indicator_name="CLI_1",
            system="climate",
            db_path=temp_db,
        )

        loaded = load_system_indicators("finance", db_path=temp_db)

        assert "FIN_1" in loaded.columns
        assert "FIN_2" in loaded.columns
        assert "CLI_1" not in loaded.columns

    def test_load_system_indicators_deprecated_panel(self, temp_db, sample_dataframe):
        """Using panel parameter works but warns."""
        write_dataframe(
            sample_dataframe,
            indicator_name="TEST",
            system="finance",
            db_path=temp_db,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_system_indicators(panel="finance", system=None, db_path=temp_db)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


class TestDbStats:
    """Tests for get_db_stats function."""

    def test_get_db_stats(self, temp_db, sample_dataframe):
        """Can get database statistics."""
        write_dataframe(
            sample_dataframe,
            indicator_name="STATS_TEST",
            system="finance",
            db_path=temp_db,
        )

        stats = get_db_stats(db_path=temp_db)

        assert stats["indicator_count"] == 1
        assert stats["value_count"] == 3
        assert stats["systems"]["finance"] == 1
