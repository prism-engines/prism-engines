"""
Tests for PRISM Database Connector module.

Tests market and economic data operations including:
- Migration runner
- Market price/dividend/TRI insertion and retrieval
- Economic value insertion and retrieval
- Date validation
- Bulk upsert operations
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from utils.db_connector import (
    get_db_path,
    get_connection,
    validate_date,
    get_migration_files,
    get_applied_migrations,
    run_pending_migrations,
    init_database,
    insert_market_price,
    insert_market_dividend,
    insert_market_tri,
    upsert_market_meta,
    upsert_market_prices,
    insert_econ_series,
    insert_econ_value,
    upsert_econ_meta,
    upsert_econ_values,
    load_market_prices,
    load_econ_values,
    log_fetch,
    get_market_stats,
    get_econ_stats,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Initialize the database with all migrations
    init_database(db_path)
    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_price_df():
    """Sample DataFrame with price data."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "price": [100.0, 101.5, 102.0],
        "ret": [None, 0.015, 0.005],
    })


@pytest.fixture
def sample_econ_df():
    """Sample DataFrame with economic data."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "value": [25000.0, 25100.0, 25200.0],
    })


class TestDateValidation:
    """Tests for date validation."""

    def test_valid_date(self):
        """Valid ISO dates pass validation."""
        assert validate_date("2024-01-15") == "2024-01-15"
        assert validate_date("2000-12-31") == "2000-12-31"
        assert validate_date("1990-01-01") == "1990-01-01"

    def test_invalid_date_format(self):
        """Invalid date formats raise ValueError."""
        with pytest.raises(ValueError, match="ISO format"):
            validate_date("01-15-2024")

        with pytest.raises(ValueError, match="ISO format"):
            validate_date("2024/01/15")

        with pytest.raises(ValueError, match="ISO format"):
            validate_date("20240115")

    def test_invalid_date_value(self):
        """Invalid date values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid date"):
            validate_date("2024-13-01")  # Month 13

        with pytest.raises(ValueError, match="Invalid date"):
            validate_date("2024-02-30")  # Feb 30

    def test_date_must_be_string(self):
        """Non-string inputs raise ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_date(20240115)


class TestMigrations:
    """Tests for migration management."""

    def test_get_migration_files(self):
        """Can get list of migration files."""
        migrations = get_migration_files()
        assert len(migrations) > 0

        # Check migrations are sorted
        versions = [int(v) for v, _ in migrations]
        assert versions == sorted(versions)

    def test_run_pending_migrations(self, temp_db):
        """Can run pending migrations."""
        # Migrations were already run during fixture setup
        applied = get_applied_migrations(temp_db)
        assert len(applied) > 0

    def test_init_database_idempotent(self, temp_db):
        """init_database can be called multiple times safely."""
        # Should not raise
        init_database(temp_db)
        init_database(temp_db)

        # Tables should still exist
        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='market_prices'"
            )
            assert cursor.fetchone() is not None


class TestMarketPrices:
    """Tests for market price operations."""

    def test_insert_market_price(self, temp_db):
        """Can insert a single market price."""
        row_id = insert_market_price(
            ticker="SPY",
            date="2024-01-15",
            price=450.25,
            db_path=temp_db,
        )
        assert row_id is not None
        assert row_id > 0

    def test_insert_market_price_with_derived(self, temp_db):
        """Can insert price with derived columns."""
        insert_market_price(
            ticker="SPY",
            date="2024-01-15",
            price=450.25,
            ret=0.005,
            price_z=1.2,
            price_log=6.11,
            db_path=temp_db,
        )

        df = load_market_prices("SPY", db_path=temp_db)
        assert len(df) == 1
        assert df.iloc[0]["price"] == 450.25
        assert df.iloc[0]["ret"] == 0.005
        assert df.iloc[0]["price_z"] == 1.2

    def test_insert_market_price_upsert(self, temp_db):
        """Inserting same ticker+date updates existing record."""
        insert_market_price("SPY", "2024-01-15", 450.0, db_path=temp_db)
        insert_market_price("SPY", "2024-01-15", 451.0, db_path=temp_db)

        df = load_market_prices("SPY", db_path=temp_db)
        assert len(df) == 1
        assert df.iloc[0]["price"] == 451.0

    def test_insert_market_price_invalid_date(self, temp_db):
        """Invalid date raises ValueError."""
        with pytest.raises(ValueError, match="ISO format"):
            insert_market_price("SPY", "01-15-2024", 450.0, db_path=temp_db)

    def test_load_market_prices_date_range(self, temp_db):
        """Can load prices with date range filter."""
        for i, date in enumerate(["2024-01-01", "2024-01-02", "2024-01-03"]):
            insert_market_price("SPY", date, 100 + i, db_path=temp_db)

        df = load_market_prices(
            "SPY",
            start_date="2024-01-02",
            end_date="2024-01-02",
            db_path=temp_db,
        )
        assert len(df) == 1
        assert df.iloc[0]["price"] == 101.0


class TestMarketDividends:
    """Tests for market dividend operations."""

    def test_insert_market_dividend(self, temp_db):
        """Can insert a single dividend record."""
        row_id = insert_market_dividend(
            ticker="SPY",
            date="2024-03-15",
            dividend=1.50,
            db_path=temp_db,
        )
        assert row_id is not None
        assert row_id > 0

    def test_insert_market_dividend_upsert(self, temp_db):
        """Inserting same ticker+date updates existing record."""
        insert_market_dividend("SPY", "2024-03-15", 1.50, db_path=temp_db)
        insert_market_dividend("SPY", "2024-03-15", 1.60, db_path=temp_db)

        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT dividend FROM market_dividends WHERE ticker = ? AND date = ?",
                ("SPY", "2024-03-15"),
            )
            row = cursor.fetchone()
            assert row["dividend"] == 1.60


class TestMarketTRI:
    """Tests for Total Return Index operations."""

    def test_insert_market_tri(self, temp_db):
        """Can insert a TRI record."""
        row_id = insert_market_tri(
            ticker="SPY",
            date="2024-01-15",
            tri_value=1000.0,
            db_path=temp_db,
        )
        assert row_id is not None
        assert row_id > 0

    def test_insert_market_tri_with_derived(self, temp_db):
        """Can insert TRI with derived columns."""
        insert_market_tri(
            ticker="SPY",
            date="2024-01-15",
            tri_value=1000.0,
            tri_z=0.5,
            tri_log=6.91,
            db_path=temp_db,
        )

        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT tri_value, tri_z, tri_log FROM market_tri WHERE ticker = ?",
                ("SPY",),
            )
            row = cursor.fetchone()
            assert row["tri_value"] == 1000.0
            assert row["tri_z"] == 0.5
            assert row["tri_log"] == 6.91


class TestMarketMeta:
    """Tests for market metadata operations."""

    def test_upsert_market_meta(self, temp_db):
        """Can insert market metadata."""
        upsert_market_meta(
            ticker="SPY",
            first_date="2000-01-01",
            last_date="2024-01-15",
            source="Yahoo Finance",
            notes="S&P 500 ETF",
            db_path=temp_db,
        )

        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT * FROM market_meta WHERE ticker = ?",
                ("SPY",),
            )
            row = cursor.fetchone()
            assert row["first_date"] == "2000-01-01"
            assert row["source"] == "Yahoo Finance"

    def test_upsert_market_meta_partial_update(self, temp_db):
        """Partial update preserves existing values."""
        upsert_market_meta(
            ticker="SPY",
            first_date="2000-01-01",
            source="Yahoo Finance",
            db_path=temp_db,
        )

        upsert_market_meta(
            ticker="SPY",
            last_date="2024-01-15",
            db_path=temp_db,
        )

        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT * FROM market_meta WHERE ticker = ?",
                ("SPY",),
            )
            row = cursor.fetchone()
            assert row["first_date"] == "2000-01-01"
            assert row["last_date"] == "2024-01-15"
            assert row["source"] == "Yahoo Finance"


class TestBulkUpsertPrices:
    """Tests for bulk market price operations."""

    def test_upsert_market_prices(self, temp_db, sample_price_df):
        """Can bulk upsert prices from DataFrame."""
        rows = upsert_market_prices(
            sample_price_df,
            ticker="SPY",
            db_path=temp_db,
        )
        assert rows == 3

        df = load_market_prices("SPY", db_path=temp_db)
        assert len(df) == 3

    def test_upsert_market_prices_empty_df(self, temp_db):
        """Empty DataFrame returns 0 rows."""
        empty_df = pd.DataFrame(columns=["date", "price"])
        rows = upsert_market_prices(empty_df, ticker="SPY", db_path=temp_db)
        assert rows == 0


class TestEconSeries:
    """Tests for economic series operations."""

    def test_insert_econ_series(self, temp_db):
        """Can insert an economic series."""
        insert_econ_series(
            code="GDP",
            human_name="Gross Domestic Product",
            frequency="quarterly",
            source="FRED",
            notes="Real GDP",
            db_path=temp_db,
        )

        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT * FROM econ_series WHERE code = ?",
                ("GDP",),
            )
            row = cursor.fetchone()
            assert row["human_name"] == "Gross Domestic Product"
            assert row["frequency"] == "quarterly"

    def test_insert_econ_series_update(self, temp_db):
        """Inserting existing code updates record."""
        insert_econ_series(code="GDP", human_name="GDP v1", db_path=temp_db)
        insert_econ_series(code="GDP", human_name="GDP v2", db_path=temp_db)

        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT human_name FROM econ_series WHERE code = ?",
                ("GDP",),
            )
            row = cursor.fetchone()
            assert row["human_name"] == "GDP v2"


class TestEconValues:
    """Tests for economic value operations."""

    def test_insert_econ_value(self, temp_db):
        """Can insert an economic value."""
        insert_econ_series(code="GDP", db_path=temp_db)

        row_id = insert_econ_value(
            code="GDP",
            date="2024-01-01",
            revision_asof="2024-03-15",
            value_raw=25000.0,
            db_path=temp_db,
        )
        assert row_id is not None
        assert row_id > 0

    def test_insert_econ_value_with_derived(self, temp_db):
        """Can insert value with derived columns."""
        insert_econ_series(code="GDP", db_path=temp_db)

        insert_econ_value(
            code="GDP",
            date="2024-01-01",
            revision_asof="2024-03-15",
            value_raw=25000.0,
            value_yoy=0.025,
            value_mom=0.008,
            value_z=1.5,
            value_log=10.13,
            db_path=temp_db,
        )

        df = load_econ_values("GDP", db_path=temp_db)
        assert len(df) == 1
        assert df.iloc[0]["value_raw"] == 25000.0
        assert df.iloc[0]["value_yoy"] == 0.025

    def test_insert_econ_value_invalid_date(self, temp_db):
        """Invalid date raises ValueError."""
        insert_econ_series(code="GDP", db_path=temp_db)

        with pytest.raises(ValueError, match="ISO format"):
            insert_econ_value(
                code="GDP",
                date="01-01-2024",
                revision_asof="2024-03-15",
                value_raw=25000.0,
                db_path=temp_db,
            )

    def test_load_econ_values_latest_revision(self, temp_db):
        """Loading with latest_revision=True gets most recent revision."""
        insert_econ_series(code="GDP", db_path=temp_db)

        # Insert two revisions for same date
        insert_econ_value("GDP", "2024-01-01", "2024-02-15", 24900.0, db_path=temp_db)
        insert_econ_value("GDP", "2024-01-01", "2024-03-15", 25000.0, db_path=temp_db)

        df = load_econ_values("GDP", latest_revision=True, db_path=temp_db)
        assert len(df) == 1
        assert df.iloc[0]["value_raw"] == 25000.0

    def test_load_econ_values_specific_revision(self, temp_db):
        """Can load specific revision."""
        insert_econ_series(code="GDP", db_path=temp_db)

        insert_econ_value("GDP", "2024-01-01", "2024-02-15", 24900.0, db_path=temp_db)
        insert_econ_value("GDP", "2024-01-01", "2024-03-15", 25000.0, db_path=temp_db)

        df = load_econ_values("GDP", revision_asof="2024-02-15", db_path=temp_db)
        assert len(df) == 1
        assert df.iloc[0]["value_raw"] == 24900.0


class TestEconMeta:
    """Tests for economic metadata operations."""

    def test_upsert_econ_meta(self, temp_db):
        """Can insert economic metadata."""
        insert_econ_series(code="GDP", db_path=temp_db)

        upsert_econ_meta(
            code="GDP",
            last_fetched="2024-03-15",
            last_revision_asof="2024-03-01",
            db_path=temp_db,
        )

        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT * FROM econ_meta WHERE code = ?",
                ("GDP",),
            )
            row = cursor.fetchone()
            assert row["last_fetched"] == "2024-03-15"


class TestBulkUpsertEcon:
    """Tests for bulk economic value operations."""

    def test_upsert_econ_values(self, temp_db, sample_econ_df):
        """Can bulk upsert economic values from DataFrame."""
        insert_econ_series(code="GDP", db_path=temp_db)

        rows = upsert_econ_values(
            sample_econ_df,
            code="GDP",
            revision_asof="2024-04-01",
            db_path=temp_db,
        )
        assert rows == 3

        df = load_econ_values("GDP", db_path=temp_db)
        assert len(df) == 3


class TestFetchLogging:
    """Tests for fetch logging operations."""

    def test_log_fetch(self, temp_db):
        """Can log a fetch operation."""
        row_id = log_fetch(
            source="yahoo",
            entity="SPY",
            operation="fetch",
            status="success",
            rows_fetched=100,
            rows_inserted=100,
            db_path=temp_db,
        )
        assert row_id is not None
        assert row_id > 0

    def test_log_fetch_with_error(self, temp_db):
        """Can log a failed fetch operation."""
        log_fetch(
            source="fred",
            entity="GDP",
            operation="fetch",
            status="error",
            error_message="API rate limit exceeded",
            db_path=temp_db,
        )

        with get_connection(temp_db) as conn:
            cursor = conn.execute(
                "SELECT * FROM fetch_log WHERE entity = ?",
                ("GDP",),
            )
            row = cursor.fetchone()
            assert row["status"] == "error"
            assert "rate limit" in row["error_message"]


class TestDatabaseStats:
    """Tests for database statistics."""

    def test_get_market_stats(self, temp_db):
        """Can get market data statistics."""
        insert_market_price("SPY", "2024-01-01", 450.0, db_path=temp_db)
        insert_market_price("SPY", "2024-01-02", 451.0, db_path=temp_db)
        insert_market_price("AAPL", "2024-01-01", 180.0, db_path=temp_db)

        stats = get_market_stats(temp_db)
        assert stats["prices"]["tickers"] == 2
        assert stats["prices"]["total_rows"] == 3

    def test_get_econ_stats(self, temp_db):
        """Can get economic data statistics."""
        insert_econ_series(code="GDP", db_path=temp_db)
        insert_econ_series(code="UNRATE", db_path=temp_db)

        insert_econ_value("GDP", "2024-01-01", "2024-03-15", 25000.0, db_path=temp_db)

        stats = get_econ_stats(temp_db)
        assert stats["series_count"] == 2
        assert stats["values"]["total_rows"] == 1
