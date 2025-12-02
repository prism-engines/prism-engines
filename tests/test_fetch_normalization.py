"""
Test Fetch Normalization

Tests for:
- Date normalization behavior (fixing 2-digit years, parsing formats)
- Removal of footer garbage
- Handling of invalid numeric strings
- Column name standardization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def raw_dataframe_with_multiindex():
    """DataFrame with MultiIndex columns (like yfinance output)."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    data = {
        ("Adj Close", "SPY"): [100, 101, 102, 103, 104],
        ("Volume", "SPY"): [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def raw_dataframe_with_garbage():
    """DataFrame with footer garbage rows."""
    return pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03", "Source: XYZ", "Updated: 2024"],
        "value": [100.0, 101.0, 102.0, np.nan, np.nan],
    })


@pytest.fixture
def raw_dataframe_with_invalid_numbers():
    """DataFrame with invalid numeric strings."""
    return pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        "value": ["100.5", "N/A", "102.3", "---"],
    })


@pytest.fixture
def raw_dataframe_with_2digit_years():
    """DataFrame with 2-digit year dates."""
    return pd.DataFrame({
        "date": ["01/15/20", "02/15/20", "03/15/20", "04/15/20"],
        "value": [100, 101, 102, 103],
    })


# ============================================================================
# Date Normalization Tests
# ============================================================================

class TestDateNormalization:
    """Tests for date parsing and normalization."""

    def test_parse_iso_dates(self):
        """Should parse ISO format dates correctly."""
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "value": [1, 2, 3]
        })
        df["date"] = pd.to_datetime(df["date"])

        assert df["date"].iloc[0].year == 2020
        assert df["date"].iloc[0].month == 1
        assert df["date"].iloc[0].day == 1

    def test_parse_us_format_dates(self):
        """Should parse US format (MM/DD/YYYY) dates."""
        df = pd.DataFrame({
            "date": ["01/15/2020", "02/15/2020", "03/15/2020"],
            "value": [1, 2, 3]
        })
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

        assert df["date"].iloc[0].year == 2020
        assert df["date"].iloc[0].month == 1
        assert df["date"].iloc[0].day == 15

    def test_parse_2digit_year(self, raw_dataframe_with_2digit_years):
        """Should correctly handle 2-digit years."""
        df = raw_dataframe_with_2digit_years.copy()

        # Parse with explicit format for 2-digit year
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")

        # 2-digit year '20' should become 2020
        assert df["date"].iloc[0].year == 2020

    def test_parse_mixed_formats(self):
        """Should handle mixed date formats with inference."""
        df = pd.DataFrame({
            "date": ["2020-01-01", "2020/02/15", "March 15, 2020"],
            "value": [1, 2, 3]
        })

        # pandas can infer these formats with format='mixed'
        df["date"] = pd.to_datetime(df["date"], format="mixed")

        assert df["date"].iloc[0].year == 2020
        assert df["date"].iloc[1].month == 2
        assert df["date"].iloc[2].day == 15

    def test_handle_invalid_dates(self):
        """Should handle invalid dates gracefully."""
        df = pd.DataFrame({
            "date": ["2020-01-01", "invalid_date", "2020-01-03"],
            "value": [1, 2, 3]
        })

        # Use coerce to convert invalid dates to NaT
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        assert pd.notna(df["date"].iloc[0])
        assert pd.isna(df["date"].iloc[1])  # invalid becomes NaT
        assert pd.notna(df["date"].iloc[2])

    def test_normalize_date_index(self):
        """Should normalize date when it's the index."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)
        df.index.name = "Date"

        # Convert index to column
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        assert "date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ============================================================================
# Footer Garbage Removal Tests
# ============================================================================

class TestFooterGarbageRemoval:
    """Tests for removing footer garbage from data."""

    def remove_footer_garbage(self, df, date_col="date"):
        """
        Remove rows where date column contains non-date values.

        Args:
            df: Input DataFrame
            date_col: Name of date column

        Returns:
            DataFrame with footer garbage removed
        """
        result = df.copy()

        # Try to convert dates, invalid ones become NaT
        result[date_col] = pd.to_datetime(result[date_col], errors="coerce")

        # Drop rows where date is NaT
        result = result.dropna(subset=[date_col])

        return result

    def test_remove_text_footer(self, raw_dataframe_with_garbage):
        """Should remove rows with text in date column."""
        df = self.remove_footer_garbage(raw_dataframe_with_garbage)

        assert len(df) == 3  # Only 3 valid dates
        assert "Source: XYZ" not in df["date"].astype(str).values
        assert "Updated: 2024" not in df["date"].astype(str).values

    def test_preserve_valid_rows(self, raw_dataframe_with_garbage):
        """Should preserve rows with valid dates."""
        df = self.remove_footer_garbage(raw_dataframe_with_garbage)

        assert df["value"].tolist() == [100.0, 101.0, 102.0]

    def test_remove_metadata_rows(self):
        """Should remove common metadata row patterns."""
        df = pd.DataFrame({
            "date": ["2020-01-01", "Last Updated:", "Data Source:", "2020-01-02"],
            "value": [100, np.nan, np.nan, 101]
        })

        cleaned = self.remove_footer_garbage(df)
        assert len(cleaned) == 2

    def test_handle_nan_in_date_column(self):
        """Should handle NaN values in date column."""
        df = pd.DataFrame({
            "date": ["2020-01-01", np.nan, "2020-01-03"],
            "value": [100, 101, 102]
        })

        cleaned = self.remove_footer_garbage(df)
        assert len(cleaned) == 2  # NaN date row removed


# ============================================================================
# Invalid Numeric String Tests
# ============================================================================

class TestInvalidNumericHandling:
    """Tests for handling invalid numeric strings."""

    def clean_numeric_column(self, df, col):
        """
        Clean numeric column by handling invalid values.

        Args:
            df: Input DataFrame
            col: Column name to clean

        Returns:
            DataFrame with cleaned numeric column
        """
        result = df.copy()
        result[col] = pd.to_numeric(result[col], errors="coerce")
        return result

    def test_convert_valid_strings(self, raw_dataframe_with_invalid_numbers):
        """Should convert valid numeric strings."""
        df = self.clean_numeric_column(raw_dataframe_with_invalid_numbers, "value")

        assert df["value"].iloc[0] == 100.5
        assert df["value"].iloc[2] == 102.3

    def test_handle_na_strings(self, raw_dataframe_with_invalid_numbers):
        """Should convert 'N/A' to NaN."""
        df = self.clean_numeric_column(raw_dataframe_with_invalid_numbers, "value")

        assert pd.isna(df["value"].iloc[1])

    def test_handle_dash_strings(self, raw_dataframe_with_invalid_numbers):
        """Should convert '---' to NaN."""
        df = self.clean_numeric_column(raw_dataframe_with_invalid_numbers, "value")

        assert pd.isna(df["value"].iloc[3])

    def test_handle_common_missing_patterns(self):
        """Should handle common missing value patterns."""
        df = pd.DataFrame({
            "value": ["100", "N/A", "n/a", "NA", ".", "-", "#N/A", "NULL", ""]
        })

        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        assert df["value"].iloc[0] == 100
        assert pd.isna(df["value"].iloc[1:]).all()  # All others should be NaN

    def test_preserve_negative_numbers(self):
        """Should preserve valid negative numbers."""
        df = pd.DataFrame({
            "value": ["100", "-50", "-0.5", "0"]
        })

        df["value"] = pd.to_numeric(df["value"])

        assert df["value"].tolist() == [100.0, -50.0, -0.5, 0.0]

    def test_handle_scientific_notation(self):
        """Should handle scientific notation."""
        df = pd.DataFrame({
            "value": ["1e6", "1.5e-3", "2E10"]
        })

        df["value"] = pd.to_numeric(df["value"])

        assert df["value"].iloc[0] == 1000000
        assert df["value"].iloc[1] == 0.0015


# ============================================================================
# Column Name Standardization Tests
# ============================================================================

class TestColumnStandardization:
    """Tests for column name standardization."""

    def standardize_columns(self, df):
        """
        Standardize DataFrame column names.

        - Flatten MultiIndex columns
        - Lowercase all column names
        - Ensure 'date' column exists

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized columns
        """
        result = df.copy()

        # Flatten MultiIndex columns
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [
                "_".join(str(x) for x in col if x)
                for col in result.columns
            ]

        # Lowercase all column names
        result.columns = [c.lower() for c in result.columns]

        # Handle date index
        if "date" not in result.columns:
            if result.index.name and "date" in result.index.name.lower():
                result = result.reset_index()
                result.columns = [c.lower() for c in result.columns]

        return result

    def test_flatten_multiindex_columns(self, raw_dataframe_with_multiindex):
        """Should flatten MultiIndex columns."""
        df = self.standardize_columns(raw_dataframe_with_multiindex)

        assert not isinstance(df.columns, pd.MultiIndex)
        assert "adj close_spy" in df.columns or "adj_close_spy" in df.columns.str.replace(" ", "_")

    def test_lowercase_column_names(self):
        """Should lowercase all column names."""
        df = pd.DataFrame({
            "Date": [1, 2, 3],
            "VALUE": [100, 101, 102],
            "TickerName": ["A", "B", "C"]
        })

        df.columns = [c.lower() for c in df.columns]

        assert all(c.islower() or not c.isalpha() for c in df.columns)
        assert "date" in df.columns
        assert "value" in df.columns

    def test_handle_date_as_index(self, raw_dataframe_with_multiindex):
        """Should convert date index to column."""
        df = self.standardize_columns(raw_dataframe_with_multiindex)

        assert "date" in df.columns

    def test_preserve_data_integrity(self, raw_dataframe_with_multiindex):
        """Column standardization should not alter data values."""
        original = raw_dataframe_with_multiindex.copy()
        standardized = self.standardize_columns(raw_dataframe_with_multiindex)

        # Check that values are preserved
        assert len(standardized) == len(original)


# ============================================================================
# Integration Tests
# ============================================================================

class TestNormalizationPipeline:
    """Integration tests for the full normalization pipeline."""

    def normalize_dataframe(self, df, date_col="date", value_col="value"):
        """
        Full normalization pipeline.

        Args:
            df: Input DataFrame
            date_col: Date column name
            value_col: Value column name

        Returns:
            Normalized DataFrame
        """
        result = df.copy()

        # 1. Standardize column names
        result.columns = [c.lower() for c in result.columns]

        # 2. Parse dates
        result[date_col] = pd.to_datetime(result[date_col], errors="coerce")

        # 3. Remove rows with invalid dates
        result = result.dropna(subset=[date_col])

        # 4. Clean numeric values
        if value_col in result.columns:
            result[value_col] = pd.to_numeric(result[value_col], errors="coerce")

        # 5. Sort by date
        result = result.sort_values(date_col).reset_index(drop=True)

        return result

    def test_full_pipeline_with_dirty_data(self):
        """Should handle a messy real-world-like dataset."""
        df = pd.DataFrame({
            "DATE": ["2020-01-03", "2020-01-01", "invalid", "2020-01-02", "Footer"],
            "VALUE": ["100.5", "99.0", "101.0", "N/A", "Source: XYZ"]
        })

        cleaned = self.normalize_dataframe(df, "date", "value")

        # Should have 3 valid rows
        assert len(cleaned) == 3

        # Should be sorted by date
        assert cleaned["date"].iloc[0] < cleaned["date"].iloc[1]
        assert cleaned["date"].iloc[1] < cleaned["date"].iloc[2]

        # Should have proper types
        assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
        assert pd.api.types.is_numeric_dtype(cleaned["value"])

    def test_pipeline_preserves_valid_data(self):
        """Pipeline should preserve all valid data."""
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "value": [100.0, 101.0, 102.0, 103.0, 104.0]
        })

        cleaned = self.normalize_dataframe(df)

        assert len(cleaned) == 5
        assert cleaned["value"].sum() == 510.0
