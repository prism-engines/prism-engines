"""
PRISM CSV Data Import

Simple utility to upload your own data to PRISM via CSV.

Expected CSV Format:
    date,indicator_id,value
    2020-01-01,GDP,21000.5
    2020-01-01,UNRATE,3.5
    2020-01-02,SPY,320.45
    ...

Or wide format:
    date,GDP,UNRATE,SPY,...
    2020-01-01,21000.5,3.5,320.45
    2020-01-02,21050.2,3.5,321.20
    ...

The script auto-detects the format.

Usage:
    # Basic import
    python prism_csv_import.py --file my_data.csv --db prism.db

    # Import and run analysis
    python prism_csv_import.py --file my_data.csv --db prism.db --run

    # Show expected format
    python prism_csv_import.py --example

    # Validate without importing
    python prism_csv_import.py --file my_data.csv --validate-only

Author: Jason (PRISM Project)
Date: December 2024
"""

import pandas as pd
import numpy as np
import duckdb
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CSV FORMAT DETECTION AND VALIDATION
# =============================================================================

class CSVFormatError(Exception):
    """Raised when CSV format is invalid."""
    pass


def detect_format(df: pd.DataFrame) -> str:
    """
    Detect CSV format: 'long' or 'wide'.

    Long format:
        date, indicator_id, value

    Wide format:
        date, indicator1, indicator2, ...
    """
    columns = [c.lower().strip() for c in df.columns]

    # Check for long format markers
    has_indicator_col = any(c in columns for c in ['indicator_id', 'indicator', 'series', 'ticker', 'symbol'])
    has_value_col = any(c in columns for c in ['value', 'close', 'price', 'level'])

    if has_indicator_col and has_value_col:
        return 'long'

    # Check for wide format (date + multiple numeric columns)
    date_cols = [c for c in columns if c in ['date', 'datetime', 'timestamp', 'time', 'period']]
    if date_cols and len(df.columns) > 2:
        # Check if other columns are numeric
        non_date_cols = [c for c in df.columns if c.lower() not in date_cols]
        numeric_count = sum(1 for c in non_date_cols if pd.api.types.is_numeric_dtype(df[c]))
        if numeric_count >= len(non_date_cols) * 0.8:  # 80% numeric
            return 'wide'

    raise CSVFormatError(
        "Could not detect CSV format. Expected either:\n"
        "  Long: columns [date, indicator_id, value]\n"
        "  Wide: columns [date, indicator1, indicator2, ...]"
    )


def find_date_column(df: pd.DataFrame) -> str:
    """Find the date column."""
    candidates = ['date', 'datetime', 'timestamp', 'time', 'period', 'obs_date']
    columns_lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in columns_lower:
            return columns_lower[candidate]

    # Try first column
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col].head(10))
        return first_col
    except:
        pass

    raise CSVFormatError(f"Could not find date column. Expected one of: {candidates}")


def find_indicator_column(df: pd.DataFrame) -> str:
    """Find the indicator ID column (long format)."""
    candidates = ['indicator_id', 'indicator', 'series', 'ticker', 'symbol', 'name', 'id']
    columns_lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in columns_lower:
            return columns_lower[candidate]

    raise CSVFormatError(f"Could not find indicator column. Expected one of: {candidates}")


def find_value_column(df: pd.DataFrame) -> str:
    """Find the value column (long format)."""
    candidates = ['value', 'close', 'price', 'level', 'observation', 'val']
    columns_lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in columns_lower:
            return columns_lower[candidate]

    raise CSVFormatError(f"Could not find value column. Expected one of: {candidates}")


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(df: pd.DataFrame, format_type: str) -> List[str]:
    """
    Validate data quality. Returns list of warnings.
    """
    warnings = []

    if len(df) == 0:
        raise CSVFormatError("CSV file is empty")

    # Check for required columns based on format
    if format_type == 'long':
        date_col = find_date_column(df)
        indicator_col = find_indicator_column(df)
        value_col = find_value_column(df)

        # Check for nulls
        null_dates = df[date_col].isna().sum()
        null_indicators = df[indicator_col].isna().sum()
        null_values = df[value_col].isna().sum()

        if null_dates > 0:
            warnings.append(f"Found {null_dates} rows with missing dates")
        if null_indicators > 0:
            warnings.append(f"Found {null_indicators} rows with missing indicator IDs")
        if null_values > 0:
            pct = 100 * null_values / len(df)
            warnings.append(f"Found {null_values} rows with missing values ({pct:.1f}%)")

        # Check indicator count
        n_indicators = df[indicator_col].nunique()
        if n_indicators < 2:
            warnings.append(f"Only {n_indicators} indicator(s) found. PRISM works best with multiple indicators.")

        # Check date range
        dates = pd.to_datetime(df[date_col])
        date_range = (dates.max() - dates.min()).days
        if date_range < 30:
            warnings.append(f"Date range is only {date_range} days. Consider longer history.")

    elif format_type == 'wide':
        date_col = find_date_column(df)
        indicator_cols = [c for c in df.columns if c != date_col]

        # Check for nulls per indicator
        for col in indicator_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                pct = 100 * null_count / len(df)
                if pct > 20:
                    warnings.append(f"Indicator '{col}' has {pct:.1f}% missing values")

        # Check indicator count
        if len(indicator_cols) < 2:
            warnings.append(f"Only {len(indicator_cols)} indicator(s) found. PRISM works best with multiple.")

    return warnings


def get_data_summary(df: pd.DataFrame, format_type: str) -> Dict[str, Any]:
    """Get summary statistics for the data."""

    if format_type == 'long':
        date_col = find_date_column(df)
        indicator_col = find_indicator_column(df)
        value_col = find_value_column(df)

        dates = pd.to_datetime(df[date_col])
        indicators = df[indicator_col].unique()

        summary = {
            'format': 'long',
            'total_rows': len(df),
            'n_indicators': len(indicators),
            'indicators': list(indicators)[:20],  # First 20
            'date_range': {
                'start': dates.min().strftime('%Y-%m-%d'),
                'end': dates.max().strftime('%Y-%m-%d'),
                'days': (dates.max() - dates.min()).days,
            },
            'observations_per_indicator': df.groupby(indicator_col).size().describe().to_dict(),
        }

    elif format_type == 'wide':
        date_col = find_date_column(df)
        indicator_cols = [c for c in df.columns if c != date_col]
        dates = pd.to_datetime(df[date_col])

        summary = {
            'format': 'wide',
            'total_rows': len(df),
            'n_indicators': len(indicator_cols),
            'indicators': indicator_cols[:20],
            'date_range': {
                'start': dates.min().strftime('%Y-%m-%d'),
                'end': dates.max().strftime('%Y-%m-%d'),
                'days': (dates.max() - dates.min()).days,
            },
        }

    return summary


# =============================================================================
# DATA TRANSFORMATION
# =============================================================================

def normalize_to_long(df: pd.DataFrame, format_type: str) -> pd.DataFrame:
    """
    Normalize data to long format for PRISM.

    Output columns: date, indicator_id, value
    """
    if format_type == 'long':
        date_col = find_date_column(df)
        indicator_col = find_indicator_column(df)
        value_col = find_value_column(df)

        result = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'indicator_id': df[indicator_col].astype(str).str.strip(),
            'value': pd.to_numeric(df[value_col], errors='coerce'),
        })

    elif format_type == 'wide':
        date_col = find_date_column(df)
        indicator_cols = [c for c in df.columns if c != date_col]

        # Melt wide to long
        melted = df.melt(
            id_vars=[date_col],
            value_vars=indicator_cols,
            var_name='indicator_id',
            value_name='value'
        )

        result = pd.DataFrame({
            'date': pd.to_datetime(melted[date_col]),
            'indicator_id': melted['indicator_id'].astype(str).str.strip(),
            'value': pd.to_numeric(melted['value'], errors='coerce'),
        })

    # Clean up
    result = result.dropna(subset=['date', 'indicator_id'])
    result = result.sort_values(['indicator_id', 'date'])
    result = result.reset_index(drop=True)

    return result


# =============================================================================
# DATABASE IMPORT
# =============================================================================

def import_to_duckdb(
    df: pd.DataFrame,
    db_path: str,
    table_name: str = 'imported_series',
    replace: bool = False
) -> int:
    """
    Import normalized data to DuckDB.

    Returns number of rows imported.
    """
    con = duckdb.connect(db_path)

    try:
        # Check if table exists
        existing = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name]
        ).fetchone()[0]

        if existing and not replace:
            # Append
            con.execute(f"""
                INSERT INTO {table_name} (date, indicator_id, value)
                SELECT date, indicator_id, value FROM df
            """)
            logger.info(f"Appended {len(df)} rows to existing table '{table_name}'")
        else:
            # Create/replace
            if existing:
                con.execute(f"DROP TABLE {table_name}")

            con.execute(f"""
                CREATE TABLE {table_name} AS
                SELECT
                    date,
                    indicator_id,
                    value,
                    CURRENT_TIMESTAMP as imported_at
                FROM df
            """)
            logger.info(f"Created table '{table_name}' with {len(df)} rows")

        # Create index
        try:
            con.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_indicator
                ON {table_name}(indicator_id, date)
            """)
        except:
            pass  # Index might already exist

        # Get final count
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        return count

    finally:
        con.close()


def create_prism_views(db_path: str, source_table: str = 'imported_series'):
    """
    Create views that PRISM expects from imported data.
    """
    con = duckdb.connect(db_path)

    try:
        # Create indicator registry if not exists
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS indicator_registry AS
            SELECT DISTINCT
                indicator_id,
                indicator_id as name,
                'imported' as source,
                MIN(date) as first_date,
                MAX(date) as last_date,
                COUNT(*) as n_observations,
                'unknown' as frequency
            FROM {source_table}
            GROUP BY indicator_id
        """)

        # Update frequency estimate
        con.execute(f"""
            UPDATE indicator_registry
            SET frequency = CASE
                WHEN n_observations / GREATEST(1, DATEDIFF('day', first_date, last_date)) > 0.9 THEN 'daily'
                WHEN n_observations / GREATEST(1, DATEDIFF('week', first_date, last_date)) > 0.8 THEN 'weekly'
                WHEN n_observations / GREATEST(1, DATEDIFF('month', first_date, last_date)) > 0.8 THEN 'monthly'
                ELSE 'irregular'
            END
        """)

        logger.info("Created indicator_registry table")

        # Create time series view that PRISM expects
        con.execute(f"""
            CREATE OR REPLACE VIEW series_data AS
            SELECT
                date as obs_date,
                indicator_id,
                value,
                ROW_NUMBER() OVER (PARTITION BY indicator_id ORDER BY date) as obs_num
            FROM {source_table}
            WHERE value IS NOT NULL
        """)

        logger.info("Created series_data view")

    finally:
        con.close()


# =============================================================================
# EXAMPLE DATA
# =============================================================================

EXAMPLE_LONG = """date,indicator_id,value
2020-01-02,SPY,324.87
2020-01-02,TLT,139.20
2020-01-02,GLD,146.38
2020-01-03,SPY,323.41
2020-01-03,TLT,140.11
2020-01-03,GLD,148.95
2020-01-06,SPY,321.73
2020-01-06,TLT,140.64
2020-01-06,GLD,150.24
2020-01-07,SPY,322.41
2020-01-07,TLT,139.98
2020-01-07,GLD,149.82"""

EXAMPLE_WIDE = """date,SPY,TLT,GLD,VIX
2020-01-02,324.87,139.20,146.38,12.47
2020-01-03,323.41,140.11,148.95,13.01
2020-01-06,321.73,140.64,150.24,13.78
2020-01-07,322.41,139.98,149.82,13.02
2020-01-08,324.45,139.55,150.59,12.51
2020-01-09,325.54,138.89,149.82,12.36
2020-01-10,325.14,139.30,150.13,12.56"""


def show_example():
    """Print example CSV formats."""
    print("""
PRISM CSV Import - Expected Formats
====================================

OPTION 1: Long Format (recommended for mixed frequencies)
---------------------------------------------------------
{long}

- One row per observation
- Columns: date, indicator_id, value
- Different indicators can have different dates (native sampling!)


OPTION 2: Wide Format (simpler for aligned data)
------------------------------------------------
{wide}

- One row per date
- Columns: date, then one column per indicator
- All indicators must have same dates


COLUMN NAME VARIATIONS
----------------------
Date column:     date, datetime, timestamp, time, period
Indicator column: indicator_id, indicator, series, ticker, symbol
Value column:    value, close, price, level


TIPS
----
1. Long format is better for mixed frequencies (daily + monthly)
2. Wide format is simpler if all data is daily
3. Missing values are OK - PRISM handles gaps
4. Use consistent indicator names (SPY not S&P500 vs sp500)
""".format(long=EXAMPLE_LONG, wide=EXAMPLE_WIDE))


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Import CSV data into PRISM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show expected CSV format
    python prism_csv_import.py --example

    # Import data
    python prism_csv_import.py --file my_data.csv --db prism.db

    # Validate without importing
    python prism_csv_import.py --file my_data.csv --validate-only

    # Replace existing data
    python prism_csv_import.py --file my_data.csv --db prism.db --replace
"""
    )

    parser.add_argument('--file', '-f', type=str, help='CSV file to import')
    parser.add_argument('--db', type=str, default='prism.db', help='DuckDB database path')
    parser.add_argument('--table', type=str, default='imported_series', help='Table name')
    parser.add_argument('--replace', action='store_true', help='Replace existing data')
    parser.add_argument('--validate-only', action='store_true', help='Validate without importing')
    parser.add_argument('--example', action='store_true', help='Show example CSV formats')
    parser.add_argument('--run', action='store_true', help='Run PRISM analysis after import')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Show example and exit
    if args.example:
        show_example()
        return 0

    # Require file for other operations
    if not args.file:
        parser.print_help()
        print("\nError: --file is required (or use --example to see format)")
        return 1

    # Check file exists
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return 1

    # Read CSV
    try:
        df = pd.read_csv(args.file)
        if not args.quiet:
            logger.info(f"Read {len(df)} rows from {args.file}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1

    # Detect format
    try:
        format_type = detect_format(df)
        if not args.quiet:
            logger.info(f"Detected format: {format_type}")
    except CSVFormatError as e:
        print(f"Error: {e}")
        show_example()
        return 1

    # Validate
    try:
        warnings = validate_data(df, format_type)
        for warning in warnings:
            logger.warning(f"Warning: {warning}")
    except CSVFormatError as e:
        print(f"Error: {e}")
        return 1

    # Show summary
    if not args.quiet:
        summary = get_data_summary(df, format_type)
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Format:      {summary['format']}")
        print(f"Rows:        {summary['total_rows']:,}")
        print(f"Indicators:  {summary['n_indicators']}")
        print(f"Date range:  {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"             ({summary['date_range']['days']} days)")
        print(f"Indicators:  {', '.join(summary['indicators'][:10])}")
        if len(summary['indicators']) > 10:
            print(f"             ... and {len(summary['indicators']) - 10} more")
        print("="*50 + "\n")

    # Validate-only stops here
    if args.validate_only:
        print("Validation complete. Use without --validate-only to import.")
        return 0

    # Normalize to long format
    try:
        normalized = normalize_to_long(df, format_type)
        if not args.quiet:
            logger.info(f"Normalized to {len(normalized)} observations")
    except Exception as e:
        print(f"Error normalizing data: {e}")
        return 1

    # Import to DuckDB
    try:
        count = import_to_duckdb(
            normalized,
            args.db,
            args.table,
            replace=args.replace
        )
        logger.info(f"Database now has {count:,} total observations")
    except Exception as e:
        print(f"Error importing to database: {e}")
        return 1

    # Create PRISM views
    try:
        create_prism_views(args.db, args.table)
    except Exception as e:
        print(f"Warning: Could not create views: {e}")

    # Success message
    print("\n" + "="*50)
    print("IMPORT SUCCESSFUL")
    print("="*50)
    print(f"Database: {args.db}")
    print(f"Table:    {args.table}")
    print(f"Rows:     {count:,}")
    print()
    print("Next steps:")
    print(f"  1. Run data phase:    python run_data_phase.py --db {args.db}")
    print(f"  2. Run derived phase: python run_derived_phase.py --db {args.db}")
    print(f"  3. Run analysis:      python prism_temporal_geometry.py --db {args.db}")
    print("="*50)

    # Optionally run analysis
    if args.run:
        print("\nRunning PRISM analysis...")
        # TODO: Import and run analysis pipeline
        print("(--run not yet implemented, run manually)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
