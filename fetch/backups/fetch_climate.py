#!/usr/bin/env python3
"""
PRISM Climate Data Fetcher
==========================

Fetches climate indicators from public sources:
- NASA GISS: Global temperature anomaly
- NOAA: CO2 (Mauna Loa), ENSO index
- NSIDC: Sea ice extent

All data formatted for PRISM: date, indicator, value

Usage:
    python fetch_climate.py              # Fetch all
    python fetch_climate.py --test       # Test connections
    python fetch_climate.py --status     # Show database status

Output:
    ~/prism_data/climate.db
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
from pathlib import Path
from datetime import datetime
import logging
import argparse
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path.home() / "prism_data" / "climate.db"


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database():
    """Initialize the climate database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS climate_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL,
            UNIQUE(indicator, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS climate_metadata (
            indicator TEXT PRIMARY KEY,
            name TEXT,
            source TEXT,
            units TEXT,
            frequency TEXT,
            last_updated TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_climate_indicator_date ON climate_values(indicator, date)")
    conn.commit()
    conn.close()
    logger.info(f"Database initialized: {DB_PATH}")


def upsert_climate_data(df: pd.DataFrame, indicator: str, metadata: dict = None):
    """Upsert climate data to database."""
    if df is None or df.empty:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['indicator'] = indicator
    
    count = 0
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO climate_values (indicator, date, value)
            VALUES (?, ?, ?)
            ON CONFLICT(indicator, date) DO UPDATE SET value = excluded.value
        """, (indicator, row['date'], row['value']))
        count += 1
    
    # Update metadata
    if metadata:
        cursor.execute("""
            INSERT INTO climate_metadata (indicator, name, source, units, frequency, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(indicator) DO UPDATE SET
                name = excluded.name,
                source = excluded.source,
                units = excluded.units,
                frequency = excluded.frequency,
                last_updated = excluded.last_updated
        """, (
            indicator,
            metadata.get('name', indicator),
            metadata.get('source', 'unknown'),
            metadata.get('units', ''),
            metadata.get('frequency', 'monthly'),
            datetime.now().isoformat()
        ))
    
    conn.commit()
    conn.close()
    
    return count


# ============================================================================
# DATA FETCHERS
# ============================================================================

def fetch_nasa_giss_temperature():
    """
    Fetch NASA GISS Global Temperature Anomaly.
    Source: https://data.giss.nasa.gov/gistemp/
    """
    logger.info("Fetching NASA GISS Global Temperature Anomaly...")
    
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse the CSV (skip header rows)
        df = pd.read_csv(io.StringIO(response.text), skiprows=1)
        
        # Melt from wide to long format
        # Columns are: Year, Jan, Feb, Mar, ... Dec, J-D, D-N, DJF, MAM, JJA, SON
        month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        records = []
        for _, row in df.iterrows():
            year = row['Year']
            for i, month in enumerate(month_cols, 1):
                if month in row and pd.notna(row[month]) and row[month] != '***':
                    try:
                        value = float(row[month])
                        date = f"{int(year)}-{i:02d}-01"
                        records.append({'date': date, 'value': value})
                    except (ValueError, TypeError):
                        continue
        
        df_out = pd.DataFrame(records)
        logger.info(f"  -> {len(df_out)} records from {df_out['date'].min()} to {df_out['date'].max()}")
        
        return df_out, {
            'name': 'Global Temperature Anomaly',
            'source': 'NASA GISS',
            'units': '°C relative to 1951-1980 baseline',
            'frequency': 'monthly'
        }
        
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return None, None


def fetch_noaa_co2():
    """
    Fetch Mauna Loa CO2 data.
    Source: https://gml.noaa.gov/ccgg/trends/data.html
    """
    logger.info("Fetching NOAA Mauna Loa CO2...")
    
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Skip comment lines starting with #
        lines = [l for l in response.text.split('\n') if not l.startswith('#') and l.strip()]
        csv_text = '\n'.join(lines)
        
        df = pd.read_csv(io.StringIO(csv_text))
        
        # Columns: year, month, decimal_date, average, deseasonalized, ndays, sdev, unc
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df['value'] = df['average'].replace(-99.99, np.nan)
        
        df_out = df[['date', 'value']].dropna()
        logger.info(f"  -> {len(df_out)} records from {df_out['date'].min().date()} to {df_out['date'].max().date()}")
        
        return df_out, {
            'name': 'CO2 Concentration (Mauna Loa)',
            'source': 'NOAA GML',
            'units': 'ppm',
            'frequency': 'monthly'
        }
        
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return None, None


def fetch_enso_index():
    """
    Fetch ENSO (El Niño Southern Oscillation) Index.
    Source: NOAA Climate Prediction Center
    """
    logger.info("Fetching NOAA ENSO Index (ONI)...")
    
    # Oceanic Niño Index (ONI)
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse fixed-width format
        lines = response.text.strip().split('\n')
        
        records = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Format: SEAS  YR   TOTAL   APTS (e.g., "DJF  1950  -1.5   -1.5")
                    season = parts[0]
                    year = int(parts[1])
                    value = float(parts[2])
                    
                    # Map season to month (use middle month)
                    season_to_month = {
                        'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4,
                        'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
                        'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
                    }
                    
                    if season in season_to_month:
                        month = season_to_month[season]
                        date = f"{year}-{month:02d}-01"
                        records.append({'date': date, 'value': value})
                except (ValueError, IndexError):
                    continue
        
        df_out = pd.DataFrame(records)
        df_out = df_out.drop_duplicates(subset=['date']).sort_values('date')
        
        logger.info(f"  -> {len(df_out)} records from {df_out['date'].min()} to {df_out['date'].max()}")
        
        return df_out, {
            'name': 'Oceanic Niño Index (ENSO)',
            'source': 'NOAA CPC',
            'units': '°C anomaly',
            'frequency': 'monthly'
        }
        
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return None, None


def fetch_sea_ice_extent():
    """
    Fetch Arctic Sea Ice Extent.
    Source: NSIDC
    """
    logger.info("Fetching NSIDC Arctic Sea Ice Extent...")
    
    url = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/N_09_extent_v3.0.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text), skipinitialspace=True)
        
        # Columns typically: year, month, extent, area
        df.columns = [c.strip().lower() for c in df.columns]
        
        if 'year' in df.columns and 'mo' in df.columns:
            df['date'] = pd.to_datetime(df[['year', 'mo']].rename(columns={'mo': 'month'}).assign(day=1))
            df['value'] = pd.to_numeric(df['extent'], errors='coerce')
        elif 'year' in df.columns and 'month' in df.columns:
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            df['value'] = pd.to_numeric(df['extent'], errors='coerce')
        else:
            raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
        
        df_out = df[['date', 'value']].dropna()
        logger.info(f"  -> {len(df_out)} records")
        
        return df_out, {
            'name': 'Arctic Sea Ice Extent (September)',
            'source': 'NSIDC',
            'units': 'million sq km',
            'frequency': 'monthly'
        }
        
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return None, None


def fetch_global_sea_level():
    """
    Fetch Global Mean Sea Level.
    Source: NASA/NOAA
    """
    logger.info("Fetching NASA Global Mean Sea Level...")
    
    url = "https://climate.nasa.gov/system/internal_resources/details/original/121_Global_Sea_Level_Data_File.txt"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse the data (HDR lines are headers)
        lines = [l for l in response.text.split('\n') if not l.startswith('HDR') and l.strip()]
        
        records = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 12:
                try:
                    # Column 3 is year fraction, column 12 is smoothed GMSL
                    year_frac = float(parts[2])
                    gmsl = float(parts[11])  # GMSL with GIA applied
                    
                    # Convert year fraction to date
                    year = int(year_frac)
                    month = int((year_frac - year) * 12) + 1
                    month = min(12, max(1, month))
                    date = f"{year}-{month:02d}-01"
                    
                    records.append({'date': date, 'value': gmsl})
                except (ValueError, IndexError):
                    continue
        
        df_out = pd.DataFrame(records)
        # Average by month (multiple readings per month)
        df_out['date'] = pd.to_datetime(df_out['date'])
        df_out = df_out.groupby('date').mean().reset_index()
        
        logger.info(f"  -> {len(df_out)} records")
        
        return df_out, {
            'name': 'Global Mean Sea Level',
            'source': 'NASA',
            'units': 'mm relative to 1993-2008 baseline',
            'frequency': 'monthly'
        }
        
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return None, None


def fetch_sunspot_number():
    """
    Fetch Sunspot Number (solar activity).
    Source: SILSO (Royal Observatory of Belgium)
    """
    logger.info("Fetching SILSO Sunspot Number...")
    
    url = "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # CSV format: year;month;decimal_year;SNvalue;SNerror;Nb_observations;marker
        df = pd.read_csv(io.StringIO(response.text), sep=';', header=None,
                         names=['year', 'month', 'decimal_year', 'value', 'error', 'n_obs', 'marker'])
        
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[df['value'] >= 0]  # -1 means missing
        
        df_out = df[['date', 'value']].dropna()
        logger.info(f"  -> {len(df_out)} records from {df_out['date'].min().date()} to {df_out['date'].max().date()}")
        
        return df_out, {
            'name': 'Monthly Sunspot Number',
            'source': 'SILSO (Royal Observatory of Belgium)',
            'units': 'count',
            'frequency': 'monthly'
        }
        
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return None, None


# ============================================================================
# MAIN
# ============================================================================

def fetch_all():
    """Fetch all climate indicators."""
    print("=" * 60)
    print("🌍 PRISM CLIMATE DATA FETCHER")
    print("=" * 60)
    
    init_database()
    
    fetchers = [
        ('temp_anomaly', fetch_nasa_giss_temperature),
        ('co2', fetch_noaa_co2),
        ('enso', fetch_enso_index),
        ('sea_ice', fetch_sea_ice_extent),
        ('sea_level', fetch_global_sea_level),
        ('sunspots', fetch_sunspot_number),
    ]
    
    results = {}
    
    for indicator, fetcher in fetchers:
        try:
            df, metadata = fetcher()
            if df is not None and not df.empty:
                rows = upsert_climate_data(df, indicator, metadata)
                results[indicator] = rows
                print(f"   ✅ {indicator}: {rows} rows")
            else:
                results[indicator] = 0
                print(f"   ❌ {indicator}: no data")
        except Exception as e:
            logger.error(f"   ❌ {indicator}: {e}")
            results[indicator] = 0
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Database: {DB_PATH}")
    print(f"   Total indicators: {len(results)}")
    print(f"   Total rows: {sum(results.values())}")
    
    return results


def show_status():
    """Show database status."""
    print("=" * 60)
    print("📊 CLIMATE DATABASE STATUS")
    print("=" * 60)
    print(f"   Database: {DB_PATH}")
    
    if not DB_PATH.exists():
        print("   ❌ Database not found. Run: python fetch_climate.py")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get indicator stats
    df = pd.read_sql("""
        SELECT 
            indicator,
            COUNT(*) as rows,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM climate_values
        GROUP BY indicator
        ORDER BY indicator
    """, conn)
    
    print("\n   Indicators:")
    for _, row in df.iterrows():
        print(f"   {row['indicator']:15s} {row['rows']:5d} rows  ({row['min_date']} to {row['max_date']})")
    
    # Get metadata
    meta = pd.read_sql("SELECT * FROM climate_metadata", conn)
    conn.close()
    
    print(f"\n   Total rows: {df['rows'].sum()}")


def test_connections():
    """Test all data source connections."""
    print("=" * 60)
    print("🔌 TESTING CONNECTIONS")
    print("=" * 60)
    
    tests = [
        ("NASA GISS", "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"),
        ("NOAA CO2", "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"),
        ("NOAA ENSO", "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"),
        ("SILSO Sunspots", "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv"),
    ]
    
    for name, url in tests:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"   ✅ {name}")
            else:
                print(f"   ❌ {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Climate Data Fetcher")
    parser.add_argument("--test", "-t", action="store_true", help="Test connections")
    parser.add_argument("--status", "-s", action="store_true", help="Show database status")
    
    args = parser.parse_args()
    
    if args.test:
        test_connections()
    elif args.status:
        show_status()
    else:
        fetch_all()
