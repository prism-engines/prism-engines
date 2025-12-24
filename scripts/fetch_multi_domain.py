"""
PRISM Multi-Domain Data Fetcher

Fetches indicators from three domains to validate domain-agnostic claims:
1. Climate - NOAA, NASA, global climate indices
2. Forest/Ecosystem - MODIS, USGS, vegetation/hydrology
3. Industrial - FRED, EIA, manufacturing/production

All indicators are formatted consistently for the PRISM pipeline.

Usage:
    python fetch_multi_domain.py --domain climate --output data/climate/
    python fetch_multi_domain.py --domain forest --output data/forest/
    python fetch_multi_domain.py --domain industrial --output data/industrial/
    python fetch_multi_domain.py --all --output data/

Cross-validated by: Claude
Date: December 2024
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np
import requests
from io import StringIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# FRED API - you'll need a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.environ.get('FRED_API_KEY', 'YOUR_FRED_API_KEY_HERE')

# Date range for fetching
DEFAULT_START = '1990-01-01'
DEFAULT_END = datetime.now().strftime('%Y-%m-%d')


# =============================================================================
# INDICATOR DEFINITIONS
# =============================================================================

CLIMATE_INDICATORS = {
    # Temperature
    'GISTEMP': {
        'name': 'NASA GISS Global Temperature Anomaly',
        'frequency': 'monthly',
        'source': 'nasa_giss',
        'url': 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv',
        'unit': 'degrees_c_anomaly',
    },
    
    # Atmospheric CO2
    'CO2_MLO': {
        'name': 'Mauna Loa CO2 Concentration',
        'frequency': 'monthly',
        'source': 'noaa_esrl',
        'url': 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv',
        'unit': 'ppm',
    },
    
    # ENSO Index
    'ONI': {
        'name': 'Oceanic Nino Index (ENSO)',
        'frequency': 'monthly',
        'source': 'noaa_cpc',
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',
        'unit': 'index',
    },
    
    # Arctic Sea Ice
    'ARCTIC_ICE': {
        'name': 'Arctic Sea Ice Extent',
        'frequency': 'monthly',
        'source': 'nsidc',
        'url': 'https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/N_seaice_extent_monthly.csv',
        'unit': 'million_km2',
    },
    
    # Solar Activity
    'SUNSPOTS': {
        'name': 'Monthly Sunspot Number',
        'frequency': 'monthly',
        'source': 'silso',
        'url': 'https://www.sidc.be/SILSO/INFO/snmtotcsv.php',
        'unit': 'count',
    },
    
    # North Atlantic Oscillation
    'NAO': {
        'name': 'North Atlantic Oscillation Index',
        'frequency': 'monthly',
        'source': 'noaa_cpc',
        'url': 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table',
        'unit': 'index',
    },
    
    # Pacific Decadal Oscillation
    'PDO': {
        'name': 'Pacific Decadal Oscillation',
        'frequency': 'monthly',
        'source': 'noaa_ncei',
        'url': 'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat',
        'unit': 'index',
    },
    
    # Global Sea Level
    'SEA_LEVEL': {
        'name': 'Global Mean Sea Level',
        'frequency': 'monthly',
        'source': 'noaa',
        'url': 'https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gbl_free_txj1j2_90.csv',
        'unit': 'mm',
    },
}

FOREST_INDICATORS = {
    # Drought Index (from FRED - yes they have it)
    'PDSI_US': {
        'name': 'Palmer Drought Severity Index (US)',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'DSPI',  # Drought Severity and Coverage Index
        'unit': 'index',
    },
    
    # We'll fetch NDVI and fire data via NASA EarthData / FIRMS
    # These require authentication - providing manual download instructions
    
    # USGS Streamflow (example: Mississippi River)
    'STREAMFLOW_MISS': {
        'name': 'Mississippi River Discharge at Vicksburg',
        'frequency': 'daily',
        'source': 'usgs',
        'site_id': '07289000',
        'unit': 'cubic_feet_per_second',
    },
    
    # Lake Mead Level (water stress indicator)
    'LAKE_MEAD': {
        'name': 'Lake Mead Water Level',
        'frequency': 'daily',
        'source': 'usbr',
        'url': 'https://www.usbr.gov/lc/region/g4000/hourly/mead-elv.html',
        'unit': 'feet_above_sea_level',
    },
    
    # Wildfire data - use NIFC (National Interagency Fire Center)
    'WILDFIRE_ACRES': {
        'name': 'US Wildfire Acres Burned YTD',
        'frequency': 'daily',
        'source': 'nifc',
        'url': 'https://www.nifc.gov/fire-information/statistics',
        'unit': 'acres',
    },
}

INDUSTRIAL_INDICATORS = {
    # All from FRED
    'INDPRO': {
        'name': 'Industrial Production Index',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'INDPRO',
        'unit': 'index_2017_100',
    },
    
    'TCU': {
        'name': 'Capacity Utilization: Total Industry',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'TCU',
        'unit': 'percent',
    },
    
    'DGORDER': {
        'name': 'Durable Goods New Orders',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'DGORDER',
        'unit': 'millions_usd',
    },
    
    'RAILFRTCARLOADSD11': {
        'name': 'Rail Freight Carloads',
        'frequency': 'weekly',
        'source': 'fred',
        'series_id': 'RAILFRTCARLOADSD11',
        'unit': 'carloads',
    },
    
    'ELEC_PROD': {
        'name': 'Electric Power Generation',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'IPG2211A2N',
        'unit': 'index_2017_100',
    },
    
    'STEEL': {
        'name': 'Steel Production Capacity',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'CAPUTLG3311A2S',
        'unit': 'percent',
    },
    
    'SEMI_SHIP': {
        'name': 'Semiconductor Shipments',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'S4248SM144NCEN',  # Electronic component shipments
        'unit': 'millions_usd',
    },
    
    'TRUCK_TONNAGE': {
        'name': 'Truck Tonnage Index',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'TRUCKD11',
        'unit': 'index',
    },
    
    'ISM_PMI': {
        'name': 'ISM Manufacturing PMI',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'MANEMP',  # Manufacturing employment as proxy
        'unit': 'thousands',
    },
    
    'INVENTORY_SALES': {
        'name': 'Inventory to Sales Ratio',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'ISRATIO',
        'unit': 'ratio',
    },
    
    'CHEMICAL_PROD': {
        'name': 'Chemical Production Index',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'IPG325N',
        'unit': 'index_2017_100',
    },
    
    'MOTOR_VEHICLE': {
        'name': 'Motor Vehicle Production',
        'frequency': 'monthly',
        'source': 'fred',
        'series_id': 'DAUPSA',
        'unit': 'units_annual_rate',
    },
}


# =============================================================================
# DATA FETCHERS
# =============================================================================

def fetch_fred(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data from FRED API."""
    
    if FRED_API_KEY == 'YOUR_FRED_API_KEY_HERE':
        logger.warning("No FRED API key set. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html")
        logger.warning("Set environment variable: export FRED_API_KEY=your_key_here")
        return pd.DataFrame()
    
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'observations' not in data:
            logger.error(f"No observations in FRED response for {series_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['date', 'value']].dropna()
        df = df.set_index('date')
        
        logger.info(f"Fetched {len(df)} observations for FRED:{series_id}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch FRED {series_id}: {e}")
        return pd.DataFrame()


def fetch_nasa_giss() -> pd.DataFrame:
    """Fetch NASA GISS global temperature anomaly."""
    url = CLIMATE_INDICATORS['GISTEMP']['url']
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Skip header lines
        lines = response.text.split('\n')
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('Year'):
                data_start = i
                break
        
        df = pd.read_csv(StringIO('\n'.join(lines[data_start:])), 
                        na_values=['***', '****'])
        
        # Reshape from wide to long
        records = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for _, row in df.iterrows():
            year = int(row['Year'])
            for month_idx, month in enumerate(months, 1):
                if month in row and pd.notna(row[month]):
                    date = pd.Timestamp(year=year, month=month_idx, day=1)
                    # GISS reports in 0.01 degrees C
                    value = float(row[month]) / 100.0
                    records.append({'date': date, 'value': value})
        
        result = pd.DataFrame(records).set_index('date').sort_index()
        logger.info(f"Fetched {len(result)} observations for NASA GISS")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch NASA GISS: {e}")
        return pd.DataFrame()


def fetch_noaa_co2() -> pd.DataFrame:
    """Fetch Mauna Loa CO2 data."""
    url = CLIMATE_INDICATORS['CO2_MLO']['url']
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Skip comment lines starting with #
        lines = [l for l in response.text.split('\n') if not l.startswith('#') and l.strip()]
        
        df = pd.read_csv(StringIO('\n'.join(lines)), 
                        names=['year', 'month', 'decimal_date', 'average', 'deseasonalized', 
                               'ndays', 'std', 'uncertainty'],
                        skipinitialspace=True)
        
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + 
                                    df['month'].astype(str).str.zfill(2) + '-01')
        df['value'] = pd.to_numeric(df['average'], errors='coerce')
        
        # -99.99 is missing value indicator
        df.loc[df['value'] < 0, 'value'] = np.nan
        
        result = df[['date', 'value']].dropna().set_index('date').sort_index()
        logger.info(f"Fetched {len(result)} observations for CO2")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch CO2: {e}")
        return pd.DataFrame()


def fetch_sunspots() -> pd.DataFrame:
    """Fetch monthly sunspot numbers from SILSO."""
    url = CLIMATE_INDICATORS['SUNSPOTS']['url']
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Format: year;month;decimal_year;SNvalue;SNerror;Nb_observations;provisional
        df = pd.read_csv(StringIO(response.text), sep=';', header=None,
                        names=['year', 'month', 'decimal', 'value', 'error', 'nobs', 'provisional'])
        
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + 
                                    df['month'].astype(str).str.zfill(2) + '-01')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # -1 is missing
        df.loc[df['value'] < 0, 'value'] = np.nan
        
        result = df[['date', 'value']].dropna().set_index('date').sort_index()
        logger.info(f"Fetched {len(result)} observations for Sunspots")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch Sunspots: {e}")
        return pd.DataFrame()


def fetch_arctic_ice() -> pd.DataFrame:
    """Fetch Arctic sea ice extent from NSIDC."""
    url = CLIMATE_INDICATORS['ARCTIC_ICE']['url']
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text), skiprows=2)
        
        # Columns: year, mo, data-type, region, extent, area
        df.columns = df.columns.str.strip()
        
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + 
                                    df['mo'].astype(str).str.zfill(2) + '-01')
        df['value'] = pd.to_numeric(df['extent'], errors='coerce')
        
        result = df[['date', 'value']].dropna().set_index('date').sort_index()
        logger.info(f"Fetched {len(result)} observations for Arctic Ice")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch Arctic Ice: {e}")
        return pd.DataFrame()


def fetch_oni() -> pd.DataFrame:
    """Fetch Oceanic Nino Index (ENSO indicator)."""
    url = CLIMATE_INDICATORS['ONI']['url']
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse fixed-width format
        lines = response.text.strip().split('\n')
        
        records = []
        seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 
                   'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
        
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 13:
                year = int(parts[0])
                for i, season in enumerate(seasons):
                    try:
                        value = float(parts[i + 1])
                        # Map season to month (middle month of 3-month period)
                        month_map = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4,
                                    'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
                                    'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}
                        month = month_map[season]
                        
                        # Handle year rollover for DJF
                        actual_year = year if season != 'DJF' else year
                        
                        date = pd.Timestamp(year=actual_year, month=month, day=1)
                        records.append({'date': date, 'value': value})
                    except (ValueError, IndexError):
                        continue
        
        result = pd.DataFrame(records).set_index('date').sort_index()
        result = result[~result.index.duplicated(keep='last')]  # Remove duplicate months
        logger.info(f"Fetched {len(result)} observations for ONI")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch ONI: {e}")
        return pd.DataFrame()


def fetch_usgs_streamflow(site_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch USGS streamflow data."""
    
    url = 'https://waterservices.usgs.gov/nwis/dv/'
    params = {
        'format': 'json',
        'sites': site_id,
        'startDT': start_date,
        'endDT': end_date,
        'parameterCd': '00060',  # Discharge
        'siteStatus': 'all',
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Navigate JSON structure
        ts = data['value']['timeSeries'][0]['values'][0]['value']
        
        records = []
        for obs in ts:
            date = pd.to_datetime(obs['dateTime']).normalize()
            try:
                value = float(obs['value'])
                if value >= 0:  # Negative means missing
                    records.append({'date': date, 'value': value})
            except ValueError:
                continue
        
        result = pd.DataFrame(records).set_index('date').sort_index()
        logger.info(f"Fetched {len(result)} observations for USGS {site_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch USGS {site_id}: {e}")
        return pd.DataFrame()


# =============================================================================
# DOMAIN FETCHERS
# =============================================================================

def fetch_climate_domain(output_dir: str, start_date: str, end_date: str) -> Dict[str, str]:
    """Fetch all climate indicators."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # NASA GISS Temperature
    df = fetch_nasa_giss()
    if not df.empty:
        path = os.path.join(output_dir, 'GISTEMP.csv')
        df.to_csv(path)
        results['GISTEMP'] = path
    
    # CO2
    df = fetch_noaa_co2()
    if not df.empty:
        path = os.path.join(output_dir, 'CO2_MLO.csv')
        df.to_csv(path)
        results['CO2_MLO'] = path
    
    # Sunspots
    df = fetch_sunspots()
    if not df.empty:
        path = os.path.join(output_dir, 'SUNSPOTS.csv')
        df.to_csv(path)
        results['SUNSPOTS'] = path
    
    # Arctic Ice
    df = fetch_arctic_ice()
    if not df.empty:
        path = os.path.join(output_dir, 'ARCTIC_ICE.csv')
        df.to_csv(path)
        results['ARCTIC_ICE'] = path
    
    # ONI (ENSO)
    df = fetch_oni()
    if not df.empty:
        path = os.path.join(output_dir, 'ONI.csv')
        df.to_csv(path)
        results['ONI'] = path
    
    # NAO - requires different parsing
    # PDO - requires different parsing
    # For now, note these as TODO
    
    return results


def fetch_forest_domain(output_dir: str, start_date: str, end_date: str) -> Dict[str, str]:
    """Fetch all forest/ecosystem indicators."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # PDSI from FRED
    df = fetch_fred('DSPI', start_date, end_date)
    if not df.empty:
        path = os.path.join(output_dir, 'PDSI_US.csv')
        df.to_csv(path)
        results['PDSI_US'] = path
    
    # Mississippi Streamflow
    df = fetch_usgs_streamflow('07289000', start_date, end_date)
    if not df.empty:
        path = os.path.join(output_dir, 'STREAMFLOW_MISS.csv')
        df.to_csv(path)
        results['STREAMFLOW_MISS'] = path
    
    # Additional USGS sites for ecosystem coverage
    
    # Colorado River at Lees Ferry (drought indicator)
    df = fetch_usgs_streamflow('09380000', start_date, end_date)
    if not df.empty:
        path = os.path.join(output_dir, 'STREAMFLOW_COLORADO.csv')
        df.to_csv(path)
        results['STREAMFLOW_COLORADO'] = path
    
    # Columbia River (Pacific Northwest)
    df = fetch_usgs_streamflow('14105700', start_date, end_date)
    if not df.empty:
        path = os.path.join(output_dir, 'STREAMFLOW_COLUMBIA.csv')
        df.to_csv(path)
        results['STREAMFLOW_COLUMBIA'] = path
    
    logger.info("""
    NOTE: For complete forest/ecosystem coverage, manually download:
    
    1. NDVI (Vegetation Health):
       - Source: NASA EarthData (requires free account)
       - URL: https://earthdata.nasa.gov/
       - Product: MOD13C2 (Monthly NDVI)
       
    2. Fire Data:
       - Source: NASA FIRMS
       - URL: https://firms.modaps.eosdis.nasa.gov/download/
       - Download: Monthly fire counts by region
       
    3. Global Forest Watch:
       - URL: https://www.globalforestwatch.org/dashboards/global/
       - Download: Tree cover loss data
    """)
    
    return results


def fetch_industrial_domain(output_dir: str, start_date: str, end_date: str) -> Dict[str, str]:
    """Fetch all industrial/manufacturing indicators."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for ind_id, config in INDUSTRIAL_INDICATORS.items():
        if config['source'] == 'fred':
            time.sleep(0.5)  # Rate limiting
            df = fetch_fred(config['series_id'], start_date, end_date)
            if not df.empty:
                path = os.path.join(output_dir, f'{ind_id}.csv')
                df.to_csv(path)
                results[ind_id] = path
    
    return results


# =============================================================================
# METADATA GENERATION
# =============================================================================

def generate_indicators_yaml(domain: str, results: Dict[str, str], output_dir: str):
    """Generate indicators.yaml for PRISM pipeline."""
    
    if domain == 'climate':
        indicators = CLIMATE_INDICATORS
    elif domain == 'forest':
        indicators = FOREST_INDICATORS
    elif domain == 'industrial':
        indicators = INDUSTRIAL_INDICATORS
    else:
        return
    
    yaml_path = os.path.join(output_dir, f'{domain}_indicators.yaml')
    
    with open(yaml_path, 'w') as f:
        f.write(f"# PRISM {domain.title()} Domain Indicators\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        f.write("indicators:\n")
        
        for ind_id, file_path in results.items():
            if ind_id not in indicators:
                continue
            config = indicators[ind_id]
            f.write(f"\n  {ind_id}:\n")
            f.write(f"    name: \"{config['name']}\"\n")
            f.write(f"    frequency: {config['frequency']}\n")
            f.write(f"    source: {config['source']}\n")
            f.write(f"    unit: {config['unit']}\n")
            f.write(f"    file: {file_path}\n")
            f.write(f"    domain: {domain}\n")
    
    logger.info(f"Generated {yaml_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fetch multi-domain indicators for PRISM validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch climate data
    python fetch_multi_domain.py --domain climate --output data/climate/
    
    # Fetch all domains
    python fetch_multi_domain.py --all --output data/
    
    # With custom date range
    python fetch_multi_domain.py --domain industrial --start 2000-01-01 --output data/industrial/
    
Environment Variables:
    FRED_API_KEY - Required for FRED data (free at https://fred.stlouisfed.org/docs/api/api_key.html)
        """
    )
    
    parser.add_argument('--domain', type=str, choices=['climate', 'forest', 'industrial'],
                       help='Domain to fetch')
    parser.add_argument('--all', action='store_true', help='Fetch all domains')
    parser.add_argument('--output', type=str, default='data/', help='Output directory')
    parser.add_argument('--start', type=str, default=DEFAULT_START, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=DEFAULT_END, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if not args.domain and not args.all:
        parser.print_help()
        return
    
    domains = ['climate', 'forest', 'industrial'] if args.all else [args.domain]
    
    all_results = {}
    
    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching {domain.upper()} domain")
        logger.info(f"{'='*60}")
        
        domain_output = os.path.join(args.output, domain)
        
        if domain == 'climate':
            results = fetch_climate_domain(domain_output, args.start, args.end)
        elif domain == 'forest':
            results = fetch_forest_domain(domain_output, args.start, args.end)
        elif domain == 'industrial':
            results = fetch_industrial_domain(domain_output, args.start, args.end)
        else:
            continue
        
        generate_indicators_yaml(domain, results, domain_output)
        all_results[domain] = results
        
        logger.info(f"Fetched {len(results)} indicators for {domain}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for domain, results in all_results.items():
        print(f"\n{domain.upper()}:")
        for ind_id, path in results.items():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"  {ind_id}: {len(df)} observations ({df.index.min().date()} to {df.index.max().date()})")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("""
1. Set FRED API key if not already:
   export FRED_API_KEY=your_key_here

2. Import to PRISM database:
   python scripts/run_data_phase.py --config data/climate/climate_indicators.yaml
   python scripts/run_data_phase.py --config data/forest/forest_indicators.yaml
   python scripts/run_data_phase.py --config data/industrial/industrial_indicators.yaml

3. Run derived phase:
   python scripts/run_derived_phase.py --domain climate
   python scripts/run_derived_phase.py --domain forest
   python scripts/run_derived_phase.py --domain industrial

4. Run structure phase:
   python prism_system_geometry.py --domain climate
   python prism_state_vector.py --domain climate --detect-hidden-mass
""")


if __name__ == "__main__":
    main()
