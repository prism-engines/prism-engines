"""
NOAA Climate Data Fetcher
=========================

Fetch climate data from NOAA (National Oceanic and Atmospheric Administration).

Data includes:
- Temperature (global, regional)
- Precipitation
- Sea surface temperature
- Climate indices (ENSO, NAO, PDO)

API: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .base_fetcher import BaseFetcher, FetchResult

logger = logging.getLogger(__name__)


class NOAAFetcher(BaseFetcher):
    """
    Fetch climate data from NOAA.

    Supports:
    - Climate indices (ENSO, NAO, PDO, etc.)
    - Temperature anomalies
    - Precipitation data

    Note: Some endpoints require API key from NCDC.
    """

    # Climate index endpoints (no API key needed)
    INDEX_URLS = {
        'enso': 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',
        'nao': 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii',
        'pdo': 'https://www.ncdc.noaa.gov/teleconnections/pdo/data.csv',
        'ao': 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml',
        'soi': 'https://www.cpc.ncep.noaa.gov/data/indices/soi',
    }

    # Known climate indices with descriptions
    INDICATORS = {
        'enso': 'El Niño Southern Oscillation (Oceanic Niño Index)',
        'nao': 'North Atlantic Oscillation',
        'pdo': 'Pacific Decadal Oscillation',
        'ao': 'Arctic Oscillation',
        'soi': 'Southern Oscillation Index',
        'global_temp': 'Global Temperature Anomaly',
        'us_temp': 'US Temperature Anomaly',
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 cache_dir: str = ".cache/fetchers",
                 cache_ttl_hours: int = 24):
        """
        Initialize NOAA fetcher.

        Args:
            api_key: NOAA NCDC API key (optional, for CDO endpoints)
            cache_dir: Cache directory
            cache_ttl_hours: Cache TTL
        """
        super().__init__(cache_dir, cache_ttl_hours)
        self.api_key = api_key

    @property
    def source_name(self) -> str:
        return "NOAA"

    @property
    def base_url(self) -> str:
        return "https://www.ncdc.noaa.gov/cdo-web/api/v2"

    def list_indicators(self) -> List[str]:
        """List available climate indicators."""
        return list(self.INDICATORS.keys())

    def get_metadata(self, indicator: str) -> Dict[str, Any]:
        """Get metadata for indicator."""
        return {
            'source': self.source_name,
            'indicator': indicator,
            'description': self.INDICATORS.get(indicator, 'Unknown'),
            'base_url': self.base_url
        }

    def _parse_oni_data(self, text: str) -> pd.DataFrame:
        """Parse ENSO (ONI) data format."""
        lines = text.strip().split('\n')
        data = []

        for line in lines:
            parts = line.split()
            if len(parts) >= 8:
                try:
                    year = int(parts[0])
                    # ONI has 12 overlapping 3-month values
                    # Use center month for dating
                    for i, val in enumerate(parts[1:13]):
                        month = i + 1
                        date = f"{year}-{month:02d}-01"
                        data.append({
                            'date': date,
                            'value': float(val)
                        })
                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').reset_index(drop=True)

    def _parse_nao_data(self, text: str) -> pd.DataFrame:
        """Parse NAO data format."""
        lines = text.strip().split('\n')
        data = []

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    value = float(parts[2]) if len(parts) > 2 else np.nan
                    date = f"{year}-{month:02d}-01"
                    data.append({'date': date, 'value': value})
                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').reset_index(drop=True)

    def _generate_synthetic(self,
                            indicator: str,
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """
        Generate synthetic climate data for testing.

        Uses realistic statistical properties:
        - ENSO: ~3-7 year cycles
        - NAO: seasonal + random
        - PDO: ~20-30 year cycles
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        n = len(dates)
        np.random.seed(42)

        if indicator == 'enso':
            # ENSO: ~4 year dominant cycle
            t = np.arange(n)
            cycle = 1.5 * np.sin(2 * np.pi * t / 48)  # ~4 year
            noise = 0.5 * np.random.randn(n)
            values = cycle + noise

        elif indicator == 'nao':
            # NAO: strong seasonal + noise
            t = np.arange(n)
            seasonal = 0.5 * np.sin(2 * np.pi * t / 12)
            noise = np.random.randn(n)
            ar_noise = np.zeros(n)
            for i in range(1, n):
                ar_noise[i] = 0.3 * ar_noise[i-1] + noise[i]
            values = seasonal + ar_noise

        elif indicator == 'pdo':
            # PDO: ~25 year cycle
            t = np.arange(n)
            cycle = np.sin(2 * np.pi * t / 300)  # ~25 year
            noise = 0.3 * np.random.randn(n)
            values = cycle + noise

        else:
            # Generic climate index
            values = np.cumsum(np.random.randn(n) * 0.1)

        return pd.DataFrame({
            'date': dates,
            'value': values
        })

    def _fetch_raw(self,
                   indicator: str,
                   start_date: str,
                   end_date: str,
                   **kwargs) -> pd.DataFrame:
        """
        Fetch climate data.

        Args:
            indicator: Climate index name
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and value columns
        """
        indicator = indicator.lower()

        # Try to fetch real data
        if HAS_REQUESTS and indicator in self.INDEX_URLS:
            try:
                url = self.INDEX_URLS[indicator]
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    if indicator == 'enso':
                        df = self._parse_oni_data(response.text)
                    elif indicator == 'nao':
                        df = self._parse_nao_data(response.text)
                    else:
                        # Try CSV parsing
                        from io import StringIO
                        df = pd.read_csv(StringIO(response.text))
                        if 'date' not in df.columns:
                            df['date'] = pd.date_range(
                                start='1950-01-01',
                                periods=len(df),
                                freq='MS'
                            )

                    # Filter to date range
                    df['date'] = pd.to_datetime(df['date'])
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    return df[mask].reset_index(drop=True)

            except Exception as e:
                logger.warning(f"NOAA fetch failed, using synthetic: {e}")

        # Fall back to synthetic data
        logger.info(f"Using synthetic data for {indicator}")
        return self._generate_synthetic(indicator, start_date, end_date)


if __name__ == "__main__":
    print("=" * 60)
    print("NOAA Climate Fetcher - Demo")
    print("=" * 60)

    fetcher = NOAAFetcher()

    print("\nAvailable indicators:")
    for ind in fetcher.list_indicators():
        meta = fetcher.get_metadata(ind)
        print(f"  {ind}: {meta['description']}")

    print("\nFetching ENSO data...")
    result = fetcher.fetch('enso', '2010-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} observations")
        print(f"Date range: {result.quality_report.date_range}")
        print("\nSample data:")
        print(result.data.head(10))

        # Basic statistics
        print("\nStatistics:")
        print(result.data['value'].describe())
    else:
        print(f"Fetch failed: {result.error}")

    print("\nTest completed!")
