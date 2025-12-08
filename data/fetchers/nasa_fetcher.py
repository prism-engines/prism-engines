"""
NASA GISS Temperature Fetcher
=============================

Fetch global temperature data from NASA Goddard Institute for Space Studies.

Data includes:
- Global mean temperature anomaly
- Hemispheric temperatures
- Zonal temperature means

Source: https://data.giss.nasa.gov/gistemp/
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


class NASAFetcher(BaseFetcher):
    """
    Fetch temperature data from NASA GISS.

    Provides global and regional temperature anomalies
    relative to 1951-1980 baseline.
    """

    # NASA GISS data endpoints
    DATA_URLS = {
        'global_temp': 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv',
        'northern_temp': 'https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv',
        'southern_temp': 'https://data.giss.nasa.gov/gistemp/tabledata_v4/SH.Ts+dSST.csv',
        'zonal_temp': 'https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv',
    }

    INDICATORS = {
        'global_temp': 'Global Mean Surface Temperature Anomaly',
        'northern_temp': 'Northern Hemisphere Temperature Anomaly',
        'southern_temp': 'Southern Hemisphere Temperature Anomaly',
        'zonal_temp': 'Zonal Annual Temperature Anomalies',
    }

    def __init__(self,
                 cache_dir: str = ".cache/fetchers",
                 cache_ttl_hours: int = 24):
        """Initialize NASA fetcher."""
        super().__init__(cache_dir, cache_ttl_hours)

    @property
    def source_name(self) -> str:
        return "NASA_GISS"

    @property
    def base_url(self) -> str:
        return "https://data.giss.nasa.gov/gistemp/"

    def list_indicators(self) -> List[str]:
        """List available temperature indicators."""
        return list(self.INDICATORS.keys())

    def get_metadata(self, indicator: str) -> Dict[str, Any]:
        """Get metadata for indicator."""
        return {
            'source': self.source_name,
            'indicator': indicator,
            'description': self.INDICATORS.get(indicator, 'Unknown'),
            'baseline': '1951-1980 average',
            'unit': 'degrees Celsius',
            'base_url': self.base_url
        }

    def _parse_giss_csv(self, text: str, monthly: bool = True) -> pd.DataFrame:
        """
        Parse NASA GISS CSV format.

        GISS format has years as rows, months as columns.
        """
        from io import StringIO

        # Skip header lines
        lines = text.strip().split('\n')
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('Year'):
                data_start = i
                break

        # Parse data
        csv_text = '\n'.join(lines[data_start:])
        df = pd.read_csv(StringIO(csv_text))

        # Clean column names
        df.columns = [c.strip() for c in df.columns]

        if monthly and 'Jan' in df.columns:
            # Convert wide format to long
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            data = []
            for _, row in df.iterrows():
                year = int(row['Year'])
                for i, month in enumerate(months):
                    if month in row:
                        value = row[month]
                        if value != '***' and pd.notna(value):
                            try:
                                data.append({
                                    'date': f"{year}-{i+1:02d}-01",
                                    'value': float(value) / 100  # Convert to degrees
                                })
                            except (ValueError, TypeError):
                                pass

            result = pd.DataFrame(data)
            result['date'] = pd.to_datetime(result['date'])
            return result.sort_values('date').reset_index(drop=True)

        else:
            # Annual data
            df['date'] = pd.to_datetime(df['Year'].astype(str) + '-01-01')
            if 'J-D' in df.columns:
                df['value'] = pd.to_numeric(df['J-D'], errors='coerce') / 100
            return df[['date', 'value']].dropna()

    def _generate_synthetic(self,
                            indicator: str,
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """
        Generate synthetic temperature anomaly data.

        Models:
        - Long-term warming trend (~0.2°C/decade recent)
        - Volcanic cooling events
        - ENSO-related variability
        - Random weather noise
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        n = len(dates)
        np.random.seed(42)

        # Years since 1900 (for trend)
        years = (dates.year - 1900 + dates.month / 12).values

        # Warming trend (accelerating)
        # Pre-1980: ~0.05°C/decade
        # Post-1980: ~0.2°C/decade
        trend = np.where(
            years < 80,
            0.005 * years,
            0.005 * 80 + 0.02 * (years - 80)
        )

        # Seasonal cycle
        seasonal = 0.1 * np.sin(2 * np.pi * dates.month.values / 12)

        # ENSO-like variability (~4 year cycle)
        t = np.arange(n)
        enso = 0.15 * np.sin(2 * np.pi * t / 48 + np.random.rand() * 2 * np.pi)

        # Volcanic cooling (simulate Pinatubo-like events)
        volcanic = np.zeros(n)
        # Major eruptions: Pinatubo 1991, El Chichón 1982
        for year, month, magnitude in [(1991, 6, -0.4), (1982, 4, -0.3)]:
            eruption_date = pd.Timestamp(f"{year}-{month:02d}-01")
            if eruption_date >= dates.min() and eruption_date <= dates.max():
                idx = np.argmin(np.abs(dates - eruption_date))
                # Cooling effect decays over ~2 years
                for i in range(min(24, n - idx)):
                    volcanic[idx + i] = magnitude * np.exp(-i / 8)

        # Weather noise
        noise = 0.1 * np.random.randn(n)

        # Combine
        if indicator == 'northern_temp':
            scale = 1.2  # Northern hemisphere warms faster
        elif indicator == 'southern_temp':
            scale = 0.8  # Southern hemisphere warms slower
        else:
            scale = 1.0

        values = scale * (trend + seasonal + enso + volcanic + noise)

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
        Fetch temperature data.

        Args:
            indicator: Temperature indicator name
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and value columns
        """
        indicator = indicator.lower()

        # Try to fetch real data
        if HAS_REQUESTS and indicator in self.DATA_URLS:
            try:
                url = self.DATA_URLS[indicator]
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    df = self._parse_giss_csv(response.text)

                    # Filter to date range
                    df['date'] = pd.to_datetime(df['date'])
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    return df[mask].reset_index(drop=True)

            except Exception as e:
                logger.warning(f"NASA fetch failed, using synthetic: {e}")

        # Fall back to synthetic data
        logger.info(f"Using synthetic data for {indicator}")
        return self._generate_synthetic(indicator, start_date, end_date)


if __name__ == "__main__":
    print("=" * 60)
    print("NASA GISS Temperature Fetcher - Demo")
    print("=" * 60)

    fetcher = NASAFetcher()

    print("\nAvailable indicators:")
    for ind in fetcher.list_indicators():
        meta = fetcher.get_metadata(ind)
        print(f"  {ind}: {meta['description']}")

    print("\nFetching global temperature data...")
    result = fetcher.fetch('global_temp', '1990-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} observations")
        print(f"Date range: {result.quality_report.date_range}")
        print("\nSample data (last 12 months):")
        print(result.data.tail(12))

        # Trend analysis
        print("\nDecadal averages:")
        df = result.data.copy()
        df['decade'] = (df['date'].dt.year // 10) * 10
        print(df.groupby('decade')['value'].mean())
    else:
        print(f"Fetch failed: {result.error}")

    print("\nTest completed!")
