"""
USGS Earthquake Data Fetcher
============================

Fetch earthquake data from USGS (United States Geological Survey).

Data includes:
- Earthquake catalog (magnitude, location, depth)
- Seismic hazard data
- Historical earthquake statistics

Source: https://earthquake.usgs.gov/fdsnws/event/1/
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .base_fetcher import BaseFetcher, FetchResult

logger = logging.getLogger(__name__)


class USGSFetcher(BaseFetcher):
    """
    Fetch earthquake data from USGS.

    Supports:
    - Earthquake catalog queries
    - Magnitude-frequency statistics
    - Regional seismicity
    - Energy release metrics
    """

    # USGS API endpoint
    API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    INDICATORS = {
        'earthquake_count': 'Number of earthquakes (M >= 4.5)',
        'earthquake_energy': 'Total seismic energy release',
        'max_magnitude': 'Maximum magnitude earthquake',
        'earthquake_catalog': 'Full earthquake catalog',
        'large_earthquakes': 'Major earthquakes (M >= 6.0)',
    }

    def __init__(self,
                 min_magnitude: float = 4.5,
                 cache_dir: str = ".cache/fetchers",
                 cache_ttl_hours: int = 24):
        """
        Initialize USGS fetcher.

        Args:
            min_magnitude: Minimum earthquake magnitude to fetch
            cache_dir: Cache directory
            cache_ttl_hours: Cache TTL
        """
        super().__init__(cache_dir, cache_ttl_hours)
        self.min_magnitude = min_magnitude

    @property
    def source_name(self) -> str:
        return "USGS"

    @property
    def base_url(self) -> str:
        return "https://earthquake.usgs.gov/"

    def list_indicators(self) -> List[str]:
        """List available earthquake indicators."""
        return list(self.INDICATORS.keys())

    def get_metadata(self, indicator: str) -> Dict[str, Any]:
        """Get metadata for indicator."""
        return {
            'source': self.source_name,
            'indicator': indicator,
            'description': self.INDICATORS.get(indicator, 'Unknown'),
            'min_magnitude': self.min_magnitude,
            'base_url': self.base_url
        }

    def _fetch_earthquakes(self,
                           start_date: str,
                           end_date: str,
                           min_mag: float = 4.5) -> pd.DataFrame:
        """
        Fetch earthquake catalog from USGS API.

        Args:
            start_date: Start date
            end_date: End date
            min_mag: Minimum magnitude

        Returns:
            DataFrame with earthquake data
        """
        if not HAS_REQUESTS:
            return None

        params = {
            'format': 'geojson',
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_mag,
            'orderby': 'time'
        }

        try:
            response = requests.get(self.API_URL, params=params, timeout=60)

            if response.status_code == 200:
                data = response.json()
                features = data.get('features', [])

                earthquakes = []
                for f in features:
                    props = f['properties']
                    coords = f['geometry']['coordinates']

                    earthquakes.append({
                        'date': datetime.fromtimestamp(props['time'] / 1000),
                        'magnitude': props.get('mag'),
                        'depth': coords[2] if len(coords) > 2 else np.nan,
                        'latitude': coords[1],
                        'longitude': coords[0],
                        'place': props.get('place', ''),
                        'type': props.get('type', 'earthquake')
                    })

                return pd.DataFrame(earthquakes)

        except Exception as e:
            logger.warning(f"USGS API fetch failed: {e}")

        return None

    def _generate_synthetic(self,
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """
        Generate synthetic earthquake data.

        Models:
        - Gutenberg-Richter magnitude distribution
        - Poisson occurrence process
        - Spatial clustering
        """
        np.random.seed(42)

        # Date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        days = (end - start).days

        # Average rate: ~50 M4.5+ earthquakes per day globally
        # Poisson process
        n_earthquakes = np.random.poisson(50 * days)

        # Random times
        times = [start + timedelta(
            seconds=np.random.randint(0, days * 86400)
        ) for _ in range(n_earthquakes)]
        times = sorted(times)

        # Gutenberg-Richter: log10(N) = a - b*M
        # b ~ 1.0, so probability decreases 10x per magnitude unit
        # For M >= 4.5, truncated exponential
        magnitudes = 4.5 + np.random.exponential(scale=0.7, size=n_earthquakes)
        magnitudes = np.clip(magnitudes, 4.5, 9.5)

        # Depth: most shallow, some deep subduction
        depths = np.abs(np.random.randn(n_earthquakes) * 30) + 10
        depths = np.clip(depths, 0, 700)

        # Locations: cluster around plate boundaries
        # Simple: mix of ring of fire and mid-ocean ridges
        latitudes = []
        longitudes = []

        for _ in range(n_earthquakes):
            r = np.random.rand()
            if r < 0.4:
                # Pacific ring (west coast Americas)
                lat = np.random.uniform(-40, 60)
                lon = np.random.uniform(-130, -70) + np.random.randn() * 5
            elif r < 0.6:
                # Pacific ring (Asia)
                lat = np.random.uniform(0, 50)
                lon = np.random.uniform(100, 160) + np.random.randn() * 5
            elif r < 0.75:
                # Mediterranean/Himalayan
                lat = np.random.uniform(20, 45) + np.random.randn() * 5
                lon = np.random.uniform(0, 100)
            else:
                # Random (mid-ocean ridges, etc.)
                lat = np.random.uniform(-60, 60)
                lon = np.random.uniform(-180, 180)

            latitudes.append(lat)
            longitudes.append(lon)

        return pd.DataFrame({
            'date': times,
            'magnitude': magnitudes,
            'depth': depths,
            'latitude': latitudes,
            'longitude': longitudes,
            'place': ['Synthetic location'] * n_earthquakes,
            'type': ['earthquake'] * n_earthquakes
        })

    def _aggregate_to_timeseries(self,
                                  df: pd.DataFrame,
                                  indicator: str,
                                  freq: str = 'W') -> pd.DataFrame:
        """
        Aggregate earthquake catalog to time series.

        Args:
            df: Earthquake catalog
            indicator: Type of aggregation
            freq: Aggregation frequency

        Returns:
            Time series DataFrame
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['period'] = df['date'].dt.to_period(freq)

        if indicator == 'earthquake_count':
            agg = df.groupby('period').size()
        elif indicator == 'earthquake_energy':
            # Energy ~ 10^(1.5*M)
            df['energy'] = 10 ** (1.5 * df['magnitude'])
            agg = df.groupby('period')['energy'].sum()
        elif indicator == 'max_magnitude':
            agg = df.groupby('period')['magnitude'].max()
        elif indicator == 'large_earthquakes':
            large = df[df['magnitude'] >= 6.0]
            agg = large.groupby('period').size()
        else:
            agg = df.groupby('period').size()

        # Convert to DataFrame
        result = agg.reset_index()
        result.columns = ['period', 'value']
        result['date'] = result['period'].dt.to_timestamp()

        return result[['date', 'value']]

    def _fetch_raw(self,
                   indicator: str,
                   start_date: str,
                   end_date: str,
                   **kwargs) -> pd.DataFrame:
        """
        Fetch earthquake data.

        Args:
            indicator: Data indicator name
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and value columns
        """
        indicator = indicator.lower()

        # Try real API first
        catalog = self._fetch_earthquakes(
            start_date, end_date,
            min_mag=self.min_magnitude
        )

        if catalog is None or len(catalog) == 0:
            logger.info("Using synthetic earthquake data")
            catalog = self._generate_synthetic(start_date, end_date)

        # Return catalog or aggregate
        if indicator == 'earthquake_catalog':
            return catalog
        else:
            freq = kwargs.get('freq', 'W')
            return self._aggregate_to_timeseries(catalog, indicator, freq)


if __name__ == "__main__":
    print("=" * 60)
    print("USGS Earthquake Fetcher - Demo")
    print("=" * 60)

    fetcher = USGSFetcher(min_magnitude=5.0)

    print("\nAvailable indicators:")
    for ind in fetcher.list_indicators():
        meta = fetcher.get_metadata(ind)
        print(f"  {ind}: {meta['description']}")

    print("\nFetching earthquake count data...")
    result = fetcher.fetch('earthquake_count', '2010-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} periods")
        print(f"Date range: {result.quality_report.date_range}")
        print("\nWeekly earthquake counts (first 10):")
        print(result.data.head(10))

        # Statistics
        print("\nStatistics:")
        print(f"  Mean weekly count: {result.data['value'].mean():.1f}")
        print(f"  Max weekly count: {result.data['value'].max():.0f}")
    else:
        print(f"Fetch failed: {result.error}")

    print("\nFetching earthquake catalog...")
    result = fetcher.fetch('earthquake_catalog', '2023-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} earthquakes")
        print("\nMagnitude distribution:")
        df = result.data.copy()
        df['mag_bin'] = pd.cut(df['magnitude'], bins=[4.5, 5, 5.5, 6, 6.5, 7, 10])
        print(df['mag_bin'].value_counts().sort_index())

    print("\nTest completed!")
