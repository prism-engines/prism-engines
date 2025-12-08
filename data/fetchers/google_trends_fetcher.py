"""
Google Trends Data Fetcher
==========================

Fetch search interest data from Google Trends.

Data includes:
- Search interest over time
- Related queries
- Geographic interest

Note: Uses pytrends library if available, otherwise synthetic data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False

from .base_fetcher import BaseFetcher, FetchResult

logger = logging.getLogger(__name__)


class GoogleTrendsFetcher(BaseFetcher):
    """
    Fetch search interest data from Google Trends.

    Supports:
    - Interest over time for keywords
    - Comparison of multiple terms
    - Geographic breakdown
    - Related queries

    Note: Google Trends data is normalized 0-100 relative to
    the peak value in the time range.
    """

    # Pre-defined topics for testing cross-domain coherence
    PREDEFINED_TOPICS = {
        'recession': 'Economic recession search interest',
        'unemployment': 'Unemployment search interest',
        'stock_market': 'Stock market search interest',
        'inflation': 'Inflation search interest',
        'bitcoin': 'Bitcoin search interest',
        'covid': 'COVID-19 search interest',
        'flu': 'Flu/influenza search interest',
        'climate_change': 'Climate change search interest',
        'hurricane': 'Hurricane search interest',
        'earthquake': 'Earthquake search interest',
    }

    def __init__(self,
                 geo: str = 'US',
                 cache_dir: str = ".cache/fetchers",
                 cache_ttl_hours: int = 24):
        """
        Initialize Google Trends fetcher.

        Args:
            geo: Geographic region (default: US)
            cache_dir: Cache directory
            cache_ttl_hours: Cache TTL
        """
        super().__init__(cache_dir, cache_ttl_hours)
        self.geo = geo

        if HAS_PYTRENDS:
            self.pytrends = TrendReq(hl='en-US', tz=360)
        else:
            self.pytrends = None

    @property
    def source_name(self) -> str:
        return "GoogleTrends"

    @property
    def base_url(self) -> str:
        return "https://trends.google.com/trends/"

    def list_indicators(self) -> List[str]:
        """List predefined topic indicators."""
        return list(self.PREDEFINED_TOPICS.keys())

    def get_metadata(self, indicator: str) -> Dict[str, Any]:
        """Get metadata for indicator."""
        return {
            'source': self.source_name,
            'indicator': indicator,
            'description': self.PREDEFINED_TOPICS.get(indicator, 'Custom search term'),
            'geo': self.geo,
            'base_url': self.base_url,
            'note': 'Values are normalized 0-100 relative to peak in time range'
        }

    def _keyword_for_topic(self, topic: str) -> str:
        """Map topic to Google Trends keyword."""
        keyword_map = {
            'recession': 'recession',
            'unemployment': 'unemployment',
            'stock_market': 'stock market',
            'inflation': 'inflation',
            'bitcoin': 'bitcoin',
            'covid': 'covid',
            'flu': 'flu symptoms',
            'climate_change': 'climate change',
            'hurricane': 'hurricane',
            'earthquake': 'earthquake',
        }
        return keyword_map.get(topic.lower(), topic)

    def _fetch_real_data(self,
                         keyword: str,
                         start_date: str,
                         end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch real data from Google Trends.

        Args:
            keyword: Search term
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame or None if fetch fails
        """
        if not HAS_PYTRENDS or self.pytrends is None:
            return None

        try:
            timeframe = f"{start_date} {end_date}"
            self.pytrends.build_payload(
                kw_list=[keyword],
                cat=0,
                timeframe=timeframe,
                geo=self.geo
            )

            df = self.pytrends.interest_over_time()

            if df is not None and len(df) > 0:
                df = df.reset_index()
                df = df.rename(columns={'date': 'date', keyword: 'value'})
                return df[['date', 'value']]

        except Exception as e:
            logger.warning(f"Google Trends fetch failed: {e}")

        return None

    def _generate_synthetic(self,
                            indicator: str,
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """
        Generate synthetic search interest data.

        Models different patterns for different topics.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
        n = len(dates)
        np.random.seed(hash(indicator) % 2**32)

        t = np.arange(n)

        if indicator in ['recession', 'unemployment']:
            # Spikes during recessions
            # 2008-2009 GFC, 2020 COVID
            base = 20 + 5 * np.sin(2 * np.pi * t / 52)

            # GFC spike
            gfc_start = pd.Timestamp('2008-09-01')
            gfc_peak = pd.Timestamp('2009-03-01')

            # COVID spike
            covid_start = pd.Timestamp('2020-03-01')

            values = base.copy()
            for i, d in enumerate(dates):
                if gfc_start <= d <= gfc_start + pd.Timedelta(days=365):
                    days_in = (d - gfc_start).days
                    values[i] = base[i] + 60 * np.exp(-((days_in - 180) ** 2) / 10000)
                elif covid_start <= d <= covid_start + pd.Timedelta(days=180):
                    days_in = (d - covid_start).days
                    values[i] = base[i] + 70 * np.exp(-((days_in - 30) ** 2) / 2000)

        elif indicator == 'bitcoin':
            # Bubble patterns: 2017, 2021
            base = 10 + np.random.randn(n) * 3

            for i, d in enumerate(dates):
                year = d.year + d.month / 12

                # 2017 bubble
                if 2017 <= year <= 2018:
                    values = base[i] + 80 * np.exp(-((year - 2017.95) ** 2) * 50)
                # 2021 bubble
                elif 2020.8 <= year <= 2022:
                    values = base[i] + 70 * np.exp(-((year - 2021.3) ** 2) * 20)
                else:
                    values = base[i]
                base[i] = values

            values = base

        elif indicator == 'covid':
            # Massive spike in 2020
            values = np.full(n, 5.0)

            covid_start = pd.Timestamp('2020-01-15')
            for i, d in enumerate(dates):
                if d >= covid_start:
                    days_in = (d - covid_start).days
                    # Peak around March 2020, then gradual decline
                    peak_val = 100 * np.exp(-((days_in - 60) ** 2) / 3000)
                    decay_val = 30 * np.exp(-days_in / 500)
                    values[i] = max(5, peak_val + decay_val)

        elif indicator == 'flu':
            # Strong seasonal pattern
            week = dates.isocalendar().week.values
            values = 30 + 50 * np.exp(-((week - 5) ** 2) / 30)
            values = np.where(week > 20, 10 + np.random.randn(n) * 3, values)
            values = np.clip(values, 0, 100)

        elif indicator in ['hurricane', 'earthquake']:
            # Spike pattern around events
            base = 10 + np.random.randn(n) * 5

            # Random spikes (simulating major events)
            n_events = n // 52  # ~1 major event per year
            event_indices = np.random.choice(n, n_events, replace=False)

            for idx in event_indices:
                spike_mag = np.random.uniform(40, 90)
                for offset in range(-2, 4):
                    if 0 <= idx + offset < n:
                        base[idx + offset] += spike_mag * np.exp(-abs(offset))

            values = np.clip(base, 0, 100)

        else:
            # Generic trend + noise
            trend = 30 + 20 * np.sin(2 * np.pi * t / 156)  # ~3 year cycle
            noise = 10 * np.random.randn(n)
            values = np.clip(trend + noise, 0, 100)

        # Normalize to 0-100
        values = 100 * (values - values.min()) / (values.max() - values.min() + 1e-10)

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
        Fetch search interest data.

        Args:
            indicator: Topic or search term
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and value columns
        """
        keyword = self._keyword_for_topic(indicator)

        # Try real data first
        df = self._fetch_real_data(keyword, start_date, end_date)

        if df is None or len(df) == 0:
            logger.info(f"Using synthetic data for {indicator}")
            df = self._generate_synthetic(indicator, start_date, end_date)

        # Filter to date range
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        return df[mask].reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Google Trends Fetcher - Demo")
    print("=" * 60)

    fetcher = GoogleTrendsFetcher()

    print(f"\nPytrends available: {HAS_PYTRENDS}")

    print("\nAvailable topics:")
    for ind in fetcher.list_indicators():
        meta = fetcher.get_metadata(ind)
        print(f"  {ind}: {meta['description']}")

    print("\nFetching 'recession' interest...")
    result = fetcher.fetch('recession', '2005-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} observations")
        print(f"Date range: {result.quality_report.date_range}")

        # Find peaks
        df = result.data.copy()
        df_sorted = df.sort_values('value', ascending=False)
        print("\nTop 10 highest interest periods:")
        print(df_sorted.head(10))

    print("\nFetching 'covid' interest...")
    result = fetcher.fetch('covid', '2019-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} observations")
        df = result.data.copy()
        print("\nCOVID interest peaks:")
        print(df.sort_values('value', ascending=False).head(5))

    print("\nTest completed!")
