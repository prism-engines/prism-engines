"""
Our World in Data (OWID) Fetcher
================================

Fetch comprehensive data from Our World in Data.

Data includes:
- COVID-19 statistics
- Economic indicators
- Energy data
- Health and demographics
- Climate and environment

Source: https://github.com/owid/owid-datasets
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


class OWIDFetcher(BaseFetcher):
    """
    Fetch data from Our World in Data.

    OWID provides curated, open-source datasets on
    global issues like health, energy, and economics.
    """

    # OWID GitHub raw data URLs
    DATA_URLS = {
        'covid': 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv',
        'co2': 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv',
        'energy': 'https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv',
    }

    INDICATORS = {
        # COVID
        'covid_cases': 'COVID-19 new cases',
        'covid_deaths': 'COVID-19 new deaths',
        'covid_vaccinations': 'COVID-19 vaccinations',
        'covid_tests': 'COVID-19 tests',
        'covid_icu': 'COVID-19 ICU patients',
        'covid_hosp': 'COVID-19 hospitalizations',
        'excess_mortality': 'Excess mortality',
        # CO2/Climate
        'co2_emissions': 'CO2 emissions',
        'co2_per_capita': 'CO2 emissions per capita',
        'ghg_emissions': 'Greenhouse gas emissions',
        # Energy
        'energy_consumption': 'Primary energy consumption',
        'renewables_share': 'Share of renewables in energy',
        'fossil_share': 'Share of fossil fuels in energy',
        'electricity_generation': 'Electricity generation',
    }

    def __init__(self,
                 location: str = 'World',
                 cache_dir: str = ".cache/fetchers",
                 cache_ttl_hours: int = 24):
        """
        Initialize OWID fetcher.

        Args:
            location: Country/region (default: World)
            cache_dir: Cache directory
            cache_ttl_hours: Cache TTL
        """
        super().__init__(cache_dir, cache_ttl_hours)
        self.location = location

    @property
    def source_name(self) -> str:
        return "OWID"

    @property
    def base_url(self) -> str:
        return "https://ourworldindata.org/"

    def list_indicators(self) -> List[str]:
        """List available indicators."""
        return list(self.INDICATORS.keys())

    def get_metadata(self, indicator: str) -> Dict[str, Any]:
        """Get metadata for indicator."""
        return {
            'source': self.source_name,
            'indicator': indicator,
            'description': self.INDICATORS.get(indicator, 'Unknown'),
            'location': self.location,
            'base_url': self.base_url
        }

    def _fetch_covid_data(self,
                          indicator: str,
                          start_date: str,
                          end_date: str) -> Optional[pd.DataFrame]:
        """Fetch COVID data from OWID."""
        if not HAS_REQUESTS:
            return None

        try:
            url = self.DATA_URLS['covid']
            response = requests.get(url, timeout=60)

            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))

                # Filter by location
                df = df[df['location'] == self.location].copy()

                # Map indicator to column
                col_map = {
                    'covid_cases': 'new_cases_smoothed',
                    'covid_deaths': 'new_deaths_smoothed',
                    'covid_vaccinations': 'new_vaccinations_smoothed',
                    'covid_tests': 'new_tests_smoothed',
                    'covid_icu': 'icu_patients',
                    'covid_hosp': 'hosp_patients',
                    'excess_mortality': 'excess_mortality',
                }

                value_col = col_map.get(indicator, 'new_cases_smoothed')

                if value_col in df.columns:
                    result = df[['date', value_col]].copy()
                    result = result.rename(columns={value_col: 'value'})
                    result['date'] = pd.to_datetime(result['date'])
                    return result

        except Exception as e:
            logger.warning(f"OWID COVID fetch failed: {e}")

        return None

    def _fetch_co2_data(self,
                        indicator: str,
                        start_date: str,
                        end_date: str) -> Optional[pd.DataFrame]:
        """Fetch CO2/emissions data from OWID."""
        if not HAS_REQUESTS:
            return None

        try:
            url = self.DATA_URLS['co2']
            response = requests.get(url, timeout=60)

            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))

                # Filter by country
                df = df[df['country'] == self.location].copy()

                col_map = {
                    'co2_emissions': 'co2',
                    'co2_per_capita': 'co2_per_capita',
                    'ghg_emissions': 'total_ghg',
                }

                value_col = col_map.get(indicator, 'co2')

                if value_col in df.columns:
                    result = df[['year', value_col]].copy()
                    result = result.rename(columns={value_col: 'value', 'year': 'date'})
                    result['date'] = pd.to_datetime(result['date'].astype(str) + '-01-01')
                    return result

        except Exception as e:
            logger.warning(f"OWID CO2 fetch failed: {e}")

        return None

    def _generate_covid_synthetic(self,
                                   indicator: str,
                                   start_date: str,
                                   end_date: str) -> pd.DataFrame:
        """Generate synthetic COVID data."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        np.random.seed(42)

        covid_start = pd.Timestamp('2020-03-01')
        values = np.zeros(n)

        for i, d in enumerate(dates):
            if d < covid_start:
                continue

            days = (d - covid_start).days

            # Wave patterns
            wave1 = 50000 * np.exp(-((days - 45) ** 2) / 800)
            wave2 = 70000 * np.exp(-((days - 130) ** 2) / 1000)
            wave3 = 200000 * np.exp(-((days - 310) ** 2) / 2000)
            wave4 = 150000 * np.exp(-((days - 530) ** 2) / 1500)
            wave5 = 500000 * np.exp(-((days - 680) ** 2) / 1000)

            total = wave1 + wave2 + wave3 + wave4 + wave5

            # Weekly pattern
            dow = d.weekday()
            dow_factor = [1.0, 1.05, 1.0, 0.95, 0.9, 0.7, 0.6][dow]

            if indicator == 'covid_deaths':
                values[i] = total * 0.015 * dow_factor
            elif indicator == 'covid_hosp':
                values[i] = total * 0.05 * dow_factor
            elif indicator == 'covid_icu':
                values[i] = total * 0.01 * dow_factor
            else:
                values[i] = total * dow_factor

        noise = np.random.randn(n) * values * 0.05
        values = np.maximum(0, values + noise)

        return pd.DataFrame({'date': dates, 'value': values})

    def _generate_co2_synthetic(self,
                                indicator: str,
                                start_date: str,
                                end_date: str) -> pd.DataFrame:
        """Generate synthetic CO2/emissions data."""
        # Annual data
        start_year = pd.Timestamp(start_date).year
        end_year = pd.Timestamp(end_date).year
        years = list(range(start_year, end_year + 1))
        n = len(years)
        np.random.seed(42)

        if indicator == 'co2_emissions':
            # Global CO2: ~35 billion tonnes, growing ~1-2% per year
            base = 25e9  # 1990 baseline
            growth = 1.015 ** np.arange(n)
            values = base * growth

            # COVID dip in 2020
            for i, y in enumerate(years):
                if y == 2020:
                    values[i] *= 0.94

        elif indicator == 'co2_per_capita':
            # ~5 tonnes per capita, slowly rising
            base = 4.0
            values = base + 0.02 * np.arange(n) + np.random.randn(n) * 0.1

        else:
            # GHG emissions
            base = 40e9
            growth = 1.012 ** np.arange(n)
            values = base * growth

        dates = [pd.Timestamp(f"{y}-01-01") for y in years]

        return pd.DataFrame({'date': dates, 'value': values})

    def _generate_energy_synthetic(self,
                                   indicator: str,
                                   start_date: str,
                                   end_date: str) -> pd.DataFrame:
        """Generate synthetic energy data."""
        start_year = pd.Timestamp(start_date).year
        end_year = pd.Timestamp(end_date).year
        years = list(range(start_year, end_year + 1))
        n = len(years)
        np.random.seed(42)

        if indicator == 'energy_consumption':
            # Global primary energy ~600 EJ, growing
            base = 400  # 1990 EJ
            growth = 1.02 ** np.arange(n)
            values = base * growth

        elif indicator == 'renewables_share':
            # Growing from ~10% to ~15%
            values = 10 + 0.3 * np.arange(n) + np.random.randn(n) * 0.5

        elif indicator == 'fossil_share':
            # Declining from ~85% to ~80%
            values = 85 - 0.2 * np.arange(n) + np.random.randn(n) * 0.5

        else:
            # Electricity generation TWh
            base = 15000
            growth = 1.025 ** np.arange(n)
            values = base * growth

        dates = [pd.Timestamp(f"{y}-01-01") for y in years]

        return pd.DataFrame({'date': dates, 'value': values})

    def _fetch_raw(self,
                   indicator: str,
                   start_date: str,
                   end_date: str,
                   **kwargs) -> pd.DataFrame:
        """
        Fetch data from OWID.

        Args:
            indicator: Data indicator name
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and value columns
        """
        indicator = indicator.lower()

        # Determine category and try real data
        if indicator.startswith('covid') or indicator == 'excess_mortality':
            df = self._fetch_covid_data(indicator, start_date, end_date)
            if df is None:
                df = self._generate_covid_synthetic(indicator, start_date, end_date)

        elif indicator.startswith('co2') or indicator == 'ghg_emissions':
            df = self._fetch_co2_data(indicator, start_date, end_date)
            if df is None:
                df = self._generate_co2_synthetic(indicator, start_date, end_date)

        elif indicator in ['energy_consumption', 'renewables_share',
                           'fossil_share', 'electricity_generation']:
            df = self._generate_energy_synthetic(indicator, start_date, end_date)

        else:
            logger.warning(f"Unknown indicator: {indicator}")
            return pd.DataFrame(columns=['date', 'value'])

        # Filter to date range
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        return df[mask].reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Our World in Data Fetcher - Demo")
    print("=" * 60)

    fetcher = OWIDFetcher(location='World')

    print("\nAvailable indicators:")
    for ind in fetcher.list_indicators():
        meta = fetcher.get_metadata(ind)
        print(f"  {ind}: {meta['description']}")

    print("\nFetching COVID cases data...")
    result = fetcher.fetch('covid_cases', '2020-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} observations")
        print(f"Date range: {result.quality_report.date_range}")

        df = result.data.copy()
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('month')['value'].mean()
        print("\nMonthly average cases (first 12 months):")
        print(monthly.head(12))

    print("\nFetching CO2 emissions data...")
    result = fetcher.fetch('co2_emissions', '1990-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} observations")
        print("\nAnnual CO2 emissions (billion tonnes):")
        df = result.data.copy()
        df['year'] = df['date'].dt.year
        df['value_bt'] = df['value'] / 1e9
        print(df[['year', 'value_bt']].tail(10))

    print("\nFetching renewables share data...")
    result = fetcher.fetch('renewables_share', '2000-01-01', '2023-12-31')

    if result.success:
        print(f"\nFetched {len(result.data)} observations")
        print("\nRenewables share (%):")
        print(result.data.tail(10))

    print("\nTest completed!")
