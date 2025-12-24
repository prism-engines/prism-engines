"""
PRISM Epidemiology Fetcher (COVID-19)

Fetches COVID-19 data from Our World in Data and converts to PRISM-compatible time series.

DOMAIN TRANSLATION:
    Finance         ->  Epidemiology
    ─────────────────────────────────────────
    Indicator       ->  Country / Region
    Daily return    ->  Daily new cases or deaths
    Volatility      ->  Case count variance
    Correlation     ->  Cross-country synchronization
    Regime change   ->  Wave onset / variant emergence
    Hidden mass     ->  Latent spread before detection

INDICATORS GENERATED:
    For each region, we generate multiple time series:
    - {REGION}_CASES       : Daily new cases (smoothed)
    - {REGION}_DEATHS      : Daily new deaths (smoothed)
    - {REGION}_CASES_PC    : Cases per million population
    - {REGION}_DEATHS_PC   : Deaths per million population
    - {REGION}_HOSP        : Hospital patients (where available)
    - {REGION}_VAX         : Vaccination rate (% fully vaccinated)
    - {REGION}_POSITIVITY  : Test positivity rate (where available)
    - {REGION}_STRINGENCY  : Government stringency index (0-100)

REGIONS:
    Major countries/regions with good data quality:
    - USA, GBR, DEU, FRA, ITA, ESP (Western)
    - JPN, KOR, AUS, NZL (Pacific)
    - BRA, MEX, ARG (Americas)
    - IND, IDN (Asia)
    - ZAF (Africa)

Data Source:
    Our World in Data COVID-19 Dataset
    https://github.com/owid/covid-19-data

Usage:
    python -m prism.fetch.owid --start 2020-01-01 --end 2024-12-31

    Or import and use programmatically:

    from prism.fetch.owid import OWIDFetcher
    fetcher = OWIDFetcher()
    result = fetcher.fetch("USA_CASES", start_date=date(2020, 1, 1))
"""

import logging
import requests
import time
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from io import StringIO

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# REGION DEFINITIONS
# =============================================================================

@dataclass
class CovidRegion:
    """Definition of a COVID-19 monitoring region."""
    code: str           # ISO code or OWID code
    name: str           # Human readable name
    population: int     # Approximate population (for reference)
    owid_location: str  # Name as it appears in OWID data


# Major regions with good data quality
REGIONS: Dict[str, CovidRegion] = {
    # North America
    "USA": CovidRegion("USA", "United States", 331_000_000, "United States"),
    "CAN": CovidRegion("CAN", "Canada", 38_000_000, "Canada"),
    "MEX": CovidRegion("MEX", "Mexico", 128_000_000, "Mexico"),

    # Europe
    "GBR": CovidRegion("GBR", "United Kingdom", 67_000_000, "United Kingdom"),
    "DEU": CovidRegion("DEU", "Germany", 83_000_000, "Germany"),
    "FRA": CovidRegion("FRA", "France", 67_000_000, "France"),
    "ITA": CovidRegion("ITA", "Italy", 60_000_000, "Italy"),
    "ESP": CovidRegion("ESP", "Spain", 47_000_000, "Spain"),

    # Asia-Pacific
    "JPN": CovidRegion("JPN", "Japan", 126_000_000, "Japan"),
    "KOR": CovidRegion("KOR", "South Korea", 52_000_000, "South Korea"),
    "AUS": CovidRegion("AUS", "Australia", 26_000_000, "Australia"),
    "NZL": CovidRegion("NZL", "New Zealand", 5_000_000, "New Zealand"),
    "IND": CovidRegion("IND", "India", 1_400_000_000, "India"),
    "IDN": CovidRegion("IDN", "Indonesia", 274_000_000, "Indonesia"),

    # South America
    "BRA": CovidRegion("BRA", "Brazil", 214_000_000, "Brazil"),
    "ARG": CovidRegion("ARG", "Argentina", 45_000_000, "Argentina"),

    # Africa
    "ZAF": CovidRegion("ZAF", "South Africa", 60_000_000, "South Africa"),

    # Aggregates (useful for global view)
    "WORLD": CovidRegion("WORLD", "World", 8_000_000_000, "World"),
    "EU": CovidRegion("EU", "European Union", 447_000_000, "European Union"),
}


# Indicator types we generate for each region
INDICATOR_TYPES = {
    "CASES": {
        "owid_column": "new_cases_smoothed",
        "description": "Daily new cases (7-day smoothed)",
        "unit": "cases/day",
    },
    "DEATHS": {
        "owid_column": "new_deaths_smoothed",
        "description": "Daily new deaths (7-day smoothed)",
        "unit": "deaths/day",
    },
    "CASES_PC": {
        "owid_column": "new_cases_smoothed_per_million",
        "description": "Daily new cases per million population",
        "unit": "cases/day/million",
    },
    "DEATHS_PC": {
        "owid_column": "new_deaths_smoothed_per_million",
        "description": "Daily new deaths per million population",
        "unit": "deaths/day/million",
    },
    "HOSP": {
        "owid_column": "hosp_patients",
        "description": "Current hospital patients",
        "unit": "patients",
    },
    "ICU": {
        "owid_column": "icu_patients",
        "description": "Current ICU patients",
        "unit": "patients",
    },
    "VAX": {
        "owid_column": "people_fully_vaccinated_per_hundred",
        "description": "Percent fully vaccinated",
        "unit": "percent",
    },
    "POSITIVITY": {
        "owid_column": "positive_rate",
        "description": "Test positivity rate",
        "unit": "ratio (0-1)",
    },
    "STRINGENCY": {
        "owid_column": "stringency_index",
        "description": "Government response stringency (0-100)",
        "unit": "index",
    },
    "REPRODUCTION": {
        "owid_column": "reproduction_rate",
        "description": "Effective reproduction number (R)",
        "unit": "ratio",
    },
}

# Core indicators (always available for most countries)
CORE_INDICATORS = ["CASES", "DEATHS", "CASES_PC", "DEATHS_PC", "STRINGENCY"]


# =============================================================================
# FETCH RESULT
# =============================================================================

@dataclass
class FetchResult:
    """Result from fetching a single indicator."""
    indicator_id: str
    success: bool
    data: Optional[pd.DataFrame] = None
    rows: int = 0
    error: Optional[str] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_indicator_id(indicator_id: str) -> Tuple[str, str]:
    """
    Parse indicator ID into region and type.

    Examples:
        "USA_CASES" -> ("USA", "CASES")
        "JPN_DEATHS_PC" -> ("JPN", "DEATHS_PC")
    """
    for type_suffix in sorted(INDICATOR_TYPES.keys(), key=len, reverse=True):
        if indicator_id.endswith(f"_{type_suffix}"):
            region = indicator_id[:-len(type_suffix)-1]
            return region, type_suffix

    raise ValueError(f"Unknown indicator format: {indicator_id}")


def get_all_indicator_ids(core_only: bool = True) -> List[str]:
    """Return list of all available indicator IDs."""
    types = CORE_INDICATORS if core_only else INDICATOR_TYPES.keys()
    return [
        f"{region}_{indicator_type}"
        for region in REGIONS
        for indicator_type in types
    ]


# =============================================================================
# OWID DATA CLIENT
# =============================================================================

class OWIDClient:
    """
    Client for Our World in Data COVID-19 dataset.

    Data source: https://github.com/owid/covid-19-data
    """

    # Direct link to the full dataset CSV
    DATA_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

    # Columns we need
    REQUIRED_COLUMNS = [
        "date", "location", "iso_code",
        "new_cases_smoothed", "new_deaths_smoothed",
        "new_cases_smoothed_per_million", "new_deaths_smoothed_per_million",
        "hosp_patients", "icu_patients",
        "people_fully_vaccinated_per_hundred",
        "positive_rate", "stringency_index", "reproduction_rate",
        "population",
    ]

    def __init__(self, cache_hours: int = 24):
        """
        Initialize client.

        Args:
            cache_hours: Hours to cache the downloaded data
        """
        self.cache_hours = cache_hours
        self._data: Optional[pd.DataFrame] = None
        self._last_fetch: Optional[datetime] = None

    def _cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._data is None or self._last_fetch is None:
            return False
        elapsed = datetime.now() - self._last_fetch
        return elapsed.total_seconds() < self.cache_hours * 3600

    def fetch_all(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch the complete OWID dataset.

        Returns cached data if available and valid.
        """
        if not force_refresh and self._cache_valid():
            logger.debug("Using cached OWID data")
            return self._data

        logger.info("Downloading OWID COVID-19 dataset...")

        try:
            response = requests.get(self.DATA_URL, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to download OWID data: {e}")
            if self._data is not None:
                logger.warning("Using stale cached data")
                return self._data
            raise

        # Parse CSV
        df = pd.read_csv(
            StringIO(response.text),
            parse_dates=["date"],
            usecols=lambda c: c in self.REQUIRED_COLUMNS or c == "date" or c == "location",
        )

        logger.info(f"Downloaded {len(df)} rows, {df['location'].nunique()} locations")

        self._data = df
        self._last_fetch = datetime.now()

        return df

    def get_region_data(
        self,
        region: CovidRegion,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get data for a specific region and date range.
        """
        df = self.fetch_all()

        # Filter by location
        mask = df["location"] == region.owid_location
        region_df = df[mask].copy()

        if region_df.empty:
            logger.warning(f"No data found for {region.name} ({region.owid_location})")
            return pd.DataFrame()

        # Filter by date
        region_df = region_df[
            (region_df["date"].dt.date >= start_date) &
            (region_df["date"].dt.date <= end_date)
        ]

        return region_df.sort_values("date").reset_index(drop=True)


# =============================================================================
# PRISM FETCHER
# =============================================================================

class OWIDFetcher:
    """
    PRISM-compatible fetcher for COVID-19 epidemiological data.

    Converts OWID data into standardized time series for geometric analysis.
    """

    def __init__(self):
        self.client = OWIDClient()

    def fetch(
        self,
        indicator_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> FetchResult:
        """
        Fetch a single epidemiological indicator.

        Args:
            indicator_id: e.g., "USA_CASES", "JPN_DEATHS_PC"
            start_date: Start date (default: 2020-01-01)
            end_date: End date (default: yesterday)

        Returns:
            FetchResult with daily time series
        """
        # Default date range
        if start_date is None:
            start_date = date(2020, 1, 1)
        if end_date is None:
            end_date = date.today() - timedelta(days=1)

        # Parse indicator ID
        try:
            region_code, indicator_type = parse_indicator_id(indicator_id)
        except ValueError as e:
            return FetchResult(
                indicator_id=indicator_id,
                success=False,
                error=str(e)
            )

        if region_code not in REGIONS:
            return FetchResult(
                indicator_id=indicator_id,
                success=False,
                error=f"Unknown region: {region_code}. Available: {list(REGIONS.keys())}"
            )

        if indicator_type not in INDICATOR_TYPES:
            return FetchResult(
                indicator_id=indicator_id,
                success=False,
                error=f"Unknown indicator type: {indicator_type}. Available: {list(INDICATOR_TYPES.keys())}"
            )

        region = REGIONS[region_code]
        indicator_info = INDICATOR_TYPES[indicator_type]
        owid_column = indicator_info["owid_column"]

        # Get region data
        try:
            region_data = self.client.get_region_data(region, start_date, end_date)
        except Exception as e:
            return FetchResult(
                indicator_id=indicator_id,
                success=False,
                error=f"Data fetch error: {e}"
            )

        if region_data.empty:
            return FetchResult(
                indicator_id=indicator_id,
                success=False,
                error="No data available for region"
            )

        if owid_column not in region_data.columns:
            return FetchResult(
                indicator_id=indicator_id,
                success=False,
                error=f"Column {owid_column} not available for {region.name}"
            )

        # Extract the series
        df = region_data[["date", owid_column]].copy()
        df.columns = ["date", "value"]

        # Convert date to date type
        df["date"] = df["date"].dt.date

        return FetchResult(
            indicator_id=indicator_id,
            success=True,
            data=df,
            rows=len(df)
        )

    def fetch_region_all(
        self,
        region_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        core_only: bool = True,
    ) -> List[FetchResult]:
        """
        Fetch all indicator types for a single region.

        Args:
            region_code: Region code (e.g., "USA")
            start_date: Start date
            end_date: End date
            core_only: If True, only fetch core indicators

        Returns:
            List of FetchResult, one per indicator type
        """
        types = CORE_INDICATORS if core_only else INDICATOR_TYPES.keys()
        results = []

        for indicator_type in types:
            indicator_id = f"{region_code}_{indicator_type}"
            results.append(self.fetch(indicator_id, start_date, end_date))

        return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fetch COVID-19 epidemiology data for PRISM analysis"
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["USA", "GBR", "DEU", "JPN", "BRA", "IND"],
        help="Regions to fetch (default: major countries)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: yesterday)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file",
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today() - timedelta(days=1)

    fetcher = OWIDFetcher()

    print(f"Fetching COVID-19 data from {start_date} to {end_date}")
    print(f"Regions: {args.regions}")
    print("-" * 60)

    all_results = []
    for region_code in args.regions:
        print(f"\n[{region_code}] {REGIONS[region_code].name}")
        results = fetcher.fetch_region_all(region_code, start_date, end_date)

        for r in results:
            if r.success:
                non_null = r.data["value"].notna().sum()
                print(f"  + {r.indicator_id}: {r.rows} days ({non_null} with data)")
                if r.data is not None:
                    df = r.data.copy()
                    df["indicator_id"] = r.indicator_id
                    all_results.append(df)
            else:
                print(f"  - {r.indicator_id}: {r.error}")

    if args.output and all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined[["indicator_id", "date", "value"]]
        combined.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
