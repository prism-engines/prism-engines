"""
PRISM USGS Earthquake Fetcher

Fetches earthquake data from USGS and converts to PRISM-compatible time series.

DOMAIN TRANSLATION:
    Finance         →  Seismology
    ─────────────────────────────────────────
    Indicator       →  Fault zone / Region
    Daily return    →  Daily earthquake count or energy release
    Volatility      →  Magnitude variance
    Correlation     →  Cross-fault triggering
    Regime change   →  Aftershock sequence / Swarm onset
    Hidden mass     →  Stress accumulation before rupture

INDICATORS GENERATED:
    For each region, we generate multiple time series:
    - {REGION}_COUNT     : Daily earthquake count
    - {REGION}_ENERGY    : Daily seismic energy release (Joules, log scale)
    - {REGION}_MAXMAG    : Maximum magnitude per day
    - {REGION}_MEANMAG   : Mean magnitude per day
    - {REGION}_DEPTH     : Mean depth per day (km)

REGIONS:
    - SAN_ANDREAS   : California (San Andreas fault system)
    - CASCADIA      : Pacific Northwest (Cascadia subduction zone)
    - NEW_MADRID    : Central US (New Madrid seismic zone)
    - ALASKA        : Alaska-Aleutian subduction zone
    - JAPAN         : Japan trench system
    - INDONESIA     : Sunda megathrust
    - CHILE         : Chile-Peru trench
    - MEDITERRANEAN : Mediterranean-Alpine belt

Usage:
    python -m prism.fetch.usgs --start 2014-01-01 --end 2024-12-31 --output earthquakes.db

    Or import and use programmatically:

    from prism.fetch.usgs import USGSFetcher
    fetcher = USGSFetcher()
    result = fetcher.fetch("SAN_ANDREAS_COUNT", start_date=date(2020, 1, 1))
"""

import logging
import requests
import time
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import math

import pandas as pd
import numpy as np

from prism.fetch.base import BaseFetcher, FetchResult


logger = logging.getLogger(__name__)


# =============================================================================
# REGION DEFINITIONS
# =============================================================================

@dataclass
class SeismicRegion:
    """Definition of a seismic monitoring region."""
    name: str
    description: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    min_magnitude: float = 2.5  # Minimum magnitude to include

    @property
    def bbox_params(self) -> Dict[str, float]:
        """Return USGS API bounding box parameters."""
        return {
            "minlatitude": self.min_lat,
            "maxlatitude": self.max_lat,
            "minlongitude": self.min_lon,
            "maxlongitude": self.max_lon,
        }


# Major seismic regions for cross-domain analysis
REGIONS: Dict[str, SeismicRegion] = {
    # North America
    "SAN_ANDREAS": SeismicRegion(
        name="SAN_ANDREAS",
        description="California - San Andreas Fault System",
        min_lat=32.0, max_lat=42.0,
        min_lon=-125.0, max_lon=-114.0,
        min_magnitude=2.0,
    ),
    "CASCADIA": SeismicRegion(
        name="CASCADIA",
        description="Pacific Northwest - Cascadia Subduction Zone",
        min_lat=40.0, max_lat=52.0,
        min_lon=-130.0, max_lon=-120.0,
        min_magnitude=2.5,
    ),
    "NEW_MADRID": SeismicRegion(
        name="NEW_MADRID",
        description="Central US - New Madrid Seismic Zone",
        min_lat=35.0, max_lat=40.0,
        min_lon=-92.0, max_lon=-88.0,
        min_magnitude=2.0,
    ),
    "ALASKA": SeismicRegion(
        name="ALASKA",
        description="Alaska-Aleutian Subduction Zone",
        min_lat=50.0, max_lat=72.0,
        min_lon=-180.0, max_lon=-130.0,
        min_magnitude=3.0,
    ),
    "YELLOWSTONE": SeismicRegion(
        name="YELLOWSTONE",
        description="Yellowstone Volcanic Region",
        min_lat=44.0, max_lat=45.5,
        min_lon=-111.5, max_lon=-109.5,
        min_magnitude=1.5,
    ),

    # Pacific Ring of Fire
    "JAPAN": SeismicRegion(
        name="JAPAN",
        description="Japan Trench System",
        min_lat=30.0, max_lat=46.0,
        min_lon=128.0, max_lon=146.0,
        min_magnitude=4.0,
    ),
    "INDONESIA": SeismicRegion(
        name="INDONESIA",
        description="Sunda Megathrust",
        min_lat=-12.0, max_lat=8.0,
        min_lon=95.0, max_lon=140.0,
        min_magnitude=4.5,
    ),
    "CHILE": SeismicRegion(
        name="CHILE",
        description="Chile-Peru Trench",
        min_lat=-56.0, max_lat=-17.0,
        min_lon=-80.0, max_lon=-66.0,
        min_magnitude=4.0,
    ),
    "PHILIPPINES": SeismicRegion(
        name="PHILIPPINES",
        description="Philippine Trench",
        min_lat=4.0, max_lat=22.0,
        min_lon=116.0, max_lon=128.0,
        min_magnitude=4.0,
    ),

    # Other major zones
    "MEDITERRANEAN": SeismicRegion(
        name="MEDITERRANEAN",
        description="Mediterranean-Alpine Seismic Belt",
        min_lat=30.0, max_lat=48.0,
        min_lon=-10.0, max_lon=45.0,
        min_magnitude=4.0,
    ),
    "HIMALAYA": SeismicRegion(
        name="HIMALAYA",
        description="Himalayan Collision Zone",
        min_lat=24.0, max_lat=38.0,
        min_lon=68.0, max_lon=98.0,
        min_magnitude=4.0,
    ),

    # Global (for comparison)
    "GLOBAL": SeismicRegion(
        name="GLOBAL",
        description="Worldwide M5.0+",
        min_lat=-90.0, max_lat=90.0,
        min_lon=-180.0, max_lon=180.0,
        min_magnitude=5.0,
    ),
}


# Indicator types we generate for each region
INDICATOR_TYPES = [
    "COUNT",     # Daily earthquake count
    "ENERGY",    # Daily energy release (log10 Joules)
    "MAXMAG",    # Maximum magnitude
    "MEANMAG",   # Mean magnitude
    "DEPTH",     # Mean depth (km)
]


def magnitude_to_energy(magnitude: float) -> float:
    """
    Convert earthquake magnitude to energy in Joules.

    Uses the Gutenberg-Richter relation:
        log10(E) = 1.5 * M + 4.8

    where E is in Joules and M is moment magnitude.

    Returns energy in Joules.
    """
    return 10 ** (1.5 * magnitude + 4.8)


def parse_indicator_id(indicator_id: str) -> Tuple[str, str]:
    """
    Parse indicator ID into region and type.

    Examples:
        "SAN_ANDREAS_COUNT" -> ("SAN_ANDREAS", "COUNT")
        "JAPAN_ENERGY" -> ("JAPAN", "ENERGY")
        "NEW_MADRID_MAXMAG" -> ("NEW_MADRID", "MAXMAG")
    """
    for type_suffix in INDICATOR_TYPES:
        if indicator_id.endswith(f"_{type_suffix}"):
            region = indicator_id[:-len(type_suffix)-1]
            return region, type_suffix

    raise ValueError(f"Unknown indicator format: {indicator_id}")


def get_all_indicator_ids() -> List[str]:
    """Return list of all available indicator IDs."""
    return [
        f"{region}_{indicator_type}"
        for region in REGIONS
        for indicator_type in INDICATOR_TYPES
    ]


# =============================================================================
# USGS API CLIENT
# =============================================================================

class USGSClient:
    """
    Client for USGS Earthquake Catalog API.

    API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/
    """

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # USGS limits to 20,000 results per query
    MAX_RESULTS = 20000

    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize client.

        Args:
            rate_limit_delay: Seconds to wait between API calls
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.session = requests.Session()

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def fetch_earthquakes(
        self,
        region: SeismicRegion,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch earthquakes for a region and date range.

        Args:
            region: SeismicRegion definition
            start_date: Start of date range
            end_date: End of date range (inclusive)

        Returns:
            DataFrame with columns: time, latitude, longitude, depth, mag, place
        """
        self._rate_limit()

        params = {
            "format": "csv",
            "starttime": start_date.isoformat(),
            "endtime": (end_date + timedelta(days=1)).isoformat(),  # API is exclusive
            "minmagnitude": region.min_magnitude,
            "orderby": "time",
            **region.bbox_params,
        }

        logger.debug(f"Fetching {region.name}: {start_date} to {end_date}")

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"USGS API error for {region.name}: {e}")
            return pd.DataFrame()

        if not response.text.strip():
            return pd.DataFrame()

        try:
            df = pd.read_csv(
                pd.io.common.StringIO(response.text),
                parse_dates=["time"],
                usecols=["time", "latitude", "longitude", "depth", "mag", "place"],
            )
        except Exception as e:
            logger.error(f"Failed to parse USGS response: {e}")
            return pd.DataFrame()

        logger.debug(f"  Received {len(df)} earthquakes")
        return df

    def fetch_region_range(
        self,
        region: SeismicRegion,
        start_date: date,
        end_date: date,
        chunk_months: int = 6,
    ) -> pd.DataFrame:
        """
        Fetch earthquakes for a region, chunking to avoid API limits.

        Args:
            region: SeismicRegion definition
            start_date: Start date
            end_date: End date
            chunk_months: Months per chunk (to stay under 20k limit)

        Returns:
            Combined DataFrame of all earthquakes
        """
        all_data = []
        current_start = start_date

        while current_start < end_date:
            # Calculate chunk end
            chunk_end = min(
                current_start + timedelta(days=chunk_months * 30),
                end_date
            )

            df = self.fetch_earthquakes(region, current_start, chunk_end)
            if not df.empty:
                all_data.append(df)

            current_start = chunk_end + timedelta(days=1)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # Remove duplicates (can happen at chunk boundaries)
        combined = combined.drop_duplicates(subset=["time", "latitude", "longitude", "mag"])

        return combined.sort_values("time").reset_index(drop=True)


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_to_daily(
    earthquakes: pd.DataFrame,
    indicator_type: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Aggregate earthquake events to daily time series.

    Args:
        earthquakes: DataFrame with time, depth, mag columns
        indicator_type: One of INDICATOR_TYPES
        start_date: Start date for output series
        end_date: End date for output series

    Returns:
        DataFrame with columns [date, value]
    """
    if earthquakes.empty:
        # Return series of zeros/NaN for the date range
        dates = pd.date_range(start_date, end_date, freq="D")
        if indicator_type == "COUNT":
            return pd.DataFrame({"date": dates, "value": 0})
        else:
            return pd.DataFrame({"date": dates, "value": np.nan})

    # Add date column (date only, no time)
    eq = earthquakes.copy()
    eq["date"] = pd.to_datetime(eq["time"]).dt.date

    # Create full date range to ensure no gaps
    all_dates = pd.date_range(start_date, end_date, freq="D")
    date_df = pd.DataFrame({"date": all_dates.date})

    if indicator_type == "COUNT":
        # Daily earthquake count
        daily = eq.groupby("date").size().reset_index(name="value")

    elif indicator_type == "ENERGY":
        # Daily energy release (log10 Joules)
        eq["energy"] = eq["mag"].apply(magnitude_to_energy)
        daily = eq.groupby("date")["energy"].sum().reset_index(name="value")
        # Convert to log scale for better handling
        daily["value"] = np.log10(daily["value"])

    elif indicator_type == "MAXMAG":
        # Maximum magnitude per day
        daily = eq.groupby("date")["mag"].max().reset_index(name="value")

    elif indicator_type == "MEANMAG":
        # Mean magnitude per day
        daily = eq.groupby("date")["mag"].mean().reset_index(name="value")

    elif indicator_type == "DEPTH":
        # Mean depth per day
        daily = eq.groupby("date")["depth"].mean().reset_index(name="value")

    else:
        raise ValueError(f"Unknown indicator type: {indicator_type}")

    # Merge with full date range
    daily["date"] = pd.to_datetime(daily["date"])
    date_df["date"] = pd.to_datetime(date_df["date"])

    result = date_df.merge(daily, on="date", how="left")

    # Fill missing days
    if indicator_type == "COUNT":
        result["value"] = result["value"].fillna(0).astype(int)
    # Other types keep NaN for days with no earthquakes

    return result[["date", "value"]]


# =============================================================================
# PRISM FETCHER
# =============================================================================

class USGSFetcher(BaseFetcher):
    """
    PRISM-compatible fetcher for USGS earthquake data.

    Converts earthquake catalogs into daily time series for geometric analysis.
    """

    def __init__(self):
        super().__init__()
        self.client = USGSClient()
        self._cache: Dict[str, pd.DataFrame] = {}  # Region -> raw earthquakes

    @property
    def source_name(self) -> str:
        return "usgs"

    def _get_region_data(
        self,
        region: SeismicRegion,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get raw earthquake data for a region, using cache.
        """
        cache_key = f"{region.name}_{start_date}_{end_date}"

        if cache_key not in self._cache:
            logger.info(f"Fetching raw data for {region.name}...")
            self._cache[cache_key] = self.client.fetch_region_range(
                region, start_date, end_date
            )
            logger.info(f"  Cached {len(self._cache[cache_key])} earthquakes")

        return self._cache[cache_key]

    def fetch(
        self,
        indicator_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> FetchResult:
        """
        Fetch a single seismic indicator.

        Args:
            indicator_id: e.g., "SAN_ANDREAS_COUNT", "JAPAN_ENERGY"
            start_date: Start date (default: 10 years ago)
            end_date: End date (default: yesterday)

        Returns:
            FetchResult with daily time series
        """
        # Default date range
        if end_date is None:
            end_date = date.today() - timedelta(days=1)
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 10)

        # Parse indicator ID
        try:
            region_name, indicator_type = parse_indicator_id(indicator_id)
        except ValueError as e:
            return FetchResult.from_error(indicator_id, str(e))

        if region_name not in REGIONS:
            return FetchResult.from_error(
                indicator_id,
                f"Unknown region: {region_name}. Available: {list(REGIONS.keys())}"
            )

        region = REGIONS[region_name]

        # Get raw earthquake data
        try:
            earthquakes = self._get_region_data(region, start_date, end_date)
        except Exception as e:
            return FetchResult.from_error(indicator_id, f"API error: {e}")

        # Aggregate to daily series
        try:
            df = aggregate_to_daily(earthquakes, indicator_type, start_date, end_date)
        except Exception as e:
            return FetchResult.from_error(indicator_id, f"Aggregation error: {e}")

        return FetchResult.from_dataframe(indicator_id, df)

    def fetch_region_all(
        self,
        region_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[FetchResult]:
        """
        Fetch all indicator types for a single region.

        More efficient than calling fetch() multiple times because
        it shares the cached raw earthquake data.

        Args:
            region_name: Region name (e.g., "SAN_ANDREAS")
            start_date: Start date
            end_date: End date

        Returns:
            List of FetchResult, one per indicator type
        """
        results = []
        for indicator_type in INDICATOR_TYPES:
            indicator_id = f"{region_name}_{indicator_type}"
            results.append(self.fetch(indicator_id, start_date, end_date))
        return results

    def clear_cache(self):
        """Clear the raw earthquake data cache."""
        self._cache.clear()


# =============================================================================
# CLI
# =============================================================================

def main():
    """
    Command-line interface for fetching USGS earthquake data.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch USGS earthquake data for PRISM analysis"
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["SAN_ANDREAS", "CASCADIA", "JAPAN", "INDONESIA"],
        help="Regions to fetch (default: major Pacific zones)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2014-01-01",
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
        default="earthquakes.csv",
        help="Output file (CSV)",
    )
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List available regions and exit",
    )
    parser.add_argument(
        "--list-indicators",
        action="store_true",
        help="List all available indicators and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.list_regions:
        print("\nAvailable Regions:")
        print("-" * 60)
        for name, region in REGIONS.items():
            print(f"  {name:15} : {region.description}")
        print()
        return

    if args.list_indicators:
        print("\nAvailable Indicators:")
        print("-" * 60)
        for indicator_id in get_all_indicator_ids():
            print(f"  {indicator_id}")
        print()
        return

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = date.today() - timedelta(days=1)

    # Validate regions
    for region in args.regions:
        if region not in REGIONS:
            print(f"Error: Unknown region '{region}'")
            print(f"Available: {list(REGIONS.keys())}")
            return

    # Fetch data
    fetcher = USGSFetcher()
    all_results = []

    print(f"\nFetching earthquake data from {start_date} to {end_date}")
    print(f"Regions: {args.regions}")
    print("-" * 60)

    for region_name in args.regions:
        print(f"\n[{region_name}]")
        results = fetcher.fetch_region_all(region_name, start_date, end_date)

        for result in results:
            if result.success:
                print(f"  ✓ {result.indicator_id}: {result.rows} days")

                # Add indicator_id column and append
                df = result.data.copy()
                df["indicator_id"] = result.indicator_id
                all_results.append(df)
            else:
                print(f"  ✗ {result.indicator_id}: {result.error}")

    # Combine and save
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined[["indicator_id", "date", "value"]]
        combined.to_csv(args.output, index=False)

        print(f"\n{'=' * 60}")
        print(f"Saved {len(combined)} rows to {args.output}")
        print(f"Indicators: {combined['indicator_id'].nunique()}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    else:
        print("\nNo data fetched!")


if __name__ == "__main__":
    main()
