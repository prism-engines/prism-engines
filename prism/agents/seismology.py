"""
PRISM Seismology Agent

Fetches seismological data from USGS and other sources.
No API key required.

Data sources:
- USGS Earthquake Hazards Program
- IRIS (Incorporated Research Institutions for Seismology)

Available indicators:
- Global earthquake counts (by magnitude threshold)
- Regional seismic activity
- Cumulative seismic moment release
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests

from .base import DomainAgent, IndicatorMeta

logger = logging.getLogger(__name__)


# USGS Earthquake API
USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Seismology indicators
SEISMOLOGY_SOURCES = {
    # Global earthquake counts by magnitude
    "EQ_M4_GLOBAL": {
        "name": "Global M4+ Earthquakes (Monthly Count)",
        "category": "activity",
        "description": "Count of magnitude 4+ earthquakes globally per month",
        "frequency": "monthly",
        "min_magnitude": 4.0,
        "region": None,
    },
    "EQ_M5_GLOBAL": {
        "name": "Global M5+ Earthquakes (Monthly Count)",
        "category": "activity",
        "description": "Count of magnitude 5+ earthquakes globally per month",
        "frequency": "monthly",
        "min_magnitude": 5.0,
        "region": None,
    },
    "EQ_M6_GLOBAL": {
        "name": "Global M6+ Earthquakes (Monthly Count)",
        "category": "activity",
        "description": "Count of magnitude 6+ earthquakes globally per month",
        "frequency": "monthly",
        "min_magnitude": 6.0,
        "region": None,
    },
    "EQ_M7_GLOBAL": {
        "name": "Global M7+ Earthquakes (Monthly Count)",
        "category": "activity",
        "description": "Count of magnitude 7+ earthquakes globally per month",
        "frequency": "monthly",
        "min_magnitude": 7.0,
        "region": None,
    },

    # Cumulative energy release
    "SEISMIC_MOMENT_GLOBAL": {
        "name": "Global Seismic Moment Release",
        "category": "energy",
        "description": "Cumulative seismic moment release per month (log scale)",
        "frequency": "monthly",
        "min_magnitude": 5.0,
        "region": None,
    },

    # Regional - US
    "EQ_M3_US": {
        "name": "US M3+ Earthquakes (Monthly Count)",
        "category": "regional",
        "description": "Count of magnitude 3+ earthquakes in contiguous US",
        "frequency": "monthly",
        "min_magnitude": 3.0,
        "region": {"minlat": 24, "maxlat": 50, "minlon": -125, "maxlon": -66},
    },

    # Regional - California
    "EQ_M3_CA": {
        "name": "California M3+ Earthquakes (Monthly Count)",
        "category": "regional",
        "description": "Count of magnitude 3+ earthquakes in California",
        "frequency": "monthly",
        "min_magnitude": 3.0,
        "region": {"minlat": 32, "maxlat": 42, "minlon": -125, "maxlon": -114},
    },

    # Regional - Pacific Ring of Fire
    "EQ_M5_PACIFIC": {
        "name": "Pacific Ring M5+ Earthquakes (Monthly Count)",
        "category": "regional",
        "description": "Count of magnitude 5+ earthquakes in Pacific Ring of Fire",
        "frequency": "monthly",
        "min_magnitude": 5.0,
        "region": {"minlat": -60, "maxlat": 60, "minlon": -180, "maxlon": -100},
    },

    # Regional - Japan
    "EQ_M4_JAPAN": {
        "name": "Japan M4+ Earthquakes (Monthly Count)",
        "category": "regional",
        "description": "Count of magnitude 4+ earthquakes near Japan",
        "frequency": "monthly",
        "min_magnitude": 4.0,
        "region": {"minlat": 24, "maxlat": 46, "minlon": 122, "maxlon": 150},
    },

    # Depth categories
    "EQ_SHALLOW_GLOBAL": {
        "name": "Global Shallow Earthquakes M5+ (0-70km)",
        "category": "depth",
        "description": "Count of shallow (0-70km) M5+ earthquakes",
        "frequency": "monthly",
        "min_magnitude": 5.0,
        "region": None,
        "max_depth": 70,
    },
    "EQ_DEEP_GLOBAL": {
        "name": "Global Deep Earthquakes M5+ (300km+)",
        "category": "depth",
        "description": "Count of deep (300km+) M5+ earthquakes",
        "frequency": "monthly",
        "min_magnitude": 5.0,
        "region": None,
        "min_depth": 300,
    },
}


class SeismologyAgent(DomainAgent):
    """
    Seismology domain agent.

    Fetches earthquake data from USGS.
    No API key required.
    """

    domain = "seismology"
    sources = ["usgs", "iris"]

    def __init__(self):
        super().__init__(domain=self.domain)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PRISM-Observatory/1.0 (Research)'
        })

    def discover(self) -> List[IndicatorMeta]:
        """Return available seismology indicators."""
        self.indicators = []

        for ind_id, config in SEISMOLOGY_SOURCES.items():
            self.indicators.append(IndicatorMeta(
                indicator_id=ind_id,
                name=config["name"],
                source="usgs",
                category=config["category"],
                frequency=config["frequency"],
                description=config.get("description", "")
            ))

        logger.info(f"Discovered {len(self.indicators)} seismology indicators")
        return self.indicators

    def _fetch_usgs_earthquakes(
        self,
        start_date: datetime,
        end_date: datetime,
        min_magnitude: float = 4.0,
        region: Optional[Dict] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Fetch earthquake events from USGS.

        Returns DataFrame with columns: time, magnitude, latitude, longitude, depth
        """
        params = {
            "format": "csv",
            "starttime": start_date.strftime("%Y-%m-%d"),
            "endtime": end_date.strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
            "orderby": "time-asc",
        }

        if region:
            params.update(region)

        if min_depth is not None:
            params["mindepth"] = min_depth
        if max_depth is not None:
            params["maxdepth"] = max_depth

        try:
            response = self.session.get(USGS_BASE, params=params, timeout=60)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))

            if len(df) == 0:
                return pd.DataFrame()

            # Parse time
            df['time'] = pd.to_datetime(df['time'])

            # Select relevant columns
            cols = ['time', 'mag', 'latitude', 'longitude', 'depth']
            cols = [c for c in cols if c in df.columns]
            df = df[cols].copy()

            return df

        except requests.RequestException as e:
            logger.error(f"USGS request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"USGS parse failed: {e}")
            return pd.DataFrame()

    def _aggregate_monthly(self, events: pd.DataFrame, agg_type: str = "count") -> pd.DataFrame:
        """Aggregate events to monthly frequency."""
        if len(events) == 0:
            return pd.DataFrame()

        events['month'] = events['time'].dt.to_period('M')

        if agg_type == "count":
            monthly = events.groupby('month').size().reset_index(name='value')
        elif agg_type == "moment":
            # Seismic moment from magnitude: M0 = 10^(1.5*M + 9.1)
            events['moment'] = 10 ** (1.5 * events['mag'] + 9.1)
            monthly = events.groupby('month')['moment'].sum().reset_index(name='value')
            # Log scale for readability
            monthly['value'] = monthly['value'].apply(lambda x: x if x == 0 else pd.np.log10(x))
        else:
            monthly = events.groupby('month').size().reset_index(name='value')

        monthly['date'] = monthly['month'].dt.to_timestamp()
        monthly = monthly[['date', 'value']]

        return monthly

    def fetch_one(
        self,
        indicator_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch a single seismology indicator."""
        if indicator_id not in SEISMOLOGY_SOURCES:
            logger.error(f"Unknown indicator: {indicator_id}")
            return None

        config = SEISMOLOGY_SOURCES[indicator_id]

        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = datetime(1970, 1, 1)

        # Fetch in chunks (USGS limits results)
        all_events = []
        chunk_start = start_date
        chunk_days = 365  # 1 year chunks

        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end_date)

            events = self._fetch_usgs_earthquakes(
                start_date=chunk_start,
                end_date=chunk_end,
                min_magnitude=config["min_magnitude"],
                region=config.get("region"),
                min_depth=config.get("min_depth"),
                max_depth=config.get("max_depth"),
            )

            if len(events) > 0:
                all_events.append(events)

            chunk_start = chunk_end

        if not all_events:
            logger.warning(f"No data for {indicator_id}")
            return None

        events = pd.concat(all_events, ignore_index=True)

        # Aggregate
        agg_type = "moment" if "MOMENT" in indicator_id else "count"
        df = self._aggregate_monthly(events, agg_type)

        if len(df) == 0:
            return None

        logger.info(f"Fetched {indicator_id}: {len(df)} months, "
                   f"{df['date'].min().date()} to {df['date'].max().date()}")

        return df

    def fetch_category(self, category: str) -> Dict[str, pd.DataFrame]:
        """Fetch all indicators in a category."""
        results = {}
        for ind_id, config in SEISMOLOGY_SOURCES.items():
            if config["category"] == category:
                df = self.fetch_one(ind_id)
                if df is not None:
                    results[ind_id] = df
        return results

    def get_categories(self) -> List[str]:
        """Get available categories."""
        return list(set(c["category"] for c in SEISMOLOGY_SOURCES.values()))


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    agent = SeismologyAgent()

    # Discover
    indicators = agent.discover()
    print(f"\nDiscovered {len(indicators)} indicators:")
    for cat in agent.get_categories():
        cat_inds = [i for i in indicators if i.category == cat]
        print(f"  {cat}: {len(cat_inds)}")

    # Fetch one
    print("\nFetching EQ_M5_GLOBAL (last 5 years)...")
    df = agent.fetch_one(
        "EQ_M5_GLOBAL",
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2024, 12, 31)
    )
    if df is not None:
        print(f"  Rows: {len(df)}")
        print(f"  Range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Sample:\n{df.tail(10)}")
