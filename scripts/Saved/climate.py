from typing import List, Dict, Optional
"""
PRISM Climate Agent

Fetches climate indices from NOAA and other sources.
No API key required - direct text file downloads.

Data sources:
- NOAA PSL (Physical Sciences Laboratory)
- NOAA CPC (Climate Prediction Center)
- NASA GISTEMP

Available indices:
- ENSO (El Niño Southern Oscillation)
- NAO (North Atlantic Oscillation)
- PDO (Pacific Decadal Oscillation)
- AMO (Atlantic Multidecadal Oscillation)
- AO (Arctic Oscillation)
- SOI (Southern Oscillation Index)
- PNA (Pacific-North American)
- Global Temperature Anomaly
"""

import logging
from typing import List, Optional
from io import StringIO

import pandas as pd
import requests

from .base import DomainAgent, IndicatorMeta

logger = logging.getLogger(__name__)


# NOAA data sources - direct text file URLs
CLIMATE_SOURCES = {
    # ENSO indices
    "NINO34": {
        "url": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
        "name": "Niño 3.4 Anomaly",
        "category": "enso",
        "description": "El Niño/La Niña indicator - sea surface temperature anomaly",
        "frequency": "monthly"
    },
    "NINO12": {
        "url": "https://psl.noaa.gov/data/correlation/nina1.anom.data",
        "name": "Niño 1+2 Anomaly",
        "category": "enso",
        "description": "Eastern Pacific SST anomaly",
        "frequency": "monthly"
    },
    "NINO3": {
        "url": "https://psl.noaa.gov/data/correlation/nina3.anom.data",
        "name": "Niño 3 Anomaly",
        "category": "enso",
        "description": "Central-Eastern Pacific SST anomaly",
        "frequency": "monthly"
    },
    "NINO4": {
        "url": "https://psl.noaa.gov/data/correlation/nina4.anom.data",
        "name": "Niño 4 Anomaly",
        "category": "enso",
        "description": "Central Pacific SST anomaly",
        "frequency": "monthly"
    },
    "ONI": {
        "url": "https://psl.noaa.gov/data/correlation/oni.data",
        "name": "Oceanic Niño Index",
        "category": "enso",
        "description": "3-month running mean of Niño 3.4 SST anomalies",
        "frequency": "monthly"
    },
    "SOI": {
        "url": "https://psl.noaa.gov/data/correlation/soi.data",
        "name": "Southern Oscillation Index",
        "category": "enso",
        "description": "Tahiti - Darwin sea level pressure difference",
        "frequency": "monthly"
    },
    "MEI": {
        "url": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
        "name": "Multivariate ENSO Index",
        "category": "enso",
        "description": "Combined oceanic-atmospheric ENSO index",
        "frequency": "monthly"
    },
    
    # Atlantic indices
    "NAO": {
        "url": "https://psl.noaa.gov/data/correlation/nao.data",
        "name": "North Atlantic Oscillation",
        "category": "atlantic",
        "description": "Iceland-Azores pressure difference",
        "frequency": "monthly"
    },
    "AMO": {
        "url": "https://psl.noaa.gov/data/correlation/amon.us.data",
        "name": "Atlantic Multidecadal Oscillation",
        "category": "atlantic",
        "description": "North Atlantic SST anomaly (detrended)",
        "frequency": "monthly"
    },
    "TNA": {
        "url": "https://psl.noaa.gov/data/correlation/tna.data",
        "name": "Tropical Northern Atlantic",
        "category": "atlantic",
        "description": "Tropical North Atlantic SST anomaly",
        "frequency": "monthly"
    },
    "TSA": {
        "url": "https://psl.noaa.gov/data/correlation/tsa.data",
        "name": "Tropical Southern Atlantic",
        "category": "atlantic",
        "description": "Tropical South Atlantic SST anomaly",
        "frequency": "monthly"
    },
    
    # Pacific indices
    "PDO": {
        "url": "https://psl.noaa.gov/data/correlation/pdo.data",
        "name": "Pacific Decadal Oscillation",
        "category": "pacific",
        "description": "North Pacific SST pattern",
        "frequency": "monthly"
    },
    "PNA": {
        "url": "https://psl.noaa.gov/data/correlation/pna.data",
        "name": "Pacific-North American",
        "category": "pacific",
        "description": "Atmospheric circulation pattern",
        "frequency": "monthly"
    },
    "WP": {
        "url": "https://psl.noaa.gov/data/correlation/wp.data",
        "name": "Western Pacific",
        "category": "pacific",
        "description": "Western Pacific oscillation pattern",
        "frequency": "monthly"
    },
    "EP_NP": {
        "url": "https://psl.noaa.gov/data/correlation/epo.data",
        "name": "East Pacific-North Pacific",
        "category": "pacific",
        "description": "EP-NP oscillation pattern",
        "frequency": "monthly"
    },
    
    # Arctic/Polar indices
    "AO": {
        "url": "https://psl.noaa.gov/data/correlation/ao.data",
        "name": "Arctic Oscillation",
        "category": "polar",
        "description": "Arctic sea level pressure pattern",
        "frequency": "monthly"
    },
    "AAO": {
        "url": "https://psl.noaa.gov/data/correlation/aao.data",
        "name": "Antarctic Oscillation",
        "category": "polar",
        "description": "Southern Annular Mode",
        "frequency": "monthly"
    },
    
    # Global indices
    "WHWP": {
        "url": "https://psl.noaa.gov/data/correlation/whwp.data",
        "name": "Western Hemisphere Warm Pool",
        "category": "global",
        "description": "Warm pool area index",
        "frequency": "monthly"
    },
    "GLOBAL_TEMP": {
        "url": "https://psl.noaa.gov/data/correlation/glbts.data",
        "name": "Global Temperature Anomaly",
        "category": "global",
        "description": "Global mean surface temperature anomaly",
        "frequency": "monthly"
    },
    
    # Indian Ocean
    "DMI": {
        "url": "https://psl.noaa.gov/data/correlation/dmi.data",
        "name": "Dipole Mode Index",
        "category": "indian",
        "description": "Indian Ocean Dipole",
        "frequency": "monthly"
    },
}


class ClimateAgent(DomainAgent):
    """
    Climate domain agent.
    
    Fetches climate indices from NOAA PSL and other sources.
    No API key required.
    """
    
    domain = "climate"
    sources = ["noaa_psl", "noaa_cpc", "nasa"]
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PRISM-Observatory/1.0 (Research)'
        })
    
    def discover(self) -> List[IndicatorMeta]:
        """Return available climate indicators."""
        self.indicators = []
        
        for ind_id, config in CLIMATE_SOURCES.items():
            self.indicators.append(IndicatorMeta(
                id=ind_id,
                name=config["name"],
                domain=self.domain,
                source="noaa_psl",
                category=config["category"],
                frequency=config["frequency"],
                description=config.get("description", "")
            ))
        
        logger.info(f"Discovered {len(self.indicators)} climate indicators")
        return self.indicators
    
    def _parse_noaa_format(self, text: str, indicator_id: str) -> pd.DataFrame:
        """
        Parse NOAA PSL text format.
        
        Format is typically:
        YEAR  JAN  FEB  MAR  APR  MAY  JUN  JUL  AUG  SEP  OCT  NOV  DEC
        1950  0.1  0.2  0.3  ...
        
        Missing values are usually -99.9 or -99.99
        """
        lines = text.strip().split('\n')
        
        records = []
        
        for line in lines:
            parts = line.split()
            
            if not parts:
                continue
            
            # Try to parse year
            try:
                year = int(parts[0])
                if year < 1800 or year > 2100:
                    continue
            except ValueError:
                continue
            
            # Parse monthly values
            for month_idx, value_str in enumerate(parts[1:13], start=1):
                try:
                    value = float(value_str)
                    
                    # Skip missing values
                    if value <= -99 or value >= 999:
                        continue
                    
                    date = pd.Timestamp(year=year, month=month_idx, day=1)
                    records.append({"date": date, "value": value})
                    
                except ValueError:
                    continue
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        
        return df
    
    def fetch_one(self, indicator_id: str) -> Optional[pd.DataFrame]:
        """Fetch a single climate indicator."""
        if indicator_id not in CLIMATE_SOURCES:
            logger.error(f"Unknown indicator: {indicator_id}")
            return None
        
        config = CLIMATE_SOURCES[indicator_id]
        url = config["url"]
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            df = self._parse_noaa_format(response.text, indicator_id)
            
            if len(df) == 0:
                logger.warning(f"No data parsed for {indicator_id}")
                return None
            
            logger.info(f"Fetched {indicator_id}: {len(df)} rows, "
                       f"{df['date'].min().date()} to {df['date'].max().date()}")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Request failed for {indicator_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Parse failed for {indicator_id}: {e}")
            return None
    
    def fetch_category(self, category: str, delay: float = 0.5) -> Dict[str, pd.DataFrame]:
        """Fetch all indicators in a category."""
        indicators = [
            ind_id for ind_id, config in CLIMATE_SOURCES.items()
            if config["category"] == category
        ]
        return self.fetch_all(delay=delay, indicators=indicators)
    
    def get_categories(self) -> List[str]:
        """Get available categories."""
        return list(set(c["category"] for c in CLIMATE_SOURCES.values()))


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = ClimateAgent()
    
    # Discover
    indicators = agent.discover()
    print(f"\nDiscovered {len(indicators)} indicators:")
    for cat in agent.get_categories():
        cat_inds = [i for i in indicators if i.category == cat]
        print(f"  {cat}: {len(cat_inds)}")
    
    # Fetch one
    print("\nFetching NINO34...")
    df = agent.fetch_one("NINO34")
    if df is not None:
        print(f"  Rows: {len(df)}")
        print(f"  Range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Sample:\n{df.tail()}")
