from typing import List, Dict, Optional
"""
PRISM Epidemiology Agent

Fetches disease surveillance data from CMU Delphi Epidata API.
No API key required for most endpoints.

Data sources:
- CMU Delphi Epidata API
- CDC FluView (via Delphi)
- COVID-19 indicators

Available signals:
- ILI (Influenza-like Illness)
- Flu hospitalizations
- COVID indicators
- Syndromic surveillance
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import requests

from .base import DomainAgent, IndicatorMeta

logger = logging.getLogger(__name__)


# Delphi API configuration
DELPHI_BASE_URL = "https://api.delphi.cmu.edu/epidata"

# Available epidemiological signals
EPI_SOURCES = {
    # CDC FluView - Influenza surveillance
    "ILI_NATIONAL": {
        "endpoint": "fluview",
        "name": "ILI National Rate",
        "category": "influenza",
        "description": "National influenza-like illness rate from CDC",
        "frequency": "weekly",
        "params": {"regions": "nat"}
    },
    "ILI_HHS1": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 1",
        "category": "influenza",
        "description": "ILI rate - New England",
        "frequency": "weekly",
        "params": {"regions": "hhs1"}
    },
    "ILI_HHS2": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 2",
        "category": "influenza",
        "description": "ILI rate - NY, NJ, PR, VI",
        "frequency": "weekly",
        "params": {"regions": "hhs2"}
    },
    "ILI_HHS3": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 3",
        "category": "influenza",
        "description": "ILI rate - Mid-Atlantic",
        "frequency": "weekly",
        "params": {"regions": "hhs3"}
    },
    "ILI_HHS4": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 4",
        "category": "influenza",
        "description": "ILI rate - Southeast",
        "frequency": "weekly",
        "params": {"regions": "hhs4"}
    },
    "ILI_HHS5": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 5",
        "category": "influenza",
        "description": "ILI rate - Midwest",
        "frequency": "weekly",
        "params": {"regions": "hhs5"}
    },
    "ILI_HHS6": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 6",
        "category": "influenza",
        "description": "ILI rate - South Central",
        "frequency": "weekly",
        "params": {"regions": "hhs6"}
    },
    "ILI_HHS7": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 7",
        "category": "influenza",
        "description": "ILI rate - Central",
        "frequency": "weekly",
        "params": {"regions": "hhs7"}
    },
    "ILI_HHS8": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 8",
        "category": "influenza",
        "description": "ILI rate - Mountain",
        "frequency": "weekly",
        "params": {"regions": "hhs8"}
    },
    "ILI_HHS9": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 9",
        "category": "influenza",
        "description": "ILI rate - Pacific",
        "frequency": "weekly",
        "params": {"regions": "hhs9"}
    },
    "ILI_HHS10": {
        "endpoint": "fluview",
        "name": "ILI HHS Region 10",
        "category": "influenza",
        "description": "ILI rate - Pacific Northwest",
        "frequency": "weekly",
        "params": {"regions": "hhs10"}
    },
    
    # Clinical surveillance
    "FLU_HOSP_NATIONAL": {
        "endpoint": "flusurv",
        "name": "Flu Hospitalizations National",
        "category": "hospitalizations",
        "description": "FluSurv-NET hospitalization rate",
        "frequency": "weekly",
        "params": {"locations": "network_all"}
    },
    
    # Wikipedia access (proxy for public interest/awareness)
    "WIKI_FLU": {
        "endpoint": "wiki",
        "name": "Wikipedia Flu Article",
        "category": "digital",
        "description": "Wikipedia access to flu-related articles",
        "frequency": "daily",
        "params": {"articles": "influenza"}
    },
    
    # Google Health Trends (if available)
    "GHT_FLU": {
        "endpoint": "ght",
        "name": "Google Health Trends Flu",
        "category": "digital",
        "description": "Google search trends for flu",
        "frequency": "weekly",
        "params": {"query": "flu"}
    },
}


def epiweek_to_date(epiweek: int) -> pd.Timestamp:
    """
    Convert CDC epiweek (YYYYWW) to date.
    
    Args:
        epiweek: Integer like 202301 (year 2023, week 1)
    
    Returns:
        Timestamp for the start of that epiweek
    """
    year = epiweek // 100
    week = epiweek % 100
    
    # First day of the year
    jan1 = datetime(year, 1, 1)
    
    # Find first Sunday
    days_to_sunday = (6 - jan1.weekday()) % 7
    first_sunday = jan1 + timedelta(days=days_to_sunday)
    
    # Add weeks
    target_date = first_sunday + timedelta(weeks=week - 1)
    
    return pd.Timestamp(target_date)


class EpidemiologyAgent(DomainAgent):
    """
    Epidemiology domain agent.
    
    Fetches disease surveillance data from CMU Delphi API.
    """
    
    domain = "epidemiology"
    sources = ["delphi", "cdc", "who"]
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PRISM-Observatory/1.0 (Research)'
        })
        self.base_url = DELPHI_BASE_URL
    
    def discover(self) -> List[IndicatorMeta]:
        """Return available epi indicators."""
        self.indicators = []
        
        for ind_id, config in EPI_SOURCES.items():
            self.indicators.append(IndicatorMeta(
                id=ind_id,
                name=config["name"],
                domain=self.domain,
                source="delphi",
                category=config["category"],
                frequency=config["frequency"],
                description=config.get("description", "")
            ))
        
        logger.info(f"Discovered {len(self.indicators)} epidemiology indicators")
        return self.indicators
    
    def _fetch_fluview(self, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch from FluView endpoint."""
        url = f"{self.base_url}/fluview/"
        
        # Build epiweek range (last 20 years)
        current_year = datetime.now().year
        start_week = (current_year - 20) * 100 + 1  # 20 years ago, week 1
        end_week = current_year * 100 + 52
        
        request_params = {
            "regions": params.get("regions", "nat"),
            "epiweeks": f"{start_week}-{end_week}"
        }
        
        try:
            response = self.session.get(url, params=request_params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result") != 1:
                logger.warning(f"FluView API error: {data.get('message', 'Unknown')}")
                return None
            
            epidata = data.get("epidata", [])
            if not epidata:
                return None
            
            records = []
            for row in epidata:
                epiweek = row.get("epiweek")
                ili = row.get("wili") or row.get("ili")  # weighted or unweighted
                
                if epiweek and ili is not None:
                    date = epiweek_to_date(epiweek)
                    records.append({"date": date, "value": float(ili)})
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            return df.sort_values("date").reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"FluView fetch failed: {e}")
            return None
    
    def _fetch_flusurv(self, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch from FluSurv-NET endpoint."""
        url = f"{self.base_url}/flusurv/"
        
        current_year = datetime.now().year
        start_week = (current_year - 10) * 100 + 1
        end_week = current_year * 100 + 52
        
        request_params = {
            "locations": params.get("locations", "network_all"),
            "epiweeks": f"{start_week}-{end_week}"
        }
        
        try:
            response = self.session.get(url, params=request_params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result") != 1:
                return None
            
            epidata = data.get("epidata", [])
            if not epidata:
                return None
            
            records = []
            for row in epidata:
                epiweek = row.get("epiweek")
                rate = row.get("rate_overall") or row.get("rate")
                
                if epiweek and rate is not None:
                    date = epiweek_to_date(epiweek)
                    records.append({"date": date, "value": float(rate)})
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            return df.sort_values("date").reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"FluSurv fetch failed: {e}")
            return None
    
    def _fetch_wiki(self, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch Wikipedia access data."""
        url = f"{self.base_url}/wiki/"
        
        # Last 5 years of daily data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        request_params = {
            "articles": params.get("articles", "influenza"),
            "dates": f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
        }
        
        try:
            response = self.session.get(url, params=request_params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result") != 1:
                return None
            
            epidata = data.get("epidata", [])
            if not epidata:
                return None
            
            records = []
            for row in epidata:
                date_str = str(row.get("date"))
                count = row.get("count") or row.get("value")
                
                if date_str and count is not None:
                    try:
                        date = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
                        records.append({"date": date, "value": float(count)})
                    except:
                        continue
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            return df.sort_values("date").reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Wiki fetch failed: {e}")
            return None
    
    def fetch_one(self, indicator_id: str) -> Optional[pd.DataFrame]:
        """Fetch a single epidemiology indicator."""
        if indicator_id not in EPI_SOURCES:
            logger.error(f"Unknown indicator: {indicator_id}")
            return None
        
        config = EPI_SOURCES[indicator_id]
        endpoint = config["endpoint"]
        params = config.get("params", {})
        
        # Route to appropriate fetcher
        if endpoint == "fluview":
            df = self._fetch_fluview(params)
        elif endpoint == "flusurv":
            df = self._fetch_flusurv(params)
        elif endpoint == "wiki":
            df = self._fetch_wiki(params)
        else:
            logger.error(f"Unknown endpoint: {endpoint}")
            return None
        
        if df is not None and len(df) > 0:
            logger.info(f"Fetched {indicator_id}: {len(df)} rows, "
                       f"{df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    
    def get_categories(self) -> List[str]:
        """Get available categories."""
        return list(set(c["category"] for c in EPI_SOURCES.values()))


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = EpidemiologyAgent()
    
    # Discover
    indicators = agent.discover()
    print(f"\nDiscovered {len(indicators)} indicators:")
    for cat in agent.get_categories():
        cat_inds = [i for i in indicators if i.category == cat]
        print(f"  {cat}: {len(cat_inds)}")
    
    # Fetch one
    print("\nFetching ILI_NATIONAL...")
    df = agent.fetch_one("ILI_NATIONAL")
    if df is not None:
        print(f"  Rows: {len(df)}")
        print(f"  Range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Sample:\n{df.tail()}")
