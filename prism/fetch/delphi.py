"""
PRISM Delphi Epidata Fetcher

Fetches epidemiological data from CMU Delphi Epidata API.
No API key required for most endpoints.

DOMAIN TRANSLATION:
    Finance             ->  Epidemiology
    ─────────────────────────────────────────
    Indicator           ->  Disease signal / Region
    Daily return        ->  Weekly ILI rate change
    Volatility          ->  Epidemic intensity variance
    Correlation         ->  Disease co-occurrence / Geographic spread
    Regime change       ->  Epidemic onset / Peak detection
    Hidden mass         ->  Unreported cases / Surveillance lag

SIGNAL TYPES:
    ILI         - Influenza-like illness rate (%)
    HOSP        - Hospitalization rate
    WIKI        - Wikipedia article views (proxy for awareness)
    SEARCH      - Search trends (Google, etc.)

Usage:
    from prism.fetch.delphi import DelphiFetcher, REGIONS, SIGNAL_TYPES

    fetcher = DelphiFetcher()
    results = fetcher.fetch_region_all("NATIONAL", start_date, end_date)
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# REGION DEFINITIONS
# =============================================================================

@dataclass
class EpiRegion:
    """Definition of an epidemiological surveillance region."""
    name: str
    description: str
    region_code: str  # Code used in API calls
    region_type: str  # 'national', 'hhs', 'state', 'metro'
    population: Optional[int] = None
    notable_events: Optional[List[str]] = None


REGIONS: Dict[str, EpiRegion] = {
    # National
    "NATIONAL": EpiRegion(
        name="National",
        description="United States National",
        region_code="nat",
        region_type="national",
        population=330_000_000,
        notable_events=[
            "2009 H1N1 Pandemic",
            "2017-18 Severe Flu Season",
            "2020-21 COVID-19 Pandemic",
        ]
    ),

    # HHS Regions (10 regions)
    "HHS1": EpiRegion(
        name="HHS Region 1",
        description="New England (CT, ME, MA, NH, RI, VT)",
        region_code="hhs1",
        region_type="hhs",
        population=14_800_000,
    ),
    "HHS2": EpiRegion(
        name="HHS Region 2",
        description="NY, NJ, Puerto Rico, Virgin Islands",
        region_code="hhs2",
        region_type="hhs",
        population=27_200_000,
    ),
    "HHS3": EpiRegion(
        name="HHS Region 3",
        description="Mid-Atlantic (DE, DC, MD, PA, VA, WV)",
        region_code="hhs3",
        region_type="hhs",
        population=30_600_000,
    ),
    "HHS4": EpiRegion(
        name="HHS Region 4",
        description="Southeast (AL, FL, GA, KY, MS, NC, SC, TN)",
        region_code="hhs4",
        region_type="hhs",
        population=65_400_000,
    ),
    "HHS5": EpiRegion(
        name="HHS Region 5",
        description="Midwest (IL, IN, MI, MN, OH, WI)",
        region_code="hhs5",
        region_type="hhs",
        population=52_500_000,
    ),
    "HHS6": EpiRegion(
        name="HHS Region 6",
        description="South Central (AR, LA, NM, OK, TX)",
        region_code="hhs6",
        region_type="hhs",
        population=40_300_000,
    ),
    "HHS7": EpiRegion(
        name="HHS Region 7",
        description="Central (IA, KS, MO, NE)",
        region_code="hhs7",
        region_type="hhs",
        population=13_900_000,
    ),
    "HHS8": EpiRegion(
        name="HHS Region 8",
        description="Mountain (CO, MT, ND, SD, UT, WY)",
        region_code="hhs8",
        region_type="hhs",
        population=12_100_000,
    ),
    "HHS9": EpiRegion(
        name="HHS Region 9",
        description="Pacific (AZ, CA, HI, NV)",
        region_code="hhs9",
        region_type="hhs",
        population=51_600_000,
    ),
    "HHS10": EpiRegion(
        name="HHS Region 10",
        description="Pacific Northwest (AK, ID, OR, WA)",
        region_code="hhs10",
        region_type="hhs",
        population=13_500_000,
    ),
}


# Signal types (equivalent to USGS indicator types)
SIGNAL_TYPES = [
    "ILI",      # Influenza-like illness rate
    "WILI",     # Weighted ILI
    "HOSP",     # Hospitalization rate (where available)
]


# =============================================================================
# API CONFIGURATION
# =============================================================================

DELPHI_BASE_URL = "https://api.delphi.cmu.edu/epidata"


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
# EPIWEEK UTILITIES
# =============================================================================

def epiweek_to_date(epiweek: int) -> date:
    """
    Convert CDC epiweek (YYYYWW) to date.

    Args:
        epiweek: Integer like 202301 (year 2023, week 1)

    Returns:
        Date for the start of that epiweek (Sunday)
    """
    year = epiweek // 100
    week = epiweek % 100

    # First day of the year
    jan1 = datetime(year, 1, 1)

    # Find first Sunday (CDC epiweek starts on Sunday)
    days_to_sunday = (6 - jan1.weekday()) % 7
    first_sunday = jan1 + timedelta(days=days_to_sunday)

    # Add weeks
    target_date = first_sunday + timedelta(weeks=week - 1)

    return target_date.date()


def date_to_epiweek(d: date) -> int:
    """
    Convert date to CDC epiweek (YYYYWW).
    """
    # Simplified calculation
    year = d.year
    jan1 = datetime(year, 1, 1)
    days_to_sunday = (6 - jan1.weekday()) % 7
    first_sunday = jan1 + timedelta(days=days_to_sunday)

    if d < first_sunday.date():
        # Belongs to previous year's last week
        return (year - 1) * 100 + 52

    days_since = (d - first_sunday.date()).days
    week = days_since // 7 + 1

    if week > 52:
        week = 52

    return year * 100 + week


# =============================================================================
# DELPHI FETCHER
# =============================================================================

class DelphiFetcher:
    """
    Fetch epidemiological data from CMU Delphi Epidata API.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PRISM-Observatory/1.0 (Research)'
        })
        self.base_url = DELPHI_BASE_URL
        self.cache_dir = cache_dir
        self._last_request_time = 0
        self._min_request_interval = 0.5  # 500ms between requests

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def fetch_fluview(
        self,
        region_code: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch FluView ILI data for a region.

        Returns DataFrame with columns: date, ili, wili, num_ili, num_patients
        """
        self._rate_limit()

        # Convert dates to epiweeks
        start_epiweek = date_to_epiweek(start_date)
        end_epiweek = date_to_epiweek(end_date)

        url = f"{self.base_url}/fluview/"
        params = {
            "regions": region_code,
            "epiweeks": f"{start_epiweek}-{end_epiweek}"
        }

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if data.get("result") != 1:
                logger.warning(f"FluView API returned: {data.get('message', 'Unknown error')}")
                return None

            epidata = data.get("epidata", [])
            if not epidata:
                return None

            records = []
            for row in epidata:
                epiweek = row.get("epiweek")
                if not epiweek:
                    continue

                records.append({
                    "date": epiweek_to_date(epiweek),
                    "ili": row.get("ili"),
                    "wili": row.get("wili"),
                    "num_ili": row.get("num_ili"),
                    "num_patients": row.get("num_patients"),
                    "num_providers": row.get("num_providers"),
                })

            if not records:
                return None

            df = pd.DataFrame(records)
            # Deduplicate by date (epiweek boundary issues can create duplicates)
            df = df.drop_duplicates(subset=["date"], keep="first")
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except requests.RequestException as e:
            logger.error(f"FluView request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"FluView parse failed: {e}")
            return None

    def fetch_region_signal(
        self,
        region_name: str,
        signal_type: str,
        start_date: date,
        end_date: date,
    ) -> FetchResult:
        """
        Fetch a specific signal for a region.

        Args:
            region_name: Key from REGIONS dict (e.g., "NATIONAL", "HHS1")
            signal_type: One of SIGNAL_TYPES (e.g., "ILI", "WILI")
            start_date: Start date
            end_date: End date

        Returns:
            FetchResult with data DataFrame containing (date, value) columns
        """
        if region_name not in REGIONS:
            return FetchResult(
                indicator_id=f"{region_name}_{signal_type}",
                success=False,
                error=f"Unknown region: {region_name}"
            )

        region = REGIONS[region_name]
        indicator_id = f"{region_name}_{signal_type}"

        # Fetch raw data
        df = self.fetch_fluview(region.region_code, start_date, end_date)

        if df is None or df.empty:
            return FetchResult(
                indicator_id=indicator_id,
                success=False,
                error="No data returned"
            )

        # Extract the requested signal
        signal_col = signal_type.lower()
        if signal_col not in df.columns:
            return FetchResult(
                indicator_id=indicator_id,
                success=False,
                error=f"Signal {signal_type} not available"
            )

        result_df = df[["date", signal_col]].copy()
        result_df.columns = ["date", "value"]
        result_df = result_df.dropna(subset=["value"])

        return FetchResult(
            indicator_id=indicator_id,
            success=True,
            data=result_df,
            rows=len(result_df)
        )

    def fetch_region_all(
        self,
        region_name: str,
        start_date: date,
        end_date: date,
    ) -> List[FetchResult]:
        """
        Fetch all signal types for a region.

        Returns list of FetchResults, one per signal type.
        """
        results = []

        # Fetch raw data once
        if region_name not in REGIONS:
            for signal_type in SIGNAL_TYPES:
                results.append(FetchResult(
                    indicator_id=f"{region_name}_{signal_type}",
                    success=False,
                    error=f"Unknown region: {region_name}"
                ))
            return results

        region = REGIONS[region_name]
        df = self.fetch_fluview(region.region_code, start_date, end_date)

        if df is None or df.empty:
            for signal_type in SIGNAL_TYPES:
                results.append(FetchResult(
                    indicator_id=f"{region_name}_{signal_type}",
                    success=False,
                    error="No data returned"
                ))
            return results

        # Extract each signal
        for signal_type in SIGNAL_TYPES:
            indicator_id = f"{region_name}_{signal_type}"
            signal_col = signal_type.lower()

            if signal_col not in df.columns:
                results.append(FetchResult(
                    indicator_id=indicator_id,
                    success=False,
                    error=f"Signal {signal_type} not in data"
                ))
                continue

            result_df = df[["date", signal_col]].copy()
            result_df.columns = ["date", "value"]
            result_df = result_df.dropna(subset=["value"])

            if result_df.empty:
                results.append(FetchResult(
                    indicator_id=indicator_id,
                    success=False,
                    error="No valid values"
                ))
            else:
                results.append(FetchResult(
                    indicator_id=indicator_id,
                    success=True,
                    data=result_df,
                    rows=len(result_df)
                ))

        return results


def get_all_indicator_ids() -> List[str]:
    """Get all possible indicator IDs (region × signal combinations)."""
    ids = []
    for region_name in REGIONS:
        for signal_type in SIGNAL_TYPES:
            ids.append(f"{region_name}_{signal_type}")
    return ids


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Fetch Delphi epidemiological data")
    parser.add_argument("--region", default="NATIONAL", help="Region to fetch")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    fetcher = DelphiFetcher()

    print(f"Fetching {args.region} from {start_date} to {end_date}")
    results = fetcher.fetch_region_all(args.region, start_date, end_date)

    for r in results:
        if r.success:
            print(f"  {r.indicator_id}: {r.rows} weeks")
            if r.data is not None:
                print(f"    Range: {r.data['date'].min()} to {r.data['date'].max()}")
        else:
            print(f"  {r.indicator_id}: FAILED - {r.error}")
