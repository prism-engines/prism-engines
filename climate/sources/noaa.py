"""
NOAA Climate Data Source Handler

STATUS: SKELETON - Placeholder only, no active fetch logic.

This module will handle data fetching from NOAA Climate Data Online (CDO)
and related NOAA data services.

Planned data types:
    - Global Historical Climatology Network (GHCN)
    - Integrated Surface Database (ISD)
    - Climate Normals
    - Storm Events Database
"""

from typing import Optional, Dict, Any
from . import BaseClimateSource


class NOAASource(BaseClimateSource):
    """
    NOAA Climate Data Online source handler.

    STATUS: SKELETON - No active fetch logic.
    """

    # API endpoints (for future reference)
    ENDPOINTS = {
        "cdo_api": "https://www.ncdc.noaa.gov/cdo-web/api/v2/",
        "ghcn_ftp": "ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/",
    }

    # Dataset identifiers
    DATASETS = {
        "GHCND": "Global Historical Climatology Network - Daily",
        "GSOM": "Global Summary of the Month",
        "GSOY": "Global Summary of the Year",
        "NORMAL_DLY": "Climate Normals Daily",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NOAA source handler.

        Args:
            api_key: NOAA CDO API key (not used in skeleton)
        """
        super().__init__()
        self._source_name = "noaa_cdo"
        self._api_key = api_key  # Stored but not used

    def fetch(
        self,
        dataset: str = "GHCND",
        station_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch data from NOAA CDO.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            dataset: Dataset identifier (GHCND, GSOM, etc.)
            station_id: Weather station ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "NOAA fetch not implemented. This module is in skeleton state. "
            f"Would fetch: dataset={dataset}, station={station_id}, "
            f"dates={start_date} to {end_date}"
        )

    def get_stations(self, **kwargs) -> Dict[str, Any]:
        """
        Get available weather stations.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "NOAA station lookup not implemented. This module is in skeleton state."
        )


# Singleton instance (inactive)
_noaa_source: Optional[NOAASource] = None


def get_noaa_source() -> NOAASource:
    """
    Get the NOAA source handler instance.

    Returns:
        NOAASource instance (inactive skeleton)
    """
    global _noaa_source
    if _noaa_source is None:
        _noaa_source = NOAASource()
    return _noaa_source
