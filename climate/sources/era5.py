"""
ERA5 Reanalysis Data Source Handler

STATUS: SKELETON - Placeholder only, no active fetch logic.

This module will handle data fetching from ECMWF ERA5 reanalysis
via the Copernicus Climate Data Store.

Planned data types:
    - Surface temperature
    - Precipitation
    - Wind speed/direction
    - Sea surface temperature
    - Atmospheric pressure
"""

from typing import Optional, Dict, Any, List
from . import BaseClimateSource


class ERA5Source(BaseClimateSource):
    """
    ECMWF ERA5 reanalysis data source handler.

    STATUS: SKELETON - No active fetch logic.
    """

    # CDS API info (for future reference)
    CDS_URL = "https://cds.climate.copernicus.eu/api/v2"

    # Available variables
    VARIABLES = {
        "2m_temperature": "2 metre temperature",
        "total_precipitation": "Total precipitation",
        "10m_u_component_of_wind": "10 metre U wind component",
        "10m_v_component_of_wind": "10 metre V wind component",
        "sea_surface_temperature": "Sea surface temperature",
        "surface_pressure": "Surface pressure",
        "mean_sea_level_pressure": "Mean sea level pressure",
    }

    # Temporal resolutions
    RESOLUTIONS = ["hourly", "daily", "monthly"]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ERA5 source handler.

        Args:
            api_key: Copernicus CDS API key (not used in skeleton)
        """
        super().__init__()
        self._source_name = "era5"
        self._api_key = api_key  # Stored but not used

    def fetch(
        self,
        variables: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resolution: str = "monthly",
        area: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch data from ERA5.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            variables: List of variable names
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resolution: Temporal resolution
            area: Bounding box [North, West, South, East]

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "ERA5 fetch not implemented. This module is in skeleton state. "
            f"Would fetch: variables={variables}, resolution={resolution}, "
            f"dates={start_date} to {end_date}"
        )

    def list_variables(self) -> Dict[str, str]:
        """
        List available ERA5 variables.

        Returns:
            Dictionary of variable IDs and descriptions
        """
        return self.VARIABLES.copy()


# Singleton instance (inactive)
_era5_source: Optional[ERA5Source] = None


def get_era5_source() -> ERA5Source:
    """
    Get the ERA5 source handler instance.

    Returns:
        ERA5Source instance (inactive skeleton)
    """
    global _era5_source
    if _era5_source is None:
        _era5_source = ERA5Source()
    return _era5_source
