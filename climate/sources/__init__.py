"""
PRISM-CLIMATE Sources - Climate Data Source Handlers

This submodule will contain handlers for various climate data sources.

STATUS: SKELETON - Placeholder only, no active fetch logic.

Planned sources:
    - NOAA Climate Data Online (CDO)
    - NASA GISS temperature data
    - ECMWF ERA5 reanalysis
    - Copernicus Climate Data Store
    - USGS water data
    - Weather station networks

Each source handler will implement:
    - Data fetching (when activated)
    - Response validation
    - Data normalization
    - Caching/checkpointing
"""

__all__ = [
    "BaseClimateSource",
    "AVAILABLE_SOURCES",
]

# Registry of available climate sources (placeholder)
AVAILABLE_SOURCES = {
    "noaa_cdo": {
        "name": "NOAA Climate Data Online",
        "status": "planned",
        "url": "https://www.ncdc.noaa.gov/cdo-web/",
        "data_types": ["temperature", "precipitation", "wind", "pressure"],
    },
    "nasa_giss": {
        "name": "NASA GISS Temperature",
        "status": "planned",
        "url": "https://data.giss.nasa.gov/gistemp/",
        "data_types": ["global_temperature_anomaly"],
    },
    "era5": {
        "name": "ECMWF ERA5 Reanalysis",
        "status": "planned",
        "url": "https://cds.climate.copernicus.eu/",
        "data_types": ["temperature", "precipitation", "wind", "humidity"],
    },
    "mauna_loa": {
        "name": "Mauna Loa CO2",
        "status": "planned",
        "url": "https://gml.noaa.gov/ccgg/trends/",
        "data_types": ["co2_concentration"],
    },
    "sea_level": {
        "name": "Global Sea Level",
        "status": "planned",
        "url": "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/",
        "data_types": ["sea_level_rise"],
    },
}


class BaseClimateSource:
    """
    Abstract base class for climate data sources.

    STATUS: SKELETON - No implementation, interface definition only.

    All climate source handlers will inherit from this class
    and implement the required methods.
    """

    def __init__(self):
        """Initialize the climate source handler."""
        self._is_active = False
        self._source_name = "base"

    @property
    def is_active(self) -> bool:
        """Check if source is actively fetching data."""
        return self._is_active

    @property
    def source_name(self) -> str:
        """Get the source name."""
        return self._source_name

    def fetch(self, **kwargs):
        """
        Fetch data from the climate source.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            f"Climate source '{self._source_name}' fetch not implemented. "
            "This module is in skeleton state."
        )

    def validate(self, data) -> bool:
        """
        Validate fetched data.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            f"Climate source '{self._source_name}' validation not implemented. "
            "This module is in skeleton state."
        )
