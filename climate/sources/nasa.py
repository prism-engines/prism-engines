"""
NASA Climate Data Source Handler

STATUS: SKELETON - Placeholder only, no active fetch logic.

This module will handle data fetching from NASA climate data services.

Planned data types:
    - GISS Surface Temperature Analysis (GISTEMP)
    - MERRA-2 Reanalysis
    - Earth Observing System Data
"""

from typing import Optional, Dict, Any
from . import BaseClimateSource


class NASAGISSSource(BaseClimateSource):
    """
    NASA GISS temperature data source handler.

    STATUS: SKELETON - No active fetch logic.
    """

    # Data URLs (for future reference)
    URLS = {
        "gistemp_global": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        "gistemp_nh": "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv",
        "gistemp_sh": "https://data.giss.nasa.gov/gistemp/tabledata_v4/SH.Ts+dSST.csv",
        "gistemp_zones": "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv",
    }

    # Data products
    PRODUCTS = {
        "GLB": "Global Mean",
        "NH": "Northern Hemisphere",
        "SH": "Southern Hemisphere",
        "zones": "Zonal Means",
    }

    def __init__(self):
        """Initialize NASA GISS source handler."""
        super().__init__()
        self._source_name = "nasa_giss"

    def fetch(
        self,
        product: str = "GLB",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch data from NASA GISS.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            product: Data product (GLB, NH, SH, zones)

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "NASA GISS fetch not implemented. This module is in skeleton state. "
            f"Would fetch: product={product}"
        )

    def get_temperature_anomaly(self, **kwargs) -> Dict[str, Any]:
        """
        Get global temperature anomaly data.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "NASA GISS temperature anomaly fetch not implemented. "
            "This module is in skeleton state."
        )


# Singleton instance (inactive)
_nasa_source: Optional[NASAGISSSource] = None


def get_nasa_source() -> NASAGISSSource:
    """
    Get the NASA GISS source handler instance.

    Returns:
        NASAGISSSource instance (inactive skeleton)
    """
    global _nasa_source
    if _nasa_source is None:
        _nasa_source = NASAGISSSource()
    return _nasa_source
