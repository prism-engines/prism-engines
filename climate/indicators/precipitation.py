"""
Precipitation-Based Climate Indicators

STATUS: SKELETON - Placeholder only, no active logic.

This module will contain precipitation-derived indicators:
    - Standardized Precipitation Index (SPI)
    - Standardized Precipitation Evapotranspiration Index (SPEI)
    - Drought indices
    - Flood risk indicators
"""

from typing import Optional, Dict, Any
from . import BaseClimateIndicator


class StandardizedPrecipitationIndex(BaseClimateIndicator):
    """
    Standardized Precipitation Index (SPI).

    SPI is a normalized index representing precipitation deviation
    from historical mean, useful for drought/flood assessment.

    STATUS: SKELETON - No active logic.
    """

    # Standard SPI timescales (months)
    TIMESCALES = [1, 3, 6, 9, 12, 24]

    def __init__(self, timescale: int = 3):
        """
        Initialize SPI indicator.

        Args:
            timescale: SPI timescale in months (1, 3, 6, 12, etc.)
        """
        super().__init__()
        self._indicator_name = f"spi_{timescale}"
        self._timescale = timescale

    def compute(
        self,
        precipitation_data,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute SPI from precipitation data.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            precipitation_data: Time series of precipitation values

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            f"SPI-{self._timescale} computation not implemented. "
            "This module is in skeleton state."
        )

    @staticmethod
    def interpret_spi(value: float) -> str:
        """
        Interpret SPI value.

        Args:
            value: SPI value

        Returns:
            Human-readable interpretation
        """
        if value >= 2.0:
            return "Extremely wet"
        elif value >= 1.5:
            return "Very wet"
        elif value >= 1.0:
            return "Moderately wet"
        elif value > -1.0:
            return "Near normal"
        elif value > -1.5:
            return "Moderately dry"
        elif value > -2.0:
            return "Severely dry"
        else:
            return "Extremely dry"


class DroughtIndex(BaseClimateIndicator):
    """
    Composite drought index.

    Combines multiple factors for drought assessment.

    STATUS: SKELETON - No active logic.
    """

    def __init__(self):
        """Initialize drought index indicator."""
        super().__init__()
        self._indicator_name = "drought_index"

    def compute(
        self,
        precipitation_data,
        temperature_data=None,
        soil_moisture_data=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute drought index.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            precipitation_data: Time series of precipitation
            temperature_data: Optional temperature data
            soil_moisture_data: Optional soil moisture data

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "Drought index computation not implemented. "
            "This module is in skeleton state."
        )


class FloodRiskIndicator(BaseClimateIndicator):
    """
    Flood risk indicator.

    Assesses flood risk based on precipitation patterns.

    STATUS: SKELETON - No active logic.
    """

    def __init__(self, threshold_percentile: float = 95.0):
        """
        Initialize flood risk indicator.

        Args:
            threshold_percentile: Precipitation percentile for extreme events
        """
        super().__init__()
        self._indicator_name = "flood_risk"
        self._threshold_percentile = threshold_percentile

    def compute(
        self,
        precipitation_data,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute flood risk indicator.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            precipitation_data: Time series of precipitation values

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "Flood risk computation not implemented. "
            "This module is in skeleton state. "
            f"Would use {self._threshold_percentile}th percentile threshold"
        )
