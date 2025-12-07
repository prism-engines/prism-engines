"""
PRISM-CLIMATE Indicators - Climate to Scalar Indicator Converters

This submodule will contain transformers that convert raw climate data
into scalar indicators suitable for PRISM analysis pipelines.

STATUS: SKELETON - Placeholder only, no active logic.

Planned indicator types:
    - Temperature anomaly indices
    - Precipitation indices (SPI, SPEI)
    - Climate volatility measures
    - Extreme weather event scores
    - Climate risk indices

Each indicator will:
    - Accept standardized climate data
    - Output scalar time series
    - Be compatible with PRISM temporal alignment
"""

__all__ = [
    "BaseClimateIndicator",
    "AVAILABLE_INDICATORS",
]

# Registry of available climate indicators (placeholder)
AVAILABLE_INDICATORS = {
    "temp_anomaly": {
        "name": "Temperature Anomaly Index",
        "status": "planned",
        "description": "Deviation from baseline temperature",
        "output_range": [-5.0, 5.0],
        "unit": "degrees_c",
    },
    "precipitation_index": {
        "name": "Standardized Precipitation Index",
        "status": "planned",
        "description": "Standardized precipitation deviation",
        "output_range": [-3.0, 3.0],
        "unit": "standard_deviations",
    },
    "climate_volatility": {
        "name": "Climate Volatility Index",
        "status": "planned",
        "description": "Rolling volatility of climate variables",
        "output_range": [0.0, 1.0],
        "unit": "normalized",
    },
    "extreme_weather_score": {
        "name": "Extreme Weather Score",
        "status": "planned",
        "description": "Composite score of extreme weather events",
        "output_range": [0.0, 100.0],
        "unit": "score",
    },
    "climate_risk_index": {
        "name": "Climate Risk Index",
        "status": "planned",
        "description": "Aggregate climate risk scoring",
        "output_range": [0.0, 10.0],
        "unit": "risk_score",
    },
}


class BaseClimateIndicator:
    """
    Abstract base class for climate indicators.

    STATUS: SKELETON - No implementation, interface definition only.

    All climate indicators will inherit from this class
    and implement the required methods.
    """

    def __init__(self):
        """Initialize the climate indicator."""
        self._indicator_name = "base"
        self._is_active = False

    @property
    def indicator_name(self) -> str:
        """Get the indicator name."""
        return self._indicator_name

    @property
    def is_active(self) -> bool:
        """Check if indicator is active."""
        return self._is_active

    def compute(self, data, **kwargs):
        """
        Compute the indicator from climate data.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            data: Input climate data

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            f"Climate indicator '{self._indicator_name}' compute not implemented. "
            "This module is in skeleton state."
        )

    def validate_input(self, data) -> bool:
        """
        Validate input data for indicator computation.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            f"Climate indicator '{self._indicator_name}' validation not implemented. "
            "This module is in skeleton state."
        )

    def get_metadata(self) -> dict:
        """
        Get indicator metadata.

        Returns:
            Dictionary with indicator metadata
        """
        return {
            "name": self._indicator_name,
            "is_active": self._is_active,
            "status": "skeleton",
        }
