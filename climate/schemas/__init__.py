"""
PRISM-CLIMATE Schemas - Data Validation Schemas

This submodule will contain data validation schemas
for climate data structures.

STATUS: SKELETON - Placeholder only, no active validation.

Planned schemas:
    - Climate data point schema
    - Time series schema
    - Indicator output schema
    - Configuration schema
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

__all__ = [
    "ClimateDataPoint",
    "ClimateTimeSeries",
    "IndicatorOutput",
    "SCHEMA_REGISTRY",
]


@dataclass
class ClimateDataPoint:
    """
    Schema for a single climate data point.

    STATUS: SKELETON - Data structure definition only.
    """

    timestamp: str  # ISO format datetime
    value: float
    unit: str
    source: str
    quality_flag: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """
        Validate the data point.

        STATUS: NOT IMPLEMENTED - Returns True always.
        """
        # Placeholder - no actual validation
        return True


@dataclass
class ClimateTimeSeries:
    """
    Schema for a climate time series.

    STATUS: SKELETON - Data structure definition only.
    """

    series_id: str
    variable: str
    unit: str
    source: str
    frequency: str  # D, W, M, Q, Y
    data_points: List[ClimateDataPoint]
    start_date: str
    end_date: str
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """
        Validate the time series.

        STATUS: NOT IMPLEMENTED - Returns True always.
        """
        # Placeholder - no actual validation
        return True

    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.data_points)


@dataclass
class IndicatorOutput:
    """
    Schema for indicator computation output.

    STATUS: SKELETON - Data structure definition only.
    """

    indicator_name: str
    indicator_version: str
    computation_timestamp: str
    input_sources: List[str]
    output_values: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """
        Validate the indicator output.

        STATUS: NOT IMPLEMENTED - Returns True always.
        """
        # Placeholder - no actual validation
        return True


# Registry of available schemas
SCHEMA_REGISTRY = {
    "climate_data_point": {
        "class": ClimateDataPoint,
        "description": "Single climate observation",
        "required_fields": ["timestamp", "value", "unit", "source"],
    },
    "climate_time_series": {
        "class": ClimateTimeSeries,
        "description": "Time series of climate observations",
        "required_fields": ["series_id", "variable", "unit", "source", "frequency"],
    },
    "indicator_output": {
        "class": IndicatorOutput,
        "description": "Output from indicator computation",
        "required_fields": ["indicator_name", "indicator_version", "output_values"],
    },
}
