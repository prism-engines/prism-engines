"""
PRISM-CLIMATE Transforms - Data Transformation Utilities

This submodule will contain data transformation utilities
for climate data processing.

STATUS: SKELETON - Placeholder only, no active logic.

Planned transformations:
    - Temporal alignment (daily/monthly/yearly aggregation)
    - Spatial aggregation (regional averages)
    - Normalization (z-score, min-max, percentile)
    - Interpolation (missing data filling)
    - Resampling (frequency conversion)
    - Deseasonalization
"""

__all__ = [
    "BaseTransform",
    "AVAILABLE_TRANSFORMS",
]

# Registry of available transforms (placeholder)
AVAILABLE_TRANSFORMS = {
    "temporal_resample": {
        "name": "Temporal Resampling",
        "status": "planned",
        "description": "Resample data to different temporal frequencies",
    },
    "spatial_aggregate": {
        "name": "Spatial Aggregation",
        "status": "planned",
        "description": "Aggregate spatial data to regions",
    },
    "normalize": {
        "name": "Normalization",
        "status": "planned",
        "description": "Normalize values to standard scales",
    },
    "interpolate": {
        "name": "Interpolation",
        "status": "planned",
        "description": "Fill missing values",
    },
    "deseasonalize": {
        "name": "Deseasonalization",
        "status": "planned",
        "description": "Remove seasonal components",
    },
}


class BaseTransform:
    """
    Abstract base class for data transforms.

    STATUS: SKELETON - No implementation, interface definition only.

    All transforms will inherit from this class
    and implement the required methods.
    """

    def __init__(self):
        """Initialize the transform."""
        self._transform_name = "base"

    @property
    def transform_name(self) -> str:
        """Get the transform name."""
        return self._transform_name

    def transform(self, data, **kwargs):
        """
        Apply the transformation to data.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            data: Input data

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            f"Transform '{self._transform_name}' not implemented. "
            "This module is in skeleton state."
        )

    def inverse_transform(self, data, **kwargs):
        """
        Apply inverse transformation (if applicable).

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            data: Transformed data

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            f"Inverse transform '{self._transform_name}' not implemented. "
            "This module is in skeleton state."
        )
