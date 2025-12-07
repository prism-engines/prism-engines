"""
Temperature-Based Climate Indicators

STATUS: SKELETON - Placeholder only, no active logic.

This module will contain temperature-derived indicators:
    - Temperature anomaly index
    - Heating/cooling degree days
    - Temperature volatility
    - Extreme temperature events
"""

from typing import Optional, Dict, Any
from . import BaseClimateIndicator


class TemperatureAnomalyIndicator(BaseClimateIndicator):
    """
    Temperature anomaly indicator.

    Computes deviation from historical baseline temperature.

    STATUS: SKELETON - No active logic.
    """

    # Baseline period for anomaly calculation
    DEFAULT_BASELINE_START = "1951-01-01"
    DEFAULT_BASELINE_END = "1980-12-31"

    def __init__(
        self,
        baseline_start: Optional[str] = None,
        baseline_end: Optional[str] = None,
    ):
        """
        Initialize temperature anomaly indicator.

        Args:
            baseline_start: Start of baseline period (YYYY-MM-DD)
            baseline_end: End of baseline period (YYYY-MM-DD)
        """
        super().__init__()
        self._indicator_name = "temp_anomaly"
        self._baseline_start = baseline_start or self.DEFAULT_BASELINE_START
        self._baseline_end = baseline_end or self.DEFAULT_BASELINE_END

    def compute(
        self,
        temperature_data,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute temperature anomaly from raw temperature data.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            temperature_data: Time series of temperature values

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "Temperature anomaly computation not implemented. "
            "This module is in skeleton state. "
            f"Would use baseline: {self._baseline_start} to {self._baseline_end}"
        )


class DegreeDaysIndicator(BaseClimateIndicator):
    """
    Heating and cooling degree days indicator.

    STATUS: SKELETON - No active logic.
    """

    # Standard base temperatures
    DEFAULT_HEATING_BASE = 65.0  # Fahrenheit
    DEFAULT_COOLING_BASE = 65.0  # Fahrenheit

    def __init__(
        self,
        heating_base: float = DEFAULT_HEATING_BASE,
        cooling_base: float = DEFAULT_COOLING_BASE,
    ):
        """
        Initialize degree days indicator.

        Args:
            heating_base: Base temperature for heating degree days
            cooling_base: Base temperature for cooling degree days
        """
        super().__init__()
        self._indicator_name = "degree_days"
        self._heating_base = heating_base
        self._cooling_base = cooling_base

    def compute(
        self,
        temperature_data,
        indicator_type: str = "both",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute heating/cooling degree days.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            temperature_data: Time series of temperature values
            indicator_type: 'heating', 'cooling', or 'both'

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "Degree days computation not implemented. "
            "This module is in skeleton state. "
            f"Would compute {indicator_type} degree days"
        )


class TemperatureVolatilityIndicator(BaseClimateIndicator):
    """
    Temperature volatility indicator.

    Measures rolling volatility of temperature values.

    STATUS: SKELETON - No active logic.
    """

    DEFAULT_WINDOW = 30  # days

    def __init__(self, window: int = DEFAULT_WINDOW):
        """
        Initialize temperature volatility indicator.

        Args:
            window: Rolling window size in days
        """
        super().__init__()
        self._indicator_name = "temp_volatility"
        self._window = window

    def compute(
        self,
        temperature_data,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute temperature volatility.

        STATUS: NOT IMPLEMENTED - Placeholder only.

        Args:
            temperature_data: Time series of temperature values

        Raises:
            NotImplementedError: Always, as this is a skeleton.
        """
        raise NotImplementedError(
            "Temperature volatility computation not implemented. "
            "This module is in skeleton state. "
            f"Would use {self._window}-day rolling window"
        )
