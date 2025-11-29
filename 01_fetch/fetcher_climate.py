"""
Climate Data Fetcher - NOAA, NASA, and other climate data sources
"""

from pathlib import Path
from typing import Optional, Any
import pandas as pd
import requests
import logging

from .fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)


class ClimateFetcher(BaseFetcher):
    """
    Fetcher for climate data from various sources.

    Supports:
    - NOAA Climate Data Online (CDO)
    - NASA GISS temperature data
    - Mauna Loa CO2 data
    """

    # Data source URLs
    SOURCES = {
        "nasa_giss": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        "mauna_loa_co2": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
        "sea_level": "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gbl_free_txj1j2_90.csv",
    }

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize Climate fetcher.

        Args:
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(checkpoint_dir)

    def validate_response(self, response: Any) -> bool:
        """Validate climate data response."""
        if response is None:
            return False
        if isinstance(response, pd.DataFrame) and response.empty:
            return False
        return True

    def fetch_single(
        self,
        ticker: str,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch climate data series.

        Args:
            ticker: Data series name ('nasa_giss', 'mauna_loa_co2', 'sea_level')

        Returns:
            DataFrame with date and values
        """
        ticker_lower = ticker.lower()

        if ticker_lower == "nasa_giss":
            return self._fetch_nasa_giss()
        elif ticker_lower == "mauna_loa_co2":
            return self._fetch_mauna_loa()
        elif ticker_lower == "sea_level":
            return self._fetch_sea_level()
        else:
            logger.error(f"Unknown climate series: {ticker}")
            return None

    def _fetch_nasa_giss(self) -> Optional[pd.DataFrame]:
        """Fetch NASA GISS global temperature anomaly data."""
        try:
            url = self.SOURCES["nasa_giss"]

            # Read CSV, skipping header rows
            df = pd.read_csv(url, skiprows=1)

            # The data has Year as first column, then monthly values
            # Melt to long format
            df = df.melt(
                id_vars=["Year"],
                var_name="month",
                value_name="temp_anomaly"
            )

            # Convert month names to numbers
            month_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
            }
            df["month_num"] = df["month"].map(month_map)
            df = df.dropna(subset=["month_num"])

            # Create date column
            df["date"] = pd.to_datetime(
                df["Year"].astype(int).astype(str) + "-" +
                df["month_num"].astype(int).astype(str) + "-01"
            )

            # Clean up
            df = df[["date", "temp_anomaly"]].dropna()
            df = df.sort_values("date").reset_index(drop=True)

            # Convert anomaly to float (handle '***' missing values)
            df["temp_anomaly"] = pd.to_numeric(df["temp_anomaly"], errors="coerce")

            return df

        except Exception as e:
            logger.error(f"Error fetching NASA GISS data: {e}")
            return None

    def _fetch_mauna_loa(self) -> Optional[pd.DataFrame]:
        """Fetch Mauna Loa CO2 measurements."""
        try:
            url = self.SOURCES["mauna_loa_co2"]

            # Read CSV, skipping comment lines
            df = pd.read_csv(url, comment="#", header=None)

            # Columns: year, month, decimal_date, monthly_avg, deseasonalized, ...
            df.columns = ["year", "month", "decimal_date", "co2_ppm",
                         "co2_deseasonalized", "days", "std", "uncertainty"]

            # Create date
            df["date"] = pd.to_datetime(
                df["year"].astype(str) + "-" +
                df["month"].astype(str).str.zfill(2) + "-01"
            )

            # Handle missing values (marked as -99.99)
            df.loc[df["co2_ppm"] < 0, "co2_ppm"] = None

            return df[["date", "co2_ppm", "co2_deseasonalized"]].dropna()

        except Exception as e:
            logger.error(f"Error fetching Mauna Loa CO2 data: {e}")
            return None

    def _fetch_sea_level(self) -> Optional[pd.DataFrame]:
        """Fetch global sea level rise data."""
        try:
            url = self.SOURCES["sea_level"]

            # Try to fetch and parse
            response = requests.get(url)
            response.raise_for_status()

            # Parse the data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), skiprows=5)

            # Typical columns: year, TOPEX, Jason-1, etc.
            # Simplify to year and sea level
            if "year" in df.columns.str.lower():
                year_col = [c for c in df.columns if "year" in c.lower()][0]
                value_cols = [c for c in df.columns if c != year_col]

                df["date"] = pd.to_datetime(df[year_col], format="%Y")
                df["sea_level_mm"] = df[value_cols].mean(axis=1)

                return df[["date", "sea_level_mm"]].dropna()

            return None

        except Exception as e:
            logger.error(f"Error fetching sea level data: {e}")
            return None


# Available climate data series
AVAILABLE_CLIMATE_SERIES = {
    "nasa_giss": "Global temperature anomaly (NASA GISS)",
    "mauna_loa_co2": "Atmospheric CO2 concentration (Mauna Loa)",
    "sea_level": "Global sea level rise",
}
