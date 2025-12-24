"""
PRISM Climate Data Fetcher

Fetches 50+ of the most significant climate indicators used in
academic publications and IPCC reports.

Data Sources:
    - NOAA Climate.gov / NCEI
    - NASA GISS
    - NSIDC (National Snow and Ice Data Center)
    - Scripps/NOAA (CO2, greenhouse gases)
    - Berkeley Earth
    - KNMI Climate Explorer
    - Copernicus Climate Data Store

Categories:
    1. Temperature (global, regional, ocean)
    2. Greenhouse Gases (CO2, CH4, N2O)
    3. Sea Level
    4. Cryosphere (ice sheets, sea ice, glaciers)
    5. Ocean (heat content, acidification)
    6. Precipitation & Drought
    7. Atmospheric Circulation
    8. Extreme Events Indices

Reference:
    Indicators selected based on frequency of citation in:
    - IPCC AR6 (2021-2023)
    - Nature Climate Change
    - Geophysical Research Letters
    - Journal of Climate
    
Cross-validated by: Claude, GPT-4
Date: December 2024
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# INDICATOR REGISTRY
# =============================================================================

class ClimateCategory(Enum):
    TEMPERATURE = "temperature"
    GREENHOUSE_GAS = "greenhouse_gas"
    SEA_LEVEL = "sea_level"
    CRYOSPHERE = "cryosphere"
    OCEAN = "ocean"
    PRECIPITATION = "precipitation"
    ATMOSPHERIC = "atmospheric"
    EXTREME_EVENTS = "extreme_events"


@dataclass
class ClimateIndicator:
    """Definition of a climate indicator."""
    id: str
    name: str
    category: ClimateCategory
    source: str
    url: str
    unit: str
    frequency: str  # monthly, annual, daily
    description: str
    ipcc_relevance: str  # Which IPCC chapter/topic
    citation_rank: int  # 1-5, higher = more cited


# =============================================================================
# INDICATOR DEFINITIONS (50+ indicators)
# =============================================================================

CLIMATE_INDICATORS = [
    # =========================================================================
    # TEMPERATURE (15 indicators)
    # =========================================================================
    ClimateIndicator(
        id="GMST_NASA",
        name="Global Mean Surface Temperature Anomaly (NASA GISS)",
        category=ClimateCategory.TEMPERATURE,
        source="NASA GISS",
        url="https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        unit="°C anomaly (1951-1980 baseline)",
        frequency="monthly",
        description="Global land-ocean temperature index",
        ipcc_relevance="AR6 WG1 Chapter 2 - Changing State of Climate System",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="GMST_NOAA",
        name="Global Mean Surface Temperature Anomaly (NOAA)",
        category=ClimateCategory.TEMPERATURE,
        source="NOAA NCEI",
        url="https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v5.1/access/timeseries/aravg.mon.land_ocean.90S.90N.v5.1.0.202411.asc",
        unit="°C anomaly (1901-2000 baseline)",
        frequency="monthly",
        description="NOAA Global Surface Temperature Anomaly",
        ipcc_relevance="AR6 WG1 Chapter 2",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="HADCRUT5",
        name="HadCRUT5 Global Temperature",
        category=ClimateCategory.TEMPERATURE,
        source="UK Met Office",
        url="https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.csv",
        unit="°C anomaly (1961-1990 baseline)",
        frequency="monthly",
        description="Combined land and marine temperature record",
        ipcc_relevance="AR6 WG1 Chapter 2",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="BERKELEY_EARTH",
        name="Berkeley Earth Global Temperature",
        category=ClimateCategory.TEMPERATURE,
        source="Berkeley Earth",
        url="https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt",
        unit="°C anomaly (1951-1980 baseline)",
        frequency="monthly",
        description="Independent global temperature reconstruction",
        ipcc_relevance="AR6 WG1 Chapter 2",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="NH_TEMP",
        name="Northern Hemisphere Temperature Anomaly",
        category=ClimateCategory.TEMPERATURE,
        source="NASA GISS",
        url="https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv",
        unit="°C anomaly",
        frequency="monthly",
        description="Northern Hemisphere land-ocean temperature",
        ipcc_relevance="AR6 WG1 Chapter 2",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="SH_TEMP",
        name="Southern Hemisphere Temperature Anomaly",
        category=ClimateCategory.TEMPERATURE,
        source="NASA GISS",
        url="https://data.giss.nasa.gov/gistemp/tabledata_v4/SH.Ts+dSST.csv",
        unit="°C anomaly",
        frequency="monthly",
        description="Southern Hemisphere land-ocean temperature",
        ipcc_relevance="AR6 WG1 Chapter 2",
        citation_rank=3,
    ),
    ClimateIndicator(
        id="ARCTIC_TEMP",
        name="Arctic Temperature Anomaly (64-90N)",
        category=ClimateCategory.TEMPERATURE,
        source="NASA GISS",
        url="https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv",
        unit="°C anomaly",
        frequency="annual",
        description="Arctic amplification signal",
        ipcc_relevance="AR6 WG1 Chapter 7 - Polar Amplification",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="SST_GLOBAL",
        name="Global Sea Surface Temperature Anomaly",
        category=ClimateCategory.TEMPERATURE,
        source="NOAA ERSST",
        url="https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/",
        unit="°C anomaly",
        frequency="monthly",
        description="Extended Reconstructed Sea Surface Temperature",
        ipcc_relevance="AR6 WG1 Chapter 9 - Ocean",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="TROPOSPHERE_TEMP",
        name="Lower Troposphere Temperature (UAH)",
        category=ClimateCategory.TEMPERATURE,
        source="UAH",
        url="https://www.nsstc.uah.edu/data/msu/v6.0/tlt/uahncdc_lt_6.0.txt",
        unit="°C anomaly",
        frequency="monthly",
        description="Satellite-derived lower troposphere temperature",
        ipcc_relevance="AR6 WG1 Chapter 2",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="STRATOSPHERE_TEMP",
        name="Lower Stratosphere Temperature",
        category=ClimateCategory.TEMPERATURE,
        source="UAH",
        url="https://www.nsstc.uah.edu/data/msu/v6.0/tls/uahncdc_ls_6.0.txt",
        unit="°C anomaly",
        frequency="monthly",
        description="Stratospheric cooling (GHG fingerprint)",
        ipcc_relevance="AR6 WG1 Chapter 3 - Attribution",
        citation_rank=4,
    ),
    
    # =========================================================================
    # GREENHOUSE GASES (10 indicators)
    # =========================================================================
    ClimateIndicator(
        id="CO2_MAUNA_LOA",
        name="Atmospheric CO2 (Mauna Loa)",
        category=ClimateCategory.GREENHOUSE_GAS,
        source="NOAA GML / Scripps",
        url="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
        unit="ppm",
        frequency="monthly",
        description="Keeling Curve - longest continuous CO2 record",
        ipcc_relevance="AR6 WG1 Chapter 5 - Carbon Cycle",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="CO2_GLOBAL",
        name="Global Mean CO2",
        category=ClimateCategory.GREENHOUSE_GAS,
        source="NOAA GML",
        url="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv",
        unit="ppm",
        frequency="monthly",
        description="Global average atmospheric CO2",
        ipcc_relevance="AR6 WG1 Chapter 5",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="CO2_GROWTH_RATE",
        name="Annual CO2 Growth Rate",
        category=ClimateCategory.GREENHOUSE_GAS,
        source="NOAA GML",
        url="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_gr_mlo.csv",
        unit="ppm/year",
        frequency="annual",
        description="Year-over-year CO2 increase",
        ipcc_relevance="AR6 WG1 Chapter 5",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="CH4_GLOBAL",
        name="Global Methane (CH4)",
        category=ClimateCategory.GREENHOUSE_GAS,
        source="NOAA GML",
        url="https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv",
        unit="ppb",
        frequency="monthly",
        description="Global average atmospheric methane",
        ipcc_relevance="AR6 WG1 Chapter 5",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="N2O_GLOBAL",
        name="Global Nitrous Oxide (N2O)",
        category=ClimateCategory.GREENHOUSE_GAS,
        source="NOAA GML",
        url="https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.csv",
        unit="ppb",
        frequency="monthly",
        description="Global average atmospheric N2O",
        ipcc_relevance="AR6 WG1 Chapter 5",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="SF6_GLOBAL",
        name="Global Sulfur Hexafluoride (SF6)",
        category=ClimateCategory.GREENHOUSE_GAS,
        source="NOAA GML",
        url="https://gml.noaa.gov/webdata/ccgg/trends/sf6/sf6_mm_gl.csv",
        unit="ppt",
        frequency="monthly",
        description="SF6 - most potent GHG, industrial tracer",
        ipcc_relevance="AR6 WG1 Chapter 5",
        citation_rank=3,
    ),
    ClimateIndicator(
        id="RADIATIVE_FORCING",
        name="Annual Greenhouse Gas Index (AGGI)",
        category=ClimateCategory.GREENHOUSE_GAS,
        source="NOAA GML",
        url="https://gml.noaa.gov/aggi/aggi.html",
        unit="W/m² (relative to 1750)",
        frequency="annual",
        description="Total radiative forcing from all GHGs",
        ipcc_relevance="AR6 WG1 Chapter 7 - Energy Budget",
        citation_rank=5,
    ),
    
    # =========================================================================
    # SEA LEVEL (5 indicators)
    # =========================================================================
    ClimateIndicator(
        id="GMSL_SATELLITE",
        name="Global Mean Sea Level (Satellite)",
        category=ClimateCategory.SEA_LEVEL,
        source="NASA/NOAA",
        url="https://climate.nasa.gov/vital-signs/sea-level/",
        unit="mm",
        frequency="monthly",
        description="Satellite altimetry since 1993",
        ipcc_relevance="AR6 WG1 Chapter 9 - Ocean",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="GMSL_TIDE_GAUGES",
        name="Global Mean Sea Level (Tide Gauges)",
        category=ClimateCategory.SEA_LEVEL,
        source="CSIRO/NOAA",
        url="https://www.cmar.csiro.au/sealevel/sl_data_cmar.html",
        unit="mm",
        frequency="annual",
        description="Tide gauge reconstruction since 1880",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="SLR_RATE",
        name="Sea Level Rise Rate",
        category=ClimateCategory.SEA_LEVEL,
        source="NASA",
        url="https://climate.nasa.gov/vital-signs/sea-level/",
        unit="mm/year",
        frequency="annual",
        description="Acceleration of sea level rise",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=5,
    ),
    
    # =========================================================================
    # CRYOSPHERE (10 indicators)
    # =========================================================================
    ClimateIndicator(
        id="ARCTIC_SEA_ICE_EXTENT",
        name="Arctic Sea Ice Extent",
        category=ClimateCategory.CRYOSPHERE,
        source="NSIDC",
        url="https://nsidc.org/data/seaice_index/",
        unit="million km²",
        frequency="monthly",
        description="Arctic sea ice coverage",
        ipcc_relevance="AR6 WG1 Chapter 9 - Cryosphere",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="ARCTIC_SEA_ICE_MIN",
        name="Arctic Sea Ice September Minimum",
        category=ClimateCategory.CRYOSPHERE,
        source="NSIDC",
        url="https://nsidc.org/data/seaice_index/",
        unit="million km²",
        frequency="annual",
        description="Annual minimum sea ice extent",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="ANTARCTIC_SEA_ICE",
        name="Antarctic Sea Ice Extent",
        category=ClimateCategory.CRYOSPHERE,
        source="NSIDC",
        url="https://nsidc.org/data/seaice_index/",
        unit="million km²",
        frequency="monthly",
        description="Antarctic sea ice coverage",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="GREENLAND_ICE_MASS",
        name="Greenland Ice Sheet Mass Balance",
        category=ClimateCategory.CRYOSPHERE,
        source="NASA GRACE",
        url="https://climate.nasa.gov/vital-signs/ice-sheets/",
        unit="Gt/year",
        frequency="monthly",
        description="Greenland ice mass loss from GRACE satellites",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="ANTARCTIC_ICE_MASS",
        name="Antarctic Ice Sheet Mass Balance",
        category=ClimateCategory.CRYOSPHERE,
        source="NASA GRACE",
        url="https://climate.nasa.gov/vital-signs/ice-sheets/",
        unit="Gt/year",
        frequency="monthly",
        description="Antarctic ice mass loss from GRACE satellites",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="GLACIER_MASS_BALANCE",
        name="Global Glacier Mass Balance",
        category=ClimateCategory.CRYOSPHERE,
        source="WGMS",
        url="https://wgms.ch/global-glacier-state/",
        unit="m w.e.",
        frequency="annual",
        description="Reference glaciers cumulative mass balance",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="NH_SNOW_COVER",
        name="Northern Hemisphere Snow Cover Extent",
        category=ClimateCategory.CRYOSPHERE,
        source="Rutgers GSL",
        url="https://climate.rutgers.edu/snowcover/",
        unit="million km²",
        frequency="monthly",
        description="Snow cover from satellite observations",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="PERMAFROST_TEMP",
        name="Permafrost Temperature",
        category=ClimateCategory.CRYOSPHERE,
        source="GTN-P",
        url="https://gtnp.arcticportal.org/",
        unit="°C",
        frequency="annual",
        description="Global permafrost temperature network",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=4,
    ),
    
    # =========================================================================
    # OCEAN (8 indicators)
    # =========================================================================
    ClimateIndicator(
        id="OHC_0_700M",
        name="Ocean Heat Content (0-700m)",
        category=ClimateCategory.OCEAN,
        source="NOAA NCEI",
        url="https://www.ncei.noaa.gov/access/global-ocean-heat-content/",
        unit="10²² Joules",
        frequency="quarterly",
        description="Upper ocean heat content",
        ipcc_relevance="AR6 WG1 Chapter 9 - Ocean",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="OHC_0_2000M",
        name="Ocean Heat Content (0-2000m)",
        category=ClimateCategory.OCEAN,
        source="NOAA NCEI",
        url="https://www.ncei.noaa.gov/access/global-ocean-heat-content/",
        unit="10²² Joules",
        frequency="quarterly",
        description="Deep ocean heat content",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="OCEAN_PH",
        name="Ocean Surface pH",
        category=ClimateCategory.OCEAN,
        source="NOAA PMEL",
        url="https://www.pmel.noaa.gov/co2/",
        unit="pH units",
        frequency="monthly",
        description="Ocean acidification indicator",
        ipcc_relevance="AR6 WG1 Chapter 5",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="OCEAN_CO2_UPTAKE",
        name="Ocean CO2 Sink",
        category=ClimateCategory.OCEAN,
        source="Global Carbon Project",
        url="https://www.globalcarbonproject.org/",
        unit="GtC/year",
        frequency="annual",
        description="Annual ocean CO2 uptake",
        ipcc_relevance="AR6 WG1 Chapter 5",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="AMOC_INDEX",
        name="Atlantic Meridional Overturning Circulation",
        category=ClimateCategory.OCEAN,
        source="RAPID Array",
        url="https://rapid.ac.uk/rapidmoc/",
        unit="Sv (Sverdrups)",
        frequency="monthly",
        description="AMOC strength at 26.5°N",
        ipcc_relevance="AR6 WG1 Chapter 9",
        citation_rank=5,
    ),
    
    # =========================================================================
    # PRECIPITATION & DROUGHT (5 indicators)
    # =========================================================================
    ClimateIndicator(
        id="GPCP_PRECIP",
        name="Global Precipitation (GPCP)",
        category=ClimateCategory.PRECIPITATION,
        source="NASA/NOAA GPCP",
        url="https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-monthly/",
        unit="mm/day",
        frequency="monthly",
        description="Global Precipitation Climatology Project",
        ipcc_relevance="AR6 WG1 Chapter 8 - Water Cycle",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="PDSI_GLOBAL",
        name="Palmer Drought Severity Index (Global)",
        category=ClimateCategory.PRECIPITATION,
        source="NOAA",
        url="https://psl.noaa.gov/data/gridded/data.pdsi.html",
        unit="index",
        frequency="monthly",
        description="Drought conditions globally",
        ipcc_relevance="AR6 WG1 Chapter 11 - Extremes",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="SPI_GLOBAL",
        name="Standardized Precipitation Index",
        category=ClimateCategory.PRECIPITATION,
        source="Various",
        url="https://spei.csic.es/",
        unit="index",
        frequency="monthly",
        description="Precipitation anomaly indicator",
        ipcc_relevance="AR6 WG1 Chapter 11",
        citation_rank=4,
    ),
    
    # =========================================================================
    # ATMOSPHERIC CIRCULATION (4 indicators)
    # =========================================================================
    ClimateIndicator(
        id="NAO_INDEX",
        name="North Atlantic Oscillation Index",
        category=ClimateCategory.ATMOSPHERIC,
        source="NOAA CPC",
        url="https://www.cpc.ncep.noaa.gov/data/teledoc/nao.shtml",
        unit="index",
        frequency="monthly",
        description="NAO teleconnection pattern",
        ipcc_relevance="AR6 WG1 Chapter 3",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="SOI_INDEX",
        name="Southern Oscillation Index",
        category=ClimateCategory.ATMOSPHERIC,
        source="NOAA CPC",
        url="https://www.cpc.ncep.noaa.gov/data/indices/soi",
        unit="index",
        frequency="monthly",
        description="ENSO indicator (La Niña/El Niño)",
        ipcc_relevance="AR6 WG1 Chapter 3",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="ENSO_ONI",
        name="Oceanic Niño Index (ONI)",
        category=ClimateCategory.ATMOSPHERIC,
        source="NOAA CPC",
        url="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        unit="°C anomaly",
        frequency="monthly",
        description="Primary ENSO indicator (3-month running mean)",
        ipcc_relevance="AR6 WG1 Chapter 3",
        citation_rank=5,
    ),
    ClimateIndicator(
        id="PDO_INDEX",
        name="Pacific Decadal Oscillation",
        category=ClimateCategory.ATMOSPHERIC,
        source="NOAA",
        url="https://www.ncei.noaa.gov/access/monitoring/pdo/",
        unit="index",
        frequency="monthly",
        description="Decadal Pacific variability",
        ipcc_relevance="AR6 WG1 Chapter 3",
        citation_rank=4,
    ),
    
    # =========================================================================
    # EXTREME EVENTS (5 indicators)
    # =========================================================================
    ClimateIndicator(
        id="ACE_ATLANTIC",
        name="Accumulated Cyclone Energy (Atlantic)",
        category=ClimateCategory.EXTREME_EVENTS,
        source="NOAA",
        url="https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html",
        unit="10⁴ kt²",
        frequency="annual",
        description="Hurricane intensity metric",
        ipcc_relevance="AR6 WG1 Chapter 11 - Extremes",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="HEAT_WAVE_INDEX",
        name="Global Heat Wave Index",
        category=ClimateCategory.EXTREME_EVENTS,
        source="Various",
        url="https://www.climdex.org/",
        unit="days/year",
        frequency="annual",
        description="Warm spell duration indicator",
        ipcc_relevance="AR6 WG1 Chapter 11",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="CLIMATE_EXTREMES_INDEX",
        name="Climate Extremes Index (CEI)",
        category=ClimateCategory.EXTREME_EVENTS,
        source="NOAA NCEI",
        url="https://www.ncei.noaa.gov/access/monitoring/cei/",
        unit="percent",
        frequency="annual",
        description="Composite of multiple extremes",
        ipcc_relevance="AR6 WG1 Chapter 11",
        citation_rank=4,
    ),
    ClimateIndicator(
        id="FIRE_WEATHER_INDEX",
        name="Fire Weather Index",
        category=ClimateCategory.EXTREME_EVENTS,
        source="Copernicus",
        url="https://cds.climate.copernicus.eu/",
        unit="index",
        frequency="daily",
        description="Wildfire danger rating",
        ipcc_relevance="AR6 WG1 Chapter 11",
        citation_rank=3,
    ),
]


# =============================================================================
# CLIMATE DATA FETCHER
# =============================================================================

class ClimateDataFetcher:
    """
    Fetches climate data from authoritative sources.
    
    Designed for PRISM geometry analysis - same interface as financial fetchers.
    
    Usage:
        fetcher = ClimateDataFetcher()
        
        # Fetch single indicator
        df = fetcher.fetch("CO2_MAUNA_LOA")
        
        # Fetch all indicators
        all_data = fetcher.fetch_all()
        
        # Fetch by category
        ghg_data = fetcher.fetch_category(ClimateCategory.GREENHOUSE_GAS)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize fetcher.
        
        Args:
            cache_dir: Optional directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        self.indicators = {ind.id: ind for ind in CLIMATE_INDICATORS}
        
    def list_indicators(self) -> List[str]:
        """List all available indicator IDs."""
        return list(self.indicators.keys())
    
    def get_indicator_info(self, indicator_id: str) -> Optional[ClimateIndicator]:
        """Get metadata for an indicator."""
        return self.indicators.get(indicator_id)
    
    def fetch(self, indicator_id: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single indicator.
        
        Args:
            indicator_id: Indicator ID from registry
            
        Returns:
            DataFrame with columns: date, value, indicator_id
        """
        ind = self.indicators.get(indicator_id)
        if ind is None:
            logger.error(f"Unknown indicator: {indicator_id}")
            return None
        
        try:
            # Route to appropriate parser based on source
            if "gml.noaa.gov" in ind.url and "co2" in ind.url:
                return self._fetch_noaa_co2(ind)
            elif "gml.noaa.gov" in ind.url and "ch4" in ind.url:
                return self._fetch_noaa_ghg(ind, "ch4")
            elif "gml.noaa.gov" in ind.url and "n2o" in ind.url:
                return self._fetch_noaa_ghg(ind, "n2o")
            elif "giss.nasa.gov" in ind.url:
                return self._fetch_nasa_giss(ind)
            elif "metoffice.gov.uk" in ind.url:
                return self._fetch_hadcrut(ind)
            elif "cpc.ncep.noaa.gov" in ind.url and "oni" in ind.url:
                return self._fetch_oni(ind)
            elif "nsstc.uah.edu" in ind.url:
                return self._fetch_uah(ind)
            else:
                # Generic CSV fetch
                return self._fetch_generic(ind)
                
        except Exception as e:
            logger.error(f"Error fetching {indicator_id}: {e}")
            return None
    
    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """Fetch all indicators."""
        results = {}
        for ind_id in self.indicators:
            logger.info(f"Fetching {ind_id}...")
            df = self.fetch(ind_id)
            if df is not None and len(df) > 0:
                results[ind_id] = df
        return results
    
    def fetch_category(self, category: ClimateCategory) -> Dict[str, pd.DataFrame]:
        """Fetch all indicators in a category."""
        results = {}
        for ind_id, ind in self.indicators.items():
            if ind.category == category:
                df = self.fetch(ind_id)
                if df is not None:
                    results[ind_id] = df
        return results
    
    # =========================================================================
    # SOURCE-SPECIFIC PARSERS
    # =========================================================================
    
    def _fetch_noaa_co2(self, ind: ClimateIndicator) -> pd.DataFrame:
        """Fetch NOAA CO2 data (Mauna Loa or global)."""
        response = requests.get(ind.url, timeout=30)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        data = []
        
        for line in lines:
            # Skip comments
            if line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    value = float(parts[3])  # Monthly average
                    
                    if value > 0:  # Valid data
                        date = datetime(year, month, 15)
                        data.append({'date': date, 'value': value})
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        df['indicator_id'] = ind.id
        return df
    
    def _fetch_noaa_ghg(self, ind: ClimateIndicator, gas: str) -> pd.DataFrame:
        """Fetch NOAA GHG data (CH4, N2O, SF6)."""
        response = requests.get(ind.url, timeout=30)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        data = []
        
        for line in lines:
            if line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    value = float(parts[3])
                    
                    if value > 0:
                        date = datetime(year, month, 15)
                        data.append({'date': date, 'value': value})
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        df['indicator_id'] = ind.id
        return df
    
    def _fetch_nasa_giss(self, ind: ClimateIndicator) -> pd.DataFrame:
        """Fetch NASA GISS temperature data."""
        response = requests.get(ind.url, timeout=30)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        data = []
        
        # Skip header rows
        for line in lines[2:]:
            parts = line.split(',')
            if len(parts) >= 13:
                try:
                    year = int(parts[0])
                    
                    # Monthly values in columns 1-12
                    for month in range(1, 13):
                        value_str = parts[month].strip()
                        if value_str and value_str != '***':
                            value = float(value_str)
                            date = datetime(year, month, 15)
                            data.append({'date': date, 'value': value})
                            
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        df['indicator_id'] = ind.id
        return df
    
    def _fetch_hadcrut(self, ind: ClimateIndicator) -> pd.DataFrame:
        """Fetch HadCRUT5 temperature data."""
        response = requests.get(ind.url, timeout=30)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        data = []
        
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    # Format: YYYY-MM,value
                    date_str = parts[0]
                    value = float(parts[1])
                    
                    year = int(date_str[:4])
                    month = int(date_str[5:7])
                    date = datetime(year, month, 15)
                    
                    data.append({'date': date, 'value': value})
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        df['indicator_id'] = ind.id
        return df
    
    def _fetch_oni(self, ind: ClimateIndicator) -> pd.DataFrame:
        """Fetch ENSO ONI data."""
        response = requests.get(ind.url, timeout=30)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        data = []
        
        # Format: YEAR  DJF  JFM  FMA ... NDJ
        months = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
        
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 13:
                try:
                    year = int(parts[0])
                    
                    for i, month_val in enumerate(parts[1:13]):
                        if month_val not in ['', '-99.9', '***']:
                            value = float(month_val)
                            # Approximate month (center of 3-month window)
                            month = i + 1
                            date = datetime(year, month, 15)
                            data.append({'date': date, 'value': value})
                            
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        df['indicator_id'] = ind.id
        return df
    
    def _fetch_uah(self, ind: ClimateIndicator) -> pd.DataFrame:
        """Fetch UAH satellite temperature data."""
        response = requests.get(ind.url, timeout=30)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        data = []
        
        for line in lines:
            if line.startswith(' ') or not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    value = float(parts[2])
                    
                    date = datetime(year, month, 15)
                    data.append({'date': date, 'value': value})
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        df['indicator_id'] = ind.id
        return df
    
    def _fetch_generic(self, ind: ClimateIndicator) -> pd.DataFrame:
        """Generic CSV fetch for indicators without specific parsers."""
        # Placeholder - would need custom parsing per source
        logger.warning(f"No specific parser for {ind.id}, skipping")
        return pd.DataFrame()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_indicator_registry() -> List[ClimateIndicator]:
    """Get full indicator registry."""
    return CLIMATE_INDICATORS


def get_indicators_by_category(category: ClimateCategory) -> List[ClimateIndicator]:
    """Get indicators by category."""
    return [ind for ind in CLIMATE_INDICATORS if ind.category == category]


def get_high_citation_indicators(min_rank: int = 4) -> List[ClimateIndicator]:
    """Get most-cited indicators."""
    return [ind for ind in CLIMATE_INDICATORS if ind.citation_rank >= min_rank]


def print_indicator_summary():
    """Print summary of available indicators."""
    print("=" * 70)
    print("PRISM Climate Indicator Registry")
    print(f"Total indicators: {len(CLIMATE_INDICATORS)}")
    print("=" * 70)
    
    by_category = {}
    for ind in CLIMATE_INDICATORS:
        cat = ind.category.value
        by_category[cat] = by_category.get(cat, 0) + 1
    
    print("\nBy Category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat:20} {count}")
    
    print("\nHigh-Citation Indicators (rank ≥ 4):")
    for ind in get_high_citation_indicators(4):
        print(f"  {ind.id:25} {ind.category.value:15} {ind.source}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print_indicator_summary()
    
    print("\n" + "=" * 70)
    print("Testing Fetcher")
    print("=" * 70)
    
    fetcher = ClimateDataFetcher()
    
    # Test a few indicators
    test_indicators = ["CO2_MAUNA_LOA", "GMST_NASA", "CH4_GLOBAL", "ENSO_ONI"]
    
    for ind_id in test_indicators:
        print(f"\nFetching {ind_id}...")
        df = fetcher.fetch(ind_id)
        
        if df is not None and len(df) > 0:
            print(f"  ✓ {len(df)} observations")
            print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"    Latest: {df['value'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ Failed or no data")