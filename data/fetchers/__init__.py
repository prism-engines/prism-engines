"""
Multi-Domain Data Fetchers
==========================

Fetch data from diverse sources for cross-domain coherence analysis.

Modules:
- base_fetcher: Base class with common functionality
- noaa_fetcher: NOAA climate data (temperature, precipitation)
- nasa_fetcher: NASA GISS temperature anomalies
- cdc_fetcher: CDC epidemiology data (flu, mortality)
- usgs_fetcher: USGS earthquake data
- google_trends_fetcher: Google Trends interest data
- owid_fetcher: Our World in Data (COVID, economics)

Purpose:
Test cross-domain coherence hypotheses across:
- Climate systems
- Epidemiological dynamics
- Geophysical events
- Social/behavioral patterns
"""

from .base_fetcher import (
    BaseFetcher,
    FetchResult,
    DataQualityReport
)

from .noaa_fetcher import NOAAFetcher
from .nasa_fetcher import NASAFetcher
from .cdc_fetcher import CDCFetcher
from .usgs_fetcher import USGSFetcher
from .google_trends_fetcher import GoogleTrendsFetcher
from .owid_fetcher import OWIDFetcher

__all__ = [
    'BaseFetcher',
    'FetchResult',
    'DataQualityReport',
    'NOAAFetcher',
    'NASAFetcher',
    'CDCFetcher',
    'USGSFetcher',
    'GoogleTrendsFetcher',
    'OWIDFetcher'
]
