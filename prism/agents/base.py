"""
PRISM Agent Base Classes

Base classes for domain agents and data acquisition.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class IndicatorMeta:
    """Metadata for an indicator."""
    indicator_id: str
    name: Optional[str] = None
    source: Optional[str] = None
    frequency: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    units: Optional[str] = None


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    indicator_id: str
    success: bool
    rows_fetched: int = 0
    first_date: Optional[datetime] = None
    last_date: Optional[datetime] = None
    error: Optional[str] = None


class DomainAgent:
    """
    Base class for domain-specific data acquisition agents.

    Subclass this for ClimateAgent, EpidemiologyAgent, etc.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.indicators: List[IndicatorMeta] = []

    def fetch(self, indicator_id: str) -> FetchResult:
        """Fetch data for an indicator. Override in subclass."""
        raise NotImplementedError("Subclass must implement fetch()")

    def list_indicators(self) -> List[str]:
        """Return list of available indicator IDs."""
        return [ind.indicator_id for ind in self.indicators]
