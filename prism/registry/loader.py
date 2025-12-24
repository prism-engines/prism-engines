"""
PRISM Registry Loader

Loads indicator definitions from YAML configuration.

The registry defines WHAT indicators exist and WHERE they come from.
It does NOT fetch data or write to databases.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml
import logging


logger = logging.getLogger(__name__)


# Default registry location
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "indicators.yaml"


@dataclass
class Indicator:
    """Single indicator definition."""
    id: str
    source: str
    name: Optional[str] = None
    frequency: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    units: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for FetchRunner."""
        return {
            "id": self.id,
            "source": self.source,
            "frequency": self.frequency,
        }


class RegistryLoader:
    """
    Loads and queries indicator definitions from YAML.
    
    Usage:
        registry = RegistryLoader()
        
        # Get all indicators
        all_indicators = registry.get_all()
        
        # Get by source
        fred_indicators = registry.get_by_source("fred")
        
        # Get by category
        macro_indicators = registry.get_by_category("macro")
        
        # Get specific indicators
        selected = registry.get_indicators(["GDP", "UNRATE", "SPY"])
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize registry loader.
        
        Args:
            registry_path: Path to indicators.yaml. Uses default if not provided.
        """
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self._indicators: Dict[str, Indicator] = {}
        self._loaded = False
    
    def _load(self) -> None:
        """Load registry from YAML file."""
        if self._loaded:
            return
            
        if not self.registry_path.exists():
            logger.warning(f"Registry file not found: {self.registry_path}")
            self._loaded = True
            return
        
        with open(self.registry_path) as f:
            data = yaml.safe_load(f)
        
        if not data or "indicators" not in data:
            logger.warning("No indicators found in registry")
            self._loaded = True
            return
        
        for ind_data in data["indicators"]:
            indicator = Indicator(
                id=ind_data["id"],
                source=ind_data["source"],
                name=ind_data.get("name"),
                frequency=ind_data.get("frequency"),
                category=ind_data.get("category"),
                description=ind_data.get("description"),
                units=ind_data.get("units"),
            )
            self._indicators[indicator.id] = indicator
        
        logger.info(f"Loaded {len(self._indicators)} indicators from registry")
        self._loaded = True
    
    def get_all(self) -> List[Indicator]:
        """Get all registered indicators."""
        self._load()
        return list(self._indicators.values())
    
    def get_indicator(self, indicator_id: str) -> Optional[Indicator]:
        """Get a single indicator by ID."""
        self._load()
        return self._indicators.get(indicator_id)
    
    def get_indicators(self, indicator_ids: List[str]) -> List[Indicator]:
        """Get multiple indicators by ID."""
        self._load()
        result = []
        for ind_id in indicator_ids:
            if ind_id in self._indicators:
                result.append(self._indicators[ind_id])
            else:
                logger.warning(f"Indicator not found in registry: {ind_id}")
        return result
    
    def get_by_source(self, source: str) -> List[Indicator]:
        """Get all indicators from a specific source."""
        self._load()
        return [i for i in self._indicators.values() if i.source == source]
    
    def get_by_category(self, category: str) -> List[Indicator]:
        """Get all indicators in a specific category."""
        self._load()
        return [i for i in self._indicators.values() if i.category == category]
    
    def get_sources(self) -> Set[str]:
        """Get all unique sources."""
        self._load()
        return {i.source for i in self._indicators.values()}
    
    def get_categories(self) -> Set[str]:
        """Get all unique categories."""
        self._load()
        return {i.category for i in self._indicators.values() if i.category}
    
    def to_fetch_list(self, indicators: Optional[List[Indicator]] = None) -> List[Dict]:
        """
        Convert indicators to format expected by FetchRunner.
        
        Args:
            indicators: List of Indicator objects. If None, uses all indicators.
            
        Returns:
            List of dicts with 'id', 'source', 'frequency' keys
        """
        if indicators is None:
            indicators = self.get_all()
        return [i.to_dict() for i in indicators]
