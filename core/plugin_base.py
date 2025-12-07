"""
PRISM Plugin Base Classes
==========================

Base classes for creating PRISM plugins.

Usage:
    from core import EnginePlugin

    class MyEngine(EnginePlugin):
        name = "my_engine"
        version = "1.0.0"

        def analyze(self, data):
            ...

        def rank_indicators(self, data):
            ...
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    settings_schema: Dict[str, Any] = field(default_factory=dict)


class PluginBase(ABC):
    """Base class for all PRISM plugins."""

    # Override in subclass
    name: str = "unnamed_plugin"
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    category: str = "general"
    tags: List[str] = []
    dependencies: List[str] = []

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin with optional settings.

        Args:
            settings: Plugin-specific settings
        """
        self.settings = settings or {}
        self._validate_settings()
        self.logger = logging.getLogger(f"plugin.{self.name}")

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            category=self.category,
            tags=self.tags,
            dependencies=self.dependencies,
            settings_schema=self.get_settings_schema(),
        )

    def get_settings_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema for plugin settings.
        Override in subclass if settings are needed.
        """
        return {}

    def _validate_settings(self) -> None:
        """Validate settings against schema."""
        schema = self.get_settings_schema()
        if not schema:
            return

        # Basic validation - check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in self.settings:
                raise ValueError(f"Missing required setting: {field}")

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate plugin is properly configured.
        Returns True if valid, raises exception otherwise.
        """
        pass


class EnginePlugin(PluginBase):
    """Base class for analysis engine plugins."""

    category: str = "engine"

    # Engine-specific metadata
    engine_type: str = "lens"  # lens, aggregator, utility
    supports_ranking: bool = True
    supports_analysis: bool = True

    @abstractmethod
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Perform analysis on data.

        Args:
            data: Input data (typically DataFrame)
            **kwargs: Additional parameters

        Returns:
            Analysis results dictionary
        """
        pass

    def rank_indicators(self, data: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Rank indicators by importance/relevance.

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            List of ranked indicators with scores
        """
        # Default implementation - override in subclass
        return []

    def validate(self) -> bool:
        """Validate engine plugin."""
        if not self.name:
            raise ValueError("Engine must have a name")
        if self.supports_analysis and not hasattr(self, 'analyze'):
            raise ValueError("Engine must implement analyze()")
        return True


class WorkflowPlugin(PluginBase):
    """Base class for workflow plugins."""

    category: str = "workflow"

    # Workflow-specific metadata
    estimated_runtime: str = "1-5 minutes"
    required_lenses: List[str] = []
    output_type: str = "report"  # report, data, visualization

    @abstractmethod
    def execute(
        self,
        panel: Any,
        engines: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the workflow.

        Args:
            panel: Panel instance with indicators
            engines: Available engines
            **kwargs: Workflow parameters

        Returns:
            Workflow results
        """
        pass

    def validate(self) -> bool:
        """Validate workflow plugin."""
        if not self.name:
            raise ValueError("Workflow must have a name")
        if not hasattr(self, 'execute'):
            raise ValueError("Workflow must implement execute()")
        return True


class PanelPlugin(PluginBase):
    """Base class for panel plugins."""

    category: str = "panel"

    # Panel-specific metadata
    domain: str = ""
    indicator_count: int = 0
    data_sources: List[str] = []

    @abstractmethod
    def get_indicators(self) -> List[Dict[str, Any]]:
        """
        Get list of indicators for this panel.

        Returns:
            List of indicator configurations
        """
        pass

    @abstractmethod
    def load_data(self, **kwargs) -> Any:
        """
        Load data for panel indicators.

        Returns:
            Loaded data (typically DataFrame)
        """
        pass

    def validate(self) -> bool:
        """Validate panel plugin."""
        if not self.name:
            raise ValueError("Panel must have a name")
        if not hasattr(self, 'get_indicators'):
            raise ValueError("Panel must implement get_indicators()")
        return True
