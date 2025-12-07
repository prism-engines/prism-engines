"""
PRISM Plugin Loader
====================

Auto-discovers and loads plugins from designated directories.
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Type, Any, Optional

import yaml

from .plugin_base import PluginBase, EnginePlugin, WorkflowPlugin, PanelPlugin

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Discovers and loads PRISM plugins.

    Plugin sources (in priority order):
    1. Built-in (engines/, workflows/, panels/)
    2. User plugins (plugins/engines/, plugins/workflows/, plugins/panels/)
    3. External packages (via entry points - future)
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize plugin loader.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.plugins_dir = self.project_root / "plugins"

        # Plugin registries
        self._engines: Dict[str, Type[EnginePlugin]] = {}
        self._workflows: Dict[str, Type[WorkflowPlugin]] = {}
        self._panels: Dict[str, Type[PanelPlugin]] = {}

        # Track sources
        self._sources: Dict[str, str] = {}

    def discover_all(self) -> Dict[str, int]:
        """
        Discover all plugins.

        Returns:
            Count of discovered plugins by type
        """
        counts = {
            "engines": self.discover_engines(),
            "workflows": self.discover_workflows(),
            "panels": self.discover_panels(),
        }

        logger.info(f"Discovered plugins: {counts}")
        return counts

    def discover_engines(self) -> int:
        """Discover engine plugins."""
        count = 0

        # Built-in engines (from registry)
        builtin_registry = self.project_root / "engines" / "registry.yaml"
        if builtin_registry.exists():
            count += self._load_from_registry(builtin_registry, "engine")

        # Plugin engines
        plugin_engines_dir = self.plugins_dir / "engines"
        if plugin_engines_dir.exists():
            count += self._discover_python_plugins(plugin_engines_dir, EnginePlugin)

        return count

    def discover_workflows(self) -> int:
        """Discover workflow plugins."""
        count = 0

        # Built-in workflows (from registry)
        builtin_registry = self.project_root / "workflows" / "registry.yaml"
        if builtin_registry.exists():
            count += self._load_from_registry(builtin_registry, "workflow")

        # Plugin workflows
        plugin_workflows_dir = self.plugins_dir / "workflows"
        if plugin_workflows_dir.exists():
            count += self._discover_python_plugins(plugin_workflows_dir, WorkflowPlugin)

        return count

    def discover_panels(self) -> int:
        """Discover panel plugins."""
        count = 0

        # Built-in panels
        panels_dir = self.project_root / "panels"
        if panels_dir.exists():
            for panel_dir in panels_dir.iterdir():
                if panel_dir.is_dir() and (panel_dir / "registry.yaml").exists():
                    count += 1
                    self._sources[panel_dir.name] = "builtin"

        # Plugin panels
        plugin_panels_dir = self.plugins_dir / "panels"
        if plugin_panels_dir.exists():
            for panel_dir in plugin_panels_dir.iterdir():
                if panel_dir.is_dir():
                    if (panel_dir / "registry.yaml").exists() or (panel_dir / "panel.py").exists():
                        count += 1
                        self._sources[panel_dir.name] = "plugin"

        return count

    def _load_from_registry(self, registry_path: Path, plugin_type: str) -> int:
        """Load plugins from YAML registry."""
        try:
            with open(registry_path) as f:
                registry = yaml.safe_load(f) or {}

            items = registry.get(f"{plugin_type}s", registry.get("items", []))
            if isinstance(items, list):
                return len(items)
            elif isinstance(items, dict):
                return len(items)
            return 0

        except Exception as e:
            logger.warning(f"Failed to load registry {registry_path}: {e}")
            return 0

    def _discover_python_plugins(
        self,
        directory: Path,
        base_class: Type[PluginBase]
    ) -> int:
        """Discover Python plugin files."""
        count = 0

        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                # Load module
                spec = importlib.util.spec_from_file_location(
                    py_file.stem,
                    py_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[py_file.stem] = module
                    spec.loader.exec_module(module)

                    # Find plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, base_class)
                            and attr is not base_class
                            and hasattr(attr, 'name')
                        ):
                            plugin_name = getattr(attr, 'name', attr_name.lower())

                            if base_class == EnginePlugin:
                                self._engines[plugin_name] = attr
                            elif base_class == WorkflowPlugin:
                                self._workflows[plugin_name] = attr
                            elif base_class == PanelPlugin:
                                self._panels[plugin_name] = attr

                            self._sources[plugin_name] = f"plugin:{py_file.name}"
                            count += 1
                            logger.info(f"Discovered plugin: {plugin_name} from {py_file.name}")

            except Exception as e:
                logger.warning(f"Failed to load plugin {py_file}: {e}")

        return count

    def list_engines(self) -> List[str]:
        """List all discovered engines."""
        # Combine built-in and plugin engines
        engines = set()

        # Built-in from registry
        builtin_registry = self.project_root / "engines" / "registry.yaml"
        if builtin_registry.exists():
            try:
                with open(builtin_registry) as f:
                    registry = yaml.safe_load(f) or {}
                for engine in registry.get("engines", []):
                    if isinstance(engine, dict):
                        engines.add(engine.get("key", engine.get("name", "")))
                    else:
                        engines.add(str(engine))
            except Exception:
                pass

        # Plugin engines
        engines.update(self._engines.keys())

        return sorted(engines)

    def list_workflows(self) -> List[str]:
        """List all discovered workflows."""
        workflows = set()

        # Built-in from registry
        builtin_registry = self.project_root / "workflows" / "registry.yaml"
        if builtin_registry.exists():
            try:
                with open(builtin_registry) as f:
                    registry = yaml.safe_load(f) or {}
                for wf in registry.get("workflows", []):
                    if isinstance(wf, dict):
                        workflows.add(wf.get("key", wf.get("name", "")))
                    else:
                        workflows.add(str(wf))
            except Exception:
                pass

        # Plugin workflows
        workflows.update(self._workflows.keys())

        return sorted(workflows)

    def list_panels(self) -> List[str]:
        """List all discovered panels."""
        panels = set()

        # Built-in panels
        panels_dir = self.project_root / "panels"
        if panels_dir.exists():
            for panel_dir in panels_dir.iterdir():
                if panel_dir.is_dir() and not panel_dir.name.startswith("_"):
                    if (panel_dir / "registry.yaml").exists():
                        panels.add(panel_dir.name)

        # Plugin panels
        plugin_panels_dir = self.plugins_dir / "panels"
        if plugin_panels_dir.exists():
            for panel_dir in plugin_panels_dir.iterdir():
                if panel_dir.is_dir() and not panel_dir.name.startswith("_"):
                    panels.add(panel_dir.name)

        return sorted(panels)

    def get_engine(self, name: str) -> Optional[Type[EnginePlugin]]:
        """Get engine class by name."""
        return self._engines.get(name)

    def get_workflow(self, name: str) -> Optional[Type[WorkflowPlugin]]:
        """Get workflow class by name."""
        return self._workflows.get(name)

    def get_panel(self, name: str) -> Optional[Type[PanelPlugin]]:
        """Get panel class by name."""
        return self._panels.get(name)

    def get_source(self, plugin_name: str) -> str:
        """Get source of a plugin (builtin or plugin path)."""
        return self._sources.get(plugin_name, "unknown")
