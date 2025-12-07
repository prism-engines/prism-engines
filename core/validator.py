"""
PRISM Plugin Validator
=======================

Validates plugins meet requirements.
"""

import logging
from typing import Dict, Any, List, Type, Optional
from pathlib import Path
import yaml

from .plugin_base import PluginBase, EnginePlugin, WorkflowPlugin, PanelPlugin

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Plugin validation error."""
    pass


class PluginValidator:
    """Validates PRISM plugins."""

    @classmethod
    def validate_engine(cls, engine_class: Type[EnginePlugin]) -> Dict[str, Any]:
        """
        Validate an engine plugin.

        Returns:
            Validation result with status and details
        """
        result = {
            "valid": False,
            "name": getattr(engine_class, 'name', 'unknown'),
            "errors": [],
            "warnings": [],
        }

        try:
            # Check required attributes
            if not hasattr(engine_class, 'name') or not engine_class.name:
                result["errors"].append("Missing 'name' attribute")

            # Check required methods
            if not hasattr(engine_class, 'analyze'):
                result["errors"].append("Missing 'analyze' method")

            # Check method signatures
            if hasattr(engine_class, 'analyze'):
                import inspect
                sig = inspect.signature(engine_class.analyze)
                params = list(sig.parameters.keys())
                if 'data' not in params and len(params) < 2:
                    result["warnings"].append("analyze() should accept 'data' parameter")

            # Check optional attributes
            if not hasattr(engine_class, 'version'):
                result["warnings"].append("Missing 'version' attribute")

            if not hasattr(engine_class, 'description'):
                result["warnings"].append("Missing 'description' attribute")

            # Try instantiation
            try:
                instance = engine_class()
                if hasattr(instance, 'validate'):
                    instance.validate()
            except Exception as e:
                result["errors"].append(f"Instantiation failed: {e}")

            result["valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["errors"].append(f"Validation error: {e}")

        return result

    @classmethod
    def validate_workflow(cls, workflow_class: Type[WorkflowPlugin]) -> Dict[str, Any]:
        """Validate a workflow plugin."""
        result = {
            "valid": False,
            "name": getattr(workflow_class, 'name', 'unknown'),
            "errors": [],
            "warnings": [],
        }

        try:
            if not hasattr(workflow_class, 'name') or not workflow_class.name:
                result["errors"].append("Missing 'name' attribute")

            if not hasattr(workflow_class, 'execute'):
                result["errors"].append("Missing 'execute' method")

            if not hasattr(workflow_class, 'estimated_runtime'):
                result["warnings"].append("Missing 'estimated_runtime' attribute")

            result["valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["errors"].append(f"Validation error: {e}")

        return result

    @classmethod
    def validate_panel(cls, panel_path: Path) -> Dict[str, Any]:
        """Validate a panel plugin directory."""
        result = {
            "valid": False,
            "name": panel_path.name,
            "errors": [],
            "warnings": [],
        }

        try:
            # Check registry.yaml exists
            registry_file = panel_path / "registry.yaml"
            if not registry_file.exists():
                result["errors"].append("Missing registry.yaml")
            else:
                # Validate registry content
                try:
                    with open(registry_file) as f:
                        registry = yaml.safe_load(f) or {}

                    if "name" not in registry:
                        result["warnings"].append("registry.yaml missing 'name'")

                    if "indicators" not in registry:
                        result["warnings"].append("registry.yaml missing 'indicators'")

                except Exception as e:
                    result["errors"].append(f"Invalid registry.yaml: {e}")

            # Check panel.py exists
            panel_file = panel_path / "panel.py"
            if not panel_file.exists():
                result["warnings"].append("Missing panel.py (optional)")

            result["valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["errors"].append(f"Validation error: {e}")

        return result

    @classmethod
    def validate_all_plugins(cls, project_root: Path) -> Dict[str, List[Dict]]:
        """Validate all discovered plugins."""
        from .plugin_loader import PluginLoader

        loader = PluginLoader(project_root)
        loader.discover_all()

        results = {
            "engines": [],
            "workflows": [],
            "panels": [],
        }

        # Validate engines
        for name, engine_class in loader._engines.items():
            result = cls.validate_engine(engine_class)
            results["engines"].append(result)

        # Validate workflows
        for name, workflow_class in loader._workflows.items():
            result = cls.validate_workflow(workflow_class)
            results["workflows"].append(result)

        # Validate panels
        panels_dir = project_root / "panels"
        if panels_dir.exists():
            for panel_dir in panels_dir.iterdir():
                if panel_dir.is_dir() and not panel_dir.name.startswith("_"):
                    result = cls.validate_panel(panel_dir)
                    results["panels"].append(result)

        # Plugin panels
        plugin_panels = project_root / "plugins" / "panels"
        if plugin_panels.exists():
            for panel_dir in plugin_panels.iterdir():
                if panel_dir.is_dir() and not panel_dir.name.startswith("_"):
                    result = cls.validate_panel(panel_dir)
                    results["panels"].append(result)

        return results
