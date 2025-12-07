"""
PRISM Workflow Loader
======================

Auto-discovers and loads workflows from the registry.
Workflows orchestrate multiple engines to produce analysis reports.

Usage:
    from workflows.workflow_loader import WorkflowLoader

    loader = WorkflowLoader()

    # List available workflows
    workflows = loader.list_workflows()

    # Get workflow info
    info = loader.get_workflow_info("regime_comparison")

    # Run a workflow
    results = loader.run_workflow("regime_comparison", panel="market")
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "workflows" / "registry.yaml"


class WorkflowLoadError(Exception):
    """Raised when a workflow cannot be loaded."""
    pass


class WorkflowLoader:
    """
    Loads and manages analysis workflows from the registry.

    Workflows orchestrate:
    - Which engines/lenses to run
    - What parameters to use
    - How to combine results
    - What outputs to generate
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the workflow loader.

        Args:
            registry_path: Path to registry.yaml
        """
        self.registry_path = registry_path or REGISTRY_PATH
        self._registry: Optional[Dict] = None
        self._workflow_cache: Dict[str, Any] = {}

    @property
    def registry(self) -> Dict:
        """Lazy-load the registry."""
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    def _load_registry(self) -> Dict:
        """Load the workflow registry from YAML."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Workflow registry not found: {self.registry_path}")

        with open(self.registry_path, 'r') as f:
            registry = yaml.safe_load(f)

        logger.info(f"Loaded workflow registry: {len(registry.get('workflows', {}))} workflows")
        return registry

    def reload_registry(self) -> None:
        """Force reload of the registry."""
        self._registry = None
        self._workflow_cache.clear()
        logger.info("Workflow registry reloaded")

    # -------------------------------------------------------------------------
    # DISCOVERY METHODS
    # -------------------------------------------------------------------------

    def list_workflows(self) -> List[str]:
        """List all available workflow names."""
        return list(self.registry.get("workflows", {}).keys())

    def get_workflow_info(self, workflow_key: str) -> Optional[Dict]:
        """Get full configuration for a workflow."""
        return self.registry.get("workflows", {}).get(workflow_key)

    def get_workflows_by_category(self, category: str) -> List[str]:
        """Get all workflows in a category."""
        return [
            key for key, config in self.registry.get("workflows", {}).items()
            if config.get("category") == category
        ]

    def get_compatible_workflows(self, panel: str) -> List[str]:
        """Get workflows compatible with a specific panel."""
        compatible = []

        for key, config in self.registry.get("workflows", {}).items():
            panels = config.get("compatible_panels", [])
            if "*" in panels or panel in panels:
                compatible.append(key)

        return compatible

    def get_categories(self) -> Dict[str, Dict]:
        """Get all workflow categories with descriptions."""
        return self.registry.get("categories", {})

    def get_presets(self) -> Dict[str, Dict]:
        """Get workflow presets (quick configurations)."""
        return self.registry.get("presets", {})

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configurations."""
        return self.registry.get("defaults", {})

    # -------------------------------------------------------------------------
    # PARAMETER HANDLING
    # -------------------------------------------------------------------------

    def get_workflow_parameters(self, workflow_key: str) -> Dict[str, Dict]:
        """
        Get parameter definitions for a workflow.

        Returns dict mapping param_name -> param_config
        """
        info = self.get_workflow_info(workflow_key)
        if not info:
            return {}
        return info.get("parameters", {})

    def get_default_parameters(self, workflow_key: str) -> Dict[str, Any]:
        """
        Get default parameter values for a workflow.

        Returns dict mapping param_name -> default_value
        """
        params = self.get_workflow_parameters(workflow_key)
        return {
            name: config.get("default")
            for name, config in params.items()
        }

    def validate_parameters(
        self,
        workflow_key: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and fill in missing parameters with defaults.

        Args:
            workflow_key: Workflow identifier
            params: User-provided parameters

        Returns:
            Complete parameter dict with defaults filled in
        """
        param_defs = self.get_workflow_parameters(workflow_key)
        defaults = self.get_default_parameters(workflow_key)

        # Start with defaults
        validated = defaults.copy()

        # Override with user params
        for key, value in params.items():
            if key in param_defs:
                # Validate type if specified
                expected_type = param_defs[key].get("type")
                choices = param_defs[key].get("choices")

                if choices and value not in choices:
                    raise ValueError(
                        f"Parameter {key} must be one of {choices}, got {value}"
                    )

                validated[key] = value
            else:
                logger.warning(f"Unknown parameter '{key}' for workflow {workflow_key}")

        return validated

    # -------------------------------------------------------------------------
    # ENGINE REQUIREMENTS
    # -------------------------------------------------------------------------

    def get_required_engines(self, workflow_key: str) -> Dict[str, List[str]]:
        """
        Get engines required by a workflow.

        Returns dict with keys: lenses, analysis, weighting, diagnostics
        """
        info = self.get_workflow_info(workflow_key)
        if not info:
            return {}

        return info.get("engines_required", {})

    def get_required_lenses(self, workflow_key: str) -> List[str]:
        """Get lens engines required by workflow."""
        required = self.get_required_engines(workflow_key)
        lenses = required.get("lenses", [])

        # Handle "all" keyword
        if lenses == "all":
            from engines.engine_loader import EngineLoader
            return EngineLoader().list_lenses()
        elif isinstance(lenses, str):
            # It's a lens set name
            from engines.engine_loader import EngineLoader
            return EngineLoader().get_lens_set(lenses)

        return lenses

    # -------------------------------------------------------------------------
    # OUTPUT CONFIGURATION
    # -------------------------------------------------------------------------

    def get_output_config(self, workflow_key: str) -> List[Dict]:
        """Get output configuration for a workflow."""
        info = self.get_workflow_info(workflow_key)
        if not info:
            return []
        return info.get("outputs", [])

    def resolve_output_path(
        self,
        output_config: Dict,
        base_dir: Optional[Path] = None
    ) -> Path:
        """
        Resolve output filename with timestamp substitution.

        Args:
            output_config: Output configuration dict
            base_dir: Base output directory

        Returns:
            Resolved output path
        """
        defaults = self.get_defaults()

        if base_dir is None:
            # Use default output pattern
            pattern = defaults.get("output_dir", "output/{date}_{time}")
            now = datetime.now()
            base_dir = PROJECT_ROOT / pattern.format(
                date=now.strftime("%Y-%m-%d"),
                time=now.strftime("%H%M")
            )

        base_dir.mkdir(parents=True, exist_ok=True)

        filename = output_config.get("filename", "output")

        # Substitute placeholders
        now = datetime.now()
        filename = filename.format(
            timestamp=now.strftime("%Y%m%d_%H%M%S"),
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H%M")
        )

        return base_dir / filename

    # -------------------------------------------------------------------------
    # LOADING & EXECUTION
    # -------------------------------------------------------------------------

    def load_workflow_class(self, workflow_key: str) -> Type:
        """
        Load a workflow class (without instantiating).

        Args:
            workflow_key: Workflow identifier

        Returns:
            Workflow class
        """
        config = self.get_workflow_info(workflow_key)
        if not config:
            raise WorkflowLoadError(f"Unknown workflow: {workflow_key}")

        module_path = config.get("module")
        class_name = config.get("class")

        if not module_path or not class_name:
            raise WorkflowLoadError(
                f"Workflow {workflow_key} missing module or class in registry"
            )

        try:
            module = importlib.import_module(module_path)
            workflow_class = getattr(module, class_name)
            return workflow_class

        except ImportError as e:
            raise WorkflowLoadError(f"Could not import {module_path}: {e}")
        except AttributeError as e:
            raise WorkflowLoadError(f"Class {class_name} not found in {module_path}: {e}")

    def load_workflow(
        self,
        workflow_key: str,
        **init_kwargs
    ) -> Any:
        """
        Load and instantiate a workflow.

        Args:
            workflow_key: Workflow identifier
            **init_kwargs: Arguments passed to workflow constructor

        Returns:
            Instantiated workflow
        """
        workflow_class = self.load_workflow_class(workflow_key)

        try:
            workflow = workflow_class(**init_kwargs)
            logger.info(f"Instantiated workflow: {workflow_key}")
            return workflow

        except Exception as e:
            raise WorkflowLoadError(f"Could not instantiate {workflow_key}: {e}")

    def run_workflow(
        self,
        workflow_key: str,
        panel: Optional[str] = None,
        output_dir: Optional[Path] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Load and run a workflow with given parameters.

        Args:
            workflow_key: Workflow identifier
            panel: Panel to use (market, economy, etc.)
            output_dir: Output directory (optional)
            **params: Workflow parameters

        Returns:
            Workflow results
        """
        info = self.get_workflow_info(workflow_key)
        if not info:
            raise WorkflowLoadError(f"Unknown workflow: {workflow_key}")

        # Validate panel compatibility
        if panel:
            compatible_panels = info.get("compatible_panels", ["*"])
            if "*" not in compatible_panels and panel not in compatible_panels:
                raise ValueError(
                    f"Workflow {workflow_key} not compatible with panel {panel}. "
                    f"Compatible panels: {compatible_panels}"
                )

        # Validate and fill parameters
        validated_params = self.validate_parameters(workflow_key, params)

        # Load workflow
        workflow = self.load_workflow(workflow_key)

        # Run workflow
        logger.info(f"Running workflow: {workflow_key}")
        logger.info(f"  Panel: {panel}")
        logger.info(f"  Parameters: {validated_params}")

        # Most workflows have a 'run' method
        if hasattr(workflow, 'run'):
            results = workflow.run(**validated_params)
        elif hasattr(workflow, 'analyze'):
            results = workflow.analyze(**validated_params)
        else:
            raise WorkflowLoadError(
                f"Workflow {workflow_key} has no run() or analyze() method"
            )

        logger.info(f"Workflow complete: {workflow_key}")
        return results

    def run_preset(
        self,
        preset_name: str,
        **override_params
    ) -> Dict[str, Any]:
        """
        Run a workflow preset with optional parameter overrides.

        Args:
            preset_name: Preset name from registry
            **override_params: Parameters to override

        Returns:
            Workflow results
        """
        presets = self.get_presets()

        if preset_name not in presets:
            raise WorkflowLoadError(f"Unknown preset: {preset_name}")

        preset = presets[preset_name]
        workflow_key = preset.get("workflow")
        preset_params = preset.get("parameters", {})

        # Merge preset params with overrides
        final_params = {**preset_params, **override_params}

        logger.info(f"Running preset: {preset_name} ({workflow_key})")
        return self.run_workflow(workflow_key, **final_params)

    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------

    def get_workflow_summary(self) -> str:
        """Get a formatted summary of all workflows."""
        lines = [
            "=" * 60,
            "PRISM WORKFLOW REGISTRY",
            "=" * 60,
            "",
        ]

        categories = self.get_categories()

        for cat_key, cat_info in categories.items():
            workflows = self.get_workflows_by_category(cat_key)
            if workflows:
                lines.append(f"📁 {cat_info.get('name', cat_key)}")
                lines.append(f"   {cat_info.get('description', '')}")
                for wf in workflows:
                    info = self.get_workflow_info(wf)
                    runtime = info.get("estimated_runtime", "unknown")
                    lines.append(f"   • {wf}: {info.get('name', '')} ({runtime})")
                lines.append("")

        # Presets
        presets = self.get_presets()
        if presets:
            lines.append("⚡ PRESETS (Quick Configurations)")
            for name, preset in presets.items():
                lines.append(f"   • {name}: {preset.get('description', '')}")
            lines.append("")

        lines.extend([
            "-" * 60,
            f"Total workflows: {len(self.list_workflows())}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"WorkflowLoader(workflows={len(self.list_workflows())})"


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_loader: Optional[WorkflowLoader] = None


def get_loader() -> WorkflowLoader:
    """Get the default workflow loader singleton."""
    global _default_loader
    if _default_loader is None:
        _default_loader = WorkflowLoader()
    return _default_loader


def list_workflows() -> List[str]:
    """List all available workflows."""
    return get_loader().list_workflows()


def run_workflow(workflow_key: str, **params) -> Dict[str, Any]:
    """Run a workflow by key."""
    return get_loader().run_workflow(workflow_key, **params)


def run_preset(preset_name: str, **params) -> Dict[str, Any]:
    """Run a workflow preset."""
    return get_loader().run_preset(preset_name, **params)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="PRISM Workflow Loader")
    parser.add_argument("--list", action="store_true", help="List all workflows")
    parser.add_argument("--presets", action="store_true", help="List presets")
    parser.add_argument("--info", type=str, help="Show info for specific workflow")
    parser.add_argument("--run", type=str, help="Run a workflow")
    parser.add_argument("--preset", type=str, help="Run a preset")
    parser.add_argument("--panel", type=str, default="market", help="Panel to use")

    args = parser.parse_args()

    loader = WorkflowLoader()

    if args.list:
        print(loader.get_workflow_summary())

    elif args.presets:
        print("\nWorkflow Presets:")
        for name, preset in loader.get_presets().items():
            print(f"  {name}:")
            print(f"    Workflow: {preset.get('workflow')}")
            print(f"    Description: {preset.get('description', '')}")
            print()

    elif args.info:
        info = loader.get_workflow_info(args.info)
        if info:
            print(f"\nWorkflow: {args.info}")
            print("-" * 40)
            print(f"Name: {info.get('name')}")
            print(f"Description: {info.get('description')}")
            print(f"Category: {info.get('category')}")
            print(f"Runtime: {info.get('estimated_runtime')}")
            print(f"Compatible panels: {info.get('compatible_panels')}")
            print(f"\nParameters:")
            for pname, pconfig in info.get("parameters", {}).items():
                print(f"  {pname}: {pconfig.get('type')} = {pconfig.get('default')}")
        else:
            print(f"Workflow not found: {args.info}")

    elif args.run:
        results = loader.run_workflow(args.run, panel=args.panel)
        print(f"\nResults: {results.get('report_path', 'complete')}")

    elif args.preset:
        results = loader.run_preset(args.preset)
        print(f"\nResults: {results.get('report_path', 'complete')}")

    else:
        print(loader.get_workflow_summary())
