"""
PRISM Engine Loader
====================

Auto-discovers and loads engines from the registry.
This is the foundation for the plugin system.

Usage:
    from engines.engine_loader import EngineLoader

    loader = EngineLoader()

    # Get all available engines
    engines = loader.list_engines()

    # Get engines by category
    lenses = loader.get_engines_by_category("geometry")

    # Load a specific engine
    pca = loader.load_engine("pca")

    # Run an engine
    results = loader.run_engine("pca", panel_data=df)
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
import yaml

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "engines" / "registry.yaml"


class EngineLoadError(Exception):
    """Raised when an engine cannot be loaded."""
    pass


class EngineLoader:
    """
    Loads and manages analysis engines from the registry.

    The loader provides:
    - Auto-discovery of engines from registry.yaml
    - Lazy loading of engine classes
    - Validation of engine compatibility
    - Caching of loaded engines
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize the engine loader.

        Args:
            registry_path: Path to registry.yaml (default: engines/registry.yaml)
        """
        self.registry_path = registry_path or REGISTRY_PATH
        self._registry: Optional[Dict] = None
        self._engine_cache: Dict[str, Any] = {}
        self._class_cache: Dict[str, Type] = {}

    @property
    def registry(self) -> Dict:
        """Lazy-load the registry."""
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    def _load_registry(self) -> Dict:
        """Load the engine registry from YAML."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Engine registry not found: {self.registry_path}")

        with open(self.registry_path, 'r') as f:
            registry = yaml.safe_load(f)

        logger.info(f"Loaded engine registry: {len(registry.get('engines', {}))} engines")
        return registry

    def reload_registry(self) -> None:
        """Force reload of the registry (for hot-reload during development)."""
        self._registry = None
        self._engine_cache.clear()
        self._class_cache.clear()
        logger.info("Registry reloaded")

    # -------------------------------------------------------------------------
    # DISCOVERY METHODS
    # -------------------------------------------------------------------------

    def list_engines(self) -> List[str]:
        """
        List all available engine names including plugins.

        Returns:
            List of engine keys
        """
        engines = set(self.registry.get("engines", {}).keys())

        # Plugin engines
        try:
            from core.plugin_loader import PluginLoader
            plugin_loader = PluginLoader()
            plugin_loader.discover_engines()
            engines.update(plugin_loader._engines.keys())
        except ImportError:
            pass

        return sorted(e for e in engines if e)

    def list_lenses(self) -> List[str]:
        """
        List all lens engines (subset used in consensus).

        Returns:
            List of lens engine keys
        """
        return [
            key for key, config in self.registry.get("engines", {}).items()
            if config.get("lens", False)
        ]

    def get_engine_info(self, engine_key: str) -> Optional[Dict]:
        """
        Get full configuration for an engine.

        Args:
            engine_key: Engine identifier

        Returns:
            Engine configuration dict or None
        """
        return self.registry.get("engines", {}).get(engine_key)

    def get_engines_by_category(self, category: str) -> List[str]:
        """
        Get all engines in a category.

        Args:
            category: Category name (geometry, causality, regime, etc.)

        Returns:
            List of engine keys in that category
        """
        return [
            key for key, config in self.registry.get("engines", {}).items()
            if config.get("category") == category
        ]

    def get_compatible_engines(
        self,
        panel: str,
        resolution: str = "daily"
    ) -> List[str]:
        """
        Get engines compatible with a specific panel and resolution.

        Args:
            panel: Panel name (market, economy, climate, etc.)
            resolution: Data resolution (daily, weekly, monthly)

        Returns:
            List of compatible engine keys
        """
        compatible = []

        for key, config in self.registry.get("engines", {}).items():
            panels = config.get("compatible_panels", [])
            resolutions = config.get("compatible_resolutions", [])

            # Check panel compatibility
            panel_ok = "*" in panels or panel in panels

            # Check resolution compatibility
            resolution_ok = "*" in resolutions or resolution in resolutions

            if panel_ok and resolution_ok:
                compatible.append(key)

        return compatible

    def get_lens_set(self, set_name: str = "full") -> List[str]:
        """
        Get a predefined set of lenses.

        Args:
            set_name: Set name (full, quick, geometry_only, etc.)

        Returns:
            List of lens keys in the set
        """
        defaults = self.registry.get("defaults", {})
        lens_sets = defaults.get("lens_sets", {})
        return lens_sets.get(set_name, self.list_lenses())

    def get_categories(self) -> Dict[str, Dict]:
        """
        Get all engine categories with descriptions.

        Returns:
            Dictionary of category info
        """
        return self.registry.get("categories", {})

    # -------------------------------------------------------------------------
    # LOADING METHODS
    # -------------------------------------------------------------------------

    def load_engine_class(self, engine_key: str) -> Type:
        """
        Load an engine class (without instantiating).

        Args:
            engine_key: Engine identifier

        Returns:
            Engine class
        """
        if engine_key in self._class_cache:
            return self._class_cache[engine_key]

        config = self.get_engine_info(engine_key)
        if not config:
            raise EngineLoadError(f"Unknown engine: {engine_key}")

        module_path = config.get("module")
        class_name = config.get("class")

        if not module_path or not class_name:
            raise EngineLoadError(
                f"Engine {engine_key} missing module or class in registry"
            )

        try:
            module = importlib.import_module(module_path)
            engine_class = getattr(module, class_name)
            self._class_cache[engine_key] = engine_class
            logger.debug(f"Loaded engine class: {engine_key}")
            return engine_class

        except ImportError as e:
            raise EngineLoadError(
                f"Could not import {module_path}: {e}"
            )
        except AttributeError as e:
            raise EngineLoadError(
                f"Class {class_name} not found in {module_path}: {e}"
            )

    def load_engine(
        self,
        engine_key: str,
        **init_kwargs
    ) -> Any:
        """
        Load and instantiate an engine.

        Args:
            engine_key: Engine identifier
            **init_kwargs: Arguments passed to engine constructor

        Returns:
            Instantiated engine
        """
        cache_key = f"{engine_key}_{hash(frozenset(init_kwargs.items()))}"

        if cache_key in self._engine_cache:
            return self._engine_cache[cache_key]

        engine_class = self.load_engine_class(engine_key)

        try:
            engine = engine_class(**init_kwargs)
            self._engine_cache[cache_key] = engine
            logger.info(f"Instantiated engine: {engine_key}")
            return engine

        except Exception as e:
            raise EngineLoadError(
                f"Could not instantiate {engine_key}: {e}"
            )

    def load_multiple_engines(
        self,
        engine_keys: List[str],
        **shared_kwargs
    ) -> Dict[str, Any]:
        """
        Load multiple engines at once.

        Args:
            engine_keys: List of engine identifiers
            **shared_kwargs: Arguments passed to all engine constructors

        Returns:
            Dictionary mapping engine_key -> engine instance
        """
        engines = {}
        errors = []

        for key in engine_keys:
            try:
                engines[key] = self.load_engine(key, **shared_kwargs)
            except EngineLoadError as e:
                logger.warning(f"Could not load {key}: {e}")
                errors.append((key, str(e)))

        if errors:
            logger.warning(f"Failed to load {len(errors)} engines: {errors}")

        return engines

    def load_all_lenses(self, set_name: str = "full") -> Dict[str, Any]:
        """
        Load all lenses in a set.

        Args:
            set_name: Lens set name (full, quick, etc.)

        Returns:
            Dictionary mapping lens_key -> lens instance
        """
        lens_keys = self.get_lens_set(set_name)
        return self.load_multiple_engines(lens_keys)

    # -------------------------------------------------------------------------
    # EXECUTION METHODS
    # -------------------------------------------------------------------------

    def run_engine(
        self,
        engine_key: str,
        method: str = "analyze",
        **kwargs
    ) -> Any:
        """
        Load and run an engine method.

        Args:
            engine_key: Engine identifier
            method: Method to call (analyze, rank_indicators, etc.)
            **kwargs: Arguments passed to the method

        Returns:
            Method result
        """
        engine = self.load_engine(engine_key)

        if not hasattr(engine, method):
            raise EngineLoadError(
                f"Engine {engine_key} has no method '{method}'"
            )

        method_func = getattr(engine, method)
        return method_func(**kwargs)

    def run_lens_rankings(
        self,
        engine_key: str,
        df,
        **kwargs
    ):
        """
        Run a lens engine and get indicator rankings.

        Args:
            engine_key: Lens engine identifier
            df: Input DataFrame
            **kwargs: Additional arguments

        Returns:
            Rankings DataFrame
        """
        return self.run_engine(
            engine_key,
            method="rank_indicators",
            df=df,
            **kwargs
        )

    def run_all_lenses(
        self,
        df,
        set_name: str = "full",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run all lenses and collect rankings.

        Args:
            df: Input DataFrame
            set_name: Lens set to use
            **kwargs: Additional arguments for each lens

        Returns:
            Dictionary mapping lens_key -> rankings DataFrame
        """
        results = {}
        lens_keys = self.get_lens_set(set_name)

        for key in lens_keys:
            try:
                results[key] = self.run_lens_rankings(key, df, **kwargs)
                logger.info(f"  {key}: completed")
            except Exception as e:
                logger.warning(f"  {key}: failed - {e}")
                results[key] = None

        return results

    # -------------------------------------------------------------------------
    # VALIDATION METHODS
    # -------------------------------------------------------------------------

    def validate_engine(self, engine_key: str) -> Dict[str, Any]:
        """
        Validate that an engine can be loaded and has required interface.

        Args:
            engine_key: Engine identifier

        Returns:
            Validation result dict
        """
        result = {
            "engine": engine_key,
            "valid": False,
            "loadable": False,
            "has_analyze": False,
            "has_rank_indicators": False,
            "errors": [],
        }

        config = self.get_engine_info(engine_key)
        if not config:
            result["errors"].append(f"Engine not in registry")
            return result

        try:
            engine_class = self.load_engine_class(engine_key)
            result["loadable"] = True

            # Check for required methods
            result["has_analyze"] = hasattr(engine_class, "analyze")
            result["has_rank_indicators"] = hasattr(engine_class, "rank_indicators")

            # For lenses, rank_indicators is required
            if config.get("lens", False):
                if not result["has_rank_indicators"]:
                    result["errors"].append("Lens missing rank_indicators method")

            result["valid"] = result["loadable"] and not result["errors"]

        except EngineLoadError as e:
            result["errors"].append(str(e))

        return result

    def validate_all_engines(self) -> Dict[str, Dict]:
        """
        Validate all engines in the registry.

        Returns:
            Dictionary mapping engine_key -> validation result
        """
        results = {}
        for key in self.list_engines():
            results[key] = self.validate_engine(key)

        valid_count = sum(1 for r in results.values() if r["valid"])
        logger.info(f"Validation complete: {valid_count}/{len(results)} engines valid")

        return results

    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------

    def get_engine_summary(self) -> str:
        """Get a formatted summary of all engines."""
        lines = [
            "=" * 60,
            "PRISM ENGINE REGISTRY",
            "=" * 60,
            "",
        ]

        categories = self.get_categories()

        for cat_key, cat_info in categories.items():
            engines = self.get_engines_by_category(cat_key)
            if engines:
                lines.append(f"📁 {cat_info.get('name', cat_key)}")
                lines.append(f"   {cat_info.get('description', '')}")
                for eng in engines:
                    info = self.get_engine_info(eng)
                    lens_marker = "🔍" if info.get("lens") else "  "
                    lines.append(f"   {lens_marker} {eng}: {info.get('name', '')}")
                lines.append("")

        lines.extend([
            "-" * 60,
            f"Total engines: {len(self.list_engines())}",
            f"Lens engines: {len(self.list_lenses())}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"EngineLoader(engines={len(self.list_engines())})"


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_loader: Optional[EngineLoader] = None


def get_loader() -> EngineLoader:
    """Get the default engine loader singleton."""
    global _default_loader
    if _default_loader is None:
        _default_loader = EngineLoader()
    return _default_loader


def list_engines() -> List[str]:
    """List all available engines."""
    return get_loader().list_engines()


def list_lenses() -> List[str]:
    """List all lens engines."""
    return get_loader().list_lenses()


def load_engine(engine_key: str, **kwargs) -> Any:
    """Load an engine by key."""
    return get_loader().load_engine(engine_key, **kwargs)


def run_engine(engine_key: str, method: str = "analyze", **kwargs) -> Any:
    """Load and run an engine method."""
    return get_loader().run_engine(engine_key, method, **kwargs)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="PRISM Engine Loader")
    parser.add_argument("--list", action="store_true", help="List all engines")
    parser.add_argument("--lenses", action="store_true", help="List lens engines")
    parser.add_argument("--validate", action="store_true", help="Validate all engines")
    parser.add_argument("--info", type=str, help="Show info for specific engine")
    parser.add_argument("--category", type=str, help="List engines in category")

    args = parser.parse_args()

    loader = EngineLoader()

    if args.list:
        print(loader.get_engine_summary())

    elif args.lenses:
        print("Lens Engines:")
        for lens in loader.list_lenses():
            info = loader.get_engine_info(lens)
            print(f"  {lens}: {info.get('name', '')}")

    elif args.validate:
        results = loader.validate_all_engines()
        print("\nValidation Results:")
        for key, result in results.items():
            status = "✅" if result["valid"] else "❌"
            print(f"  {status} {key}")
            if result["errors"]:
                for err in result["errors"]:
                    print(f"      Error: {err}")

    elif args.info:
        info = loader.get_engine_info(args.info)
        if info:
            print(f"\nEngine: {args.info}")
            print("-" * 40)
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"Engine not found: {args.info}")

    elif args.category:
        engines = loader.get_engines_by_category(args.category)
        print(f"\nEngines in category '{args.category}':")
        for eng in engines:
            info = loader.get_engine_info(eng)
            print(f"  {eng}: {info.get('name', '')}")

    else:
        print(loader.get_engine_summary())
