# prism/engine/registry.py
"""
Registry for engine components.

Two registries:
- LENS_REGISTRY: Analytical lenses (observe data, emit metrics)
- ENGINE_REGISTRY: Legacy engines (for backward compatibility)

Use get_lens_class() for new code.
"""

from typing import Dict, Type
import importlib

from prism.engine.lens_base import LensBase


# ---------------------------------------------------------------------------
# Lens Registry (new architecture)
# ---------------------------------------------------------------------------

LENS_REGISTRY: Dict[str, str] = {
    "geometry": "prism.engine.geometry.GeometryLens",
    "entropy": "prism.engine.entropy.EntropyLens",
    "gap": "prism.engine.gap.GapLens",
    "curvature": "prism.engine.curvature.CurvatureLens",
    "changepoint": "prism.engine.changepoint.ChangepointLens",
    "sparsity": "prism.engine.sparsity.SparsityLens",
    "compare": "prism.engine.compare.CompareLens",
}


def get_lens_class(name: str) -> Type[LensBase]:
    """
    Resolve a lens class from the registry.

    Args:
        name: Lens name (e.g., "entropy", "curvature")

    Returns:
        LensBase subclass

    Raises:
        KeyError: If lens name not found
    """
    if name not in LENS_REGISTRY:
        available = ", ".join(sorted(LENS_REGISTRY.keys()))
        raise KeyError(f"Unknown lens: '{name}'. Available: {available}")

    path = LENS_REGISTRY[name]
    module_path, class_name = path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_lens(name: str) -> LensBase:
    """
    Get an instantiated lens from the registry.

    Args:
        name: Lens name

    Returns:
        LensBase instance
    """
    cls = get_lens_class(name)
    return cls()


def list_lenses() -> list:
    """List all available lens names."""
    return sorted(LENS_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Legacy Engine Registry (backward compatibility)
# ---------------------------------------------------------------------------

ENGINE_REGISTRY: Dict[str, str] = {
    # These map to the same classes but via old naming
    # Kept for backward compatibility with existing code
    "normalize": "prism.engine.normalize.normalize_vector",  # function, not class
    "entropy": "prism.engine.entropy.EntropyLens",
    "gap": "prism.engine.gap.GapLens",
    "curvature": "prism.engine.curvature.CurvatureLens",
    "changepoint": "prism.engine.changepoint.ChangepointLens",
    "sparsity": "prism.engine.sparsity.SparsityLens",
    "compare": "prism.engine.compare.CompareLens",
    "cluster": "prism.engine.cluster.ClusterEngine",  # Not a lens (different pattern)
}


def get_engine_class(name: str):
    """
    Legacy: Resolve an engine class from the registry.

    Prefer get_lens_class() for new code.
    """
    if name not in ENGINE_REGISTRY:
        raise KeyError(f"Unknown engine: {name}")

    path = ENGINE_REGISTRY[name]
    module_path, class_name = path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
