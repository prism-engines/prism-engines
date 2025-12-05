"""
Registry module for PRISM Engine.

Provides centralized metric registry loading and validation.
"""

from .registry_loader import (
    load_metric_registry,
    validate_registry,
    get_all_metric_names,
    get_dependencies,
)

__all__ = [
    "load_metric_registry",
    "validate_registry",
    "get_all_metric_names",
    "get_dependencies",
]
