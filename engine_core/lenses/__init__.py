# engine_core/lenses/__init__.py
"""
PRISM Lens Registry - Auto-Discovery

Convention:
    File:  {name}_lens.py
    Class: {Name}Lens
    Key:   {name}

All lenses following this convention are automatically registered.
No manual imports required - just create a new file and it's discovered.
"""

import importlib
import pkgutil
from pathlib import Path

# Base class always available
from .base_lens import BaseLens

LENS_CLASSES = {}


def _to_class_name(module_name: str) -> str:
    """Convert module name to class name: pca_lens -> PCALens"""
    parts = module_name.replace('_lens', '').split('_')
    # Handle acronyms (pca, dmd, tda) vs words (wavelet, network)
    return ''.join(
        p.upper() if len(p) <= 3 else p.title()
        for p in parts
    ) + 'Lens'


def _discover_lenses():
    """Auto-discover all *_lens.py modules."""
    _lens_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(_lens_dir)]):
        if module_info.name.endswith('_lens') and module_info.name != 'base_lens':
            try:
                module = importlib.import_module(f'.{module_info.name}', __package__)
                class_name = _to_class_name(module_info.name)

                if hasattr(module, class_name):
                    key = module_info.name.replace('_lens', '')
                    LENS_CLASSES[key] = getattr(module, class_name)
            except ImportError:
                pass  # Skip lenses with missing dependencies


def get_lens(name: str) -> BaseLens:
    """
    Get a lens instance by name.

    Args:
        name: Lens key (e.g., 'pca', 'granger', 'network')

    Returns:
        Instantiated lens object
    """
    if name not in LENS_CLASSES:
        raise KeyError(f"Unknown lens: {name}. Available: {list(LENS_CLASSES.keys())}")
    return LENS_CLASSES[name]()


def list_lenses() -> list:
    """List all available lens names."""
    return sorted(LENS_CLASSES.keys())


# Run discovery on import
_discover_lenses()

__all__ = ['BaseLens', 'LENS_CLASSES', 'get_lens', 'list_lenses']
