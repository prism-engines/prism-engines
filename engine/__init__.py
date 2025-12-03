"""
Engine - Probability Research and Integrated Statistical Mathematics Engine

A quantitative analysis framework with multiple mathematical "lenses" for
analyzing market data and financial indicators.

Usage:
    from engine import IndicatorEngine

    engine = IndicatorEngine()
    results = engine.analyze(panel_data, mode="basic")
    print(results["top_indicators"])

    # Domain-specific engines (registry-driven)
    from engine import PrismMacroEngine, PrismMarketEngine

    macro = PrismMacroEngine()
    macro_results = macro.analyze()

    market = PrismMarketEngine()
    market_results = market.analyze()
"""

__version__ = "0.1.0"

# Import subpackages - they reference the numbered directories
# via sys.path manipulation in each subpackage __init__

import sys
from pathlib import Path

# Add parent directory to path for importing numbered directories
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


# Lazy loading to avoid circular imports
def __getattr__(name):
    """Lazy load submodules and classes."""

    if name == "fetch":
        from importlib import import_module
        return import_module("01_fetch")

    elif name == "cleaning":
        from importlib import import_module
        return import_module("03_cleaning")

    elif name == "engine":
        from importlib import import_module
        return import_module("05_engine")

    elif name == "validation":
        from importlib import import_module
        return import_module("validation")

    elif name == "utils":
        from importlib import import_module
        return import_module("utils")

    elif name == "IndicatorEngine":
        from importlib import import_module
        engine = import_module("05_engine")
        return engine.IndicatorEngine

    elif name == "LensComparator":
        from importlib import import_module
        engine = import_module("05_engine")
        return engine.LensComparator

    elif name == "get_lens":
        from importlib import import_module
        engine = import_module("05_engine")
        return engine.get_lens

    # Domain-specific engines (registry-driven)
    elif name == "PrismMacroEngine":
        from .prism_macro_engine import PrismMacroEngine
        return PrismMacroEngine

    elif name == "PrismMarketEngine":
        from .prism_market_engine import PrismMarketEngine
        return PrismMarketEngine

    elif name == "PrismStressEngine":
        from .prism_stress_engine import PrismStressEngine
        return PrismStressEngine

    elif name == "PrismMLEngine":
        from .prism_ml_engine import PrismMLEngine
        return PrismMLEngine

    raise AttributeError(f"module 'engine' has no attribute '{name}'")


__all__ = [
    'fetch',
    'cleaning',
    'engine',
    'validation',
    'utils',
    'IndicatorEngine',
    'LensComparator',
    'get_lens',
    # Domain-specific engines
    'PrismMacroEngine',
    'PrismMarketEngine',
    'PrismStressEngine',
    'PrismMLEngine',
]
