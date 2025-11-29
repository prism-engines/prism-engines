"""
PRISM Engine - Probability Research and Integrated Statistical Mathematics Engine

A quantitative analysis framework with multiple mathematical "lenses" for
analyzing market data and financial indicators.

Usage:
    from prism_engine import IndicatorEngine

    engine = IndicatorEngine()
    results = engine.analyze(panel_data, mode="basic")
    print(results["top_indicators"])
"""

__version__ = "0.1.0"

# Re-export main components for easy imports
# These will be available after the package structure is finalized

# For now, provide import paths
PACKAGE_STRUCTURE = """
prism_engine/
├── fetch/          -> from prism_engine.fetch import FREDFetcher, YahooFetcher
├── cleaning/       -> from prism_engine.cleaning import NaNAnalyzer, get_strategy
├── engine/
│   ├── lenses/     -> from prism_engine.engine.lenses import PCALens, GrangerLens
│   └── orchestration/ -> from prism_engine.engine import IndicatorEngine
├── validation/     -> from prism_engine.validation import LensValidator
└── utils/          -> from prism_engine.utils import setup_logging
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "IndicatorEngine":
        from .engine.orchestration import IndicatorEngine
        return IndicatorEngine
    elif name == "LensComparator":
        from .engine.orchestration import LensComparator
        return LensComparator
    elif name == "get_lens":
        from .engine import get_lens
        return get_lens
    raise AttributeError(f"module 'prism_engine' has no attribute '{name}'")
