"""
Fetch submodule - re-exports from fetch
"""
import sys
from pathlib import Path

# Ensure parent is in path
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# Re-export everything from fetch
from importlib import import_module as _import

_fetch = _import("fetch")

BaseFetcher = _fetch.BaseFetcher
FREDFetcher = _fetch.FREDFetcher
YahooFetcher = _fetch.YahooFetcher
ClimateFetcher = _fetch.ClimateFetcher
CustomFetcher = _fetch.CustomFetcher

__all__ = [
    'BaseFetcher',
    'FREDFetcher',
    'YahooFetcher',
    'ClimateFetcher',
    'CustomFetcher',
]
