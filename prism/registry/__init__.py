"""
PRISM Registry

Indicator definitions and configuration.

The registry defines WHAT indicators exist and WHERE they come from.
It does NOT fetch data or write to databases.
"""

from .loader import RegistryLoader, Indicator

__all__ = ["RegistryLoader", "Indicator"]
