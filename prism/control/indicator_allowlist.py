"""
Indicator Allowlist Registry

Single source of truth for which indicators are active.
All downstream scripts consume this, never compute their own exclusions.
"""

from typing import Set, Optional, FrozenSet
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class IndicatorAllowlist:
    """
    Immutable allowlist passed through pipeline.

    Scripts check: `if indicator_id in allowlist`
    Scripts NEVER check thresholds directly.
    """
    active_indicators: FrozenSet[str]
    excluded_indicators: FrozenSet[str]
    policy_mode: str

    def __contains__(self, indicator_id: str) -> bool:
        return indicator_id in self.active_indicators

    def __iter__(self):
        return iter(self.active_indicators)

    def __len__(self):
        return len(self.active_indicators)


class AllowlistRegistry:
    """
    Thread-safe global registry for current allowlist.

    Usage in scripts:
        from prism.control.indicator_allowlist import get_current_allowlist

        allowlist = get_current_allowlist()
        for indicator in my_indicators:
            if indicator not in allowlist:
                continue  # Skip excluded
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._allowlist = None
        return cls._instance

    def set_allowlist(self, allowlist: IndicatorAllowlist):
        with self._lock:
            self._allowlist = allowlist

    def get_allowlist(self) -> Optional[IndicatorAllowlist]:
        return self._allowlist

    def clear(self):
        with self._lock:
            self._allowlist = None


# Convenience functions
_registry = AllowlistRegistry()


def set_current_allowlist(allowlist: IndicatorAllowlist):
    _registry.set_allowlist(allowlist)


def get_current_allowlist() -> Optional[IndicatorAllowlist]:
    return _registry.get_allowlist()


def indicator_is_active(indicator_id: str) -> bool:
    """Check if indicator is in current allowlist."""
    allowlist = get_current_allowlist()
    if allowlist is None:
        return True  # No policy = allow all
    return indicator_id in allowlist
