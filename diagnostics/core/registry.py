"""
Diagnostic Registry

Manages registration and discovery of diagnostic checks.
Supports auto-discovery of diagnostics from the diagnostics module.
"""

from typing import Dict, List, Optional, Type
import importlib
import pkgutil
from pathlib import Path

from diagnostics.core.base import BaseDiagnostic, DiagnosticCategory


class DiagnosticRegistry:
    """
    Registry for diagnostic checks.

    Handles registration, discovery, and retrieval of diagnostics.
    Supports filtering by category and automatic discovery.
    """

    def __init__(self):
        self._diagnostics: Dict[str, BaseDiagnostic] = {}
        self._discovered = False

    def register(self, diagnostic: BaseDiagnostic) -> None:
        """
        Register a diagnostic instance.

        Args:
            diagnostic: Diagnostic instance to register
        """
        if not isinstance(diagnostic, BaseDiagnostic):
            raise TypeError(f"Expected BaseDiagnostic, got {type(diagnostic)}")
        self._diagnostics[diagnostic.name] = diagnostic

    def register_class(self, diagnostic_class: Type[BaseDiagnostic]) -> None:
        """
        Register a diagnostic class (will be instantiated).

        Args:
            diagnostic_class: Diagnostic class to register
        """
        if not issubclass(diagnostic_class, BaseDiagnostic):
            raise TypeError(f"Expected BaseDiagnostic subclass, got {diagnostic_class}")
        instance = diagnostic_class()
        self.register(instance)

    def unregister(self, name: str) -> bool:
        """
        Unregister a diagnostic by name.

        Args:
            name: Name of diagnostic to unregister

        Returns:
            True if diagnostic was removed, False if not found
        """
        if name in self._diagnostics:
            del self._diagnostics[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseDiagnostic]:
        """
        Get a diagnostic by name.

        Args:
            name: Name of diagnostic

        Returns:
            Diagnostic instance or None if not found
        """
        self._ensure_discovered()
        return self._diagnostics.get(name)

    def get_all(self) -> List[BaseDiagnostic]:
        """
        Get all registered diagnostics.

        Returns:
            List of all diagnostic instances
        """
        self._ensure_discovered()
        return list(self._diagnostics.values())

    def get_by_category(self, category: DiagnosticCategory) -> List[BaseDiagnostic]:
        """
        Get diagnostics filtered by category.

        Args:
            category: Category to filter by

        Returns:
            List of diagnostics in the specified category
        """
        self._ensure_discovered()
        return [d for d in self._diagnostics.values() if d.category == category]

    def get_quick_diagnostics(self) -> List[BaseDiagnostic]:
        """
        Get diagnostics marked as quick (for fast runs).

        Returns:
            List of quick diagnostics
        """
        self._ensure_discovered()
        return [d for d in self._diagnostics.values() if d.is_quick]

    def list_names(self) -> List[str]:
        """
        Get list of all registered diagnostic names.

        Returns:
            List of diagnostic names
        """
        self._ensure_discovered()
        return list(self._diagnostics.keys())

    def _ensure_discovered(self) -> None:
        """Ensure auto-discovery has been run."""
        if not self._discovered:
            self.discover()

    def discover(self) -> int:
        """
        Auto-discover and register diagnostics from submodules.

        Scans the health/, performance/, and validation/ subdirectories
        for diagnostic classes.

        Returns:
            Number of diagnostics discovered
        """
        count_before = len(self._diagnostics)

        # Import submodules to trigger registration
        submodules = ['health', 'performance', 'validation']
        for submodule in submodules:
            try:
                module = importlib.import_module(f'diagnostics.{submodule}')
                # Check for get_diagnostics function
                if hasattr(module, 'get_diagnostics'):
                    for diag in module.get_diagnostics():
                        self.register(diag)
            except ImportError:
                pass  # Module doesn't exist or has import errors

        self._discovered = True
        return len(self._diagnostics) - count_before

    def clear(self) -> None:
        """Clear all registered diagnostics."""
        self._diagnostics.clear()
        self._discovered = False

    def __len__(self) -> int:
        return len(self._diagnostics)

    def __iter__(self):
        self._ensure_discovered()
        return iter(self._diagnostics.values())

    def __contains__(self, name: str) -> bool:
        return name in self._diagnostics


# Global registry instance
_global_registry: Optional[DiagnosticRegistry] = None


def get_registry() -> DiagnosticRegistry:
    """
    Get the global diagnostic registry.

    Returns:
        The global DiagnosticRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = DiagnosticRegistry()
    return _global_registry


def register_diagnostic(diagnostic: BaseDiagnostic) -> None:
    """
    Register a diagnostic in the global registry.

    Args:
        diagnostic: Diagnostic to register
    """
    get_registry().register(diagnostic)


def diagnostic(cls: Type[BaseDiagnostic]) -> Type[BaseDiagnostic]:
    """
    Decorator to auto-register a diagnostic class.

    Example:
        @diagnostic
        class MyDiagnostic(BaseDiagnostic):
            name = "my_diagnostic"
            ...
    """
    get_registry().register_class(cls)
    return cls
