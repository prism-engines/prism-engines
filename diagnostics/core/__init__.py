"""
Diagnostics Core Module

Contains base classes, runner, and utilities for the diagnostics framework.
"""

from diagnostics.core.base import (
    BaseDiagnostic,
    DiagnosticStatus,
    DiagnosticResult,
    DiagnosticCategory,
)
from diagnostics.core.runner import DiagnosticRunner
from diagnostics.core.registry import DiagnosticRegistry, get_registry
from diagnostics.core.report import DiagnosticReporter

__all__ = [
    'BaseDiagnostic',
    'DiagnosticStatus',
    'DiagnosticResult',
    'DiagnosticCategory',
    'DiagnosticRunner',
    'DiagnosticRegistry',
    'DiagnosticReporter',
    'get_registry',
]
