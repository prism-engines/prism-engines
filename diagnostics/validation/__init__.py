"""
Validation Diagnostics

Plugin and configuration validation checks.
"""

from diagnostics.validation.plugins import (
    EngineValidationDiagnostic,
    WorkflowValidationDiagnostic,
    PanelValidationDiagnostic,
)
from diagnostics.validation.config import (
    ConfigurationDiagnostic,
)


def get_diagnostics():
    """Return all validation diagnostics."""
    return [
        EngineValidationDiagnostic(),
        WorkflowValidationDiagnostic(),
        PanelValidationDiagnostic(),
        ConfigurationDiagnostic(),
    ]


__all__ = [
    'get_diagnostics',
    'EngineValidationDiagnostic',
    'WorkflowValidationDiagnostic',
    'PanelValidationDiagnostic',
    'ConfigurationDiagnostic',
]
