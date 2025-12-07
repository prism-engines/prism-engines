"""
Health Diagnostics

System and application health checks for PRISM.
"""

from diagnostics.health.system import (
    PythonVersionDiagnostic,
    DiskSpaceDiagnostic,
    MemoryDiagnostic,
    DependenciesDiagnostic,
)
from diagnostics.health.database import (
    DatabaseConnectionDiagnostic,
    DatabaseIntegrityDiagnostic,
    DatabaseTablesDiagnostic,
)
from diagnostics.health.directories import (
    DirectoryStructureDiagnostic,
    OutputDirectoryDiagnostic,
    RegistryFilesDiagnostic,
)


def get_diagnostics():
    """Return all health diagnostics."""
    return [
        PythonVersionDiagnostic(),
        DiskSpaceDiagnostic(),
        MemoryDiagnostic(),
        DependenciesDiagnostic(),
        DatabaseConnectionDiagnostic(),
        DatabaseIntegrityDiagnostic(),
        DatabaseTablesDiagnostic(),
        DirectoryStructureDiagnostic(),
        OutputDirectoryDiagnostic(),
        RegistryFilesDiagnostic(),
    ]


__all__ = [
    'get_diagnostics',
    'PythonVersionDiagnostic',
    'DiskSpaceDiagnostic',
    'MemoryDiagnostic',
    'DependenciesDiagnostic',
    'DatabaseConnectionDiagnostic',
    'DatabaseIntegrityDiagnostic',
    'DatabaseTablesDiagnostic',
    'DirectoryStructureDiagnostic',
    'OutputDirectoryDiagnostic',
    'RegistryFilesDiagnostic',
]
