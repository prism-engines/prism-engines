"""
Performance Diagnostics

Runtime performance checks for PRISM engines and data operations.
"""

from diagnostics.performance.engine_perf import (
    EngineLoadTimeDiagnostic,
    LensExecutionDiagnostic,
)
from diagnostics.performance.data_perf import (
    DataFetchDiagnostic,
    DatabaseQueryDiagnostic,
)


def get_diagnostics():
    """Return all performance diagnostics."""
    return [
        EngineLoadTimeDiagnostic(),
        LensExecutionDiagnostic(),
        DataFetchDiagnostic(),
        DatabaseQueryDiagnostic(),
    ]


__all__ = [
    'get_diagnostics',
    'EngineLoadTimeDiagnostic',
    'LensExecutionDiagnostic',
    'DataFetchDiagnostic',
    'DatabaseQueryDiagnostic',
]
