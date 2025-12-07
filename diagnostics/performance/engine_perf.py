"""
Engine Performance Diagnostics

Checks engine loading and execution performance.
"""

import time
from pathlib import Path
from typing import Dict, List

from diagnostics.core.base import (
    BaseDiagnostic,
    DiagnosticCategory,
    DiagnosticResult,
)


def get_project_root() -> Path:
    """Get the PRISM project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "engines").exists() and (parent / "workflows").exists():
            return parent
    return Path.cwd()


class EngineLoadTimeDiagnostic(BaseDiagnostic):
    """Check engine loader performance."""

    name = "engine_load_time"
    description = "Measure time to load engine registry"
    category = DiagnosticCategory.PERFORMANCE
    is_quick = True

    MAX_LOAD_TIME_MS = 2000  # 2 seconds max
    WARN_LOAD_TIME_MS = 500  # Warn over 500ms

    def check(self) -> DiagnosticResult:
        try:
            from engines.engine_loader import EngineLoader
        except ImportError:
            return self.skip_result("EngineLoader not available")

        start = time.time()
        try:
            root = get_project_root()
            loader = EngineLoader(root)
            engines = loader.list_engines()
            load_time_ms = (time.time() - start) * 1000

            details = {
                "load_time_ms": round(load_time_ms, 2),
                "engines_found": len(engines),
                "engine_names": engines[:10],  # First 10
            }

            if load_time_ms > self.MAX_LOAD_TIME_MS:
                return self.fail_result(
                    f"Engine loading too slow: {load_time_ms:.0f}ms",
                    details=details,
                    suggestions=["Check for import errors in engine modules"]
                )
            elif load_time_ms > self.WARN_LOAD_TIME_MS:
                return self.warn_result(
                    f"Engine loading slow: {load_time_ms:.0f}ms",
                    details=details
                )
            else:
                return self.pass_result(
                    f"Loaded {len(engines)} engines in {load_time_ms:.0f}ms",
                    details=details
                )

        except Exception as e:
            return self.error_result(f"Engine load failed: {e}")


class LensExecutionDiagnostic(BaseDiagnostic):
    """Test lens execution with sample data."""

    name = "lens_execution"
    description = "Test basic lens execution with sample data"
    category = DiagnosticCategory.PERFORMANCE
    is_quick = False  # Takes time to run
    timeout_seconds = 60

    def check(self) -> DiagnosticResult:
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            return self.skip_result("numpy/pandas not available")

        # Generate sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            index=dates,
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Try to load and run a simple lens
        lens_results = {}
        tested_lenses = []
        failed_lenses = []

        try:
            from engine_core.lenses.magnitude import MagnitudeLens

            start = time.time()
            lens = MagnitudeLens()
            result = lens.analyze(data)
            elapsed_ms = (time.time() - start) * 1000

            lens_results['magnitude'] = {
                'time_ms': round(elapsed_ms, 2),
                'success': True,
            }
            tested_lenses.append('magnitude')

        except Exception as e:
            lens_results['magnitude'] = {
                'error': str(e),
                'success': False,
            }
            failed_lenses.append('magnitude')

        # Try PCA lens
        try:
            from engine_core.lenses.pca import PCALens

            start = time.time()
            lens = PCALens()
            result = lens.analyze(data)
            elapsed_ms = (time.time() - start) * 1000

            lens_results['pca'] = {
                'time_ms': round(elapsed_ms, 2),
                'success': True,
            }
            tested_lenses.append('pca')

        except Exception as e:
            lens_results['pca'] = {
                'error': str(e),
                'success': False,
            }
            failed_lenses.append('pca')

        details = {
            "lens_results": lens_results,
            "tested": tested_lenses,
            "failed": failed_lenses,
            "sample_data_shape": list(data.shape),
        }

        if len(failed_lenses) == len(lens_results):
            return self.fail_result(
                "All tested lenses failed",
                details=details,
                suggestions=["Check lens implementations", "Verify dependencies"]
            )
        elif failed_lenses:
            return self.warn_result(
                f"{len(failed_lenses)}/{len(lens_results)} lenses failed",
                details=details
            )
        else:
            avg_time = sum(
                r.get('time_ms', 0) for r in lens_results.values() if r.get('success')
            ) / len(tested_lenses)
            return self.pass_result(
                f"Tested {len(tested_lenses)} lenses (avg {avg_time:.0f}ms)",
                details=details
            )
