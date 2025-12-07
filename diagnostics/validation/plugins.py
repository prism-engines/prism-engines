"""
Plugin Validation Diagnostics

Validates engines, workflows, and panels meet requirements.
"""

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


class EngineValidationDiagnostic(BaseDiagnostic):
    """Validate all registered engines."""

    name = "engine_validation"
    description = "Validate engine plugins meet requirements"
    category = DiagnosticCategory.VALIDATION
    is_quick = True

    def check(self) -> DiagnosticResult:
        try:
            from core.validator import PluginValidator
            from core.plugin_loader import PluginLoader
        except ImportError:
            return self.skip_result("Plugin system not available")

        root = get_project_root()

        try:
            loader = PluginLoader(root)
            loader.discover_all()

            results = []
            for name, engine_class in loader._engines.items():
                result = PluginValidator.validate_engine(engine_class)
                results.append(result)

            valid_count = sum(1 for r in results if r['valid'])
            invalid_count = len(results) - valid_count

            # Collect errors and warnings
            all_errors = []
            all_warnings = []
            for r in results:
                for err in r.get('errors', []):
                    all_errors.append(f"{r['name']}: {err}")
                for warn in r.get('warnings', []):
                    all_warnings.append(f"{r['name']}: {warn}")

            details = {
                "total_engines": len(results),
                "valid": valid_count,
                "invalid": invalid_count,
                "errors": all_errors[:10],  # First 10
                "warnings": all_warnings[:10],
            }

            if invalid_count > 0:
                return self.fail_result(
                    f"{invalid_count}/{len(results)} engines have validation errors",
                    details=details,
                    suggestions=["Fix engine implementations", "Check required attributes"]
                )
            elif all_warnings:
                return self.warn_result(
                    f"All {valid_count} engines valid, but {len(all_warnings)} warnings",
                    details=details
                )
            else:
                return self.pass_result(
                    f"All {valid_count} engines validated successfully",
                    details=details
                )

        except Exception as e:
            return self.error_result(f"Engine validation failed: {e}")


class WorkflowValidationDiagnostic(BaseDiagnostic):
    """Validate all registered workflows."""

    name = "workflow_validation"
    description = "Validate workflow plugins meet requirements"
    category = DiagnosticCategory.VALIDATION
    is_quick = True

    def check(self) -> DiagnosticResult:
        try:
            from core.validator import PluginValidator
            from core.plugin_loader import PluginLoader
        except ImportError:
            return self.skip_result("Plugin system not available")

        root = get_project_root()

        try:
            loader = PluginLoader(root)
            loader.discover_all()

            results = []
            for name, workflow_class in loader._workflows.items():
                result = PluginValidator.validate_workflow(workflow_class)
                results.append(result)

            valid_count = sum(1 for r in results if r['valid'])
            invalid_count = len(results) - valid_count

            # Collect errors
            all_errors = []
            for r in results:
                for err in r.get('errors', []):
                    all_errors.append(f"{r['name']}: {err}")

            details = {
                "total_workflows": len(results),
                "valid": valid_count,
                "invalid": invalid_count,
                "errors": all_errors[:10],
            }

            if invalid_count > 0:
                return self.fail_result(
                    f"{invalid_count}/{len(results)} workflows have validation errors",
                    details=details,
                    suggestions=["Fix workflow implementations"]
                )
            else:
                return self.pass_result(
                    f"All {valid_count} workflows validated successfully",
                    details=details
                )

        except Exception as e:
            return self.error_result(f"Workflow validation failed: {e}")


class PanelValidationDiagnostic(BaseDiagnostic):
    """Validate all panel configurations."""

    name = "panel_validation"
    description = "Validate panel configurations"
    category = DiagnosticCategory.VALIDATION
    is_quick = True

    def check(self) -> DiagnosticResult:
        try:
            from core.validator import PluginValidator
        except ImportError:
            return self.skip_result("Plugin system not available")

        root = get_project_root()
        panels_dir = root / "panels"

        if not panels_dir.exists():
            return self.fail_result(
                "Panels directory not found",
                suggestions=["Create panels/ directory"]
            )

        try:
            results = []
            for panel_dir in panels_dir.iterdir():
                if panel_dir.is_dir() and not panel_dir.name.startswith(('_', '.')):
                    result = PluginValidator.validate_panel(panel_dir)
                    results.append(result)

            valid_count = sum(1 for r in results if r['valid'])
            invalid_count = len(results) - valid_count

            # Collect errors
            all_errors = []
            for r in results:
                for err in r.get('errors', []):
                    all_errors.append(f"{r['name']}: {err}")

            details = {
                "total_panels": len(results),
                "valid": valid_count,
                "invalid": invalid_count,
                "errors": all_errors[:10],
                "panel_names": [r['name'] for r in results],
            }

            if invalid_count > 0:
                return self.fail_result(
                    f"{invalid_count}/{len(results)} panels have validation errors",
                    details=details,
                    suggestions=["Fix panel registry.yaml files"]
                )
            else:
                return self.pass_result(
                    f"All {valid_count} panels validated successfully",
                    details=details
                )

        except Exception as e:
            return self.error_result(f"Panel validation failed: {e}")
