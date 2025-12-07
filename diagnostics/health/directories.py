"""
Directory Structure Diagnostics

Checks for required directories and file structure.
"""

import os
from pathlib import Path
from typing import Dict, List

from diagnostics.core.base import (
    BaseDiagnostic,
    DiagnosticCategory,
    DiagnosticResult,
)


def get_project_root() -> Path:
    """Get the PRISM project root directory."""
    # Try to find it relative to this file
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "engines").exists() and (parent / "workflows").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


class DirectoryStructureDiagnostic(BaseDiagnostic):
    """Check required project directories exist."""

    name = "directory_structure"
    description = "Verify required project directories exist"
    category = DiagnosticCategory.SYSTEM
    is_quick = True

    # Required directories relative to project root
    REQUIRED_DIRS = [
        'engines',
        'workflows',
        'panels',
        'runner',
        'data',
        'engine_core',
    ]

    # Optional but expected directories
    OPTIONAL_DIRS = [
        'runs',
        'templates',
        'validation',
        'tests',
    ]

    def check(self) -> DiagnosticResult:
        root = get_project_root()
        details = {"project_root": str(root)}

        missing_required = []
        missing_optional = []
        found_dirs = []

        # Check required directories
        for dir_name in self.REQUIRED_DIRS:
            dir_path = root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                found_dirs.append(dir_name)
            else:
                missing_required.append(dir_name)

        # Check optional directories
        for dir_name in self.OPTIONAL_DIRS:
            dir_path = root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                found_dirs.append(dir_name)
            else:
                missing_optional.append(dir_name)

        details["found_directories"] = found_dirs
        details["missing_required"] = missing_required
        details["missing_optional"] = missing_optional

        if missing_required:
            return self.fail_result(
                f"Missing required directories: {', '.join(missing_required)}",
                details=details,
                suggestions=[
                    "Ensure you're running from the correct directory",
                    "Check out the complete repository"
                ]
            )
        elif missing_optional:
            return self.warn_result(
                f"Missing optional directories: {', '.join(missing_optional)}",
                details=details
            )
        else:
            return self.pass_result(
                f"All {len(found_dirs)} directories present",
                details=details
            )


class OutputDirectoryDiagnostic(BaseDiagnostic):
    """Check output directory is writable."""

    name = "output_directory"
    description = "Check output directory exists and is writable"
    category = DiagnosticCategory.SYSTEM
    is_quick = True

    def check(self) -> DiagnosticResult:
        root = get_project_root()
        runs_dir = root / "runs"

        details = {"runs_dir": str(runs_dir)}

        # Check if runs directory exists
        if not runs_dir.exists():
            try:
                runs_dir.mkdir(parents=True, exist_ok=True)
                details["created"] = True
            except PermissionError:
                return self.fail_result(
                    "Cannot create runs/ directory",
                    details=details,
                    suggestions=["Check directory permissions"]
                )

        # Check if writable
        test_file = runs_dir / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            details["writable"] = True

            # Count existing runs
            run_count = len([d for d in runs_dir.iterdir() if d.is_dir()])
            details["existing_runs"] = run_count

            return self.pass_result(
                f"Output directory OK ({run_count} existing runs)",
                details=details
            )
        except PermissionError:
            return self.fail_result(
                "runs/ directory is not writable",
                details=details,
                suggestions=["Check file permissions: chmod u+w runs/"]
            )
        except Exception as e:
            return self.error_result(f"Could not verify output directory: {e}")


class RegistryFilesDiagnostic(BaseDiagnostic):
    """Check registry YAML files exist and are valid."""

    name = "registry_files"
    description = "Verify registry YAML files are present and valid"
    category = DiagnosticCategory.SYSTEM
    is_quick = True

    REGISTRY_FILES = [
        'engines/registry.yaml',
        'workflows/registry.yaml',
        'panels/registry.yaml',
    ]

    def check(self) -> DiagnosticResult:
        import yaml

        root = get_project_root()
        details = {"registries": {}}

        missing = []
        invalid = []
        valid = []

        for reg_path in self.REGISTRY_FILES:
            full_path = root / reg_path

            if not full_path.exists():
                missing.append(reg_path)
                continue

            try:
                with open(full_path) as f:
                    data = yaml.safe_load(f)

                if data is None:
                    invalid.append(f"{reg_path} (empty)")
                else:
                    valid.append(reg_path)
                    # Count entries
                    if isinstance(data, dict):
                        details["registries"][reg_path] = {
                            "entries": len(data),
                            "keys": list(data.keys())[:5],  # First 5 keys
                        }
                    elif isinstance(data, list):
                        details["registries"][reg_path] = {"entries": len(data)}

            except yaml.YAMLError as e:
                invalid.append(f"{reg_path} (parse error)")
            except Exception as e:
                invalid.append(f"{reg_path} ({e})")

        details["valid"] = valid
        details["missing"] = missing
        details["invalid"] = invalid

        if missing or invalid:
            problems = missing + invalid
            return self.fail_result(
                f"Registry issues: {', '.join(problems)}",
                details=details,
                suggestions=["Check out complete repository", "Validate YAML syntax"]
            )
        else:
            return self.pass_result(
                f"All {len(valid)} registry files valid",
                details=details
            )
