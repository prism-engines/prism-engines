"""
System Health Diagnostics

Checks for Python version, disk space, memory, and dependencies.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List

from diagnostics.core.base import (
    BaseDiagnostic,
    DiagnosticCategory,
    DiagnosticResult,
)


class PythonVersionDiagnostic(BaseDiagnostic):
    """Check Python version meets minimum requirements."""

    name = "python_version"
    description = "Check Python version is compatible"
    category = DiagnosticCategory.SYSTEM
    is_quick = True

    MIN_VERSION = (3, 9)

    def check(self) -> DiagnosticResult:
        current = sys.version_info[:2]
        version_str = f"{current[0]}.{current[1]}"

        if current >= self.MIN_VERSION:
            return self.pass_result(
                f"Python {version_str} meets minimum requirement ({self.MIN_VERSION[0]}.{self.MIN_VERSION[1]}+)",
                details={
                    "current_version": version_str,
                    "minimum_version": f"{self.MIN_VERSION[0]}.{self.MIN_VERSION[1]}",
                    "full_version": sys.version,
                }
            )
        else:
            return self.fail_result(
                f"Python {version_str} is below minimum {self.MIN_VERSION[0]}.{self.MIN_VERSION[1]}",
                suggestions=[f"Upgrade to Python {self.MIN_VERSION[0]}.{self.MIN_VERSION[1]} or higher"]
            )


class DiskSpaceDiagnostic(BaseDiagnostic):
    """Check available disk space."""

    name = "disk_space"
    description = "Check available disk space for data and outputs"
    category = DiagnosticCategory.SYSTEM
    is_quick = True

    MIN_FREE_GB = 1.0  # Minimum 1GB free
    WARN_FREE_GB = 5.0  # Warn if less than 5GB

    def check(self) -> DiagnosticResult:
        # Check home directory partition
        home = Path.home()
        try:
            usage = shutil.disk_usage(home)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            used_percent = (usage.used / usage.total) * 100

            details = {
                "path": str(home),
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 1),
            }

            if free_gb < self.MIN_FREE_GB:
                return self.fail_result(
                    f"Critical: Only {free_gb:.1f}GB free disk space",
                    details=details,
                    suggestions=[
                        f"Free up disk space (minimum {self.MIN_FREE_GB}GB required)",
                        "Remove old output files from runs/ directory",
                    ]
                )
            elif free_gb < self.WARN_FREE_GB:
                return self.warn_result(
                    f"Low disk space: {free_gb:.1f}GB free",
                    details=details,
                    suggestions=["Consider freeing up disk space"]
                )
            else:
                return self.pass_result(
                    f"Disk space OK: {free_gb:.1f}GB free ({used_percent:.0f}% used)",
                    details=details
                )
        except Exception as e:
            return self.error_result(f"Could not check disk space: {e}")


class MemoryDiagnostic(BaseDiagnostic):
    """Check available system memory."""

    name = "memory"
    description = "Check available system memory"
    category = DiagnosticCategory.SYSTEM
    is_quick = True

    MIN_AVAILABLE_GB = 1.0
    WARN_AVAILABLE_GB = 2.0

    def check(self) -> DiagnosticResult:
        try:
            # Try psutil first (if available)
            try:
                import psutil
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024 ** 3)
                total_gb = mem.total / (1024 ** 3)
                used_percent = mem.percent

                details = {
                    "available_gb": round(available_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_percent": round(used_percent, 1),
                }

                if available_gb < self.MIN_AVAILABLE_GB:
                    return self.fail_result(
                        f"Critical: Only {available_gb:.1f}GB memory available",
                        details=details,
                        suggestions=["Close other applications to free memory"]
                    )
                elif available_gb < self.WARN_AVAILABLE_GB:
                    return self.warn_result(
                        f"Low memory: {available_gb:.1f}GB available",
                        details=details
                    )
                else:
                    return self.pass_result(
                        f"Memory OK: {available_gb:.1f}GB available ({used_percent:.0f}% used)",
                        details=details
                    )
            except ImportError:
                # Fallback: try reading /proc/meminfo on Linux
                if os.path.exists('/proc/meminfo'):
                    with open('/proc/meminfo') as f:
                        meminfo = {}
                        for line in f:
                            parts = line.split(':')
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip().split()[0]
                                meminfo[key] = int(value)

                    total_kb = meminfo.get('MemTotal', 0)
                    available_kb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
                    total_gb = total_kb / (1024 ** 2)
                    available_gb = available_kb / (1024 ** 2)

                    return self.pass_result(
                        f"Memory: {available_gb:.1f}GB available of {total_gb:.1f}GB total",
                        details={
                            "available_gb": round(available_gb, 2),
                            "total_gb": round(total_gb, 2),
                            "note": "psutil not installed, using /proc/meminfo"
                        }
                    )
                else:
                    return self.skip_result("Cannot check memory (psutil not installed)")

        except Exception as e:
            return self.error_result(f"Could not check memory: {e}")


class DependenciesDiagnostic(BaseDiagnostic):
    """Check required Python dependencies are installed."""

    name = "dependencies"
    description = "Check required Python packages are installed"
    category = DiagnosticCategory.SYSTEM
    is_quick = True

    # Core required packages
    REQUIRED_PACKAGES = [
        'numpy',
        'pandas',
        'scipy',
        'yaml',  # pyyaml
        'flask',
        'jinja2',
    ]

    # Optional but recommended
    OPTIONAL_PACKAGES = [
        'sklearn',  # scikit-learn
        'matplotlib',
        'requests',
        'psutil',
    ]

    def check(self) -> DiagnosticResult:
        missing_required = []
        missing_optional = []
        installed = []

        # Check required packages
        for pkg in self.REQUIRED_PACKAGES:
            if self._is_installed(pkg):
                installed.append(pkg)
            else:
                missing_required.append(pkg)

        # Check optional packages
        for pkg in self.OPTIONAL_PACKAGES:
            if self._is_installed(pkg):
                installed.append(pkg)
            else:
                missing_optional.append(pkg)

        details = {
            "installed": installed,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
        }

        if missing_required:
            return self.fail_result(
                f"Missing required packages: {', '.join(missing_required)}",
                details=details,
                suggestions=[
                    f"pip install {' '.join(missing_required)}",
                    "Run: pip install -r requirements.txt"
                ]
            )
        elif missing_optional:
            return self.warn_result(
                f"Missing optional packages: {', '.join(missing_optional)}",
                details=details,
                suggestions=[f"pip install {' '.join(missing_optional)}"]
            )
        else:
            return self.pass_result(
                f"All {len(installed)} dependencies installed",
                details=details
            )

    def _is_installed(self, package: str) -> bool:
        """Check if a package is installed."""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
