"""
Configuration Validation Diagnostics

Validates PRISM configuration files and settings.
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
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "engines").exists() and (parent / "workflows").exists():
            return parent
    return Path.cwd()


class ConfigurationDiagnostic(BaseDiagnostic):
    """Validate PRISM configuration."""

    name = "configuration"
    description = "Validate PRISM configuration files"
    category = DiagnosticCategory.VALIDATION
    is_quick = True

    def check(self) -> DiagnosticResult:
        import yaml

        root = get_project_root()
        issues = []
        validated = []

        # Check engines/registry.yaml
        engine_reg = root / "engines" / "registry.yaml"
        if engine_reg.exists():
            try:
                with open(engine_reg) as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    validated.append("engines/registry.yaml")
                    # Check each engine has required fields
                    for name, config in data.items():
                        if not isinstance(config, dict):
                            issues.append(f"Engine '{name}': invalid format")
                        elif 'module' not in config:
                            issues.append(f"Engine '{name}': missing 'module'")
                else:
                    issues.append("engines/registry.yaml: empty or invalid")
            except yaml.YAMLError as e:
                issues.append(f"engines/registry.yaml: YAML parse error")
        else:
            issues.append("engines/registry.yaml: not found")

        # Check workflows/registry.yaml
        workflow_reg = root / "workflows" / "registry.yaml"
        if workflow_reg.exists():
            try:
                with open(workflow_reg) as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    validated.append("workflows/registry.yaml")
                    # Check each workflow has required fields
                    for name, config in data.items():
                        if not isinstance(config, dict):
                            issues.append(f"Workflow '{name}': invalid format")
                        elif 'module' not in config:
                            issues.append(f"Workflow '{name}': missing 'module'")
                else:
                    issues.append("workflows/registry.yaml: empty or invalid")
            except yaml.YAMLError as e:
                issues.append(f"workflows/registry.yaml: YAML parse error")
        else:
            issues.append("workflows/registry.yaml: not found")

        # Check panels/registry.yaml
        panel_reg = root / "panels" / "registry.yaml"
        if panel_reg.exists():
            try:
                with open(panel_reg) as f:
                    data = yaml.safe_load(f)
                if data:
                    validated.append("panels/registry.yaml")
                else:
                    issues.append("panels/registry.yaml: empty")
            except yaml.YAMLError as e:
                issues.append(f"panels/registry.yaml: YAML parse error")
        else:
            issues.append("panels/registry.yaml: not found")

        # Check for optional config files
        optional_configs = ['config.yaml', 'settings.yaml', '.env']
        found_optional = []
        for cfg in optional_configs:
            cfg_path = root / cfg
            if cfg_path.exists():
                found_optional.append(cfg)

        details = {
            "validated": validated,
            "issues": issues[:15],  # First 15 issues
            "optional_configs_found": found_optional,
        }

        critical_issues = [i for i in issues if 'not found' in i or 'parse error' in i]

        if critical_issues:
            return self.fail_result(
                f"Configuration errors: {len(critical_issues)} critical issues",
                details=details,
                suggestions=[
                    "Check YAML syntax in registry files",
                    "Ensure all registry files exist"
                ]
            )
        elif issues:
            return self.warn_result(
                f"Configuration warnings: {len(issues)} issues",
                details=details
            )
        else:
            return self.pass_result(
                f"Configuration OK ({len(validated)} files validated)",
                details=details
            )
