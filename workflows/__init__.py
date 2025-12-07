"""
PRISM Workflows Package
========================

Auto-discoverable analysis workflows loaded from registry.yaml.

Usage:
    from workflows import WorkflowLoader, list_workflows, run_workflow

    # List all workflows
    workflows = list_workflows()

    # Run a workflow
    results = run_workflow("regime_comparison", panel="market")

    # Run a preset
    results = run_preset("quick_check")
"""

from .workflow_loader import (
    WorkflowLoader,
    WorkflowLoadError,
    get_loader,
    list_workflows,
    run_workflow,
    run_preset,
)

__all__ = [
    "WorkflowLoader",
    "WorkflowLoadError",
    "get_loader",
    "list_workflows",
    "run_workflow",
    "run_preset",
]
