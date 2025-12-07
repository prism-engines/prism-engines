"""
Diagnostic Runner

Orchestrates the execution of diagnostic checks with support for:
- Parallel and sequential execution
- Timeout handling
- Progress reporting
- Result aggregation
"""

import concurrent.futures
import signal
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

from diagnostics.core.base import (
    BaseDiagnostic,
    DiagnosticCategory,
    DiagnosticResult,
    DiagnosticStatus,
)
from diagnostics.core.registry import get_registry, DiagnosticRegistry


@dataclass
class DiagnosticSummary:
    """Summary of diagnostic run results."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    errors: int = 0
    skipped: int = 0
    timeouts: int = 0
    total_duration_ms: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 100.0
        return ((self.passed + self.warnings + self.skipped) / self.total) * 100

    @property
    def all_passed(self) -> bool:
        """Check if all diagnostics passed (including warns/skips)."""
        return self.failed == 0 and self.errors == 0 and self.timeouts == 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'total': self.total,
            'passed': self.passed,
            'failed': self.failed,
            'warnings': self.warnings,
            'errors': self.errors,
            'skipped': self.skipped,
            'timeouts': self.timeouts,
            'total_duration_ms': round(self.total_duration_ms, 2),
            'success_rate': round(self.success_rate, 1),
            'all_passed': self.all_passed,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }


class DiagnosticRunner:
    """
    Orchestrates diagnostic execution.

    Supports running diagnostics with:
    - Category filtering
    - Quick mode (skip slow diagnostics)
    - Timeout handling
    - Progress callbacks
    - Parallel execution (optional)
    """

    def __init__(
        self,
        registry: DiagnosticRegistry = None,
        verbose: bool = True,
        parallel: bool = False,
        max_workers: int = 4,
    ):
        """
        Initialize the diagnostic runner.

        Args:
            registry: DiagnosticRegistry to use (default: global registry)
            verbose: Print progress to stdout
            parallel: Run diagnostics in parallel
            max_workers: Max parallel workers (if parallel=True)
        """
        self._registry = registry or get_registry()
        self._verbose = verbose
        self._parallel = parallel
        self._max_workers = max_workers
        self._results: List[DiagnosticResult] = []
        self._summary = DiagnosticSummary()
        self._progress_callback: Optional[Callable] = None
        self._completed_diagnostics: Set[str] = set()

    def set_progress_callback(self, callback: Callable[[str, DiagnosticResult], None]) -> None:
        """
        Set a callback for progress updates.

        Args:
            callback: Function(diagnostic_name, result) called after each diagnostic
        """
        self._progress_callback = callback

    def run_all(
        self,
        categories: List[DiagnosticCategory] = None,
        quick_mode: bool = False,
        names: List[str] = None,
    ) -> Dict:
        """
        Run diagnostics and return results.

        Args:
            categories: List of categories to run (None = all)
            quick_mode: Skip slow diagnostics
            names: Specific diagnostic names to run (overrides categories)

        Returns:
            Dictionary with 'results' list and 'summary' dict
        """
        self._results = []
        self._summary = DiagnosticSummary()
        self._summary.start_time = datetime.now()
        self._completed_diagnostics = set()

        # Get diagnostics to run
        diagnostics = self._get_diagnostics(categories, quick_mode, names)

        if not diagnostics:
            if self._verbose:
                print("No diagnostics to run.")
            return self._build_response()

        self._summary.total = len(diagnostics)

        if self._verbose:
            print(f"\n{'='*60}")
            print(f"PRISM Diagnostics Runner")
            print(f"{'='*60}")
            print(f"Running {len(diagnostics)} diagnostic(s)...")
            if quick_mode:
                print("(Quick mode enabled - skipping slow diagnostics)")
            print()

        # Run diagnostics
        if self._parallel:
            self._run_parallel(diagnostics)
        else:
            self._run_sequential(diagnostics)

        self._summary.end_time = datetime.now()
        self._summary.total_duration_ms = sum(r.duration_ms for r in self._results)

        if self._verbose:
            self._print_summary()

        return self._build_response()

    def run_single(self, name: str) -> Optional[DiagnosticResult]:
        """
        Run a single diagnostic by name.

        Args:
            name: Name of diagnostic to run

        Returns:
            DiagnosticResult or None if not found
        """
        diagnostic = self._registry.get(name)
        if diagnostic is None:
            return None

        result = self._execute_diagnostic(diagnostic)
        return result

    def _get_diagnostics(
        self,
        categories: List[DiagnosticCategory],
        quick_mode: bool,
        names: List[str],
    ) -> List[BaseDiagnostic]:
        """Get list of diagnostics to run based on filters."""
        if names:
            # Run specific diagnostics by name
            diagnostics = []
            for name in names:
                diag = self._registry.get(name)
                if diag:
                    diagnostics.append(diag)
            return diagnostics

        # Get all diagnostics
        all_diagnostics = list(self._registry.get_all())

        # Filter by category
        if categories:
            all_diagnostics = [d for d in all_diagnostics if d.category in categories]

        # Filter for quick mode
        if quick_mode:
            all_diagnostics = [d for d in all_diagnostics if d.is_quick]

        # Sort by category and name for consistent ordering
        all_diagnostics.sort(key=lambda d: (d.category.value, d.name))

        return all_diagnostics

    def _run_sequential(self, diagnostics: List[BaseDiagnostic]) -> None:
        """Run diagnostics sequentially."""
        for diagnostic in diagnostics:
            # Check dependencies
            if not self._check_dependencies(diagnostic):
                result = DiagnosticResult(
                    name=diagnostic.name,
                    status=DiagnosticStatus.SKIP,
                    message=f"Skipped: dependency not satisfied",
                    category=diagnostic.category,
                )
            else:
                result = self._execute_diagnostic(diagnostic)

            self._record_result(result)
            self._completed_diagnostics.add(diagnostic.name)

    def _run_parallel(self, diagnostics: List[BaseDiagnostic]) -> None:
        """Run diagnostics in parallel (where possible)."""
        # Group diagnostics by dependencies
        no_deps = [d for d in diagnostics if not d.dependencies]
        with_deps = [d for d in diagnostics if d.dependencies]

        # Run diagnostics without dependencies in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_diag = {
                executor.submit(self._execute_diagnostic, d): d
                for d in no_deps
            }
            for future in concurrent.futures.as_completed(future_to_diag):
                diag = future_to_diag[future]
                result = future.result()
                self._record_result(result)
                self._completed_diagnostics.add(diag.name)

        # Run diagnostics with dependencies sequentially
        for diagnostic in with_deps:
            if not self._check_dependencies(diagnostic):
                result = DiagnosticResult(
                    name=diagnostic.name,
                    status=DiagnosticStatus.SKIP,
                    message=f"Skipped: dependency not satisfied",
                    category=diagnostic.category,
                )
            else:
                result = self._execute_diagnostic(diagnostic)
            self._record_result(result)
            self._completed_diagnostics.add(diagnostic.name)

    def _execute_diagnostic(self, diagnostic: BaseDiagnostic) -> DiagnosticResult:
        """Execute a single diagnostic with timeout handling."""
        if self._verbose:
            print(f"  Running: {diagnostic.name}...", end=" ", flush=True)

        try:
            result = diagnostic.run()
        except Exception as e:
            result = DiagnosticResult(
                name=diagnostic.name,
                status=DiagnosticStatus.ERROR,
                message=f"Unexpected error: {str(e)}",
                category=diagnostic.category,
            )

        if self._verbose:
            status_str = result.status.symbol
            duration_str = f"({result.duration_ms:.0f}ms)"
            print(f"{status_str} {duration_str}")
            if result.status == DiagnosticStatus.FAIL and result.suggestions:
                for suggestion in result.suggestions:
                    print(f"    -> {suggestion}")

        if self._progress_callback:
            self._progress_callback(diagnostic.name, result)

        return result

    def _check_dependencies(self, diagnostic: BaseDiagnostic) -> bool:
        """Check if diagnostic dependencies are satisfied."""
        for dep_name in diagnostic.dependencies:
            if dep_name not in self._completed_diagnostics:
                return False
            # Check if dependency passed
            dep_result = next(
                (r for r in self._results if r.name == dep_name),
                None
            )
            if dep_result and not dep_result.status.is_success:
                return False
        return True

    def _record_result(self, result: DiagnosticResult) -> None:
        """Record a diagnostic result and update summary."""
        self._results.append(result)

        # Update summary counts
        status_map = {
            DiagnosticStatus.PASS: 'passed',
            DiagnosticStatus.FAIL: 'failed',
            DiagnosticStatus.WARN: 'warnings',
            DiagnosticStatus.ERROR: 'errors',
            DiagnosticStatus.SKIP: 'skipped',
            DiagnosticStatus.TIMEOUT: 'timeouts',
        }
        attr = status_map.get(result.status)
        if attr:
            setattr(self._summary, attr, getattr(self._summary, attr) + 1)

    def _build_response(self) -> Dict:
        """Build the final response dictionary."""
        return {
            'results': [r.to_dict() for r in self._results],
            'summary': self._summary.to_dict(),
            'by_category': self._group_by_category(),
        }

    def _group_by_category(self) -> Dict[str, List[Dict]]:
        """Group results by category."""
        grouped = {}
        for result in self._results:
            cat = result.category.value
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(result.to_dict())
        return grouped

    def _print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        s = self._summary
        print()
        print(f"{'='*60}")
        print("DIAGNOSTIC SUMMARY")
        print(f"{'='*60}")
        print(f"  Total:     {s.total}")
        print(f"  Passed:    {s.passed}")
        print(f"  Failed:    {s.failed}")
        print(f"  Warnings:  {s.warnings}")
        print(f"  Errors:    {s.errors}")
        print(f"  Skipped:   {s.skipped}")
        print(f"  Timeouts:  {s.timeouts}")
        print(f"{'='*60}")
        print(f"  Success Rate: {s.success_rate:.1f}%")
        print(f"  Total Time:   {s.total_duration_ms:.0f}ms")
        print(f"{'='*60}")

        if s.all_passed:
            print("\n  All diagnostics PASSED!")
        else:
            print("\n  Some diagnostics FAILED. Review the results above.")
            # Print failed diagnostics
            failed = [r for r in self._results if r.status in (
                DiagnosticStatus.FAIL, DiagnosticStatus.ERROR, DiagnosticStatus.TIMEOUT
            )]
            if failed:
                print("\n  Failed diagnostics:")
                for r in failed:
                    print(f"    - {r.name}: {r.message}")

        print()

    @property
    def results(self) -> List[DiagnosticResult]:
        """Get the list of results from the last run."""
        return self._results

    @property
    def summary(self) -> DiagnosticSummary:
        """Get the summary from the last run."""
        return self._summary
