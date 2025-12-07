#!/usr/bin/env python3
"""
PRISM Diagnostics Runner
========================

Run system diagnostics to check PRISM health.

Usage:
    python -m diagnostics.run_diagnostics              # Run all diagnostics
    python -m diagnostics.run_diagnostics --quick      # Quick checks only
    python -m diagnostics.run_diagnostics --category health  # Specific category
    python -m diagnostics.run_diagnostics --json       # JSON output
    python -m diagnostics.run_diagnostics --html       # Generate HTML report
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.core.runner import DiagnosticRunner
from diagnostics.core.registry import get_registry
from diagnostics.core.report import DiagnosticReporter
from diagnostics.core.base import DiagnosticCategory


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Diagnostics Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories:
  health       - System and database health checks
  performance  - Performance benchmarks
  validation   - Plugin validation checks

Examples:
  python -m diagnostics.run_diagnostics                    # Run all
  python -m diagnostics.run_diagnostics --quick            # Quick mode
  python -m diagnostics.run_diagnostics -c health          # Health only
  python -m diagnostics.run_diagnostics --json             # JSON output
  python -m diagnostics.run_diagnostics --html report.html # HTML report
        """
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick diagnostics only (skip slow checks)"
    )
    parser.add_argument(
        "--category", "-c",
        choices=['health', 'performance', 'validation', 'all'],
        default='all',
        help="Category of diagnostics to run"
    )
    parser.add_argument(
        "--name", "-n",
        action="append",
        help="Run specific diagnostic(s) by name (can specify multiple)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available diagnostics"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--html",
        type=str,
        metavar="FILE",
        help="Generate HTML report to specified file"
    )
    parser.add_argument(
        "--markdown", "--md",
        type=str,
        metavar="FILE",
        help="Generate Markdown report to specified file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (only show errors)"
    )

    args = parser.parse_args()

    # Initialize registry
    registry = get_registry()

    # List mode
    if args.list:
        print("\nAvailable Diagnostics:")
        print("=" * 60)

        # Group by category
        by_category = {}
        for diag in registry.get_all():
            cat = diag.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(diag)

        for cat, diagnostics in sorted(by_category.items()):
            print(f"\n[{cat.upper()}]")
            for diag in diagnostics:
                quick_marker = "" if diag.is_quick else " (slow)"
                print(f"  - {diag.name}: {diag.description}{quick_marker}")

        print()
        return 0

    # Determine categories to run
    categories = None
    if args.category != 'all':
        cat_map = {
            'health': [DiagnosticCategory.HEALTH, DiagnosticCategory.SYSTEM],
            'performance': [DiagnosticCategory.PERFORMANCE],
            'validation': [DiagnosticCategory.VALIDATION],
        }
        categories = cat_map.get(args.category, None)

    # Run diagnostics
    verbose = not args.quiet and (args.verbose or not args.json)
    runner = DiagnosticRunner(verbose=verbose)

    if args.name:
        # Run specific diagnostics
        results = runner.run_all(names=args.name, quick_mode=args.quick)
    else:
        results = runner.run_all(categories=categories, quick_mode=args.quick)

    # Output handling
    reporter = DiagnosticReporter(results)

    if args.json:
        print(reporter.to_json())
    elif args.html:
        path = reporter.save(args.html, format='html')
        print(f"\nHTML report saved to: {path}")
    elif args.markdown:
        path = reporter.save(args.markdown, format='md')
        print(f"\nMarkdown report saved to: {path}")

    # Generate HTML report by default to runs directory
    if not args.json and not args.html and not args.quiet:
        runs_dir = PROJECT_ROOT / "runs" / "diagnostics"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = runs_dir / f"diagnostics_{timestamp}.html"
        reporter.save(str(html_path), format='html')
        print(f"\nReport saved: {html_path}")

    # Return exit code based on results
    summary = results.get('summary', {})
    if summary.get('all_passed', False):
        return 0
    elif summary.get('failed', 0) > 0 or summary.get('errors', 0) > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
