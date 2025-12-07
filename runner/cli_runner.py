"""
PRISM CLI Runner
=================

Interactive command-line interface for running PRISM analyses.
"""

import argparse
import logging
import sys
from typing import Optional, List, Dict, Any

from .executor import WorkflowExecutor
from .output_manager import OutputManager

logger = logging.getLogger(__name__)


class CLIRunner:
    """Interactive CLI for PRISM analysis."""

    def __init__(self):
        """Initialize the CLI runner."""
        self.executor = WorkflowExecutor()
        self.output = OutputManager()

    def run_interactive(self) -> int:
        """Run interactive menu mode."""
        self.output.print_banner("PRISM ENGINE")
        print("Modular Analysis System v2.0")
        print("=" * 60)

        try:
            # Step 1: Select Panel
            panel = self._select_panel()
            if panel is None:
                return 0

            # Step 2: Select Workflow
            workflow = self._select_workflow(panel)
            if workflow is None:
                return 0

            # Step 3: Configure Parameters
            parameters = self._configure_parameters(workflow)

            # Step 4: Confirm and Execute
            if not self._confirm_execution(panel, workflow, parameters):
                print("\nExecution cancelled.")
                return 0

            # Step 5: Execute
            print("\n" + "=" * 60)
            print("EXECUTING ANALYSIS...")
            print("=" * 60)

            results = self.executor.execute(panel, workflow, parameters)

            # Step 6: Display Results
            self._display_results(results)

            # Step 7: Save Results
            self._save_results(results)

            return 0 if results.get("status") == "completed" else 1

        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            return 130
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\nError: {e}")
            return 1

    def run_direct(
        self,
        panel: str,
        workflow: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Run in direct mode with specified arguments."""
        self.output.print_banner("PRISM ENGINE - Direct Mode")

        print(f"Panel: {panel}")
        print(f"Workflow: {workflow}")
        if parameters:
            print(f"Parameters: {parameters}")
        print("=" * 60)

        results = self.executor.execute(panel, workflow, parameters)
        self._display_results(results)
        self._save_results(results)

        return 0 if results.get("status") == "completed" else 1

    def _select_panel(self) -> Optional[str]:
        """Interactive panel selection."""
        panels = self.executor.list_panels()

        print("\nSELECT PANEL")
        print("-" * 40)
        for i, panel in enumerate(panels, 1):
            info = self.executor.get_panel_info(panel)
            icon = self._get_panel_icon(info) if info else "[*]"
            name = info.get("name", panel) if info else panel
            print(f"  {i}. {icon} {name} [{panel}]")
        print(f"  0. Exit")
        print("-" * 40)

        while True:
            try:
                choice = input("Enter selection (0-{}): ".format(len(panels)))
                if choice == "0" or choice.lower() == "exit":
                    return None

                idx = int(choice) - 1
                if 0 <= idx < len(panels):
                    selected = panels[idx]
                    print(f"\n[OK] Selected panel: {selected}")
                    return selected

                # Try direct panel name
                if choice in panels:
                    print(f"\n[OK] Selected panel: {choice}")
                    return choice

                print("Invalid selection. Try again.")
            except ValueError:
                if choice in panels:
                    print(f"\n[OK] Selected panel: {choice}")
                    return choice
                print("Please enter a number.")

    def _select_workflow(self, panel: str) -> Optional[str]:
        """Interactive workflow selection."""
        workflows = self.executor.list_workflows()

        print("\nSELECT WORKFLOW")
        print("-" * 40)
        for i, wf in enumerate(workflows, 1):
            info = self.executor.get_workflow_info(wf)
            name = info.get("name", wf) if info else wf
            desc = info.get("description", "")[:40] if info else ""
            runtime = info.get("estimated_runtime", "?") if info else "?"
            print(f"  {i}. {name}")
            if desc:
                print(f"      {desc}...")
            print(f"      Est. runtime: {runtime}")
        print(f"  0. Back")
        print("-" * 40)

        while True:
            try:
                choice = input("Enter selection (0-{}): ".format(len(workflows)))
                if choice == "0" or choice.lower() == "back":
                    return None

                idx = int(choice) - 1
                if 0 <= idx < len(workflows):
                    selected = workflows[idx]
                    print(f"\n[OK] Selected workflow: {selected}")
                    return selected

                if choice in workflows:
                    print(f"\n[OK] Selected workflow: {choice}")
                    return choice

                print("Invalid selection. Try again.")
            except ValueError:
                if choice in workflows:
                    print(f"\n[OK] Selected workflow: {choice}")
                    return choice
                print("Please enter a number.")

    def _configure_parameters(self, workflow: str) -> Dict[str, Any]:
        """Configure workflow parameters."""
        parameters = {}

        info = self.executor.get_workflow_info(workflow)
        if not info:
            return parameters

        param_defs = info.get("parameters", {})
        if not param_defs:
            return parameters

        print("\nCONFIGURE PARAMETERS")
        print("-" * 40)
        print("Press Enter to accept defaults")
        print("-" * 40)

        for param_name, param_info in param_defs.items():
            default = param_info.get("default", "")
            param_type = param_info.get("type", "string")
            description = param_info.get("description", param_name)

            prompt = f"{description}"
            if default:
                prompt += f" [{default}]"
            prompt += ": "

            value = input(prompt).strip()

            if not value and default:
                value = default

            if value:
                # Type conversion
                if param_type == "integer":
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif param_type == "float":
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                elif param_type == "boolean":
                    value = value.lower() in ("true", "yes", "1")
                elif param_type == "list":
                    value = [v.strip() for v in value.split(",")]

                parameters[param_name] = value

        return parameters

    def _confirm_execution(
        self,
        panel: str,
        workflow: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """Confirm execution with user."""
        print("\n" + "=" * 60)
        print("CONFIRM EXECUTION")
        print("=" * 60)
        print(f"  Panel:    {panel}")
        print(f"  Workflow: {workflow}")
        if parameters:
            print(f"  Parameters:")
            for k, v in parameters.items():
                print(f"    - {k}: {v}")
        print("=" * 60)

        response = input("\nProceed? [Y/n]: ").strip().lower()
        return response in ("", "y", "yes")

    def _display_results(self, results: Dict[str, Any]) -> None:
        """Display execution results."""
        status = results.get("status", "unknown")

        if status == "completed":
            print("\n" + "=" * 60)
            print("[OK] EXECUTION COMPLETED")
            print("=" * 60)
        elif status == "failed":
            print("\n" + "=" * 60)
            print("[FAILED] EXECUTION FAILED")
            print("=" * 60)
            if "error" in results:
                print(f"Error: {results['error']}")
            return

        # Summary
        print(self.output.format_summary(results))

        # Workflow-specific output
        workflow_type = results.get("workflow_type", "")

        if workflow_type == "regime_comparison":
            print(self.output.format_regime_comparison(results))

        if "rankings" in results and results["rankings"]:
            print(self.output.format_rankings(results["rankings"]))

        if "lenses_run" in results:
            lenses = results["lenses_run"]
            if isinstance(lenses, list):
                print(f"\nLenses executed: {len(lenses)}")
                for lens in lenses:
                    print(f"  * {lens}")

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to files."""
        try:
            # Create run directory
            run_dir = self.output.create_run_directory(
                prefix=results.get("workflow", "run")
            )

            # Save JSON results
            json_path = self.output.save_json(results, "results.json")

            # Save summary text
            summary = self.output.format_summary(results)
            if results.get("workflow_type") == "regime_comparison":
                summary += "\n" + self.output.format_regime_comparison(results)
            self.output.save_text(summary, "summary.txt")

            # Generate HTML report
            html_path = self.output.generate_html_report(results, open_browser=False)

            print(f"\n[SAVED] Results saved to: {run_dir}")
            if html_path:
                print(f"[HTML] Report: {html_path}")

        except Exception as e:
            logger.error(f"Could not save results: {e}")
            print(f"\n[WARNING] Could not save results: {e}")

    def _get_panel_icon(self, info: Dict[str, Any]) -> str:
        """Get text icon for panel."""
        if not info:
            return "[*]"
        icon_name = info.get("icon", "chart")
        icon_map = {
            "chart_increasing": "[M]",  # Market
            "bank": "[E]",  # Economy
            "globe": "[C]",  # Climate
            "gear": "[X]",  # Custom
        }
        return icon_map.get(icon_name, "[*]")


def run_cli() -> int:
    """Entry point for CLI runner."""
    parser = argparse.ArgumentParser(
        description="PRISM Engine - Modular Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prism_run.py                           # Interactive mode
  python prism_run.py --panel market            # Direct mode
  python prism_run.py --list-panels             # List panels
  python prism_run.py --list-workflows          # List workflows
        """
    )

    parser.add_argument(
        "--panel", "-p",
        help="Panel to analyze (market, economy, etc.)"
    )
    parser.add_argument(
        "--workflow", "-w",
        help="Workflow to run (regime_comparison, etc.)"
    )
    parser.add_argument(
        "--list-panels",
        action="store_true",
        help="List available panels"
    )
    parser.add_argument(
        "--list-workflows",
        action="store_true",
        help="List available workflows"
    )
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available engines"
    )
    parser.add_argument(
        "--comparison-periods",
        nargs="+",
        default=["2008", "2020"],
        help="Comparison periods for regime analysis"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s"
    )

    runner = CLIRunner()

    # List modes
    if args.list_panels:
        print("\nAvailable Panels:")
        print("-" * 40)
        for panel in runner.executor.list_panels():
            info = runner.executor.get_panel_info(panel)
            icon = runner._get_panel_icon(info) if info else "[*]"
            name = info.get("name", panel) if info else panel
            print(f"  {icon} {name} [{panel}]")
        return 0

    if args.list_workflows:
        print("\nAvailable Workflows:")
        print("-" * 40)
        for wf in runner.executor.list_workflows():
            info = runner.executor.get_workflow_info(wf)
            name = info.get("name", wf) if info else wf
            print(f"  * {name} [{wf}]")
        return 0

    if args.list_engines:
        print("\nAvailable Engines:")
        print("-" * 40)
        for engine in runner.executor.list_engines():
            print(f"  * {engine}")
        return 0

    # Direct mode
    if args.panel:
        workflow = args.workflow or "regime_comparison"
        parameters = {
            "comparison_periods": args.comparison_periods,
        }
        return runner.run_direct(args.panel, workflow, parameters)

    # Interactive mode
    return runner.run_interactive()


if __name__ == "__main__":
    sys.exit(run_cli())
