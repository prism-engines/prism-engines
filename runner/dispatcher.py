"""
PRISM Dispatcher
=================

Routes UI requests to the appropriate workflow execution.

The dispatcher:
1. Receives parameters from the UI
2. Validates parameters
3. Loads the appropriate workflow
4. Executes the workflow
5. Returns results for display

Usage:
    from runner.dispatcher import Dispatcher

    dispatcher = Dispatcher()
    results = dispatcher.run({
        "panel": "market",
        "workflow": "regime_comparison",
        "weighting": "hybrid",
    })
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class DispatchError(Exception):
    """Raised when dispatch fails."""
    pass


class Dispatcher:
    """
    Routes analysis requests to workflow execution.

    Acts as the bridge between the UI and the engine/workflow system.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the dispatcher.

        Args:
            output_dir: Base directory for outputs
        """
        self.output_dir = output_dir or PROJECT_ROOT / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track current run
        self.current_run_dir: Optional[Path] = None
        self.run_start_time: Optional[datetime] = None

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an analysis based on parameters.

        Args:
            params: Dictionary with:
                - panel: Panel domain (market, economy, etc.)
                - workflow: Workflow key (regime_comparison, etc.)
                - weighting: Weighting method (meta, ml, hybrid, none)
                - timeframe: Timeframe option (default, last_run, custom)
                - start_date: Optional start date
                - end_date: Optional end date
                - comparison_periods: Optional list of periods to compare

        Returns:
            Results dictionary with:
                - status: success/error
                - report_path: Path to generated report
                - output_dir: Path to output directory
                - duration: Execution time in seconds
                - Additional workflow-specific results
        """
        self.run_start_time = datetime.now()

        # Create output directory for this run
        timestamp = self.run_start_time.strftime("%Y-%m-%d_%H%M%S")
        self.current_run_dir = self.output_dir / timestamp
        self.current_run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting dispatch: {params.get('workflow')} on {params.get('panel')}")
        logger.info(f"Output directory: {self.current_run_dir}")

        try:
            # Extract parameters
            workflow_key = params.get("workflow", "regime_comparison")
            panel = params.get("panel", "market")
            weighting = params.get("weighting", "hybrid")

            # Route to appropriate handler
            if workflow_key == "regime_comparison":
                results = self._run_regime_comparison(params)
            elif workflow_key == "full_temporal":
                results = self._run_full_temporal(params)
            elif workflow_key == "daily_update":
                results = self._run_daily_update(params)
            elif workflow_key == "overnight_batch":
                results = self._run_overnight_batch(params)
            else:
                # Try to load from workflow registry
                results = self._run_workflow_from_registry(workflow_key, params)

            # Add common result fields
            duration = (datetime.now() - self.run_start_time).total_seconds()
            results.update({
                "status": "success",
                "output_dir": str(self.current_run_dir),
                "duration": duration,
                "timestamp": timestamp,
            })

            logger.info(f"Dispatch complete in {duration:.1f}s")
            return results

        except Exception as e:
            logger.exception(f"Dispatch failed: {e}")
            duration = (datetime.now() - self.run_start_time).total_seconds()
            return {
                "status": "error",
                "error": str(e),
                "output_dir": str(self.current_run_dir),
                "duration": duration,
            }

    # -------------------------------------------------------------------------
    # WORKFLOW HANDLERS
    # -------------------------------------------------------------------------

    def _run_regime_comparison(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the regime comparison workflow."""
        logger.info("Running regime comparison workflow...")

        try:
            from engine_core.tandem_regime_runner import TandemRegimeRunner

            # Extract parameters
            comparison_periods = params.get("comparison_periods") or [
                "2001_dotcom", "2008_gfc", "2020_covid", "2022_bear"
            ]

            panel = params.get("panel", "market")
            data_source = "database" if panel != "custom" else "csv"

            runner = TandemRegimeRunner(
                current_period="2024_current",
                comparison_periods=comparison_periods,
                data_source=data_source,
                output_dir=self.current_run_dir,
            )

            results = runner.run()

            return {
                "report_path": results.get("report_path"),
                "comparisons": results.get("comparisons", {}),
                "fingerprints": results.get("fingerprints", {}),
            }

        except ImportError as e:
            logger.warning(f"TandemRegimeRunner not available: {e}")
            return self._create_placeholder_report("regime_comparison", params)

    def _run_full_temporal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full temporal analysis workflow."""
        logger.info("Running full temporal analysis workflow...")

        try:
            from engine_core.orchestration.temporal_runner import TemporalAnalysisRunner

            # Extract parameters
            start_year = int(params.get("start_year", 1970))
            end_year = int(params.get("end_year", 2025))
            window_years = int(params.get("window_years", 2))

            # runner = TemporalAnalysisRunner(...)
            # results = runner.run()

            # Placeholder until fully implemented
            return self._create_placeholder_report("full_temporal", params)

        except ImportError as e:
            logger.warning(f"TemporalAnalysisRunner not available: {e}")
            return self._create_placeholder_report("full_temporal", params)

    def _run_daily_update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the daily update workflow."""
        logger.info("Running daily update workflow...")

        # Quick analysis for daily monitoring
        return self._create_placeholder_report("daily_update", params)

    def _run_overnight_batch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the overnight batch workflow."""
        logger.info("Running overnight batch workflow...")

        return self._create_placeholder_report("overnight_batch", params)

    def _run_workflow_from_registry(
        self,
        workflow_key: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a workflow loaded from the registry."""
        logger.info(f"Running workflow from registry: {workflow_key}")

        try:
            from workflows.workflow_loader import WorkflowLoader

            loader = WorkflowLoader()
            results = loader.run_workflow(
                workflow_key,
                panel=params.get("panel"),
                output_dir=self.current_run_dir,
                **{k: v for k, v in params.items() if k not in ["panel", "workflow"]}
            )

            return results

        except Exception as e:
            logger.warning(f"Could not run workflow {workflow_key}: {e}")
            return self._create_placeholder_report(workflow_key, params)

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _create_placeholder_report(
        self,
        workflow: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a placeholder report for workflows not yet implemented."""

        report_path = self.current_run_dir / f"{workflow}_report.html"

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM Report - {workflow}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            margin: 0;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}
        h1 {{ color: var(--primary); }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: #dbeafe;
            color: #1d4ed8;
            border-radius: 9999px;
            font-size: 0.875rem;
        }}
        .params {{
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🔍 PRISM Analysis Report</h1>
            <p><span class="badge">{workflow}</span></p>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="card">
            <h2>📋 Configuration</h2>
            <div class="params">
                <p><strong>Workflow:</strong> {workflow}</p>
                <p><strong>Panel:</strong> {params.get('panel', 'market')}</p>
                <p><strong>Weighting:</strong> {params.get('weighting', 'hybrid')}</p>
            </div>
        </div>

        <div class="card">
            <h2>⚠️ Placeholder Report</h2>
            <p>This is a placeholder report. The full implementation will include:</p>
            <ul>
                <li>Detailed analysis results</li>
                <li>Interactive charts</li>
                <li>Indicator rankings</li>
                <li>Regime comparison (if applicable)</li>
            </ul>
        </div>

        <div class="card">
            <p style="text-align: center; color: #64748b;">
                PRISM Engine - Panel-Based Regime Analysis
            </p>
        </div>
    </div>
</body>
</html>
        """

        with open(report_path, "w") as f:
            f.write(html_content)

        return {
            "report_path": str(report_path),
            "placeholder": True,
        }

    def get_recent_runs(self, limit: int = 10) -> list:
        """Get list of recent run directories."""
        runs = []

        if not self.output_dir.exists():
            return runs

        for run_dir in sorted(self.output_dir.iterdir(), reverse=True)[:limit]:
            if run_dir.is_dir():
                # Find the report file
                reports = list(run_dir.glob("*.html"))
                runs.append({
                    "timestamp": run_dir.name,
                    "path": str(run_dir),
                    "report": str(reports[0]) if reports else None,
                })

        return runs
