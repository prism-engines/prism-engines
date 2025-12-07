"""
PRISM Web Server
=================

Flask-based web interface for running PRISM analyses.

Usage:
    python prism_run.py --web
    # Then open http://localhost:5000
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .executor import WorkflowExecutor
from .output_manager import OutputManager

logger = logging.getLogger(__name__)

# Project root for templates
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
OUTPUT_DIR = PROJECT_ROOT / "output"


def create_app() -> 'Flask':
    """Create and configure the Flask application."""
    if not FLASK_AVAILABLE:
        raise ImportError("Flask is required. Install with: pip install flask")

    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(TEMPLATES_DIR / "static"),
    )

    # Initialize components
    executor = WorkflowExecutor()
    output_manager = OutputManager()

    def get_panels_list() -> List[Dict[str, Any]]:
        """Get panels as list of dicts for template."""
        panels = executor.list_panels()
        result = []

        # Panel icons mapping
        icons = {
            "market": "📊",
            "economy": "🏦",
            "sentiment": "💬",
            "crypto": "₿",
            "rates": "📈",
            "fx": "💱",
        }

        for panel_id in panels:
            try:
                info = executor.get_panel_info(panel_id)
                result.append({
                    "key": panel_id,
                    "name": info.get("name", panel_id.title()),
                    "description": info.get("description", f"Analysis panel for {panel_id}"),
                    "icon": icons.get(panel_id, "📊"),
                    "indicator_count": info.get("indicator_count", len(info.get("indicators", []))),
                })
            except Exception:
                result.append({
                    "key": panel_id,
                    "name": panel_id.title(),
                    "description": f"Analysis panel for {panel_id}",
                    "icon": icons.get(panel_id, "📊"),
                })

        return result

    def get_workflows_list() -> List[Dict[str, Any]]:
        """Get workflows as list of dicts for template."""
        workflows = executor.list_workflows()
        result = []

        for wf_id in workflows:
            try:
                info = executor.get_workflow_info(wf_id)
                result.append({
                    "key": wf_id,
                    "name": info.get("name", wf_id.replace("_", " ").title()),
                    "description": info.get("description", ""),
                    "runtime": info.get("estimated_runtime", "2-5 minutes"),
                })
            except Exception:
                result.append({
                    "key": wf_id,
                    "name": wf_id.replace("_", " ").title(),
                    "description": "",
                    "runtime": "2-5 minutes",
                })

        return result

    def get_weighting_methods() -> List[Dict[str, Any]]:
        """Get available weighting methods."""
        return [
            {
                "key": "equal",
                "name": "Equal Weight",
                "description": "All indicators weighted equally",
            },
            {
                "key": "hybrid",
                "name": "Hybrid",
                "description": "Blend of regime-specific and universal weights",
            },
            {
                "key": "market_cap",
                "name": "Market Cap",
                "description": "Weighted by market capitalization",
            },
            {
                "key": "volatility",
                "name": "Volatility",
                "description": "Inverse volatility weighting",
            },
        ]

    @app.route("/")
    def index():
        """Main page with panel/workflow selection."""
        return render_template(
            "runner/index.html",
            panels=get_panels_list(),
            workflows=get_workflows_list(),
            weighting_methods=get_weighting_methods(),
        )

    @app.route("/api/panels")
    def api_panels():
        """API endpoint for panel list."""
        panels = executor.list_panels()
        result = {}
        for panel_id in panels:
            try:
                info = executor.get_panel_info(panel_id)
                result[panel_id] = info
            except Exception:
                result[panel_id] = {"name": panel_id}
        return jsonify(result)

    @app.route("/api/workflows")
    def api_workflows():
        """API endpoint for workflow list."""
        workflows = executor.list_workflows()
        result = {}
        for wf_id in workflows:
            try:
                info = executor.get_workflow_info(wf_id)
                result[wf_id] = info
            except Exception:
                result[wf_id] = {"name": wf_id}
        return jsonify(result)

    @app.route("/api/panel/<panel_id>")
    def api_panel_info(panel_id: str):
        """API endpoint for panel details."""
        try:
            info = executor.get_panel_info(panel_id)
            return jsonify(info)
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    @app.route("/api/workflow/<workflow_id>")
    def api_workflow_info(workflow_id: str):
        """API endpoint for workflow details."""
        try:
            info = executor.get_workflow_info(workflow_id)
            return jsonify(info)
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    @app.route("/run", methods=["POST"])
    def run_analysis():
        """Execute analysis and show results."""
        try:
            # Get form data
            panel_id = request.form.get("panel", "market")
            workflow_id = request.form.get("workflow", "lens_validation")
            weighting = request.form.get("weighting", "hybrid")

            # Get optional parameters
            params = {"weighting": weighting}

            # Comparison periods for regime analysis
            comparison_periods = request.form.getlist("comparison_periods")
            if comparison_periods:
                params["comparison_periods"] = comparison_periods

            # Timeframe options
            timeframe = request.form.get("timeframe", "default")
            if timeframe == "custom":
                params["start_date"] = request.form.get("start_date")
                params["end_date"] = request.form.get("end_date")

            logger.info(f"Running: panel={panel_id}, workflow={workflow_id}, params={params}")

            # Execute workflow
            results = executor.execute(panel_id, workflow_id, params)

            # Create output directory and save results
            run_dir = output_manager.create_run_directory(prefix=workflow_id)
            output_manager.save_json(results, "results.json")

            # Generate summary
            summary = output_manager.format_summary(results)
            if results.get("workflow_type") == "regime_comparison":
                summary += "\n" + output_manager.format_regime_comparison(results)
            output_manager.save_text(summary, "summary.txt")

            # Generate HTML report
            html_path = output_manager.generate_html_report(results)

            # Add paths to results for template
            results["output_dir"] = str(run_dir)
            results["report_path"] = str(html_path) if html_path else None

            # Format comparisons for template if we have similarities
            if "similarities" in results:
                results["comparisons"] = {
                    period: {"overall_similarity": score}
                    for period, score in results["similarities"].items()
                }

            # Request params for display
            request_params = {
                "panel": panel_id,
                "workflow": workflow_id,
                "weighting": weighting,
                "comparison_periods": comparison_periods,
            }

            return render_template(
                "runner/results.html",
                success=results.get("status") == "completed",
                results=results,
                params=request_params,
                error=results.get("error"),
            )

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            return render_template(
                "runner/error.html",
                error=str(e),
                panel=request.form.get("panel"),
                workflow=request.form.get("workflow"),
            )

    @app.route("/api/run", methods=["POST"])
    def api_run():
        """API endpoint for running analysis (returns JSON)."""
        try:
            data = request.get_json() or {}
            panel_id = data.get("panel", "market")
            workflow_id = data.get("workflow", "lens_validation")
            params = data.get("params", {})

            results = executor.execute(panel_id, workflow_id, params)

            # Save results
            run_dir = output_manager.create_run_directory(prefix=workflow_id)
            output_manager.save_json(results, "results.json")
            html_path = output_manager.generate_html_report(results)

            results["output_dir"] = str(run_dir)
            results["report_path"] = str(html_path) if html_path else None

            return jsonify(results)

        except Exception as e:
            logger.exception(f"API run failed: {e}")
            return jsonify({"error": str(e), "status": "failed"}), 500

    @app.route("/reports")
    def list_reports():
        """List available reports."""
        reports = []

        if OUTPUT_DIR.exists():
            for run_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
                if run_dir.is_dir():
                    report_file = run_dir / "report.html"
                    results_file = run_dir / "results.json"

                    report_info = {
                        "name": run_dir.name,
                        "path": str(run_dir),
                        "has_report": report_file.exists(),
                        "has_results": results_file.exists(),
                    }

                    # Try to get metadata from results
                    if results_file.exists():
                        try:
                            with open(results_file) as f:
                                data = json.load(f)
                                report_info["panel"] = data.get("panel", "unknown")
                                report_info["workflow"] = data.get("workflow", "unknown")
                                report_info["timestamp"] = data.get("timestamp", "")
                                report_info["status"] = data.get("status", "unknown")
                        except Exception:
                            pass

                    reports.append(report_info)

        return render_template("runner/reports.html", reports=reports)

    @app.route("/output/<path:filepath>")
    def serve_output(filepath):
        """Serve files from output directory."""
        return send_from_directory(str(OUTPUT_DIR), filepath)

    @app.route("/health")
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "panels": len(executor.list_panels()),
            "workflows": len(executor.list_workflows()),
        })

    return app


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> int:
    """Run the Flask development server."""
    if not FLASK_AVAILABLE:
        print("ERROR: Flask is required. Install with: pip install flask")
        return 1

    app = create_app()

    print(f"""
+==============================================================+
|                    PRISM WEB SERVER                          |
+==============================================================+
|  URL: http://localhost:{port}                                  |
|  Press Ctrl+C to stop                                        |
+==============================================================+
    """)

    app.run(host=host, port=port, debug=debug)
    return 0


if __name__ == "__main__":
    run_server(debug=True)
