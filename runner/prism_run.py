#!/usr/bin/env python3
"""
PRISM HTML Runner
==================

Launches a local web interface for PRISM analysis.

Usage:
    python prism_run.py              # Opens browser to runner UI
    python prism_run.py --port 8080  # Custom port
    python prism_run.py --no-browser # Don't auto-open browser
    python prism_run.py --cli        # Fall back to CLI mode

The runner provides:
- Panel selection (Market, Economy, Climate, Custom)
- Workflow selection (auto-populated from registry)
- Timeframe configuration
- Engine weighting options
- One-click execution
- Auto-generated HTML reports
"""

import sys
import os
import logging
import webbrowser
import threading
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "runner" else Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for Flask
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not installed. Install with: pip install flask")


# =============================================================================
# FLASK APPLICATION
# =============================================================================

def create_app():
    """Create and configure the Flask application."""

    # Template and static folders
    template_dir = PROJECT_ROOT / "templates"
    static_dir = PROJECT_ROOT / "templates" / "static"
    output_dir = PROJECT_ROOT / "output"

    # Create directories if needed
    template_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir)
    )

    app.config['SECRET_KEY'] = 'prism-runner-local-only'
    app.config['PROJECT_ROOT'] = PROJECT_ROOT
    app.config['OUTPUT_DIR'] = output_dir

    # -------------------------------------------------------------------------
    # ROUTES
    # -------------------------------------------------------------------------

    @app.route("/")
    def index():
        """Main runner interface."""

        # Load available options from registries
        panels = get_available_panels()
        workflows = get_available_workflows()
        weighting_methods = get_weighting_methods()

        return render_template(
            "runner/index.html",
            panels=panels,
            workflows=workflows,
            weighting_methods=weighting_methods,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

    @app.route("/api/workflows/<panel>")
    def get_workflows_for_panel(panel):
        """Get workflows compatible with a panel (AJAX endpoint)."""
        workflows = get_available_workflows(panel)
        return jsonify(workflows)

    @app.route("/api/workflow/<workflow_key>")
    def get_workflow_details(workflow_key):
        """Get details for a specific workflow (AJAX endpoint)."""
        details = get_workflow_info(workflow_key)
        return jsonify(details)

    @app.route("/run", methods=["POST"])
    def run_analysis():
        """Execute the selected analysis."""

        # Get form parameters
        params = {
            "panel": request.form.get("panel", "market"),
            "workflow": request.form.get("workflow", "regime_comparison"),
            "weighting": request.form.get("weighting", "hybrid"),
            "timeframe": request.form.get("timeframe", "default"),
            "start_date": request.form.get("start_date"),
            "end_date": request.form.get("end_date"),
            "comparison_periods": request.form.getlist("comparison_periods"),
        }

        logger.info(f"Running analysis with params: {params}")

        # Execute via dispatcher
        try:
            from runner.dispatcher import Dispatcher
            dispatcher = Dispatcher()
            results = dispatcher.run(params)

            return render_template(
                "runner/results.html",
                success=True,
                results=results,
                params=params
            )

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            return render_template(
                "runner/results.html",
                success=False,
                error=str(e),
                params=params
            )

    @app.route("/api/run", methods=["POST"])
    def run_analysis_api():
        """Execute analysis and return JSON (for async calls)."""

        params = request.json or {}

        try:
            from runner.dispatcher import Dispatcher
            dispatcher = Dispatcher()
            results = dispatcher.run(params)

            return jsonify({
                "success": True,
                "results": results
            })

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route("/output/<path:filename>")
    def serve_output(filename):
        """Serve files from the output directory."""
        return send_from_directory(str(output_dir), filename)

    @app.route("/reports")
    def list_reports():
        """List recent reports."""
        reports = get_recent_reports()
        return render_template("runner/reports.html", reports=reports)

    @app.route("/health")
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

    return app


# =============================================================================
# DATA LOADERS (from registries)
# =============================================================================

def get_available_panels():
    """Get available panel domains."""
    # Start with hardcoded, will be replaced by panel registry
    panels = [
        {"key": "market", "name": "Market", "description": "ETFs, Indices, Sectors", "icon": "📈"},
        {"key": "economy", "name": "Economy", "description": "FRED Economic Data", "icon": "🏛️"},
        {"key": "market_economy", "name": "Market + Economy", "description": "Combined Analysis", "icon": "📊"},
        {"key": "climate", "name": "Climate", "description": "Climate Data (Beta)", "icon": "🌍"},
        {"key": "custom", "name": "Custom", "description": "User-defined Panel", "icon": "⚙️"},
    ]

    # Try to load from panel registry
    try:
        panel_registry = PROJECT_ROOT / "panels"
        if panel_registry.exists():
            for panel_dir in panel_registry.iterdir():
                if panel_dir.is_dir() and (panel_dir / "registry.yaml").exists():
                    # Panel exists in registry
                    pass
    except Exception as e:
        logger.debug(f"Could not load panel registry: {e}")

    return panels


def get_available_workflows(panel: str = None):
    """Get available workflows, optionally filtered by panel."""

    # Try to load from workflow registry
    try:
        from workflows.workflow_loader import WorkflowLoader
        loader = WorkflowLoader()

        if panel:
            workflow_keys = loader.get_compatible_workflows(panel)
        else:
            workflow_keys = loader.list_workflows()

        workflows = []
        for key in workflow_keys:
            info = loader.get_workflow_info(key)
            workflows.append({
                "key": key,
                "name": info.get("name", key),
                "description": info.get("description", ""),
                "category": info.get("category", ""),
                "runtime": info.get("estimated_runtime", ""),
                "parameters": info.get("parameters", {}),
            })

        return workflows

    except Exception as e:
        logger.debug(f"Could not load workflow registry: {e}")

    # Fallback to hardcoded
    return [
        {
            "key": "regime_comparison",
            "name": "Regime Comparison",
            "description": "Compare current regime to historical periods",
            "category": "regime",
            "runtime": "2-5 min",
        },
        {
            "key": "full_temporal",
            "name": "Full Temporal Analysis",
            "description": "55-window analysis from 1970-2025",
            "category": "temporal",
            "runtime": "10-30 min",
        },
        {
            "key": "daily_update",
            "name": "Daily Update",
            "description": "Quick daily market check",
            "category": "monitoring",
            "runtime": "30 sec",
        },
        {
            "key": "overnight_batch",
            "name": "Overnight Batch",
            "description": "Comprehensive overnight analysis",
            "category": "monitoring",
            "runtime": "15-30 min",
        },
    ]


def get_workflow_info(workflow_key: str):
    """Get detailed info for a workflow."""
    try:
        from workflows.workflow_loader import WorkflowLoader
        loader = WorkflowLoader()
        return loader.get_workflow_info(workflow_key) or {}
    except:
        return {}


def get_weighting_methods():
    """Get available weighting methods."""
    return [
        {"key": "meta", "name": "Meta Engine", "description": "Democratic lens consensus"},
        {"key": "ml", "name": "ML Engine", "description": "Learned from historical patterns"},
        {"key": "hybrid", "name": "Hybrid", "description": "Meta + ML combined"},
        {"key": "none", "name": "None", "description": "Equal weights"},
    ]


def get_recent_reports(limit: int = 20):
    """Get list of recent reports."""
    reports = []
    output_dir = PROJECT_ROOT / "output"

    if not output_dir.exists():
        return reports

    # Find all HTML reports
    html_files = list(output_dir.rglob("*.html"))
    html_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for f in html_files[:limit]:
        reports.append({
            "name": f.name,
            "path": str(f.relative_to(output_dir)),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size": f.stat().st_size,
        })

    return reports


# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

def open_browser(port: int, delay: float = 1.0):
    """Open browser after a short delay."""
    import time
    time.sleep(delay)
    url = f"http://localhost:{port}"
    logger.info(f"Opening browser to {url}")
    webbrowser.open(url)


def run_server(port: int = 5000, debug: bool = False, open_browser_flag: bool = True):
    """Run the Flask server."""

    if not FLASK_AVAILABLE:
        print("Error: Flask is required for the HTML runner.")
        print("Install with: pip install flask")
        print("Or use: python prism_run.py --cli")
        return

    app = create_app()

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     PRISM RUNNER                             ║
    ║                                                              ║
    ║   Starting web interface...                                  ║
    ║   URL: http://localhost:{port}                                ║
    ║                                                              ║
    ║   Press Ctrl+C to stop                                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Open browser in background thread
    if open_browser_flag:
        browser_thread = threading.Thread(target=open_browser, args=(port,))
        browser_thread.daemon = True
        browser_thread.start()

    # Run server
    app.run(
        host="127.0.0.1",
        port=port,
        debug=debug,
        use_reloader=False  # Disable reloader to prevent double browser opens
    )


# =============================================================================
# CLI FALLBACK
# =============================================================================

def run_cli():
    """Run the CLI version (fallback)."""
    # Import the CLI runner from the existing prism_run.py logic
    print("Running in CLI mode...")

    # Simple CLI menu
    print("""
    PRISM Runner (CLI Mode)
    =======================

    1. Regime Comparison
    2. Full Temporal Analysis
    3. Daily Update
    0. Exit
    """)

    choice = input("Select workflow [1]: ").strip() or "1"

    if choice == "0":
        return

    workflows = {
        "1": "regime_comparison",
        "2": "full_temporal",
        "3": "daily_update",
    }

    workflow = workflows.get(choice, "regime_comparison")

    print(f"\nRunning {workflow}...")

    try:
        from runner.dispatcher import Dispatcher
        dispatcher = Dispatcher()
        results = dispatcher.run({"workflow": workflow, "panel": "market"})
        print(f"\nComplete! Report: {results.get('report_path', 'N/A')}")
    except Exception as e:
        print(f"\nError: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prism_run.py                    # Launch web UI
  python prism_run.py --port 8080        # Custom port
  python prism_run.py --cli              # CLI mode
  python prism_run.py --no-browser       # Don't auto-open browser
        """
    )

    parser.add_argument("--port", "-p", type=int, default=5000, help="Port for web server")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--cli", action="store_true", help="Use CLI mode instead of web UI")

    args = parser.parse_args()

    if args.cli or not FLASK_AVAILABLE:
        run_cli()
    else:
        run_server(
            port=args.port,
            debug=args.debug,
            open_browser_flag=not args.no_browser
        )


if __name__ == "__main__":
    main()
