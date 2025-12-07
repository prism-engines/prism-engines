# dashboard/app.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, render_template, request, jsonify, send_from_directory

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
START_DIR = BASE_DIR / "start"
DIAG_DIR = BASE_DIR / "diagnostics"
OUTPUT_DIR = Path(
    os.environ.get(
        "PRISM_OUTPUT_DIR",
        str(
            Path.home()
            / "Library"
            / "CloudStorage"
            / "GoogleDrive-rudder.jason@gmail.com"
            / "My Drive"
            / "prism_output"
        ),
    )
)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "dashboard" / "templates"),
    static_folder=str(BASE_DIR / "dashboard" / "static"),
)

# -----------------------------------------------------------------------------
# Model metadata for /models page
# -----------------------------------------------------------------------------

MODELS_METADATA = [
    {
        "id": "wavelet",
        "name": "Wavelet Coherence Engine",
        "category": "Geometry / Time–Frequency",
        "short": "Measures how synchronized two signals are across time and frequency.",
        "inputs": [
            "Daily or weekly return series for 2+ indicators",
            "Aligned timestamp index (no timezone required)",
            "Minimum history: ~3× the longest wavelet scale used"
        ],
        "outputs": [
            "Coherence surface C(t, s) in [0, 1]",
            "Time series of aggregate coherence scores",
            "Candidate high-coherence "regime shift" windows"
        ],
        "used_for": [
            "Detecting when many indicators move in lockstep",
            "Flagging potential regime shift periods",
            "Validating lens agreement during stress"
        ],
    },
    {
        "id": "hmm",
        "name": "Hidden Markov Model (HMM) Regime Engine",
        "category": "Unsupervised Machine Learning",
        "short": "Learns latent market regimes from returns and volatility without labeled data.",
        "inputs": [
            "Panel of returns and/or volatility features",
            "Regular time step (daily recommended)",
            "Configurable number of hidden states (e.g. 2–4)"
        ],
        "outputs": [
            "State probabilities P(state | t) over time",
            "Most likely regime path",
            "Transition matrix between regimes"
        ],
        "used_for": [
            "Identifying bull/bear/transition regimes",
            "Comparing with other regime lenses (GMM, PCA)",
            "Feeding regime labels into downstream geometry"
        ],
    },
    {
        "id": "pca",
        "name": "PCA Geometry Engine",
        "category": "Dimensionality Reduction",
        "short": "Projects the indicator panel into a low-dimensional geometric space.",
        "inputs": [
            "Normalized indicator panel (returns or levels)",
            "Sufficient cross-section (N indicators) >> components",
            "Rolling windows for time-varying geometry"
        ],
        "outputs": [
            "Eigenvalues / explained variance per component",
            "Eigenvectors / factor loadings",
            "Low-dimensional coordinates for each indicator"
        ],
        "used_for": [
            "Tracking concentration of risk into few factors",
            "Detecting geometry changes in the risk surface",
            "Supporting MRF / PRF / CRF composite signals"
        ],
    },
    {
        "id": "gmm",
        "name": "Gaussian Mixture (GMM) Clustering Engine",
        "category": "Unsupervised Machine Learning",
        "short": "Clusters observations into probabilistic regimes in return–feature space.",
        "inputs": [
            "Feature matrix (returns, volatility, spreads, etc.)",
            "Number of clusters (or model selection range)",
            "Optional PCA pre-projection"
        ],
        "outputs": [
            "Cluster responsibilities per observation",
            "Cluster centroids and covariance matrices",
            "Regime labels aligned with dates"
        ],
        "used_for": [
            "Alternative regime labelling vs HMM",
            "Comparing geometric vs probabilistic regimes",
            "Stress-testing regime stability over time"
        ],
    },
    {
        "id": "network",
        "name": "Network / Graph Geometry Engine",
        "category": "Topology & Graph Theory",
        "short": "Builds a correlation network and studies its structure over time.",
        "inputs": [
            "Correlation or distance matrix between indicators",
            "Threshold or K-nearest neighbor rule",
            "Rolling windows for time evolution"
        ],
        "outputs": [
            "Graph metrics (degree, centrality, clustering)",
            "Component structure and connectivity",
            "Network-based early warning metrics"
        ],
        "used_for": [
            "Identifying central risk hubs",
            "Tracking fragmentation vs synchronization",
            "Complementing correlation and coherence lenses"
        ],
    },
    {
        "id": "dispersion",
        "name": "Dispersion & Volatility Geometry Engine",
        "category": "Distributional Geometry",
        "short": "Measures how spread out indicators are vs the market and vs each other.",
        "inputs": [
            "Panel of returns by indicator",
            "Benchmark (e.g. S&P 500) if available",
            "Rolling window size"
        ],
        "outputs": [
            "Cross-sectional dispersion measures",
            "Sector/style dispersion indexes",
            "Phase-space views of risk spreading"
        ],
        "used_for": [
            "Detecting early dispersion before major moves",
            "Measuring breadth of participation in trends",
            "Feeding dispersion into composite risk scores"
        ],
    },
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def run_python_script(script_path: Path, args: List[str] | None = None) -> Dict[str, Any]:
    """
    Run a Python script in the current virtualenv and capture stdout/stderr.
    Returns a dict {returncode, stdout, stderr}.
    """
    if args is None:
        args = []

    cmd = [sys.executable, str(script_path), *args]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=60 * 60,  # 1 hour max, just to be safe
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": f"TimeoutExpired: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Exception when running {script_path}: {exc}",
        }


def summarize_result(label: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Small helper to normalize result blocks for the UI."""
    status = "ok"
    if result["returncode"] != 0:
        status = "error"

    # Trim long outputs a bit for JSON payload
    def _trim(text: str, limit: int = 4000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n\n...[truncated]..."

    return {
        "label": label,
        "status": status,
        "returncode": result["returncode"],
        "stdout": _trim(result.get("stdout", "")),
        "stderr": _trim(result.get("stderr", "")),
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.route("/")
def index():
    """
    Main dashboard page.
    The form controls are in dashboard/templates/dashboard.html.
    """
    return render_template("dashboard.html")


@app.route("/api/run", methods=["POST"])
def run_controller():
    """
    Central PRISM controller endpoint.

    Expects JSON like:
    {
      "analysis": "coherence",
      "diagnostics": "db",
      "fetch_source": ["fred"],
      "fetch_range": "30d",
      "mlmeta": "none",
      "systems": ["market"]
    }
    """
    payload = request.get_json(force=True) or {}
    analysis = payload.get("analysis")
    diagnostics_mode = payload.get("diagnostics")
    fetch_sources = payload.get("fetch_source", [])
    fetch_range = payload.get("fetch_range", "30d")
    mlmeta_mode = payload.get("mlmeta", "none")
    systems = payload.get("systems", [])

    results: Dict[str, Any] = {
        "payload_received": payload,
        "actions": [],
    }

    # ------------------------------------------------------
    # 1) Optional: run diagnostics
    # ------------------------------------------------------
    if diagnostics_mode and diagnostics_mode != "none":
        diag_script = DIAG_DIR / "run_diagnostics.py"
        if diag_script.exists():
            diag_args: List[str] = []
            if diagnostics_mode == "full":
                diag_args = []  # run all
            elif diagnostics_mode == "db":
                diag_args = ["--category", "health"]
            elif diagnostics_mode == "engine":
                diag_args = ["--category", "performance"]
            elif diagnostics_mode == "fetch":
                diag_args = ["--category", "validation"]
            elif diagnostics_mode == "lenses":
                diag_args = ["--category", "validation"]
            elif diagnostics_mode == "engines":
                diag_args = ["--category", "performance"]

            r = run_python_script(diag_script, diag_args)
            results["actions"].append(
                summarize_result("diagnostics", r),
            )
        else:
            results["actions"].append(
                {
                    "label": "diagnostics",
                    "status": "missing",
                    "message": f"{diag_script} not found",
                }
            )

    # ------------------------------------------------------
    # 2) Optional: fetch data
    # (For now we just note the intent; can wire to update_all.py later)
    # ------------------------------------------------------
    if fetch_sources and "none" not in fetch_sources:
        # Placeholder for future:
        # Could pass --source fred,tiingo etc. to update_all.py
        results["actions"].append(
            {
                "label": "fetch",
                "status": "skipped",
                "message": (
                    "Fetch requested for sources="
                    f"{', '.join(fetch_sources)} (range={fetch_range}), "
                    "but fetch wiring is intentionally disabled here to "
                    "avoid hammering APIs. Implement in start/update_all.py "
                    "with source filters."
                ),
            }
        )

    # ------------------------------------------------------
    # 3) Run analysis
    # ------------------------------------------------------
    analysis_scripts = {
        "coherence": START_DIR / "run_coherance.py",
        "correlation": START_DIR / "run_correlation.py",
        "calibrated": START_DIR / "run_calibrated.py",
        "regime": START_DIR / "run_regime_analysis.py",
        "20yr": START_DIR / "run_full_20y_analysis.py",
        "40yr": START_DIR / "run_full_40y_analysis.py",
        "families": START_DIR / "run_correlation.py",  # Cross-family uses correlation
    }

    if analysis and analysis != "none":
        script = analysis_scripts.get(analysis)
        if script and script.exists():
            r = run_python_script(script)
            results["actions"].append(
                summarize_result(analysis, r),
            )
        elif script:
            results["actions"].append(
                {
                    "label": analysis,
                    "status": "missing",
                    "message": f"{script} not found",
                }
            )
        else:
            results["actions"].append(
                {
                    "label": "analysis",
                    "status": "unknown",
                    "message": f"Unknown analysis mode: {analysis}",
                }
            )
    elif not analysis or analysis == "none":
        results["actions"].append(
            {
                "label": "analysis",
                "status": "skipped",
                "message": "No analysis selected.",
            }
        )

    # ------------------------------------------------------
    # 4) ML / Meta mode – placeholder for now
    # ------------------------------------------------------
    if mlmeta_mode and mlmeta_mode != "none":
        results["actions"].append(
            {
                "label": "mlmeta",
                "status": "todo",
                "message": (
                    f"ML/Meta mode '{mlmeta_mode}' selected; "
                    "hook this into hidden_variation_detector + meta-engine "
                    "when ready."
                ),
            }
        )

    # ------------------------------------------------------
    # 5) System selection – just echo back for now
    # ------------------------------------------------------
    if systems:
        results["systems_used"] = systems

    return jsonify(
        {
            "status": "ok",
            "message": "PRISM controller executed.",
            "result": results,
        }
    )


# -----------------------------------------------------------------------------
# Models page routes
# -----------------------------------------------------------------------------


@app.route("/models")
def models():
    """
    Models & Geometry documentation page.
    """
    return render_template("models.html", models=MODELS_METADATA)


@app.route("/downloads/models_overview.md")
def download_models_overview():
    """
    Serve the Markdown documentation for all models.
    """
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    return send_from_directory(docs_dir, "models_overview.md", as_attachment=True)


if __name__ == "__main__":
    # Run dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
