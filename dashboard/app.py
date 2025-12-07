# dashboard/app.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np

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


# -----------------------------------------------------------------------------
# Engine vs Series API
# -----------------------------------------------------------------------------

# Series mapping: user-friendly names to indicator IDs
SERIES_MAPPING = {
    "sp500": "sp500_d",
    "nasdaq": "nasdaq_d",
    "t10y2y": "t10y2y_d",
    "vix": "vix_d",
    "m2sl": "m2sl_m",
    "dgs10": "dgs10_d",
    "dgs2": "dgs2_d",
    "effr": "effr_d",
}


def _compute_rolling_correlation(panel: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    Compute rolling mean pairwise correlation as a coherence proxy.
    Returns a Series indexed by date with values in [0, 1].
    """
    if panel.shape[1] < 2:
        return pd.Series(index=panel.index, data=0.5)

    scores = []
    dates = []

    for i in range(window, len(panel)):
        window_data = panel.iloc[i - window:i]
        corr_matrix = window_data.corr()
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_corrs = corr_matrix.where(mask).stack()
        if len(upper_corrs) > 0:
            # Mean absolute correlation as coherence score
            mean_corr = upper_corrs.abs().mean()
            scores.append(float(mean_corr))
        else:
            scores.append(0.5)
        dates.append(panel.index[i])

    return pd.Series(data=scores, index=dates)


def _compute_pca_instability(panel: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    Compute rolling PCA instability: change in PC1 explained variance ratio.
    High variance in explained ratio = unstable geometry.
    """
    from sklearn.decomposition import PCA

    if panel.shape[1] < 2:
        return pd.Series(index=panel.index, data=0.5)

    pc1_ratios = []
    dates = []

    for i in range(window, len(panel)):
        window_data = panel.iloc[i - window:i].dropna(axis=1, how="all")
        window_data = window_data.dropna()

        if window_data.shape[0] < 10 or window_data.shape[1] < 2:
            pc1_ratios.append(np.nan)
            dates.append(panel.index[i])
            continue

        try:
            pca = PCA(n_components=1)
            pca.fit(window_data)
            pc1_ratios.append(pca.explained_variance_ratio_[0])
        except Exception:
            pc1_ratios.append(np.nan)
        dates.append(panel.index[i])

    result = pd.Series(data=pc1_ratios, index=dates)

    # Compute rolling change as instability
    instability = result.diff().abs().rolling(window=21, min_periods=1).mean()
    # Normalize to 0-1
    if instability.max() > 0:
        instability = instability / instability.max()

    return instability.fillna(0.5)


def _compute_dispersion(panel: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Compute rolling cross-sectional dispersion (standard deviation across indicators).
    Higher dispersion = more divergence in behavior.
    """
    # Normalize each column to returns
    returns = panel.pct_change()

    scores = []
    dates = []

    for i in range(window, len(returns)):
        window_data = returns.iloc[i - window:i]
        # Mean cross-sectional std for each day, then average over window
        cross_std = window_data.std(axis=1).mean()
        scores.append(float(cross_std) if pd.notna(cross_std) else 0)
        dates.append(returns.index[i])

    result = pd.Series(data=scores, index=dates)

    # Normalize to 0-1
    if result.max() > 0:
        result = result / result.max()

    return result.fillna(0)


def _compute_regime_score(panel: pd.DataFrame, window: int = 126) -> pd.Series:
    """
    Compute regime stress score based on rolling volatility regime.
    Uses volatility clustering as a regime proxy.
    """
    if panel.shape[1] < 1:
        return pd.Series(index=panel.index, data=0.5)

    # Use first column or mean for volatility
    if panel.shape[1] > 1:
        mean_series = panel.mean(axis=1)
    else:
        mean_series = panel.iloc[:, 0]

    returns = mean_series.pct_change().dropna()
    vol = returns.rolling(window=21, min_periods=5).std()

    # Rolling percentile of volatility as regime score
    scores = []
    dates = []

    for i in range(window, len(vol)):
        window_vol = vol.iloc[i - window:i]
        current_vol = vol.iloc[i]
        if pd.notna(current_vol) and len(window_vol.dropna()) > 0:
            percentile = (window_vol < current_vol).sum() / len(window_vol.dropna())
            scores.append(float(percentile))
        else:
            scores.append(0.5)
        dates.append(vol.index[i])

    return pd.Series(data=scores, index=dates)


def compute_engine_series(
    panel: pd.DataFrame,
    engine_name: str,
    window: int = 63
) -> pd.Series:
    """
    Compute engine scores over time.

    Args:
        panel: Wide-format DataFrame with indicators as columns
        engine_name: One of coherence, hmm_regime, pca_instability, dispersion, meta, composite
        window: Rolling window size

    Returns:
        pd.Series indexed by date with engine scores in [0, 1]
    """
    engine_name = engine_name.lower()

    if engine_name == "coherence":
        return _compute_rolling_correlation(panel, window=window)
    elif engine_name in ("hmm_regime", "regime"):
        return _compute_regime_score(panel, window=window)
    elif engine_name == "pca_instability":
        return _compute_pca_instability(panel, window=window)
    elif engine_name == "dispersion":
        return _compute_dispersion(panel, window=21)
    elif engine_name in ("meta", "composite"):
        # Average of all engines
        coherence = _compute_rolling_correlation(panel, window=window)
        regime = _compute_regime_score(panel, window=window)
        pca_inst = _compute_pca_instability(panel, window=window)
        dispersion = _compute_dispersion(panel, window=21)

        # Align and average
        combined = pd.DataFrame({
            "coherence": coherence,
            "regime": regime,
            "pca": pca_inst,
            "dispersion": dispersion,
        })
        return combined.mean(axis=1).fillna(0.5)
    else:
        raise ValueError(f"Unknown engine: {engine_name}")


def align_engine_and_series(engine_series: pd.Series, data_series: pd.Series) -> pd.DataFrame:
    """
    Align engine scores and data series on shared dates.
    Returns DataFrame with columns: ['engine', 'series'].
    """
    df = pd.DataFrame({
        "engine": engine_series,
        "series": data_series,
    }).dropna(how="any")
    return df


@app.route("/api/engine_vs_series", methods=["GET"])
def api_engine_vs_series():
    """
    Return engine output (bars) and selected series (line) on a common time axis.

    Query params:
      engine:     coherence | hmm_regime | pca_instability | dispersion | meta | composite
      series:     indicator id (e.g., sp500, vix, t10y2y)
      start:      YYYY-MM-DD (optional)
      end:        YYYY-MM-DD (optional)
      frequency:  daily | weekly | monthly (optional, default daily)
    """
    engine = request.args.get("engine", "coherence")
    series_id = request.args.get("series", "sp500")
    start = request.args.get("start")
    end = request.args.get("end")
    frequency = request.args.get("frequency", "daily")

    try:
        # Import data loaders
        from panel.runtime_loader import load_panel, list_available_indicators
        from data.sql.db_connector import load_indicator

        # Get available indicators
        available = list_available_indicators()

        if not available:
            return jsonify({
                "error": "No indicators available in database",
                "dates": [],
                "bars": [],
                "line": [],
            }), 200

        # Load panel for engine computation (use all available indicators)
        panel = load_panel(
            indicator_names=available,
            start_date=start,
            end_date=end,
            skip_hvd_check=True,
        )

        if panel.empty:
            return jsonify({
                "error": "Panel is empty - no data in selected date range",
                "dates": [],
                "bars": [],
                "line": [],
            }), 200

        # Handle frequency resampling
        if frequency == "weekly":
            panel = panel.resample("W").last()
        elif frequency == "monthly":
            panel = panel.resample("ME").last()

        # Compute engine scores
        engine_scores = compute_engine_series(panel, engine)

        # Load selected series
        indicator_id = SERIES_MAPPING.get(series_id.lower(), series_id)
        series_df = load_indicator(indicator_id)

        if series_df.empty:
            return jsonify({
                "error": f"Series '{series_id}' not found in database",
                "dates": [],
                "bars": [],
                "line": [],
            }), 200

        # Convert to series
        series_df["date"] = pd.to_datetime(series_df["date"])
        data_series = series_df.set_index("date")["value"]

        # Apply date filters to series
        if start:
            data_series = data_series[data_series.index >= start]
        if end:
            data_series = data_series[data_series.index <= end]

        # Handle frequency resampling for series
        if frequency == "weekly":
            data_series = data_series.resample("W").last()
        elif frequency == "monthly":
            data_series = data_series.resample("ME").last()

        # Align engine and series
        aligned = align_engine_and_series(engine_scores, data_series)

        if aligned.empty:
            return jsonify({
                "error": "No overlapping dates between engine and series",
                "dates": [],
                "bars": [],
                "line": [],
            }), 200

        # Normalize series values for display (z-score normalization)
        series_values = aligned["series"].values
        if len(series_values) > 1 and np.std(series_values) > 0:
            # Keep original values for display - frontend will handle dual axis
            pass

        return jsonify({
            "dates": [d.strftime("%Y-%m-%d") for d in aligned.index],
            "bars": [round(v, 4) for v in aligned["engine"].tolist()],
            "line": [round(v, 4) if pd.notna(v) else None for v in aligned["series"].tolist()],
            "meta": {
                "engine": engine,
                "series": series_id,
                "start": aligned.index.min().strftime("%Y-%m-%d") if not aligned.empty else None,
                "end": aligned.index.max().strftime("%Y-%m-%d") if not aligned.empty else None,
                "frequency": frequency,
                "n_points": len(aligned),
            }
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "dates": [],
            "bars": [],
            "line": [],
        }), 200


@app.route("/api/available_series", methods=["GET"])
def api_available_series():
    """Return list of available series for the dropdown."""
    try:
        from panel.runtime_loader import list_available_indicators

        available = list_available_indicators()

        # Build response with friendly names
        series_list = []
        for key, indicator_id in SERIES_MAPPING.items():
            if indicator_id in available:
                series_list.append({
                    "id": key,
                    "name": key.upper().replace("_", " "),
                    "indicator_id": indicator_id,
                })

        # Also add any other available indicators
        mapped_ids = set(SERIES_MAPPING.values())
        for ind in available:
            if ind not in mapped_ids:
                series_list.append({
                    "id": ind,
                    "name": ind,
                    "indicator_id": ind,
                })

        return jsonify({"series": series_list})

    except Exception as e:
        return jsonify({"series": [], "error": str(e)})


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    env_port = os.environ.get("FLASK_RUN_PORT")
    port = args.port or (int(env_port) if env_port else 5000)

    app.run(host=args.host, port=port, debug=True, use_reloader=False)
