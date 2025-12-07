PR TITLE

Add Engine-vs-Series Visualization to PRISM Dashboard

SUMMARY

This PR adds a new “Engine vs Series” visualization mode to the PRISM dashboard:

Bars (left axis) show selected engine outputs (coherence, HMM regime, PCA instability, etc.).
Line (right axis) shows a user-selected indicator or indicator group (e.g., SP500, NASDAQ, Treasuries, climate series).

The goal is to let users see:

“Here’s what happened in the data (line), and here’s what PRISM’s engines thought about it (bars).”

This becomes a core, default PRISM view.

GOALS

Implement a new API endpoint that returns:

Time series of engine scores (bars)

Time series of indicator / group levels (line)

Add UI controls to the dashboard so the user can select:

Engine (for bars)

Indicator / family / group (for line)

Date range

Frequency (daily / weekly / monthly)

Render a dual-axis chart:

Left axis: engine score (0–1 or normalized)

Right axis: data series level (or normalized)

Bars for engine, line for series

Keep it safe & performant by:

Using existing DB paths & loaders

Reusing runtime_loader / panel tools where possible

Avoiding expensive, default-heavy workloads

SCOPE

In scope:

Flask route & JSON API to serve engine + series data.

Frontend controls + chart rendering in dashboard/templates/dashboard.html and dashboard/static/dashboard.js.

Minimal documentation in dashboard/README.md or in a new short section.

Wiring this into the existing Dashboard as a primary visualization.

Not in scope (yet):

New engines, ML models, or meta logic.

Heavy 20y/40y full analyses (those remain manual triggers).

Authentication, multi-user, or production-grade deployment.

IMPLEMENTATION PLAN
1. Backend: Data & API

Key design:

Add a new Flask route under dashboard/app.py:

# dashboard/app.py

from flask import Flask, render_template, request, jsonify
from panel.runtime_loader import load_runtime_panel  # already exists
from diagnostics.core.runner import resolve_db_path  # if available, or reuse existing helper
from data.family_manager import FamilyManager        # families for grouped series

@app.route("/api/engine_vs_series", methods=["GET"])
def api_engine_vs_series():
    """
    Return engine output (bars) and selected series (line) on a common time axis.

    Query params:
      engine:     coherence | hmm_regime | pca_instability | dispersion | meta | composite
      series:     indicator id OR family id OR special alias (e.g. 'treasury_curve', 'growth_style')
      start:      YYYY-MM-DD (optional)
      end:        YYYY-MM-DD (optional)
      frequency:  daily | weekly | monthly (optional, default daily)
    """
    engine = request.args.get("engine", "coherence")
    series_id = request.args.get("series", "sp500")
    start = request.args.get("start")
    end = request.args.get("end")
    frequency = request.args.get("frequency", "daily")

    # 1) Resolve DB and load panel via existing runtime loader
    db_path = resolve_db_path()  # or use the same logic used in diagnostics/dash
    panel = load_runtime_panel(db_path=db_path, frequency=frequency)

    # 2) Get underlying series (indicator or family)
    fm = FamilyManager()
    # Pseudocode: resolve whether 'series_id' is a family or direct indicator
    # Implement helper on FamilyManager if needed:
    #   - fm.get_series(panel, series_id) -> pd.Series
    series = fm.get_series(panel, series_id)  # daily/weekly/monthly level series

    # 3) Compute engine outputs over same date range
    # Option A: use existing coherence / engine modules
    # Option B: add a thin wrapper that computes:
    #   - coherence over sliding windows (default 63d)
    #   - HMM probabilities
    #   - PCA instability, etc.
    from analysis.hidden_variation_detector import HiddenVariationDetector
    # OR import existing engine modules as appropriate

    engine_scores = compute_engine_series(
        panel=panel,
        engine_name=engine,
        start=start,
        end=end,
    )

    # Align engine_scores and series on intersecting dates
    aligned = align_engine_and_series(engine_scores, series)

    return jsonify({
        "dates": [d.strftime("%Y-%m-%d") for d in aligned.index],
        "bars": aligned["engine"].tolist(),
        "line": aligned["series"].tolist(),
        "meta": {
            "engine": engine,
            "series": series_id,
            "start": aligned.index.min().strftime("%Y-%m-%d") if not aligned.empty else None,
            "end": aligned.index.max().strftime("%Y-%m-%d") if not aligned.empty else None,
            "frequency": frequency,
        }
    })


Add helper functions in a small backend module or in dashboard/app.py for now (Claude can choose):

import pandas as pd

def compute_engine_series(panel, engine_name: str, start=None, end=None) -> pd.Series:
    """
    Returns a pandas Series indexed by date with engine scores in [0,1] or normalized.

    engine_name options:
      - coherence
      - hmm_regime
      - pca_instability
      - dispersion
      - meta
      - composite
    """
    # Filter panel by date if start/end provided
    if start:
        panel = panel[panel.index >= start]
    if end:
        panel = panel[panel.index <= end]

    # Example: coherence engine using existing logic
    if engine_name == "coherence":
        from start.run_coherance import compute_coherence_series  # or refactor coherence logic into a shared function
        return compute_coherence_series(panel)

    # Example: placeholder for other engines
    elif engine_name == "hmm_regime":
        # Use existing HMM/regime engine used in overnight / regime analysis
        from analysis.regime_engine import compute_hmm_regime_scores
        return compute_hmm_regime_scores(panel)

    # etc. for pca_instability / dispersion / meta / composite
    # For v1, it's okay to implement coherence + 1 other engine, then stub others with clear TODOs.

    raise ValueError(f"Unsupported engine: {engine_name}")


def align_engine_and_series(engine_series: pd.Series, series: pd.Series) -> pd.DataFrame:
    """
    Align engine scores and data series on shared dates.
    Returns DataFrame with columns: ['engine', 'series'].
    """
    df = pd.DataFrame({
        "engine": engine_series,
        "series": series,
    }).dropna(how="any")
    return df


Note for Claude:
Use existing, tested logic for coherence, regime, etc., instead of rewriting from scratch. If needed, refactor the relevant computation from the start/ scripts into reusable functions in analysis/ or panel/.

2. Frontend: Dashboard Controls & Chart

Files to modify:

dashboard/templates/dashboard.html

dashboard/static/dashboard.js

Possibly minor updates to dashboard/templates/base.html

a) Add Controls Section

In dashboard/templates/dashboard.html, add a new row or card dedicated to Engine vs Series:

<div class="card mb-4">
  <div class="card-header">
    <h3 class="card-title">Engine vs Series</h3>
    <p class="card-subtitle text-muted">
      Compare PRISM engine outputs (bars) vs a selected indicator or group (line).
    </p>
  </div>
  <div class="card-body">
    <form id="engine-series-form" class="row g-3 align-items-end">
      <div class="col-md-3">
        <label for="engineSelect" class="form-label">Engine (Bars)</label>
        <select id="engineSelect" class="form-select">
          <option value="coherence">Coherence Engine</option>
          <option value="hmm_regime">HMM Regime Probabilities</option>
          <option value="pca_instability">PCA Instability</option>
          <option value="dispersion">Dispersion Engine</option>
          <option value="meta">Meta-Lens Composite</option>
          <option value="composite">PRISM Composite</option>
        </select>
      </div>
      <div class="col-md-3">
        <label for="seriesSelect" class="form-label">Line Series</label>
        <select id="seriesSelect" class="form-select">
          <!-- For v1: hard-code a few good defaults; later this can be dynamic -->
          <option value="sp500">S&P 500 (sp500)</option>
          <option value="nasdaq">NASDAQ Composite (nasdaq)</option>
          <option value="t10y2y">Yield Curve (T10Y2Y)</option>
          <option value="vix">VIX (vix)</option>
          <option value="m2sl">Money Supply (M2SL)</option>
          <!-- If families/groups are wired, we can use family IDs here -->
        </select>
      </div>
      <div class="col-md-2">
        <label for="startDate" class="form-label">Start Date</label>
        <input type="date" id="startDate" class="form-control" />
      </div>
      <div class="col-md-2">
        <label for="endDate" class="form-label">End Date</label>
        <input type="date" id="endDate" class="form-control" />
      </div>
      <div class="col-md-2">
        <label for="frequencySelect" class="form-label">Frequency</label>
        <select id="frequencySelect" class="form-select">
          <option value="daily" selected>Daily</option>
          <option value="weekly">Weekly</option>
          <option value="monthly">Monthly</option>
        </select>
      </div>
      <div class="col-12 mt-2">
        <button type="button" id="runEngineSeriesBtn" class="btn btn-primary">
          Run Engine vs Series
        </button>
      </div>
    </form>

    <div id="engine-series-chart-container" class="mt-4">
      <!-- Chart canvas goes here -->
      <canvas id="engineSeriesChart"></canvas>
    </div>
  </div>
</div>


Note: Use whatever charting approach the dashboard already uses (Chart.js, etc.). Above assumes a <canvas> element with ID engineSeriesChart.

b) JS Logic to Call API & Render Chart

In dashboard/static/dashboard.js:

Add a function to call /api/engine_vs_series

Add chart rendering using the existing chart library

Example (assuming Chart.js is available):

let engineSeriesChart = null;

async function fetchEngineVsSeries() {
  const engine = document.getElementById("engineSelect").value;
  const series = document.getElementById("seriesSelect").value;
  const start = document.getElementById("startDate").value;
  const end = document.getElementById("endDate").value;
  const frequency = document.getElementById("frequencySelect").value;

  const params = new URLSearchParams({
    engine,
    series,
    frequency,
  });

  if (start) params.append("start", start);
  if (end) params.append("end", end);

  const resp = await fetch(`/api/engine_vs_series?${params.toString()}`);
  if (!resp.ok) {
    console.error("Failed to fetch engine vs series data");
    return;
  }

  const data = await resp.json();
  renderEngineSeriesChart(data);
}

function renderEngineSeriesChart(data) {
  const ctx = document.getElementById("engineSeriesChart").getContext("2d");
  const dates = data.dates || [];
  const bars = data.bars || [];
  const line = data.line || [];

  if (engineSeriesChart) {
    engineSeriesChart.destroy();
  }

  engineSeriesChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: dates,
      datasets: [
        {
          type: "bar",
          label: "Engine Score",
          data: bars,
          yAxisID: "yEngine",
          // keep colors consistent with PRISM palette
        },
        {
          type: "line",
          label: "Series Level",
          data: line,
          yAxisID: "ySeries",
          tension: 0.2,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        yEngine: {
          type: "linear",
          position: "left",
          title: {
            display: true,
            text: "Engine Score",
          },
          min: 0,
          max: 1,
        },
        ySeries: {
          type: "linear",
          position: "right",
          title: {
            display: true,
            text: "Series Level",
          },
          grid: {
            drawOnChartArea: false,
          },
        },
        x: {
          ticks: {
            maxTicksLimit: 15, // readability
          },
        },
      },
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          mode: "index",
          intersect: false,
        },
      },
    },
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("runEngineSeriesBtn");
  if (btn) {
    btn.addEventListener("click", fetchEngineVsSeries);
  }

  // Optionally: auto-run a default view on load
  // fetchEngineVsSeries();
});

3. Documentation / Readme

Update dashboard/README.md with a new section:

## Engine vs Series View

The dashboard now supports an **Engine vs Series** visualization:

- **Bars (left axis)** — output from a selected PRISM engine (e.g. Coherence, HMM Regime, PCA Instability).
- **Line (right axis)** — a selected indicator or indicator group (e.g. SP500, NASDAQ, Yield curve, Money supply).

This makes it easy to see how the engines behave around major events:
spikes in coherence, regime shifts, or instability alongside actual price or macro behavior.

### How to use

1. Start the dashboard:

   ```bash
   source ~/venvs/prism-mac-venv/bin/activate
   cd prism-engine/dashboard
   python app.py


Open the app in your browser (default):

http://127.0.0.1:5000 or the port you configured.

In the Engine vs Series card:

Choose an engine for bars.

Choose a series for the line.

Optionally set date range and frequency.

Click Run Engine vs Series.

A chart will render with:

Bars on the left y-axis for engine scores.

Line on the right y-axis for the data series.


---

## TESTING / ACCEPTANCE CRITERIA

Before merging, verify:

1. **API works**
   - `GET /api/engine_vs_series?engine=coherence&series=sp500` returns JSON with `dates`, `bars`, `line`.
   - Handles empty result ranges gracefully (returns empty arrays, no 500).

2. **Dashboard loads**
   - No JS errors in browser console.
   - “Engine vs Series” section renders with dropdowns + button.

3. **Chart renders**
   - Selecting `engine=coherence`, `series=sp500`, recent years → chart shows:
     - Bars that roughly spike around known events (e.g. March 2020).
     - Line that follows SP500 level.

4. **Performance**
   - Requests complete comfortably under a few seconds on normal ranges.
   - Very long date ranges still function, but user can always narrow via date selector.

5. **Non-regression**
   - Existing dashboard cards still work.
   - Diagnostics + `run_calibrated.py` remain functional.

---

## OPTIONAL NICE-TO-HAVES (If Claude Has Time)

- Populate `seriesSelect` from a **live endpoint** listing:
  - All `geometry_enabled` indicators
  - Families (spx, treasury, etc.)
- Add a toggle for **“Normalize series (z-score)”** for the line.
- Add a small **info tooltip**:
  - “Bars: engine score (left axis). Line: selected series (right axis).”

---

There you go — **full PR package** for #100. 🥂  

Copy this straight to Claude and let him wire it up. Once that chart is live, PRISM stops being “an engine” and starts being **a living instrument panel**.
::contentReference[oaicite:0]{index=0}
