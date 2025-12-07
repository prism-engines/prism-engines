// dashboard/static/dashboard.js

// =============================================================================
// Engine vs Series Chart
// =============================================================================

let engineSeriesChart = null;

async function fetchEngineVsSeries() {
  const engine = document.getElementById("engineSelect").value;
  const series = document.getElementById("seriesSelect").value;
  const start = document.getElementById("startDate").value;
  const end = document.getElementById("endDate").value;
  const frequency = document.getElementById("frequencySelect").value;
  const statusEl = document.getElementById("engine-series-status");
  const errorEl = document.getElementById("engine-series-error");
  const btn = document.getElementById("runEngineSeriesBtn");

  // Reset error display
  errorEl.style.display = "none";
  errorEl.textContent = "";

  // Update status
  statusEl.textContent = "Loading...";
  btn.disabled = true;

  const params = new URLSearchParams({
    engine,
    series,
    frequency,
  });

  if (start) params.append("start", start);
  if (end) params.append("end", end);

  try {
    const resp = await fetch(`/api/engine_vs_series?${params.toString()}`);
    const data = await resp.json();

    if (data.error) {
      errorEl.textContent = data.error;
      errorEl.style.display = "block";
      statusEl.textContent = "Error loading data.";
      btn.disabled = false;
      return;
    }

    if (!data.dates || data.dates.length === 0) {
      errorEl.textContent = "No data available for the selected parameters.";
      errorEl.style.display = "block";
      statusEl.textContent = "No data.";
      btn.disabled = false;
      return;
    }

    renderEngineSeriesChart(data);
    statusEl.textContent = `Loaded ${data.meta.n_points} data points (${data.meta.start} to ${data.meta.end})`;
  } catch (err) {
    console.error("Failed to fetch engine vs series data:", err);
    errorEl.textContent = `Request failed: ${err.message}`;
    errorEl.style.display = "block";
    statusEl.textContent = "Error.";
  } finally {
    btn.disabled = false;
  }
}

function renderEngineSeriesChart(data) {
  const ctx = document.getElementById("engineSeriesChart").getContext("2d");
  const dates = data.dates || [];
  const bars = data.bars || [];
  const line = data.line || [];
  const engineName = data.meta?.engine || "Engine";
  const seriesName = data.meta?.series || "Series";

  // Destroy existing chart
  if (engineSeriesChart) {
    engineSeriesChart.destroy();
  }

  // Subsample labels for readability if too many points
  const maxLabels = 20;
  const skipLabels = Math.ceil(dates.length / maxLabels);

  engineSeriesChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: dates,
      datasets: [
        {
          type: "bar",
          label: `${engineName.charAt(0).toUpperCase() + engineName.slice(1)} Score`,
          data: bars,
          yAxisID: "yEngine",
          backgroundColor: "rgba(32, 107, 196, 0.7)",
          borderColor: "rgba(32, 107, 196, 1)",
          borderWidth: 1,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
        },
        {
          type: "line",
          label: `${seriesName.toUpperCase()} Level`,
          data: line,
          yAxisID: "ySeries",
          borderColor: "rgba(255, 170, 0, 1)",
          backgroundColor: "rgba(255, 170, 0, 0.1)",
          tension: 0.2,
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 2,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      scales: {
        yEngine: {
          type: "linear",
          position: "left",
          title: {
            display: true,
            text: "Engine Score (0-1)",
            color: "rgba(32, 107, 196, 1)",
          },
          min: 0,
          max: 1,
          ticks: {
            color: "rgba(32, 107, 196, 1)",
          },
          grid: {
            color: "rgba(255, 255, 255, 0.1)",
          },
        },
        ySeries: {
          type: "linear",
          position: "right",
          title: {
            display: true,
            text: "Series Level",
            color: "rgba(255, 170, 0, 1)",
          },
          ticks: {
            color: "rgba(255, 170, 0, 1)",
          },
          grid: {
            drawOnChartArea: false,
          },
        },
        x: {
          ticks: {
            maxTicksLimit: 15,
            maxRotation: 45,
            minRotation: 0,
            autoSkip: true,
            color: "rgba(255, 255, 255, 0.7)",
          },
          grid: {
            color: "rgba(255, 255, 255, 0.05)",
          },
        },
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: "rgba(255, 255, 255, 0.8)",
          },
        },
        tooltip: {
          mode: "index",
          intersect: false,
          backgroundColor: "rgba(30, 41, 59, 0.95)",
          titleColor: "rgba(255, 255, 255, 0.9)",
          bodyColor: "rgba(255, 255, 255, 0.8)",
          borderColor: "rgba(255, 255, 255, 0.2)",
          borderWidth: 1,
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || "";
              if (label) {
                label += ": ";
              }
              if (context.parsed.y !== null) {
                if (context.datasetIndex === 0) {
                  // Engine score - format as percentage
                  label += (context.parsed.y * 100).toFixed(1) + "%";
                } else {
                  // Series value - format with commas
                  label += context.parsed.y.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  });
                }
              }
              return label;
            },
          },
        },
      },
    },
  });
}

// =============================================================================
// PRISM Controller Form
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Engine vs Series button handler
  const engineSeriesBtn = document.getElementById("runEngineSeriesBtn");
  if (engineSeriesBtn) {
    engineSeriesBtn.addEventListener("click", fetchEngineVsSeries);
  }

  // PRISM Controller form
  const form = document.getElementById("prism-form");
  const runButton = document.getElementById("run-button");
  const runStatus = document.getElementById("run-status");
  const outputJson = document.getElementById("output-json");
  const outputActions = document.getElementById("output-actions");

  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    runButton.disabled = true;
    runStatus.textContent = "Running...";
    outputActions.innerHTML = "";

    const formData = new FormData(form);

    const payload = {
      diagnostics: formData.get("diagnostics"),
      analysis: formData.get("analysis"),
      fetch_range: formData.get("fetch_range"),
      mlmeta: formData.get("mlmeta"),
      fetch_source: formData.getAll("fetch_source"),
      systems: formData.getAll("systems"),
    };

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      outputJson.textContent = JSON.stringify(data, null, 2);

      if (data.result && Array.isArray(data.result.actions)) {
        data.result.actions.forEach((action) => {
          const card = document.createElement("div");
          card.className = "card mb-3";

          const statusBadge = `<span class="badge ${
            action.status === "ok"
              ? "bg-green"
              : action.status === "skipped"
              ? "bg-azure"
              : action.status === "todo"
              ? "bg-yellow"
              : action.status === "warning"
              ? "bg-yellow"
              : "bg-red"
          }">${action.status}</span>`;

          card.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center">
              <span>${action.label || "action"}</span>
              ${statusBadge}
            </div>
            <div class="card-body">
              ${
                action.message
                  ? `<p class="mb-2 text-secondary">${escapeHtml(action.message)}</p>`
                  : ""
              }
              ${
                action.stdout
                  ? `<details class="mb-2"><summary>stdout</summary><pre style="max-height: 200px; overflow-y: auto;">${escapeHtml(
                      action.stdout
                    )}</pre></details>`
                  : ""
              }
              ${
                action.stderr
                  ? `<details class="mb-0"><summary>stderr</summary><pre style="max-height: 200px; overflow-y: auto;">${escapeHtml(
                      action.stderr
                    )}</pre></details>`
                  : ""
              }
            </div>
          `;
          outputActions.appendChild(card);
        });
      }

      runStatus.textContent = "Run complete.";
    } catch (err) {
      outputJson.textContent = `Request failed: ${err}`;
      runStatus.textContent = "Error.";
    } finally {
      runButton.disabled = false;
    }
  });

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
});
