// dashboard/static/dashboard.js
document.addEventListener("DOMContentLoaded", () => {
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
