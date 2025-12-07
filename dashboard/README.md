# PRISM Dashboard

This folder provides the interactive PRISM dashboard:

- Flask backend (`app.py`)
- Tabler UI (`templates/`)
- UI controls for diagnostics, fetch, analysis, ML/Meta, systems
- **Engine vs Series** visualization
- Output viewer

## Running the Dashboard

```bash
cd dashboard
python app.py
```

Or with custom host/port:

```bash
python app.py --host 0.0.0.0 --port 8080
```

Default: http://127.0.0.1:5000

---

## Engine vs Series View

The dashboard supports an **Engine vs Series** visualization:

- **Bars (left axis)** — output from a selected PRISM engine (e.g. Coherence, HMM Regime, PCA Instability).
- **Line (right axis)** — a selected indicator or indicator group (e.g. SP500, NASDAQ, Yield curve, Money supply).

This makes it easy to see how the engines behave around major events:
spikes in coherence, regime shifts, or instability alongside actual price or macro behavior.

### Available Engines

| Engine | Description |
|--------|-------------|
| `coherence` | Rolling mean pairwise correlation across all indicators (coherence proxy) |
| `hmm_regime` | Volatility regime score based on rolling percentile of volatility |
| `pca_instability` | Rolling PCA stability - change in PC1 explained variance |
| `dispersion` | Cross-sectional dispersion (std across indicators) |
| `meta` / `composite` | Average of all engine scores |

### Available Series

Pre-mapped series for easy selection:
- `sp500` - S&P 500 Index
- `nasdaq` - NASDAQ Composite
- `t10y2y` - Treasury Yield Curve (10Y-2Y spread)
- `vix` - VIX Volatility Index
- `m2sl` - Money Supply M2
- `dgs10` - 10-Year Treasury Rate
- `dgs2` - 2-Year Treasury Rate

Additional indicators from the database are also available.

### How to Use

1. Start the dashboard:
   ```bash
   cd prism-engine/dashboard
   python app.py
   ```

2. Open in your browser: http://127.0.0.1:5000

3. In the **Engine vs Series** card:
   - Choose an engine for bars
   - Choose a series for the line
   - Optionally set date range and frequency
   - Click **Run Engine vs Series**

4. A chart will render with:
   - Bars on the left y-axis for engine scores (0-1)
   - Line on the right y-axis for the data series

### API Endpoints

#### `GET /api/engine_vs_series`

Returns engine output (bars) and selected series (line) on a common time axis.

**Query Parameters:**
- `engine`: coherence | hmm_regime | pca_instability | dispersion | meta | composite
- `series`: indicator id (e.g., sp500, vix, t10y2y)
- `start`: YYYY-MM-DD (optional)
- `end`: YYYY-MM-DD (optional)
- `frequency`: daily | weekly | monthly (optional, default: daily)

**Response:**
```json
{
  "dates": ["2020-01-02", "2020-01-03", ...],
  "bars": [0.45, 0.52, ...],
  "line": [3245.12, 3256.78, ...],
  "meta": {
    "engine": "coherence",
    "series": "sp500",
    "start": "2020-01-02",
    "end": "2024-12-06",
    "frequency": "daily",
    "n_points": 1200
  }
}
```

#### `GET /api/available_series`

Returns list of available series for the dropdown.

**Response:**
```json
{
  "series": [
    {"id": "sp500", "name": "SP500", "indicator_id": "sp500_d"},
    {"id": "vix", "name": "VIX", "indicator_id": "vix_d"},
    ...
  ]
}
```

---

## PRISM Controller

The main controller provides UI controls for:
- Diagnostics (full, db, fetch, lenses, engines)
- Analysis (coherence, correlation, calibrated, regime, 20yr, 40yr)
- Data fetch (FRED, Tiingo)
- ML/Meta modes
- System/domain selection
