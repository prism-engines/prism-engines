# PRISM Engine - AI Assistant Memory

> **Purpose**: Provide context for AI assistants (Claude, ChatGPT, etc.) working on this codebase.
> **Last Updated**: December 2025 (PR #11)
> **Maintainer**: Jason @ Rudder Research

---

## Project Identity

**PRISM Engine** (formerly VCF - Vector Coherence Framework) is a multi-domain analytics framework that evaluates structural behavior across time-series systems. The project was rebranded from VCF to PRISM to better reflect its evolved scope.

**Tagline**: "Translating liquidity, macro, and behavioral momentum into a unified 3-D model of market equilibrium — a behavioral geometry of cycles revealing when forces align, diverge, or break."

**Philosophy**: The creator (Jason) describes himself as "intellectually curious rather than market-focused" — "not an academic, not doing it to time the market, just a nerd." The project emphasizes:
- Continuous mathematical approaches over discrete classifications
- Removing "artificial thresholds and predetermined structures"
- Type-agnostic framework applicable across domains

---

## Core Architecture

### Pipeline Flow
```
systems → indicators → timeseries → windows → lenses → consensus → regime stability → events
```

### Key Concepts

| Term | Definition |
|------|------------|
| **System** | Top-level domain (finance, climate, chemistry, biology, physics, anthropology) |
| **Indicator** | A single time series (SPY, GDP, CPI, temperature, etc.) |
| **Panel** | Consolidated DataFrame with multiple indicators on common date index |
| **Lens** | Mathematical perspective that ranks indicators by importance |
| **Window** | Time period for rolling analysis |
| **Consensus** | Aggregated agreement across all lenses |
| **Coherence Event** | When normally disagreeing lenses suddenly align (the signal) |

### The 14 Analytical Lenses

1. **magnitude** - Raw signal strength
2. **pca** - Principal component contribution
3. **influence** - Cross-indicator influence
4. **clustering** - Grouping behavior
5. **decomposition** - Trend/seasonal/residual breakdown
6. **granger** - Granger causality
7. **dmd** - Dynamic Mode Decomposition
8. **mutual_info** - Mutual information
9. **network** - Network centrality
10. **regime_switching** - Markov regime detection
11. **anomaly** - Anomaly detection
12. **transfer_entropy** - Information transfer
13. **tda** - Topological Data Analysis
14. **wavelet** - Wavelet coherence

**Key Insight**: Mathematical disagreement between lenses is *normal*. The signal occurs when all lenses suddenly agree around significant events — this is the "Coherence Index" concept.

---

## Directory Structure

```
prism-engine/
├── cleaning/              # Data cleaning & normalization
│   ├── alignment.py       # Date alignment
│   ├── nan_strategies.py  # NaN handling
│   └── outlier_detection.py
├── data/
│   ├── raw/              # Fetched raw CSVs
│   ├── cleaned/          # Normalized data
│   ├── panels/           # Analysis-ready panels
│   ├── registry/         # JSON registries (market, economic, metric, system)
│   └── sql/              # SQLite schema, migrations, Python API
├── data_fetch/           # Data fetchers
│   ├── fetch_market_data.py   # Yahoo Finance
│   ├── fetch_economic_data.py # FRED
│   └── *_registry.json        # Source configs
├── engine/               # High-level engine wrappers
│   ├── prism_macro_engine.py
│   ├── prism_market_engine.py
│   ├── prism_stress_engine.py
│   └── prism_ml_engine.py
├── engine_core/          # Core analysis logic
│   ├── lenses/           # 14 lens implementations
│   │   └── base_lens.py  # Abstract base class
│   └── orchestration/    # Lens coordination
│       ├── consensus.py  # Borda count, score fusion, voting
│       ├── temporal_analysis.py
│       └── temporal_runner.py
├── fetch/                # Source-specific fetchers
│   ├── fetcher_fred.py
│   ├── fetcher_yahoo.py
│   └── fetcher_climate.py
├── interpretation/       # AI interpretation layer
│   ├── ai_context.py
│   ├── interpreter.py
│   └── prompt_templates.py
├── panel/                # Panel construction
│   ├── build_panel.py
│   ├── transforms_econ.py  # YoY, MoM transforms
│   ├── transforms_market.py
│   └── validators.py
├── output/
│   └── latest/           # Most recent run outputs
├── start/                # Entry points
│   ├── temporal_runner.py
│   └── fetcher.py
├── tests/                # Test suite
├── utils/                # Shared utilities
│   ├── db_connector.py   # SQLite connection
│   ├── db_manager.py
│   ├── registry.py       # Registry loader
│   └── date_cleaner.py
├── validation/           # Statistical validation
│   ├── backtester.py
│   ├── bootstrap_analysis.py
│   └── permutation_tests.py
├── visualization/
│   └── plotters/         # Temporal plots, heatmaps
├── doc/                  # PR documentation
│   ├── PR1_*.md through PR6_*.md
│   └── reports/          # Analysis outputs, Streamlit app
└── docs/architecture/    # Technical documentation
```

---

## Data Sources

### Market Data (Yahoo Finance via yfinance)
- **Equities**: SPY, QQQ, IWM
- **Volatility**: VIX
- **Bonds**: TLT, IEF, SHY, BND, TIP, LQD, HYG
- **Commodities**: GLD, SLV, USO
- **Currency**: DXY
- **Sectors**: XLU (utilities - Jason holds 20% position)

### Economic Data (FRED)
- **Interest Rates**: DGS10, DGS2, DGS3MO, T10Y2Y, T10Y3M
- **Inflation**: CPIAUCSL, CPILFESL, PPIACO
- **Labor**: UNRATE, PAYEMS
- **Production**: INDPRO
- **Housing**: HOUST, PERMIT
- **Monetary**: M2SL, WALCL (Fed balance sheet)
- **Financial Conditions**: NFCI, ANFCI

### Data Notes
- VCF/PRISM data starts ~1970 due to source limitations
- Stooq is being evaluated as alternative to yfinance
- Total Return Index (TRI) calculation methodology established
- Economic data often monthly/quarterly; market data daily

---

## Database Schema (SQLite v2)

### Reference Tables
- `systems` - Valid domains (finance, climate, etc.)
- `indicators` - Indicator metadata
- `lenses` - Available analytical lenses

### Input Tables
- `timeseries` - Raw observations
- `data_quality_log` - Data quality tracking

### Output Tables
- `windows` - Analysis time periods
- `lens_results` - Per-lens rankings
- `consensus` - Aggregated rankings
- `regime_stability` - Transition metrics
- `coherence_events` - Detected signals

### Key API Functions (from `data/sql/prism_db.py`)
```python
from data.sql.prism_db import init_db, add_indicator, write_dataframe, load_indicator

init_db()
add_indicator("SPY", system="finance", units="USD")
write_dataframe(df, "SPY", system="finance")
data = load_indicator("SPY")
```

---

## Development Workflow

### AI Collaboration Model
- **Claude**: Used for pull requests and project planning
- **ChatGPT**: Also collaborates on PRs
- **Agent Mode**: For running code, debugging, git ops
- **Claude Code**: For major refactors, zip creation, schema migrations

### Environment
- **Primary**: Google Colab + Google Drive (GitHub abandoned for simplicity)
- **Python**: 3.12+
- **Database**: SQLite

### Running Analysis
```python
# Quick start
from start.temporal_runner import quick_start
results, summary = quick_start(panel_clean)

# Full control
from start.temporal_runner import run_temporal_analysis, generate_all_plots
results = run_temporal_analysis(panel, profile='standard')
generate_all_plots(results, output_dir='./plots')
```

### Performance Profiles
- **chromebook**: 3 fast lenses, 2-month steps, streaming mode
- **standard**: 4 lenses, 1-month steps
- **powerful**: 5 lenses, 0.5-month steps

---

## Phase III Pilot Run

Currently testing with 4 specific inputs:
1. S&P 500 50d-200d moving average
2. 10-Year Treasury yield (DGS10)
3. DXY (US Dollar Index)
4. AGG (bond ETF)

**Goal**: Validate framework before full implementation over next few weeks.

---

## Key Findings to Date

- **2024 Regime Break**: Most complete regime break in dataset — only 0.02 Spearman correlation to 2023 rankings
- **55 Annual Periods**: 1970-2025 analyzed
- **Top Indicators** (recent run): M2SL, XLU, WALCL, DGS2, HOUST, CPILFESL, PERMIT, PAYEMS, NFCI, CPIAUCSL

---

## PR History

| PR | Title | Focus |
|----|-------|-------|
| 1 | Foundation & Folder Layout | Initial structure, remove VCF references |
| 2 | Registry System | JSON registries as "quarterback" |
| 3 | Fetch & Normalization Layer | Yahoo + FRED fetchers |
| 4 | DB Schema & Migrations | SQLite v2 schema |
| 5 | Panel Builder & Column Selection | Panel construction |
| 6 | Engine Integration | Connect engines to registries |
| 7+ | Lenses, Orchestration, Reports | Current work |

---

## Future Roadmap

1. **Phase III Completion**: Pilot with 4 inputs
2. **Full US Markets**: Expand indicator set
3. **International Markets**: Global expansion
4. **Global Model**: Cross-market coherence
5. **Natural Systems**: Climate, biology, chemistry applications
6. **Physicist Review**: PNAS-published physicist to review outcomes

---

## Common Tasks for AI Assistants

### Adding a New Lens
1. Create `engine_core/lenses/new_lens.py`
2. Inherit from `BaseLens`
3. Implement `analyze()` and `rank_indicators()`
4. Register in `engine_core/lenses/__init__.py`

### Adding a New Indicator
1. Add to appropriate registry (`market_registry.json` or `economic_registry.json`)
2. Run fetcher to pull data
3. Verify in `data/raw/`

### Running Full Analysis
```bash
cd prism-engine
python -m start.temporal_runner
```

### Database Operations
```python
from data.sql.prism_db import get_connection, init_db

with get_connection() as conn:
    # Execute queries
    pass
```

---

## Style Preferences

- **Code**: Python with type hints preferred
- **Docs**: Markdown
- **Analysis**: Prefer continuous math over discrete thresholds
- **Naming**: Lowercase with underscores for files, CamelCase for classes

---

## Contact

- **Creator**: Jason
- **Organization**: Rudder Research
- **Repo**: github.com/rudder-research/prism-engine

---

*This document should be updated with each significant PR to maintain accurate context for AI assistants.*
