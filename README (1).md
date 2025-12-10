# PRISM Engine

**P**attern **R**ecognition via **I**ndependent **S**ignal **M**easurement

A multi-lens financial analysis framework that detects regime changes and coherence patterns by applying multiple independent mathematical perspectives to economic/market data.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/rudder-research/prism-engine.git
cd prism-engine
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 2. Configure API keys (create .env file)
echo "FRED_API_KEY=your_fred_key" >> .env
echo "TIINGO_API_KEY=your_tiingo_key" >> .env

# 3. Fetch data
python start/update_all.py

# 4. Run analysis
python start/analyze.py

# 5. (Optional) Compute lens weights
python start/lens_geometry.py --save

# 6. (Optional) Run weighted analysis
python start/analyze.py --weighted combined
```

## What It Does

PRISM applies 14 different mathematical "lenses" to financial data, each detecting different patterns:

| Category | Lenses | What They See |
|----------|--------|---------------|
| **Causality** | granger, transfer_entropy, influence | Who leads whom |
| **Structure** | pca, clustering, network | Correlation patterns |
| **Dynamics** | regime, wavelet, decomposition, dmd | Regime changes, cycles |
| **Information** | mutual_info | Nonlinear dependencies |
| **Outliers** | anomaly, magnitude | Stress, volatility |

### The Key Insight

Not all lenses are independent. Our geometry analysis found:
- 4 structure lenses correlate r > 0.8 (redundant)
- Granger is orthogonal to everything (unique)

**Unweighted consensus** over-counts structure 4x.  
**Weighted consensus** gives each *perspective* equal voice.

## Core Scripts (in `start/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `update_all.py` | Fetch data from FRED/Tiingo/Yahoo | Run daily/weekly |
| `analyze.py` | Main analysis runner | `--weighted combined` for balanced consensus |
| `lens_geometry.py` | Compute lens independence weights | `--save` to store weights |
| `compare_consensus.py` | Compare weighted vs unweighted | See methodology impact |
| `query_results.py` | Query historical runs | `--run N` for specific run |
| `full_monty.py` | Comprehensive analysis (Mac) | Heavy compute |
| `benchmark.py` | Validate lenses on synthetic data | 10/10 passing |

## Project Structure

```
prism-engine/
├── start/                  # Entry points (run scripts from here)
│   ├── analyze.py          # Main analysis
│   ├── lens_geometry.py    # Lens weight computation
│   ├── update_all.py       # Data fetching
│   └── prism_config.yaml   # Configuration
│
├── engine_core/
│   └── lenses/             # 14 analytical lenses
│       ├── granger_lens.py
│       ├── pca_lens.py
│       ├── regime_switching_lens.py
│       └── ...
│
├── data/
│   ├── sql/                # Database layer
│   │   └── db_connector.py # Main DB interface
│   ├── benchmark/          # Synthetic test data
│   └── registry/           # Indicator definitions
│
├── fetch/                  # Data fetchers (FRED, Tiingo, Yahoo)
└── analysis/               # Statistical utilities
```

## Configuration

Edit `start/prism_config.yaml`:

```yaml
data:
  start_date: "2000-01-01"
  min_coverage: 0.7  # Drop indicators with <70% data

analysis:
  lenses:
    - granger
    - regime
    - pca
    # ... etc
```

## Weighted Consensus

```bash
# Standard analysis (equal weights)
python start/analyze.py

# Weighted by lens independence
python start/analyze.py --weighted independence

# Weighted by cluster membership
python start/analyze.py --weighted cluster

# Combined approach (recommended)
python start/analyze.py --weighted combined
```

## Database

Results are stored in SQLite (`data/prism.db`):

```bash
# Query recent runs
python start/query_results.py

# Specific run details
python start/query_results.py --run 3

# Indicator history
python start/query_results.py --indicator uso
```

## Requirements

- Python 3.9+
- ~2GB RAM minimum (TDA lens needs more)
- API keys: FRED (free), Tiingo (free tier works)

## Versioning

- **v2.0** - Multi-lens framework, benchmarks passing
- **v2.1** - Lens geometry analysis, weight computation
- **v2.2** - Weighted consensus, methodology demo

## Theory

PRISM is based on the observation that financial regimes can be detected by measuring *coherence* across multiple independent analytical perspectives. When diverse mathematical methods (causality, structure, dynamics) agree, the signal is robust. When they disagree, the system may be in transition.

The lens geometry analysis ensures we don't over-count redundant perspectives. See `reports/benchmarks/` for validation against synthetic data with known ground truth.

## License

MIT

## Author

Jason Rudderman / Rudder Research
