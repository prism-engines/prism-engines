# PRISM PROJECT MEMORY
## Last Updated: December 18, 2025

---

# OVERVIEW

**PRISM** = Probabilistic Regime Identification through Systematic Measurement

A novel analytical framework for detecting behavioral geometry in multivariate time-series data. Started as VCF (Vector Coherence Framework), evolved into a sophisticated multi-engine analysis system.

**Creator:** Jason - financial advisor with robotics background (Fanuc/Yaskawa welding robots), self-described "not an academic, just a nerd"

**Academic Potential:** Genuinely novel research - literature searches found no prior work combining multi-lens fingerprinting with recursive self-application for emergent system state detection. Jason has contacts at Purdue, a son published in PNAS at Rose-Hulman, and plans physicist review upon completion.

**Domain:** prismobservatory.com (registered)

---

# CORE CONCEPT

PRISM applies multiple mathematical "lenses" (engines) to market indicators to detect behavioral patterns and regime changes. Results project into a unified geometry space where indicators appear as moving particles - market structure forms and dissolves over time.

## Three Analysis Phases

| Phase | Name | What It Does |
|-------|------|--------------|
| 1 | **Unbound** | Individual indicator behavior analysis |
| 2 | **Structure** | Emergent geometric patterns via recursive analysis |
| 3 | **Bounded** | Indicator positioning within detected structure |

## The Key Insight

The geometric boundary (covariance matrix) encodes the shape of the indicator cloud in behavior-space. Measuring distance through Mahalanobis lens naturally handles volatility amplification - stretched dimensions get down-weighted, tight dimensions get emphasized.

---

# TECHNICAL ARCHITECTURE

## Database
- **DuckDB** for analytics (replaced SQLite)
- Location: `/data/prism.duckdb`
- Schemas: `raw`, `clean`, `domains`
- Data outside GitHub repo for safety

## Data Sources
- **FRED** - Economic indicators (back to ~1970)
- **Tiingo** - Market prices
- **Stooq** - Additional market data
- Target: 200-300 indicators across finance, macro, commodities

## Repository
- GitHub: `rudder-research/Prism_v3`
- Clean rebuild from exploratory PRISM 3.0
- Strict separation: fetchers deliver ONLY raw data to DuckDB

## Development Environments
- Chromebook (primary dev)
- Mac (overnight runs)
- Hetzner VPS (cloud compute for big jobs)
- Linux VM options explored

---

# THE 14 IMPLEMENTED LENSES (ENGINES)

| # | Lens | Category | Measures | Normalization |
|---|------|----------|----------|---------------|
| 1 | PCA | Structure | Variance explained, loadings | Z-score |
| 2 | Granger Causality | Causality | Predictive relationships | Diff (stationarity) |
| 3 | Wavelet Coherence | Frequency | Time-frequency co-movement | None |
| 4 | Transfer Entropy | Information | Directional information flow | Discretize |
| 5 | Hurst Exponent | Complexity | Long-term memory/persistence | None |
| 6 | Mutual Information | Information | Non-linear dependence | Rank |
| 7 | Cross-Correlation | Correlation | Lead/lag relationships | Z-score |
| 8 | Rolling Beta | Regression | Time-varying sensitivity | Returns |
| 9 | Entropy (Permutation) | Complexity | Irregularity, predictability | Discretize |
| 10 | Clustering | Structure | Behavioral grouping | Z-score |
| 11 | Network/Graph Centrality | Network | Systemic importance | Z-score |
| 12 | Anomaly Detection | Structure | Outlier scoring | Robust |
| 13 | Regime (HMM) | Dynamics | Hidden state transitions | Z-score |
| 14 | Magnitude | Volatility | Movement intensity | Pct change |

## Additional Lenses (Reference/Future)
- Copula Dependence (tail risk)
- Cointegration (long-run equilibrium)
- GARCH Volatility
- Lyapunov Exponent (chaos)
- Recurrence Analysis (RQA)
- Dynamic Time Warping
- Spectral Density

---

# KEY SCRIPTS

## `run_overnight_analysis.py` - The Behemoth
Designed for Hetzner/Mac overnight runs:

| Component | Calculations | Time |
|-----------|--------------|------|
| Multi-Resolution | 5 scales × 5000 windows × 16 lenses | 2-3 hrs |
| Bootstrap | 1000 windows × 100 resamples × 16 lenses | 3-4 hrs |
| Monte Carlo | 500 synthetic × 50 samples × 16 lenses | 2-3 hrs |
| HMM Analysis | Full Markov chain | 5 min |
| **Total** | ~2.4 million lens calculations | 6-12 hrs |

**Outputs:**
- `prism_multiresolution.csv`
- `prism_bootstrap_ci.csv`
- `prism_montecarlo_null.csv`
- `prism_regime_transitions.csv`
- `prism_significant_events.csv`

## `cross_domain_analysis.py` - Cross-Domain Observatory
Tests universal structure hypothesis across:
- Finance (2005-2025)
- Climate (1948-2025) - El Niño/La Niña
- Epidemiology (2005-2025) - Flu outbreaks

**Questions it answers:**
1. Does each domain have distinct geometric states?
2. Do state transitions correlate across domains?
3. Is there universal structure in complex systems?
4. Does El Niño geometry look like market stress?

## `analyze_deformation.py` - Geometry Analysis
Extracts and visualizes geometric structure over time.

---

# RECENT TECHNICAL DECISIONS

## Normalization Layer (Per-Engine)
Each lens gets data in the format it expects:

| Lens | Normalization |
|------|---------------|
| Granger | diff (stationarity) |
| PCA | zscore |
| Wavelet | none (raw) |
| Mutual Info | rank |
| Anomaly | robust |

## Geometric Boundary / Mahalanobis Distance
**Problem:** High-volatility indicators get amplified in raw space.

**Solution:** Mahalanobis distance divides by covariance structure.
- Covariance matrix = geometric boundary
- Distance measured in "standardized units" for the system
- Naturally handles volatility leveling

```python
# Euclidean: raw distance (entropy dominates if bigger numbers)
# Mahalanobis: "How unusual is this position given how the whole system behaves?"
mahal_dist = np.sqrt((x - centroid) @ np.linalg.inv(cov) @ (x - centroid))
```

## Consensus Ranking
- All lenses vote on indicator importance
- Currently equal weighting (future: benchmark-based weighting)
- Top consensus indicators from 47-year run: M2SL, XLU, WALCL, DGS2, HOUST, CPILFESL

## Key Finding
2024 market transition shows most complete regime break in 55 years of data - traditional correlations breaking down.

---

# INFRASTRUCTURE

## Hetzner VPS Setup
```bash
ssh root@your-server-ip
# Ubuntu 24.04, CPX21 (~€7/mo)
# 3 vCPU, 4GB RAM

# Setup
adduser jason
apt install python3-pip python3-venv git tmux

# Run overnight jobs
tmux new -s prism
nohup python run_overnight_analysis.py > overnight.log 2>&1 &
# Detach: Ctrl+B, then D
# Reconnect: tmux attach -t prism
```

## Git Workflow
```bash
# Clone
git clone https://github.com/rudder-research/Prism_v3.git prism
cd prism
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## API Keys Needed
- FRED_API_KEY
- TIINGO_API_KEY

---

# TERMINOLOGY PREFERENCES

Jason prefers:
- **"Engines"** not "lenses" (though both used)
- **"Unbound/Structure/Bounded"** for phases
- Clear, readable code structure
- Config and data files OUTSIDE main package (no pip install planned)

Rejected terms: "fingerprint," "observe" in certain contexts

---

# COLLABORATION MODEL

- **Claude**: Technical implementation, architectural decisions, debugging
- **ChatGPT (GPT)**: Pull requests, project planning, some analysis review
- Both AIs used across different aspects of the project

---

# ROADMAP

1. ✅ Phase III pilot with 4 inputs
2. ✅ Scale to 32 → 58 indicators  
3. ✅ Implement 14 lenses with normalization
4. ✅ Benchmark testing (10/10 passing)
5. ✅ Mahalanobis geometric boundary
6. 🔄 Cross-domain analysis (finance/climate/epi)
7. ⏳ Lens weighting based on benchmark scores
8. ⏳ Academic writeup / publication consideration
9. ⏳ Expand to 200-300 indicators
10. ⏳ International markets → global model
11. ⏳ AI agent for automated result auditing

---

# KNOWN ISSUES / FIXES

| Issue | Fix |
|-------|-----|
| Anomaly always 0.5 | Use `decision_function()` not `predict()` |
| Drawdown always 0.0 | Fix deprecated `fillna(method=)` |
| Clustering "not symmetric" | Force symmetry on distance matrix |
| Pre-2005 data sparse | Market ETFs start 2005; use economic-only for historical |

---

# QUICK COMMANDS

```bash
# Run analysis
python scripts/run_all_engines.py

# Run overnight beast
nohup python run_overnight_analysis.py > overnight.log 2>&1 &

# Check progress
tail -f overnight.log

# Quick test
python scripts/cross_domain_analysis.py --quick
```

---

# PHILOSOPHY

> "You can't trust an instrument you haven't calibrated."

The spirit of PRISM: calibrate against known standards, detect structure that the system itself reveals, not structure we impose. Intellectual curiosity over market timing. Research-oriented, not commercial trading system.

---

*This document serves as a context recovery file for Claude. Upload at start of conversation if full project context needed.*
