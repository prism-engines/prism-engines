# Prism Engines
**Translating liquidity, macro, and behavioral momentum into a unified 3-D model of market equilibrium — a behavioral geometry of cycles revealing when forces align, diverge, or break.**

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![SQLite](https://img.shields.io/badge/Database-SQLite_v2-brightgreen)
![Domains](https://img.shields.io/badge/Systems-finance|climate|chemistry|biology|physics-purple)
![Status](https://img.shields.io/badge/Project-active-success)

---

## Overview

**Prism Engines** is a multi-domain analytics framework designed to evaluate structural behavior across systems such as:

- finance  
- climate  
- chemistry  
- biology  
- physics  

The engine ingests time-series indicators, processes them through 14 independent mathematical lenses, aggregates cross-lens consensus, and detects regime shifts and coherence events — all backed by a unified SQL architecture.

This README is a hybrid overview for contributors, researchers, and quant developers.

---

## Architecture (High-Level)

Prism’s analytics pipeline follows a simple, scalable flow:



systems → indicators → timeseries → windows → lenses → consensus → regime stability → events


This defines how raw inputs become ranked, geometric signals.

---

## SQL Schema v2 (Summary)

The SQL schema is structured into clear layers:

### Reference Tables
- `systems`
- `indicators`
- `lenses`

### Input Tables
- `timeseries`
- `data_quality_log`

### Output Tables
- `windows`
- `lens_results`
- `consensus`
- `regime_stability`
- `coherence_events`

### Derived Lineage
- `derived_indicator_lineage`

Full schema:  
`data/sql/prism_schema.sql`

---

## The 14 Prism Lenses

1. magnitude  
2. pca  
3. influence  
4. clustering  
5. decomposition  
6. granger  
7. dmd  
8. mutual_info  
9. network  
10. regime_switching  
11. anomaly  
12. transfer_entropy  
13. tda  
14. wavelet  

Each lens produces:
- raw score  
- normalized score  
- rank  

---

## Installation

```bash
git clone https://github.com/rudder-research/prism-engine.git
cd prism-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Quick Database Start
from data.sql.prism_db import init_db, add_indicator, write_dataframe

init_db()

add_indicator("SPY", system="finance", units="USD")
write_dataframe(df, "SPY", system="finance")

Running Prism Engines
python Start/prism_runner.py --system finance --step 12 --window 60


Window outputs are saved under:

data/outputs/windows/

Example: Finance Ingest
import yfinance as yf
from data.sql.prism_db import write_dataframe

df = yf.download("SPY").reset_index()[["Date", "Close"]]
df.columns = ["date", "value"]

write_dataframe(df, "SPY", system="finance", frequency="daily", source="Yahoo")

Example: Climate Ingest
import pandas as pd
from data.sql.prism_db import write_dataframe

df = pd.read_csv("nyc_temp.csv")  # or any API source
write_dataframe(df, "temperature_nyc", system="climate", units="celsius")
---

Used in Prism’s Stress Controls Block to stabilize transitions and suppress false positives.

AI Workflow (Claude Code + Agent Mode)

Prism Engines was built for AI co-development.

Use Agent Mode for:

running code

environment setup

debugging

git operations

directory management

Use Claude Code for:

major refactors

zip creation

schema migrations

heavy codegen

notebook → module conversion

Together they create a fast, dual-engine workflow.

Roadmap

 Multi-frequency & seasonal improvements

 Energy & climate domain expansion

 Wavelet coherence dashboards

 Automated PCA component selection

 MVSS calibration testbench

 Regime dampening module

 Prism Engines web dashboard

 Cloud deployment

 Documentation site

License

MIT (or your chosen license)

End of README

