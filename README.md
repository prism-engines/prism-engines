# 🚧 PRISM Engine Roadmap

*A forward-first view of ongoing development, challenges ahead, and open research problems.*

---

## 9. Fixed Broken Benchmark Signals  
- Inverted SNR ratios repaired  
- Leader variance corrected  
- Noise-only sets no longer falsely indicate structure  
- Rewrote benchmark generator with correct signal-to-noise architecture  


# 🔮 FUTURE WORK

Below is the **master list of work ahead**, organized into major engineering, research, and architecture categories.

These are the tasks that define PRISM’s **next-stage evolution**.

---

## **10. Multi-Normalization Stack (MNS v1.0)**

PRISM’s differentiator.
We need to build the system that runs each lens across *multiple* normalization geometries simultaneously.

Tasks:

* Implement multi-geometry runs
* Build comparison scoring system
* Define consensus-selection logic
* Add stability / robustness measurement
* Integrate with temporal & structural lenses

---

## **11. MVF Integration (Market Vector Framework v2)**

This includes the advanced geometric modeling you defined:

Tasks:

* Implement updated normalization weights
* Add Stress Controls Block
* Add MVSS metric (already defined conceptually)
* Add divergence & correlation anomaly detection
* Add regime confirmation dampener

---

## **12. Unified Data Panel Overhaul**

Consolidate all sources into a clean, metadata-aware panel builder.

Tasks:

* Standardize transforms per indicator type
* Finalize metadata schema
* Add cross-source cleaning rules
* Add NaN-handling engine
* Ensure consistency through all PRISM lenses

---

## **13. Diagnostic & Reporting Dashboards**

Needed for interpreting outcomes and performance.

Tasks:

* HTML / PDF report generation
* Temporal comparative lens plots
* Normalization geometry plots
* Benchmark summaries
* Lens-level stats & validation

---

## **14. Phase-Space Geometry Modules**

The long-term “geometric” heart of PRISM.

Tasks:

* Theta/phi decomposition
* Wave & harmonic modeling
* Divergence mapping
* Cross-lens geometric alignment
* Indicator vibration/oscillation modeling

---

## **15. Engine Execution Modes**

We need predictable run profiles.

Tasks:

* “Quick Mode” for intraday
* “Deep Mode” for overnight analysis
* Fix parallelization instability
* Add resource-aware scheduling
* Add deterministic run outputs

---

## **8. Reflective Error Handling**

PRISM should be self-diagnosing.

Tasks:

* Structured error categories
* Execution trace logs
* Auto-generated debugging notes
* Early misconfiguration warnings
* External environment consistency checks

---

## **9. Expanded Research Data Integration**

PRISM needs multi-domain capability.

Tasks:

* Climate data (ECMWF)
* Ecology data (GBIF, DataONE)
* Economic long-timescale datasets
* Universal metadata layer for external domains
* Domain-appropriate transforms & normalizers

---

## **8. PRISM Documentation & Identity**

For collaborators, research release, or publishing.

Tasks:

* Full documentation set
* Architecture diagrams
* Contributor onboarding
* Whitepaper / academic description
* Versioning policy docs

---

# ✔️ PAST PROBLEMS SOLVED

This section documents the **engineering and research milestones already completed**.
This serves as the retrospective progress ledger.

---

## ~~**7. Solved the One-Dimensional Normalization Problem~~**

Key accomplishments:

* Implemented lens-specific transforms & normalizers
* Added rolling, rank, robust, minmax, volatility normalization
* Built full LensNormalizer orchestrator
* Enabled type-aware default normalization based on indicator category
* Removed legacy forced z-score dependency

---

## ~~**6. Stabilized Stationarity & Transform Flow**~~

Fixed:

* Lenses receiving non-stationary data
* Mismatched transformation behavior across modules
* Introduced standardized: returns, diffs, bps, YoY, pct change
  Created stability across all lenses.

---

## ~~**5. Tamed Claude Code’s “Repo Rewrite Chaos”~~**

Major workflow improvements:

* Introduced DIAGNOSIS MODE (read-only)
* Introduced FIX MODE (diff-based, restricted)
* Removed multi-file cascading fixes
* Added controlled sandbox branches
* Prevented unwanted branch switching
* Claude now writes only after explicit approval

---

## ~~**4. Secured Main Branch with Versioned Snapshots**~~

Accomplished:

* Created and locked `Prism_v1.0`
* Added GitHub protection rules
* Established durable versioning strategy
* Main remains live, but releases are immutable
  Creates scientific reproducibility.

---

## ~~**3. Fixed DB & Fetcher Foundation Issues**~~

Resolved:

* Broken DB initialization
* Missing schema tables
* Path inconsistencies
* FRED & Yahoo fetcher issues
* Established unified DB resolution system

---

## ~~**2. Improved Temporal Engine Consistency**~~

Fixed:

* Misaligned temporal windows
* Parallelization overshooting
* Inconsistent step behavior
* Clarified engine runners and scripts

---

## ~~**1. Standardized Development Environments**~~

Resolved:

* Mac / Linux / Chromebook drift
* Conflicting Python versions
* Dependency mismatches
  Now converged on Python 3.12 and standardized path behavior.

---

# 📘 PRISM ENGINE — PROJECT OVERVIEW

*(Basic README content — placed intentionally at the bottom per your request)*

---

## 🎯 What Is PRISM?

PRISM is a multi-lens, geometric, normalization-aware analytical engine for financial and complex-system data.
Instead of relying on a single normalized dataset, PRISM studies:

* multiple **transforms**
* multiple **normalization geometries**
* multiple **independent analytical lenses**
* multiple **temporal resolutions**
* multiple **indicator categories**

…to reveal **structural coherence**, **regime shifts**, and **geometric alignment** in markets and other complex datasets.

---

## 🧱 Core Architecture

* **Lens System** — modular analytical components
* **LensNormalizer** — type-aware data preparation
* **Transforms** — stationarity enforcement
* **Normalizers** — geometry selection (zscore, rank, rolling, robust, etc.)
* **Benchmarks** — synthetic structured vs noise datasets
* **Temporal Analysis** — reversible, multi-resolution lens execution
* **Panel Builder** — unified, metadata-driven data ingestion
* **Versioning** — locked release branches (`Prism_v1.x`)

---

## 👥 Development Workflow

* Active development on `main`
* AI coding using `claude-sandbox`
* Release snapshots as `Prism_v1.x`
* Claude allowed to modify only after diffs approved
* Locked branches ensure reproducibility

---

## 🧭 Mission Statement

PRISM aims to translate *market structure, behavior, and oscillation patterns* into a unified geometric model — bridging macroeconomics, signals, normalization domains, and regime dynamics.

---
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

## Quickstart for Researchers (Clone → DB → Dashboard)

This project is designed so that an external user can:

1. Clone the repo
2. Create a virtual environment
3. Set API keys
4. Fetch data
5. Run the dashboard and explore the models

### 1. Clone the repository

```bash
git clone https://github.com/rudder-research/prism-engine.git
cd prism-engine
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set API keys (FRED, Tiingo, optional climate APIs)

```bash
export FRED_API_KEY="YOUR_FRED_KEY"
export TIINGO_API_KEY="YOUR_TIINGO_KEY"
# Optional climate keys / .cdsapirc can be configured separately
```

On macOS you can store these permanently in `~/.zshrc` as environment variables.

### 4. Initialize and populate the database

```bash
python start/update_all.py
```

This will:

- create the SQLite database (e.g. `~/prism_data/prism.db`)
- fetch economic series from FRED
- fetch market series from Tiingo
- register indicators and families for the geometry engines

You can inspect the result with:

```bash
python start/data_report.py
python start/check_indicator_health.py
python start/check_families.py
```

### 5. Run the dashboard

```bash
cd dashboard
python app.py
```

Then open the URL printed in the console (e.g. `http://127.0.0.1:5000`) in your browser.

From there you can:

- View the main **Dashboard**
- Open **Models & Geometry** for detailed descriptions
- Download `docs/models_overview.md` to review the math and data requirements offline

---

## Climate Module (Frozen)

A future standalone project. Present in repository but not active.
Please do not integrate, import, or modify the climate folder until the PRISM core engine is complete.

---

## License

MIT (or your chosen license)

---

End of README

