# PRISM Repository Structure

```
prism/
│
├── docs/                       # Documentation (you are here)
│   ├── REPO_STRUCTURE.md       # This file
│   ├── ARCHITECTURE.md         # System design
│   └── TERMINOLOGY.md          # Canonical terms
│
├── config/                     # Configuration (outside package)
│   ├── .env.example            # API keys template
│   └── settings.yaml           # Runtime settings
│
├── data/                       # Data storage (outside package)
│   └── prism.duckdb            # DuckDB database
│
├── scripts/                    # Entry points (outside package)
│   ├── fetch.py                # Run fetch operations
│   ├── clean.py                # Run cleaning (future)
│   └── run_phase.py            # Run analysis phases (future)
│
├── prism/                      # Main package
│   ├── __init__.py
│   │
│   ├── fetch/                  # Fetch system
│   │   ├── __init__.py
│   │   ├── base.py             # BaseFetcher contract
│   │   ├── fred.py             # FRED API fetcher
│   │   ├── tiingo.py           # Tiingo API fetcher
│   │   ├── stooq.py            # Stooq fetcher (future)
│   │   └── runner.py           # FetchRunner (ONLY DB writer)
│   │
│   ├── registry/               # Indicator definitions
│   │   ├── __init__.py
│   │   ├── indicators.yaml     # Unified indicator registry
│   │   └── loader.py           # Registry loader
│   │
│   ├── db/                     # Database layer
│   │   ├── __init__.py
│   │   ├── connection.py       # DuckDB connection factory
│   │   └── schema.sql          # Schema definitions
│   │
│   ├── clean/                  # Data cleaning (future)
│   │   ├── __init__.py
│   │   └── cleaner.py
│   │
│   ├── normalize/              # Normalization (future)
│   │   ├── __init__.py
│   │   └── normalizer.py
│   │
│   ├── engines/                # Math methods (future)
│   │   ├── __init__.py
│   │   ├── base.py             # BaseEngine contract
│   │   ├── pca.py
│   │   ├── granger.py
│   │   ├── hurst.py
│   │   └── ...
│   │
│   ├── phases/                 # Processing phases (future)
│   │   ├── __init__.py
│   │   ├── unbound.py          # Phase 1: Indicator Behavior Unbound
│   │   ├── structure.py        # Phase 2: System Geometry
│   │   └── bounded.py          # Phase 3: Indicator Behavior in Geometry
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       └── logging.py          # Logging configuration
│
├── tests/                      # Tests
│   ├── __init__.py
│   ├── test_fetch.py
│   └── test_registry.py
│
└── requirements.txt            # Dependencies
```

## Design Principles

### 1. Separation of Concerns

| Component | Responsibility | DB Access |
|-----------|----------------|-----------|
| Fetchers | Call APIs, return DataFrames | **NONE** |
| FetchRunner | Orchestrate fetches, write to DB | **WRITE** |
| Registry | Define indicators | **NONE** |
| DB Layer | Connection management | **READ/WRITE** |
| Scripts | CLI interface | **NONE** |

### 2. Data Immutability

Raw data is **never modified** after fetch. Transformations create new tables:

```
raw.indicators       ← Fetch writes here (immutable)
cleaned.indicators   ← Cleaning creates this
normalized.indicators← Normalization creates this
```

### 3. Single Writer Rule

**Only `FetchRunner` writes to the `raw` schema.**

Fetchers return DataFrames. They do not:
- Open database connections
- Write data
- Transform data
- Compute derived values

### 4. Outside-Package Separation

Things that **change per environment** live outside `prism/`:
- Configuration (`config/`)
- Data files (`data/`)
- Entry point scripts (`scripts/`)

The `prism/` package is pure logic.
