# Prism Engine – Complete Data Pipeline Implementation Specification

**Version**: 2.0  
**Purpose**: Implementation-ready specification for Claude Code  
**Architecture**: Registry-driven, compute-on-demand, single-value storage

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Folder Structure](#2-folder-structure)
3. [Registry System](#3-registry-system)
4. [Database Schema](#4-database-schema)
5. [Fetch Pipeline](#5-fetch-pipeline)
6. [Panel Builder](#6-panel-builder)
7. [Transform System](#7-transform-system)
8. [Error Handling](#8-error-handling)
9. [Testing Requirements](#9-testing-requirements)
10. [Migration Plan](#10-migration-plan)
11. [Implementation Phases](#11-implementation-phases)

---

## 1. Design Principles

### 1.1 Core Philosophy

- **Single source of truth**: Registry files control all behavior
- **Compute-on-demand**: Store raw values only; compute transforms at panel build time
- **Fail loudly**: Validation failures halt processing with clear error messages
- **Incremental by default**: Fetch only new data unless full refresh requested
- **Frequency-aware**: Explicit handling of daily vs monthly data alignment

### 1.2 Why Compute-on-Demand?

Instead of storing 200+ computed columns per series:

```
# BAD: Stored columns explosion
value_raw, value_z, value_yoy, value_mom, value_log, value_pct, value_tri...
```

We store ONE column and compute transforms when building panels:

```
# GOOD: Single stored value
value (raw)

# Transforms computed at panel build time based on registry config
```

**Benefits**:
- Simpler schema that never needs migration for new transforms
- No stale computed values when raw data is corrected
- Registry controls which transforms each engine needs
- Adding new series = adding registry entry, not schema change

---

## 2. Folder Structure

```
prism-engine/
│
├── config/
│   ├── system_registry.json      # Master config: paths, DB, logging
│   ├── market_registry.json      # Market series definitions
│   ├── economic_registry.json    # Economic series definitions
│   └── transform_registry.json   # Available transforms and their params
│
├── fetch/
│   ├── __init__.py
│   ├── base_fetcher.py           # Abstract base class for all fetchers
│   ├── market_fetcher.py         # Yahoo Finance / market data
│   ├── fred_fetcher.py           # FRED API
│   ├── bls_fetcher.py            # BLS API (future)
│   └── validators.py             # Fetch validation functions
│
├── db/
│   ├── __init__.py
│   ├── connector.py              # SQLite connection manager
│   ├── models.py                 # Table definitions (SQLAlchemy or raw SQL)
│   ├── queries.py                # Common query functions
│   └── migrations/
│       ├── 001_initial_schema.sql
│       ├── 002_create_market_tables.sql
│       ├── 003_create_economic_tables.sql
│       ├── 004_create_metadata_tables.sql
│       └── 005_create_fetch_log_table.sql
│
├── panel/
│   ├── __init__.py
│   ├── builder.py                # Main panel assembly logic
│   ├── frequency.py              # Frequency alignment functions
│   └── output.py                 # Panel export functions
│
├── transforms/
│   ├── __init__.py
│   ├── base.py                   # Transform base class
│   ├── returns.py                # pct_change, log_return
│   ├── growth.py                 # yoy, mom, annualized
│   ├── normalize.py              # z_score, min_max, rank
│   ├── smooth.py                 # rolling_mean, ewma
│   └── registry.py               # Transform dispatcher
│
├── utils/
│   ├── __init__.py
│   ├── date_utils.py             # Date parsing, normalization
│   ├── numeric_utils.py          # Number cleaning, validation
│   └── logging_utils.py          # Structured logging
│
├── engines/
│   ├── __init__.py
│   ├── base_engine.py            # Abstract engine class
│   ├── macro_engine.py
│   ├── market_engine.py
│   ├── stress_engine.py
│   └── ml_engine.py
│
├── data/
│   ├── prism.db                  # SQLite database
│   └── output/
│       └── panels/               # Generated panel CSVs
│
├── logs/
│   ├── fetch.log
│   ├── panel.log
│   └── error.log
│
├── tests/
│   ├── test_fetch/
│   ├── test_db/
│   ├── test_panel/
│   ├── test_transforms/
│   └── fixtures/
│
├── scripts/
│   ├── run_fetch.py              # CLI for fetching
│   ├── run_panel.py              # CLI for panel building
│   ├── run_migrate.py            # CLI for DB migrations
│   └── run_validate.py           # CLI for data validation
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── .env.example                  # API keys template
```

---

## 3. Registry System

### 3.1 system_registry.json

```json
{
  "version": "2.0",
  "project_name": "prism-engine",
  
  "paths": {
    "database": "data/prism.db",
    "output_dir": "data/output/panels",
    "log_dir": "logs",
    "migrations_dir": "db/migrations"
  },
  
  "registries": {
    "market": "config/market_registry.json",
    "economic": "config/economic_registry.json",
    "transforms": "config/transform_registry.json"
  },
  
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    "rotate_mb": 10,
    "backup_count": 5
  },
  
  "fetch": {
    "default_mode": "incremental",
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "timeout_seconds": 30,
    "quarantine_on_error": true
  },
  
  "panel": {
    "default_frequency": "monthly",
    "default_align_method": "end_of_month",
    "min_history_years": 10,
    "required_coverage_pct": 80
  }
}
```

### 3.2 market_registry.json

```json
{
  "version": "2.0",
  "description": "Market data series definitions",
  "default_source": "yahoo",
  
  "series": {
    "SPX": {
      "enabled": true,
      "ticker": "^GSPC",
      "name": "S&P 500 Index",
      "source": "yahoo",
      "frequency": "daily",
      "fields": ["open", "high", "low", "close", "volume", "adj_close"],
      "primary_field": "adj_close",
      "start_date": "1970-01-01",
      "transforms_available": ["pct_change", "log_return", "z_score", "yoy", "rolling_vol"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        },
        "market_engine": {
          "use_transform": "pct_change",
          "params": {"periods": 1}
        },
        "stress_engine": {
          "use_transform": "rolling_vol",
          "params": {"window": 21}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_daily_change_pct": 25
      },
      "metadata": {
        "category": "equity_index",
        "region": "US",
        "currency": "USD"
      }
    },
    
    "XLU": {
      "enabled": true,
      "ticker": "XLU",
      "name": "Utilities Select Sector SPDR",
      "source": "yahoo",
      "frequency": "daily",
      "fields": ["open", "high", "low", "close", "volume", "adj_close"],
      "primary_field": "adj_close",
      "start_date": "1998-12-22",
      "transforms_available": ["pct_change", "log_return", "z_score", "yoy", "relative_strength"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_daily_change_pct": 20
      },
      "metadata": {
        "category": "sector_etf",
        "sector": "utilities",
        "region": "US",
        "currency": "USD"
      }
    },
    
    "TLT": {
      "enabled": true,
      "ticker": "TLT",
      "name": "iShares 20+ Year Treasury Bond ETF",
      "source": "yahoo",
      "frequency": "daily",
      "fields": ["open", "high", "low", "close", "volume", "adj_close"],
      "primary_field": "adj_close",
      "start_date": "2002-07-30",
      "transforms_available": ["pct_change", "log_return", "z_score", "yoy"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_daily_change_pct": 15
      },
      "metadata": {
        "category": "bond_etf",
        "duration": "long",
        "region": "US",
        "currency": "USD"
      }
    },
    
    "GLD": {
      "enabled": true,
      "ticker": "GLD",
      "name": "SPDR Gold Shares",
      "source": "yahoo",
      "frequency": "daily",
      "fields": ["open", "high", "low", "close", "volume", "adj_close"],
      "primary_field": "adj_close",
      "start_date": "2004-11-18",
      "transforms_available": ["pct_change", "log_return", "z_score", "yoy"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_daily_change_pct": 15
      },
      "metadata": {
        "category": "commodity_etf",
        "commodity": "gold",
        "region": "US",
        "currency": "USD"
      }
    },
    
    "DXY": {
      "enabled": true,
      "ticker": "DX-Y.NYB",
      "name": "US Dollar Index",
      "source": "yahoo",
      "frequency": "daily",
      "fields": ["open", "high", "low", "close", "adj_close"],
      "primary_field": "adj_close",
      "start_date": "1970-01-01",
      "transforms_available": ["pct_change", "log_return", "z_score", "yoy"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 50,
        "max_value": 200,
        "max_daily_change_pct": 5
      },
      "metadata": {
        "category": "currency_index",
        "region": "US",
        "currency": "USD"
      }
    },
    
    "VIX": {
      "enabled": true,
      "ticker": "^VIX",
      "name": "CBOE Volatility Index",
      "source": "yahoo",
      "frequency": "daily",
      "fields": ["open", "high", "low", "close"],
      "primary_field": "close",
      "start_date": "1990-01-02",
      "transforms_available": ["z_score", "percentile_rank", "rolling_mean"],
      "engine_config": {
        "stress_engine": {
          "use_transform": "percentile_rank",
          "params": {"window": 252}
        }
      },
      "validation": {
        "min_value": 5,
        "max_value": 100,
        "max_daily_change_pct": 50
      },
      "metadata": {
        "category": "volatility_index",
        "region": "US"
      }
    }
  }
}
```

### 3.3 economic_registry.json

```json
{
  "version": "2.0",
  "description": "Economic data series definitions",
  "default_source": "fred",
  
  "series": {
    "GDP": {
      "enabled": true,
      "series_id": "GDP",
      "name": "Gross Domestic Product",
      "source": "fred",
      "frequency": "quarterly",
      "units": "billions_usd",
      "seasonal_adjustment": "saar",
      "start_date": "1947-01-01",
      "revision_behavior": "revised",
      "revision_lag_months": 3,
      "transforms_available": ["yoy", "qoq", "z_score", "log"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_period_change_pct": 20
      },
      "metadata": {
        "category": "output",
        "subcategory": "gdp",
        "region": "US"
      }
    },
    
    "UNRATE": {
      "enabled": true,
      "series_id": "UNRATE",
      "name": "Unemployment Rate",
      "source": "fred",
      "frequency": "monthly",
      "units": "percent",
      "seasonal_adjustment": "sa",
      "start_date": "1948-01-01",
      "revision_behavior": "minimal",
      "revision_lag_months": 1,
      "transforms_available": ["diff", "yoy_diff", "z_score", "ma"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy_diff",
          "params": {}
        },
        "stress_engine": {
          "use_transform": "diff",
          "params": {"periods": 3}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": 30,
        "max_period_change_pct": 50
      },
      "metadata": {
        "category": "labor",
        "subcategory": "unemployment",
        "region": "US"
      }
    },
    
    "CPIAUCSL": {
      "enabled": true,
      "series_id": "CPIAUCSL",
      "name": "Consumer Price Index for All Urban Consumers",
      "source": "fred",
      "frequency": "monthly",
      "units": "index_1982_84_100",
      "seasonal_adjustment": "sa",
      "start_date": "1947-01-01",
      "revision_behavior": "revised",
      "revision_lag_months": 2,
      "transforms_available": ["yoy", "mom", "z_score", "log"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_period_change_pct": 5
      },
      "metadata": {
        "category": "prices",
        "subcategory": "cpi",
        "region": "US"
      }
    },
    
    "FEDFUNDS": {
      "enabled": true,
      "series_id": "FEDFUNDS",
      "name": "Federal Funds Effective Rate",
      "source": "fred",
      "frequency": "monthly",
      "units": "percent",
      "seasonal_adjustment": "nsa",
      "start_date": "1954-07-01",
      "revision_behavior": "minimal",
      "revision_lag_months": 0,
      "transforms_available": ["diff", "yoy_diff", "z_score", "level"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "level",
          "params": {}
        },
        "stress_engine": {
          "use_transform": "diff",
          "params": {"periods": 1}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": 25,
        "max_period_change_pct": null
      },
      "metadata": {
        "category": "interest_rates",
        "subcategory": "policy",
        "region": "US"
      }
    },
    
    "GS10": {
      "enabled": true,
      "series_id": "GS10",
      "name": "10-Year Treasury Constant Maturity Rate",
      "source": "fred",
      "frequency": "monthly",
      "units": "percent",
      "seasonal_adjustment": "nsa",
      "start_date": "1953-04-01",
      "revision_behavior": "none",
      "revision_lag_months": 0,
      "transforms_available": ["diff", "yoy_diff", "z_score", "level"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "level",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": 20,
        "max_period_change_pct": null
      },
      "metadata": {
        "category": "interest_rates",
        "subcategory": "treasury",
        "region": "US"
      }
    },
    
    "T10Y2Y": {
      "enabled": true,
      "series_id": "T10Y2Y",
      "name": "10-Year Treasury Minus 2-Year Treasury",
      "source": "fred",
      "frequency": "daily",
      "units": "percent",
      "seasonal_adjustment": "nsa",
      "start_date": "1976-06-01",
      "revision_behavior": "none",
      "revision_lag_months": 0,
      "transforms_available": ["level", "z_score", "ma", "sign"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "level",
          "params": {}
        },
        "stress_engine": {
          "use_transform": "sign",
          "params": {}
        }
      },
      "validation": {
        "min_value": -5,
        "max_value": 5,
        "max_period_change_pct": null
      },
      "metadata": {
        "category": "interest_rates",
        "subcategory": "spread",
        "region": "US"
      }
    },
    
    "INDPRO": {
      "enabled": true,
      "series_id": "INDPRO",
      "name": "Industrial Production Index",
      "source": "fred",
      "frequency": "monthly",
      "units": "index_2017_100",
      "seasonal_adjustment": "sa",
      "start_date": "1919-01-01",
      "revision_behavior": "revised",
      "revision_lag_months": 2,
      "transforms_available": ["yoy", "mom", "z_score", "log"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_period_change_pct": 20
      },
      "metadata": {
        "category": "output",
        "subcategory": "industrial",
        "region": "US"
      }
    },
    
    "PERMIT": {
      "enabled": true,
      "series_id": "PERMIT",
      "name": "New Private Housing Units Authorized by Building Permits",
      "source": "fred",
      "frequency": "monthly",
      "units": "thousands",
      "seasonal_adjustment": "saar",
      "start_date": "1960-01-01",
      "revision_behavior": "revised",
      "revision_lag_months": 2,
      "transforms_available": ["yoy", "mom", "z_score", "log"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_period_change_pct": 50
      },
      "metadata": {
        "category": "housing",
        "subcategory": "permits",
        "region": "US"
      }
    },
    
    "UMCSENT": {
      "enabled": true,
      "series_id": "UMCSENT",
      "name": "University of Michigan Consumer Sentiment",
      "source": "fred",
      "frequency": "monthly",
      "units": "index_1966_q1_100",
      "seasonal_adjustment": "nsa",
      "start_date": "1952-11-01",
      "revision_behavior": "revised",
      "revision_lag_months": 1,
      "transforms_available": ["yoy", "diff", "z_score", "level"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "z_score",
          "params": {"window": null}
        }
      },
      "validation": {
        "min_value": 40,
        "max_value": 120,
        "max_period_change_pct": 30
      },
      "metadata": {
        "category": "sentiment",
        "subcategory": "consumer",
        "region": "US"
      }
    },
    
    "DTWEXBGS": {
      "enabled": true,
      "series_id": "DTWEXBGS",
      "name": "Trade Weighted U.S. Dollar Index: Broad, Goods and Services",
      "source": "fred",
      "frequency": "daily",
      "units": "index_jan_2006_100",
      "seasonal_adjustment": "nsa",
      "start_date": "2006-01-02",
      "revision_behavior": "none",
      "revision_lag_months": 0,
      "transforms_available": ["yoy", "pct_change", "z_score", "log"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 50,
        "max_value": 200,
        "max_period_change_pct": 10
      },
      "metadata": {
        "category": "currency",
        "subcategory": "trade_weighted",
        "region": "US"
      }
    },
    
    "M2SL": {
      "enabled": true,
      "series_id": "M2SL",
      "name": "M2 Money Stock",
      "source": "fred",
      "frequency": "monthly",
      "units": "billions_usd",
      "seasonal_adjustment": "sa",
      "start_date": "1959-01-01",
      "revision_behavior": "revised",
      "revision_lag_months": 2,
      "transforms_available": ["yoy", "mom", "z_score", "log"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_period_change_pct": 30
      },
      "metadata": {
        "category": "money",
        "subcategory": "m2",
        "region": "US"
      }
    },
    
    "PPIACO": {
      "enabled": true,
      "series_id": "PPIACO",
      "name": "Producer Price Index: All Commodities",
      "source": "fred",
      "frequency": "monthly",
      "units": "index_1982_100",
      "seasonal_adjustment": "nsa",
      "start_date": "1913-01-01",
      "revision_behavior": "revised",
      "revision_lag_months": 2,
      "transforms_available": ["yoy", "mom", "z_score", "log"],
      "engine_config": {
        "macro_engine": {
          "use_transform": "yoy",
          "params": {}
        }
      },
      "validation": {
        "min_value": 0,
        "max_value": null,
        "max_period_change_pct": 20
      },
      "metadata": {
        "category": "prices",
        "subcategory": "ppi",
        "region": "US"
      }
    }
  }
}
```

### 3.4 transform_registry.json

```json
{
  "version": "2.0",
  "description": "Available transforms and their parameters",
  
  "transforms": {
    "level": {
      "description": "Raw value, no transformation",
      "function": "transforms.base.level",
      "params": {},
      "requires_history": false
    },
    
    "pct_change": {
      "description": "Percentage change over N periods",
      "function": "transforms.returns.pct_change",
      "params": {
        "periods": {"type": "int", "default": 1, "min": 1, "max": 252}
      },
      "requires_history": true,
      "min_history_periods": 2
    },
    
    "log_return": {
      "description": "Log return over N periods",
      "function": "transforms.returns.log_return",
      "params": {
        "periods": {"type": "int", "default": 1, "min": 1, "max": 252}
      },
      "requires_history": true,
      "min_history_periods": 2
    },
    
    "yoy": {
      "description": "Year-over-year percentage change",
      "function": "transforms.growth.yoy",
      "params": {},
      "requires_history": true,
      "min_history_periods": 12,
      "frequency_dependent": true,
      "frequency_map": {
        "daily": 252,
        "monthly": 12,
        "quarterly": 4
      }
    },
    
    "yoy_diff": {
      "description": "Year-over-year difference (for rates)",
      "function": "transforms.growth.yoy_diff",
      "params": {},
      "requires_history": true,
      "min_history_periods": 12,
      "frequency_dependent": true,
      "frequency_map": {
        "daily": 252,
        "monthly": 12,
        "quarterly": 4
      }
    },
    
    "mom": {
      "description": "Month-over-month percentage change",
      "function": "transforms.growth.mom",
      "params": {},
      "requires_history": true,
      "min_history_periods": 2,
      "frequency_dependent": true,
      "frequency_map": {
        "daily": 21,
        "monthly": 1,
        "quarterly": null
      }
    },
    
    "qoq": {
      "description": "Quarter-over-quarter percentage change",
      "function": "transforms.growth.qoq",
      "params": {},
      "requires_history": true,
      "min_history_periods": 2,
      "allowed_frequencies": ["quarterly"]
    },
    
    "diff": {
      "description": "Simple difference over N periods",
      "function": "transforms.growth.diff",
      "params": {
        "periods": {"type": "int", "default": 1, "min": 1, "max": 12}
      },
      "requires_history": true,
      "min_history_periods": 2
    },
    
    "z_score": {
      "description": "Z-score normalization",
      "function": "transforms.normalize.z_score",
      "params": {
        "window": {"type": "int", "default": null, "min": 20, "max": null}
      },
      "requires_history": true,
      "min_history_periods": 20,
      "notes": "window=null uses full history"
    },
    
    "percentile_rank": {
      "description": "Percentile rank within rolling window",
      "function": "transforms.normalize.percentile_rank",
      "params": {
        "window": {"type": "int", "default": 252, "min": 20, "max": null}
      },
      "requires_history": true,
      "min_history_periods": 20
    },
    
    "min_max": {
      "description": "Min-max scaling to 0-1 range",
      "function": "transforms.normalize.min_max",
      "params": {
        "window": {"type": "int", "default": null, "min": 20, "max": null}
      },
      "requires_history": true,
      "min_history_periods": 20
    },
    
    "rolling_mean": {
      "description": "Rolling mean (moving average)",
      "function": "transforms.smooth.rolling_mean",
      "params": {
        "window": {"type": "int", "default": 20, "min": 2, "max": 252}
      },
      "requires_history": true,
      "min_history_periods": 2
    },
    
    "ewma": {
      "description": "Exponentially weighted moving average",
      "function": "transforms.smooth.ewma",
      "params": {
        "span": {"type": "int", "default": 20, "min": 2, "max": 252}
      },
      "requires_history": true,
      "min_history_periods": 2
    },
    
    "rolling_vol": {
      "description": "Rolling volatility (standard deviation of returns)",
      "function": "transforms.smooth.rolling_vol",
      "params": {
        "window": {"type": "int", "default": 21, "min": 5, "max": 252},
        "annualize": {"type": "bool", "default": true}
      },
      "requires_history": true,
      "min_history_periods": 5
    },
    
    "log": {
      "description": "Natural logarithm",
      "function": "transforms.base.log",
      "params": {},
      "requires_history": false,
      "notes": "Requires positive values"
    },
    
    "sign": {
      "description": "Sign of value (-1, 0, 1)",
      "function": "transforms.base.sign",
      "params": {},
      "requires_history": false
    },
    
    "ma": {
      "description": "Alias for rolling_mean",
      "function": "transforms.smooth.rolling_mean",
      "params": {
        "window": {"type": "int", "default": 12, "min": 2, "max": 60}
      },
      "requires_history": true,
      "min_history_periods": 2
    },
    
    "relative_strength": {
      "description": "Ratio relative to benchmark (requires benchmark_series param)",
      "function": "transforms.base.relative_strength",
      "params": {
        "benchmark_series": {"type": "str", "default": "SPX", "required": true}
      },
      "requires_history": false,
      "requires_additional_series": true
    }
  }
}
```

---

## 4. Database Schema

### 4.1 Design Philosophy

- **Single value column** for each observation (no pre-computed transforms)
- **Separate tables** for market vs economic data
- **Metadata tables** for series information
- **Fetch log table** for audit trail and incremental fetch support
- **Quarantine table** for failed/suspicious data

### 4.2 Migration 001: Initial Schema

```sql
-- migrations/001_initial_schema.sql

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Create schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

INSERT INTO schema_version (version, description) 
VALUES (1, 'Initial schema creation');
```

### 4.3 Migration 002: Market Tables

```sql
-- migrations/002_create_market_tables.sql

-- Market price data (daily)
CREATE TABLE IF NOT EXISTS market_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(series_id, date)
);

-- Indexes for market_prices
CREATE INDEX IF NOT EXISTS idx_market_prices_series_date 
ON market_prices(series_id, date);

CREATE INDEX IF NOT EXISTS idx_market_prices_date 
ON market_prices(date);

-- Trigger to update updated_at
CREATE TRIGGER IF NOT EXISTS trg_market_prices_updated
AFTER UPDATE ON market_prices
BEGIN
    UPDATE market_prices 
    SET updated_at = datetime('now') 
    WHERE id = NEW.id;
END;

INSERT INTO schema_version (version, description) 
VALUES (2, 'Create market tables');
```

### 4.4 Migration 003: Economic Tables

```sql
-- migrations/003_create_economic_tables.sql

-- Economic data (various frequencies)
CREATE TABLE IF NOT EXISTS economic_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL NOT NULL,
    vintage_date TEXT,  -- When this value was published (for revision tracking)
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(series_id, date, vintage_date)
);

-- Indexes for economic_values
CREATE INDEX IF NOT EXISTS idx_economic_values_series_date 
ON economic_values(series_id, date);

CREATE INDEX IF NOT EXISTS idx_economic_values_date 
ON economic_values(date);

CREATE INDEX IF NOT EXISTS idx_economic_values_vintage 
ON economic_values(series_id, vintage_date);

-- Trigger to update updated_at
CREATE TRIGGER IF NOT EXISTS trg_economic_values_updated
AFTER UPDATE ON economic_values
BEGIN
    UPDATE economic_values 
    SET updated_at = datetime('now') 
    WHERE id = NEW.id;
END;

INSERT INTO schema_version (version, description) 
VALUES (3, 'Create economic tables');
```

### 4.5 Migration 004: Metadata Tables

```sql
-- migrations/004_create_metadata_tables.sql

-- Series metadata (cached from registry + computed stats)
CREATE TABLE IF NOT EXISTS series_metadata (
    series_id TEXT PRIMARY KEY,
    series_type TEXT NOT NULL CHECK(series_type IN ('market', 'economic')),
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    frequency TEXT NOT NULL,
    start_date TEXT,
    end_date TEXT,
    row_count INTEGER,
    last_fetched_at TEXT,
    last_value REAL,
    last_value_date TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Trigger to update updated_at
CREATE TRIGGER IF NOT EXISTS trg_series_metadata_updated
AFTER UPDATE ON series_metadata
BEGIN
    UPDATE series_metadata 
    SET updated_at = datetime('now') 
    WHERE series_id = NEW.series_id;
END;

INSERT INTO schema_version (version, description) 
VALUES (4, 'Create metadata tables');
```

### 4.6 Migration 005: Fetch Log Table

```sql
-- migrations/005_create_fetch_log_table.sql

-- Fetch operation log (for audit and incremental support)
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    series_type TEXT NOT NULL CHECK(series_type IN ('market', 'economic')),
    fetch_mode TEXT NOT NULL CHECK(fetch_mode IN ('full', 'incremental')),
    fetch_started_at TEXT NOT NULL,
    fetch_completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'success', 'failed', 'partial')),
    rows_fetched INTEGER DEFAULT 0,
    rows_inserted INTEGER DEFAULT 0,
    rows_updated INTEGER DEFAULT 0,
    rows_quarantined INTEGER DEFAULT 0,
    date_range_start TEXT,
    date_range_end TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Index for fetch_log lookups
CREATE INDEX IF NOT EXISTS idx_fetch_log_series 
ON fetch_log(series_id, fetch_started_at DESC);

-- Quarantine table for failed/suspicious data
CREATE TABLE IF NOT EXISTS quarantine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    series_type TEXT NOT NULL,
    date TEXT NOT NULL,
    raw_data TEXT NOT NULL,  -- JSON of original row
    reason TEXT NOT NULL,
    fetch_log_id INTEGER,
    reviewed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (fetch_log_id) REFERENCES fetch_log(id)
);

CREATE INDEX IF NOT EXISTS idx_quarantine_series 
ON quarantine(series_id, created_at DESC);

INSERT INTO schema_version (version, description) 
VALUES (5, 'Create fetch log and quarantine tables');
```

---

## 5. Fetch Pipeline

### 5.1 Base Fetcher Class

```python
# fetch/base_fetcher.py
"""
Abstract base class for all data fetchers.
Enforces consistent interface and validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

from utils.date_utils import normalize_date, validate_date_format
from utils.numeric_utils import clean_numeric
from db.connector import DatabaseConnector


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    series_id: str
    status: str  # 'success', 'failed', 'partial'
    rows_fetched: int
    rows_inserted: int
    rows_updated: int
    rows_quarantined: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    error_message: Optional[str]
    duration_seconds: float


class BaseFetcher(ABC):
    """
    Abstract base class for data fetchers.
    
    All fetchers must implement:
    - fetch_raw(): Get raw data from source
    - validate_row(): Validate a single row
    
    Base class provides:
    - Normalization pipeline
    - Database insertion
    - Logging
    - Error handling
    """
    
    def __init__(
        self,
        db: DatabaseConnector,
        series_config: Dict[str, Any],
        system_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        self.db = db
        self.series_config = series_config
        self.system_config = system_config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Extract common config
        self.series_id = series_config.get('series_id') or series_config.get('ticker')
        self.series_type = 'market' if 'ticker' in series_config else 'economic'
        self.validation_rules = series_config.get('validation', {})
        
    @abstractmethod
    def fetch_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Fetch raw data from source.
        Must return DataFrame with at minimum: date column + value column(s)
        """
        pass
    
    @abstractmethod
    def validate_row(self, row: pd.Series, prev_row: Optional[pd.Series] = None) -> tuple[bool, str]:
        """
        Validate a single row of data.
        Returns: (is_valid, reason_if_invalid)
        """
        pass
    
    def fetch(
        self,
        mode: str = 'incremental',
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> FetchResult:
        """
        Main fetch entry point.
        
        Args:
            mode: 'incremental' or 'full'
            start_date: Override start date
            end_date: Override end date (default: today)
        """
        fetch_start = datetime.now()
        self.logger.info(f"Starting {mode} fetch for {self.series_id}")
        
        # Determine date range
        if mode == 'incremental' and start_date is None:
            start_date = self._get_last_date()
            if start_date:
                # Start from day after last date
                start_date = start_date + pd.Timedelta(days=1)
        
        if start_date is None:
            start_date = pd.to_datetime(
                self.series_config.get('start_date', '1970-01-01')
            ).date()
        
        if end_date is None:
            end_date = date.today()
        
        try:
            # Step 1: Fetch raw data
            self.logger.info(f"Fetching {start_date} to {end_date}")
            raw_df = self.fetch_raw(start_date, end_date)
            
            if raw_df is None or len(raw_df) == 0:
                self.logger.info("No new data available")
                return FetchResult(
                    series_id=self.series_id,
                    status='success',
                    rows_fetched=0,
                    rows_inserted=0,
                    rows_updated=0,
                    rows_quarantined=0,
                    date_range_start=str(start_date),
                    date_range_end=str(end_date),
                    error_message=None,
                    duration_seconds=(datetime.now() - fetch_start).total_seconds()
                )
            
            rows_fetched = len(raw_df)
            self.logger.info(f"Fetched {rows_fetched} rows")
            
            # Step 2: Normalize
            normalized_df = self._normalize(raw_df)
            
            # Step 3: Validate
            valid_df, quarantine_df = self._validate(normalized_df)
            
            # Step 4: Insert to database
            rows_inserted, rows_updated = self._insert(valid_df)
            
            # Step 5: Quarantine invalid rows
            rows_quarantined = self._quarantine(quarantine_df)
            
            # Step 6: Update metadata
            self._update_metadata()
            
            status = 'success' if rows_quarantined == 0 else 'partial'
            
            result = FetchResult(
                series_id=self.series_id,
                status=status,
                rows_fetched=rows_fetched,
                rows_inserted=rows_inserted,
                rows_updated=rows_updated,
                rows_quarantined=rows_quarantined,
                date_range_start=str(valid_df['date'].min()) if len(valid_df) > 0 else None,
                date_range_end=str(valid_df['date'].max()) if len(valid_df) > 0 else None,
                error_message=None,
                duration_seconds=(datetime.now() - fetch_start).total_seconds()
            )
            
            self._log_fetch(result, mode)
            return result
            
        except Exception as e:
            self.logger.error(f"Fetch failed: {str(e)}", exc_info=True)
            result = FetchResult(
                series_id=self.series_id,
                status='failed',
                rows_fetched=0,
                rows_inserted=0,
                rows_updated=0,
                rows_quarantined=0,
                date_range_start=str(start_date),
                date_range_end=str(end_date),
                error_message=str(e),
                duration_seconds=(datetime.now() - fetch_start).total_seconds()
            )
            self._log_fetch(result, mode)
            raise
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard normalization:
        1. Normalize dates to YYYY-MM-DD
        2. Normalize column names (lowercase, underscores)
        3. Remove duplicate dates
        4. Convert numerics
        5. Sort by date
        6. Drop invalid rows
        """
        df = df.copy()
        
        # Normalize date column
        date_col = self._find_date_column(df)
        df['date'] = df[date_col].apply(normalize_date)
        df = df.dropna(subset=['date'])
        
        # Normalize column names
        df.columns = [self._normalize_column_name(c) for c in df.columns]
        
        # Remove duplicates (keep last)
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Convert numeric columns
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if col != 'date':
                df[col] = df[col].apply(clean_numeric)
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate each row, separating valid from invalid.
        """
        valid_rows = []
        quarantine_rows = []
        
        prev_row = None
        for idx, row in df.iterrows():
            is_valid, reason = self.validate_row(row, prev_row)
            
            if is_valid:
                valid_rows.append(row)
                prev_row = row
            else:
                quarantine_rows.append({
                    'row': row,
                    'reason': reason
                })
                self.logger.warning(
                    f"Row failed validation: date={row.get('date')}, reason={reason}"
                )
        
        valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame()
        quarantine_df = pd.DataFrame(quarantine_rows) if quarantine_rows else pd.DataFrame()
        
        return valid_df, quarantine_df
    
    def _insert(self, df: pd.DataFrame) -> tuple[int, int]:
        """Insert validated data to database. Returns (inserted, updated)."""
        if len(df) == 0:
            return 0, 0
        
        # Implementation depends on series_type
        # Subclasses may override for custom behavior
        raise NotImplementedError("Subclass must implement _insert")
    
    def _quarantine(self, quarantine_df: pd.DataFrame) -> int:
        """Insert invalid rows to quarantine table."""
        if len(quarantine_df) == 0:
            return 0
        
        import json
        count = 0
        for _, item in quarantine_df.iterrows():
            row_data = item['row'].to_dict()
            self.db.execute(
                """
                INSERT INTO quarantine (series_id, series_type, date, raw_data, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.series_id,
                    self.series_type,
                    row_data.get('date', 'unknown'),
                    json.dumps(row_data, default=str),
                    item['reason']
                )
            )
            count += 1
        
        return count
    
    def _get_last_date(self) -> Optional[date]:
        """Get the last date we have data for this series."""
        table = 'market_prices' if self.series_type == 'market' else 'economic_values'
        result = self.db.fetch_one(
            f"SELECT MAX(date) FROM {table} WHERE series_id = ?",
            (self.series_id,)
        )
        if result and result[0]:
            return pd.to_datetime(result[0]).date()
        return None
    
    def _update_metadata(self):
        """Update series_metadata table."""
        table = 'market_prices' if self.series_type == 'market' else 'economic_values'
        value_col = 'adj_close' if self.series_type == 'market' else 'value'
        
        stats = self.db.fetch_one(
            f"""
            SELECT 
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as row_count
            FROM {table}
            WHERE series_id = ?
            """,
            (self.series_id,)
        )
        
        last = self.db.fetch_one(
            f"""
            SELECT date, {value_col}
            FROM {table}
            WHERE series_id = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (self.series_id,)
        )
        
        self.db.execute(
            """
            INSERT INTO series_metadata 
                (series_id, series_type, name, source, frequency, 
                 start_date, end_date, row_count, last_fetched_at, 
                 last_value_date, last_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)
            ON CONFLICT(series_id) DO UPDATE SET
                start_date = excluded.start_date,
                end_date = excluded.end_date,
                row_count = excluded.row_count,
                last_fetched_at = excluded.last_fetched_at,
                last_value_date = excluded.last_value_date,
                last_value = excluded.last_value
            """,
            (
                self.series_id,
                self.series_type,
                self.series_config.get('name', self.series_id),
                self.series_config.get('source', 'unknown'),
                self.series_config.get('frequency', 'unknown'),
                stats[0] if stats else None,
                stats[1] if stats else None,
                stats[2] if stats else 0,
                last[0] if last else None,
                last[1] if last else None
            )
        )
    
    def _log_fetch(self, result: FetchResult, mode: str):
        """Log fetch operation to fetch_log table."""
        self.db.execute(
            """
            INSERT INTO fetch_log 
                (series_id, series_type, fetch_mode, fetch_started_at, 
                 fetch_completed_at, status, rows_fetched, rows_inserted,
                 rows_updated, rows_quarantined, date_range_start, 
                 date_range_end, error_message)
            VALUES (?, ?, ?, datetime('now', ?), datetime('now'), 
                    ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.series_id,
                self.series_type,
                mode,
                f'-{result.duration_seconds} seconds',
                result.status,
                result.rows_fetched,
                result.rows_inserted,
                result.rows_updated,
                result.rows_quarantined,
                result.date_range_start,
                result.date_range_end,
                result.error_message
            )
        )
    
    @staticmethod
    def _find_date_column(df: pd.DataFrame) -> str:
        """Find the date column in a DataFrame."""
        candidates = ['date', 'Date', 'DATE', 'datetime', 'Datetime', 'timestamp']
        for c in candidates:
            if c in df.columns:
                return c
        # Try to find any column with 'date' in name
        for c in df.columns:
            if 'date' in c.lower():
                return c
        # Fall back to index if it looks like dates
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            return df.columns[0]
        raise ValueError("Could not identify date column")
    
    @staticmethod
    def _normalize_column_name(name: str) -> str:
        """Normalize column name: lowercase, replace spaces with underscores."""
        return str(name).lower().strip().replace(' ', '_').replace('-', '_')
```

### 5.2 Market Fetcher Implementation

```python
# fetch/market_fetcher.py
"""
Market data fetcher using yfinance.
"""

from datetime import date
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf
import logging

from fetch.base_fetcher import BaseFetcher, FetchResult
from db.connector import DatabaseConnector


class MarketFetcher(BaseFetcher):
    """Fetcher for market data (equities, ETFs, indices)."""
    
    def __init__(
        self,
        db: DatabaseConnector,
        series_config: Dict[str, Any],
        system_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(db, series_config, system_config, logger)
        self.ticker = series_config['ticker']
        self.series_id = series_config.get('series_id', self.ticker.replace('^', '').replace('-', '_'))
    
    def fetch_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        
        ticker = yf.Ticker(self.ticker)
        
        # yfinance uses string dates
        start_str = start_date.strftime('%Y-%m-%d') if start_date else '1970-01-01'
        end_str = end_date.strftime('%Y-%m-%d') if end_date else None
        
        df = ticker.history(start=start_str, end=end_str, auto_adjust=False)
        
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        # Reset index to get date as column
        df = df.reset_index()
        
        # Rename columns to standard names
        column_map = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_map)
        
        # Keep only columns we need
        keep_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[[c for c in keep_cols if c in df.columns]]
        
        return df
    
    def validate_row(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series] = None
    ) -> tuple[bool, str]:
        """Validate a market data row."""
        
        # Check for required fields
        if pd.isna(row.get('close')) and pd.isna(row.get('adj_close')):
            return False, "Missing close price"
        
        # Check for valid date
        if pd.isna(row.get('date')):
            return False, "Missing date"
        
        # Check min value
        min_val = self.validation_rules.get('min_value')
        price = row.get('adj_close') or row.get('close')
        if min_val is not None and price < min_val:
            return False, f"Price {price} below minimum {min_val}"
        
        # Check max value
        max_val = self.validation_rules.get('max_value')
        if max_val is not None and price > max_val:
            return False, f"Price {price} above maximum {max_val}"
        
        # Check daily change limit
        max_change = self.validation_rules.get('max_daily_change_pct')
        if max_change is not None and prev_row is not None:
            prev_price = prev_row.get('adj_close') or prev_row.get('close')
            if prev_price and prev_price > 0:
                pct_change = abs((price - prev_price) / prev_price * 100)
                if pct_change > max_change:
                    return False, f"Daily change {pct_change:.1f}% exceeds limit {max_change}%"
        
        # Check OHLC consistency
        o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
        if all(pd.notna([o, h, l, c])):
            if not (l <= o <= h and l <= c <= h):
                return False, f"OHLC inconsistency: O={o}, H={h}, L={l}, C={c}"
        
        return True, ""
    
    def _insert(self, df: pd.DataFrame) -> tuple[int, int]:
        """Insert market data to database."""
        if len(df) == 0:
            return 0, 0
        
        inserted = 0
        updated = 0
        
        for _, row in df.iterrows():
            # Try insert, on conflict update
            result = self.db.execute(
                """
                INSERT INTO market_prices 
                    (series_id, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(series_id, date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    adj_close = excluded.adj_close,
                    volume = excluded.volume,
                    updated_at = datetime('now')
                """,
                (
                    self.series_id,
                    row['date'],
                    row.get('open'),
                    row.get('high'),
                    row.get('low'),
                    row.get('close'),
                    row.get('adj_close'),
                    row.get('volume')
                )
            )
            
            # SQLite doesn't easily distinguish insert vs update
            # We'll count based on rowcount
            if result.rowcount > 0:
                inserted += 1  # Simplified - actual logic would check
        
        self.db.commit()
        return inserted, updated
```

### 5.3 FRED Fetcher Implementation

```python
# fetch/fred_fetcher.py
"""
Economic data fetcher using FRED API.
"""

from datetime import date
from typing import Optional, Dict, Any
import pandas as pd
import requests
import os
import logging

from fetch.base_fetcher import BaseFetcher
from db.connector import DatabaseConnector


class FREDFetcher(BaseFetcher):
    """Fetcher for FRED economic data."""
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def __init__(
        self,
        db: DatabaseConnector,
        series_config: Dict[str, Any],
        system_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(db, series_config, system_config, logger)
        self.series_id = series_config['series_id']
        self.api_key = os.environ.get('FRED_API_KEY')
        
        if not self.api_key:
            raise ValueError("FRED_API_KEY environment variable not set")
    
    def fetch_raw(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Fetch data from FRED API."""
        
        params = {
            'series_id': self.series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date.strftime('%Y-%m-%d') if start_date else '1900-01-01',
            'observation_end': end_date.strftime('%Y-%m-%d') if end_date else None,
            'sort_order': 'asc'
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=self.system_config.get('fetch', {}).get('timeout_seconds', 30)
        )
        response.raise_for_status()
        
        data = response.json()
        
        if 'observations' not in data:
            self.logger.warning(f"No observations in response: {data}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['observations'])
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # FRED returns 'date' and 'value' columns
        # Value can be '.' for missing
        df = df[['date', 'value']]
        df['value'] = df['value'].replace('.', pd.NA)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        return df
    
    def validate_row(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series] = None
    ) -> tuple[bool, str]:
        """Validate an economic data row."""
        
        # Check for required fields
        if pd.isna(row.get('value')):
            return False, "Missing value"
        
        if pd.isna(row.get('date')):
            return False, "Missing date"
        
        value = row['value']
        
        # Check min value
        min_val = self.validation_rules.get('min_value')
        if min_val is not None and value < min_val:
            return False, f"Value {value} below minimum {min_val}"
        
        # Check max value
        max_val = self.validation_rules.get('max_value')
        if max_val is not None and value > max_val:
            return False, f"Value {value} above maximum {max_val}"
        
        # Check period change limit
        max_change = self.validation_rules.get('max_period_change_pct')
        if max_change is not None and prev_row is not None:
            prev_value = prev_row.get('value')
            if prev_value and prev_value != 0:
                pct_change = abs((value - prev_value) / prev_value * 100)
                if pct_change > max_change:
                    return False, f"Period change {pct_change:.1f}% exceeds limit {max_change}%"
        
        return True, ""
    
    def _insert(self, df: pd.DataFrame) -> tuple[int, int]:
        """Insert economic data to database."""
        if len(df) == 0:
            return 0, 0
        
        inserted = 0
        updated = 0
        today = date.today().isoformat()
        
        for _, row in df.iterrows():
            result = self.db.execute(
                """
                INSERT INTO economic_values 
                    (series_id, date, value, vintage_date)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(series_id, date, vintage_date) DO UPDATE SET
                    value = excluded.value,
                    updated_at = datetime('now')
                """,
                (
                    self.series_id,
                    row['date'],
                    row['value'],
                    today  # Current vintage
                )
            )
            
            if result.rowcount > 0:
                inserted += 1
        
        self.db.commit()
        return inserted, updated
```

### 5.4 Utility Functions

```python
# utils/date_utils.py
"""Date parsing and normalization utilities."""

from datetime import datetime, date
from typing import Optional, Union
import re


def normalize_date(value: Union[str, datetime, date, None]) -> Optional[str]:
    """
    Normalize any date input to YYYY-MM-DD string format.
    
    Handles:
    - datetime objects
    - date objects  
    - ISO format strings
    - US format (MM/DD/YYYY)
    - 2-digit years
    - Excel serial dates (integers)
    
    Returns None for unparseable inputs.
    """
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d')
    
    if isinstance(value, date):
        return value.strftime('%Y-%m-%d')
    
    if isinstance(value, (int, float)):
        # Excel serial date (days since 1899-12-30)
        try:
            from datetime import timedelta
            excel_epoch = datetime(1899, 12, 30)
            dt = excel_epoch + timedelta(days=int(value))
            return dt.strftime('%Y-%m-%d')
        except:
            return None
    
    if not isinstance(value, str):
        return None
    
    value = str(value).strip()
    
    # Try common formats
    formats = [
        '%Y-%m-%d',      # ISO
        '%Y/%m/%d',      # ISO with slashes
        '%m/%d/%Y',      # US
        '%m-%d-%Y',      # US with dashes
        '%d/%m/%Y',      # EU
        '%d-%m-%Y',      # EU with dashes
        '%Y%m%d',        # Compact
        '%m/%d/%y',      # US 2-digit year
        '%d/%m/%y',      # EU 2-digit year
        '%Y-%m-%dT%H:%M:%S',  # ISO with time
        '%Y-%m-%d %H:%M:%S',  # Datetime
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            # Handle 2-digit years (assume 1950-2049 range)
            if dt.year < 50:
                dt = dt.replace(year=dt.year + 2000)
            elif dt.year < 100:
                dt = dt.replace(year=dt.year + 1900)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # Try pandas as fallback
    try:
        import pandas as pd
        dt = pd.to_datetime(value)
        return dt.strftime('%Y-%m-%d')
    except:
        pass
    
    return None


def validate_date_format(value: str) -> bool:
    """Check if string is in YYYY-MM-DD format."""
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(pattern, value))


def get_end_of_month(dt: Union[str, date, datetime]) -> str:
    """Get the last day of the month for a given date."""
    import calendar
    
    if isinstance(dt, str):
        dt = datetime.strptime(dt, '%Y-%m-%d')
    
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return f"{dt.year}-{dt.month:02d}-{last_day:02d}"


def get_end_of_quarter(dt: Union[str, date, datetime]) -> str:
    """Get the last day of the quarter for a given date."""
    if isinstance(dt, str):
        dt = datetime.strptime(dt, '%Y-%m-%d')
    
    quarter = (dt.month - 1) // 3 + 1
    quarter_end_month = quarter * 3
    
    import calendar
    last_day = calendar.monthrange(dt.year, quarter_end_month)[1]
    return f"{dt.year}-{quarter_end_month:02d}-{last_day:02d}"
```

```python
# utils/numeric_utils.py
"""Numeric cleaning and validation utilities."""

from typing import Union, Optional
import re
import math


def clean_numeric(value: Union[str, int, float, None]) -> Optional[float]:
    """
    Clean and convert a value to float.
    
    Handles:
    - Already numeric values
    - Strings with commas
    - Strings with currency symbols
    - Percentage strings
    - Parentheses for negatives
    - Various NA representations
    
    Returns None for unparseable inputs.
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    
    if not isinstance(value, str):
        return None
    
    value = str(value).strip()
    
    # Check for NA representations
    na_values = {'', 'na', 'n/a', 'nan', 'null', 'none', '.', '-', '--', '#n/a', '#value!'}
    if value.lower() in na_values:
        return None
    
    # Handle parentheses as negative
    is_negative = False
    if value.startswith('(') and value.endswith(')'):
        is_negative = True
        value = value[1:-1]
    
    # Remove currency symbols and whitespace
    value = re.sub(r'[$€£¥₹,\s]', '', value)
    
    # Handle percentage
    is_percent = False
    if value.endswith('%'):
        is_percent = True
        value = value[:-1]
    
    # Handle leading minus/plus
    if value.startswith('-'):
        is_negative = True
        value = value[1:]
    elif value.startswith('+'):
        value = value[1:]
    
    try:
        result = float(value)
        
        if is_percent:
            result = result / 100
        
        if is_negative:
            result = -result
        
        if math.isnan(result) or math.isinf(result):
            return None
        
        return result
    
    except ValueError:
        return None


def validate_numeric_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> bool:
    """Check if value is within specified range."""
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True
```

---

## 6. Panel Builder

### 6.1 Frequency Alignment

```python
# panel/frequency.py
"""
Frequency alignment functions for panel building.
Handles conversion between daily, monthly, and quarterly data.
"""

from typing import Literal, Optional
import pandas as pd
import numpy as np


FrequencyType = Literal['daily', 'monthly', 'quarterly']
AlignMethod = Literal['end_of_month', 'end_of_period', 'mean', 'last']


def align_to_monthly(
    df: pd.DataFrame,
    date_col: str = 'date',
    value_col: str = 'value',
    source_frequency: FrequencyType = 'daily',
    method: AlignMethod = 'end_of_month'
) -> pd.DataFrame:
    """
    Align data to monthly frequency.
    
    Args:
        df: DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column
        source_frequency: Original frequency of data
        method: How to select/compute monthly values
            - 'end_of_month': Last value of each month
            - 'end_of_period': Last available value in period
            - 'mean': Average of all values in month
            - 'last': Last value (same as end_of_period)
    
    Returns:
        DataFrame with monthly observations
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    if source_frequency == 'monthly':
        # Already monthly, just ensure end-of-month dates
        df['month_end'] = df[date_col] + pd.offsets.MonthEnd(0)
        df[date_col] = df['month_end']
        return df.drop(columns=['month_end'])
    
    if source_frequency == 'quarterly':
        # Quarterly data: forward-fill to monthly
        df['month_end'] = df[date_col] + pd.offsets.MonthEnd(0)
        
        # Create full monthly date range
        full_range = pd.date_range(
            start=df['month_end'].min(),
            end=df['month_end'].max(),
            freq='ME'
        )
        
        # Reindex and forward-fill
        df = df.set_index('month_end')
        df = df.reindex(full_range)
        df[value_col] = df[value_col].ffill()
        df = df.reset_index()
        df = df.rename(columns={'index': date_col})
        
        return df
    
    # Daily to monthly
    if method == 'end_of_month' or method == 'end_of_period' or method == 'last':
        # Group by month, take last value
        df['year_month'] = df[date_col].dt.to_period('M')
        result = df.groupby('year_month').last().reset_index()
        result[date_col] = result['year_month'].dt.to_timestamp() + pd.offsets.MonthEnd(0)
        result = result.drop(columns=['year_month'])
        
    elif method == 'mean':
        df['year_month'] = df[date_col].dt.to_period('M')
        result = df.groupby('year_month').agg({value_col: 'mean'}).reset_index()
        result[date_col] = result['year_month'].dt.to_timestamp() + pd.offsets.MonthEnd(0)
        result = result.drop(columns=['year_month'])
    
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    return result


def align_dates(
    dfs: dict[str, pd.DataFrame],
    target_frequency: FrequencyType = 'monthly',
    date_col: str = 'date',
    method: AlignMethod = 'end_of_month',
    min_coverage: float = 0.8
) -> pd.DataFrame:
    """
    Align multiple DataFrames to common date index.
    
    Args:
        dfs: Dictionary of {series_name: DataFrame}
        target_frequency: Target frequency for alignment
        date_col: Name of date column in each DataFrame
        method: Alignment method
        min_coverage: Minimum fraction of dates a series must have (0-1)
    
    Returns:
        Combined DataFrame with aligned dates
    """
    aligned = {}
    
    for name, df in dfs.items():
        # Determine source frequency from data
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Guess frequency from date gaps
        if len(df) > 1:
            median_gap = df[date_col].diff().median().days
            if median_gap <= 5:
                source_freq = 'daily'
            elif median_gap <= 35:
                source_freq = 'monthly'
            else:
                source_freq = 'quarterly'
        else:
            source_freq = 'monthly'  # Default
        
        # Align to target frequency
        if target_frequency == 'monthly':
            aligned_df = align_to_monthly(
                df,
                date_col=date_col,
                value_col=[c for c in df.columns if c != date_col][0],
                source_frequency=source_freq,
                method=method
            )
        else:
            # For now, only monthly target supported
            raise NotImplementedError(f"Target frequency {target_frequency} not implemented")
        
        aligned[name] = aligned_df.set_index(date_col)
    
    # Combine all series
    combined = pd.DataFrame()
    
    for name, df in aligned.items():
        if len(combined) == 0:
            combined = df.rename(columns={df.columns[0]: name})
        else:
            other = df.rename(columns={df.columns[0]: name})
            combined = combined.join(other, how='outer')
    
    # Filter by coverage
    total_dates = len(combined)
    for col in combined.columns:
        coverage = combined[col].notna().sum() / total_dates
        if coverage < min_coverage:
            print(f"Warning: {col} has only {coverage:.1%} coverage, excluding")
            combined = combined.drop(columns=[col])
    
    # Sort by date
    combined = combined.sort_index()
    
    return combined.reset_index().rename(columns={'index': date_col})
```

### 6.2 Panel Builder Main Module

```python
# panel/builder.py
"""
Main panel building module.
Reads registries, loads data, applies transforms, outputs panel.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import logging
from datetime import datetime

from db.connector import DatabaseConnector
from panel.frequency import align_dates
from transforms.registry import TransformRegistry


class PanelBuilder:
    """
    Builds analysis panels from database data.
    
    Process:
    1. Load system and series registries
    2. Query raw data from database
    3. Apply frequency alignment
    4. Apply transforms per engine config
    5. Output panel CSV
    """
    
    def __init__(
        self,
        config_dir: str = 'config',
        logger: Optional[logging.Logger] = None
    ):
        self.config_dir = Path(config_dir)
        self.logger = logger or logging.getLogger('PanelBuilder')
        
        # Load registries
        self.system_config = self._load_json('system_registry.json')
        self.market_registry = self._load_json('market_registry.json')
        self.economic_registry = self._load_json('economic_registry.json')
        self.transform_registry = TransformRegistry(
            self._load_json('transform_registry.json')
        )
        
        # Initialize database connection
        db_path = self.system_config['paths']['database']
        self.db = DatabaseConnector(db_path)
    
    def _load_json(self, filename: str) -> dict:
        """Load a JSON config file."""
        path = self.config_dir / filename
        with open(path) as f:
            return json.load(f)
    
    def build(
        self,
        engine: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build a panel for a specific engine.
        
        Args:
            engine: Engine name (e.g., 'macro_engine', 'stress_engine')
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            output_path: Optional path to save CSV
        
        Returns:
            DataFrame with panel data
        """
        self.logger.info(f"Building panel for engine: {engine}")
        
        # Collect series configurations
        series_configs = self._get_series_for_engine(engine)
        
        if not series_configs:
            raise ValueError(f"No series configured for engine: {engine}")
        
        self.logger.info(f"Found {len(series_configs)} series for {engine}")
        
        # Load raw data for each series
        raw_data = {}
        for series_id, config in series_configs.items():
            df = self._load_series_data(series_id, config, start_date, end_date)
            if df is not None and len(df) > 0:
                raw_data[series_id] = df
                self.logger.info(f"Loaded {len(df)} rows for {series_id}")
            else:
                self.logger.warning(f"No data for {series_id}")
        
        if not raw_data:
            raise ValueError("No data loaded for any series")
        
        # Align to common frequency
        target_freq = self.system_config['panel']['default_frequency']
        align_method = self.system_config['panel']['default_align_method']
        min_coverage = self.system_config['panel']['required_coverage_pct'] / 100
        
        self.logger.info(f"Aligning to {target_freq} frequency")
        aligned = align_dates(
            raw_data,
            target_frequency=target_freq,
            method=align_method,
            min_coverage=min_coverage
        )
        
        self.logger.info(f"Aligned panel shape: {aligned.shape}")
        
        # Apply transforms
        transformed = self._apply_transforms(aligned, series_configs, engine)
        
        # Output
        if output_path:
            output_file = Path(output_path)
        else:
            output_dir = Path(self.system_config['paths']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"panel_{engine}_{timestamp}.csv"
        
        transformed.to_csv(output_file, index=False)
        self.logger.info(f"Saved panel to {output_file}")
        
        return transformed
    
    def _get_series_for_engine(self, engine: str) -> Dict[str, Dict[str, Any]]:
        """Get all series configured for a specific engine."""
        series_configs = {}
        
        # Check market series
        for series_id, config in self.market_registry.get('series', {}).items():
            if not config.get('enabled', True):
                continue
            if engine in config.get('engine_config', {}):
                series_configs[series_id] = {
                    'type': 'market',
                    'config': config,
                    'engine_config': config['engine_config'][engine]
                }
        
        # Check economic series
        for series_id, config in self.economic_registry.get('series', {}).items():
            if not config.get('enabled', True):
                continue
            if engine in config.get('engine_config', {}):
                series_configs[series_id] = {
                    'type': 'economic',
                    'config': config,
                    'engine_config': config['engine_config'][engine]
                }
        
        return series_configs
    
    def _load_series_data(
        self,
        series_id: str,
        config: Dict[str, Any],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Load raw data for a series from database."""
        series_type = config['type']
        
        if series_type == 'market':
            table = 'market_prices'
            value_col = config['config'].get('primary_field', 'adj_close')
            query = f"""
                SELECT date, {value_col} as value
                FROM {table}
                WHERE series_id = ?
            """
        else:
            table = 'economic_values'
            query = f"""
                SELECT date, value
                FROM {table}
                WHERE series_id = ?
                AND vintage_date = (
                    SELECT MAX(vintage_date) 
                    FROM {table} t2 
                    WHERE t2.series_id = {table}.series_id 
                    AND t2.date = {table}.date
                )
            """
        
        params = [series_id]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        df = self.db.fetch_df(query, params)
        
        return df if len(df) > 0 else None
    
    def _apply_transforms(
        self,
        df: pd.DataFrame,
        series_configs: Dict[str, Dict[str, Any]],
        engine: str
    ) -> pd.DataFrame:
        """Apply transforms to each series based on engine config."""
        result = df[['date']].copy()
        
        for series_id, config in series_configs.items():
            if series_id not in df.columns:
                continue
            
            engine_config = config['engine_config']
            transform_name = engine_config.get('use_transform', 'level')
            transform_params = engine_config.get('params', {})
            
            # Get the raw values
            raw_values = df[series_id].copy()
            
            # Apply transform
            try:
                transformed = self.transform_registry.apply(
                    transform_name,
                    raw_values,
                    **transform_params
                )
                
                # Name the column with transform suffix
                col_name = f"{series_id}_{transform_name}"
                result[col_name] = transformed
                
                self.logger.debug(f"Applied {transform_name} to {series_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to apply {transform_name} to {series_id}: {e}")
                # Include raw values as fallback
                result[f"{series_id}_raw"] = raw_values
        
        return result


def build_panel_cli():
    """Command-line interface for panel building."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build analysis panel')
    parser.add_argument('engine', help='Engine name (macro_engine, stress_engine, etc.)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--config-dir', default='config', help='Config directory')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    builder = PanelBuilder(config_dir=args.config_dir)
    panel = builder.build(
        engine=args.engine,
        start_date=args.start,
        end_date=args.end,
        output_path=args.output
    )
    
    print(f"\nPanel built successfully!")
    print(f"Shape: {panel.shape}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Columns: {list(panel.columns)}")


if __name__ == '__main__':
    build_panel_cli()
```

---

## 7. Transform System

### 7.1 Transform Registry

```python
# transforms/registry.py
"""
Transform registry and dispatcher.
"""

from typing import Dict, Any, Callable, Optional
import pandas as pd
import numpy as np


class TransformRegistry:
    """
    Registry of available transforms.
    Dispatches transform calls to appropriate functions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._transforms = self._register_transforms()
    
    def _register_transforms(self) -> Dict[str, Callable]:
        """Register all transform functions."""
        from transforms import returns, growth, normalize, smooth, base
        
        return {
            # Base transforms
            'level': base.level,
            'log': base.log_transform,
            'sign': base.sign,
            
            # Return transforms
            'pct_change': returns.pct_change,
            'log_return': returns.log_return,
            
            # Growth transforms
            'yoy': growth.yoy,
            'yoy_diff': growth.yoy_diff,
            'mom': growth.mom,
            'qoq': growth.qoq,
            'diff': growth.diff,
            
            # Normalization transforms
            'z_score': normalize.z_score,
            'percentile_rank': normalize.percentile_rank,
            'min_max': normalize.min_max,
            
            # Smoothing transforms
            'rolling_mean': smooth.rolling_mean,
            'ma': smooth.rolling_mean,  # Alias
            'ewma': smooth.ewma,
            'rolling_vol': smooth.rolling_vol,
        }
    
    def apply(
        self,
        transform_name: str,
        series: pd.Series,
        **params
    ) -> pd.Series:
        """
        Apply a transform to a series.
        
        Args:
            transform_name: Name of transform
            series: Input pandas Series
            **params: Transform parameters
        
        Returns:
            Transformed Series
        """
        if transform_name not in self._transforms:
            raise ValueError(f"Unknown transform: {transform_name}")
        
        # Get default params from config
        transform_config = self.config.get('transforms', {}).get(transform_name, {})
        default_params = {}
        for param_name, param_spec in transform_config.get('params', {}).items():
            if isinstance(param_spec, dict) and 'default' in param_spec:
                default_params[param_name] = param_spec['default']
        
        # Merge with provided params
        final_params = {**default_params, **params}
        
        # Call transform function
        func = self._transforms[transform_name]
        return func(series, **final_params)
    
    def list_transforms(self) -> list:
        """List available transform names."""
        return list(self._transforms.keys())
    
    def get_info(self, transform_name: str) -> Dict[str, Any]:
        """Get info about a transform."""
        if transform_name not in self.config.get('transforms', {}):
            return {}
        return self.config['transforms'][transform_name]
```

### 7.2 Transform Implementations

```python
# transforms/base.py
"""Base transforms: level, log, sign."""

import pandas as pd
import numpy as np


def level(series: pd.Series, **kwargs) -> pd.Series:
    """Return values unchanged."""
    return series.copy()


def log_transform(series: pd.Series, **kwargs) -> pd.Series:
    """Natural logarithm. Requires positive values."""
    result = series.copy()
    result[result <= 0] = np.nan
    return np.log(result)


def sign(series: pd.Series, **kwargs) -> pd.Series:
    """Sign of value (-1, 0, 1)."""
    return np.sign(series)
```

```python
# transforms/returns.py
"""Return calculations: pct_change, log_return."""

import pandas as pd
import numpy as np


def pct_change(series: pd.Series, periods: int = 1, **kwargs) -> pd.Series:
    """Percentage change over N periods."""
    return series.pct_change(periods=periods) * 100


def log_return(series: pd.Series, periods: int = 1, **kwargs) -> pd.Series:
    """Log return over N periods."""
    return np.log(series / series.shift(periods)) * 100
```

```python
# transforms/growth.py
"""Growth rate calculations: yoy, mom, qoq, diff."""

import pandas as pd
import numpy as np


def yoy(series: pd.Series, **kwargs) -> pd.Series:
    """
    Year-over-year percentage change.
    Assumes monthly data (12 periods).
    """
    return series.pct_change(periods=12) * 100


def yoy_diff(series: pd.Series, **kwargs) -> pd.Series:
    """
    Year-over-year difference (for rates).
    Assumes monthly data (12 periods).
    """
    return series.diff(periods=12)


def mom(series: pd.Series, **kwargs) -> pd.Series:
    """Month-over-month percentage change."""
    return series.pct_change(periods=1) * 100


def qoq(series: pd.Series, **kwargs) -> pd.Series:
    """Quarter-over-quarter percentage change."""
    return series.pct_change(periods=1) * 100


def diff(series: pd.Series, periods: int = 1, **kwargs) -> pd.Series:
    """Simple difference over N periods."""
    return series.diff(periods=periods)
```

```python
# transforms/normalize.py
"""Normalization transforms: z_score, percentile_rank, min_max."""

import pandas as pd
import numpy as np


def z_score(series: pd.Series, window: int = None, **kwargs) -> pd.Series:
    """
    Z-score normalization.
    
    Args:
        window: Rolling window size. None = full history.
    """
    if window is None:
        # Expanding z-score
        mean = series.expanding().mean()
        std = series.expanding().std()
    else:
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
    
    return (series - mean) / std


def percentile_rank(series: pd.Series, window: int = 252, **kwargs) -> pd.Series:
    """
    Percentile rank within rolling window.
    Returns value from 0 to 100.
    """
    def rank_pct(x):
        if len(x) < 2:
            return np.nan
        return (x.argsort().argsort()[-1] / (len(x) - 1)) * 100
    
    return series.rolling(window=window).apply(rank_pct, raw=False)


def min_max(series: pd.Series, window: int = None, **kwargs) -> pd.Series:
    """
    Min-max scaling to 0-1 range.
    
    Args:
        window: Rolling window size. None = full history.
    """
    if window is None:
        min_val = series.expanding().min()
        max_val = series.expanding().max()
    else:
        min_val = series.rolling(window=window).min()
        max_val = series.rolling(window=window).max()
    
    return (series - min_val) / (max_val - min_val)
```

```python
# transforms/smooth.py
"""Smoothing transforms: rolling_mean, ewma, rolling_vol."""

import pandas as pd
import numpy as np


def rolling_mean(series: pd.Series, window: int = 20, **kwargs) -> pd.Series:
    """Rolling mean (simple moving average)."""
    return series.rolling(window=window).mean()


def ewma(series: pd.Series, span: int = 20, **kwargs) -> pd.Series:
    """Exponentially weighted moving average."""
    return series.ewm(span=span).mean()


def rolling_vol(
    series: pd.Series,
    window: int = 21,
    annualize: bool = True,
    **kwargs
) -> pd.Series:
    """
    Rolling volatility (standard deviation of returns).
    
    Args:
        window: Rolling window size
        annualize: If True, multiply by sqrt(252) for daily data
    """
    returns = series.pct_change()
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol * 100  # As percentage
```

---

## 8. Error Handling

### 8.1 Error Strategy

```python
# utils/errors.py
"""Custom exceptions and error handling."""


class PrismError(Exception):
    """Base exception for Prism Engine."""
    pass


class FetchError(PrismError):
    """Error during data fetching."""
    def __init__(self, series_id: str, message: str, recoverable: bool = True):
        self.series_id = series_id
        self.recoverable = recoverable
        super().__init__(f"Fetch error for {series_id}: {message}")


class ValidationError(PrismError):
    """Error during data validation."""
    def __init__(self, series_id: str, date: str, reason: str):
        self.series_id = series_id
        self.date = date
        self.reason = reason
        super().__init__(f"Validation error for {series_id} on {date}: {reason}")


class TransformError(PrismError):
    """Error during transform application."""
    def __init__(self, transform_name: str, series_id: str, message: str):
        self.transform_name = transform_name
        self.series_id = series_id
        super().__init__(f"Transform {transform_name} failed on {series_id}: {message}")


class ConfigError(PrismError):
    """Error in configuration."""
    pass


class DatabaseError(PrismError):
    """Error in database operations."""
    pass
```

### 8.2 Error Handling in Fetch Pipeline

```python
# Error handling configuration in system_registry.json
{
  "fetch": {
    "on_series_error": "continue",  # "halt" or "continue"
    "quarantine_threshold_pct": 10,  # Fail if >10% of rows quarantined
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "alert_on_failure": true
  }
}
```

```python
# fetch/runner.py
"""Fetch runner with error handling."""

import logging
from typing import List, Dict, Any
from datetime import datetime

from fetch.market_fetcher import MarketFetcher
from fetch.fred_fetcher import FREDFetcher
from db.connector import DatabaseConnector
from utils.errors import FetchError


class FetchRunner:
    """
    Runs fetch operations across all configured series.
    Handles errors according to configuration.
    """
    
    def __init__(
        self,
        system_config: Dict[str, Any],
        market_registry: Dict[str, Any],
        economic_registry: Dict[str, Any],
        db: DatabaseConnector,
        logger: logging.Logger = None
    ):
        self.system_config = system_config
        self.market_registry = market_registry
        self.economic_registry = economic_registry
        self.db = db
        self.logger = logger or logging.getLogger('FetchRunner')
        
        self.fetch_config = system_config.get('fetch', {})
        self.on_error = self.fetch_config.get('on_series_error', 'continue')
        self.quarantine_threshold = self.fetch_config.get('quarantine_threshold_pct', 10)
    
    def run_all(self, mode: str = 'incremental') -> Dict[str, Any]:
        """
        Run fetch for all enabled series.
        
        Returns:
            Summary of fetch results
        """
        results = {
            'started_at': datetime.now().isoformat(),
            'mode': mode,
            'series_results': [],
            'total_success': 0,
            'total_failed': 0,
            'total_partial': 0,
            'errors': []
        }
        
        # Fetch market series
        for series_id, config in self.market_registry.get('series', {}).items():
            if not config.get('enabled', True):
                continue
            
            try:
                result = self._fetch_market_series(series_id, config, mode)
                results['series_results'].append(result.__dict__)
                
                if result.status == 'success':
                    results['total_success'] += 1
                elif result.status == 'partial':
                    results['total_partial'] += 1
                    self._check_quarantine_threshold(result)
                elif result.status == 'failed':
                    results['total_failed'] += 1
                    results['errors'].append({
                        'series_id': series_id,
                        'error': result.error_message
                    })
                    
                    if self.on_error == 'halt':
                        raise FetchError(series_id, result.error_message, recoverable=False)
                        
            except FetchError as e:
                if not e.recoverable:
                    raise
                self.logger.error(f"Failed to fetch {series_id}: {e}")
                results['total_failed'] += 1
                results['errors'].append({
                    'series_id': series_id,
                    'error': str(e)
                })
        
        # Fetch economic series
        for series_id, config in self.economic_registry.get('series', {}).items():
            if not config.get('enabled', True):
                continue
            
            try:
                result = self._fetch_economic_series(series_id, config, mode)
                results['series_results'].append(result.__dict__)
                
                if result.status == 'success':
                    results['total_success'] += 1
                elif result.status == 'partial':
                    results['total_partial'] += 1
                elif result.status == 'failed':
                    results['total_failed'] += 1
                    results['errors'].append({
                        'series_id': series_id,
                        'error': result.error_message
                    })
                    
                    if self.on_error == 'halt':
                        raise FetchError(series_id, result.error_message, recoverable=False)
                        
            except FetchError as e:
                if not e.recoverable:
                    raise
                self.logger.error(f"Failed to fetch {series_id}: {e}")
                results['total_failed'] += 1
        
        results['completed_at'] = datetime.now().isoformat()
        return results
    
    def _fetch_market_series(self, series_id: str, config: Dict, mode: str):
        """Fetch a single market series."""
        fetcher = MarketFetcher(
            db=self.db,
            series_config=config,
            system_config=self.system_config,
            logger=self.logger
        )
        return fetcher.fetch(mode=mode)
    
    def _fetch_economic_series(self, series_id: str, config: Dict, mode: str):
        """Fetch a single economic series."""
        fetcher = FREDFetcher(
            db=self.db,
            series_config=config,
            system_config=self.system_config,
            logger=self.logger
        )
        return fetcher.fetch(mode=mode)
    
    def _check_quarantine_threshold(self, result):
        """Check if quarantine rate exceeds threshold."""
        if result.rows_fetched > 0:
            quarantine_pct = (result.rows_quarantined / result.rows_fetched) * 100
            if quarantine_pct > self.quarantine_threshold:
                self.logger.warning(
                    f"{result.series_id}: {quarantine_pct:.1f}% rows quarantined "
                    f"(threshold: {self.quarantine_threshold}%)"
                )
```

---

## 9. Testing Requirements

### 9.1 Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_fetch/
│   ├── test_base_fetcher.py
│   ├── test_market_fetcher.py
│   ├── test_fred_fetcher.py
│   └── test_validators.py
├── test_db/
│   ├── test_connector.py
│   ├── test_migrations.py
│   └── test_schema.py
├── test_panel/
│   ├── test_builder.py
│   ├── test_frequency.py
│   └── test_output.py
├── test_transforms/
│   ├── test_returns.py
│   ├── test_growth.py
│   ├── test_normalize.py
│   └── test_smooth.py
└── fixtures/
    ├── sample_market_data.csv
    ├── sample_economic_data.csv
    └── test_registry.json
```

### 9.2 Key Test Cases

```python
# tests/conftest.py
"""Shared test fixtures."""

import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd

from db.connector import DatabaseConnector


@pytest.fixture
def temp_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = DatabaseConnector(db_path)
    
    # Run migrations
    migrations_dir = Path(__file__).parent.parent / 'db' / 'migrations'
    for migration in sorted(migrations_dir.glob('*.sql')):
        with open(migration) as f:
            db.execute_script(f.read())
    
    yield db
    
    # Cleanup
    db.close()
    Path(db_path).unlink()


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [101 + i * 0.1 for i in range(100)],
        'low': [99 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'adj_close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000000] * 100
    })


@pytest.fixture
def sample_economic_data():
    """Sample economic data for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=24, freq='ME'),
        'value': [100 + i * 0.5 for i in range(24)]
    })
```

```python
# tests/test_fetch/test_validators.py
"""Tests for fetch validation."""

import pytest
import pandas as pd
from utils.date_utils import normalize_date
from utils.numeric_utils import clean_numeric


class TestDateNormalization:
    """Tests for date normalization."""
    
    def test_iso_format(self):
        assert normalize_date('2024-01-15') == '2024-01-15'
    
    def test_us_format(self):
        assert normalize_date('01/15/2024') == '2024-01-15'
    
    def test_two_digit_year(self):
        assert normalize_date('01/15/24') == '2024-01-15'
        assert normalize_date('01/15/99') == '1999-01-15'
    
    def test_datetime_object(self):
        from datetime import datetime
        dt = datetime(2024, 1, 15)
        assert normalize_date(dt) == '2024-01-15'
    
    def test_invalid_returns_none(self):
        assert normalize_date('not a date') is None
        assert normalize_date(None) is None
        assert normalize_date('') is None


class TestNumericCleaning:
    """Tests for numeric cleaning."""
    
    def test_already_numeric(self):
        assert clean_numeric(123.45) == 123.45
        assert clean_numeric(100) == 100.0
    
    def test_string_with_commas(self):
        assert clean_numeric('1,234.56') == 1234.56
    
    def test_currency_symbols(self):
        assert clean_numeric('$1,234.56') == 1234.56
        assert clean_numeric('€100') == 100.0
    
    def test_percentage(self):
        assert clean_numeric('50%') == 0.5
    
    def test_negative_parentheses(self):
        assert clean_numeric('(100)') == -100.0
    
    def test_na_values(self):
        assert clean_numeric('NA') is None
        assert clean_numeric('.') is None
        assert clean_numeric('') is None
        assert clean_numeric(None) is None


class TestDuplicateHandling:
    """Tests for duplicate date handling."""
    
    def test_keep_last(self):
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'value': [100, 101, 102]
        })
        deduped = df.drop_duplicates(subset=['date'], keep='last')
        assert len(deduped) == 2
        assert deduped[deduped['date'] == '2024-01-01']['value'].iloc[0] == 101
```

```python
# tests/test_panel/test_frequency.py
"""Tests for frequency alignment."""

import pytest
import pandas as pd
import numpy as np
from panel.frequency import align_to_monthly, align_dates


class TestAlignToMonthly:
    """Tests for monthly alignment."""
    
    def test_daily_to_monthly_end(self):
        """Daily data should align to month-end."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': range(len(dates))
        })
        
        result = align_to_monthly(df, source_frequency='daily', method='end_of_month')
        
        assert len(result) == 3  # Jan, Feb, Mar
        assert result['date'].iloc[0] == pd.Timestamp('2024-01-31')
        assert result['date'].iloc[1] == pd.Timestamp('2024-02-29')
        assert result['date'].iloc[2] == pd.Timestamp('2024-03-31')
    
    def test_quarterly_forward_fill(self):
        """Quarterly data should forward-fill to monthly."""
        df = pd.DataFrame({
            'date': ['2024-03-31', '2024-06-30'],
            'value': [100, 110]
        })
        
        result = align_to_monthly(df, source_frequency='quarterly')
        
        assert len(result) == 4  # Mar, Apr, May, Jun
        assert result['value'].iloc[0] == 100  # March
        assert result['value'].iloc[1] == 100  # April (forward-filled)
        assert result['value'].iloc[2] == 100  # May (forward-filled)
        assert result['value'].iloc[3] == 110  # June


class TestAlignDates:
    """Tests for multi-series alignment."""
    
    def test_combine_daily_and_monthly(self):
        """Should properly combine daily and monthly series."""
        daily_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-03-31', freq='D'),
            'value': range(91)
        })
        
        monthly_df = pd.DataFrame({
            'date': ['2024-01-31', '2024-02-29', '2024-03-31'],
            'value': [100, 110, 120]
        })
        
        result = align_dates(
            {'daily_series': daily_df, 'monthly_series': monthly_df},
            target_frequency='monthly'
        )
        
        assert len(result) == 3
        assert 'daily_series' in result.columns
        assert 'monthly_series' in result.columns
```

```python
# tests/test_transforms/test_growth.py
"""Tests for growth transforms."""

import pytest
import pandas as pd
import numpy as np
from transforms.growth import yoy, yoy_diff, mom


class TestYoY:
    """Tests for year-over-year growth."""
    
    def test_basic_yoy(self):
        """Basic YoY calculation."""
        values = [100] * 12 + [110] * 12  # 100 for year 1, 110 for year 2
        series = pd.Series(values)
        
        result = yoy(series)
        
        # First 12 values should be NaN
        assert result[:12].isna().all()
        # 13th value onwards should be 10%
        assert np.isclose(result.iloc[12], 10.0)
    
    def test_negative_growth(self):
        """Negative growth calculation."""
        values = [100] * 12 + [90] * 12
        series = pd.Series(values)
        
        result = yoy(series)
        
        assert np.isclose(result.iloc[12], -10.0)


class TestYoYDiff:
    """Tests for year-over-year difference (for rates)."""
    
    def test_rate_diff(self):
        """Rate difference calculation."""
        values = [5.0] * 12 + [5.25] * 12  # Rate went from 5% to 5.25%
        series = pd.Series(values)
        
        result = yoy_diff(series)
        
        assert np.isclose(result.iloc[12], 0.25)
```

---

## 10. Migration Plan

### 10.1 Phase 1: Setup (Week 1)

1. Create folder structure
2. Create all config JSON files
3. Set up database connector
4. Run schema migrations
5. Set up logging

**Verification:**
- [ ] All folders exist
- [ ] JSON configs parse without error
- [ ] Database creates successfully
- [ ] All tables created

### 10.2 Phase 2: Fetch Pipeline (Week 2)

1. Implement BaseFetcher
2. Implement MarketFetcher
3. Implement FREDFetcher
4. Implement date/numeric utilities
5. Write fetch tests

**Verification:**
- [ ] Can fetch SPX data
- [ ] Can fetch UNRATE data
- [ ] Data inserts to correct tables
- [ ] Validation catches bad data
- [ ] Quarantine table populated for failures

### 10.3 Phase 3: Transform System (Week 3)

1. Implement transform registry
2. Implement all transform functions
3. Write transform tests

**Verification:**
- [ ] All transforms produce expected output
- [ ] Transform registry dispatches correctly
- [ ] Edge cases handled (NaN, division by zero)

### 10.4 Phase 4: Panel Builder (Week 4)

1. Implement frequency alignment
2. Implement panel builder
3. Implement CLI
4. Write panel tests

**Verification:**
- [ ] Daily → monthly alignment works
- [ ] Quarterly forward-fill works
- [ ] Panel outputs correct shape
- [ ] Transforms applied correctly

### 10.5 Phase 5: Integration (Week 5)

1. Wire up existing engines
2. Full system test
3. Documentation
4. Remove legacy files

**Verification:**
- [ ] Engines can load new panels
- [ ] Results match expected output
- [ ] No import errors
- [ ] Documentation complete

---

## 11. Implementation Phases for Claude Code

### Recommended PR Sequence

**PR 1: Foundation**
- Folder structure
- config/*.json files
- db/connector.py
- utils/logging_utils.py
- requirements.txt

**PR 2: Database**
- All migration files
- db/models.py
- db/queries.py
- tests/test_db/

**PR 3: Utilities**
- utils/date_utils.py
- utils/numeric_utils.py
- utils/errors.py
- tests for utilities

**PR 4: Fetch Pipeline**
- fetch/base_fetcher.py
- fetch/market_fetcher.py
- fetch/fred_fetcher.py
- fetch/validators.py
- tests/test_fetch/

**PR 5: Transforms**
- transforms/*.py
- transforms/registry.py
- tests/test_transforms/

**PR 6: Panel Builder**
- panel/frequency.py
- panel/builder.py
- panel/output.py
- tests/test_panel/

**PR 7: Scripts & Integration**
- scripts/run_fetch.py
- scripts/run_panel.py
- scripts/run_migrate.py
- Integration tests

---

## Appendix A: Environment Setup

```bash
# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.28
requests>=2.31.0
python-dotenv>=1.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

```bash
# .env.example
FRED_API_KEY=your_fred_api_key_here
```

```bash
# Initial setup commands
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your FRED API key
python scripts/run_migrate.py
python scripts/run_fetch.py --mode full
python scripts/run_panel.py macro_engine
```

---

## Appendix B: Quick Reference

### Adding a New Market Series

1. Add entry to `config/market_registry.json`:
```json
"NEW_TICKER": {
  "enabled": true,
  "ticker": "NEW",
  "name": "New Series Name",
  "source": "yahoo",
  "frequency": "daily",
  "fields": ["adj_close"],
  "primary_field": "adj_close",
  "start_date": "2000-01-01",
  "transforms_available": ["pct_change", "yoy"],
  "engine_config": {
    "macro_engine": {"use_transform": "yoy"}
  },
  "validation": {"min_value": 0}
}
```

2. Run fetch: `python scripts/run_fetch.py --series NEW_TICKER`

### Adding a New Economic Series

1. Add entry to `config/economic_registry.json`:
```json
"NEWSERIES": {
  "enabled": true,
  "series_id": "NEWSERIES",
  "name": "New Economic Series",
  "source": "fred",
  "frequency": "monthly",
  "start_date": "1990-01-01",
  "transforms_available": ["yoy", "z_score"],
  "engine_config": {
    "macro_engine": {"use_transform": "yoy"}
  },
  "validation": {"min_value": 0}
}
```

2. Run fetch: `python scripts/run_fetch.py --series NEWSERIES`

### Adding a New Transform

1. Add function to appropriate `transforms/*.py` file
2. Add entry to `config/transform_registry.json`
3. Register in `transforms/registry.py`
4. Add tests

---

*End of Specification*
