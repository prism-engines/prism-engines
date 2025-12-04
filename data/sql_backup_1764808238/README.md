# PRISM Engine SQL Database

## Quick Setup (5 minutes)

### Step 1: Create the folder
```bash
mkdir -p /mnt/chromeos/GoogleDrive/MyDrive/prismsql
```

### Step 2: Set the environment variable
Add this to your shell config (~/.bashrc or ~/.zshrc):
```bash
export PRISM_DB="/mnt/chromeos/GoogleDrive/MyDrive/prismsql/prism.db"
```

Then reload:
```bash
source ~/.bashrc
```

### Step 3: Copy the files
Put these files in your PRISM project:
- `prism_schema.sql` - Database structure
- `prism_db.py` - Python helper library

### Step 4: Initialize the database
```python
from prism_db import init_database
init_database()
```

Done! The database is ready.

---

## Daily Usage

### Import data
```python
from prism_db import add_indicator, write_values
import pandas as pd

# Add an indicator
add_indicator('SPY', panel='equity', frequency='daily', source='Yahoo')

# Write data (assuming df has 'date' and 'value' columns)
write_values('SPY', df)

# Or do both in one line:
from prism_db import write_dataframe
write_dataframe(df, 'SPY', 'equity', source='Yahoo')
```

### Load data
```python
from prism_db import load_indicator, load_multiple

# Single indicator
df = load_indicator('SPY')
df = load_indicator('SPY', start_date='2020-01-01')

# Multiple indicators (returns pivoted table)
df = load_multiple(['SPY', 'DXY', 'AGG', 'TLT'])
```

### Query anything
```python
from prism_db import query

# See all indicators
df = query("SELECT * FROM indicators")

# Count data points per indicator
df = query("""
    SELECT i.name, COUNT(*) as n_points
    FROM indicator_values iv
    JOIN indicators i ON iv.indicator_id = i.id
    GROUP BY i.id
    ORDER BY n_points DESC
""")
```

---

## What's in the Schema

### Input Tables
| Table | Purpose |
|-------|---------|
| `indicators` | Master list (name, panel, frequency, source) |
| `indicator_values` | Time series data (date, value, optional value_2) |

### Engine Output Tables
| Table | Purpose |
|-------|---------|
| `lenses` | The 14 analytical lenses |
| `windows` | Temporal analysis periods (e.g., 2005-2010) |
| `lens_results` | Per-lens rankings for each indicator/window |
| `consensus` | Aggregated rankings across lenses |
| `regime_stability` | Spearman correlations between windows |
| `coherence_events` | When lenses agreed (your key insight!) |
| `engine_runs` | Metadata for reproducibility |

### Pre-built Views
| View | What it shows |
|------|---------------|
| `v_top_indicators` | Current top-ranked indicators |
| `v_regime_transitions` | Regime break summary |
| `v_indicator_history` | Ranking trajectory over time |

---

## GUI Option

Download **DB Browser for SQLite** (free):
- https://sqlitebrowser.org

Open your `.db` file, browse tables, run queries visually.

---

## Comparison to GPT's PR

| Feature | GPT Version | Claude Version |
|---------|-------------|----------------|
| Input storage | Yes | Yes |
| Panel/frequency fields | Yes | Yes |
| Multiple value columns | No | Yes (value, value_2, adjusted) |
| Engine outputs | No | Yes (lens_results, consensus) |
| Temporal windows | No | Yes |
| Regime tracking | No | Yes |
| Coherence events | No | Yes |
| Run metadata | No | Yes |
| Indexes for speed | No | Yes |
| Pre-built views | No | Yes |
| Duplicate prevention | No | Yes (UNIQUE constraints) |

---

## File Locations

| File | Put it in... |
|------|--------------|
| `prism_schema.sql` | Your code folder |
| `prism_db.py` | Your code folder |
| `prism.db` (the actual database) | `/mnt/chromeos/GoogleDrive/MyDrive/prismsql/` |

The `.db` file stays OUT of GitHub. The `.py` and `.sql` files go IN your repo.
