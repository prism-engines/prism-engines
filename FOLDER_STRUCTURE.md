# PRISM Engine - Folder Structure
# ================================
# Reorganized for human workflow, not technical components
#
# BEFORE (technical thinking):
#   01_fetch/ 02_data_raw/ 03_cleaning/ 04_data_clean/ 05_engine/...
#   Problem: "Where do I start? What runs what?"
#
# AFTER (human thinking):
#   "Here's my data. Run analysis. Show me results."

prism-engine/
│
├── data/
│   ├── raw/                    # Original fetched data (don't touch)
│   ├── clean/                  # Processed, ready for analysis
│   └── indicators.yaml         # Defines your indicator categories
│
├── run.py                      # THE ONLY FILE YOU RUN
│
├── engine/
│   ├── consolidate.py          # Groups & weights indicators
│   ├── lenses/                 # All 14+ analysis engines
│   │   ├── __init__.py
│   │   ├── correlation.py
│   │   ├── pca.py
│   │   ├── granger.py
│   │   └── ...
│   ├── coherence.py            # Coherence Index calculation
│   └── temporal.py             # Time windowing logic
│
├── results/
│   ├── latest/                 # Always the most recent run
│   │   ├── summary.txt         # Human-readable summary
│   │   ├── coherence.csv       # Coherence over time
│   │   ├── rankings.csv        # Indicator rankings
│   │   └── charts/             # Generated visualizations
│   │
│   └── archive/                # Past runs (auto-dated)
│       └── 2024-11-30_143022/
│
├── reports/                    # Formatted outputs for humans
│   └── latest_report.html      # Nice visual report
│
├── config/
│   └── settings.yaml           # Tweak without touching code
│
├── requirements.txt
└── README.md


# ============================================================
# THE KEY INSIGHT: 
# ============================================================
#
# You only ever run ONE command:
#
#     python run.py
#
# That's it. Everything else is options:
#
#     python run.py                  # Full analysis, all defaults
#     python run.py --quick          # Fast version (fewer windows)
#     python run.py --focus monetary # Only monetary indicators  
#     python run.py --since 2020     # Recent data only
#     python run.py --report         # Generate HTML report at end
#
# The "run.py" script handles:
#   1. Loading & consolidating data
#   2. Running ALL relevant lenses
#   3. Computing coherence
#   4. Saving results
#   5. Printing a human summary
#
# You don't think about temporal vs coherence vs lenses.
# PRISM thinks about that. You just run it.
