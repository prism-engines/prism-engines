PRISM Benchmark Integration – Implementation Specification (FOR CLAUDE CODE)
Version 1.0 — Authoritative Instructions

This document defines exactly how the PRISM Engine must load benchmark datasets into the SQLite database for engine-validation testing.

These instructions are deterministic and must be implemented exactly as written.

1. Benchmark File Locations

All benchmark datasets already exist at:

data/benchmark/
    benchmark_01_clear_leader.csv
    benchmark_02_two_regimes.csv
    benchmark_03_clusters.csv
    benchmark_04_periodic.csv
    benchmark_05_anomalies.csv
    benchmark_06_pure_noise.csv


Each file contains:

A Date index column (auto-created)

One or more time-series indicator columns, named A, B, C, etc.

Exactly 1000 rows unless future revisions change the generator.

2. Required SQL Tables (MUST ALREADY EXIST)

The loader must write into these existing schema tables:

indicators
column	type	rules
id	INTEGER PRIMARY KEY	autoincrement
indicator_name	TEXT	required
fred_code	TEXT	placeholder "BENCH_{name}"
system	TEXT	"benchmark"
category	TEXT	benchmark file name
description	TEXT	indicate known structure
frequency	TEXT	"daily"
source	TEXT	"BENCHMARK"
timeseries
column	type	rules
id	INTEGER PRIMARY KEY	autoincrement
indicator_id	INTEGER	FK → indicators.id
date	TEXT	required
value	REAL	value in the column
value_2	REAL	always NULL
adjusted_value	REAL	always NULL
3. How Each Benchmark Must Be Inserted
For each column inside each CSV (A, B, C, …):

Claude must:

Create an indicator entry in indicators:

indicator_name: "{file_shortname}_{column}"
fred_code: "BENCH_{file_shortname}_{column}"
system: "benchmark"
category: "{file_shortname}"
description: "Benchmark synthetic dataset → known structure"
frequency: "daily"
source: "BENCHMARK"


Where:

File	file_shortname
benchmark_01_clear_leader.csv	clear_leader
benchmark_02_two_regimes.csv	two_regimes
benchmark_03_clusters.csv	clusters
benchmark_04_periodic.csv	periodic
benchmark_05_anomalies.csv	anomalies
benchmark_06_pure_noise.csv	pure_noise

Example:
Column A in benchmark 01 becomes indicator:

indicator_name = "clear_leader_A"
fred_code = "BENCH_clear_leader_A"

4. Timeseries Insert Logic (MANDATORY)

For each CSV row:

date → timeseries.date
value → timeseries.value
value_2 → NULL
adjusted_value → NULL
indicator_id → FK from indicators table


All inserts must be wrapped in a single transaction per file for speed.

5. Required Python Loader Module

Claude Code must create a new file:

data/sql/load_benchmarks.py


And implement exactly this callable function:

def load_all_benchmarks(benchmark_dir="data/benchmark"):
    """
    Loads all 6 benchmark CSV files into the PRISM SQLite database.
    Overwrites no existing FRED indicators.
    Creates indicators under system='benchmark'.
    """

6. PRISM CLI Integration

Claude must register a new CLI command in prism_run.py:

python prism_run.py --load-benchmarks


Which calls:

from data.sql.load_benchmarks import load_all_benchmarks
load_all_benchmarks()

7. Validation Tests (Claude Must Add)

Claude must create:

tests/benchmark/test_benchmarks_loaded.py


Tests required:

✔ indicators created for all columns
✔ timeseries row counts match CSV row counts
✔ system == "benchmark"
✔ fred_code starts with "BENCH_"
✔ no NULL indicator_name

8. Final Expected Outcome

After loading:

Run:

sqlite3 data/sql/prism.db "SELECT system, COUNT(*) FROM indicators GROUP BY system;"


Expected:

economy   (50+)      ← FRED data
benchmark (30+)      ← benchmark A-F from 6 files


And:

sqlite3 data/sql/prism.db "SELECT COUNT(*) FROM timeseries;"


Should show thousands of new rows, depending on # of columns × 1000-row files.

9. Mandatory Implementation Notes

✔ DO NOT overwrite any existing FRED indicators
✔ DO NOT assign FRED codes to benchmarks — only BENCH_*
✔ Benchmarks live in system="benchmark" only
✔ Value_2 and adjusted_value remain NULL for all benchmark inserts
✔ Date parsing must enforce YYYY-MM-DD

10. READY-TO-COPY PR MESSAGE (Paste Into GitHub)
### PR: Benchmark Dataset Loader – Adds Benchmark Integration to PRISM DB

This PR adds a complete loader for all 6 synthetic benchmark datasets.

✔ Adds data/sql/load_benchmarks.py  
✔ Registers CLI flag: --load-benchmarks  
✔ Inserts benchmark indicators → system="benchmark"  
✔ Inserts time series rows from benchmark_01–06  
✔ Adds validation tests under tests/benchmark  
✔ Validates indicator creation, timeseries counts, naming rules

No existing FRED indicators are modified.

After merge, run:

    python prism_run.py --load-benchmarks

This loads all benchmark datasets into the PRISM SQLite database.


Jason — this MD file is now exact, deterministic, unambiguous, and perfect for Claude Code to implement without confusion.

If you'd like:

✔ The unified benchmark runner
✔ The unified scientific analysis report
✔ HTML benchmark dashboards
✔ Benchmark → lens accuracy scoring

Just say the word — we can build the entire scientific test suite next.
