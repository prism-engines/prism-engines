# PRISM Fetchers — Canonical Standards

This directory contains all **data fetchers** used by PRISM.

Fetchers are responsible for **retrieving raw time series data from external sources**
(e.g. FRED, Tiingo, Yahoo, NOAA), and **nothing else**.

If you have questions about how fetching is supposed to work, **this README is the source of truth**.

---

## Design Philosophy (Non-Negotiable)

PRISM fetchers are built on the following principles:

1. **Domain-agnostic**
2. **Source-specific**
3. **Mechanics centralized**
4. **Observable at runtime**
5. **Silent failure is forbidden**

Fetchers do not decide *what* to run, *why* it is run, or *where* the data is stored.
They only know **how to fetch** from a specific source, reliably.

---

## File Structure

prism/fetchers/
├── README.md ← this file (canonical standard)
├── base_fetcher.py ← abstract base class (infrastructure)
├── fred.py ← FRED source fetcher
├── tiingo.py ← Tiingo source fetcher
├── yahoo.py ← Yahoo / market data fetcher
└── ...


---

## `base_fetcher.py` (Core Infrastructure)

### Naming standard

- File name: **`base_fetcher.py`**
- Lowercase
- Snake_case
- No abbreviations
- No alternative spellings

This is intentional and enforced.

---

### What `BaseFetcher` IS

`BaseFetcher` provides **shared mechanics** for all fetchers:

- retry logic
- validation
- schema normalization
- failure framing
- live fetch status emission (START / OK / FAIL)
- attempt tracking

It is **abstract** and must never be instantiated directly.

---

### What `BaseFetcher` is NOT

`BaseFetcher` does **not**:

- read `pipeline.yaml`
- know about `domain_id`
- select indicators
- choose data sources
- write to DuckDB
- control execution order
- act as a runner or orchestrator

If any of the above appear in `base_fetcher.py`, it is a design violation.

---

## Source-Specific Fetchers (e.g. `fred.py`, `tiingo.py`)

Each concrete fetcher:

- subclasses `BaseFetcher`
- implements exactly **one method**:

```python
def _fetch_once(self) -> pd.DataFrame:
    ...

Contract for _fetch_once

It must return a pandas.DataFrame with at least:

    date

    value

It must not:

    add PRISM metadata columns

    handle retries

    emit status

    catch and suppress errors

All of that is handled by BaseFetcher.
Execution Model (Important)

Fetchers do not run themselves.

There are two valid execution modes:
1. Manual / Interactive Mode (bootstrap & testing)

You may manually instantiate and run a fetcher:

fetcher = FredFetcher(
    indicator_id="CPI_YOY",
    source="fred",
    params={}
)

df = fetcher.fetch()

In this mode, the developer is acting as the runner.

This is valid and supported.
2. Pipeline / Runner Mode (intended steady state)

In production:

    a runner parses pipeline.yaml

    resolves indicators → sources

    instantiates concrete fetchers

    calls fetcher.fetch()

    persists results to DuckDB

BaseFetcher behaves identically in both modes.
Observability (Mandatory)

Every fetch attempt must be observable at runtime.

PRISM enforces this via stderr-based status emission:

[FETCH START ] CPI_YOY
[FETCH OK    ] CPI_YOY — rows=914
[FETCH FAIL  ] TLT — HTTP 429

This output must be visible even when stdout is captured.

Print statements to stdout are not allowed for fetch status.
Failure Semantics (Strict)

Failure is first-class data.

Rules:

    A failed fetch must:

        emit FETCH FAIL

        return a failure row

        be persisted by the runner

    Silent failure is not allowed

    Empty success is not allowed

If an indicator fails, PRISM must know it failed.
Relationship to Domains

Domains (economic, market, climate, etc.) are planning concepts.

They live in:

    pipeline.yaml

    indicator registries

    runner logic

    downstream analysis

Domains must never leak into fetcher logic.

Fetchers operate on indicator_id, not domain_id.
Relationship to DuckDB

Fetchers do not write to DuckDB.

They return a DataFrame.

Persistence is the responsibility of the runner.

This separation is intentional and required.
If You Are Unsure

If you are unsure where a piece of logic belongs:

    Fetch mechanics? → base_fetcher.py

    Source API logic? → source fetcher (fred.py, etc.)

    Indicator selection? → runner / registry

    Persistence? → DuckDB layer

    Domain meaning? → analysis / lenses

When in doubt, do not add logic to fetchers.

Consult this README first.
Final Note

This directory is foundational.

Breaking fetch semantics breaks:

    lenses

    cloud compute

    reproducibility

    trust in results

Changes here must be deliberate, minimal, and reviewed against this standard.

When in doubt: read this README again.


---

If you want, next I can:

- review your existing `fred.py` / `tiingo.py` against this standard
- add a short checklist comment at the top of `base_fetcher.py`
- generate a matching README for `prism/observe`
- or write the DuckDB `fetch_runs` migration that this README implicitly relies on

This README is the right move — it locks in correctness before scale.

