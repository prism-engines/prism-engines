"""
PRISM Fetch Runner

Orchestrates the fetch → clean → store pipeline.

Architecture:
    FetchRunner coordinates:
        1. Fetcher gets raw data from source
        2. Cleaner processes raw → clean
        3. Writer stores both to DB (data.raw_indicators + data.indicators)

CANONICAL PHASE MODEL:
    data      - fetched + cleaned + admissible indicator data
    derived   - first-order engine measurements
    structure - geometric organization
    binding   - final attachment / positioning
    meta      - orchestration, audits, windows
"""

import logging
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import uuid

import duckdb
import pandas as pd

from prism.db.open import open_prism_db
from prism.fetch.fred import FredFetcher
from prism.fetch.tiingo import TiingoFetcher
from prism.cleaning import clean_series

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Fetcher dispatch (NO external fetchers module)
# ---------------------------------------------------------------------
def get_fetcher(source: str):
    if source == "fred":
        return FredFetcher()
    elif source == "tiingo":
        return TiingoFetcher()
    else:
        raise ValueError(f"Unknown data source: {source}")


# ---------------------------------------------------------------------
# Summary object
# ---------------------------------------------------------------------
@dataclass
class FetchSummary:
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.succeeded + self.failed + self.skipped

    def print_summary(self):
        duration = ""
        if self.completed_at:
            duration = f" ({(self.completed_at - self.started_at).total_seconds():.1f}s)"

        print("\n" + "=" * 60)
        print(f"Fetch Run: {self.run_id}")
        print("=" * 60)
        print(f"Total:     {self.total}")
        print(f"Succeeded: {self.succeeded}")
        print(f"Failed:    {self.failed}")
        print(f"Skipped:   {self.skipped}")
        print(f"Completed: {self.completed_at}{duration}")
        print()


# ---------------------------------------------------------------------
# FetchRunner
# ---------------------------------------------------------------------
class FetchRunner:

    def __init__(self):
        # Schema guardrail: verify canonical table exists
        self._assert_canonical_schema()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        return open_prism_db()

    def _assert_canonical_schema(self):
        """Verify canonical data.indicators table exists."""
        with self._get_connection() as conn:
            exists = conn.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'data'
                  AND table_name = 'indicators'
            """).fetchone()[0]

        assert exists == 1, (
            "Canonical table data.indicators missing. "
            "Run migration: prism/db/migrations/2025_12_21_canonical_data_schema.sql"
        )

    # ---------------------------------------------------------------
    def run(
        self,
        indicators: List[Dict[str, Any]],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> FetchSummary:

        run_id = self._generate_run_id()
        summary = FetchSummary(run_id=run_id, started_at=datetime.now())

        logger.info(f"Starting fetch run {run_id} for {len(indicators)} indicators")

        self._record_run_start(run_id, len(indicators))

        for cfg in indicators:
            indicator_id = cfg.get("indicator_id") or cfg.get("id")
            source = cfg["source"]
            frequency = cfg.get("frequency", "daily")

            try:
                result = self._fetch_one(
                    indicator_id,
                    source,
                    frequency,
                    start_date,
                    end_date,
                    run_id,
                )
                summary.results.append(result)

                if result["status"] == "success":
                    summary.succeeded += 1
                else:
                    summary.failed += 1

            except Exception as e:
                logger.exception(f"Unexpected error fetching {indicator_id}")
                summary.failed += 1
                summary.results.append({
                    "indicator_id": indicator_id,
                    "source": source,
                    "status": "failed",
                    "error": str(e),
                })

        summary.completed_at = datetime.now()
        self._record_run_complete(run_id, summary)

        logger.info(
            f"Fetch run {run_id} complete: "
            f"{summary.succeeded} succeeded, {summary.failed} failed"
        )

        return summary

    # ---------------------------------------------------------------
    def _fetch_one(
        self,
        indicator_id: str,
        source: str,
        frequency: str,
        start_date: Optional[date],
        end_date: Optional[date],
        run_id: str,
    ) -> Dict[str, Any]:

        logger.info(f"Fetching {indicator_id} from {source}")
        fetcher = get_fetcher(source)

        fetch_result = fetcher.fetch(
            indicator_id=indicator_id,
            start_date=start_date,
            end_date=end_date,
        )

        if not fetch_result.success:
            return {
                "indicator_id": indicator_id,
                "source": source,
                "status": "failed",
                "error": fetch_result.error,
            }

        raw_df = fetch_result.data

        clean_result = clean_series(raw_df, frequency=frequency)
        clean_df = clean_result.data

        # Write directly to canonical data.indicators (fetch + clean combined)
        self._write_to_data(indicator_id, source, frequency, clean_df, clean_result.method, run_id)

        self._record_fetch_result(
            run_id,
            indicator_id,
            source,
            "success",
            rows_raw=len(raw_df),
            rows_clean=len(clean_df),
            first_date=clean_df["date"].min() if not clean_df.empty else None,
            last_date=clean_df["date"].max() if not clean_df.empty else None,
        )

        return {
            "indicator_id": indicator_id,
            "source": source,
            "status": "success",
            "rows_raw": len(raw_df),
            "rows_clean": len(clean_df),
            "cleaning_method": clean_result.method,
        }

    # ---------------------------------------------------------------
    def _write_to_data(self, indicator_id, source, frequency, df, cleaning_method, run_id):
        """
        Write to canonical data.indicators table.

        This combines fetch + clean into a single write to the canonical
        data schema. No separate raw/clean tables - data.indicators is
        the single source of truth.
        """
        if df.empty:
            return

        now = datetime.now()
        write_df = df.copy()
        write_df["indicator_id"] = indicator_id
        write_df["source"] = source
        write_df["frequency"] = frequency
        write_df["cleaning_method"] = cleaning_method
        write_df["fetched_at"] = now
        write_df["cleaned_at"] = now
        write_df["run_id"] = run_id

        write_df = write_df[
            ["indicator_id", "date", "value", "source", "frequency",
             "cleaning_method", "fetched_at", "cleaned_at", "run_id"]
        ]

        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM data.indicators WHERE indicator_id = ?",
                [indicator_id],
            )
            conn.register("write_df", write_df)
            conn.execute("INSERT INTO data.indicators SELECT * FROM write_df")

    # ---------------------------------------------------------------
    # DEPRECATED: Legacy methods kept for reference
    # ---------------------------------------------------------------
    def _write_raw(self, indicator_id, source, frequency, df, run_id):
        """Deprecated: Use _write_to_data() instead."""
        # Canonical rule: cleaning is part of the data phase, not separate
        pass

    def _write_clean(self, indicator_id, df, method, run_id):
        """Deprecated: Use _write_to_data() instead."""
        # Canonical rule: cleaning writes directly to data.indicators
        pass

    # ---------------------------------------------------------------
    def _record_run_start(self, run_id, count):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO meta.fetch_runs
                (run_id, started_at, status, indicators_requested)
                VALUES (?, ?, 'running', ?)
                """,
                [run_id, datetime.now(), count],
            )

    def _record_run_complete(self, run_id, summary):
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE meta.fetch_runs
                SET completed_at = ?,
                    status = 'completed',
                    indicators_succeeded = ?,
                    indicators_failed = ?
                WHERE run_id = ?
                """,
                [summary.completed_at, summary.succeeded, summary.failed, run_id],
            )

    def _record_fetch_result(
        self,
        run_id,
        indicator_id,
        source,
        status,
        rows_raw=0,
        rows_clean=0,
        first_date=None,
        last_date=None,
        error=None,
    ):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO meta.fetch_results
                (run_id, indicator_id, source, status,
                 rows_fetched, first_date, last_date,
                 error_message, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    indicator_id,
                    source,
                    status,
                    rows_clean,
                    first_date,
                    last_date,
                    error,
                    datetime.now(),
                ],
            )

    def _generate_run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"fetch_{ts}_{uuid.uuid4().hex[:8]}"
