"""
Data Performance Diagnostics

Checks data fetching and query performance.
"""

import os
import time
import sqlite3
from pathlib import Path
from typing import Dict

from diagnostics.core.base import (
    BaseDiagnostic,
    DiagnosticCategory,
    DiagnosticResult,
)


def get_db_path() -> str:
    """Get the database path using the standard PRISM method."""
    try:
        from data.sql.db_path import get_db_path as prism_get_db_path
        return prism_get_db_path()
    except ImportError:
        return os.path.expanduser("~/prism_data/prism.db")


class DataFetchDiagnostic(BaseDiagnostic):
    """Test data fetching performance."""

    name = "data_fetch"
    description = "Test data fetching from database"
    category = DiagnosticCategory.PERFORMANCE
    is_quick = True

    MAX_FETCH_TIME_MS = 5000  # 5 seconds max
    WARN_FETCH_TIME_MS = 1000  # Warn over 1 second

    def check(self) -> DiagnosticResult:
        db_path = get_db_path()

        if not os.path.exists(db_path):
            return self.skip_result("Database not available")

        try:
            conn = sqlite3.connect(db_path)

            # Test simple query
            start = time.time()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM indicator_data"
            )
            row_count = cursor.fetchone()[0]
            count_time_ms = (time.time() - start) * 1000

            # Test fetching sample data
            start = time.time()
            cursor = conn.execute(
                "SELECT * FROM indicator_data LIMIT 1000"
            )
            rows = cursor.fetchall()
            fetch_time_ms = (time.time() - start) * 1000

            conn.close()

            total_time_ms = count_time_ms + fetch_time_ms
            details = {
                "total_rows": row_count,
                "count_query_ms": round(count_time_ms, 2),
                "fetch_1000_rows_ms": round(fetch_time_ms, 2),
                "total_time_ms": round(total_time_ms, 2),
            }

            if total_time_ms > self.MAX_FETCH_TIME_MS:
                return self.fail_result(
                    f"Data fetch too slow: {total_time_ms:.0f}ms",
                    details=details,
                    suggestions=[
                        "Check database indexes",
                        "Consider running VACUUM on database"
                    ]
                )
            elif total_time_ms > self.WARN_FETCH_TIME_MS:
                return self.warn_result(
                    f"Data fetch slow: {total_time_ms:.0f}ms",
                    details=details
                )
            else:
                return self.pass_result(
                    f"Fetched from {row_count:,} rows in {total_time_ms:.0f}ms",
                    details=details
                )

        except sqlite3.Error as e:
            return self.error_result(f"Data fetch failed: {e}")


class DatabaseQueryDiagnostic(BaseDiagnostic):
    """Test database query patterns."""

    name = "database_query"
    description = "Test common database query patterns"
    category = DiagnosticCategory.PERFORMANCE
    is_quick = True

    def check(self) -> DiagnosticResult:
        db_path = get_db_path()

        if not os.path.exists(db_path):
            return self.skip_result("Database not available")

        try:
            conn = sqlite3.connect(db_path)
            query_times = {}

            # Test 1: Get indicators list
            start = time.time()
            cursor = conn.execute(
                "SELECT DISTINCT indicator_id FROM indicator_data"
            )
            indicators = cursor.fetchall()
            query_times['list_indicators'] = (time.time() - start) * 1000

            # Test 2: Date range query
            start = time.time()
            cursor = conn.execute("""
                SELECT date, value FROM indicator_data
                WHERE indicator_id = (SELECT indicator_id FROM indicator_data LIMIT 1)
                ORDER BY date
            """)
            cursor.fetchall()
            query_times['date_range'] = (time.time() - start) * 1000

            # Test 3: Aggregate query
            start = time.time()
            cursor = conn.execute("""
                SELECT indicator_id, COUNT(*), MIN(date), MAX(date)
                FROM indicator_data
                GROUP BY indicator_id
            """)
            cursor.fetchall()
            query_times['aggregate'] = (time.time() - start) * 1000

            conn.close()

            total_time_ms = sum(query_times.values())
            avg_time_ms = total_time_ms / len(query_times)

            details = {
                "query_times_ms": {k: round(v, 2) for k, v in query_times.items()},
                "total_time_ms": round(total_time_ms, 2),
                "avg_time_ms": round(avg_time_ms, 2),
                "indicators_count": len(indicators),
            }

            # Check for slow queries
            slow_queries = [k for k, v in query_times.items() if v > 1000]

            if slow_queries:
                return self.warn_result(
                    f"Slow queries detected: {', '.join(slow_queries)}",
                    details=details,
                    suggestions=["Add indexes for commonly queried columns"]
                )
            else:
                return self.pass_result(
                    f"All queries OK (avg {avg_time_ms:.0f}ms)",
                    details=details
                )

        except sqlite3.Error as e:
            return self.error_result(f"Query test failed: {e}")
