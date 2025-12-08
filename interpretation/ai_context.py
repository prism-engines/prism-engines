"""
PRISM AI Context Builder
=========================

Assembles structured context from PRISM outputs for AI interpretation.

This module bridges the gap between raw database queries and the formatted
context that prompt templates expect. It handles:
- Window context assembly
- Indicator history retrieval
- Event context packaging
- Cross-domain comparison data

Usage:
    from interpretation.ai_context import AIContext

    context = AIContext(db_path="path/to/temporal.db")
    window_ctx = context.build_window_context(window_id=42)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Default paths
_default_temporal_db = Path(__file__).parent.parent / "06_output" / "temporal" / "temporal.db"


def _get_default_prism_db() -> Path:
    """Get default prism DB path using centralized resolver."""
    from data.sql.db_path import get_db_path
    return Path(get_db_path())


@dataclass
class WindowContext:
    """Container for window interpretation context."""

    window_id: int
    start_year: int
    end_year: int
    window_label: str
    n_days: Optional[int] = None
    top_indicators: List[Dict[str, Any]] = field(default_factory=list)
    bottom_indicators: List[Dict[str, Any]] = field(default_factory=list)
    lens_agreement: Dict[str, Any] = field(default_factory=dict)
    coherence_score: float = 0.0
    regime_transition: Optional[Dict[str, Any]] = None
    historical_comparison: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "window_id": self.window_id,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "window_label": self.window_label,
            "n_days": self.n_days,
            "top_indicators": self.top_indicators,
            "bottom_indicators": self.bottom_indicators,
            "lens_agreement": self.lens_agreement,
            "coherence_score": self.coherence_score,
            "regime_transition": self.regime_transition,
            "historical_comparison": self.historical_comparison,
        }


@dataclass
class IndicatorContext:
    """Container for indicator deep-dive context."""

    indicator_name: str
    system: Optional[str] = None
    frequency: Optional[str] = None
    data_start: Optional[str] = None
    data_end: Optional[str] = None
    n_windows: int = 0
    ranking_history: List[Dict[str, Any]] = field(default_factory=list)
    high_importance_periods: List[Dict[str, Any]] = field(default_factory=list)
    low_importance_periods: List[Dict[str, Any]] = field(default_factory=list)
    lens_breakdown: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "indicator_name": self.indicator_name,
            "system": self.system,
            "frequency": self.frequency,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "n_windows": self.n_windows,
            "ranking_history": self.ranking_history,
            "high_importance_periods": self.high_importance_periods,
            "low_importance_periods": self.low_importance_periods,
            "lens_breakdown": self.lens_breakdown,
        }


@dataclass
class EventContext:
    """Container for coherence event context."""

    event_id: int
    event_date: str
    event_type: str
    coherence_score: float
    participating_lenses: List[str] = field(default_factory=list)
    indicator_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    similar_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_date": self.event_date,
            "event_type": self.event_type,
            "coherence_score": self.coherence_score,
            "participating_lenses": self.participating_lenses,
            "indicator_snapshot": self.indicator_snapshot,
            "similar_events": self.similar_events,
        }


@dataclass
class RegimeBreakContext:
    """Container for regime break context."""

    window_before_id: int
    window_after_id: int
    window_before_label: str
    window_after_label: str
    before_start: int
    before_end: int
    after_start: int
    after_end: int
    spearman_corr: float
    p_value: Optional[float] = None
    n_indicators: int = 0
    top_10_overlap: float = 0.0
    rank_changes: List[Dict[str, Any]] = field(default_factory=list)
    historical_breaks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "window_before_id": self.window_before_id,
            "window_after_id": self.window_after_id,
            "window_before_label": self.window_before_label,
            "window_after_label": self.window_after_label,
            "before_start": self.before_start,
            "before_end": self.before_end,
            "after_start": self.after_start,
            "after_end": self.after_end,
            "spearman_corr": self.spearman_corr,
            "p_value": self.p_value,
            "n_indicators": self.n_indicators,
            "top_10_overlap": self.top_10_overlap,
            "rank_changes": self.rank_changes,
            "historical_breaks": self.historical_breaks,
        }


class AIContext:
    """
    Builds structured context from PRISM outputs for AI interpretation.

    This class queries the temporal database and assembles context
    in the format expected by prompt templates.
    """

    def __init__(
        self,
        temporal_db_path: Optional[Path] = None,
        prism_db_path: Optional[Path] = None,
    ):
        """
        Initialize the context builder.

        Args:
            temporal_db_path: Path to temporal analysis database
            prism_db_path: Path to main PRISM database (for indicator metadata)
        """
        self.temporal_db_path = Path(temporal_db_path) if temporal_db_path else _default_temporal_db
        self.prism_db_path = Path(prism_db_path) if prism_db_path else _get_default_prism_db()

    @contextmanager
    def _temporal_connection(self):
        """Context manager for temporal database connections."""
        if not self.temporal_db_path.exists():
            raise FileNotFoundError(f"Temporal database not found: {self.temporal_db_path}")

        conn = sqlite3.connect(str(self.temporal_db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _prism_connection(self):
        """Context manager for PRISM database connections."""
        if not self.prism_db_path.exists():
            raise FileNotFoundError(f"PRISM database not found: {self.prism_db_path}")

        conn = sqlite3.connect(str(self.prism_db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def build_window_context(
        self,
        window_id: int,
        top_n: int = 10,
        bottom_n: int = 5,
    ) -> WindowContext:
        """
        Assemble all relevant data for a specific analysis window.

        Args:
            window_id: ID of the window to analyze
            top_n: Number of top indicators to include
            bottom_n: Number of bottom indicators to include

        Returns:
            WindowContext with assembled data
        """
        with self._temporal_connection() as conn:
            # Get window metadata
            cursor = conn.execute(
                "SELECT id, start_year, end_year, increment, n_days FROM windows WHERE id = ?",
                (window_id,)
            )
            window_row = cursor.fetchone()
            if window_row is None:
                raise ValueError(f"Window {window_id} not found")

            window_label = f"{window_row['start_year']}-{window_row['end_year']}"

            # Get top indicators
            cursor = conn.execute("""
                SELECT
                    i.name as indicator,
                    i.category,
                    c.consensus_rank,
                    c.consensus_score,
                    c.n_lenses
                FROM consensus c
                JOIN indicators i ON c.indicator_id = i.id
                WHERE c.window_id = ?
                ORDER BY c.consensus_rank ASC
                LIMIT ?
            """, (window_id, top_n))
            top_indicators = [dict(row) for row in cursor.fetchall()]

            # Get bottom indicators
            cursor = conn.execute("""
                SELECT
                    i.name as indicator,
                    i.category,
                    c.consensus_rank,
                    c.consensus_score,
                    c.n_lenses
                FROM consensus c
                JOIN indicators i ON c.indicator_id = i.id
                WHERE c.window_id = ?
                ORDER BY c.consensus_rank DESC
                LIMIT ?
            """, (window_id, bottom_n))
            bottom_indicators = [dict(row) for row in cursor.fetchall()]

            # Calculate lens agreement metrics
            lens_agreement = self._calculate_lens_agreement(conn, window_id)

            # Calculate coherence score (average n_lenses normalized)
            cursor = conn.execute("""
                SELECT AVG(n_lenses) as avg_lenses, MAX(n_lenses) as max_lenses
                FROM consensus WHERE window_id = ?
            """, (window_id,))
            lens_stats = cursor.fetchone()
            coherence_score = 0.0
            if lens_stats and lens_stats['max_lenses'] and lens_stats['max_lenses'] > 0:
                coherence_score = lens_stats['avg_lenses'] / lens_stats['max_lenses']

            # Get regime transition info (if this is after a break)
            regime_transition = self._get_regime_transition_for_window(conn, window_id)

            # Get historical comparison
            historical_comparison = self._get_historical_comparison(conn, window_id, top_n)

        return WindowContext(
            window_id=window_id,
            start_year=window_row['start_year'],
            end_year=window_row['end_year'],
            window_label=window_label,
            n_days=window_row['n_days'],
            top_indicators=top_indicators,
            bottom_indicators=bottom_indicators,
            lens_agreement=lens_agreement,
            coherence_score=coherence_score,
            regime_transition=regime_transition,
            historical_comparison=historical_comparison,
        )

    def _calculate_lens_agreement(self, conn, window_id: int) -> Dict[str, Any]:
        """Calculate lens agreement statistics for a window."""
        cursor = conn.execute("""
            SELECT l.name as lens, COUNT(DISTINCT lr.indicator_id) as n_ranked
            FROM lens_results lr
            JOIN lenses l ON lr.lens_id = l.id
            WHERE lr.window_id = ?
            GROUP BY l.id
        """, (window_id,))

        lenses_active = [dict(row) for row in cursor.fetchall()]

        # Get top-10 overlap between lenses
        cursor = conn.execute("""
            SELECT
                l1.name as lens1,
                l2.name as lens2,
                COUNT(*) as overlap
            FROM (
                SELECT lens_id, indicator_id
                FROM lens_results
                WHERE window_id = ? AND rank <= 10
            ) t1
            JOIN (
                SELECT lens_id, indicator_id
                FROM lens_results
                WHERE window_id = ? AND rank <= 10
            ) t2 ON t1.indicator_id = t2.indicator_id AND t1.lens_id < t2.lens_id
            JOIN lenses l1 ON t1.lens_id = l1.id
            JOIN lenses l2 ON t2.lens_id = l2.id
            GROUP BY t1.lens_id, t2.lens_id
        """, (window_id, window_id))

        lens_overlaps = [dict(row) for row in cursor.fetchall()]

        # Determine agreeing vs disagreeing lenses
        # Lenses with high top-10 overlap are "agreeing"
        agreeing = []
        disagreeing = []

        if lens_overlaps:
            avg_overlap = sum(o['overlap'] for o in lens_overlaps) / len(lens_overlaps)
            lens_names = set(l['lens'] for l in lenses_active)

            # Simple heuristic: lenses with above-average overlap are agreeing
            overlap_by_lens = {}
            for o in lens_overlaps:
                overlap_by_lens.setdefault(o['lens1'], []).append(o['overlap'])
                overlap_by_lens.setdefault(o['lens2'], []).append(o['overlap'])

            for lens in lens_names:
                if lens in overlap_by_lens:
                    lens_avg = sum(overlap_by_lens[lens]) / len(overlap_by_lens[lens])
                    if lens_avg >= avg_overlap:
                        agreeing.append(lens)
                    else:
                        disagreeing.append(lens)
                else:
                    disagreeing.append(lens)

        return {
            "lenses_active": lenses_active,
            "lens_overlaps": lens_overlaps,
            "agreeing_lenses": agreeing,
            "disagreeing_lenses": disagreeing,
        }

    def _get_regime_transition_for_window(
        self,
        conn,
        window_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get regime transition info if this window follows a break."""
        cursor = conn.execute("""
            SELECT
                rs.transition_year,
                rs.spearman_corr,
                rs.p_value,
                rs.n_indicators
            FROM regime_stability rs
            WHERE rs.window_after_id = ?
        """, (window_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "transition_year": row['transition_year'],
            "spearman_corr": row['spearman_corr'],
            "p_value": row['p_value'],
            "n_indicators": row['n_indicators'],
            "is_break": row['spearman_corr'] < 0.7,  # Heuristic threshold
        }

    def _get_historical_comparison(
        self,
        conn,
        window_id: int,
        top_n: int
    ) -> Dict[str, Any]:
        """Get comparison with historical windows."""
        # Get current window's top indicators
        cursor = conn.execute("""
            SELECT i.name as indicator
            FROM consensus c
            JOIN indicators i ON c.indicator_id = i.id
            WHERE c.window_id = ?
            ORDER BY c.consensus_rank ASC
            LIMIT ?
        """, (window_id, top_n))
        current_top = {row['indicator'] for row in cursor.fetchall()}

        # Get all previous windows
        cursor = conn.execute("""
            SELECT w2.id, w2.start_year, w2.end_year
            FROM windows w1
            JOIN windows w2 ON w2.start_year < w1.start_year
            WHERE w1.id = ?
            ORDER BY w2.start_year DESC
            LIMIT 5
        """, (window_id,))

        previous_windows = []
        for row in cursor.fetchall():
            # Get that window's top indicators
            cursor2 = conn.execute("""
                SELECT i.name as indicator
                FROM consensus c
                JOIN indicators i ON c.indicator_id = i.id
                WHERE c.window_id = ?
                ORDER BY c.consensus_rank ASC
                LIMIT ?
            """, (row['id'], top_n))
            prev_top = {r['indicator'] for r in cursor2.fetchall()}

            overlap = len(current_top & prev_top)
            previous_windows.append({
                "window_id": row['id'],
                "label": f"{row['start_year']}-{row['end_year']}",
                "overlap_count": overlap,
                "overlap_pct": overlap / top_n * 100 if top_n > 0 else 0,
            })

        return {
            "previous_windows": previous_windows,
            "current_top_indicators": list(current_top),
        }

    def build_indicator_context(
        self,
        indicator_name: str,
    ) -> IndicatorContext:
        """
        Assemble historical context for a single indicator.

        Args:
            indicator_name: Name of the indicator to analyze

        Returns:
            IndicatorContext with assembled data
        """
        context = IndicatorContext(indicator_name=indicator_name)

        # Try to get indicator metadata from PRISM database
        try:
            with self._prism_connection() as conn:
                cursor = conn.execute("""
                    SELECT system, frequency, MIN(date) as data_start, MAX(date) as data_end
                    FROM indicators i
                    LEFT JOIN indicator_values v ON i.id = v.indicator_id
                    WHERE i.name = ?
                    GROUP BY i.id
                """, (indicator_name,))
                row = cursor.fetchone()
                if row:
                    context.system = row['system']
                    context.frequency = row['frequency']
                    context.data_start = row['data_start']
                    context.data_end = row['data_end']
        except FileNotFoundError:
            pass  # PRISM database not available

        # Get ranking history from temporal database
        with self._temporal_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    w.id as window_id,
                    w.start_year || '-' || w.end_year as window_label,
                    w.start_year,
                    w.end_year,
                    c.consensus_rank,
                    c.consensus_score,
                    c.n_lenses
                FROM consensus c
                JOIN windows w ON c.window_id = w.id
                JOIN indicators i ON c.indicator_id = i.id
                WHERE i.name = ?
                ORDER BY w.start_year
            """, (indicator_name,))

            ranking_history = [dict(row) for row in cursor.fetchall()]
            context.ranking_history = ranking_history
            context.n_windows = len(ranking_history)

            if ranking_history:
                # Calculate quartile thresholds
                # Get total indicators per window to determine quartiles
                cursor = conn.execute("""
                    SELECT window_id, COUNT(*) as n_indicators
                    FROM consensus
                    GROUP BY window_id
                """)
                window_counts = {row['window_id']: row['n_indicators'] for row in cursor.fetchall()}

                for entry in ranking_history:
                    n_ind = window_counts.get(entry['window_id'], 1)
                    quartile_threshold = n_ind / 4

                    if entry['consensus_rank'] <= quartile_threshold:
                        context.high_importance_periods.append({
                            "window": entry['window_label'],
                            "rank": entry['consensus_rank'],
                            "percentile": (1 - entry['consensus_rank'] / n_ind) * 100,
                        })
                    elif entry['consensus_rank'] >= n_ind - quartile_threshold:
                        context.low_importance_periods.append({
                            "window": entry['window_label'],
                            "rank": entry['consensus_rank'],
                            "percentile": (1 - entry['consensus_rank'] / n_ind) * 100,
                        })

            # Get lens-specific breakdown
            cursor = conn.execute("""
                SELECT
                    l.name as lens,
                    w.start_year || '-' || w.end_year as window_label,
                    lr.rank,
                    lr.raw_score
                FROM lens_results lr
                JOIN lenses l ON lr.lens_id = l.id
                JOIN windows w ON lr.window_id = w.id
                JOIN indicators i ON lr.indicator_id = i.id
                WHERE i.name = ?
                ORDER BY l.name, w.start_year
            """, (indicator_name,))

            lens_data = {}
            for row in cursor.fetchall():
                lens_name = row['lens']
                if lens_name not in lens_data:
                    lens_data[lens_name] = []
                lens_data[lens_name].append({
                    "window": row['window_label'],
                    "rank": row['rank'],
                    "score": row['raw_score'],
                })

            context.lens_breakdown = lens_data

        return context

    def build_event_context(self, event_id: int) -> EventContext:
        """
        Assemble context around a coherence event.

        Args:
            event_id: ID of the event to analyze

        Returns:
            EventContext with assembled data
        """
        with self._temporal_connection() as conn:
            # Check if coherence_events table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='coherence_events'
            """)
            if cursor.fetchone() is None:
                raise ValueError("coherence_events table not found. Run migration 007 first.")

            # Get event details
            cursor = conn.execute("""
                SELECT * FROM coherence_events WHERE id = ?
            """, (event_id,))
            event_row = cursor.fetchone()
            if event_row is None:
                raise ValueError(f"Event {event_id} not found")

            # Parse JSON fields
            participating_lenses = json.loads(event_row['participating_lenses'] or '[]')
            indicator_snapshot = json.loads(event_row['indicator_snapshot'] or '[]')

            # Get similar events
            cursor = conn.execute("""
                SELECT id, event_date, event_type, coherence_score
                FROM coherence_events
                WHERE event_type = ? AND id != ?
                ORDER BY event_date DESC
                LIMIT 5
            """, (event_row['event_type'], event_id))
            similar_events = [dict(row) for row in cursor.fetchall()]

        return EventContext(
            event_id=event_id,
            event_date=event_row['event_date'],
            event_type=event_row['event_type'],
            coherence_score=event_row['coherence_score'],
            participating_lenses=participating_lenses,
            indicator_snapshot=indicator_snapshot,
            similar_events=similar_events,
        )

    def build_regime_break_context(
        self,
        window_before_id: int,
        window_after_id: int,
        top_n: int = 10,
    ) -> RegimeBreakContext:
        """
        Assemble context for a regime transition between windows.

        Args:
            window_before_id: ID of the window before the break
            window_after_id: ID of the window after the break
            top_n: Number of top indicators to compare

        Returns:
            RegimeBreakContext with assembled data
        """
        with self._temporal_connection() as conn:
            # Get window metadata
            cursor = conn.execute("""
                SELECT id, start_year, end_year FROM windows WHERE id IN (?, ?)
            """, (window_before_id, window_after_id))

            windows = {row['id']: dict(row) for row in cursor.fetchall()}
            if window_before_id not in windows or window_after_id not in windows:
                raise ValueError("One or both windows not found")

            before = windows[window_before_id]
            after = windows[window_after_id]

            # Get regime stability data
            cursor = conn.execute("""
                SELECT spearman_corr, p_value, n_indicators, transition_year
                FROM regime_stability
                WHERE window_before_id = ? AND window_after_id = ?
            """, (window_before_id, window_after_id))

            stability_row = cursor.fetchone()
            spearman_corr = stability_row['spearman_corr'] if stability_row else 0.0
            p_value = stability_row['p_value'] if stability_row else None
            n_indicators = stability_row['n_indicators'] if stability_row else 0

            # Get rankings for both windows
            cursor = conn.execute("""
                SELECT
                    i.name as indicator,
                    c.consensus_rank
                FROM consensus c
                JOIN indicators i ON c.indicator_id = i.id
                WHERE c.window_id = ?
            """, (window_before_id,))
            before_ranks = {row['indicator']: row['consensus_rank'] for row in cursor.fetchall()}

            cursor = conn.execute("""
                SELECT
                    i.name as indicator,
                    c.consensus_rank
                FROM consensus c
                JOIN indicators i ON c.indicator_id = i.id
                WHERE c.window_id = ?
            """, (window_after_id,))
            after_ranks = {row['indicator']: row['consensus_rank'] for row in cursor.fetchall()}

            # Calculate rank changes
            rank_changes = []
            for indicator in set(before_ranks.keys()) | set(after_ranks.keys()):
                before_rank = before_ranks.get(indicator)
                after_rank = after_ranks.get(indicator)
                if before_rank is not None and after_rank is not None:
                    change = before_rank - after_rank  # Positive = improved
                    rank_changes.append({
                        "indicator": indicator,
                        "before_rank": before_rank,
                        "after_rank": after_rank,
                        "change": change,
                        "abs_change": abs(change),
                    })

            # Sort by absolute change
            rank_changes.sort(key=lambda x: x['abs_change'], reverse=True)

            # Calculate top-10 overlap
            before_top10 = set(
                k for k, v in sorted(before_ranks.items(), key=lambda x: x[1])[:top_n]
            )
            after_top10 = set(
                k for k, v in sorted(after_ranks.items(), key=lambda x: x[1])[:top_n]
            )
            overlap = len(before_top10 & after_top10)
            top_10_overlap = overlap / top_n * 100 if top_n > 0 else 0

            # Get historical regime breaks
            cursor = conn.execute("""
                SELECT
                    rs.transition_year,
                    rs.spearman_corr,
                    w1.start_year || '-' || w1.end_year as window_before,
                    w2.start_year || '-' || w2.end_year as window_after
                FROM regime_stability rs
                JOIN windows w1 ON rs.window_before_id = w1.id
                JOIN windows w2 ON rs.window_after_id = w2.id
                WHERE rs.spearman_corr < 0.7
                ORDER BY rs.transition_year DESC
                LIMIT 5
            """)
            historical_breaks = [dict(row) for row in cursor.fetchall()]

        return RegimeBreakContext(
            window_before_id=window_before_id,
            window_after_id=window_after_id,
            window_before_label=f"{before['start_year']}-{before['end_year']}",
            window_after_label=f"{after['start_year']}-{after['end_year']}",
            before_start=before['start_year'],
            before_end=before['end_year'],
            after_start=after['start_year'],
            after_end=after['end_year'],
            spearman_corr=spearman_corr,
            p_value=p_value,
            n_indicators=n_indicators,
            top_10_overlap=top_10_overlap,
            rank_changes=rank_changes[:20],  # Top 20 changes
            historical_breaks=historical_breaks,
        )

    def build_cross_domain_context(
        self,
        systems: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assemble context comparing multiple systems/domains.

        Args:
            systems: List of system names to compare
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary with cross-domain comparison data
        """
        # This requires data from multiple systems which may not be
        # in the same temporal database. For now, return a placeholder
        # structure that can be populated when cross-domain data is available.

        return {
            "systems": systems,
            "start_date": start_date,
            "end_date": end_date,
            "correlation_matrix": {},
            "synchronized_events": [],
            "divergent_periods": [],
            "note": "Cross-domain analysis requires data from multiple systems. "
                    "Ensure data for all requested systems is available.",
        }

    def build_query_context(
        self,
        question: str,
        window_ids: Optional[List[int]] = None,
        indicator_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build context for answering a natural language question.

        Args:
            question: The user's question
            window_ids: Optional list of window IDs to focus on
            indicator_names: Optional list of indicators to focus on

        Returns:
            Dictionary with context for answering the question
        """
        context = {
            "question": question,
            "available_data": {},
            "context_summary": "",
            "context_details": "",
        }

        with self._temporal_connection() as conn:
            # Get available windows
            cursor = conn.execute("""
                SELECT id, start_year, end_year, n_days
                FROM windows ORDER BY start_year
            """)
            windows = [dict(row) for row in cursor.fetchall()]
            context["available_data"]["windows"] = windows

            # Get available indicators
            cursor = conn.execute("""
                SELECT DISTINCT i.name, i.category
                FROM indicators i
                ORDER BY i.name
            """)
            indicators = [dict(row) for row in cursor.fetchall()]
            context["available_data"]["indicators"] = indicators

            # Get regime stability summary
            cursor = conn.execute("""
                SELECT transition_year, spearman_corr
                FROM regime_stability
                ORDER BY transition_year
            """)
            regimes = [dict(row) for row in cursor.fetchall()]
            context["available_data"]["regime_stability"] = regimes

            # Build summary
            summary_parts = []
            if windows:
                summary_parts.append(
                    f"Analysis windows: {len(windows)} windows from "
                    f"{windows[0]['start_year']} to {windows[-1]['end_year']}"
                )
            if indicators:
                summary_parts.append(f"Indicators: {len(indicators)} tracked")
            if regimes:
                breaks = [r for r in regimes if r['spearman_corr'] < 0.7]
                summary_parts.append(f"Regime breaks detected: {len(breaks)}")

            context["context_summary"] = "; ".join(summary_parts)

            # If specific windows or indicators requested, get detailed data
            if window_ids:
                context["focused_windows"] = []
                for wid in window_ids[:3]:  # Limit to 3 windows
                    try:
                        wctx = self.build_window_context(wid)
                        context["focused_windows"].append(wctx.to_dict())
                    except ValueError:
                        continue

            if indicator_names:
                context["focused_indicators"] = []
                for name in indicator_names[:5]:  # Limit to 5 indicators
                    try:
                        ictx = self.build_indicator_context(name)
                        context["focused_indicators"].append(ictx.to_dict())
                    except ValueError:
                        continue

        return context
