"""
PRISM Interpreter
=================

Main interpretation engine for generating AI interpretations of PRISM outputs.

Supports multiple backends:
- manual: Generates prompts for user to submit to their preferred AI
- claude: Uses Anthropic's Claude API
- openai: Uses OpenAI's API
- ollama: Uses local Ollama instance

Usage:
    from interpretation import PRISMInterpreter

    # Manual mode (generates prompts)
    interpreter = PRISMInterpreter(backend="manual")
    result = interpreter.interpret_window(window_id=42)
    print(result.prompt_used)  # Copy to Claude/GPT

    # API mode (requires API key)
    interpreter = PRISMInterpreter(
        backend="claude",
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    result = interpreter.interpret_window(window_id=42)
    print(result.interpretation)
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ai_context import (
    AIContext,
    EventContext,
    IndicatorContext,
    RegimeBreakContext,
    WindowContext,
)
from .prompt_templates import TEMPLATES, get_template, render_template


# Default paths
_default_temporal_db = Path(__file__).parent.parent / "06_output" / "temporal" / "temporal.db"
_default_interpretation_db = Path(__file__).parent.parent / "06_output" / "temporal" / "temporal.db"


@dataclass
class InterpretationResult:
    """Container for AI interpretation with metadata."""

    interpretation: str
    prompt_used: str
    template_name: str
    context_snapshot: Dict[str, Any]
    backend: str
    model: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    interpretation_id: Optional[int] = None
    human_feedback: Optional[str] = None
    validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "interpretation": self.interpretation,
            "prompt_used": self.prompt_used,
            "template_name": self.template_name,
            "context_snapshot": self.context_snapshot,
            "backend": self.backend,
            "model": self.model,
            "timestamp": self.timestamp,
            "interpretation_id": self.interpretation_id,
            "human_feedback": self.human_feedback,
            "validated": self.validated,
        }


class PRISMInterpreter:
    """
    Generates AI interpretations of PRISM outputs.

    Supports multiple backends:
    - manual: Returns prompt for human to submit
    - claude: Calls Anthropic Claude API
    - openai: Calls OpenAI API
    - ollama: Calls local Ollama instance
    """

    def __init__(
        self,
        backend: str = "manual",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temporal_db_path: Optional[Path] = None,
        save_interpretations: bool = True,
    ):
        """
        Initialize the interpreter.

        Args:
            backend: AI backend to use ('manual', 'claude', 'openai', 'ollama')
            model: Model to use (defaults based on backend)
            api_key: API key for claude/openai backends
            temporal_db_path: Path to temporal database
            save_interpretations: Whether to save interpretations to database
        """
        self.backend = backend.lower()
        self.api_key = api_key or self._get_api_key_from_env()
        self.save_interpretations = save_interpretations

        # Set default models
        if model is None:
            model = self._get_default_model()
        self.model = model

        # Initialize context builder
        self.temporal_db_path = Path(temporal_db_path) if temporal_db_path else _default_temporal_db
        self.context_builder = AIContext(temporal_db_path=self.temporal_db_path)

        # Validate backend
        valid_backends = ["manual", "claude", "openai", "ollama"]
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {backend}. Valid: {valid_backends}")

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        if self.backend == "claude":
            return os.environ.get("ANTHROPIC_API_KEY")
        elif self.backend == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None

    def _get_default_model(self) -> str:
        """Get default model for backend."""
        defaults = {
            "claude": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "ollama": "llama3.1:70b",
            "manual": "manual",
        }
        return defaults.get(self.backend, "manual")

    @contextmanager
    def _db_connection(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(str(self.temporal_db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _format_indicators_table(self, indicators: List[Dict[str, Any]]) -> str:
        """Format indicators as a markdown table."""
        if not indicators:
            return "No indicators available"

        lines = ["| Rank | Indicator | Category | Score | Lenses |",
                 "|------|-----------|----------|-------|--------|"]
        for i, ind in enumerate(indicators, 1):
            lines.append(
                f"| {ind.get('consensus_rank', i)} | {ind.get('indicator', 'N/A')} | "
                f"{ind.get('category', 'N/A')} | {ind.get('consensus_score', 0):.3f} | "
                f"{ind.get('n_lenses', 0)} |"
            )
        return "\n".join(lines)

    def _format_rank_changes_table(self, changes: List[Dict[str, Any]]) -> str:
        """Format rank changes as a markdown table."""
        if not changes:
            return "No significant rank changes"

        lines = ["| Indicator | Before | After | Change |",
                 "|-----------|--------|-------|--------|"]
        for ch in changes[:15]:  # Top 15 changes
            direction = "+" if ch['change'] > 0 else ""
            lines.append(
                f"| {ch['indicator']} | {ch['before_rank']:.0f} | "
                f"{ch['after_rank']:.0f} | {direction}{ch['change']:.0f} |"
            )
        return "\n".join(lines)

    def _format_ranking_history_table(self, history: List[Dict[str, Any]]) -> str:
        """Format ranking history as a markdown table."""
        if not history:
            return "No ranking history available"

        lines = ["| Window | Rank | Score | Lenses |",
                 "|--------|------|-------|--------|"]
        for h in history:
            lines.append(
                f"| {h.get('window_label', 'N/A')} | {h.get('consensus_rank', 0):.0f} | "
                f"{h.get('consensus_score', 0):.3f} | {h.get('n_lenses', 0)} |"
            )
        return "\n".join(lines)

    def _format_historical_breaks(self, breaks: List[Dict[str, Any]]) -> str:
        """Format historical regime breaks."""
        if not breaks:
            return "No previous regime breaks detected in dataset"

        lines = []
        for b in breaks:
            lines.append(
                f"- Year {b.get('transition_year', 'N/A')}: "
                f"{b.get('window_before', 'N/A')} → {b.get('window_after', 'N/A')} "
                f"(correlation: {b.get('spearman_corr', 0):.3f})"
            )
        return "\n".join(lines)

    def interpret_window(self, window_id: int) -> InterpretationResult:
        """
        Generate interpretation for a specific analysis window.

        Args:
            window_id: ID of the window to interpret

        Returns:
            InterpretationResult with interpretation and metadata
        """
        # Build context
        context = self.context_builder.build_window_context(window_id)

        # Prepare template variables
        template_vars = {
            "window_label": context.window_label,
            "start_date": str(context.start_year),
            "end_date": str(context.end_year),
            "top_indicators_table": self._format_indicators_table(context.top_indicators),
            "coherence_score": context.coherence_score,
            "agreeing_lenses": ", ".join(context.lens_agreement.get("agreeing_lenses", [])) or "None identified",
            "disagreeing_lenses": ", ".join(context.lens_agreement.get("disagreeing_lenses", [])) or "None identified",
            "regime_transition_text": self._format_regime_transition(context.regime_transition),
            "historical_comparison": self._format_historical_comparison(context.historical_comparison),
        }

        # Render prompt
        prompt = render_template("window_summary", **template_vars)

        # Call backend
        interpretation = self._call_backend(prompt)

        # Create result
        result = InterpretationResult(
            interpretation=interpretation,
            prompt_used=prompt,
            template_name="window_summary",
            context_snapshot=context.to_dict(),
            backend=self.backend,
            model=self.model,
        )

        # Save if enabled
        if self.save_interpretations:
            result.interpretation_id = self._save_interpretation(
                interpretation_type="window",
                target_id=window_id,
                result=result,
            )

        return result

    def _format_regime_transition(self, transition: Optional[Dict[str, Any]]) -> str:
        """Format regime transition info."""
        if transition is None:
            return "No regime transition detected for this window."

        if transition.get("is_break"):
            return (
                f"**Regime Break Detected**: Transition year {transition.get('transition_year')}. "
                f"Spearman correlation with previous window: {transition.get('spearman_corr', 0):.3f} "
                f"(p-value: {transition.get('p_value', 'N/A')}). "
                f"This indicates a significant structural shift in indicator importance."
            )
        else:
            return (
                f"Regime Stable: Spearman correlation with previous window: "
                f"{transition.get('spearman_corr', 0):.3f}. "
                f"Indicator rankings show continuity from the previous period."
            )

    def _format_historical_comparison(self, comparison: Optional[Dict[str, Any]]) -> str:
        """Format historical comparison."""
        if comparison is None or not comparison.get("previous_windows"):
            return "No historical windows available for comparison."

        lines = ["Comparison with previous windows:"]
        for pw in comparison["previous_windows"]:
            lines.append(
                f"- {pw['label']}: {pw['overlap_count']} of top 10 indicators match "
                f"({pw['overlap_pct']:.0f}% overlap)"
            )
        return "\n".join(lines)

    def interpret_event(self, event_id: int) -> InterpretationResult:
        """
        Generate interpretation for a coherence event.

        Args:
            event_id: ID of the event to interpret

        Returns:
            InterpretationResult with interpretation and metadata
        """
        # Build context
        context = self.context_builder.build_event_context(event_id)

        # Prepare template variables
        template_vars = {
            "event_date": context.event_date,
            "event_type": context.event_type,
            "coherence_score": context.coherence_score,
            "lens_list": ", ".join(context.participating_lenses) or "Not specified",
            "indicator_snapshot": self._format_indicator_snapshot(context.indicator_snapshot),
            "similar_events": self._format_similar_events(context.similar_events),
        }

        # Render prompt
        prompt = render_template("coherence_event", **template_vars)

        # Call backend
        interpretation = self._call_backend(prompt)

        # Create result
        result = InterpretationResult(
            interpretation=interpretation,
            prompt_used=prompt,
            template_name="coherence_event",
            context_snapshot=context.to_dict(),
            backend=self.backend,
            model=self.model,
        )

        # Save if enabled
        if self.save_interpretations:
            result.interpretation_id = self._save_interpretation(
                interpretation_type="event",
                target_id=event_id,
                result=result,
            )

        return result

    def _format_indicator_snapshot(self, snapshot: List[Dict[str, Any]]) -> str:
        """Format indicator snapshot."""
        if not snapshot:
            return "No indicator snapshot available"

        lines = ["| Indicator | Rank | Score |",
                 "|-----------|------|-------|"]
        for s in snapshot[:10]:
            lines.append(f"| {s.get('indicator', 'N/A')} | {s.get('rank', 'N/A')} | {s.get('score', 'N/A')} |")
        return "\n".join(lines)

    def _format_similar_events(self, events: List[Dict[str, Any]]) -> str:
        """Format similar events."""
        if not events:
            return "No similar events found in historical data"

        lines = []
        for e in events:
            lines.append(
                f"- {e.get('event_date', 'N/A')}: {e.get('event_type', 'N/A')} "
                f"(coherence: {e.get('coherence_score', 0):.2f})"
            )
        return "\n".join(lines)

    def interpret_regime_break(
        self,
        window_before_id: int,
        window_after_id: int,
    ) -> InterpretationResult:
        """
        Generate interpretation for a regime transition.

        Args:
            window_before_id: ID of window before the break
            window_after_id: ID of window after the break

        Returns:
            InterpretationResult with interpretation and metadata
        """
        # Build context
        context = self.context_builder.build_regime_break_context(
            window_before_id, window_after_id
        )

        # Prepare template variables
        template_vars = {
            "window_before_label": context.window_before_label,
            "before_start": str(context.before_start),
            "before_end": str(context.before_end),
            "window_after_label": context.window_after_label,
            "after_start": str(context.after_start),
            "after_end": str(context.after_end),
            "spearman_corr": context.spearman_corr,
            "top_10_overlap": context.top_10_overlap,
            "n_indicators": context.n_indicators,
            "rank_change_table": self._format_rank_changes_table(context.rank_changes),
            "historical_breaks": self._format_historical_breaks(context.historical_breaks),
        }

        # Render prompt
        prompt = render_template("regime_break", **template_vars)

        # Call backend
        interpretation = self._call_backend(prompt)

        # Create result
        result = InterpretationResult(
            interpretation=interpretation,
            prompt_used=prompt,
            template_name="regime_break",
            context_snapshot=context.to_dict(),
            backend=self.backend,
            model=self.model,
        )

        # Save if enabled
        if self.save_interpretations:
            result.interpretation_id = self._save_interpretation(
                interpretation_type="regime",
                target_id=window_before_id,
                target_secondary_id=window_after_id,
                result=result,
            )

        return result

    def interpret_indicator(self, indicator_name: str) -> InterpretationResult:
        """
        Generate deep-dive interpretation for an indicator.

        Args:
            indicator_name: Name of the indicator

        Returns:
            InterpretationResult with interpretation and metadata
        """
        # Build context
        context = self.context_builder.build_indicator_context(indicator_name)

        # Format lens breakdown
        lens_breakdown_text = []
        for lens, data in context.lens_breakdown.items():
            if data:
                avg_rank = sum(d['rank'] for d in data) / len(data)
                lens_breakdown_text.append(f"- **{lens}**: Average rank {avg_rank:.1f} across {len(data)} windows")

        # Format high/low importance periods
        high_periods = "\n".join([
            f"- {p['window']}: Rank {p['rank']:.0f} (top {100-p['percentile']:.0f}%)"
            for p in context.high_importance_periods[:5]
        ]) or "None identified"

        low_periods = "\n".join([
            f"- {p['window']}: Rank {p['rank']:.0f} (bottom {p['percentile']:.0f}%)"
            for p in context.low_importance_periods[:5]
        ]) or "None identified"

        # Prepare template variables
        template_vars = {
            "indicator_name": context.indicator_name,
            "system": context.system or "Unknown",
            "frequency": context.frequency or "Unknown",
            "data_start": context.data_start or "Unknown",
            "data_end": context.data_end or "Unknown",
            "n_windows": context.n_windows,
            "ranking_history_table": self._format_ranking_history_table(context.ranking_history),
            "high_importance_periods": high_periods,
            "low_importance_periods": low_periods,
            "lens_breakdown": "\n".join(lens_breakdown_text) or "No lens data available",
        }

        # Render prompt
        prompt = render_template("indicator_deep_dive", **template_vars)

        # Call backend
        interpretation = self._call_backend(prompt)

        # Create result
        result = InterpretationResult(
            interpretation=interpretation,
            prompt_used=prompt,
            template_name="indicator_deep_dive",
            context_snapshot=context.to_dict(),
            backend=self.backend,
            model=self.model,
        )

        # Save if enabled
        if self.save_interpretations:
            # For indicators, we don't have a numeric ID, so use hash
            result.interpretation_id = self._save_interpretation(
                interpretation_type="indicator",
                target_id=hash(indicator_name) % (2**31),  # Ensure positive int
                result=result,
            )

        return result

    def compare_domains(
        self,
        systems: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> InterpretationResult:
        """
        Generate cross-domain comparison interpretation.

        Args:
            systems: List of system names to compare
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            InterpretationResult with interpretation and metadata
        """
        # Build context
        context = self.context_builder.build_cross_domain_context(
            systems, start_date, end_date
        )

        # Prepare template variables
        template_vars = {
            "systems_list": ", ".join(systems),
            "correlation_matrix": json.dumps(context.get("correlation_matrix", {}), indent=2),
            "synchronized_events": self._format_synchronized_events(context.get("synchronized_events", [])),
            "divergent_periods": self._format_divergent_periods(context.get("divergent_periods", [])),
        }

        # Render prompt
        prompt = render_template("cross_domain_pattern", **template_vars)

        # Call backend
        interpretation = self._call_backend(prompt)

        # Create result
        result = InterpretationResult(
            interpretation=interpretation,
            prompt_used=prompt,
            template_name="cross_domain_pattern",
            context_snapshot=context,
            backend=self.backend,
            model=self.model,
        )

        # Save if enabled
        if self.save_interpretations:
            result.interpretation_id = self._save_interpretation(
                interpretation_type="cross_domain",
                target_id=None,
                result=result,
            )

        return result

    def _format_synchronized_events(self, events: List[Dict[str, Any]]) -> str:
        """Format synchronized events."""
        if not events:
            return "No synchronized events identified"
        lines = []
        for e in events:
            lines.append(f"- {e.get('date', 'N/A')}: {e.get('description', 'N/A')}")
        return "\n".join(lines)

    def _format_divergent_periods(self, periods: List[Dict[str, Any]]) -> str:
        """Format divergent periods."""
        if not periods:
            return "No significant divergent periods identified"
        lines = []
        for p in periods:
            lines.append(f"- {p.get('period', 'N/A')}: {p.get('description', 'N/A')}")
        return "\n".join(lines)

    def answer_question(self, question: str) -> InterpretationResult:
        """
        Answer a natural language question about PRISM results.

        Args:
            question: The user's question

        Returns:
            InterpretationResult with answer and metadata
        """
        # Build context
        context = self.context_builder.build_query_context(question)

        # Format context details
        context_details = []
        if context.get("focused_windows"):
            context_details.append("## Focused Windows")
            for w in context["focused_windows"]:
                context_details.append(f"- {w.get('window_label')}: Top indicators include "
                                      f"{', '.join(i['indicator'] for i in w.get('top_indicators', [])[:5])}")

        if context.get("focused_indicators"):
            context_details.append("\n## Focused Indicators")
            for ind in context["focused_indicators"]:
                context_details.append(f"- {ind.get('indicator_name')}: "
                                      f"Analyzed in {ind.get('n_windows', 0)} windows")

        # Prepare template variables
        template_vars = {
            "context_summary": context.get("context_summary", "No context available"),
            "context_details": "\n".join(context_details) or "No detailed context available",
            "user_question": question,
        }

        # Render prompt
        prompt = render_template("natural_language_query", **template_vars)

        # Call backend
        interpretation = self._call_backend(prompt)

        # Create result
        result = InterpretationResult(
            interpretation=interpretation,
            prompt_used=prompt,
            template_name="natural_language_query",
            context_snapshot=context,
            backend=self.backend,
            model=self.model,
        )

        # Save to nl_queries table
        if self.save_interpretations:
            self._save_query(question, context, interpretation)

        return result

    def _call_backend(self, prompt: str) -> str:
        """
        Route to appropriate AI backend.

        Args:
            prompt: The rendered prompt

        Returns:
            AI-generated interpretation
        """
        if self.backend == "manual":
            return self._format_manual_output(prompt)
        elif self.backend == "claude":
            return self._call_claude(prompt)
        elif self.backend == "openai":
            return self._call_openai(prompt)
        elif self.backend == "ollama":
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _format_manual_output(self, prompt: str) -> str:
        """Format output for manual mode."""
        return (
            "[PROMPT FOR MANUAL SUBMISSION]\n"
            "=" * 50 + "\n\n"
            "Copy the prompt below and submit to your preferred AI assistant "
            "(Claude, GPT-4, etc.):\n\n"
            "=" * 50 + "\n\n"
            f"{prompt}\n\n"
            "=" * 50 + "\n\n"
            "After receiving a response, you can save it using the FeedbackManager "
            "or by updating this interpretation in the database."
        )

    def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set via environment or api_key parameter.")

        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        message = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set. Set via environment or api_key parameter.")

        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        client = openai.OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3,
        )

        return response.choices[0].message.content

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama instance."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")

        endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")

        response = requests.post(
            f"{endpoint}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama request failed: {response.text}")

        return response.json().get("response", "")

    def _save_interpretation(
        self,
        interpretation_type: str,
        target_id: Optional[int],
        result: InterpretationResult,
        target_secondary_id: Optional[int] = None,
    ) -> Optional[int]:
        """Save interpretation to database."""
        try:
            with self._db_connection() as conn:
                # Check if interpretations table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='interpretations'
                """)
                if cursor.fetchone() is None:
                    # Table doesn't exist, skip saving
                    return None

                cursor = conn.execute("""
                    INSERT INTO interpretations (
                        interpretation_type, target_id, target_secondary_id,
                        prompt_template, prompt_rendered, context_json,
                        interpretation, backend, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    interpretation_type,
                    target_id,
                    target_secondary_id,
                    result.template_name,
                    result.prompt_used,
                    json.dumps(result.context_snapshot),
                    result.interpretation,
                    result.backend,
                    result.model,
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception:
            # Don't fail interpretation if save fails
            return None

    def _save_query(
        self,
        question: str,
        context: Dict[str, Any],
        response: str,
    ) -> Optional[int]:
        """Save natural language query to database."""
        try:
            with self._db_connection() as conn:
                # Check if nl_queries table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='nl_queries'
                """)
                if cursor.fetchone() is None:
                    return None

                cursor = conn.execute("""
                    INSERT INTO nl_queries (
                        question, context_json, context_summary, response, backend, model
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    question,
                    json.dumps(context),
                    context.get("context_summary", ""),
                    response,
                    self.backend,
                    self.model,
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception:
            return None

    def get_interpretation(self, interpretation_id: int) -> Optional[InterpretationResult]:
        """
        Retrieve a saved interpretation by ID.

        Args:
            interpretation_id: ID of the interpretation

        Returns:
            InterpretationResult or None if not found
        """
        with self._db_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM interpretations WHERE id = ?
            """, (interpretation_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return InterpretationResult(
                interpretation=row['interpretation'],
                prompt_used=row['prompt_rendered'],
                template_name=row['prompt_template'],
                context_snapshot=json.loads(row['context_json']),
                backend=row['backend'],
                model=row['model'],
                timestamp=row['created_at'],
                interpretation_id=row['id'],
                validated=bool(row['validated']),
                human_feedback=row['feedback_notes'],
            )

    def list_interpretations(
        self,
        interpretation_type: Optional[str] = None,
        validated_only: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        List saved interpretations.

        Args:
            interpretation_type: Filter by type
            validated_only: Only return validated interpretations
            limit: Maximum number to return

        Returns:
            List of interpretation summaries
        """
        with self._db_connection() as conn:
            query = "SELECT id, interpretation_type, target_id, backend, validated, created_at FROM interpretations"
            conditions = []
            params = []

            if interpretation_type:
                conditions.append("interpretation_type = ?")
                params.append(interpretation_type)

            if validated_only:
                conditions.append("validated = 1")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += f" ORDER BY created_at DESC LIMIT {limit}"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
