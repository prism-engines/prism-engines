"""
PRISM Feedback Manager
======================

Manages human feedback on AI interpretations for the feedback loop.

This module enables:
- Recording validation/rejection of AI interpretations
- Extracting validated patterns for reuse
- Analyzing rejection reasons for prompt improvement
- Building a knowledge base of confirmed interpretations

Usage:
    from interpretation.feedback import FeedbackManager

    feedback = FeedbackManager(db_path="path/to/temporal.db")

    # Record feedback
    feedback.record_feedback(
        interpretation_id=42,
        feedback_type="validated",
        notes="Accurate interpretation of the regime break"
    )

    # Get validated patterns
    patterns = feedback.get_validated_patterns(pattern_type="regime_break")
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# Default path
_default_db_path = Path(__file__).parent.parent / "06_output" / "temporal" / "temporal.db"


@dataclass
class FeedbackRecord:
    """Container for feedback on an interpretation."""

    interpretation_id: int
    feedback_type: str  # 'validated', 'rejected', 'refined'
    notes: Optional[str] = None
    refined_interpretation: Optional[str] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interpretation_id": self.interpretation_id,
            "feedback_type": self.feedback_type,
            "notes": self.notes,
            "refined_interpretation": self.refined_interpretation,
            "timestamp": self.timestamp,
        }


@dataclass
class ValidatedPattern:
    """Container for a validated pattern."""

    id: int
    pattern_type: str
    pattern_name: str
    description: str
    conditions: Dict[str, Any]
    interpretation_template: Optional[str] = None
    examples: List[Dict[str, Any]] = None
    source_interpretation_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "pattern_name": self.pattern_name,
            "description": self.description,
            "conditions": self.conditions,
            "interpretation_template": self.interpretation_template,
            "examples": self.examples or [],
            "source_interpretation_id": self.source_interpretation_id,
        }


class FeedbackManager:
    """
    Manages human feedback on AI interpretations.

    Provides methods for:
    - Recording validation, rejection, or refinement of interpretations
    - Retrieving validated patterns
    - Analyzing rejection patterns for prompt improvement
    - Creating reusable pattern templates
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the feedback manager.

        Args:
            db_path: Path to the temporal database
        """
        self.db_path = Path(db_path) if db_path else _default_db_path

    @contextmanager
    def _connection(self):
        """Context manager for database connection."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _check_tables_exist(self, conn) -> bool:
        """Check if required tables exist."""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('interpretations', 'validated_patterns')
        """)
        tables = {row['name'] for row in cursor.fetchall()}
        return 'interpretations' in tables

    def record_feedback(
        self,
        interpretation_id: int,
        feedback_type: str,
        notes: Optional[str] = None,
        refined_interpretation: Optional[str] = None,
    ) -> bool:
        """
        Record human feedback on an interpretation.

        Args:
            interpretation_id: ID of the interpretation
            feedback_type: Type of feedback ('validated', 'rejected', 'refined')
            notes: Optional notes explaining the feedback
            refined_interpretation: If type is 'refined', the corrected interpretation

        Returns:
            True if feedback was recorded successfully

        Raises:
            ValueError: If feedback_type is invalid or interpretation not found
        """
        valid_types = ['validated', 'rejected', 'refined']
        if feedback_type not in valid_types:
            raise ValueError(f"Invalid feedback_type: {feedback_type}. Valid: {valid_types}")

        if feedback_type == 'refined' and not refined_interpretation:
            raise ValueError("refined_interpretation required when feedback_type is 'refined'")

        # Determine validated value
        validated_value = {
            'validated': 1,
            'rejected': -1,
            'refined': 1,  # Refined counts as validated with corrections
        }[feedback_type]

        with self._connection() as conn:
            if not self._check_tables_exist(conn):
                raise RuntimeError("Interpretation tables not found. Run migration 007 first.")

            # Check interpretation exists
            cursor = conn.execute(
                "SELECT id FROM interpretations WHERE id = ?",
                (interpretation_id,)
            )
            if cursor.fetchone() is None:
                raise ValueError(f"Interpretation {interpretation_id} not found")

            # Update interpretation with feedback
            conn.execute("""
                UPDATE interpretations
                SET validated = ?,
                    feedback_type = ?,
                    feedback_notes = ?,
                    refined_interpretation = ?,
                    feedback_at = datetime('now')
                WHERE id = ?
            """, (
                validated_value,
                feedback_type,
                notes,
                refined_interpretation,
                interpretation_id,
            ))
            conn.commit()

        return True

    def get_interpretation_feedback(
        self,
        interpretation_id: int
    ) -> Optional[FeedbackRecord]:
        """
        Get feedback for a specific interpretation.

        Args:
            interpretation_id: ID of the interpretation

        Returns:
            FeedbackRecord or None if no feedback recorded
        """
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT feedback_type, feedback_notes, refined_interpretation, feedback_at
                FROM interpretations
                WHERE id = ? AND feedback_type IS NOT NULL
            """, (interpretation_id,))

            row = cursor.fetchone()
            if row is None:
                return None

            return FeedbackRecord(
                interpretation_id=interpretation_id,
                feedback_type=row['feedback_type'],
                notes=row['feedback_notes'],
                refined_interpretation=row['refined_interpretation'],
                timestamp=row['feedback_at'] or "",
            )

    def get_validated_patterns(
        self,
        pattern_type: Optional[str] = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Retrieve human-validated patterns.

        Args:
            pattern_type: Filter by pattern type (e.g., 'regime_break')
            limit: Maximum number of patterns to return

        Returns:
            DataFrame with validated patterns
        """
        with self._connection() as conn:
            # Check if table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='validated_patterns'
            """)
            if cursor.fetchone() is None:
                return pd.DataFrame()

            query = """
                SELECT id, pattern_type, pattern_name, description,
                       conditions_json, interpretation_template, examples_json,
                       source_interpretation_id, created_at
                FROM validated_patterns
            """

            params = []
            if pattern_type:
                query += " WHERE pattern_type = ?"
                params.append(pattern_type)

            query += f" ORDER BY created_at DESC LIMIT {limit}"

            df = pd.read_sql_query(query, conn, params=params)

            # Parse JSON columns
            if not df.empty:
                df['conditions'] = df['conditions_json'].apply(
                    lambda x: json.loads(x) if x else {}
                )
                df['examples'] = df['examples_json'].apply(
                    lambda x: json.loads(x) if x else []
                )
                df = df.drop(columns=['conditions_json', 'examples_json'])

            return df

    def create_pattern(
        self,
        pattern_type: str,
        pattern_name: str,
        description: str,
        conditions: Dict[str, Any],
        interpretation_template: Optional[str] = None,
        source_interpretation_id: Optional[int] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Create a validated pattern from a successful interpretation.

        Args:
            pattern_type: Type of pattern (e.g., 'regime_break', 'coherence_spike')
            pattern_name: Human-readable name for the pattern
            description: Description of what this pattern means
            conditions: Dictionary describing when this pattern applies
            interpretation_template: Reusable interpretation text
            source_interpretation_id: ID of the original interpretation
            examples: List of example occurrences

        Returns:
            ID of the created pattern
        """
        with self._connection() as conn:
            cursor = conn.execute("""
                INSERT INTO validated_patterns (
                    pattern_type, pattern_name, description, conditions_json,
                    interpretation_template, source_interpretation_id, examples_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_type,
                pattern_name,
                description,
                json.dumps(conditions),
                interpretation_template,
                source_interpretation_id,
                json.dumps(examples or []),
            ))
            conn.commit()
            return cursor.lastrowid

    def add_pattern_example(
        self,
        pattern_id: int,
        example: Dict[str, Any],
    ) -> bool:
        """
        Add an example occurrence to an existing pattern.

        Args:
            pattern_id: ID of the pattern
            example: Dictionary describing the example

        Returns:
            True if example was added successfully
        """
        with self._connection() as conn:
            # Get current examples
            cursor = conn.execute(
                "SELECT examples_json FROM validated_patterns WHERE id = ?",
                (pattern_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Pattern {pattern_id} not found")

            examples = json.loads(row['examples_json'] or '[]')
            examples.append(example)

            # Update pattern
            conn.execute(
                "UPDATE validated_patterns SET examples_json = ? WHERE id = ?",
                (json.dumps(examples), pattern_id)
            )
            conn.commit()

        return True

    def get_rejection_reasons(
        self,
        interpretation_type: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Analyze why interpretations were rejected.

        This is useful for improving prompts based on failure patterns.

        Args:
            interpretation_type: Filter by interpretation type
            limit: Maximum number of rejections to analyze

        Returns:
            DataFrame with rejection details
        """
        with self._connection() as conn:
            if not self._check_tables_exist(conn):
                return pd.DataFrame()

            query = """
                SELECT
                    id,
                    interpretation_type,
                    prompt_template,
                    feedback_notes,
                    feedback_at,
                    target_id
                FROM interpretations
                WHERE validated = -1
            """

            params = []
            if interpretation_type:
                query += " AND interpretation_type = ?"
                params.append(interpretation_type)

            query += f" ORDER BY feedback_at DESC LIMIT {limit}"

            return pd.read_sql_query(query, conn, params=params)

    def get_validated_interpretations(
        self,
        interpretation_type: Optional[str] = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Get validated interpretations for reference.

        Args:
            interpretation_type: Filter by type
            limit: Maximum number to return

        Returns:
            DataFrame with validated interpretations
        """
        with self._connection() as conn:
            if not self._check_tables_exist(conn):
                return pd.DataFrame()

            query = """
                SELECT
                    id,
                    interpretation_type,
                    target_id,
                    interpretation,
                    COALESCE(refined_interpretation, interpretation) as final_interpretation,
                    feedback_notes,
                    feedback_at,
                    backend,
                    model
                FROM interpretations
                WHERE validated = 1
            """

            params = []
            if interpretation_type:
                query += " AND interpretation_type = ?"
                params.append(interpretation_type)

            query += f" ORDER BY feedback_at DESC LIMIT {limit}"

            return pd.read_sql_query(query, conn, params=params)

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics on feedback collected.

        Returns:
            Dictionary with feedback statistics
        """
        with self._connection() as conn:
            if not self._check_tables_exist(conn):
                return {"error": "Tables not found"}

            stats = {}

            # Total interpretations
            cursor = conn.execute("SELECT COUNT(*) as count FROM interpretations")
            stats['total_interpretations'] = cursor.fetchone()['count']

            # By validation status
            cursor = conn.execute("""
                SELECT
                    validated,
                    COUNT(*) as count
                FROM interpretations
                GROUP BY validated
            """)
            status_map = {0: 'pending', 1: 'validated', -1: 'rejected'}
            for row in cursor.fetchall():
                status = status_map.get(row['validated'], 'unknown')
                stats[f'{status}_count'] = row['count']

            # By type
            cursor = conn.execute("""
                SELECT
                    interpretation_type,
                    COUNT(*) as count,
                    SUM(CASE WHEN validated = 1 THEN 1 ELSE 0 END) as validated_count
                FROM interpretations
                GROUP BY interpretation_type
            """)
            stats['by_type'] = {
                row['interpretation_type']: {
                    'total': row['count'],
                    'validated': row['validated_count'],
                }
                for row in cursor.fetchall()
            }

            # Validated patterns count
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='validated_patterns'
            """)
            if cursor.fetchone():
                cursor = conn.execute("SELECT COUNT(*) as count FROM validated_patterns")
                stats['validated_patterns'] = cursor.fetchone()['count']

            return stats

    def rate_query_helpfulness(
        self,
        query_id: int,
        helpful: bool,
    ) -> bool:
        """
        Rate whether a natural language query response was helpful.

        Args:
            query_id: ID of the query in nl_queries table
            helpful: True if helpful, False if not

        Returns:
            True if rating was recorded
        """
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='nl_queries'
            """)
            if cursor.fetchone() is None:
                return False

            cursor = conn.execute(
                "UPDATE nl_queries SET helpful = ? WHERE id = ?",
                (1 if helpful else 0, query_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_query_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze effectiveness of natural language queries.

        Returns:
            Statistics on query helpfulness ratings
        """
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='nl_queries'
            """)
            if cursor.fetchone() is None:
                return {"error": "nl_queries table not found"}

            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as helpful,
                    SUM(CASE WHEN helpful = 0 THEN 1 ELSE 0 END) as not_helpful,
                    SUM(CASE WHEN helpful IS NULL THEN 1 ELSE 0 END) as unrated
                FROM nl_queries
            """)
            row = cursor.fetchone()

            total = row['total']
            helpful = row['helpful'] or 0
            not_helpful = row['not_helpful'] or 0
            rated = helpful + not_helpful

            return {
                'total_queries': total,
                'rated_queries': rated,
                'unrated_queries': row['unrated'] or 0,
                'helpful_count': helpful,
                'not_helpful_count': not_helpful,
                'helpfulness_rate': helpful / rated if rated > 0 else None,
            }

    def export_training_data(
        self,
        output_path: Path,
        include_rejected: bool = False,
    ) -> str:
        """
        Export validated interpretations as training data.

        This can be used to fine-tune models or improve prompts.

        Args:
            output_path: Path to save the export
            include_rejected: Whether to include rejected interpretations

        Returns:
            Path to the exported file
        """
        with self._connection() as conn:
            if not self._check_tables_exist(conn):
                raise RuntimeError("Tables not found")

            query = """
                SELECT
                    interpretation_type,
                    prompt_rendered as prompt,
                    COALESCE(refined_interpretation, interpretation) as response,
                    feedback_type,
                    feedback_notes,
                    context_json
                FROM interpretations
                WHERE validated = 1
            """

            if include_rejected:
                query = query.replace("WHERE validated = 1", "WHERE validated != 0")

            df = pd.read_sql_query(query, conn)

            # Convert to JSONL format for training
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            records = []
            for _, row in df.iterrows():
                records.append({
                    "type": row['interpretation_type'],
                    "prompt": row['prompt'],
                    "response": row['response'],
                    "feedback": row['feedback_type'],
                    "notes": row['feedback_notes'],
                })

            with open(output_path, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')

            return str(output_path)
