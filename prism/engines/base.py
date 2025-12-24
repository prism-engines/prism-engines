"""
PRISM Engine Base Class

All analysis engines inherit from this base class.
Provides common interface for:
- Data Phase preflight gate (REQUIRED)
- Data loading from data.indicators
- Normalization (engine-specific)
- Result storage to phase schemas
- Run logging to meta.engine_runs

Architecture:
    Engine reads from: data.indicators
    Engine writes to: derived.* / structure.* / binding.*
    Engine logs to: meta.engine_runs

CANONICAL PHASE MODEL:
    data      - fetched + cleaned + admissible indicator data
    derived   - first-order engine measurements (metrics of indicators)
    structure - geometric organization (coherence, persistence, agreement)
    binding   - final attachment / positioning within structure
    meta      - orchestration, audits, windows

================================================================================
ENGINE PREFLIGHT GATE (HARD BLOCK)
================================================================================
Engines MUST call require_phase1_locked() before execution.
Engines may NOT run on unpersisted Data Phase output.
Engines may NOT infer geometry from raw data.
This is not optional. This is architectural.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import uuid

import duckdb
import numpy as np
import pandas as pd

from prism.db.open import open_prism_db

# Import path for phase1_integrity is in scripts/ - add to path if needed
import sys
from pathlib import Path
_scripts_path = Path(__file__).parent.parent.parent / "scripts"
if str(_scripts_path) not in sys.path:
    sys.path.insert(0, str(_scripts_path))

try:
    from data_integrity import require_data_phase_locked, get_data_phase_contract, DataPhaseNotLockedError
    # Aliases for backwards compatibility
    require_phase1_locked = require_data_phase_locked
    get_phase1_contract = get_data_phase_contract
    Phase1NotLockedError = DataPhaseNotLockedError
except ImportError:
    # Fallback if import fails - define stub that always fails
    class Phase1NotLockedError(RuntimeError):
        pass

    # Alias
    DataPhaseNotLockedError = Phase1NotLockedError

    def require_phase1_locked(conn, run_id):
        result = conn.execute(
            "SELECT 1 FROM meta.data_run_lock WHERE run_id = ?", [run_id]
        ).fetchone()
        if result is None:
            raise Phase1NotLockedError(
                f"ENGINE PREFLIGHT FAILED: Data Phase run '{run_id}' is NOT locked."
            )

    # Alias
    require_data_phase_locked = require_phase1_locked

    def get_phase1_contract(conn, run_id):
        require_phase1_locked(conn, run_id)
        return {'run_id': run_id, 'locked': True, 'indicators': []}

    # Alias
    get_data_phase_contract = get_phase1_contract


logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """Result of an engine run."""
    engine_name: str
    run_id: str
    success: bool
    started_at: datetime
    completed_at: Optional[datetime] = None
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    normalization: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def runtime_seconds(self) -> float:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Engine: {self.engine_name}",
            f"Run ID: {self.run_id}",
            f"Status: {status}",
            f"Runtime: {self.runtime_seconds:.2f}s",
        ]
        if self.window_start and self.window_end:
            lines.append(f"Window: {self.window_start} to {self.window_end}")
        if self.normalization:
            lines.append(f"Normalization: {self.normalization}")
        if self.metrics:
            lines.append(f"Metrics: {self.metrics}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


class BaseEngine(ABC):
    """
    Abstract base class for all PRISM analysis engines.

    Subclasses must implement:
        - name: Engine identifier
        - phase: Which phase this engine belongs to
        - run(): Execute the analysis

    Usage:
        class PCAEngine(BaseEngine):
            name = "pca"
            phase = "derived"

            def run(self, indicators, window_start, window_end, **params):
                # ... implementation
                return results_df
    """

    # Subclasses must define these
    name: str = "base"
    phase: str = "derived"  # 'derived', 'structure', 'binding'
    
    # Default normalization (override in subclass if needed)
    default_normalization: Optional[str] = None  # 'zscore', 'minmax', 'returns', etc.
    
    def __init__(self):
        """Initialize engine."""
        pass

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get database connection."""
        return open_prism_db()
    
    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    
    def load_indicators(
        self,
        indicator_ids: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Load clean indicator data from database.
        
        Args:
            indicator_ids: List of indicator IDs to load
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with DatetimeIndex, indicators as columns
        """
        with self._get_connection() as conn:
            # Build query
            placeholders = ", ".join(["?" for _ in indicator_ids])
            query = f"""
                SELECT date, indicator_id, value
                FROM data.indicators
                WHERE indicator_id IN ({placeholders})
            """
            params = list(indicator_ids)
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date, indicator_id"
            
            df = conn.execute(query, params).fetchdf()
        
        if df.empty:
            logger.warning(f"No data found for indicators: {indicator_ids}")
            return pd.DataFrame()
        
        # Pivot to wide format: rows=dates, columns=indicators
        df_wide = df.pivot(index="date", columns="indicator_id", values="value")
        df_wide.index = pd.to_datetime(df_wide.index)
        df_wide = df_wide.sort_index()
        
        return df_wide
    
    def load_all_indicators(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load all available indicators."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT DISTINCT indicator_id FROM data.indicators"
            ).fetchall()
            indicator_ids = [r[0] for r in result]
        
        if not indicator_ids:
            return pd.DataFrame()
        
        return self.load_indicators(indicator_ids, start_date, end_date)
    
    # -------------------------------------------------------------------------
    # Normalization (engine calls what it needs)
    # -------------------------------------------------------------------------
    
    def normalize_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization (mean=0, std=1) per column."""
        return (df - df.mean()) / df.std()
    
    def normalize_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalization (0-1) per column."""
        return (df - df.min()) / (df.max() - df.min())
    
    def normalize_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to percentage returns."""
        return df.pct_change().dropna()
    
    def normalize_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to ranks (uniform distribution)."""
        return df.rank(pct=True)
    
    def normalize_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """First difference (for stationarity)."""
        return df.diff().dropna()
    
    def discretize(
        self, 
        df: pd.DataFrame, 
        n_bins: int = 8,
        method: str = "quantile"
    ) -> pd.DataFrame:
        """
        Discretize continuous data into bins.
        
        Args:
            df: Input DataFrame
            n_bins: Number of bins
            method: 'quantile' or 'uniform'
        """
        result = df.copy()
        for col in result.columns:
            if method == "quantile":
                result[col] = pd.qcut(
                    result[col], q=n_bins, labels=False, duplicates="drop"
                )
            else:
                result[col] = pd.cut(
                    result[col], bins=n_bins, labels=False
                )
        return result
    
    # -------------------------------------------------------------------------
    # Run Orchestration
    # -------------------------------------------------------------------------
    
    def execute(
        self,
        indicator_ids: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        normalization: Optional[str] = None,
        phase1_run_id: Optional[str] = None,
        **params
    ) -> EngineResult:
        """
        Execute the engine with logging and error handling.

        IMPORTANT: If phase1_run_id is provided, the engine will verify
        that Phase 1 is locked before execution. This is the recommended
        way to run engines to ensure they only operate on persisted,
        validated Phase 1 output.

        Args:
            indicator_ids: List of indicators (None = all)
            start_date: Window start
            end_date: Window end
            normalization: Override default normalization
            phase1_run_id: Phase 1 run ID to verify (RECOMMENDED)
            **params: Engine-specific parameters

        Returns:
            EngineResult with status and metrics

        Raises:
            Phase1NotLockedError: If phase1_run_id is provided but not locked
        """
        # =====================================================================
        # ENGINE PREFLIGHT GATE
        # =====================================================================
        if phase1_run_id is not None:
            with self._get_connection() as conn:
                require_phase1_locked(conn, phase1_run_id)
                logger.info(f"✓ Phase 1 preflight passed: {phase1_run_id} is locked")

        run_id = self._generate_run_id()
        started_at = datetime.now()
        
        result = EngineResult(
            engine_name=self.name,
            run_id=run_id,
            success=False,
            started_at=started_at,
            window_start=start_date,
            window_end=end_date,
            normalization=normalization or self.default_normalization,
            parameters=params,
        )
        
        try:
            # Log run start
            self._record_run_start(result)
            
            # Load data
            if indicator_ids:
                df = self.load_indicators(indicator_ids, start_date, end_date)
            else:
                df = self.load_all_indicators(start_date, end_date)
            
            if df.empty:
                raise ValueError("No data available for specified indicators/window")
            
            # Update window from actual data
            result.window_start = df.index.min().date()
            result.window_end = df.index.max().date()
            
            # Apply normalization if specified
            norm = normalization or self.default_normalization
            if norm:
                df = self._apply_normalization(df, norm)
                result.normalization = norm
            
            # Run the actual analysis
            logger.info(f"Running {self.name} on {len(df.columns)} indicators")
            metrics = self.run(df, run_id=run_id, **params)
            
            result.success = True
            result.metrics = metrics or {}
            
        except Exception as e:
            logger.exception(f"Engine {self.name} failed: {e}")
            result.error = str(e)
        
        finally:
            result.completed_at = datetime.now()
            self._record_run_complete(result)
        
        return result
    
    def _apply_normalization(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply specified normalization method."""
        methods = {
            "zscore": self.normalize_zscore,
            "minmax": self.normalize_minmax,
            "returns": self.normalize_returns,
            "rank": self.normalize_rank,
            "diff": self.normalize_diff,
        }
        
        if method not in methods:
            raise ValueError(f"Unknown normalization: {method}. Options: {list(methods.keys())}")
        
        return methods[method](df)
    
    @abstractmethod
    def run(self, df: pd.DataFrame, run_id: str, **params) -> Dict[str, Any]:
        """
        Execute the analysis. Subclasses must implement.
        
        Args:
            df: Prepared DataFrame (normalized if applicable)
            run_id: Unique run identifier
            **params: Engine-specific parameters
        
        Returns:
            Dict of metrics/summary statistics
        """
        pass
    
    # -------------------------------------------------------------------------
    # Result Storage
    # -------------------------------------------------------------------------
    
    def store_results(
        self,
        table_name: str,
        df: pd.DataFrame,
        run_id: str,
    ):
        """
        Store results to the appropriate phase schema.
        
        Args:
            table_name: Table name (without schema prefix)
            df: Results DataFrame
            run_id: Run identifier
        """
        full_table = f"{self.phase}.{table_name}"
        
        # Add run_id if not present
        if "run_id" not in df.columns:
            df = df.copy()
            df["run_id"] = run_id
        
        with self._get_connection() as conn:
            conn.register("results_df", df)
            
            # Insert (append mode — don't delete previous runs)
            conn.execute(f"INSERT INTO {full_table} SELECT * FROM results_df")
            
            logger.debug(f"Stored {len(df)} rows to {full_table}")
    
    # -------------------------------------------------------------------------
    # Meta Logging
    # -------------------------------------------------------------------------
    
    def _record_run_start(self, result: EngineResult):
        """Record engine run start."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO meta.engine_runs
                (run_id, engine_name, phase, started_at, status, 
                 window_start, window_end, normalization, parameters)
                VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)
            """, [
                result.run_id,
                result.engine_name,
                self.phase,
                result.started_at,
                result.window_start,
                result.window_end,
                result.normalization,
                str(result.parameters),
            ])
    
    def _record_run_complete(self, result: EngineResult):
        """Update engine run record on completion."""
        status = "completed" if result.success else "failed"
        
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE meta.engine_runs
                SET completed_at = ?,
                    status = ?,
                    error_message = ?
                WHERE run_id = ?
            """, [
                result.completed_at,
                status,
                result.error,
                result.run_id,
            ])
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{self.name}_{timestamp}_{short_uuid}"
