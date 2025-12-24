"""
PRISM Engine Output Validation Agent

Evaluates engine outputs for numerical validity and informational content.
Does NOT modify or rerun engines. Writes annotations only.

IMPORTANT CONSTRAINTS (NON-NEGOTIABLE):
1. Phase-0 data is immutable. No writes to Phase-0 tables.
2. Agents operate ONLY on DB-persisted engine outputs.
3. Agents are NOT real-time.
4. Agents do NOT rerun engines.
5. Agents do NOT introduce meaning or labels.
6. Agents write annotations only.
7. All runnable scripts must live at /scripts level.
8. DB paths must be resolved from repo root (no relative drift).
9. No parallel execution inside agents unless explicitly requested.

Author: PRISM Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import numpy as np

from prism.config import get_validation_config

logger = logging.getLogger(__name__)


class ValidityFlag(Enum):
    VALID = "valid"
    DEGRADED = "degraded"
    INVALID = "invalid"


class ReasonCode(Enum):
    # Fatal
    ZERO_VARIANCE = "ZERO_VARIANCE"
    NAN_VALUES = "NAN_VALUES"
    INF_VALUES = "INF_VALUES"
    NUMERICAL_OVERFLOW = "NUMERICAL_OVERFLOW"
    
    # Error
    FLAT_EIGENVALUES = "FLAT_EIGENVALUES"
    DEGENERATE_STATES = "DEGENERATE_STATES"
    CONSTANT_OUTPUT = "CONSTANT_OUTPUT"
    WINDOW_MINIMAL = "WINDOW_MINIMAL"
    GRANGER_UNSTABLE = "GRANGER_UNSTABLE"
    
    # Warning
    LOW_ENTROPY = "LOW_ENTROPY"
    INSUFFICIENT_TRANSITIONS = "INSUFFICIENT_TRANSITIONS"
    NUMERICAL_UNDERFLOW = "NUMERICAL_UNDERFLOW"
    WINDOW_SPARSE = "WINDOW_SPARSE"
    CORRELATION_DEGENERATE = "CORRELATION_DEGENERATE"
    WAVELET_EDGE_EFFECTS = "WAVELET_EDGE_EFFECTS"
    DMD_UNSTABLE_MODES = "DMD_UNSTABLE_MODES"
    HURST_BOUNDARY = "HURST_BOUNDARY"


# Severity and penalty weights
REASON_SEVERITY = {
    ReasonCode.ZERO_VARIANCE: ("fatal", 1.0),
    ReasonCode.NAN_VALUES: ("fatal", 1.0),
    ReasonCode.INF_VALUES: ("fatal", 1.0),
    ReasonCode.NUMERICAL_OVERFLOW: ("fatal", 1.0),
    ReasonCode.FLAT_EIGENVALUES: ("error", 0.3),
    ReasonCode.DEGENERATE_STATES: ("error", 0.4),
    ReasonCode.CONSTANT_OUTPUT: ("error", 0.5),
    ReasonCode.WINDOW_MINIMAL: ("error", 0.5),
    ReasonCode.GRANGER_UNSTABLE: ("error", 0.4),
    ReasonCode.LOW_ENTROPY: ("warning", 0.2),
    ReasonCode.INSUFFICIENT_TRANSITIONS: ("warning", 0.2),
    ReasonCode.NUMERICAL_UNDERFLOW: ("warning", 0.1),
    ReasonCode.WINDOW_SPARSE: ("warning", 0.2),
    ReasonCode.CORRELATION_DEGENERATE: ("warning", 0.2),
    ReasonCode.WAVELET_EDGE_EFFECTS: ("warning", 0.15),
    ReasonCode.DMD_UNSTABLE_MODES: ("warning", 0.2),
    ReasonCode.HURST_BOUNDARY: ("warning", 0.15),
}


@dataclass
class ValidationResult:
    """Result of validating a single engine output."""
    indicator_id: str
    window_years: float
    engine: str
    
    validity_flag: ValidityFlag = ValidityFlag.VALID
    confidence_penalty: float = 0.0
    reason_codes: List[ReasonCode] = field(default_factory=list)
    
    numerical_stability: float = 1.0
    information_content: float = 1.0
    degeneracy_score: float = 0.0
    window_coverage: float = 1.0
    
    def add_reason(self, code: ReasonCode):
        """Add a reason code and update penalty."""
        if code not in self.reason_codes:
            self.reason_codes.append(code)
            severity, penalty = REASON_SEVERITY[code]
            
            # Update penalty (capped at 1.0)
            self.confidence_penalty = min(1.0, self.confidence_penalty + penalty)
            
            # Update validity flag
            if severity == "fatal":
                self.validity_flag = ValidityFlag.INVALID
            elif severity == "error" and self.validity_flag != ValidityFlag.INVALID:
                self.validity_flag = ValidityFlag.DEGRADED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "window_years": self.window_years,
            "engine": self.engine,
            "validity_flag": self.validity_flag.value,
            "confidence_penalty": self.confidence_penalty,
            "reason_codes": ",".join(rc.value for rc in self.reason_codes) if self.reason_codes else None,
            "numerical_stability": self.numerical_stability,
            "information_content": self.information_content,
            "degeneracy_score": self.degeneracy_score,
            "window_coverage": self.window_coverage,
        }


@dataclass
class ValidationPolicy:
    """
    Configurable thresholds for validation.

    Parameters can be loaded from config/validation.yaml via from_config().
    Default values are preserved for backward compatibility.
    """
    version: str = "1.0.0"

    # Eigenvalue thresholds (PCA)
    eigenvalue_ratio_min: float = 0.1      # Min ratio of 1st to 2nd eigenvalue
    explained_variance_min: float = 0.5    # Min cumulative explained variance

    # HMM thresholds
    state_entropy_min: float = 0.3         # Min entropy of state distribution
    transition_count_min: int = 3          # Min state transitions

    # Numerical thresholds
    variance_min: float = 1e-10            # Min variance to not be constant
    condition_number_max: float = 1e10     # Max condition number

    # Window coverage
    window_coverage_warn: float = 0.5      # Below this = warning
    window_coverage_error: float = 0.2     # Below this = error

    # Hurst exponent
    hurst_boundary_tolerance: float = 0.05  # Distance from 0 or 1

    @classmethod
    def from_config(cls) -> "ValidationPolicy":
        """
        Create policy from config/validation.yaml.

        Falls back to defaults if config not found.
        """
        try:
            cfg = get_validation_config()
        except FileNotFoundError:
            logger.warning("validation.yaml not found, using defaults")
            return cls()

        return cls(
            version=cfg.get("version", "1.0.0"),
            eigenvalue_ratio_min=cfg.get("eigenvalue_ratio_min", 0.1),
            explained_variance_min=cfg.get("explained_variance_min", 0.5),
            state_entropy_min=cfg.get("state_entropy_min", 0.3),
            transition_count_min=cfg.get("transition_count_min", 3),
            variance_min=cfg.get("variance_min", 1e-10),
            condition_number_max=cfg.get("condition_number_max", 1e10),
            window_coverage_warn=cfg.get("window_coverage_warn", 0.5),
            window_coverage_error=cfg.get("window_coverage_error", 0.2),
            hurst_boundary_tolerance=cfg.get("hurst_boundary_tolerance", 0.05),
        )


class EngineOutputValidationAgent:
    """
    Validates engine outputs for numerical validity and informational content.
    
    This agent:
    - Reads engine outputs from DB (read-only)
    - Checks numerical stability, degeneracy, information content
    - Writes validation annotations to Phase-1 tables
    
    This agent does NOT:
    - Rerun engines
    - Modify engine outputs
    - Exclude math or make admission decisions
    - Write to Phase-0 tables
    """
    
    def __init__(self, policy: Optional[ValidationPolicy] = None):
        self.policy = policy or ValidationPolicy()
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            from prism.db.open import open_prism_db
            self._conn = open_prism_db()
        return self._conn
    
    def validate_run(self, run_id: str) -> List[ValidationResult]:
        """
        Validate all engine outputs for a given run.
        
        Args:
            run_id: The run identifier
            
        Returns:
            List of ValidationResult objects
        """
        logger.info(f"Validating engine outputs for run_id={run_id}")
        
        # Get all engine outputs for this run
        outputs = self._load_engine_outputs(run_id)
        
        results = []
        for output in outputs:
            result = self._validate_output(output)
            results.append(result)
        
        logger.info(f"Validated {len(results)} engine outputs")
        return results
    
    def validate_and_persist(self, run_id: str) -> List[ValidationResult]:
        """Validate outputs and persist results to DB."""
        results = self.validate_run(run_id)
        self._persist_results(run_id, results)
        return results
    
    def _load_engine_outputs(self, run_id: str) -> List[Dict]:
        """Load engine outputs from DB (read-only)."""
        # Check which tables exist and load from them
        outputs = []
        
        # Try to load from common engine output tables
        engine_tables = [
            ("engine_pca", "pca"),
            ("engine_hmm", "hmm"),
            ("engine_correlation", "correlation"),
            ("engine_granger", "granger"),
            ("engine_wavelet", "wavelet"),
            ("engine_dmd", "dmd"),
            ("engine_hurst", "hurst"),
            ("engine_clustering", "clustering"),
        ]
        
        for table_name, engine_name in engine_tables:
            try:
                df = self.conn.execute(f"""
                    SELECT * FROM {table_name}
                    WHERE run_id = ?
                """, [run_id]).fetchdf()
                
                if len(df) > 0:
                    for _, row in df.iterrows():
                        outputs.append({
                            "engine": engine_name,
                            "indicator_id": row.get("indicator_id", "unknown"),
                            "window_years": row.get("window_years", 0),
                            "data": row.to_dict(),
                        })
            except Exception as e:
                logger.debug(f"Table {table_name} not found or empty: {e}")
                continue
        
        return outputs
    
    def _validate_output(self, output: Dict) -> ValidationResult:
        """Validate a single engine output."""
        result = ValidationResult(
            indicator_id=output["indicator_id"],
            window_years=output["window_years"],
            engine=output["engine"],
        )
        
        data = output["data"]
        
        # Basic numerical checks (all engines)
        self._check_numerical_validity(data, result)
        
        # Engine-specific checks
        engine = output["engine"]
        if engine == "pca":
            self._check_pca(data, result)
        elif engine == "hmm":
            self._check_hmm(data, result)
        elif engine == "correlation":
            self._check_correlation(data, result)
        elif engine == "granger":
            self._check_granger(data, result)
        elif engine == "wavelet":
            self._check_wavelet(data, result)
        elif engine == "dmd":
            self._check_dmd(data, result)
        elif engine == "hurst":
            self._check_hurst(data, result)
        
        # Compute aggregate scores
        self._compute_aggregate_scores(result)
        
        return result
    
    def _check_numerical_validity(self, data: Dict, result: ValidationResult):
        """Check for NaN, Inf, and numerical issues."""
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    result.add_reason(ReasonCode.NAN_VALUES)
                elif np.isinf(value):
                    result.add_reason(ReasonCode.INF_VALUES)
            elif isinstance(value, (list, np.ndarray)):
                arr = np.array(value)
                if np.any(np.isnan(arr)):
                    result.add_reason(ReasonCode.NAN_VALUES)
                elif np.any(np.isinf(arr)):
                    result.add_reason(ReasonCode.INF_VALUES)
    
    def _check_pca(self, data: Dict, result: ValidationResult):
        """Validate PCA output."""
        # Check eigenvalues
        eigenvalues = data.get("eigenvalues") or data.get("explained_variance_ratio")
        if eigenvalues is not None:
            ev = np.array(eigenvalues) if not isinstance(eigenvalues, np.ndarray) else eigenvalues
            
            if len(ev) > 1:
                # Check for flat eigenvalues
                if ev[0] > 0 and ev[1] / ev[0] > (1 - self.policy.eigenvalue_ratio_min):
                    result.add_reason(ReasonCode.FLAT_EIGENVALUES)
                    result.degeneracy_score = max(result.degeneracy_score, 0.5)
                
                # Check explained variance
                cumsum = np.cumsum(ev) / np.sum(ev) if np.sum(ev) > 0 else ev
                if cumsum[-1] < self.policy.explained_variance_min:
                    result.information_content *= 0.7
            
            # Check for zero variance
            if np.sum(ev) < self.policy.variance_min:
                result.add_reason(ReasonCode.ZERO_VARIANCE)
    
    def _check_hmm(self, data: Dict, result: ValidationResult):
        """Validate HMM output."""
        # Check state distribution
        stationary_dist = data.get("stationary_distribution")
        if stationary_dist is not None:
            dist = np.array(stationary_dist)
            
            # Check entropy
            entropy = -np.sum(dist * np.log(dist + 1e-10))
            max_entropy = np.log(len(dist))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            if normalized_entropy < self.policy.state_entropy_min:
                result.add_reason(ReasonCode.LOW_ENTROPY)
            
            # Check for degenerate (single state dominance)
            if np.max(dist) > 0.95:
                result.add_reason(ReasonCode.DEGENERATE_STATES)
                result.degeneracy_score = max(result.degeneracy_score, 0.7)
        
        # Check transition count
        n_transitions = data.get("n_transitions", 0)
        if n_transitions < self.policy.transition_count_min:
            result.add_reason(ReasonCode.INSUFFICIENT_TRANSITIONS)
    
    def _check_correlation(self, data: Dict, result: ValidationResult):
        """Validate correlation output."""
        corr_matrix = data.get("correlation_matrix")
        if corr_matrix is not None:
            matrix = np.array(corr_matrix) if not isinstance(corr_matrix, np.ndarray) else corr_matrix
            
            # Check condition number
            try:
                cond = np.linalg.cond(matrix)
                if cond > self.policy.condition_number_max:
                    result.add_reason(ReasonCode.CORRELATION_DEGENERATE)
                    result.numerical_stability *= 0.7
            except np.linalg.LinAlgError:
                result.add_reason(ReasonCode.CORRELATION_DEGENERATE)
    
    def _check_granger(self, data: Dict, result: ValidationResult):
        """Validate Granger causality output."""
        # Check for numerical instability indicators
        if data.get("unstable", False) or data.get("condition_warning", False):
            result.add_reason(ReasonCode.GRANGER_UNSTABLE)
            result.numerical_stability *= 0.6
    
    def _check_wavelet(self, data: Dict, result: ValidationResult):
        """Validate wavelet output."""
        # Check edge effects
        edge_ratio = data.get("edge_effect_ratio", 0)
        if edge_ratio > 0.3:
            result.add_reason(ReasonCode.WAVELET_EDGE_EFFECTS)
    
    def _check_dmd(self, data: Dict, result: ValidationResult):
        """Validate DMD output."""
        # Check for unstable modes
        eigenvalues = data.get("eigenvalues") or data.get("dmd_eigenvalues")
        if eigenvalues is not None:
            ev = np.array(eigenvalues)
            # DMD eigenvalues should have magnitude <= 1 for stability
            magnitudes = np.abs(ev)
            if np.any(magnitudes > 1.1):  # Small tolerance
                result.add_reason(ReasonCode.DMD_UNSTABLE_MODES)
    
    def _check_hurst(self, data: Dict, result: ValidationResult):
        """Validate Hurst exponent output."""
        hurst = data.get("hurst_exponent") or data.get("H")
        if hurst is not None:
            # Hurst at boundaries (0 or 1) is suspicious
            if hurst < self.policy.hurst_boundary_tolerance:
                result.add_reason(ReasonCode.HURST_BOUNDARY)
            elif hurst > (1 - self.policy.hurst_boundary_tolerance):
                result.add_reason(ReasonCode.HURST_BOUNDARY)
    
    def _compute_aggregate_scores(self, result: ValidationResult):
        """Compute final aggregate scores."""
        # Numerical stability is degraded by fatal/error codes
        fatal_count = sum(1 for rc in result.reason_codes 
                         if REASON_SEVERITY[rc][0] == "fatal")
        error_count = sum(1 for rc in result.reason_codes 
                         if REASON_SEVERITY[rc][0] == "error")
        
        if fatal_count > 0:
            result.numerical_stability = 0.0
        else:
            result.numerical_stability *= (0.7 ** error_count)
        
        # Information content based on degeneracy
        result.information_content *= (1.0 - result.degeneracy_score)
    
    def _persist_results(self, run_id: str, results: List[ValidationResult]):
        """Persist validation results to Phase-1 tables."""
        logger.info(f"Persisting {len(results)} validation results")

        # Assert required tables exist (no runtime schema creation)
        from prism.db.schema_guard import assert_table_exists
        assert_table_exists(self.conn, "phase1.engine_validation")
        
        # Insert results
        for result in results:
            self.conn.execute("""
                INSERT OR REPLACE INTO phase1.engine_validation
                (run_id, indicator_id, window_years, engine, validity_flag,
                 confidence_penalty, reason_codes, numerical_stability,
                 information_content, degeneracy_score, window_coverage,
                 validator_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                result.indicator_id,
                result.window_years,
                result.engine,
                result.validity_flag.value,
                result.confidence_penalty,
                ",".join(rc.value for rc in result.reason_codes) if result.reason_codes else None,
                result.numerical_stability,
                result.information_content,
                result.degeneracy_score,
                result.window_coverage,
                self.policy.version,
            ])
        
        logger.info(f"Persisted validation results to phase1.engine_validation")
    
    def get_validation_summary(self, run_id: str) -> Dict:
        """Get summary of validation results for a run."""
        df = self.conn.execute("""
            SELECT
                validity_flag,
                COUNT(*) as count,
                AVG(confidence_penalty) as avg_penalty,
                AVG(numerical_stability) as avg_stability,
                AVG(information_content) as avg_info
            FROM phase1.engine_validation
            WHERE run_id = ?
            GROUP BY validity_flag
        """, [run_id]).fetchdf()
        
        return df.to_dict(orient="records")
    
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# CLI INTERFACE (for orchestrator)
# =============================================================================

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validate engine outputs for numerical validity and information content"
    )
    parser.add_argument("--run-id", required=True, help="Run ID to validate")
    parser.add_argument("--dry-run", action="store_true", help="Validate without persisting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Run validation
    agent = EngineOutputValidationAgent()
    
    try:
        if args.dry_run:
            results = agent.validate_run(args.run_id)
            print(f"\nValidation Results (DRY RUN):")
        else:
            results = agent.validate_and_persist(args.run_id)
            print(f"\nValidation Results (PERSISTED):")
        
        # Print summary
        valid = sum(1 for r in results if r.validity_flag == ValidityFlag.VALID)
        degraded = sum(1 for r in results if r.validity_flag == ValidityFlag.DEGRADED)
        invalid = sum(1 for r in results if r.validity_flag == ValidityFlag.INVALID)
        
        print(f"  Total: {len(results)}")
        print(f"  Valid: {valid}")
        print(f"  Degraded: {degraded}")
        print(f"  Invalid: {invalid}")
        
        # Print reason code summary
        all_codes = []
        for r in results:
            all_codes.extend(r.reason_codes)
        
        if all_codes:
            print(f"\nReason Code Frequency:")
            from collections import Counter
            for code, count in Counter(all_codes).most_common():
                print(f"  {code.value}: {count}")
        
    finally:
        agent.close()


if __name__ == "__main__":
    main()
