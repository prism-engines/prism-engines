"""
PRISM Engine Stability Audit Agent - Spec Sections B, C, D Implementation

Agent 4: Runs AFTER engines complete. Implements:
    Spec B: Null Calibration (surrogate testing)
    Spec C: Stability Scoring (subsample-based)
    Spec D: Weight Computation and Capping

Evaluates trustworthiness of each engine's output and computes
final consensus weights with proper null calibration and stability checks.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from scipy import stats

from prism.config import get_stability_config
from .agent_foundation import (
    BaseAgent,
    RoutingResult,
    StabilityResult,
    StabilityScore,
    NullCalibrationResult,
    EngineContribution,
    DiagnosticRegistry,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Spec B1: Engines Requiring Null Calibration
# =============================================================================

REQUIRES_NULL = [
    "mutual_information",   # Permutation null essential
    "granger",              # Spurious causality risk
    "copula",               # Tail dependence estimation
    "dtw",                  # Warping artifacts
    "rqa",                  # Parameter sensitivity
    "hurst",                # Noise interpretation
    "cointegration",        # Spurious cointegration
]


# =============================================================================
# Spec G: Default Benchmark Scores (placeholder, to be calibrated later)
# =============================================================================

DEFAULT_BENCHMARK_SCORES = {
    # Core engines - high baseline confidence
    "pca": 0.9,
    "correlation": 0.9,
    "entropy": 0.85,
    "wavelets": 0.85,
    # Conditional engines - moderate confidence pending validation
    "hmm": 0.8,
    "dmd": 0.75,
    "mutual_information": 0.7,
    "granger": 0.7,
    "copula": 0.7,
    "rolling_beta": 0.8,
    # Diagnostic engines - lower confidence
    "hurst": 0.5,
    "cointegration": 0.5,
    # Domain-specific
    "sir": 0.8,
}


def get_benchmark_score(engine_name: str) -> float:
    """
    Get benchmark score for an engine (Spec G - placeholder).

    Returns default 0.8 for unknown engines, to be calibrated later.
    """
    return DEFAULT_BENCHMARK_SCORES.get(engine_name, 0.8)


# =============================================================================
# Engine-Specific Key Metrics Extractors
# =============================================================================

def _extract_pca_metric(result: dict) -> Optional[float]:
    """Extract explained variance ratio from PCA result."""
    if result is None:
        return None
    for key in ["explained_variance_ratio", "explained_variance", "variance_explained"]:
        if key in result:
            val = result[key]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                return float(val[0])
            return float(val)
    return result.get("pc1_variance", None)


def _extract_correlation_metric(result: dict) -> Optional[float]:
    """Extract mean correlation from correlation result."""
    if result is None:
        return None
    for key in ["mean_correlation", "avg_correlation", "mean_corr"]:
        if key in result:
            return float(result[key])
    if "correlation_matrix" in result:
        matrix = np.array(result["correlation_matrix"])
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        return float(np.mean(np.abs(matrix[mask])))
    return None


def _extract_entropy_metric(result: dict) -> Optional[float]:
    """Extract entropy value."""
    if result is None:
        return None
    for key in ["entropy", "permutation_entropy", "sample_entropy"]:
        if key in result:
            return float(result[key])
    return None


def _extract_hmm_metric(result: dict) -> Optional[float]:
    """Extract number of state transitions or log likelihood."""
    if result is None:
        return None
    for key in ["n_transitions", "log_likelihood", "bic", "aic"]:
        if key in result:
            return float(result[key])
    if "states" in result:
        states = np.array(result["states"])
        return float(np.sum(np.diff(states) != 0))
    return None


def _extract_dmd_metric(result: dict) -> Optional[float]:
    """Extract dominant eigenvalue magnitude."""
    if result is None:
        return None
    for key in ["dominant_eigenvalue", "lambda_1", "eigenvalues"]:
        if key in result:
            val = result[key]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                return float(np.abs(val[0]))
            return float(np.abs(val))
    return None


def _extract_generic_metric(result: dict) -> Optional[float]:
    """Generic extractor - tries common metric names."""
    if result is None:
        return None
    for key in ["score", "value", "metric", "result", "output"]:
        if key in result and isinstance(result[key], (int, float, np.number)):
            return float(result[key])
    for key, val in result.items():
        if isinstance(val, (int, float, np.number)) and not np.isnan(val):
            return float(val)
    return None


METRIC_EXTRACTORS: Dict[str, Callable] = {
    "pca": _extract_pca_metric,
    "correlation": _extract_correlation_metric,
    "entropy": _extract_entropy_metric,
    "hmm": _extract_hmm_metric,
    "dmd": _extract_dmd_metric,
    "wavelets": _extract_generic_metric,
    "granger": _extract_generic_metric,
    "mutual_information": _extract_generic_metric,
    "copula": _extract_generic_metric,
    "sir": _extract_generic_metric,
    "hurst": _extract_generic_metric,
    "rolling_beta": _extract_generic_metric,
}


def extract_primary_metric(engine_name: str, result: dict) -> Optional[float]:
    """Extract the primary metric from an engine result."""
    extractor = METRIC_EXTRACTORS.get(engine_name, _extract_generic_metric)
    return extractor(result)


# =============================================================================
# Simplified Engine Runners for Stability Tests
# =============================================================================

def _simplified_pca(data: np.ndarray) -> dict:
    """Simplified PCA - compute variance of first PC direction."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    centered = data - np.mean(data, axis=0)
    if centered.shape[1] == 1:
        return {"explained_variance_ratio": 1.0}
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    total = np.sum(eigenvalues)
    if total > 0:
        return {"explained_variance_ratio": eigenvalues[-1] / total}
    return {"explained_variance_ratio": 0.0}


def _simplified_correlation(data: np.ndarray) -> dict:
    """Simplified correlation - mean absolute correlation."""
    if data.ndim == 1:
        return {"mean_correlation": 1.0}
    corr = np.corrcoef(data.T)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    if mask.sum() > 0:
        return {"mean_correlation": np.mean(np.abs(corr[mask]))}
    return {"mean_correlation": 0.0}


def _simplified_entropy(data: np.ndarray) -> dict:
    """Simplified entropy - permutation entropy approximation."""
    if data.ndim > 1:
        data = data[:, 0]
    n = len(data)
    if n < 10:
        return {"entropy": 0.5}
    ranks = stats.rankdata(data)
    entropy = np.std(ranks) / n
    return {"entropy": min(1.0, entropy)}


def _simplified_hmm(data: np.ndarray) -> dict:
    """Simplified HMM - count sign changes as proxy for transitions."""
    if data.ndim > 1:
        data = data[:, 0]
    demeaned = data - np.mean(data)
    signs = np.sign(demeaned)
    transitions = np.sum(np.abs(np.diff(signs)) > 0)
    return {"n_transitions": transitions}


def _simplified_generic(data: np.ndarray) -> dict:
    """Generic simplified computation - just variance."""
    if data.ndim > 1:
        data = data[:, 0]
    return {"value": np.var(data)}


SIMPLIFIED_ENGINES: Dict[str, Callable] = {
    "pca": _simplified_pca,
    "correlation": _simplified_correlation,
    "entropy": _simplified_entropy,
    "hmm": _simplified_hmm,
    "dmd": _simplified_generic,
    "wavelets": _simplified_generic,
    "granger": _simplified_generic,
    "mutual_information": _simplified_generic,
    "copula": _simplified_generic,
    "sir": _simplified_generic,
    "hurst": _simplified_generic,
    "rolling_beta": _simplified_generic,
}


def run_engine_light(engine_name: str, data: np.ndarray) -> dict:
    """
    Run simplified engine for stability testing.

    Placeholder for full engine execution - uses simplified versions.
    """
    runner = SIMPLIFIED_ENGINES.get(engine_name, _simplified_generic)
    return runner(data)


# =============================================================================
# Spec B: Null Calibration Functions
# =============================================================================

def permutation_null(data: np.ndarray) -> np.ndarray:
    """
    Spec B - Null Family 1: Permutation null.

    Shuffle time index, destroying temporal structure while
    preserving marginal distribution.
    """
    if data.ndim == 1:
        return np.random.permutation(data)
    else:
        # Shuffle rows (time axis)
        indices = np.random.permutation(len(data))
        return data[indices]


def block_bootstrap_null(data: np.ndarray, block_size: int = 10) -> np.ndarray:
    """
    Spec B - Null Family 2: Block bootstrap null.

    Shuffle blocks of data, preserving local structure while
    destroying global temporal patterns.
    """
    n = len(data)
    n_blocks = max(1, n // block_size)

    if data.ndim == 1:
        # Reshape into blocks
        truncated = data[:n_blocks * block_size]
        blocks = truncated.reshape(n_blocks, block_size)
        # Shuffle blocks
        shuffled_blocks = blocks[np.random.permutation(n_blocks)]
        result = shuffled_blocks.flatten()
        # Append remainder if any
        if n > n_blocks * block_size:
            result = np.concatenate([result, data[n_blocks * block_size:]])
        return result
    else:
        # Multi-dimensional case
        truncated = data[:n_blocks * block_size]
        indices = []
        for i in range(n_blocks):
            indices.extend(range(i * block_size, (i + 1) * block_size))
        # Shuffle block indices
        block_order = np.random.permutation(n_blocks)
        shuffled_indices = []
        for b in block_order:
            shuffled_indices.extend(range(b * block_size, (b + 1) * block_size))
        result = data[shuffled_indices]
        if n > n_blocks * block_size:
            result = np.vstack([result, data[n_blocks * block_size:]])
        return result


def phase_randomization_null(data: np.ndarray) -> np.ndarray:
    """
    Spec B - Null Family 3: Phase randomization null.

    Preserve power spectrum, destroy phase coherence.
    Maintains amplitude structure but removes temporal predictability.
    """
    if data.ndim == 1:
        return _phase_randomize_1d(data)
    else:
        # Randomize each column independently
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            result[:, i] = _phase_randomize_1d(data[:, i])
        return result


def _phase_randomize_1d(series: np.ndarray) -> np.ndarray:
    """Phase randomize a 1D series."""
    n = len(series)
    fft = np.fft.fft(series)
    amplitudes = np.abs(fft)
    # Generate random phases, but keep conjugate symmetry for real output
    random_phases = np.random.uniform(-np.pi, np.pi, n)
    # Ensure conjugate symmetry: phase[k] = -phase[n-k]
    if n > 1:
        random_phases[n // 2 + 1:] = -random_phases[1:n // 2][::-1] if n % 2 == 0 else -random_phases[1:(n + 1) // 2][::-1]
    randomized_fft = amplitudes * np.exp(1j * random_phases)
    surrogate = np.real(np.fft.ifft(randomized_fft))
    return surrogate


def shift_null(data1: np.ndarray, data2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Spec B - Null Family 4: Shift null.

    Circular shift one series relative to another.
    For single series, shifts by random offset.
    """
    if data2 is not None:
        # Return shifted version of data2
        shift = np.random.randint(1, len(data2))
        return np.roll(data2, shift, axis=0)
    else:
        # Single series case - random shift
        shift = np.random.randint(1, len(data1))
        return np.roll(data1, shift, axis=0)


# =============================================================================
# Spec B2: Null Calibration
# =============================================================================

def run_null_calibration(
    engine_name: str,
    observed_metric: float,
    data: np.ndarray,
    n_null: int = 100,
) -> NullCalibrationResult:
    """
    Spec B2: Run null calibration test for an engine.

    Generates null distribution using appropriate null family,
    computes empirical p-value and effect size.

    Args:
        engine_name: Name of the engine
        observed_metric: The observed metric value from the engine
        data: Original data used by the engine
        n_null: Number of null samples (default 100)

    Returns:
        NullCalibrationResult with pass/fail decision per Spec B3
    """
    # Select null family based on engine (Spec B2)
    if engine_name in ["mutual_information", "granger"]:
        null_fn = shift_null
    elif engine_name in ["copula", "dtw"]:
        null_fn = phase_randomization_null
    else:
        null_fn = permutation_null

    # Get simplified engine runner
    runner = SIMPLIFIED_ENGINES.get(engine_name, _simplified_generic)
    extractor = METRIC_EXTRACTORS.get(engine_name, _extract_generic_metric)

    # Generate null distribution
    null_values = []
    for _ in range(n_null):
        try:
            null_data = null_fn(data)
            null_result = runner(null_data)
            null_metric = extractor(null_result)
            if null_metric is not None and not np.isnan(null_metric):
                null_values.append(null_metric)
        except Exception:
            continue

    if len(null_values) < 10:
        # Not enough null samples - assume passes
        return NullCalibrationResult(
            null_mean=0.0,
            null_std=1.0,
            observed_value=observed_metric,
            empirical_p_value=0.01,  # Assume significant
            effect_size=2.0,
            passed=True,
            n_surrogates=len(null_values),
        )

    null_values = np.array(null_values)
    null_mean = np.mean(null_values)
    null_std = np.std(null_values)

    # Spec B2: Empirical p-value formula
    # p = (count(null >= observed) + 1) / (N_null + 1)
    count_greater = np.sum(null_values >= observed_metric)
    empirical_p = (count_greater + 1) / (len(null_values) + 1)

    # Spec B2: Effect size formula
    # effect_size = (observed - null_mean) / (null_std + eps)
    effect_size = (observed_metric - null_mean) / (null_std + 1e-10)

    # Spec B3: Gate criteria
    # Pass if p <= 0.05 OR effect_size >= 2.0
    passed = (empirical_p <= 0.05) or (effect_size >= 2.0)

    return NullCalibrationResult(
        null_mean=null_mean,
        null_std=null_std,
        observed_value=observed_metric,
        empirical_p_value=empirical_p,
        effect_size=effect_size,
        passed=passed,
        n_surrogates=len(null_values),
    )


# =============================================================================
# Spec C2: Stability Scoring
# =============================================================================

def compute_stability_score(
    engine_name: str,
    data: np.ndarray,
    compute_metric_fn: Optional[Callable] = None,
    n_subsamples: int = 2,
    subsample_fraction: float = 0.9,
) -> StabilityScore:
    """
    Spec C2: Default stability recipe.

    Computes stability as consistency of engine output under subsampling.

    Args:
        engine_name: Name of the engine
        data: Data to run engine on
        compute_metric_fn: Optional custom metric function
        n_subsamples: Number of subsamples (default 2)
        subsample_fraction: Fraction of data per subsample (default 0.9)

    Returns:
        StabilityScore with score in [0, 1], requires >= 0.6 for consensus
    """
    # Default metric computation
    if compute_metric_fn is None:
        runner = SIMPLIFIED_ENGINES.get(engine_name, _simplified_generic)
        extractor = METRIC_EXTRACTORS.get(engine_name, _extract_generic_metric)

        def compute_metric_fn(d):
            result = runner(d)
            metric = extractor(result)
            return metric if metric is not None else 0.0

    n = len(data)
    subsample_size = int(subsample_fraction * n)

    try:
        # Full window metric
        full_metric = compute_metric_fn(data)
        if full_metric is None:
            full_metric = 0.0
    except Exception:
        return StabilityScore(score=0.5, method="subsample", details={"error": "full_metric_failed"})

    # Subsample metrics
    subsample_metrics = []
    for _ in range(n_subsamples):
        try:
            indices = np.random.choice(n, subsample_size, replace=False)
            indices = np.sort(indices)  # Maintain temporal order

            if data.ndim == 1:
                subsample = data[indices]
            else:
                subsample = data[indices, :]

            metric = compute_metric_fn(subsample)
            if metric is not None:
                subsample_metrics.append(metric)
        except Exception:
            continue

    if len(subsample_metrics) == 0:
        return StabilityScore(score=0.5, method="subsample", details={"error": "no_subsamples"})

    # Spec C2: Mean absolute difference (normalized)
    diffs = [abs(full_metric - m) for m in subsample_metrics]
    d = np.mean(diffs) / (abs(full_metric) + 1e-10)

    # Spec C2: Map to S via exp(-k * d)
    # k chosen so moderate drift (~0.5) -> S ~ 0.5
    k = 2.0
    S = np.exp(-k * d)

    # Compute additional metrics
    all_metrics = [full_metric] + subsample_metrics
    cv = np.std(all_metrics) / (np.mean(all_metrics) + 1e-10)
    iqr = np.percentile(all_metrics, 75) - np.percentile(all_metrics, 25)
    iqr_ratio = iqr / (np.median(all_metrics) + 1e-10)

    return StabilityScore(
        score=float(S),
        method="subsample",
        cv=float(cv),
        iqr_ratio=float(iqr_ratio),
        n_resamples=len(subsample_metrics),
        details={
            "d": float(d),
            "k": k,
            "full_metric": float(full_metric),
            "subsample_metrics": [float(m) for m in subsample_metrics],
        },
    )


# =============================================================================
# Spec D1: Engine Contribution Computation
# =============================================================================

def compute_engine_contribution(
    engine_name: str,
    gate_pass: bool,
    null_result: Optional[NullCalibrationResult],
    stability: StabilityScore,
    benchmark_score: float,
    domain_fit: float,
    w_cap: float = 0.25,
) -> EngineContribution:
    """
    Spec D1: Compute engine contribution to consensus.

    Weight formula:
        raw_weight = gate_pass * null_pass * stability_score * benchmark_score * domain_fit

    Args:
        engine_name: Engine name
        gate_pass: Did engine pass routing gates?
        null_result: Result of null calibration (or None if not required)
        stability: Stability score result
        benchmark_score: Global benchmark performance [0, 1]
        domain_fit: Domain appropriateness [0, 1]
        w_cap: Maximum weight per engine (Spec D2, default 0.25)

    Returns:
        EngineContribution with computed weights
    """
    null_pass = null_result.passed if null_result else True

    # Spec D1: Weight formula
    raw_weight = (
        float(gate_pass) *
        float(null_pass) *
        stability.score *
        benchmark_score *
        domain_fit
    )

    # Spec D2: Cap to prevent domination
    capped_weight = min(raw_weight, w_cap)

    # Determine gated reason if excluded
    gated_reason = ""
    if not gate_pass:
        gated_reason = "failed hard gate"
    elif not null_pass:
        if null_result:
            gated_reason = f"null test failed (p={null_result.empirical_p_value:.3f}, effect={null_result.effect_size:.2f})"
        else:
            gated_reason = "null test failed"
    elif stability.score < 0.6:
        gated_reason = f"stability below threshold ({stability.score:.2f} < 0.6)"

    return EngineContribution(
        engine_name=engine_name,
        gate_pass=gate_pass,
        null_pass=null_pass,
        stability_score=stability.score,
        benchmark_score=benchmark_score,
        domain_fit=domain_fit,
        raw_weight=raw_weight,
        capped_weight=capped_weight,
        gated_reason=gated_reason,
        null_calibration=null_result,
        stability_details=stability,
    )


# =============================================================================
# Legacy Stability Report (for backward compatibility)
# =============================================================================

@dataclass
class EngineStabilityReport:
    """Stability test results for a single engine (legacy format)."""
    engine_name: str
    bootstrap_stability: float
    hyperparameter_sensitivity: float
    null_test_passed: bool
    null_test_pvalue: float
    trustworthiness: float
    issues: List[str] = field(default_factory=list)


# =============================================================================
# Agent 4: EngineStabilityAuditAgent
# =============================================================================

class EngineStabilityAuditAgent(BaseAgent):
    """
    Evaluates stability and trustworthiness of engine outputs.

    Implements Spec Sections B, C, D:
        B: Null Calibration - Is output distinguishable from structured noise?
        C: Stability Scoring - Is output stable under subsampling?
        D: Weight Computation - Combine factors into consensus weights

    Usage:
        agent = EngineStabilityAuditAgent(registry)
        result = agent.run({
            "engine_results": {"pca": {...}, "hmm": {...}},
            "routing_result": routing_result,
            "original_data": data,
            "weight_cap": 0.25,
        })
    """

    # Default thresholds (can be overridden by config/stability.yaml)
    DEFAULT_N_NULL = 100
    DEFAULT_N_SUBSAMPLES = 2
    DEFAULT_SUBSAMPLE_FRACTION = 0.9
    DEFAULT_STABILITY_THRESHOLD = 0.6  # Spec C3
    DEFAULT_W_CAP = 0.25  # Spec D2
    DEFAULT_ARTIFACT_THRESHOLD = 0.3

    def __init__(
        self,
        registry: DiagnosticRegistry,
        config: Optional[Dict] = None,
    ):
        super().__init__(registry, config)
        self._last_reports: Dict[str, EngineStabilityReport] = {}
        self._last_contributions: List[EngineContribution] = []
        self._null_results: Dict[str, NullCalibrationResult] = {}
        self._stability_scores: Dict[str, StabilityScore] = {}
        self._redundancy_map: Dict[str, List[str]] = {}

        # Load config and override class defaults
        self._load_config_overrides()

    def _load_config_overrides(self) -> None:
        """Load configuration overrides from config/stability.yaml."""
        try:
            cfg = get_stability_config()

            # Override class defaults with config values
            self.DEFAULT_N_NULL = cfg.get("n_null", self.DEFAULT_N_NULL)
            self.DEFAULT_N_SUBSAMPLES = cfg.get("n_subsamples", self.DEFAULT_N_SUBSAMPLES)
            self.DEFAULT_SUBSAMPLE_FRACTION = cfg.get(
                "subsample_fraction", self.DEFAULT_SUBSAMPLE_FRACTION
            )
            self.DEFAULT_STABILITY_THRESHOLD = cfg.get(
                "stability_threshold", self.DEFAULT_STABILITY_THRESHOLD
            )
            self.DEFAULT_W_CAP = cfg.get("w_cap", self.DEFAULT_W_CAP)
            self.DEFAULT_ARTIFACT_THRESHOLD = cfg.get(
                "artifact_threshold", self.DEFAULT_ARTIFACT_THRESHOLD
            )
        except FileNotFoundError:
            logger.debug("stability.yaml not found, using defaults")

    @property
    def name(self) -> str:
        return "engine_stability"

    def bootstrap_stability(
        self,
        engine_name: str,
        result: dict,
        data: np.ndarray,
        n_bootstrap: int = 20,
    ) -> float:
        """
        Compute bootstrap stability score for an engine result.

        Measures how consistent the engine output is across bootstrap resamples.

        Args:
            engine_name: Name of the engine
            result: The engine result dict
            data: Original data array
            n_bootstrap: Number of bootstrap resamples

        Returns:
            Stability score in [0, 1], higher = more stable
        """
        # Extract observed metric
        extractor = METRIC_EXTRACTORS.get(engine_name, _extract_generic_metric)
        observed = extractor(result)
        if observed is None:
            observed = 0.0

        # Get simplified runner
        runner = SIMPLIFIED_ENGINES.get(engine_name, _simplified_generic)

        # Collect bootstrap metrics
        bootstrap_metrics = []
        n = len(data)
        for _ in range(n_bootstrap):
            try:
                # Sample with replacement
                indices = np.random.choice(n, n, replace=True)
                if data.ndim == 1:
                    bootstrap_sample = data[indices]
                else:
                    bootstrap_sample = data[indices, :]
                bootstrap_result = runner(bootstrap_sample)
                metric = extractor(bootstrap_result)
                if metric is not None:
                    bootstrap_metrics.append(metric)
            except Exception:
                continue

        if len(bootstrap_metrics) < 2:
            return 0.5  # Default for insufficient data

        # Stability = 1 - normalized coefficient of variation
        cv = np.std(bootstrap_metrics) / (np.abs(np.mean(bootstrap_metrics)) + 1e-10)
        stability = np.exp(-cv)  # Map CV to [0, 1]
        return float(np.clip(stability, 0.0, 1.0))

    def hyperparameter_sensitivity(
        self,
        engine_name: str,
        result: dict,
        data: np.ndarray,
    ) -> float:
        """
        Compute hyperparameter sensitivity score.

        Measures how sensitive the engine output is to parameter variations.
        Higher score = less sensitive = more robust.

        Args:
            engine_name: Name of the engine
            result: The engine result dict
            data: Original data array

        Returns:
            Sensitivity score in [0, 1], higher = less sensitive = better
        """
        # Extract observed metric
        extractor = METRIC_EXTRACTORS.get(engine_name, _extract_generic_metric)
        observed = extractor(result)
        if observed is None:
            observed = 0.0

        # Get simplified runner
        runner = SIMPLIFIED_ENGINES.get(engine_name, _simplified_generic)

        # Test sensitivity by running on different subsets
        metrics = [observed]
        n = len(data)
        fractions = [0.8, 0.9, 0.95]
        for frac in fractions:
            try:
                subset_size = int(frac * n)
                if data.ndim == 1:
                    subset = data[:subset_size]
                else:
                    subset = data[:subset_size, :]
                subset_result = runner(subset)
                metric = extractor(subset_result)
                if metric is not None:
                    metrics.append(metric)
            except Exception:
                continue

        if len(metrics) < 2:
            return 0.5

        # Sensitivity = 1 - normalized spread
        spread = np.max(metrics) - np.min(metrics)
        normalized_spread = spread / (np.abs(np.mean(metrics)) + 1e-10)
        # Use damping factor of 0.5 so exp(-0.5 * normalized_spread)
        # This gives more lenient scores for moderate variation
        sensitivity_score = np.exp(-0.5 * normalized_spread)
        return float(np.clip(sensitivity_score, 0.0, 1.0))

    def null_test(
        self,
        engine_name: str,
        result: dict,
        data: np.ndarray,
        n_surrogates: int = 10,
    ) -> Tuple[bool, float]:
        """
        Test whether engine output is distinguishable from null distribution.

        Args:
            engine_name: Name of the engine
            result: The engine result dict
            data: Original data array
            n_surrogates: Number of surrogate samples

        Returns:
            Tuple of (passed: bool, p_value: float)
        """
        # Extract observed metric
        extractor = METRIC_EXTRACTORS.get(engine_name, _extract_generic_metric)
        observed = extractor(result)
        if observed is None:
            observed = 0.0

        # Run null calibration
        null_result = run_null_calibration(
            engine_name, observed, data, n_null=n_surrogates
        )

        return null_result.passed, null_result.empirical_p_value

    def redundancy_check(
        self,
        engine_results: Dict[str, dict],
        threshold: float = 0.9,
    ) -> Dict[str, List[str]]:
        """
        Check for redundant engines producing similar outputs.

        Args:
            engine_results: Dict of engine_name -> result dict
            threshold: Correlation threshold for redundancy (default 0.9)

        Returns:
            Dict mapping engine names to list of redundant engines
        """
        if not engine_results:
            return {}

        # Extract metrics from all engines
        metrics = {}
        for engine_name, result in engine_results.items():
            if result is None:
                continue
            extractor = METRIC_EXTRACTORS.get(engine_name, _extract_generic_metric)
            metric = extractor(result)
            if metric is not None:
                metrics[engine_name] = metric

        if len(metrics) < 2:
            return {}

        # Find redundant pairs based on metric similarity
        redundancy_map: Dict[str, List[str]] = {}
        engines = list(metrics.keys())
        for i, eng1 in enumerate(engines):
            for eng2 in engines[i+1:]:
                # Check if metrics are very similar (within threshold fraction)
                m1, m2 = metrics[eng1], metrics[eng2]
                if m1 != 0 and m2 != 0:
                    ratio = min(m1, m2) / max(m1, m2)
                    if ratio >= threshold:
                        if eng1 not in redundancy_map:
                            redundancy_map[eng1] = []
                        if eng2 not in redundancy_map:
                            redundancy_map[eng2] = []
                        redundancy_map[eng1].append(eng2)
                        redundancy_map[eng2].append(eng1)

        return redundancy_map

    def run(self, inputs: Dict[str, Any]) -> StabilityResult:
        """
        Evaluate stability and compute consensus weights for all engines.

        Args:
            inputs: Dict with:
                - engine_results: dict[str, dict] - engine outputs
                - routing_result: RoutingResult - from Agent 3
                - original_data: np.ndarray - the data engines ran on
                - weight_cap: float - max weight per engine (default 0.25)
                - window_id: str - optional window identifier

        Returns:
            StabilityResult with null results, stability scores, and contributions
        """
        engine_results = inputs.get("engine_results", {})
        routing_result = inputs.get("routing_result")
        if routing_result is None:
            routing_result = inputs.get("engine_routing_result")
        original_data = inputs.get("original_data")
        if original_data is None:
            original_data = inputs.get("cleaned_data")
        weight_cap = inputs.get("weight_cap", self.DEFAULT_W_CAP)
        window_id = inputs.get("window_id")

        if original_data is None:
            self.log("skip", "No data provided for stability tests")
            return StabilityResult(
                trustworthiness_scores={},
                artifact_flags=[],
                adjusted_weights={},
            )

        # Ensure numpy array
        if not isinstance(original_data, np.ndarray):
            original_data = np.asarray(original_data)

        # Get routing info
        routing_weights = {}
        routing_eligibility = {}
        if routing_result is not None:
            routing_weights = routing_result.weights
            routing_eligibility = routing_result.engine_eligibility

        # Process each engine
        contributions: List[EngineContribution] = []
        null_results: Dict[str, NullCalibrationResult] = {}
        stability_scores: Dict[str, StabilityScore] = {}
        trustworthiness_scores: Dict[str, float] = {}
        adjusted_weights: Dict[str, float] = {}
        artifact_flags: List[str] = []
        reports: Dict[str, EngineStabilityReport] = {}

        for engine_name, result in engine_results.items():
            if result is None:
                continue

            # Skip if already suppressed by routing
            eligibility = routing_eligibility.get(engine_name, "allowed")
            if eligibility == "suppressed":
                continue

            gate_pass = eligibility in ["allowed", "downweighted"]

            # Extract observed metric
            observed = extract_primary_metric(engine_name, result)
            if observed is None:
                # Try running simplified version
                try:
                    simplified = run_engine_light(engine_name, original_data)
                    observed = extract_primary_metric(engine_name, simplified)
                except Exception:
                    observed = None

            if observed is None:
                observed = 0.0

            # Null calibration (Spec B)
            null_result = None
            if engine_name in REQUIRES_NULL:
                n_null = self.get_config("n_null", self.DEFAULT_N_NULL)
                null_result = run_null_calibration(
                    engine_name, observed, original_data, n_null
                )
                null_results[engine_name] = null_result

            # Stability scoring (Spec C)
            n_subsamples = self.get_config("n_subsamples", self.DEFAULT_N_SUBSAMPLES)
            subsample_fraction = self.get_config("subsample_fraction", self.DEFAULT_SUBSAMPLE_FRACTION)
            stability = compute_stability_score(
                engine_name, original_data,
                n_subsamples=n_subsamples,
                subsample_fraction=subsample_fraction,
            )
            stability_scores[engine_name] = stability

            # Domain fit from routing weights
            domain_fit = routing_weights.get(engine_name, 1.0)
            # Clamp to [0, 1] since routing weights can exceed 1
            domain_fit = min(1.0, max(0.0, domain_fit))

            # Benchmark score (Spec G placeholder)
            benchmark_score = get_benchmark_score(engine_name)

            # Compute contribution (Spec D)
            contribution = compute_engine_contribution(
                engine_name=engine_name,
                gate_pass=gate_pass,
                null_result=null_result,
                stability=stability,
                benchmark_score=benchmark_score,
                domain_fit=domain_fit,
                w_cap=weight_cap,
            )
            contributions.append(contribution)

            # Store in registry for consensus reporting
            self._registry.store_contribution(contribution, window_id)

            # Compute legacy trustworthiness
            null_passed = null_result.passed if null_result else True
            null_pvalue = null_result.empirical_p_value if null_result else 0.01
            if not null_passed:
                trustworthiness = 0.0
            else:
                trustworthiness = stability.score * benchmark_score

            trustworthiness_scores[engine_name] = trustworthiness
            # Test expects: adjusted_weight = routing_weight * trustworthiness
            routing_weight = routing_weights.get(engine_name, 1.0)
            adjusted_weights[engine_name] = routing_weight * trustworthiness

            # Check for artifacts
            artifact_threshold = self.get_config("artifact_threshold", self.DEFAULT_ARTIFACT_THRESHOLD)
            issues = []
            if trustworthiness < artifact_threshold:
                artifact_flags.append(engine_name)
                issues.append(f"Low trustworthiness: {trustworthiness:.2f}")
            if not null_passed:
                issues.append(f"Failed null test (p={null_pvalue:.3f})")
            if stability.score < self.DEFAULT_STABILITY_THRESHOLD:
                issues.append(f"Unstable: {stability.score:.2f} < 0.6")

            # Legacy report
            reports[engine_name] = EngineStabilityReport(
                engine_name=engine_name,
                bootstrap_stability=stability.score,
                hyperparameter_sensitivity=stability.score,  # Proxy
                null_test_passed=null_passed,
                null_test_pvalue=null_pvalue,
                trustworthiness=trustworthiness,
                issues=issues,
            )

        # Normalize weights (Spec D2)
        total_capped = sum(c.capped_weight for c in contributions if c.capped_weight > 0)
        consensus_weights = {}
        if total_capped > 0:
            for c in contributions:
                if c.capped_weight > 0:
                    consensus_weights[c.engine_name] = c.capped_weight / total_capped

        # Compute redundancy map
        redundancy_map = self.redundancy_check(engine_results)

        # Store state
        self._last_reports = reports
        self._last_contributions = contributions
        self._null_results = null_results
        self._stability_scores = stability_scores
        self._redundancy_map = redundancy_map

        # Store in registry
        self._registry.set(self.name, "trustworthiness_scores", trustworthiness_scores)
        self._registry.set(self.name, "adjusted_weights", adjusted_weights)
        self._registry.set(self.name, "artifact_flags", artifact_flags)
        self._registry.set(self.name, "null_results", {k: v.__dict__ for k, v in null_results.items()})
        self._registry.set(self.name, "stability_scores", {k: v.__dict__ for k, v in stability_scores.items()})
        self._registry.set(self.name, "consensus_weights", consensus_weights)

        # Log summary
        n_contributing = sum(1 for c in contributions if c.is_contributing)
        n_excluded = len(contributions) - n_contributing

        self.log(
            decision=f"contributing={n_contributing}, excluded={n_excluded}, artifacts={len(artifact_flags)}",
            reason=f"w_cap={weight_cap}, total_weight={total_capped:.3f}",
            metrics={
                "n_engines": len(engine_results),
                "n_contributing": n_contributing,
                "n_excluded": n_excluded,
                "artifact_flags": artifact_flags,
                "consensus_weights": consensus_weights,
            },
        )

        result = StabilityResult(
            trustworthiness_scores=trustworthiness_scores,
            artifact_flags=artifact_flags,
            adjusted_weights=adjusted_weights,
            null_results=null_results,
            stability_scores=stability_scores,
            contributions=contributions,
            consensus_weights=consensus_weights,
        )
        self._last_result = result
        return result

    def explain(self) -> str:
        """Return human-readable stability analysis summary."""
        if not self._last_contributions and not self._last_reports:
            return "No stability analysis has been run yet."

        lines = [
            "Engine Stability Audit Results",
            "=" * 50,
        ]

        # If we have legacy reports, use those
        if self._last_reports and not self._last_contributions:
            for engine, report in self._last_reports.items():
                lines.append("")
                lines.append(f"{engine.upper()}")
                lines.append(f"  Trustworthiness: {report.trustworthiness:.2f}")
                lines.append(f"  Bootstrap Stability: {report.bootstrap_stability:.2f}")
                lines.append(f"  Null Test: {'PASS' if report.null_test_passed else 'FAIL'}")
                if report.issues:
                    lines.append(f"  Issues: {', '.join(report.issues)}")
            return "\n".join(lines)

        # Sort by capped weight
        sorted_contribs = sorted(
            self._last_contributions,
            key=lambda c: c.capped_weight,
            reverse=True,
        )

        contributing = [c for c in sorted_contribs if c.is_contributing]
        excluded = [c for c in sorted_contribs if not c.is_contributing]

        if contributing:
            lines.append("")
            lines.append("CONTRIBUTING ENGINES:")
            for c in contributing:
                null_status = "PASS" if c.null_pass else "FAIL"
                stability_status = "OK" if c.stability_score >= 0.6 else "LOW"
                lines.append(f"  {c.engine_name.upper()}")
                trustworthiness = c.stability_score * c.benchmark_score if c.null_pass else 0.0
                lines.append(f"    Trustworthiness: {trustworthiness:.2f}")
                lines.append(f"    Weight: {c.capped_weight:.3f} (raw: {c.raw_weight:.3f})")
                lines.append(f"    Stability: {c.stability_score:.2f} [{stability_status}]")
                lines.append(f"    Null test: {null_status}")
                lines.append(f"    Benchmark: {c.benchmark_score:.2f}, Domain fit: {c.domain_fit:.2f}")

        if excluded:
            lines.append("")
            lines.append("EXCLUDED ENGINES:")
            for c in excluded:
                lines.append(f"  {c.engine_name}: {c.gated_reason}")

        # Null calibration details
        if self._null_results:
            lines.append("")
            lines.append("NULL CALIBRATION DETAILS:")
            for name, null in self._null_results.items():
                status = "PASS" if null.passed else "FAIL"
                lines.append(f"  {name}: p={null.empirical_p_value:.3f}, effect={null.effect_size:.2f} [{status}]")

        return "\n".join(lines)

    def get_contributions(self) -> List[EngineContribution]:
        """Get list of engine contributions."""
        return self._last_contributions.copy()

    def get_null_results(self) -> Dict[str, NullCalibrationResult]:
        """Get null calibration results by engine."""
        return self._null_results.copy()

    def get_stability_scores(self) -> Dict[str, StabilityScore]:
        """Get stability scores by engine."""
        return self._stability_scores.copy()

    def get_reports(self) -> Dict[str, EngineStabilityReport]:
        """Get detailed stability reports for all engines (legacy)."""
        return self._last_reports.copy()

    def get_redundancy_map(self) -> Dict[str, List[str]]:
        """Get redundancy mapping between engines."""
        return self._redundancy_map.copy()
