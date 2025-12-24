"""
PRISM Data Quality Audit Agent - Spec Section A Implementation

Validates preprocessing against Engine Admission Control Spec Section A
(Standard Inputs) requirements including:
    A2: Frequency alignment (same timestamp grid)
    A3: Missing data thresholds (5% multivariate, 10% univariate)
    A4: Domain-specific outlier handling
    A5: Scaling method detection

Also includes additional quality checks:
    - Variance collapse detection
    - Stationarity forcing detection
    - Distribution shift detection
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from scipy import stats

from prism.config import get_data_quality_config
from .agent_foundation import (
    BaseAgent,
    DataQualityResult,
    DiagnosticRegistry,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Climate Indices (Spec A4: No winsorization)
# =============================================================================

CLIMATE_INDICES = {"enso", "nao", "pdo", "soi", "mei", "oni", "amo", "ao"}


# =============================================================================
# Diagnostic Result Container
# =============================================================================

@dataclass
class SpecADiagnostics:
    """Container for Spec Section A diagnostic results."""
    # A2: Frequency alignment
    frequency_aligned: bool = True
    frequency_details: str = ""

    # A3: Missing data
    missing_rate: float = 0.0
    missing_threshold: float = 0.05  # Depends on engine type
    missing_gate_passed: bool = True

    # A4: Outlier handling
    outlier_policy: str = "none"
    outlier_compliant: bool = True
    outlier_details: str = ""

    # A5: Scaling
    scaling_method: str = "none"

    # Additional checks
    variance_ratio: float = 1.0
    variance_collapsed: bool = False

    stationarity_forced: bool = False
    stationarity_raw_pvalue: float = 1.0
    stationarity_cleaned_pvalue: float = 1.0

    distribution_shift_pvalue: float = 1.0
    distribution_shifted: bool = False

    @property
    def critical_failures(self) -> int:
        """Count critical failures (missing > threshold, variance collapse)."""
        count = 0
        if not self.missing_gate_passed:
            count += 1
        if self.variance_collapsed:
            count += 1
        return count

    @property
    def warnings(self) -> int:
        """Count non-critical warnings."""
        count = 0
        if not self.frequency_aligned:
            count += 1
        if not self.outlier_compliant:
            count += 1
        if self.stationarity_forced:
            count += 1
        if self.distribution_shifted:
            count += 1
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for registry storage."""
        return {
            "frequency_aligned": self.frequency_aligned,
            "frequency_details": self.frequency_details,
            "missing_rate": self.missing_rate,
            "missing_threshold": self.missing_threshold,
            "missing_gate_passed": self.missing_gate_passed,
            "outlier_policy": self.outlier_policy,
            "outlier_compliant": self.outlier_compliant,
            "outlier_details": self.outlier_details,
            "scaling_method": self.scaling_method,
            "variance_ratio": self.variance_ratio,
            "variance_collapsed": self.variance_collapsed,
            "stationarity_forced": self.stationarity_forced,
            "stationarity_raw_pvalue": self.stationarity_raw_pvalue,
            "stationarity_cleaned_pvalue": self.stationarity_cleaned_pvalue,
            "distribution_shift_pvalue": self.distribution_shift_pvalue,
            "distribution_shifted": self.distribution_shifted,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
        }


# =============================================================================
# Data Quality Audit Agent
# =============================================================================

class DataQualityAuditAgent(BaseAgent):
    """
    Validates preprocessing against Spec Section A requirements.

    Implements:
        A2: Frequency alignment check
        A3: Missing data thresholds
        A4: Domain-specific outlier handling
        A5: Scaling method detection

    Plus additional quality checks:
        - Variance collapse detection
        - Stationarity forcing detection
        - Distribution shift detection

    Safety classification:
        - "safe": All critical checks pass, 0-1 warnings
        - "questionable": All critical pass, 2+ warnings
        - "unsafe": Any critical check fails (missing > threshold OR variance collapse)

    Parameters can be configured via config/data_quality.yaml.
    """

    # Class-level defaults (can be overridden by config)
    # Spec A3 thresholds
    MISSING_RATE_MULTIVARIATE = 0.05  # 5%
    MISSING_RATE_UNIVARIATE = 0.10    # 10%

    # Variance collapse threshold
    VARIANCE_COLLAPSE_THRESHOLD = 0.5

    # Distribution shift significance
    DISTRIBUTION_ALPHA = 0.01

    # Stationarity significance
    STATIONARITY_ALPHA = 0.05

    def __init__(
        self,
        registry: DiagnosticRegistry,
        config: Optional[Dict] = None,
    ):
        super().__init__(registry, config)
        self._last_diagnostics: Optional[SpecADiagnostics] = None
        self._last_reasons: List[str] = []

        # Load config and override class defaults
        self._load_config_overrides()

    @property
    def name(self) -> str:
        return "data_quality"

    def _load_config_overrides(self) -> None:
        """Load configuration overrides from config/data_quality.yaml."""
        try:
            cfg = get_data_quality_config()

            # Override class constants with config values
            self.MISSING_RATE_MULTIVARIATE = cfg.get(
                "missing_rate_multivariate", self.MISSING_RATE_MULTIVARIATE
            )
            self.MISSING_RATE_UNIVARIATE = cfg.get(
                "missing_rate_univariate", self.MISSING_RATE_UNIVARIATE
            )
            self.VARIANCE_COLLAPSE_THRESHOLD = cfg.get(
                "variance_collapse_threshold", self.VARIANCE_COLLAPSE_THRESHOLD
            )
            self.DISTRIBUTION_ALPHA = cfg.get(
                "distribution_alpha", self.DISTRIBUTION_ALPHA
            )
            self.STATIONARITY_ALPHA = cfg.get(
                "stationarity_alpha", self.STATIONARITY_ALPHA
            )
        except FileNotFoundError:
            logger.debug("data_quality.yaml not found, using defaults")

    # =========================================================================
    # Spec A2: Frequency Alignment Check
    # =========================================================================

    def frequency_alignment_check(
        self,
        raw: np.ndarray,
        cleaned: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """
        Spec A2: All series must share same timestamp grid.

        Verifies:
        - Uniform spacing in timestamps (if provided)
        - Same length between raw and cleaned
        - No mixed-frequency indicators

        Args:
            raw: Raw time series
            cleaned: Cleaned time series
            timestamps: Optional timestamp array

        Returns:
            (aligned: bool, details: str)
        """
        details = []

        # Check same length
        if len(raw) != len(cleaned):
            details.append(f"Length mismatch: raw={len(raw)}, cleaned={len(cleaned)}")

        # Check timestamp uniformity if provided
        if timestamps is not None and len(timestamps) > 1:
            diffs = np.diff(timestamps.astype(np.float64))
            if len(diffs) > 0:
                # Check for uniform spacing (allow 1% tolerance)
                median_diff = np.median(diffs)
                if median_diff > 0:
                    irregularity = np.std(diffs) / median_diff
                    if irregularity > 0.01:
                        details.append(f"Non-uniform spacing detected (irregularity={irregularity:.4f})")

        aligned = len(details) == 0
        detail_str = "; ".join(details) if details else "Frequency alignment OK"

        return aligned, detail_str

    # =========================================================================
    # Spec A3: Missing Data Check
    # =========================================================================

    def missing_data_check(
        self,
        data: np.ndarray,
        engine_type: str = "multivariate",
    ) -> Tuple[float, bool]:
        """
        Spec A3: Missing rate thresholds.

        Args:
            data: Time series data (may contain NaN)
            engine_type: "multivariate" (5% threshold) or "univariate" (10% threshold)

        Returns:
            (missing_rate: float, passes_gate: bool)
        """
        if data.size == 0:
            return 0.0, True

        # Handle multi-dimensional data
        if data.ndim > 1:
            total_elements = data.size
            nan_count = np.sum(np.isnan(data))
        else:
            total_elements = len(data)
            nan_count = np.sum(np.isnan(data))

        missing_rate = nan_count / total_elements

        # Spec A3 thresholds
        if engine_type == "multivariate":
            threshold = self.get_config(
                "missing_rate_multivariate",
                self.MISSING_RATE_MULTIVARIATE
            )
        else:
            threshold = self.get_config(
                "missing_rate_univariate",
                self.MISSING_RATE_UNIVARIATE
            )

        passes_gate = missing_rate <= threshold

        return missing_rate, passes_gate

    # =========================================================================
    # Spec A4: Outlier Handling Check
    # =========================================================================

    def outlier_handling_check(
        self,
        raw: np.ndarray,
        cleaned: np.ndarray,
        domain: str = "finance",
        indicator_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Spec A4: Domain-specific outlier policy.

        For finance:
            - Verify log returns transformation
            - Verify winsorization at 1%/99%

        For climate indices (ENSO/NAO/PDO):
            - Verify NO winsorization applied

        Args:
            raw: Raw time series
            cleaned: Cleaned time series
            domain: "finance", "climate", "epidemiology"
            indicator_name: Name of indicator (for climate index detection)

        Returns:
            Dict with policy, compliance, and details
        """
        result = {
            "policy_applied": "none",
            "compliant": True,
            "details": "",
        }

        raw_clean = raw[~np.isnan(raw)]
        cleaned_clean = cleaned[~np.isnan(cleaned)]

        if len(raw_clean) < 5 or len(cleaned_clean) < 5:
            return result

        # Detect if log transformation was applied
        is_log_transformed = self._detect_log_transform(raw_clean, cleaned_clean)

        # Detect if winsorization was applied
        is_winsorized, winsorize_pct = self._detect_winsorization(raw_clean, cleaned_clean)

        # Check for climate indices (no winsorize allowed)
        is_climate_index = False
        if indicator_name:
            is_climate_index = indicator_name.lower() in CLIMATE_INDICES

        if domain == "finance":
            # Finance: expect log returns + winsorize
            if is_log_transformed:
                result["policy_applied"] = "log_returns"
                if is_winsorized:
                    result["policy_applied"] += f"+winsorize_{winsorize_pct:.1f}%"
                result["compliant"] = True
                result["details"] = "Finance-compliant transformation"
            elif is_winsorized:
                result["policy_applied"] = f"winsorize_{winsorize_pct:.1f}%"
                result["compliant"] = True
                result["details"] = "Winsorization applied"
            else:
                result["policy_applied"] = "none_detected"
                result["compliant"] = True  # Not strictly required
                result["details"] = "No outlier handling detected"

        elif domain == "climate" or is_climate_index:
            # Climate indices: NO winsorization
            if is_winsorized:
                result["policy_applied"] = f"winsorize_{winsorize_pct:.1f}%"
                result["compliant"] = False
                result["details"] = "WARNING: Winsorization applied to climate index (not allowed per Spec A4)"
            else:
                result["policy_applied"] = "none"
                result["compliant"] = True
                result["details"] = "Climate index correctly not winsorized"

        else:
            # Other domains: just detect what was applied
            if is_winsorized:
                result["policy_applied"] = f"winsorize_{winsorize_pct:.1f}%"
            elif is_log_transformed:
                result["policy_applied"] = "log_transform"
            result["details"] = f"Detected transformation: {result['policy_applied']}"

        return result

    def _detect_log_transform(self, raw: np.ndarray, cleaned: np.ndarray) -> bool:
        """Detect if log transformation was applied."""
        if len(raw) != len(cleaned):
            return False

        # Check if cleaned looks like log(raw) or log returns
        raw_positive = raw[raw > 0]
        if len(raw_positive) < len(raw) * 0.9:
            return False

        try:
            log_raw = np.log(raw_positive)
            # Check correlation with cleaned subset
            cleaned_subset = cleaned[:len(log_raw)]
            if len(log_raw) > 0 and len(cleaned_subset) > 0:
                corr = np.corrcoef(log_raw, cleaned_subset[:len(log_raw)])[0, 1]
                if not np.isnan(corr) and corr > 0.95:
                    return True
        except (ValueError, RuntimeWarning):
            pass

        # Check for log returns pattern (diff of log)
        if len(raw_positive) > 1:
            try:
                log_returns = np.diff(np.log(raw_positive))
                cleaned_diff = np.diff(cleaned)
                if len(log_returns) > 0 and len(cleaned_diff) >= len(log_returns):
                    corr = np.corrcoef(log_returns, cleaned_diff[:len(log_returns)])[0, 1]
                    if not np.isnan(corr) and corr > 0.95:
                        return True
            except (ValueError, RuntimeWarning):
                pass

        return False

    def _detect_winsorization(
        self,
        raw: np.ndarray,
        cleaned: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Detect if winsorization was applied.

        Returns:
            (is_winsorized: bool, percentile: float)
        """
        if len(raw) != len(cleaned) or len(raw) < 10:
            return False, 0.0

        # Check if extreme values were clipped
        raw_sorted = np.sort(raw)
        cleaned_sorted = np.sort(cleaned)

        # Compare tails
        n = len(raw)
        tail_size = max(1, int(n * 0.02))  # Check ~2% tails

        # Lower tail
        lower_clipped = np.allclose(
            cleaned_sorted[:tail_size],
            np.full(tail_size, cleaned_sorted[tail_size]),
            rtol=0.01
        )

        # Upper tail
        upper_clipped = np.allclose(
            cleaned_sorted[-tail_size:],
            np.full(tail_size, cleaned_sorted[-tail_size - 1]),
            rtol=0.01
        )

        # Alternative: check if cleaned min/max are different from raw
        raw_range = np.ptp(raw_sorted)
        cleaned_range = np.ptp(cleaned_sorted)

        if raw_range > 0:
            range_reduction = 1 - (cleaned_range / raw_range)
            if range_reduction > 0.05:  # >5% range reduction
                # Estimate percentile
                pct = min(5.0, range_reduction * 50)
                return True, pct

        if lower_clipped or upper_clipped:
            return True, 1.0

        return False, 0.0

    # =========================================================================
    # Spec A5: Scaling Check
    # =========================================================================

    def scaling_check(self, cleaned: np.ndarray) -> str:
        """
        Spec A5: Detect scaling method applied.

        Detects:
        - zscore: mean~0, std~1
        - robust_z: median~0, MAD-based scaling
        - rank: uniform distribution in [0, 1] or rank-like

        Args:
            cleaned: Cleaned/scaled time series

        Returns:
            Detected scaling method name
        """
        clean = cleaned[~np.isnan(cleaned)]
        if len(clean) < 10:
            return "none"

        mean = np.mean(clean)
        std = np.std(clean, ddof=1)
        median = np.median(clean)
        mad = np.median(np.abs(clean - median))

        # For gaussian data, zscore and robust_z produce similar results.
        # Distinguish by checking which center (mean vs median) is closer to 0.
        # Robust z-score centers on median, zscore centers on mean.
        mean_closeness = abs(mean)
        median_closeness = abs(median)

        # Check robust z-score first if median is closer to 0 than mean
        # (robust z-score explicitly centers on median)
        if median_closeness <= mean_closeness and abs(median) < 0.1 and mad > 0:
            normalized_mad = mad * 1.4826  # Scale to match std
            if abs(normalized_mad - 1.0) < 0.3:
                return "robust_z"

        # Check z-score: mean ~ 0, std ~ 1
        if abs(mean) < 0.1 and abs(std - 1.0) < 0.2:
            return "zscore"

        # Fallback check for robust z-score (in case mean check was stricter)
        if abs(median) < 0.1 and mad > 0:
            normalized_mad = mad * 1.4826
            if abs(normalized_mad - 1.0) < 0.3:
                return "robust_z"

        # Check rank transform: values in [0, 1] or [0, n-1]
        min_val = np.min(clean)
        max_val = np.max(clean)

        if min_val >= 0 and max_val <= 1:
            # Check for uniform distribution (rank transform)
            _, p_uniform = stats.kstest(clean, 'uniform')
            if p_uniform > 0.05:
                return "rank"

        # Check if values look like ranks (integers or near-integers)
        if np.allclose(clean, np.round(clean), atol=0.01):
            unique_ratio = len(np.unique(clean)) / len(clean)
            if unique_ratio > 0.9:  # Most values unique
                return "rank"

        return "none"

    # =========================================================================
    # Variance Collapse Check
    # =========================================================================

    def variance_collapse_check(
        self,
        raw: np.ndarray,
        cleaned: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Check for excessive variance reduction (smoothing).

        Args:
            raw: Raw time series
            cleaned: Cleaned time series

        Returns:
            (variance_ratio: float, is_collapsed: bool)
        """
        raw_clean = raw[~np.isnan(raw)]
        cleaned_clean = cleaned[~np.isnan(cleaned)]

        if len(raw_clean) < 2 or len(cleaned_clean) < 2:
            return 1.0, False

        raw_var = np.var(raw_clean, ddof=1)
        cleaned_var = np.var(cleaned_clean, ddof=1)

        if raw_var == 0:
            return 1.0, False

        ratio = cleaned_var / raw_var
        threshold = self.get_config(
            "variance_collapse_threshold",
            self.VARIANCE_COLLAPSE_THRESHOLD
        )

        return ratio, ratio < threshold

    # =========================================================================
    # Stationarity Forcing Check
    # =========================================================================

    def stationarity_forcing_check(
        self,
        raw: np.ndarray,
        cleaned: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """
        Check if stationarity was forced (raw non-stationary, cleaned stationary).

        Args:
            raw: Raw time series
            cleaned: Cleaned time series

        Returns:
            (is_forced: bool, raw_pvalue: float, cleaned_pvalue: float)
        """
        raw_pvalue = self._adf_pvalue(raw)
        cleaned_pvalue = self._adf_pvalue(cleaned)

        alpha = self.get_config("stationarity_alpha", self.STATIONARITY_ALPHA)

        # Flag: raw non-stationary (p > alpha) but cleaned stationary (p < alpha)
        raw_nonstationary = raw_pvalue > alpha
        cleaned_stationary = cleaned_pvalue < alpha
        is_forced = raw_nonstationary and cleaned_stationary

        return is_forced, raw_pvalue, cleaned_pvalue

    def _adf_pvalue(self, series: np.ndarray) -> float:
        """Run Augmented Dickey-Fuller test, return p-value."""
        clean = series[~np.isnan(series)]
        if len(clean) < 10:
            return 1.0

        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(clean, autolag="AIC")
            return float(result[1])
        except (ImportError, ValueError):
            # Fallback: simple variance ratio test
            n = len(clean)
            first_half = clean[:n // 2]
            second_half = clean[n // 2:]
            v1, v2 = np.var(first_half), np.var(second_half)
            if v1 == 0 or v2 == 0:
                return 0.5
            ratio = v1 / v2
            return 0.5 if 0.5 < ratio < 2.0 else 0.1

    # =========================================================================
    # Distribution Shift Check
    # =========================================================================

    def distribution_shift_check(
        self,
        raw: np.ndarray,
        cleaned: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Check for significant distribution change using KS test.

        Args:
            raw: Raw time series
            cleaned: Cleaned time series

        Returns:
            (p_value: float, is_shifted: bool)
        """
        raw_clean = raw[~np.isnan(raw)]
        cleaned_clean = cleaned[~np.isnan(cleaned)]

        if len(raw_clean) < 5 or len(cleaned_clean) < 5:
            return 1.0, False

        _, pvalue = stats.ks_2samp(raw_clean, cleaned_clean)
        alpha = self.get_config("distribution_alpha", self.DISTRIBUTION_ALPHA)

        return float(pvalue), pvalue < alpha

    # =========================================================================
    # Main Interface
    # =========================================================================

    def run(self, inputs: Dict[str, Any]) -> DataQualityResult:
        """
        Execute all Spec Section A checks on raw vs cleaned data.

        Args:
            inputs: Dict with:
                - raw_data: numpy array or dict of arrays
                - cleaned_data: numpy array or dict of arrays
                - domain: str ("finance", "climate", "epidemiology")
                - engine_type: str ("multivariate" or "univariate")
                - timestamps: Optional timestamp array
                - indicator_name: Optional indicator name

        Returns:
            DataQualityResult with all Spec A fields populated
        """
        raw_data = inputs.get("raw_data")
        cleaned_data = inputs.get("cleaned_data")
        domain = inputs.get("domain", "finance")
        engine_type = inputs.get("engine_type", "multivariate")
        timestamps = inputs.get("timestamps")
        indicator_name = inputs.get("indicator_name")

        if raw_data is None or cleaned_data is None:
            self.log("skip", "Missing raw or cleaned data")
            return DataQualityResult(
                frequency_aligned=False,
                missing_rate=1.0,
                safety_flag="unsafe",
                reasons=["Missing input data"],
            )

        # Handle dict of multiple series or single series
        if isinstance(raw_data, dict):
            return self._run_multi(raw_data, cleaned_data, domain, engine_type)
        else:
            return self._run_single(
                np.asarray(raw_data),
                np.asarray(cleaned_data),
                domain,
                engine_type,
                timestamps,
                indicator_name,
            )

    def _run_single(
        self,
        raw: np.ndarray,
        cleaned: np.ndarray,
        domain: str,
        engine_type: str,
        timestamps: Optional[np.ndarray] = None,
        indicator_name: Optional[str] = None,
    ) -> DataQualityResult:
        """Run all Spec A checks on a single pair of series."""

        # A2: Frequency alignment
        freq_aligned, freq_details = self.frequency_alignment_check(
            raw, cleaned, timestamps
        )

        # A3: Missing data
        missing_rate, missing_passed = self.missing_data_check(cleaned, engine_type)
        missing_threshold = (
            self.MISSING_RATE_MULTIVARIATE if engine_type == "multivariate"
            else self.MISSING_RATE_UNIVARIATE
        )

        # A4: Outlier handling
        outlier_result = self.outlier_handling_check(
            raw, cleaned, domain, indicator_name
        )

        # A5: Scaling
        scaling_method = self.scaling_check(cleaned)

        # Additional checks
        variance_ratio, variance_collapsed = self.variance_collapse_check(raw, cleaned)
        stat_forced, stat_raw_p, stat_cleaned_p = self.stationarity_forcing_check(
            raw, cleaned
        )
        dist_pvalue, dist_shifted = self.distribution_shift_check(raw, cleaned)

        # Build diagnostics
        diag = SpecADiagnostics(
            frequency_aligned=freq_aligned,
            frequency_details=freq_details,
            missing_rate=missing_rate,
            missing_threshold=missing_threshold,
            missing_gate_passed=missing_passed,
            outlier_policy=outlier_result["policy_applied"],
            outlier_compliant=outlier_result["compliant"],
            outlier_details=outlier_result["details"],
            scaling_method=scaling_method,
            variance_ratio=variance_ratio,
            variance_collapsed=variance_collapsed,
            stationarity_forced=stat_forced,
            stationarity_raw_pvalue=stat_raw_p,
            stationarity_cleaned_pvalue=stat_cleaned_p,
            distribution_shift_pvalue=dist_pvalue,
            distribution_shifted=dist_shifted,
        )

        self._last_diagnostics = diag

        # Determine safety flag
        if diag.critical_failures > 0:
            safety_flag = "unsafe"
        elif diag.warnings >= 2:
            safety_flag = "questionable"
        else:
            safety_flag = "safe"

        # Build reasons list
        reasons = []
        if not missing_passed:
            reasons.append(
                f"Missing rate {missing_rate*100:.1f}% exceeds {engine_type} "
                f"threshold {missing_threshold*100:.0f}% (Spec A3)"
            )
        if variance_collapsed:
            reasons.append(
                f"Variance collapsed to {variance_ratio*100:.1f}% "
                f"(< {self.VARIANCE_COLLAPSE_THRESHOLD*100:.0f}% threshold)"
            )
        if not freq_aligned:
            reasons.append(f"Frequency alignment issue: {freq_details}")
        if not outlier_result["compliant"]:
            reasons.append(f"Outlier policy non-compliant: {outlier_result['details']}")
        if stat_forced:
            reasons.append(
                f"Stationarity forced: raw p={stat_raw_p:.3f}, cleaned p={stat_cleaned_p:.3f}"
            )
        if dist_shifted:
            reasons.append(f"Distribution shifted: KS p={dist_pvalue:.4f}")

        self._last_reasons = reasons

        # Store in registry
        for key, value in diag.to_dict().items():
            self._registry.set(self.name, key, value)
        self._registry.set(self.name, "safety_flag", safety_flag)

        # Log decision
        self.log(
            decision=f"safety_flag={safety_flag}",
            reason=f"critical={diag.critical_failures}, warnings={diag.warnings}",
            metrics=diag.to_dict(),
        )

        result = DataQualityResult(
            frequency_aligned=freq_aligned,
            missing_rate=missing_rate,
            missing_policy_applied="gate" if not missing_passed else "pass",
            outlier_handling=outlier_result["policy_applied"],
            scaling_method=scaling_method,
            safety_flag=safety_flag,
            reasons=reasons,
            transformation_summary=diag.to_dict(),
        )

        self._last_result = result
        return result

    def _run_multi(
        self,
        raw_dict: Dict[str, np.ndarray],
        cleaned_dict: Dict[str, np.ndarray],
        domain: str,
        engine_type: str,
    ) -> DataQualityResult:
        """Run checks on multiple series, aggregate results."""
        all_summaries = {}
        all_reasons = []
        worst_flag = "safe"
        flag_order = {"safe": 0, "questionable": 1, "unsafe": 2}

        # Aggregate metrics
        total_missing = 0.0
        all_freq_aligned = True
        all_outlier_compliant = True
        n_series = 0

        for name in raw_dict:
            if name not in cleaned_dict:
                continue

            raw = np.asarray(raw_dict[name])
            cleaned = np.asarray(cleaned_dict[name])

            result = self._run_single(
                raw, cleaned, domain, engine_type,
                indicator_name=name,
            )

            all_summaries[name] = result.transformation_summary
            if result.reasons:
                all_reasons.extend([f"[{name}] {r}" for r in result.reasons])

            if flag_order.get(result.safety_flag, 0) > flag_order.get(worst_flag, 0):
                worst_flag = result.safety_flag

            total_missing += result.missing_rate
            if not result.frequency_aligned:
                all_freq_aligned = False
            n_series += 1

        avg_missing = total_missing / max(1, n_series)
        self._last_reasons = all_reasons

        return DataQualityResult(
            frequency_aligned=all_freq_aligned,
            missing_rate=avg_missing,
            missing_policy_applied="aggregate",
            outlier_handling="mixed",
            scaling_method="mixed",
            safety_flag=worst_flag,
            reasons=all_reasons,
            transformation_summary=all_summaries,
        )

    def explain(self) -> str:
        """Return human-readable explanation of the last run."""
        if self._last_result is None:
            return "No analysis has been run yet."

        result = self._last_result
        lines = [
            "Data Quality Audit (Spec Section A)",
            "=" * 50,
            f"Safety Flag: {result.safety_flag.upper()}",
            "",
        ]

        if self._last_diagnostics:
            diag = self._last_diagnostics
            lines.extend([
                "Spec A2 - Frequency Alignment:",
                f"  Status: {'ALIGNED' if diag.frequency_aligned else 'MISALIGNED'}",
                f"  Details: {diag.frequency_details}",
                "",
                "Spec A3 - Missing Data:",
                f"  Rate: {diag.missing_rate*100:.2f}%",
                f"  Threshold: {diag.missing_threshold*100:.0f}%",
                f"  Gate: {'PASSED' if diag.missing_gate_passed else 'FAILED'}",
                "",
                "Spec A4 - Outlier Handling:",
                f"  Policy: {diag.outlier_policy}",
                f"  Compliant: {'YES' if diag.outlier_compliant else 'NO'}",
                f"  Details: {diag.outlier_details}",
                "",
                "Spec A5 - Scaling:",
                f"  Method: {diag.scaling_method}",
                "",
                "Additional Checks:",
                f"  Variance Ratio: {diag.variance_ratio:.3f} "
                f"{'[COLLAPSED]' if diag.variance_collapsed else '[OK]'}",
                f"  Stationarity: raw p={diag.stationarity_raw_pvalue:.3f}, "
                f"cleaned p={diag.stationarity_cleaned_pvalue:.3f} "
                f"{'[FORCED]' if diag.stationarity_forced else '[OK]'}",
                f"  Distribution: KS p={diag.distribution_shift_pvalue:.4f} "
                f"{'[SHIFTED]' if diag.distribution_shifted else '[OK]'}",
                "",
            ])

        if self._last_reasons:
            lines.append("Issues Found:")
            for reason in self._last_reasons:
                lines.append(f"  - {reason}")
        else:
            lines.append("No issues detected. Data quality passes all Spec A checks.")

        return "\n".join(lines)

    # =========================================================================
    # Helper Methods for External Access
    # =========================================================================

    def get_diagnostics(self) -> Optional[SpecADiagnostics]:
        """Get the last computed diagnostics."""
        return self._last_diagnostics

    def get_missing_threshold(self, engine_type: str = "multivariate") -> float:
        """Get the missing rate threshold for the given engine type."""
        if engine_type == "multivariate":
            return self.MISSING_RATE_MULTIVARIATE
        return self.MISSING_RATE_UNIVARIATE
