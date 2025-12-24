"""
PRISM Blind Spot Detection Agent

Agent 5: Asks "Given the geometry we detected, what should we expect to see?"
Then checks if we actually see it.

Identifies missing expected features, unexplained residual structure, and
cross-engine anomalies. Provides escalation recommendations.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
from scipy import stats

# Import geometry classes
from prism.agents.agent_geometry_signature import GeometryProfile, GeometryClass

from .agent_foundation import (
    BaseAgent,
    BlindSpotResult,
    StabilityResult,
    DiagnosticRegistry,
    EngineContribution,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Expected Features by Geometry Type
# =============================================================================

EXPECTED_FEATURES = {
    GeometryClass.LATENT_FLOW: {
        "cumulative_peak": "Clear peak in cumulative signal",
        "saturation": "Saturation signature (slowing growth rate)",
        "low_residual_autocorr": "Low residual autocorrelation post-peak",
    },
    GeometryClass.COUPLED_OSCILLATOR: {
        "spectral_peaks": "Significant spectral peaks (from wavelets)",
        "cross_correlation_peaks": "Cross-correlation peaks at specific lags",
        "phase_coherence": "Phase coherence between variables",
    },
    GeometryClass.REFLEXIVE_STOCHASTIC: {
        "volatility_clustering": "Volatility clustering (GARCH effects)",
        "state_persistence": "State persistence (from HMM if ran)",
        "fat_tails": "Fat tails in return distribution",
    },
    GeometryClass.PURE_NOISE: {
        # No specific features expected
    },
}


# =============================================================================
# Escalation Recommendations
# =============================================================================

ESCALATION_MAP = {
    "missing_latent_structure": "Consider latent-state model (e.g., Kalman filter, state-space model)",
    "unexplained_periodicity": "Consider harmonic regression or seasonal decomposition",
    "unexplained_fat_tails": "Consider heavy-tail copula or extreme value theory",
    "high_residual_autocorr": "Consider ARIMA residual modeling or state-switching",
    "engine_disagreement": "Consider ensemble weighting or model selection criteria",
    "missing_state_structure": "Consider hidden Markov model with more states",
    "unexplained_nonlinearity": "Consider threshold models or neural network approaches",
}


# =============================================================================
# Spec H: Geometry Expectations (alternate format for documentation)
# =============================================================================

GEOMETRY_EXPECTATIONS = {
    "latent_flow": [
        "saturation_detected",
        "low_residual_autocorr",
        "asymmetric_rise_decay",
    ],
    "coupled_oscillator": [
        "spectral_peaks_significant",
        "phase_coherence_present",
        "granger_edges_stable",
    ],
    "reflexive_stochastic": [
        "volatility_clustering",
        "state_persistence",
        "fat_tails",
    ],
}


# =============================================================================
# Spec H: Sensitivity Check
# =============================================================================

def sensitivity_check(
    contributions: List[EngineContribution],
    consensus_result: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Spec H: Does consensus change if one engine is removed?

    Computes leave-one-out sensitivity for each contributing engine.
    An engine is "critical" if removing it changes consensus by >20%.

    Args:
        contributions: List of EngineContribution from stability agent
        consensus_result: Optional consensus metrics dict

    Returns:
        Dict with:
            - engine_name: {weight, consensus_change_if_removed, is_critical}
            - stable: bool (True if no engine is critical)
            - max_change: float (largest consensus change)
    """
    sensitivity = {}

    # Filter to contributing engines only
    contributing = [c for c in contributions if c.capped_weight > 0]

    if len(contributing) < 2:
        return {
            "stable": False,
            "max_change": 1.0,
            "vulnerable_to": [],
            "message": "Too few contributing engines for sensitivity analysis",
        }

    # Compute total weight
    total_weight = sum(c.capped_weight for c in contributing)

    # For each engine, compute what happens if we remove it
    max_change = 0.0
    vulnerable_to = []

    for contrib in contributing:
        # Remaining engines
        others = [c for c in contributing if c.engine_name != contrib.engine_name]
        other_total = sum(c.capped_weight for c in others)

        if other_total == 0:
            # This was the only engine
            change = 1.0
        else:
            # Compute how much the normalized weights change
            # For each remaining engine, new_norm = c.weight / other_total
            # old_norm = c.weight / total_weight
            # change = max(|new_norm - old_norm|)
            max_weight_shift = 0.0
            for c in others:
                new_norm = c.capped_weight / other_total
                old_norm = c.capped_weight / total_weight
                shift = abs(new_norm - old_norm)
                max_weight_shift = max(max_weight_shift, shift)

            change = max_weight_shift

        is_critical = change > 0.2  # >20% change = critical dependency

        sensitivity[contrib.engine_name] = {
            "weight": contrib.capped_weight,
            "normalized_weight": contrib.capped_weight / total_weight if total_weight > 0 else 0,
            "consensus_change_if_removed": change,
            "is_critical": is_critical,
        }

        max_change = max(max_change, change)

        if is_critical:
            vulnerable_to.append(contrib.engine_name)

    sensitivity["stable"] = len(vulnerable_to) == 0
    sensitivity["max_change"] = max_change
    sensitivity["vulnerable_to"] = vulnerable_to

    return sensitivity


def normalize_weights(contributions: List[EngineContribution]) -> Dict[str, float]:
    """Normalize weights from a list of contributions."""
    total = sum(c.capped_weight for c in contributions if c.capped_weight > 0)
    if total == 0:
        return {}
    return {c.engine_name: c.capped_weight / total for c in contributions if c.capped_weight > 0}


# =============================================================================
# Blind Spot Detection Agent
# =============================================================================

class BlindSpotDetectionAgent(BaseAgent):
    """
    Detects blind spots in the analysis by checking expected features
    against actual results.

    Checks:
    1. missing_expected_features: Are geometry-expected features present?
    2. unexplained_residuals: Is there structure left unexplained?
    3. cross_engine_anomalies: Do engines disagree unexpectedly?

    Usage:
        agent = BlindSpotDetectionAgent(registry)
        result = agent.run({
            "geometry_diagnostic_result": geometry_profile,
            "engine_results": engine_results,
            "engine_stability_result": stability_result,
        })
    """

    def __init__(
        self,
        registry: DiagnosticRegistry,
        config: Optional[Dict] = None,
    ):
        super().__init__(registry, config)
        self._last_blind_spots: List[str] = []
        self._last_recommendations: List[str] = []
        self._missing_features: List[str] = []
        self._residual_issues: List[str] = []
        self._anomalies: List[str] = []
        self._last_sensitivity: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "blind_spot_detection"

    # =========================================================================
    # Blind Spot Checks
    # =========================================================================

    def missing_expected_features(
        self,
        geometry: GeometryProfile,
        results: Dict[str, dict],
    ) -> List[str]:
        """
        Check if expected features for the detected geometry are present.

        Args:
            geometry: Detected geometry profile
            results: Engine results

        Returns:
            List of missing expected features
        """
        missing = []
        dominant = geometry.dominant_geometry
        expected = EXPECTED_FEATURES.get(dominant, {})

        if dominant == GeometryClass.LATENT_FLOW:
            # Check for cumulative peak detection
            if not self._has_cumulative_peak(results):
                missing.append("cumulative_peak: No clear peak detected in cumulative signal")

            # Check for saturation signature
            if not self._has_saturation_signature(results):
                missing.append("saturation: No saturation/asymptote behavior detected")

            # Check residual autocorrelation
            if self._has_high_residual_autocorr(results):
                missing.append("low_residual_autocorr: High residual autocorrelation remains")

        elif dominant == GeometryClass.COUPLED_OSCILLATOR:
            # Check for spectral peaks
            if not self._has_spectral_peaks(results):
                missing.append("spectral_peaks: No significant spectral peaks from wavelets")

            # Check cross-correlation structure
            if not self._has_cross_correlation_structure(results):
                missing.append("cross_correlation_peaks: No significant cross-correlation lags")

        elif dominant == GeometryClass.REFLEXIVE_STOCHASTIC:
            # Check volatility clustering
            if not self._has_volatility_clustering(results):
                missing.append("volatility_clustering: No volatility clustering detected")

            # Check state persistence
            if not self._has_state_persistence(results):
                missing.append("state_persistence: HMM did not find persistent states")

            # Check fat tails
            if not self._has_fat_tails(results):
                missing.append("fat_tails: Return distribution not heavy-tailed")

        return missing

    def unexplained_residuals(
        self,
        results: Dict[str, dict],
        original_data: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Check if significant structure remains unexplained after all engines.

        Args:
            results: Engine results
            original_data: Original time series for residual analysis

        Returns:
            List of unexplained residual issues
        """
        issues = []

        # Check combined residual autocorrelation
        if original_data is not None:
            residual_autocorr = self._compute_residual_autocorr(results, original_data)
            if residual_autocorr > 0.3:
                issues.append(
                    f"High residual autocorrelation ({residual_autocorr:.2f}) - "
                    "structure remains unexplained"
                )

        # Check for unexplained periodicity
        if self._has_unexplained_periodicity(results, original_data):
            issues.append("Unexplained periodicity detected in residuals")

        # Check for unexplained nonlinearity
        if self._has_unexplained_nonlinearity(results, original_data):
            issues.append("Unexplained nonlinear structure in residuals")

        return issues

    def cross_engine_anomalies(
        self,
        results: Dict[str, dict],
        stability: Optional[StabilityResult],
    ) -> List[str]:
        """
        Check for contradictions between engine outputs.

        Args:
            results: Engine results
            stability: Stability audit results

        Returns:
            List of cross-engine anomalies
        """
        anomalies = []

        # Check HMM vs entropy disagreement
        hmm_result = results.get("hmm")
        entropy_result = results.get("entropy")

        if hmm_result and entropy_result:
            n_states = self._get_n_states(hmm_result)
            entropy_val = self._get_entropy_value(entropy_result)

            if n_states >= 3 and entropy_val is not None and entropy_val < 0.3:
                anomalies.append(
                    f"HMM found {n_states} states but entropy is low ({entropy_val:.2f}) - "
                    "possible overfitting"
                )

            if n_states <= 1 and entropy_val is not None and entropy_val > 0.7:
                anomalies.append(
                    f"HMM found only {n_states} state but entropy is high ({entropy_val:.2f}) - "
                    "possible underfitting"
                )

        # Check PCA vs correlation disagreement
        pca_result = results.get("pca")
        corr_result = results.get("correlation")

        if pca_result and corr_result:
            pca_var = self._get_pca_variance(pca_result)
            mean_corr = self._get_mean_correlation(corr_result)

            if pca_var is not None and mean_corr is not None:
                if pca_var > 0.8 and mean_corr < 0.3:
                    anomalies.append(
                        f"PCA explains {pca_var:.0%} variance but mean correlation is low ({mean_corr:.2f}) - "
                        "check for outlier-driven factor"
                    )

        # Check wavelets vs granger disagreement
        wavelets_result = results.get("wavelets")
        granger_result = results.get("granger")

        if wavelets_result and granger_result:
            has_cycles = self._has_spectral_peaks({"wavelets": wavelets_result})
            has_causality = self._has_granger_causality(granger_result)

            if has_cycles and not has_causality:
                anomalies.append(
                    "Strong cycles detected but no Granger causality - "
                    "cycles may be noise or shared external driver"
                )

        # Check stability artifacts
        if stability and stability.artifact_flags:
            for engine in stability.artifact_flags:
                if engine in results and results[engine] is not None:
                    anomalies.append(
                        f"{engine} produced results but flagged as artifact - "
                        "results may be unreliable"
                    )

        return anomalies

    # =========================================================================
    # Helper Methods - Feature Detection
    # =========================================================================

    def _has_cumulative_peak(self, results: Dict[str, dict]) -> bool:
        """Check if any engine detected a cumulative peak."""
        sir_result = results.get("sir")
        if sir_result:
            if sir_result.get("peak_detected") or sir_result.get("peak_time"):
                return True
        return False

    def _has_saturation_signature(self, results: Dict[str, dict]) -> bool:
        """Check for saturation behavior in results."""
        sir_result = results.get("sir")
        if sir_result:
            if sir_result.get("saturation_detected") or sir_result.get("carrying_capacity"):
                return True
            # Check R0 decline
            r0 = sir_result.get("r0") or sir_result.get("R0")
            if r0 is not None:
                r0_val = float(np.asarray(r0).flat[0]) if hasattr(r0, '__iter__') and not isinstance(r0, str) else float(r0)
                if r0_val < 1.0:
                    return True
        return False

    def _has_high_residual_autocorr(self, results: Dict[str, dict]) -> bool:
        """Check if any engine reports high residual autocorrelation."""
        for engine, result in results.items():
            if result is None:
                continue
            residual_autocorr = result.get("residual_autocorr")
            if residual_autocorr is not None:
                val = float(np.asarray(residual_autocorr).flat[0]) if hasattr(residual_autocorr, '__iter__') and not isinstance(residual_autocorr, str) else float(residual_autocorr)
                if val > 0.3:
                    return True
        return False

    def _has_spectral_peaks(self, results: Dict[str, dict]) -> bool:
        """Check if wavelets detected significant spectral peaks."""
        wavelets = results.get("wavelets")
        if wavelets:
            if wavelets.get("significant_peaks") or wavelets.get("dominant_frequency"):
                return True
            power = wavelets.get("max_power") or wavelets.get("peak_power")
            if power is not None:
                power_val = float(np.asarray(power).flat[0]) if hasattr(power, '__iter__') and not isinstance(power, str) else float(power)
                if power_val > 0.5:
                    return True
        return False

    def _has_cross_correlation_structure(self, results: Dict[str, dict]) -> bool:
        """Check for significant cross-correlation structure."""
        corr_result = results.get("correlation")
        if corr_result:
            max_lag_corr = corr_result.get("max_lag_correlation")
            if max_lag_corr is not None:
                corr_val = float(np.asarray(max_lag_corr).flat[0]) if hasattr(max_lag_corr, '__iter__') and not isinstance(max_lag_corr, str) else float(max_lag_corr)
                if abs(corr_val) > 0.5:
                    return True
        granger = results.get("granger")
        if granger:
            n_links = granger.get("n_causal_links", 0)
            n_links_val = int(np.asarray(n_links).flat[0]) if hasattr(n_links, '__iter__') and not isinstance(n_links, str) else int(n_links)
            if granger.get("significant_edges") or n_links_val > 0:
                return True
        return False

    def _has_volatility_clustering(self, results: Dict[str, dict]) -> bool:
        """Check for volatility clustering detection."""
        for engine, result in results.items():
            if result is None:
                continue
            if result.get("volatility_clustering") or result.get("garch_effect"):
                return True
            arch_lm = result.get("arch_lm_pvalue")
            if arch_lm is not None:
                arch_val = float(np.asarray(arch_lm).flat[0]) if hasattr(arch_lm, '__iter__') and not isinstance(arch_lm, str) else float(arch_lm)
                if arch_val < 0.05:
                    return True
        return False

    def _has_state_persistence(self, results: Dict[str, dict]) -> bool:
        """Check if HMM found persistent states."""
        hmm = results.get("hmm")
        if hmm is None:
            return True  # Can't check without HMM

        dwell_time = hmm.get("mean_dwell_time") or hmm.get("avg_state_dwell_time")
        if dwell_time is not None:
            dwell_val = float(np.asarray(dwell_time).flat[0]) if hasattr(dwell_time, '__iter__') and not isinstance(dwell_time, str) else float(dwell_time)
            if dwell_val >= 5:
                return True

        persistence = hmm.get("state_persistence")
        if persistence is not None:
            pers_val = float(np.asarray(persistence).flat[0]) if hasattr(persistence, '__iter__') and not isinstance(persistence, str) else float(persistence)
            if pers_val > 0.8:
                return True

        return False

    def _has_fat_tails(self, results: Dict[str, dict]) -> bool:
        """Check for fat-tailed distribution."""
        for engine, result in results.items():
            if result is None:
                continue
            kurtosis = result.get("kurtosis") or result.get("excess_kurtosis")
            if kurtosis is not None:
                kurt_val = float(np.asarray(kurtosis).flat[0]) if hasattr(kurtosis, '__iter__') and not isinstance(kurtosis, str) else float(kurtosis)
                if kurt_val > 3:
                    return True
            tail_index = result.get("tail_index")
            if tail_index is not None:
                tail_val = float(np.asarray(tail_index).flat[0]) if hasattr(tail_index, '__iter__') and not isinstance(tail_index, str) else float(tail_index)
                if tail_val < 4:
                    return True
        return False

    def _compute_residual_autocorr(
        self,
        results: Dict[str, dict],
        data: np.ndarray,
    ) -> float:
        """Compute autocorrelation of combined residuals."""
        # Simplified: compute autocorrelation of original data
        if data.ndim > 1:
            data = data[:, 0]

        clean = data[~np.isnan(data)]
        if len(clean) < 10:
            return 0.0

        # Lag-1 autocorrelation
        mean = np.mean(clean)
        var = np.var(clean, ddof=0)
        if var == 0:
            return 0.0

        autocov = np.mean((clean[:-1] - mean) * (clean[1:] - mean))
        return abs(autocov / var)

    def _has_unexplained_periodicity(
        self,
        results: Dict[str, dict],
        data: Optional[np.ndarray],
    ) -> bool:
        """Check for unexplained periodic structure."""
        if data is None:
            return False

        if data.ndim > 1:
            data = data[:, 0]

        clean = data[~np.isnan(data)]
        if len(clean) < 20:
            return False

        # Simple FFT check
        fft = np.fft.fft(clean - np.mean(clean))
        power = np.abs(fft[:len(fft)//2])**2
        power = power / np.sum(power)

        # Check if any frequency dominates
        max_power = np.max(power[1:])  # Exclude DC
        return max_power > 0.3

    def _has_unexplained_nonlinearity(
        self,
        results: Dict[str, dict],
        data: Optional[np.ndarray],
    ) -> bool:
        """Check for unexplained nonlinear structure."""
        # Simplified check - could use BDS test in full implementation
        return False

    def _get_n_states(self, hmm_result: dict) -> int:
        """Extract number of states from HMM result."""
        n = hmm_result.get("n_states") or hmm_result.get("n_components")
        if n is not None:
            return int(n)
        states = hmm_result.get("states")
        if states is not None:
            return len(np.unique(states))
        return 1

    def _get_entropy_value(self, entropy_result: dict) -> Optional[float]:
        """Extract entropy value."""
        for key in ["entropy", "permutation_entropy", "sample_entropy"]:
            if key in entropy_result:
                return float(entropy_result[key])
        return None

    def _get_pca_variance(self, pca_result: dict) -> Optional[float]:
        """Extract PCA explained variance."""
        for key in ["explained_variance_ratio", "explained_variance", "pc1_variance"]:
            if key in pca_result:
                val = pca_result[key]
                if isinstance(val, (list, np.ndarray)):
                    return float(val[0]) if len(val) > 0 else None
                return float(val)
        return None

    def _get_mean_correlation(self, corr_result: dict) -> Optional[float]:
        """Extract mean correlation."""
        for key in ["mean_correlation", "avg_correlation", "mean_corr"]:
            if key in corr_result:
                return float(corr_result[key])
        return None

    def _has_granger_causality(self, granger_result: dict) -> bool:
        """Check if Granger causality was found."""
        if granger_result.get("significant_edges"):
            return True
        n_links = granger_result.get("n_causal_links", 0)
        return n_links > 0

    # =========================================================================
    # Generate Recommendations
    # =========================================================================

    def _generate_recommendations(
        self,
        missing_features: List[str],
        residual_issues: List[str],
        anomalies: List[str],
        geometry: GeometryProfile,
    ) -> List[str]:
        """Generate escalation recommendations based on blind spots."""
        recommendations = []

        # Missing feature recommendations
        for missing in missing_features:
            if "cumulative_peak" in missing or "saturation" in missing:
                recommendations.append(ESCALATION_MAP["missing_latent_structure"])
            if "spectral_peaks" in missing:
                recommendations.append(ESCALATION_MAP["unexplained_periodicity"])
            if "fat_tails" in missing:
                recommendations.append(ESCALATION_MAP["unexplained_fat_tails"])
            if "state_persistence" in missing:
                recommendations.append(ESCALATION_MAP["missing_state_structure"])

        # Residual issue recommendations
        for issue in residual_issues:
            if "autocorrelation" in issue.lower():
                recommendations.append(ESCALATION_MAP["high_residual_autocorr"])
            if "periodicity" in issue.lower():
                recommendations.append(ESCALATION_MAP["unexplained_periodicity"])
            if "nonlinear" in issue.lower():
                recommendations.append(ESCALATION_MAP["unexplained_nonlinearity"])

        # Anomaly recommendations
        for anomaly in anomalies:
            if "disagreement" in anomaly.lower() or "artifact" in anomaly.lower():
                if ESCALATION_MAP["engine_disagreement"] not in recommendations:
                    recommendations.append(ESCALATION_MAP["engine_disagreement"])

        # Deduplicate while preserving order
        seen = set()
        unique_recommendations = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique_recommendations.append(r)

        return unique_recommendations

    # =========================================================================
    # Main Interface
    # =========================================================================

    def run(self, inputs: Dict[str, Any]) -> BlindSpotResult:
        """
        Detect blind spots in the analysis.

        Args:
            inputs: Dict with:
                - geometry_diagnostic_result: GeometryProfile
                - engine_results: dict[str, dict]
                - engine_stability_result: StabilityResult
                - cleaned_data: Optional[np.ndarray]

        Returns:
            BlindSpotResult with blind spots, escalation recommendations, and sensitivity_report
        """
        geometry = inputs.get("geometry_diagnostic_result")
        engine_results = inputs.get("engine_results", {})
        stability = inputs.get("engine_stability_result")
        # Use explicit None check to avoid numpy array truthiness issues
        original_data = inputs.get("cleaned_data")
        if original_data is None:
            original_data = inputs.get("original_data")

        if geometry is None:
            self.log("skip", "No geometry profile provided")
            return BlindSpotResult(
                blind_spots=["Unable to check blind spots: no geometry profile"],
                escalation_recommendations=[],
                sensitivity_report={},
            )

        # Run all blind spot checks
        missing_features = self.missing_expected_features(geometry, engine_results)
        residual_issues = self.unexplained_residuals(engine_results, original_data)
        anomalies = self.cross_engine_anomalies(engine_results, stability)

        # Combine all blind spots
        blind_spots = []
        if missing_features:
            blind_spots.append("MISSING EXPECTED FEATURES:")
            blind_spots.extend([f"  - {m}" for m in missing_features])
        if residual_issues:
            blind_spots.append("UNEXPLAINED RESIDUALS:")
            blind_spots.extend([f"  - {r}" for r in residual_issues])
        if anomalies:
            blind_spots.append("CROSS-ENGINE ANOMALIES:")
            blind_spots.extend([f"  - {a}" for a in anomalies])

        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_features, residual_issues, anomalies, geometry
        )

        # Compute sensitivity report (Spec H)
        sensitivity_report = {}
        if stability is not None and hasattr(stability, 'contributions'):
            contributions = stability.contributions
            if contributions:
                sensitivity_report = sensitivity_check(contributions)
                self._last_sensitivity = sensitivity_report
        else:
            self._last_sensitivity = {}

        # Store for explain()
        self._missing_features = missing_features
        self._residual_issues = residual_issues
        self._anomalies = anomalies
        self._last_blind_spots = blind_spots
        self._last_recommendations = recommendations

        # Store in registry
        self._registry.set(self.name, "missing_features", missing_features)
        self._registry.set(self.name, "residual_issues", residual_issues)
        self._registry.set(self.name, "anomalies", anomalies)
        self._registry.set(self.name, "n_blind_spots", len(blind_spots))
        self._registry.set(self.name, "recommendations", recommendations)
        self._registry.set(self.name, "sensitivity_report", sensitivity_report)

        # Log decision
        n_total = len(missing_features) + len(residual_issues) + len(anomalies)
        sensitivity_stable = sensitivity_report.get("stable", True)
        self.log(
            decision=f"found {n_total} blind spots, sensitivity={'stable' if sensitivity_stable else 'vulnerable'}",
            reason=f"missing={len(missing_features)}, residuals={len(residual_issues)}, anomalies={len(anomalies)}",
            metrics={
                "missing_features": missing_features,
                "residual_issues": residual_issues,
                "anomalies": anomalies,
                "sensitivity_stable": sensitivity_stable,
                "vulnerable_to": sensitivity_report.get("vulnerable_to", []),
            },
        )

        result = BlindSpotResult(
            blind_spots=blind_spots,
            escalation_recommendations=recommendations,
            sensitivity_report=sensitivity_report,
        )
        self._last_result = result
        return result

    def explain(self) -> str:
        """Return human-readable blind spot analysis summary."""
        if self._last_result is None:
            return "No blind spot analysis has been run yet."

        lines = [
            "Blind Spot Detection Results",
            "=" * 50,
        ]

        if not self._last_blind_spots:
            lines.append("")
            lines.append("No blind spots detected. Analysis appears comprehensive.")
        else:
            if self._missing_features:
                lines.append("")
                lines.append("MISSING EXPECTED FEATURES:")
                for m in self._missing_features:
                    lines.append(f"  - {m}")

            if self._residual_issues:
                lines.append("")
                lines.append("UNEXPLAINED RESIDUALS:")
                for r in self._residual_issues:
                    lines.append(f"  - {r}")

            if self._anomalies:
                lines.append("")
                lines.append("CROSS-ENGINE ANOMALIES:")
                for a in self._anomalies:
                    lines.append(f"  - {a}")

        if self._last_recommendations:
            lines.append("")
            lines.append("ESCALATION RECOMMENDATIONS:")
            for i, rec in enumerate(self._last_recommendations, 1):
                lines.append(f"  {i}. {rec}")

        # Sensitivity report (Spec H)
        if self._last_sensitivity:
            lines.append("")
            lines.append("SENSITIVITY ANALYSIS (Spec H):")
            stable = self._last_sensitivity.get("stable", True)
            lines.append(f"  Consensus Stable: {'YES' if stable else 'NO'}")
            max_change = self._last_sensitivity.get("max_change", 0)
            lines.append(f"  Max Weight Shift: {max_change:.1%}")

            vulnerable = self._last_sensitivity.get("vulnerable_to", [])
            if vulnerable:
                lines.append(f"  Critical Dependencies: {', '.join(vulnerable)}")
                lines.append("  (Removing these engines changes consensus by >20%)")

            # Show per-engine sensitivity
            for key, val in self._last_sensitivity.items():
                if isinstance(val, dict) and "weight" in val:
                    engine = key
                    weight = val.get("normalized_weight", 0)
                    change = val.get("consensus_change_if_removed", 0)
                    critical = val.get("is_critical", False)
                    marker = " [CRITICAL]" if critical else ""
                    lines.append(f"    {engine}: weight={weight:.1%}, change_if_removed={change:.1%}{marker}")

        return "\n".join(lines)
