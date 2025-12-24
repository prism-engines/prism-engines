"""
PRISM Routing Agents - Spec Section D and E Implementation

Agent 2: GeometryDiagnosticAgent - Thin wrapper around GeometrySignatureAgent
Agent 3: EngineRoutingAgent - Decides which engines run based on geometry profile

These agents bridge geometry detection to engine selection, ensuring that
engines only run when the data's geometry supports them.

Implements:
    Spec D: Gate checks (domain fit, minimum windows, geometry alignment)
    Spec E: Engine cards (MIN_N requirements, verdict status)
    Spec B1: Null calibration requirement flags
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

# Import geometry agent
from prism.agents.agent_geometry_signature import (
    GeometrySignatureAgent,
    GeometryProfile,
    GeometryClass,
)

# Stub imports for engine gates - TODO: Fix proper path
class EngineAuditor:
    """Stub - needs proper implementation."""
    pass

ENGINE_GATES = {}

class EngineStatus:
    """Stub - needs proper implementation."""
    ENABLED = "enabled"
    DISABLED = "disabled"

class Domain:
    """Stub - needs proper implementation."""
    ALL = "all"
    ECONOMIC = "economic"
    FINANCE = "finance"
    CLIMATE = "climate"
    EPIDEMIOLOGY = "epidemiology"
    PHYSICAL = "physical"
    SEISMOLOGY = "seismology"

from .agent_foundation import (
    BaseAgent,
    RoutingResult,
    DiagnosticRegistry,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Spec Section E: ENGINE_MIN_WINDOWS (MIN_N requirements)
# =============================================================================

ENGINE_MIN_WINDOWS = {
    # Core engines
    "pca": 60,              # Or 5 * n_features
    "correlation": 60,      # 21 allowed with warnings
    "entropy": 100,         # 5 * m! (embedding dimension factorial)
    "wavelets": 128,        # Power of 2 preferred

    # Conditional engines
    "granger": 120,         # Needs sufficient lag structure
    "mutual_information": 200,  # MI estimation needs samples
    "hmm": 200,             # State estimation needs data
    "dmd": 100,             # SVD stability
    "copula": 250,          # Tail estimation
    "rolling_beta": 60,     # Basic regression

    # Diagnostic only
    "hurst": 500,           # EXCLUDE from consensus (short windows unreliable)
    "cointegration": 500,   # EXCLUDE from short windows

    # Domain-specific
    "sir": 84,              # ~12 weeks minimum

    # Excluded engines (set to infinity to enforce exclusion)
    "dtw": float('inf'),
    "rqa": float('inf'),
    "lyapunov": float('inf'),
    "fft": float('inf'),
}


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
# Geometry Alignment Thresholds (Spec D)
# =============================================================================

GEOMETRY_ALIGNMENT = {
    # engine -> (required_geometry, min_score) or list of (geom, score) for OR
    "sir": ("latent_flow", 0.6),
    "granger": [("oscillator", 0.5), ("latent_flow", 0.5)],  # OR condition
    "hmm": ("reflexive", 0.4),
    "dmd": ("reflexive", 0.4),
    "copula": ("reflexive", 0.4),
    # Core engines - no geometry requirement
    "pca": None,
    "correlation": None,
    "entropy": None,
    "wavelets": None,
    # Other conditional engines
    "mutual_information": None,
    "rolling_beta": None,
    "hurst": None,
    "cointegration": None,
}


# =============================================================================
# Spec D1: Domain Fit
# =============================================================================

DOMAIN_FIT = {
    # engine -> {domain: fit_score}
    "pca": {Domain.ALL: 1.0},
    "correlation": {Domain.ALL: 1.0},
    "entropy": {Domain.ALL: 1.0},
    "wavelets": {Domain.ALL: 1.0},
    "hmm": {Domain.ALL: 0.9},
    "dmd": {Domain.ALL: 0.8},
    "mutual_information": {Domain.ALL: 0.8},
    "granger": {
        Domain.CLIMATE: 1.0,
        Domain.EPIDEMIOLOGY: 1.0,
        Domain.FINANCE: 0.0,  # Not valid for finance
    },
    "copula": {
        Domain.FINANCE: 1.0,
        Domain.CLIMATE: 0.0,
        Domain.EPIDEMIOLOGY: 0.0,
    },
    "rolling_beta": {
        Domain.FINANCE: 1.0,
        Domain.CLIMATE: 0.0,
        Domain.EPIDEMIOLOGY: 0.0,
    },
    "sir": {
        Domain.EPIDEMIOLOGY: 1.0,
        Domain.FINANCE: 0.0,
        Domain.CLIMATE: 0.0,
    },
    "hurst": {Domain.ALL: 0.5},  # Diagnostic only
    "cointegration": {
        Domain.FINANCE: 0.5,  # Diagnostic only
        Domain.CLIMATE: 0.0,
        Domain.EPIDEMIOLOGY: 0.0,
    },
}


def compute_domain_fit(engine: str, domain: Domain) -> float:
    """
    Compute domain fit score for an engine (Spec D1).

    Returns:
        float in [0, 1]: 0 = not valid, 1 = fully appropriate
    """
    if engine not in DOMAIN_FIT:
        return 0.0

    fit_map = DOMAIN_FIT[engine]

    # Check for universal domain fit
    if Domain.ALL in fit_map:
        return fit_map[Domain.ALL]

    # Check specific domain
    return fit_map.get(domain, 0.0)


# =============================================================================
# Agent 2: GeometryDiagnosticAgent
# =============================================================================

class GeometryDiagnosticAgent(BaseAgent):
    """
    Thin wrapper around GeometrySignatureAgent.

    Analyzes time series to determine what class of dynamical system
    could have generated the data, providing geometry scores that
    inform engine routing.

    Usage:
        registry = DiagnosticRegistry()
        agent = GeometryDiagnosticAgent(registry)
        result = agent.run({"cleaned_data": series})
        print(agent.explain())
    """

    def __init__(
        self,
        registry: DiagnosticRegistry,
        config: Optional[Dict] = None,
        verbose: bool = False,
    ):
        super().__init__(registry, config)
        self._geometry_agent = GeometrySignatureAgent(verbose=verbose)
        self._last_profile: Optional[GeometryProfile] = None

    @property
    def name(self) -> str:
        return "geometry_diagnostic"

    def run(self, inputs: Dict[str, Any]) -> GeometryProfile:
        """
        Analyze time series geometry.

        Args:
            inputs: Dict with "cleaned_data" as numpy array or pandas Series

        Returns:
            GeometryProfile with scores for each geometry class
        """
        data = inputs.get("cleaned_data")
        if data is None:
            data = inputs.get("raw_data")

        if data is None:
            self.log("skip", "No data provided")
            # Return high-noise profile as fallback
            profile = GeometryProfile(noise_score=0.9)
            self._last_profile = profile
            self._last_result = profile
            return profile

        # Handle dict of series - use first series or combine
        if isinstance(data, dict):
            # Use first available series
            for key, series in data.items():
                data = series
                break

        # Run geometry analysis
        profile = self._geometry_agent.analyze(data)
        self._last_profile = profile
        self._last_result = profile

        # Store in registry
        self._registry.set(self.name, "latent_flow_score", profile.latent_flow_score)
        self._registry.set(self.name, "oscillator_score", profile.oscillator_score)
        self._registry.set(self.name, "reflexive_score", profile.reflexive_score)
        self._registry.set(self.name, "noise_score", profile.noise_score)
        self._registry.set(self.name, "dominant_geometry", profile.dominant_geometry.value)
        self._registry.set(self.name, "confidence", profile.confidence)
        self._registry.set(self.name, "is_hybrid", profile.is_hybrid)

        # Log decision
        self.log(
            decision=f"dominant={profile.dominant_geometry.value}",
            reason=f"confidence={profile.confidence:.2f}, noise={profile.noise_score:.2f}",
            metrics={
                "latent_flow": profile.latent_flow_score,
                "oscillator": profile.oscillator_score,
                "reflexive": profile.reflexive_score,
                "noise": profile.noise_score,
            },
        )

        return profile

    def explain(self) -> str:
        """Return human-readable summary of geometry analysis."""
        if self._last_profile is None:
            return "No geometry analysis has been run yet."

        profile = self._last_profile
        lines = [
            "Geometry Diagnostic Results",
            "=" * 40,
            f"Dominant Geometry: {profile.dominant_geometry.value.upper()}",
            f"Confidence: {profile.confidence:.2f}",
            f"Hybrid: {'Yes' if profile.is_hybrid else 'No'}",
            "",
            "Geometry Scores:",
            f"  Latent Flow:  {profile.latent_flow_score:.2f} {'***' if profile.latent_flow_score >= 0.6 else ''}",
            f"  Oscillator:   {profile.oscillator_score:.2f} {'***' if profile.oscillator_score >= 0.5 else ''}",
            f"  Reflexive:    {profile.reflexive_score:.2f} {'***' if profile.reflexive_score >= 0.4 else ''}",
            f"  Noise:        {profile.noise_score:.2f} {'[HIGH]' if profile.noise_score >= 0.6 else ''}",
        ]

        # Get warnings from underlying agent
        warnings = self._geometry_agent.get_warnings(profile)
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in warnings:
                lines.append(f"  {w}")

        return "\n".join(lines)


# =============================================================================
# Agent 3: EngineRoutingAgent
# =============================================================================

class EngineRoutingAgent(BaseAgent):
    """
    Decides which engines run based on GeometryProfile.

    Implements Spec Section D (gate checks) and E (engine cards):
    - MIN_N window requirements per engine
    - Verdict checks (EXCLUDE status)
    - Geometry alignment requirements
    - Domain fit scoring
    - Null calibration flags

    Usage:
        registry = DiagnosticRegistry()
        agent = EngineRoutingAgent(registry)
        result = agent.run({
            "geometry_diagnostic_result": profile,
            "window_size": 126,
            "sample_count": 500,
            "domain": "finance",
        })
    """

    def __init__(
        self,
        registry: DiagnosticRegistry,
        config: Optional[Dict] = None,
        domain: Domain = Domain.FINANCE,
    ):
        super().__init__(registry, config)
        self._auditor = EngineAuditor()
        self._domain = domain
        self._last_eligibility: Dict[str, str] = {}
        self._last_weights: Dict[str, float] = {}
        self._last_reasons: Dict[str, str] = {}
        self._pending_checks: Dict[str, List[str]] = {}
        self._null_required: Dict[str, bool] = {}
        self._min_window_violations: List[str] = []

    @property
    def name(self) -> str:
        return "engine_routing"

    def check_engine_eligibility(
        self,
        engine: str,
        window_size: int,
        geometry_profile: Optional[GeometryProfile],
        domain: Domain,
        sample_count: Optional[int] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Check engine eligibility per Spec D and E.

        Args:
            engine: Engine name
            window_size: Current window size
            geometry_profile: GeometryProfile from Agent 2
            domain: Data domain
            sample_count: Number of samples available

        Returns:
            (status, reason): status is "allowed", "suppressed", or "downweighted"
                              reason is None if allowed, else explanation string
        """
        # 1. Hard gate: MIN_N
        min_n = ENGINE_MIN_WINDOWS.get(engine, 60)
        if window_size < min_n:
            return "suppressed", f"window_size {window_size} < min {min_n}"

        # 2. Verdict gate (from Spec E) and sample count
        gate = ENGINE_GATES.get(engine)
        if gate is not None:
            # Check sample count against gate's min_samples
            if sample_count is not None and sample_count < gate.min_samples:
                return "suppressed", f"sample_count {sample_count} < min {gate.min_samples}"
            if gate.status == EngineStatus.EXCLUDED:
                return "suppressed", "EXCLUDE verdict per Spec E"

            # Diagnostic-only engines are downweighted
            if gate.status == EngineStatus.DIAGNOSTIC:
                return "downweighted", "DIAGNOSTIC status (informative only)"

        # 3. Geometry alignment
        if geometry_profile is not None:
            alignment_req = GEOMETRY_ALIGNMENT.get(engine)
            if alignment_req is not None:
                if isinstance(alignment_req, list):
                    # OR condition - any match is enough
                    matched = False
                    for geom_type, min_score in alignment_req:
                        score = self._get_geometry_score(geometry_profile, geom_type)
                        if score >= min_score:
                            matched = True
                            break
                    if not matched:
                        return "suppressed", f"Geometry mismatch: requires one of {alignment_req}"
                else:
                    geom_type, min_score = alignment_req
                    score = self._get_geometry_score(geometry_profile, geom_type)
                    if score < min_score:
                        return "suppressed", f"Geometry mismatch: {geom_type}={score:.2f} < {min_score}"

        # 4. Domain fit (Spec D1)
        domain_fit = compute_domain_fit(engine, domain)
        if domain_fit == 0:
            return "suppressed", f"engine not valid for domain {domain.value}"

        # All gates passed
        return "allowed", None

    def run(self, inputs: Dict[str, Any]) -> RoutingResult:
        """
        Route engines based on geometry profile.

        Args:
            inputs: Dict with:
                - geometry_diagnostic_result: GeometryProfile
                - window_size: int
                - sample_count: int
                - domain: str or Domain

        Returns:
            RoutingResult with eligibility, weights, null_required, and suppression reasons
        """
        profile = inputs.get("geometry_diagnostic_result")
        window_size = inputs.get("window_size", 126)
        sample_count = inputs.get("sample_count", 500)
        domain_input = inputs.get("domain", self._domain)

        # Handle domain as either string or Domain enum
        if isinstance(domain_input, str):
            domain_map = {d.value: d for d in Domain}
            domain = domain_map.get(domain_input.lower(), Domain.FINANCE)
        else:
            domain = domain_input

        if profile is None:
            self.log("skip", "No geometry profile provided")
            return RoutingResult(
                engine_eligibility={},
                weights={},
                suppression_reasons={"all": "No geometry profile"},
            )

        eligibility: Dict[str, str] = {}
        weights: Dict[str, float] = {}
        suppression_reasons: Dict[str, str] = {}
        pending_checks: Dict[str, List[str]] = {}
        null_required: Dict[str, bool] = {}
        min_window_violations: List[str] = []

        # Process all known engines
        all_engines = set(ENGINE_GATES.keys()) | set(ENGINE_MIN_WINDOWS.keys())

        for engine_name in all_engines:
            # Check eligibility using the new method
            status, reason = self.check_engine_eligibility(
                engine_name, window_size, profile, domain, sample_count
            )

            eligibility[engine_name] = status
            if reason:
                suppression_reasons[engine_name] = reason
                # Track MIN_N violations
                if "MIN_N" in reason:
                    min_window_violations.append(engine_name)

            # Set null_required flag (Spec B1)
            null_required[engine_name] = engine_name in REQUIRES_NULL

            if status == "suppressed":
                weights[engine_name] = 0.0
                continue

            # Compute pre-consensus weight
            weight = 1.0

            # Get gate info for additional processing
            gate = ENGINE_GATES.get(engine_name)

            # Apply domain fit multiplier
            domain_fit = compute_domain_fit(engine_name, domain)
            weight *= domain_fit

            # Apply geometry alignment score if applicable
            alignment_req = GEOMETRY_ALIGNMENT.get(engine_name)
            if alignment_req is not None and profile is not None:
                alignment_score = self._compute_alignment_score(profile, alignment_req)
                weight *= alignment_score

            # Apply noise penalty
            if profile is not None:
                weight *= (1 - profile.noise_score * 0.5)

            # Conditional engines start at 0.8 max
            if gate and gate.status == EngineStatus.CONDITIONAL:
                weight *= 0.8

            # Diagnostic engines get heavily downweighted
            if gate and gate.status == EngineStatus.DIAGNOSTIC:
                weight *= 0.3

            # Update status based on final weight
            if weight < 0.3:
                eligibility[engine_name] = "downweighted"

            weights[engine_name] = weight

            # Note pending checks for later agents
            checks = []
            if null_required[engine_name]:
                checks.append("null_calibration")
            if gate:
                if gate.requires_null_test:
                    checks.append("null_test")
                if gate.requires_stability_check:
                    checks.append("stability_check")
                if gate.requires_benchmark_validation:
                    checks.append("benchmark_validation")
            if checks:
                pending_checks[engine_name] = checks

        # Store results
        self._last_eligibility = eligibility
        self._last_weights = weights
        self._last_reasons = suppression_reasons
        self._pending_checks = pending_checks
        self._null_required = null_required
        self._min_window_violations = min_window_violations

        # Store in registry
        self._registry.set(self.name, "eligibility", eligibility)
        self._registry.set(self.name, "weights", weights)
        self._registry.set(self.name, "suppression_reasons", suppression_reasons)
        self._registry.set(self.name, "pending_checks", pending_checks)
        self._registry.set(self.name, "null_required", null_required)
        self._registry.set(self.name, "min_window_violations", min_window_violations)

        # Count by status
        allowed_count = sum(1 for s in eligibility.values() if s == "allowed")
        downweighted_count = sum(1 for s in eligibility.values() if s == "downweighted")
        suppressed_count = sum(1 for s in eligibility.values() if s == "suppressed")

        self.log(
            decision=f"allowed={allowed_count}, downweighted={downweighted_count}, suppressed={suppressed_count}",
            reason=f"window={window_size}, samples={sample_count}, domain={domain.value}",
            metrics={
                "allowed_engines": [e for e, s in eligibility.items() if s == "allowed"],
                "downweighted_engines": [e for e, s in eligibility.items() if s == "downweighted"],
                "suppressed_engines": [e for e, s in eligibility.items() if s == "suppressed"],
                "null_required_engines": [e for e, req in null_required.items() if req],
                "min_window_violations": min_window_violations,
            },
        )

        result = RoutingResult(
            engine_eligibility=eligibility,
            weights=weights,
            suppression_reasons=suppression_reasons,
            min_window_violations=min_window_violations,
        )
        self._last_result = result
        return result

    def _get_geometry_score(self, profile: GeometryProfile, geom_type: str) -> float:
        """Get specific geometry score from profile."""
        if geom_type == "latent_flow":
            return profile.latent_flow_score
        elif geom_type == "oscillator":
            return profile.oscillator_score
        elif geom_type == "reflexive":
            return profile.reflexive_score
        elif geom_type == "noise":
            return profile.noise_score
        return 0.0

    def _compute_alignment_score(
        self,
        profile: GeometryProfile,
        alignment_req,
    ) -> float:
        """Compute geometry alignment score for weight computation."""
        if isinstance(alignment_req, list):
            # OR condition - use best score
            best_score = 0.0
            for geom_type, min_score in alignment_req:
                score = self._get_geometry_score(profile, geom_type)
                if score >= min_score:
                    best_score = max(best_score, score)
            return best_score if best_score > 0 else 0.5
        else:
            geom_type, min_score = alignment_req
            return self._get_geometry_score(profile, geom_type)

    def explain(self) -> str:
        """Return human-readable routing decision summary."""
        if not self._last_eligibility:
            return "No routing analysis has been run yet."

        lines = [
            "Engine Routing Results (Spec D/E)",
            "=" * 50,
        ]

        # Group by status
        allowed = [(e, self._last_weights.get(e, 0)) for e, s in self._last_eligibility.items() if s == "allowed"]
        downweighted = [(e, self._last_weights.get(e, 0)) for e, s in self._last_eligibility.items() if s == "downweighted"]
        suppressed = [(e, self._last_reasons.get(e, "")) for e, s in self._last_eligibility.items() if s == "suppressed"]

        if allowed:
            lines.append("")
            lines.append("ALLOWED:")
            for engine, weight in sorted(allowed, key=lambda x: -x[1]):
                checks = self._pending_checks.get(engine, [])
                null_flag = " [NULL_REQ]" if self._null_required.get(engine, False) else ""
                check_str = f" [requires: {', '.join(checks)}]" if checks else ""
                lines.append(f"  {engine}: weight={weight:.2f}{null_flag}{check_str}")

        if downweighted:
            lines.append("")
            lines.append("DOWNWEIGHTED:")
            for engine, weight in sorted(downweighted, key=lambda x: -x[1]):
                reason = self._last_reasons.get(engine, "low weight")
                lines.append(f"  {engine}: weight={weight:.2f} ({reason})")

        if suppressed:
            lines.append("")
            lines.append("SUPPRESSED:")
            for engine, reason in suppressed[:15]:  # Limit output
                lines.append(f"  {engine}: {reason[:70]}")
            if len(suppressed) > 15:
                lines.append(f"  ... and {len(suppressed) - 15} more")

        if self._min_window_violations:
            lines.append("")
            lines.append(f"MIN_N VIOLATIONS: {', '.join(self._min_window_violations)}")

        return "\n".join(lines)

    def get_pending_checks(self) -> Dict[str, List[str]]:
        """Get dict of engines requiring additional validation checks."""
        return self._pending_checks.copy()

    def get_null_required(self) -> Dict[str, bool]:
        """Get dict of engines requiring null calibration (Spec B1)."""
        return self._null_required.copy()

    def get_min_window_violations(self) -> List[str]:
        """Get list of engines that failed MIN_N gate."""
        return self._min_window_violations.copy()
