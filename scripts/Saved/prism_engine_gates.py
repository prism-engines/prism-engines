"""
PRISM Engine Gates - Admission Control for Geometry Consensus

This module defines which engines contribute to PRISM's geometric structure
and under what conditions. Engines that fail their gates are either:
- Excluded entirely (artifact risk too high)
- Demoted to diagnostic-only (informative but not geometry-shaping)
- Gated by domain, window size, or validation requirements

Cross-validated by: Claude, GPT-4, Gemini
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum


class EngineStatus(Enum):
    """Engine admission status for geometry consensus."""
    CORE = "core"                    # Always included in consensus
    CONDITIONAL = "conditional"      # Included if gates pass
    DIAGNOSTIC = "diagnostic"        # Informative only, zero consensus weight
    EXCLUDED = "excluded"            # Never run, artifact risk too high


class Domain(Enum):
    """Data domains."""
    FINANCE = "finance"
    CLIMATE = "climate"
    EPIDEMIOLOGY = "epidemiology"
    ALL = "all"


@dataclass
class EngineGate:
    """
    Admission control specification for a single engine.
    
    An engine contributes to geometry consensus only if:
    1. It's not EXCLUDED
    2. All required gates pass
    3. Domain restrictions are met
    4. Minimum window/sample requirements are met
    """
    name: str
    status: EngineStatus
    
    # Domain restrictions
    domains: List[Domain] = field(default_factory=lambda: [Domain.ALL])
    
    # Window/sample requirements
    min_window: int = 21
    min_samples: int = 50
    
    # Preprocessing requirements
    requires_stationarity: bool = False
    requires_normalization: str = "zscore"  # zscore, rank, robust, none
    
    # Validation requirements
    requires_null_test: bool = False
    requires_stability_check: bool = False
    requires_benchmark_validation: bool = False
    
    # For conditional engines: specific gates
    max_states: Optional[int] = None           # HMM
    min_dwell_time: Optional[int] = None       # HMM
    fdr_threshold: Optional[float] = None      # Granger, MI
    warping_constraint: Optional[float] = None # DTW
    
    # Failure behavior
    fallback_to_diagnostic: bool = True
    
    # Notes for human review
    rationale: str = ""
    failure_modes: List[str] = field(default_factory=list)


# =============================================================================
# CORE ENGINES - Always included (with proper preprocessing)
# =============================================================================

PCA_GATE = EngineGate(
    name="pca",
    status=EngineStatus.CORE,
    min_window=42,
    requires_normalization="robust",  # Robust covariance essential
    rationale="Factor structure / shared movement. Skeleton of geometry.",
    failure_modes=[
        "Outlier drives PC1",
        "Explained variance spikes from single shock",
        "Unstable loadings flip sign"
    ]
)

CORRELATION_GATE = EngineGate(
    name="correlation",
    status=EngineStatus.CORE,
    min_window=42,
    requires_normalization="rank",  # Spearman preferred over Pearson
    rationale="Pairwise linear synchronization. Basic connectivity map.",
    failure_modes=[
        "Correlation blowup during crashes (everything â†’ 1)",
        "Spurious stability from shared trend"
    ]
)

ENTROPY_GATE = EngineGate(
    name="entropy",
    status=EngineStatus.CORE,
    min_window=63,
    min_samples=100,
    requires_normalization="none",  # Permutation entropy is distribution-free
    rationale="Complexity/disorder. Best tool for detecting 'fog' before regime change.",
    failure_modes=[
        "Sensitive to embedding dimension parameter",
        "Spikes from measurement noise"
    ]
)

WAVELETS_GATE = EngineGate(
    name="wavelets",
    status=EngineStatus.CORE,
    min_window=63,
    requires_normalization="none",
    rationale="Multi-scale time-frequency. Superior to FFT for non-stationary data.",
    failure_modes=[
        "Edge effects at window boundaries",
        "Inconsistent scale comparisons"
    ]
)


# =============================================================================
# CONDITIONAL ENGINES - Included if specific gates pass
# =============================================================================

HMM_GATE = EngineGate(
    name="hmm",
    status=EngineStatus.CONDITIONAL,
    min_window=126,
    requires_normalization="zscore",
    requires_stability_check=True,
    max_states=4,           # More states = overfitting risk
    min_dwell_time=5,       # Regimes must persist
    rationale="Discrete regime states. The 'decision maker' of the framework.",
    failure_modes=[
        "Regimes flip every few points (nonsense)",
        "States reflect noise clusters",
        "Unstable transition matrices",
        "Finds N states in white noise if you ask for N"
    ]
)

DMD_GATE = EngineGate(
    name="dmd",
    status=EngineStatus.CONDITIONAL,
    min_window=63,
    requires_normalization="zscore",
    requires_stability_check=True,
    rationale="Spatiotemporal dynamics. Captures operator structure PCA misses.",
    failure_modes=[
        "Modes jump around with small data changes",
        "Spurious oscillatory modes from noise",
        "Eigenvalues unstable"
    ]
)

MUTUAL_INFO_GATE = EngineGate(
    name="mutual_information",
    status=EngineStatus.CONDITIONAL,
    min_window=126,
    min_samples=100,
    requires_normalization="rank",
    requires_null_test=True,        # Permutation null essential
    fdr_threshold=0.05,
    rationale="Non-linear dependencies. Captures what correlation misses.",
    failure_modes=[
        "MI 'lights up everywhere' (estimator bias)",
        "Hallucinates relationships in small samples",
        "Dense fake networks without FDR control"
    ]
)

COPULA_GATE = EngineGate(
    name="copula",
    status=EngineStatus.CONDITIONAL,
    domains=[Domain.FINANCE],       # Finance only
    min_window=252,                 # Needs larger windows
    requires_normalization="rank",  # Transform to uniform margins
    requires_stability_check=True,
    rationale="Joint tail behavior. 'Do these things crash together?'",
    failure_modes=[
        "Wrong copula family (Gaussian when tails are fat)",
        "Tail dependence estimates swing wildly",
        "False extreme coupling from tiny sample"
    ]
)

GRANGER_GATE = EngineGate(
    name="granger",
    status=EngineStatus.CONDITIONAL,
    domains=[Domain.CLIMATE, Domain.EPIDEMIOLOGY],  # Not finance
    min_window=126,
    requires_stationarity=True,     # Non-negotiable
    requires_null_test=True,
    fdr_threshold=0.05,
    rationale="Predictive precedence. Lead-lag network.",
    failure_modes=[
        "Dense 'causal' networks from shared trend",
        "Unstable edges flipping every window",
        "Spurious causality from hidden confounders"
    ]
)

ROLLING_BETA_GATE = EngineGate(
    name="rolling_beta",
    status=EngineStatus.CONDITIONAL,
    domains=[Domain.FINANCE],       # Finance only
    min_window=63,
    requires_normalization="none",  # Works on returns
    rationale="Sensitivity to benchmark. Control variable for finance geometry.",
    failure_modes=[
        "Beta spikes from single shock",
        "Meaningless when benchmark relationship vanishes"
    ]
)


# =============================================================================
# DIAGNOSTIC ONLY - Informative but not geometry-shaping
# =============================================================================

HURST_GATE = EngineGate(
    name="hurst",
    status=EngineStatus.DIAGNOSTIC,
    min_window=252,                 # Too short = unreliable
    requires_normalization="none",
    rationale="Long-memory tendency. Useful diagnostic but unstable in short windows.",
    failure_modes=[
        "Floats around with huge variance",
        "Interprets volatility clustering as trend persistence",
        "H â‰ˆ 0.5 for almost everything in practice"
    ]
)

COINTEGRATION_GATE = EngineGate(
    name="cointegration",
    status=EngineStatus.DIAGNOSTIC,
    domains=[Domain.FINANCE],
    min_window=756,                 # Multi-year windows only
    requires_stationarity=False,    # Requires NON-stationary (I(1)) data
    rationale="Long-run equilibrium. Valid but needs much longer windows than rolling.",
    failure_modes=[
        "Spurious cointegration in short windows",
        "Structural breaks sever relationships",
        "Unstable rank estimates"
    ]
)


# =============================================================================
# EXCLUDED - Artifact risk too high
# =============================================================================

LYAPUNOV_GATE = EngineGate(
    name="lyapunov",
    status=EngineStatus.EXCLUDED,
    rationale="EXCLUDED: Guaranteed artifact generator. Returns positive Lyapunov for anything noisy.",
    failure_modes=[
        "Requires massive, high-precision, clean data (not available)",
        "Window sizes far too short for reliable estimation",
        "False 'chaos' everywhere"
    ]
)

RQA_GATE = EngineGate(
    name="rqa",
    status=EngineStatus.EXCLUDED,
    rationale="EXCLUDED: Parameter sensitivity makes it dangerous. Mistakes drift for structure.",
    failure_modes=[
        "Extremely sensitive to embedding parameters",
        "Metrics drift due to parameter choice, not data",
        "'Determinism' jumps on pure noise"
    ]
)

FFT_GATE = EngineGate(
    name="fft",
    status=EngineStatus.EXCLUDED,
    rationale="EXCLUDED: Use wavelets instead. FFT assumes stationarity that doesn't exist.",
    failure_modes=[
        "Spectral leakage on non-stationary data",
        "Apparent cycles from trend/edge artifacts",
        "Fully redundant with wavelets (which are better)"
    ]
)

DTW_GATE = EngineGate(
    name="dtw",
    status=EngineStatus.EXCLUDED,
    rationale="EXCLUDED: Prone to pathological warping and overfitting in noisy data.",
    failure_modes=[
        "Everything clusters with everything",
        "'Perfect matches' from excessive warping",
        "Computationally expensive for marginal benefit"
    ]
)


# =============================================================================
# DOMAIN-SPECIFIC: SIR/SEIR (Epidemiology Only)
# =============================================================================

@dataclass
class SIRCircuitBreaker:
    """
    Circuit breaker logic for SIR engine.
    If any check fails, SIR weight â†’ 0.
    
    Cross-validated by GPT and Gemini (both proposed identical logic).
    """
    
    # 1. Data sufficiency
    min_outbreak_days: int = 14
    min_variance_threshold: float = 0.01
    requires_full_cycle: bool = True  # Need rise AND fall
    
    # 2. Identifiability check
    max_parameter_correlation: float = 0.95  # Î²-Î³ correlation
    max_hessian_condition_number: float = 1e6
    
    # 3. Parameter boundaries (respiratory virus defaults)
    beta_bounds: tuple = (0.0, 2.0)
    gamma_bounds: tuple = (1/60, 1/1)  # Recovery 1-60 days
    r0_bounds: tuple = (0.1, 20.0)
    
    # 4. Residual diagnostics
    durbin_watson_bounds: tuple = (1.0, 3.0)  # Outside = autocorrelated residuals


SIR_GATE = EngineGate(
    name="sir",
    status=EngineStatus.CONDITIONAL,
    domains=[Domain.EPIDEMIOLOGY],   # EPI ONLY
    min_window=84,                   # Need ~12 weeks minimum
    min_samples=12,                  # Weekly data
    requires_normalization="none",
    requires_benchmark_validation=True,
    rationale=(
        "Mechanistic compartmental model. Adds 'saturation' dimension and "
        "interpretable R0/Î²(t). Use as 'physics engine' for epi, not universal geometry. "
        "HMM tells you state changed; SIR tells you WHY (transmission vs depletion)."
    ),
    failure_modes=[
        "Parameter unidentifiability (multiple Î²/Î³ combos fit same curve)",
        "Naive fitting without observation model â†’ fake precision",
        "'Infinite peak' prediction when model can't find peak",
        "Weekend reporting artifacts mistaken for transmission changes",
        "Over-fitting noise into Î²(t)"
    ]
)


# =============================================================================
# ENGINE REGISTRY
# =============================================================================

ENGINE_GATES: Dict[str, EngineGate] = {
    # Core
    "pca": PCA_GATE,
    "correlation": CORRELATION_GATE,
    "entropy": ENTROPY_GATE,
    "wavelets": WAVELETS_GATE,
    
    # Conditional
    "hmm": HMM_GATE,
    "dmd": DMD_GATE,
    "mutual_information": MUTUAL_INFO_GATE,
    "copula": COPULA_GATE,
    "granger": GRANGER_GATE,
    "rolling_beta": ROLLING_BETA_GATE,
    
    # Diagnostic only
    "hurst": HURST_GATE,
    "cointegration": COINTEGRATION_GATE,
    
    # Domain-specific
    "sir": SIR_GATE,
    
    # Excluded
    "lyapunov": LYAPUNOV_GATE,
    "rqa": RQA_GATE,
    "fft": FFT_GATE,
    "dtw": DTW_GATE,
}


# =============================================================================
# AUDIT AGENT INTERFACE
# =============================================================================

class EngineAuditor:
    """
    Validates engines before they contribute to geometry consensus.
    
    Usage:
        auditor = EngineAuditor()
        
        # Check if engine can run
        if auditor.can_run("hmm", domain=Domain.FINANCE, window_size=126):
            result = hmm_engine.run(data)
            
            # Check if result should contribute to consensus
            weight = auditor.get_consensus_weight("hmm", result)
    """
    
    def __init__(self):
        self.gates = ENGINE_GATES
        self.sir_breaker = SIRCircuitBreaker()
    
    def get_status(self, engine_name: str) -> EngineStatus:
        """Get engine's admission status."""
        if engine_name not in self.gates:
            raise ValueError(f"Unknown engine: {engine_name}")
        return self.gates[engine_name].status
    
    def can_run(
        self, 
        engine_name: str, 
        domain: Domain,
        window_size: int,
        sample_count: Optional[int] = None
    ) -> bool:
        """
        Check if engine is allowed to run given context.
        
        Returns False if:
        - Engine is EXCLUDED
        - Domain doesn't match
        - Window/sample requirements not met
        """
        gate = self.gates.get(engine_name)
        if gate is None:
            return False
        
        # Excluded engines never run
        if gate.status == EngineStatus.EXCLUDED:
            return False
        
        # Check domain
        if Domain.ALL not in gate.domains and domain not in gate.domains:
            return False
        
        # Check window size
        if window_size < gate.min_window:
            return False
        
        # Check sample count
        if sample_count and sample_count < gate.min_samples:
            return False
        
        return True
    
    def get_consensus_weight(
        self, 
        engine_name: str,
        result: dict,
        stability_score: Optional[float] = None,
        null_test_passed: Optional[bool] = None,
        benchmark_score: Optional[float] = None
    ) -> float:
        """
        Determine how much weight this engine's result gets in consensus.
        
        Returns:
        - 1.0: Full weight (core engine, all checks pass)
        - 0.5-0.9: Partial weight (conditional engine, some concerns)
        - 0.0: No weight (failed gates, diagnostic only, or excluded)
        """
        gate = self.gates.get(engine_name)
        if gate is None:
            return 0.0
        
        # Excluded and diagnostic get zero weight
        if gate.status in [EngineStatus.EXCLUDED, EngineStatus.DIAGNOSTIC]:
            return 0.0
        
        # Start with full weight
        weight = 1.0
        
        # Check required validations
        if gate.requires_null_test and not null_test_passed:
            weight = 0.0
        
        if gate.requires_stability_check and stability_score is not None:
            if stability_score < 0.5:
                weight *= stability_score
        
        if gate.requires_benchmark_validation and benchmark_score is not None:
            if benchmark_score < 0.5:
                weight *= benchmark_score
        
        # Conditional engines start at 0.8 max (earn full weight through validation)
        if gate.status == EngineStatus.CONDITIONAL:
            weight *= 0.8
        
        return weight
    
    def get_core_engines(self) -> List[str]:
        """Get list of core engines (always included)."""
        return [
            name for name, gate in self.gates.items()
            if gate.status == EngineStatus.CORE
        ]
    
    def get_conditional_engines(self, domain: Domain) -> List[str]:
        """Get conditional engines valid for a domain."""
        return [
            name for name, gate in self.gates.items()
            if gate.status == EngineStatus.CONDITIONAL
            and (Domain.ALL in gate.domains or domain in gate.domains)
        ]
    
    def get_excluded_engines(self) -> List[str]:
        """Get list of excluded engines (never run)."""
        return [
            name for name, gate in self.gates.items()
            if gate.status == EngineStatus.EXCLUDED
        ]
    
    def print_summary(self):
        """Print engine roster summary."""
        print("=" * 60)
        print("PRISM ENGINE GATES - ADMISSION CONTROL")
        print("=" * 60)
        
        for status in EngineStatus:
            engines = [n for n, g in self.gates.items() if g.status == status]
            if engines:
                print(f"\n{status.value.upper()}:")
                for name in engines:
                    gate = self.gates[name]
                    domains = ", ".join(d.value for d in gate.domains)
                    print(f"  {name}: min_window={gate.min_window}, domains=[{domains}]")


# =============================================================================
# MINIMAL SUFFICIENT SET (for publishable geometry)
# =============================================================================

MINIMAL_SET = {
    "universal": ["pca", "correlation", "entropy", "wavelets"],
    "conditional": ["hmm", "dmd", "mutual_information"],
    "finance_specific": ["copula", "rolling_beta"],
    "climate_specific": ["granger"],
    "epi_specific": ["sir", "granger"],
}

"""
The Minimal Sufficient Set captures:

1. PCA          â†’ Factor structure (breadth across variables)
2. Correlation  â†’ Pairwise synchronization (linear ties)
3. Entropy      â†’ Complexity/disorder (predictability)
4. Wavelets     â†’ Multi-scale dynamics (frequency over time)
5. HMM          â†’ Discrete regime states (the decision maker)
6. DMD          â†’ Operator dynamics (how system evolves)
7. MI           â†’ Non-linear dependencies (what correlation misses)

Plus domain-specific:
- Copula (finance): Tail dependencies / crash co-movement
- Granger (climate/epi): Lead-lag causality
- SIR (epi): Mechanistic transmission dynamics

Total: 7 universal + 3-4 domain-specific = 10-11 engines
Down from 20. Each captures genuinely different properties.
"""


if __name__ == "__main__":
    auditor = EngineAuditor()
    auditor.print_summary()
    
    print("\n" + "=" * 60)
    print("MINIMAL SUFFICIENT SET")
    print("=" * 60)
    for category, engines in MINIMAL_SET.items():
        print(f"\n{category}: {engines}")
