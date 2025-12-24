"""
PRISM Geometry Signature Agent

Determines what class of dynamical system could have generated the observed 
time series, based solely on measurable properties â€” before invoking specialized models.

The agent answers one question:
    "What kind of geometry is the data asking for?"

Core Geometry Classes:
    1. Latent Flow / Compartmental (epidemics, diffusion, adoption)
    2. Coupled Oscillator (climate, cycles, physical systems)
    3. Reflexive Stochastic (finance, social, sentiment-driven)
    4. Pure Noise / Noise-Dominated

Philosophy:
    PRISM does not assume the world's structure â€” it infers which geometries 
    the data can support, and only then deploys the appropriate models.

Cross-validated by: Claude, GPT-4, Gemini
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# GEOMETRY CLASSES
# =============================================================================

class GeometryClass(Enum):
    """Core geometry classes that data can exhibit."""
    LATENT_FLOW = "latent_flow"           # Compartmental / diffusion
    COUPLED_OSCILLATOR = "coupled_oscillator"  # Physical / cyclical
    REFLEXIVE_STOCHASTIC = "reflexive_stochastic"  # Finance / behavioral
    PURE_NOISE = "pure_noise"             # Noise-dominated


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic test."""
    name: str
    score: float  # 0-1
    metric_value: float
    description: str
    passed: bool  # Score >= threshold


@dataclass 
class GeometryProfile:
    """
    Complete geometry profile for a time series.
    
    Scores are in [0, 1]:
        >= 0.70: Strong evidence
        0.40 - 0.69: Partial / mixed
        < 0.40: Weak evidence
    """
    latent_flow_score: float = 0.0
    oscillator_score: float = 0.0
    reflexive_score: float = 0.0
    noise_score: float = 0.0
    
    # Detailed diagnostics
    latent_flow_diagnostics: List[DiagnosticResult] = field(default_factory=list)
    oscillator_diagnostics: List[DiagnosticResult] = field(default_factory=list)
    reflexive_diagnostics: List[DiagnosticResult] = field(default_factory=list)
    noise_diagnostics: List[DiagnosticResult] = field(default_factory=list)
    
    # Derived properties
    dominant_geometry: Optional[GeometryClass] = None
    is_hybrid: bool = False
    confidence: float = 0.0
    
    def __post_init__(self):
        self._compute_derived()
    
    def _compute_derived(self):
        """Compute dominant geometry and confidence."""
        scores = {
            GeometryClass.LATENT_FLOW: self.latent_flow_score,
            GeometryClass.COUPLED_OSCILLATOR: self.oscillator_score,
            GeometryClass.REFLEXIVE_STOCHASTIC: self.reflexive_score,
            GeometryClass.PURE_NOISE: self.noise_score,
        }
        
        # Find dominant
        self.dominant_geometry = max(scores, key=scores.get)
        max_score = scores[self.dominant_geometry]
        
        # Check for hybrid (multiple high scores)
        high_scores = [g for g, s in scores.items() if s >= 0.5 and g != GeometryClass.PURE_NOISE]
        self.is_hybrid = len(high_scores) > 1
        
        # Confidence based on separation from noise and clarity
        if self.noise_score >= 0.6:
            self.confidence = 0.2  # Low confidence if noise-dominated
        else:
            # Confidence = how much dominant exceeds second-best
            sorted_scores = sorted(scores.values(), reverse=True)
            separation = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
            self.confidence = min(1.0, max_score * 0.7 + separation * 0.3)


@dataclass
class EngineRecommendation:
    """Recommendation for which engines to run."""
    engine: str
    weight: float  # 0-1, contribution to consensus
    reason: str
    required_gates: List[str] = field(default_factory=list)


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def _safe_compute(func, *args, default=0.0, **kwargs):
    """Safely compute a metric, returning default on failure."""
    try:
        result = func(*args, **kwargs)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except:
        return default


# --- Latent Flow / Compartmental Diagnostics ---

def diagnose_saturation(series: np.ndarray) -> DiagnosticResult:
    """
    Does cumulative signal plateau or asymptote?
    Logistic vs exponential growth comparison.
    """
    if len(series) < 20:
        return DiagnosticResult("saturation", 0.0, 0.0, "Insufficient data", False)
    
    cumulative = np.cumsum(np.maximum(series - series.min(), 0))
    t = np.arange(len(cumulative))
    
    # Fit logistic: L / (1 + exp(-k*(t-t0)))
    def logistic(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))
    
    # Fit exponential: a * exp(b*t)
    def exponential(t, a, b):
        return a * np.exp(b * t)
    
    try:
        # Logistic fit
        p0_log = [cumulative[-1], 0.1, len(t)/2]
        popt_log, _ = curve_fit(logistic, t, cumulative, p0=p0_log, maxfev=5000)
        resid_log = cumulative - logistic(t, *popt_log)
        ss_log = np.sum(resid_log**2)
        
        # Exponential fit (on subset to avoid overflow)
        subset = min(len(t), 100)
        p0_exp = [cumulative[0] + 1, 0.01]
        popt_exp, _ = curve_fit(exponential, t[:subset], cumulative[:subset], p0=p0_exp, maxfev=5000)
        resid_exp = cumulative[:subset] - exponential(t[:subset], *popt_exp)
        ss_exp = np.sum(resid_exp**2)
        
        # Score: logistic fits better = saturation present
        if ss_exp > 0:
            ratio = ss_log / ss_exp
            score = 1.0 / (1.0 + ratio)  # Lower ratio = logistic better = higher score
        else:
            score = 0.5
        
        return DiagnosticResult(
            "saturation", 
            score, 
            ratio if ss_exp > 0 else 0,
            f"Logistic/Exp SS ratio: {ratio:.2f}" if ss_exp > 0 else "Could not compare",
            score >= 0.5
        )
    except:
        return DiagnosticResult("saturation", 0.3, 0.0, "Fitting failed", False)


def diagnose_asymmetry(series: np.ndarray) -> DiagnosticResult:
    """
    Is rise shape statistically different from decay shape?
    Irreversibility / hysteresis detection.
    """
    if len(series) < 30:
        return DiagnosticResult("asymmetry", 0.0, 0.0, "Insufficient data", False)
    
    # Find peak
    peak_idx = np.argmax(series)
    
    if peak_idx < 5 or peak_idx > len(series) - 5:
        return DiagnosticResult("asymmetry", 0.3, 0.0, "Peak at boundary", False)
    
    # Rise and fall segments
    rise = series[:peak_idx]
    fall = series[peak_idx:]
    
    # Compare slopes
    rise_slope = _safe_compute(lambda: np.polyfit(np.arange(len(rise)), rise, 1)[0])
    fall_slope = _safe_compute(lambda: np.polyfit(np.arange(len(fall)), fall, 1)[0])
    
    # Asymmetry score: different slopes = asymmetric
    if rise_slope != 0:
        slope_ratio = abs(fall_slope / rise_slope)
        # Score high if ratio far from 1
        asymmetry = abs(1 - slope_ratio)
        score = min(1.0, asymmetry)
    else:
        score = 0.3
    
    return DiagnosticResult(
        "asymmetry",
        score,
        slope_ratio if rise_slope != 0 else 0,
        f"Rise/fall slope ratio: {slope_ratio:.2f}" if rise_slope != 0 else "Zero rise slope",
        score >= 0.4
    )


def diagnose_bounded_accumulation(series: np.ndarray) -> DiagnosticResult:
    """
    Does cumulative activity stabilize rather than drift indefinitely?
    """
    if len(series) < 30:
        return DiagnosticResult("bounded_accumulation", 0.0, 0.0, "Insufficient data", False)
    
    cumulative = np.cumsum(series - np.mean(series))
    
    # Check if variance of cumulative stabilizes
    first_half_var = np.var(cumulative[:len(cumulative)//2])
    second_half_var = np.var(cumulative[len(cumulative)//2:])
    
    if first_half_var > 0:
        var_ratio = second_half_var / first_half_var
        # Bounded if second half variance doesn't explode
        score = 1.0 / (1.0 + max(0, var_ratio - 1))
    else:
        score = 0.5
    
    return DiagnosticResult(
        "bounded_accumulation",
        score,
        var_ratio if first_half_var > 0 else 0,
        f"Variance ratio (2nd/1st half): {var_ratio:.2f}" if first_half_var > 0 else "Zero variance",
        score >= 0.5
    )


# --- Coupled Oscillator Diagnostics ---

def diagnose_periodicity(series: np.ndarray) -> DiagnosticResult:
    """
    Stable dominant frequencies across the series?
    """
    if len(series) < 50:
        return DiagnosticResult("periodicity", 0.0, 0.0, "Insufficient data", False)
    
    # FFT
    fft = np.fft.fft(series - np.mean(series))
    power = np.abs(fft[:len(fft)//2])**2
    
    # Concentration of power in top frequencies
    sorted_power = np.sort(power)[::-1]
    total_power = np.sum(power)
    
    if total_power > 0:
        top3_concentration = np.sum(sorted_power[:3]) / total_power
        score = min(1.0, top3_concentration * 2)  # Scale to 0-1
    else:
        score = 0.0
    
    return DiagnosticResult(
        "periodicity",
        score,
        top3_concentration if total_power > 0 else 0,
        f"Top-3 frequency concentration: {top3_concentration:.2f}" if total_power > 0 else "No power",
        score >= 0.4
    )


def diagnose_long_memory(series: np.ndarray) -> DiagnosticResult:
    """
    Strong autocorrelation without variance blow-up?
    """
    if len(series) < 50:
        return DiagnosticResult("long_memory", 0.0, 0.0, "Insufficient data", False)
    
    # Autocorrelation at various lags
    n = len(series)
    max_lag = min(50, n // 4)
    
    acf_values = []
    for lag in range(1, max_lag + 1):
        acf = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        if not np.isnan(acf):
            acf_values.append(abs(acf))
    
    if not acf_values:
        return DiagnosticResult("long_memory", 0.0, 0.0, "ACF computation failed", False)
    
    # Long memory: slow decay of ACF
    mean_acf = np.mean(acf_values)
    decay_rate = (acf_values[0] - acf_values[-1]) / len(acf_values) if len(acf_values) > 1 else 0
    
    # High mean ACF + slow decay = long memory
    score = min(1.0, mean_acf + (1 - min(1, decay_rate * 10)))
    
    return DiagnosticResult(
        "long_memory",
        score / 2,  # Scale down
        mean_acf,
        f"Mean |ACF|: {mean_acf:.2f}, decay: {decay_rate:.3f}",
        score >= 0.5
    )


# --- Reflexive Stochastic Diagnostics ---

def diagnose_fat_tails(series: np.ndarray) -> DiagnosticResult:
    """
    Returns or increments non-Gaussian? Excess kurtosis?
    """
    if len(series) < 30:
        return DiagnosticResult("fat_tails", 0.0, 0.0, "Insufficient data", False)
    
    # Use returns/differences
    returns = np.diff(series)
    
    kurtosis = _safe_compute(stats.kurtosis, returns, default=0.0)
    
    # Excess kurtosis > 0 indicates fat tails (normal = 0)
    score = min(1.0, max(0, kurtosis / 6))  # Normalize: kurtosis of 6 = score 1.0
    
    return DiagnosticResult(
        "fat_tails",
        score,
        kurtosis,
        f"Excess kurtosis: {kurtosis:.2f}",
        kurtosis > 1.0
    )


def diagnose_volatility_clustering(series: np.ndarray) -> DiagnosticResult:
    """
    Conditional variance persists? ARCH effects?
    """
    if len(series) < 50:
        return DiagnosticResult("volatility_clustering", 0.0, 0.0, "Insufficient data", False)
    
    returns = np.diff(series)
    squared_returns = returns ** 2
    
    # Autocorrelation of squared returns
    lag1_acf = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
    
    if np.isnan(lag1_acf):
        return DiagnosticResult("volatility_clustering", 0.0, 0.0, "ACF computation failed", False)
    
    score = min(1.0, max(0, lag1_acf * 2))
    
    return DiagnosticResult(
        "volatility_clustering",
        score,
        lag1_acf,
        f"Squared returns ACF(1): {lag1_acf:.2f}",
        lag1_acf > 0.1
    )


def diagnose_tail_dependence(series: np.ndarray) -> DiagnosticResult:
    """
    Dependencies strengthen during extremes?
    (Simplified: check if extreme values cluster in time)
    """
    if len(series) < 50:
        return DiagnosticResult("tail_dependence", 0.0, 0.0, "Insufficient data", False)
    
    returns = np.diff(series)
    threshold = np.percentile(np.abs(returns), 90)
    
    extreme_idx = np.where(np.abs(returns) > threshold)[0]
    
    if len(extreme_idx) < 2:
        return DiagnosticResult("tail_dependence", 0.0, 0.0, "Too few extremes", False)
    
    # Check clustering: are extremes close together?
    gaps = np.diff(extreme_idx)
    mean_gap = np.mean(gaps)
    expected_gap = len(returns) / len(extreme_idx)
    
    # Clustering score: smaller gaps than expected = clustering
    if expected_gap > 0:
        clustering_ratio = expected_gap / mean_gap
        score = min(1.0, max(0, (clustering_ratio - 1) / 2))
    else:
        score = 0.0
    
    return DiagnosticResult(
        "tail_dependence",
        score,
        clustering_ratio if expected_gap > 0 else 0,
        f"Extreme clustering ratio: {clustering_ratio:.2f}" if expected_gap > 0 else "N/A",
        score >= 0.3
    )


# --- Pure Noise Diagnostics ---

def diagnose_null_dominance(series: np.ndarray, n_surrogates: int = 20) -> DiagnosticResult:
    """
    Does structure collapse under surrogate tests?
    Compare ACF of real vs phase-randomized surrogates.
    """
    if len(series) < 50:
        return DiagnosticResult("null_dominance", 0.5, 0.0, "Insufficient data", False)
    
    # Real ACF at lag 1
    real_acf = np.corrcoef(series[:-1], series[1:])[0, 1]
    
    if np.isnan(real_acf):
        return DiagnosticResult("null_dominance", 0.5, 0.0, "ACF computation failed", False)
    
    # Phase-randomized surrogates
    surrogate_acfs = []
    for _ in range(n_surrogates):
        fft = np.fft.fft(series)
        phases = np.angle(fft)
        random_phases = np.random.uniform(-np.pi, np.pi, len(phases))
        # Keep magnitude, randomize phase
        surrogate_fft = np.abs(fft) * np.exp(1j * random_phases)
        surrogate = np.real(np.fft.ifft(surrogate_fft))
        
        surr_acf = np.corrcoef(surrogate[:-1], surrogate[1:])[0, 1]
        if not np.isnan(surr_acf):
            surrogate_acfs.append(abs(surr_acf))
    
    if not surrogate_acfs:
        return DiagnosticResult("null_dominance", 0.5, 0.0, "Surrogate test failed", False)
    
    # Score: if real ACF not much higher than surrogates, it's noise
    mean_surrogate = np.mean(surrogate_acfs)
    std_surrogate = np.std(surrogate_acfs) + 1e-6
    
    z_score = (abs(real_acf) - mean_surrogate) / std_surrogate
    
    # Noise score: low z-score = can't distinguish from noise
    noise_score = 1.0 / (1.0 + max(0, z_score))
    
    return DiagnosticResult(
        "null_dominance",
        noise_score,
        z_score,
        f"ACF z-score vs surrogates: {z_score:.2f}",
        z_score < 2.0  # Can't distinguish from noise
    )


def diagnose_parameter_instability(series: np.ndarray, n_bootstrap: int = 20) -> DiagnosticResult:
    """
    Small perturbations cause large metric swings?
    """
    if len(series) < 50:
        return DiagnosticResult("parameter_instability", 0.5, 0.0, "Insufficient data", False)
    
    # Compute variance under bootstrap
    bootstrap_vars = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(series), len(series), replace=True)
        bootstrap_sample = series[idx]
        bootstrap_vars.append(np.var(bootstrap_sample))
    
    # Coefficient of variation of bootstrap variances
    cv = np.std(bootstrap_vars) / (np.mean(bootstrap_vars) + 1e-6)
    
    # High CV = unstable
    instability_score = min(1.0, cv * 2)
    
    return DiagnosticResult(
        "parameter_instability",
        instability_score,
        cv,
        f"Bootstrap variance CV: {cv:.2f}",
        cv > 0.3
    )


# =============================================================================
# GEOMETRY SIGNATURE AGENT
# =============================================================================

class GeometrySignatureAgent:
    """
    Determines what class of dynamical system generated the observed data.
    
    Usage:
        agent = GeometrySignatureAgent()
        profile = agent.analyze(series)
        recommendations = agent.recommend_engines(profile)
        
        for rec in recommendations:
            print(f"{rec.engine}: weight={rec.weight:.2f} ({rec.reason})")
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def analyze(self, series: pd.Series | np.ndarray) -> GeometryProfile:
        """
        Run all diagnostics and return geometry profile.
        
        Args:
            series: Time series data (pandas Series or numpy array)
        
        Returns:
            GeometryProfile with scores for each geometry class
        """
        if isinstance(series, pd.Series):
            arr = series.dropna().values
        else:
            arr = series[~np.isnan(series)]
        
        if len(arr) < 30:
            return GeometryProfile(
                noise_score=0.8,
                confidence=0.1,
                dominant_geometry=GeometryClass.PURE_NOISE
            )
        
        # Run diagnostics for each geometry class
        
        # 1. Latent Flow / Compartmental
        latent_diags = [
            diagnose_saturation(arr),
            diagnose_asymmetry(arr),
            diagnose_bounded_accumulation(arr),
        ]
        latent_score = np.mean([d.score for d in latent_diags])
        
        # 2. Coupled Oscillator
        osc_diags = [
            diagnose_periodicity(arr),
            diagnose_long_memory(arr),
        ]
        osc_score = np.mean([d.score for d in osc_diags])
        
        # 3. Reflexive Stochastic
        reflex_diags = [
            diagnose_fat_tails(arr),
            diagnose_volatility_clustering(arr),
            diagnose_tail_dependence(arr),
        ]
        reflex_score = np.mean([d.score for d in reflex_diags])
        
        # 4. Pure Noise
        noise_diags = [
            diagnose_null_dominance(arr),
            diagnose_parameter_instability(arr),
        ]
        noise_score = np.mean([d.score for d in noise_diags])
        
        profile = GeometryProfile(
            latent_flow_score=latent_score,
            oscillator_score=osc_score,
            reflexive_score=reflex_score,
            noise_score=noise_score,
            latent_flow_diagnostics=latent_diags,
            oscillator_diagnostics=osc_diags,
            reflexive_diagnostics=reflex_diags,
            noise_diagnostics=noise_diags,
        )
        
        if self.verbose:
            self._print_diagnostics(profile)
        
        return profile
    
    def recommend_engines(self, profile: GeometryProfile) -> List[EngineRecommendation]:
        """
        Based on geometry profile, recommend which engines to run and with what weight.
        """
        recommendations = []
        
        # Core engines always included
        core_engines = ["pca", "correlation", "entropy", "wavelets"]
        for engine in core_engines:
            recommendations.append(EngineRecommendation(
                engine=engine,
                weight=1.0 - (profile.noise_score * 0.5),  # Downweight if noisy
                reason="Core engine"
            ))
        
        # Latent Flow geometry â†’ compartmental models
        if profile.latent_flow_score >= 0.7:
            recommendations.append(EngineRecommendation(
                engine="sir",
                weight=profile.latent_flow_score,
                reason=f"Strong compartmental signature ({profile.latent_flow_score:.2f})",
                required_gates=["full_outbreak_cycle", "identifiability_check"]
            ))
        elif profile.latent_flow_score >= 0.4:
            recommendations.append(EngineRecommendation(
                engine="sir",
                weight=profile.latent_flow_score * 0.5,
                reason=f"Partial compartmental signature ({profile.latent_flow_score:.2f})",
                required_gates=["full_outbreak_cycle", "identifiability_check", "benchmark_validation"]
            ))
        
        # Oscillator geometry â†’ wavelets emphasized, granger valid
        if profile.oscillator_score >= 0.7:
            # Boost wavelet weight
            for rec in recommendations:
                if rec.engine == "wavelets":
                    rec.weight = min(1.0, rec.weight * 1.5)
                    rec.reason = f"Boosted: strong oscillator signature ({profile.oscillator_score:.2f})"
            
            recommendations.append(EngineRecommendation(
                engine="granger",
                weight=profile.oscillator_score * 0.8,
                reason=f"Oscillator geometry supports lead-lag ({profile.oscillator_score:.2f})",
                required_gates=["stationarity"]
            ))
        
        # Reflexive geometry â†’ HMM, DMD, copula
        if profile.reflexive_score >= 0.5:
            recommendations.append(EngineRecommendation(
                engine="hmm",
                weight=profile.reflexive_score,
                reason=f"Reflexive dynamics detected ({profile.reflexive_score:.2f})",
                required_gates=["state_stability", "dwell_time"]
            ))
            
            recommendations.append(EngineRecommendation(
                engine="dmd",
                weight=profile.reflexive_score * 0.8,
                reason=f"Reflexive dynamics ({profile.reflexive_score:.2f})",
                required_gates=["rank_stability"]
            ))
            
            if profile.reflexive_score >= 0.7:
                recommendations.append(EngineRecommendation(
                    engine="copula",
                    weight=profile.reflexive_score * 0.7,
                    reason=f"Strong tail dependence signature ({profile.reflexive_score:.2f})",
                    required_gates=["min_window_252", "bootstrap_ci"]
                ))
        
        # Mutual information for non-linear dependencies
        if profile.reflexive_score >= 0.4 or profile.latent_flow_score >= 0.4:
            recommendations.append(EngineRecommendation(
                engine="mutual_information",
                weight=max(profile.reflexive_score, profile.latent_flow_score) * 0.8,
                reason="Non-linear dependencies possible",
                required_gates=["permutation_null", "fdr_control"]
            ))
        
        # Suppress if noise-dominated
        if profile.noise_score >= 0.6:
            for rec in recommendations:
                rec.weight *= (1 - profile.noise_score)
                rec.reason += f" [DOWNWEIGHTED: noise={profile.noise_score:.2f}]"
        
        return sorted(recommendations, key=lambda r: r.weight, reverse=True)
    
    def get_warnings(self, profile: GeometryProfile) -> List[str]:
        """Generate warnings based on geometry profile."""
        warnings = []
        
        if profile.noise_score >= 0.6:
            warnings.append(
                f"âš ï¸ HIGH NOISE ({profile.noise_score:.2f}): Structure may not be distinguishable "
                "from random. All engine outputs should be treated with low confidence."
            )
        
        if profile.is_hybrid:
            warnings.append(
                "âš ï¸ HYBRID GEOMETRY: Multiple geometry classes detected. Consider analyzing "
                "sub-periods separately or using layered modeling."
            )
        
        if profile.latent_flow_score >= 0.5 and profile.reflexive_score >= 0.5:
            warnings.append(
                "âš ï¸ MIXED DYNAMICS: Both compartmental and reflexive signatures present. "
                "This may indicate behavior-modified epidemic dynamics."
            )
        
        if profile.confidence < 0.5:
            warnings.append(
                f"âš ï¸ LOW CONFIDENCE ({profile.confidence:.2f}): Geometry classification uncertain. "
                "Consider longer time series or additional diagnostics."
            )
        
        return warnings
    
    def _print_diagnostics(self, profile: GeometryProfile):
        """Print detailed diagnostic results."""
        print("=" * 60)
        print("GEOMETRY SIGNATURE ANALYSIS")
        print("=" * 60)
        
        print(f"\nLatent Flow Score: {profile.latent_flow_score:.2f}")
        for d in profile.latent_flow_diagnostics:
            status = "âœ“" if d.passed else "âœ—"
            print(f"  {status} {d.name}: {d.score:.2f} - {d.description}")
        
        print(f"\nOscillator Score: {profile.oscillator_score:.2f}")
        for d in profile.oscillator_diagnostics:
            status = "âœ“" if d.passed else "âœ—"
            print(f"  {status} {d.name}: {d.score:.2f} - {d.description}")
        
        print(f"\nReflexive Score: {profile.reflexive_score:.2f}")
        for d in profile.reflexive_diagnostics:
            status = "âœ“" if d.passed else "âœ—"
            print(f"  {status} {d.name}: {d.score:.2f} - {d.description}")
        
        print(f"\nNoise Score: {profile.noise_score:.2f}")
        for d in profile.noise_diagnostics:
            status = "âœ“" if d.passed else "âœ—"
            print(f"  {status} {d.name}: {d.score:.2f} - {d.description}")
        
        print(f"\n{'=' * 60}")
        print(f"DOMINANT GEOMETRY: {profile.dominant_geometry.value}")
        print(f"HYBRID: {profile.is_hybrid}")
        print(f"CONFIDENCE: {profile.confidence:.2f}")
        print("=" * 60)


# =============================================================================
# CLI / DEMO
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    # Demo with synthetic data
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("DEMO: Testing Geometry Signature Agent on synthetic data")
    print("=" * 70)
    
    agent = GeometrySignatureAgent(verbose=True)
    
    # Test 1: Logistic growth (compartmental)
    print("\n\n>>> TEST 1: Logistic growth (should detect LATENT_FLOW)")
    t = np.linspace(0, 10, 200)
    logistic = 100 / (1 + np.exp(-1.5 * (t - 5))) + np.random.normal(0, 2, len(t))
    profile1 = agent.analyze(logistic)
    recs1 = agent.recommend_engines(profile1)
    print("\nTop recommendations:")
    for r in recs1[:5]:
        print(f"  {r.engine}: {r.weight:.2f} - {r.reason}")
    
    # Test 2: Sine wave (oscillator)
    print("\n\n>>> TEST 2: Sine wave (should detect COUPLED_OSCILLATOR)")
    sine = np.sin(np.linspace(0, 8*np.pi, 200)) + np.random.normal(0, 0.1, 200)
    profile2 = agent.analyze(sine)
    recs2 = agent.recommend_engines(profile2)
    print("\nTop recommendations:")
    for r in recs2[:5]:
        print(f"  {r.engine}: {r.weight:.2f} - {r.reason}")
    
    # Test 3: GARCH-like (reflexive)
    print("\n\n>>> TEST 3: GARCH-like volatility clustering (should detect REFLEXIVE)")
    returns = np.random.standard_t(4, 200) * 0.02
    vol = np.ones(200)
    for i in range(1, 200):
        vol[i] = 0.9 * vol[i-1] + 0.1 * returns[i-1]**2
    reflexive = np.cumsum(returns * np.sqrt(vol))
    profile3 = agent.analyze(reflexive)
    recs3 = agent.recommend_engines(profile3)
    print("\nTop recommendations:")
    for r in recs3[:5]:
        print(f"  {r.engine}: {r.weight:.2f} - {r.reason}")
    
    # Test 4: Pure noise
    print("\n\n>>> TEST 4: Pure noise (should detect PURE_NOISE)")
    noise = np.random.normal(0, 1, 200)
    profile4 = agent.analyze(noise)
    recs4 = agent.recommend_engines(profile4)
    print("\nTop recommendations:")
    for r in recs4[:5]:
        print(f"  {r.engine}: {r.weight:.2f} - {r.reason}")
    
    print("\n\n" + "=" * 70)
    print("Geometry Signature Agent ready for PRISM integration")
    print("=" * 70)
