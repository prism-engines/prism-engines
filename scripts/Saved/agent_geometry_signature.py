"""
Geometry Signature Agent v3

Core Geometry Classes:
    1. Latent Flow / Compartmental (epidemics, diffusion, adoption, S-curves)
    2. Coupled Oscillator (climate, cycles, physical systems)
    3. Reflexive Stochastic (finance, social, sentiment-driven)
    4. Pure Noise / Noise-Dominated

v3 Changes:
    - Direct S-curve (logistic) fitting for latent flow detection
    - Smoothed monotonicity to ignore micro-noise reversals
    - Derivative shape analysis (bell curve vs monotonic decay)
    - Removed asymmetry test (too strict for monotonic saturation)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import warnings

warnings.filterwarnings('ignore')


class GeometryClass(Enum):
    LATENT_FLOW = "latent_flow"
    COUPLED_OSCILLATOR = "coupled_oscillator"
    REFLEXIVE_STOCHASTIC = "reflexive_stochastic"
    PURE_NOISE = "pure_noise"


@dataclass
class DiagnosticResult:
    name: str
    score: float
    metric_value: float
    description: str
    passed: bool


@dataclass 
class GeometryProfile:
    latent_flow_score: float = 0.0
    oscillator_score: float = 0.0
    reflexive_score: float = 0.0
    noise_score: float = 0.0
    
    latent_flow_diagnostics: List[DiagnosticResult] = field(default_factory=list)
    oscillator_diagnostics: List[DiagnosticResult] = field(default_factory=list)
    reflexive_diagnostics: List[DiagnosticResult] = field(default_factory=list)
    noise_diagnostics: List[DiagnosticResult] = field(default_factory=list)
    
    dominant_geometry: Optional[GeometryClass] = None
    is_hybrid: bool = False
    confidence: float = 0.0
    
    def __post_init__(self):
        self._compute_derived()
    
    def _compute_derived(self):
        scores = {
            GeometryClass.LATENT_FLOW: self.latent_flow_score,
            GeometryClass.COUPLED_OSCILLATOR: self.oscillator_score,
            GeometryClass.REFLEXIVE_STOCHASTIC: self.reflexive_score,
            GeometryClass.PURE_NOISE: self.noise_score,
        }
        self.dominant_geometry = max(scores, key=scores.get)
        max_score = scores[self.dominant_geometry]
        
        high_scores = [g for g, s in scores.items() if s >= 0.5 and g != GeometryClass.PURE_NOISE]
        self.is_hybrid = len(high_scores) > 1
        
        if self.noise_score >= 0.6:
            self.confidence = 0.2
        else:
            sorted_scores = sorted(scores.values(), reverse=True)
            separation = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
            self.confidence = min(1.0, max_score * 0.7 + separation * 0.3)


@dataclass
class EngineRecommendation:
    engine: str
    weight: float
    reason: str
    required_gates: List[str] = field(default_factory=list)


def _safe_compute(func, *args, default=0.0, **kwargs):
    try:
        result = func(*args, **kwargs)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except:
        return default


# =============================================================================
# LATENT FLOW DIAGNOSTICS
# =============================================================================

def diagnose_scurve(series: np.ndarray) -> DiagnosticResult:
    """
    Direct S-curve (logistic) fit. High R² + beats linear = latent flow.
    """
    if len(series) < 30:
        return DiagnosticResult("scurve", 0.0, 0.0, "Insufficient data", False)
    
    t = np.arange(len(series))
    y = series.astype(float)
    
    y_min, y_max = np.min(y), np.max(y)
    if y_max - y_min < 1e-10:
        return DiagnosticResult("scurve", 0.0, 0.0, "No variation", False)
    
    y_norm = (y - y_min) / (y_max - y_min)
    
    def logistic(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))
    
    def linear(t, a, b):
        return a * t + b
    
    try:
        # Logistic fit
        p0_log = [1.0, 0.1, len(t)/2]
        bounds_log = ([0.5, 0.001, 0], [1.5, 1.0, len(t)])
        popt_log, _ = curve_fit(logistic, t, y_norm, p0=p0_log, bounds=bounds_log, maxfev=5000)
        pred_log = logistic(t, *popt_log)
        ss_log = np.sum((y_norm - pred_log)**2)
        
        # Linear fit
        popt_lin, _ = curve_fit(linear, t, y_norm, maxfev=1000)
        pred_lin = linear(t, *popt_lin)
        ss_lin = np.sum((y_norm - pred_lin)**2)
        
        ss_tot = np.sum((y_norm - np.mean(y_norm))**2)
        r2_log = 1 - ss_log / ss_tot if ss_tot > 0 else 0
        improvement = (ss_lin - ss_log) / ss_lin if ss_lin > 0 else 0
        
        k_value = popt_log[1]
        
        # Score based on fit quality
        if r2_log > 0.95 and improvement > 0.1:
            score = 0.95
        elif r2_log > 0.9 and improvement > 0.05:
            score = 0.85
        elif r2_log > 0.8:
            score = 0.7
        else:
            score = max(0, r2_log * 0.5)
        
        return DiagnosticResult(
            "scurve", score, r2_log,
            f"R²={r2_log:.3f}, improvement={improvement:.2f}, k={k_value:.3f}",
            score >= 0.5
        )
    except:
        return DiagnosticResult("scurve", 0.0, 0.0, "Fitting failed", False)


def diagnose_derivative_shape(series: np.ndarray) -> DiagnosticResult:
    """
    Check derivative: bell-curve (epidemic) or monotonic decay (saturation).
    Both are latent flow signatures.
    """
    if len(series) < 30:
        return DiagnosticResult("derivative_shape", 0.0, 0.0, "Insufficient data", False)
    
    window = max(5, len(series) // 20)
    smoothed = uniform_filter1d(series.astype(float), size=window)
    deriv = np.diff(smoothed)
    
    # Characteristics
    pos_frac = np.mean(deriv > 0)
    deriv_peak_idx = np.argmax(np.abs(deriv))
    has_interior_peak = 0.1 * len(deriv) < deriv_peak_idx < 0.9 * len(deriv)
    
    first_half = np.mean(np.abs(deriv[:len(deriv)//2]))
    second_half = np.mean(np.abs(deriv[len(deriv)//2:]))
    is_decelerating = second_half < first_half * 0.8
    
    # Mostly positive + decelerating = S-curve growth
    if pos_frac > 0.7 and is_decelerating:
        score = 0.85
    # Mostly negative + decelerating = decay
    elif pos_frac < 0.3 and is_decelerating:
        score = 0.7
    # Interior peak = epidemic bell curve
    elif has_interior_peak:
        score = 0.75
    # Mixed but structured
    elif is_decelerating:
        score = 0.5
    else:
        score = 0.2
    
    return DiagnosticResult(
        "derivative_shape", score, pos_frac,
        f"pos_frac={pos_frac:.2f}, decel={is_decelerating}, peak={has_interior_peak}",
        score >= 0.5
    )


def diagnose_smoothed_monotonicity(series: np.ndarray) -> DiagnosticResult:
    """
    Monotonicity on smoothed series - ignores micro-noise reversals.
    """
    if len(series) < 30:
        return DiagnosticResult("smooth_mono", 0.0, 0.0, "Insufficient data", False)
    
    window = max(7, len(series) // 15)
    smoothed = uniform_filter1d(series.astype(float), size=window)
    
    diffs = np.diff(smoothed)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    
    if len(signs) < 2:
        return DiagnosticResult("smooth_mono", 0.5, 0.0, "Too few movements", False)
    
    sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
    max_changes = len(signs) - 1
    monotonicity = 1 - (sign_changes / max_changes) if max_changes > 0 else 0.5
    
    # High smoothed monotonicity = latent flow
    if monotonicity > 0.85:
        score = 0.95
    elif monotonicity > 0.7:
        score = 0.8
    elif monotonicity > 0.5:
        score = 0.6
    else:
        score = 0.3
    
    return DiagnosticResult(
        "smooth_mono", score, monotonicity,
        f"Smoothed monotonicity: {monotonicity:.2f} (changes: {sign_changes})",
        score >= 0.5
    )


def diagnose_bounded_accumulation(series: np.ndarray) -> DiagnosticResult:
    """Cumulative variance stabilizes?"""
    if len(series) < 30:
        return DiagnosticResult("bounded", 0.0, 0.0, "Insufficient data", False)
    
    cumulative = np.cumsum(series - np.mean(series))
    
    first_half_var = np.var(cumulative[:len(cumulative)//2])
    second_half_var = np.var(cumulative[len(cumulative)//2:])
    
    if first_half_var > 0:
        var_ratio = second_half_var / first_half_var
        base_score = 1.0 / (1.0 + max(0, var_ratio - 1))
    else:
        var_ratio = 1.0
        base_score = 0.5
    
    return DiagnosticResult(
        "bounded", base_score, var_ratio,
        f"Var ratio: {var_ratio:.2f}",
        base_score >= 0.5
    )


# =============================================================================
# OSCILLATOR DIAGNOSTICS
# =============================================================================

def diagnose_periodicity(series: np.ndarray) -> DiagnosticResult:
    """Concentrated spectral power + harmonics = oscillator."""
    if len(series) < 50:
        return DiagnosticResult("periodicity", 0.0, 0.0, "Insufficient data", False)
    
    detrended = series - np.mean(series)
    fft = np.fft.fft(detrended)
    power = np.abs(fft[:len(fft)//2])**2
    
    min_freq_bin = min(3, len(power) // 10)
    if len(power) > min_freq_bin + 5:
        power_no_trend = power[min_freq_bin:]
    else:
        power_no_trend = power
    
    sorted_power = np.sort(power_no_trend)[::-1]
    total_power = np.sum(power_no_trend)
    
    if total_power > 0:
        top3_concentration = np.sum(sorted_power[:3]) / total_power
        
        peak_idx = np.argmax(power_no_trend)
        has_harmonics = False
        if peak_idx > 0:
            median_power = np.median(power_no_trend)
            for mult in [2, 3]:
                harm_idx = peak_idx * mult
                if harm_idx < len(power_no_trend):
                    if power_no_trend[harm_idx] > median_power * 1.5:
                        has_harmonics = True
        
        base_score = top3_concentration
        score = min(1.0, base_score * 1.3) if has_harmonics else base_score * 0.6
    else:
        score = 0.0
        top3_concentration = 0.0
        has_harmonics = False
    
    return DiagnosticResult(
        "periodicity", score, top3_concentration,
        f"Top3={top3_concentration:.2f}, harmonics={has_harmonics}",
        score >= 0.5
    )


def diagnose_long_memory(series: np.ndarray) -> DiagnosticResult:
    """High ACF with slow decay, penalized if monotonic."""
    if len(series) < 50:
        return DiagnosticResult("long_memory", 0.0, 0.0, "Insufficient data", False)
    
    max_lag = min(50, len(series) // 4)
    acf_values = []
    for lag in range(1, max_lag + 1):
        acf = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        if not np.isnan(acf):
            acf_values.append(abs(acf))
    
    if not acf_values:
        return DiagnosticResult("long_memory", 0.0, 0.0, "ACF failed", False)
    
    mean_acf = np.mean(acf_values)
    decay_rate = (acf_values[0] - acf_values[-1]) / len(acf_values) if len(acf_values) > 1 else 0
    
    # Check monotonicity (smoothed)
    window = max(5, len(series) // 20)
    smoothed = uniform_filter1d(series.astype(float), size=window)
    diffs = np.diff(smoothed)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) > 1:
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        monotonicity = 1 - (sign_changes / (len(signs) - 1))
    else:
        monotonicity = 0.5
    
    base_score = mean_acf * 0.6 + max(0, 1 - decay_rate * 20) * 0.4
    
    # Penalize monotonic (high ACF from trend, not oscillation)
    if monotonicity > 0.7:
        penalty = (monotonicity - 0.7) / 0.3
        base_score *= (1 - penalty * 0.8)
    
    return DiagnosticResult(
        "long_memory", max(0, min(1, base_score)), mean_acf,
        f"ACF={mean_acf:.2f}, mono={monotonicity:.2f}",
        base_score >= 0.5
    )


def diagnose_zero_crossings(series: np.ndarray) -> DiagnosticResult:
    """Oscillators cross mean regularly."""
    if len(series) < 50:
        return DiagnosticResult("zero_cross", 0.0, 0.0, "Insufficient data", False)
    
    centered = series - np.mean(series)
    signs = np.sign(centered)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    
    crossing_rate = crossings / (len(series) / 2)
    
    crossing_idx = np.where(np.abs(np.diff(signs)) > 0)[0]
    if len(crossing_idx) > 2:
        gaps = np.diff(crossing_idx)
        regularity = 1.0 / (1.0 + np.std(gaps) / (np.mean(gaps) + 1e-6))
    else:
        regularity = 0.0
    
    if 0.3 < crossing_rate < 1.5:
        score = regularity * 0.8 + 0.2 * (1 - abs(crossing_rate - 0.7))
    else:
        score = regularity * 0.3
    
    return DiagnosticResult(
        "zero_cross", max(0, min(1, score)), crossing_rate,
        f"Rate={crossing_rate:.2f}, regularity={regularity:.2f}",
        score >= 0.5
    )


# =============================================================================
# REFLEXIVE STOCHASTIC DIAGNOSTICS
# =============================================================================

def diagnose_fat_tails(series: np.ndarray) -> DiagnosticResult:
    if len(series) < 30:
        return DiagnosticResult("fat_tails", 0.0, 0.0, "Insufficient data", False)
    
    returns = np.diff(series)
    kurtosis = _safe_compute(stats.kurtosis, returns, default=0.0)
    score = min(1.0, max(0, kurtosis / 6))
    
    return DiagnosticResult("fat_tails", score, kurtosis, f"Kurt={kurtosis:.2f}", kurtosis > 1.0)


def diagnose_volatility_clustering(series: np.ndarray) -> DiagnosticResult:
    if len(series) < 50:
        return DiagnosticResult("vol_cluster", 0.0, 0.0, "Insufficient data", False)
    
    returns = np.diff(series)
    sq_ret = returns ** 2
    acf = np.corrcoef(sq_ret[:-1], sq_ret[1:])[0, 1]
    
    if np.isnan(acf):
        return DiagnosticResult("vol_cluster", 0.0, 0.0, "ACF failed", False)
    
    score = min(1.0, max(0, acf * 2))
    return DiagnosticResult("vol_cluster", score, acf, f"Sq.ACF={acf:.2f}", acf > 0.1)


def diagnose_tail_dependence(series: np.ndarray) -> DiagnosticResult:
    if len(series) < 50:
        return DiagnosticResult("tail_dep", 0.0, 0.0, "Insufficient data", False)
    
    returns = np.diff(series)
    threshold = np.percentile(np.abs(returns), 90)
    extreme_idx = np.where(np.abs(returns) > threshold)[0]
    
    if len(extreme_idx) < 2:
        return DiagnosticResult("tail_dep", 0.0, 0.0, "Few extremes", False)
    
    gaps = np.diff(extreme_idx)
    mean_gap = np.mean(gaps)
    expected_gap = len(returns) / len(extreme_idx)
    
    if expected_gap > 0:
        ratio = expected_gap / mean_gap
        score = min(1.0, max(0, (ratio - 1) / 2))
    else:
        score = 0.0
        ratio = 0.0
    
    return DiagnosticResult("tail_dep", score, ratio, f"Cluster ratio={ratio:.2f}", score >= 0.3)


# =============================================================================
# NOISE DIAGNOSTICS
# =============================================================================

def diagnose_spectral_flatness(series: np.ndarray) -> DiagnosticResult:
    if len(series) < 50:
        return DiagnosticResult("spectral_flat", 0.5, 0.0, "Insufficient data", False)
    
    fft = np.fft.fft(series - np.mean(series))
    power = np.abs(fft[:len(fft)//2])**2 + 1e-10
    
    geo_mean = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)
    flatness = geo_mean / arith_mean
    
    return DiagnosticResult("spectral_flat", flatness, flatness, f"Flatness={flatness:.2f}", flatness > 0.5)


def diagnose_null_dominance(series: np.ndarray, n_surr: int = 20) -> DiagnosticResult:
    if len(series) < 50:
        return DiagnosticResult("null_dom", 0.5, 0.0, "Insufficient data", False)
    
    real_acf = np.corrcoef(series[:-1], series[1:])[0, 1]
    if np.isnan(real_acf):
        return DiagnosticResult("null_dom", 0.5, 0.0, "ACF failed", False)
    
    surr_acfs = []
    for _ in range(n_surr):
        fft = np.fft.fft(series)
        rand_phase = np.random.uniform(-np.pi, np.pi, len(fft))
        surr = np.real(np.fft.ifft(np.abs(fft) * np.exp(1j * rand_phase)))
        acf = np.corrcoef(surr[:-1], surr[1:])[0, 1]
        if not np.isnan(acf):
            surr_acfs.append(abs(acf))
    
    if not surr_acfs:
        return DiagnosticResult("null_dom", 0.5, 0.0, "Surrogates failed", False)
    
    z = (abs(real_acf) - np.mean(surr_acfs)) / (np.std(surr_acfs) + 1e-6)
    noise_score = 1.0 / (1.0 + max(0, z))
    
    return DiagnosticResult("null_dom", noise_score, z, f"z={z:.2f}", z < 2.0)


def diagnose_randomness(series: np.ndarray) -> DiagnosticResult:
    if len(series) < 50:
        return DiagnosticResult("randomness", 0.5, 0.0, "Insufficient data", False)
    
    median = np.median(series)
    above = series > median
    
    runs = 1
    for i in range(1, len(above)):
        if above[i] != above[i-1]:
            runs += 1
    
    n1, n2 = np.sum(above), len(above) - np.sum(above)
    
    if n1 > 0 and n2 > 0:
        expected = (2 * n1 * n2) / (n1 + n2) + 1
        var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
        z = (runs - expected) / np.sqrt(var) if var > 0 else 0
        score = np.exp(-z**2 / 8)
    else:
        score = 0.5
        z = 0.0
    
    return DiagnosticResult("randomness", score, z, f"Runs z={z:.2f}", score > 0.5)


def diagnose_drift(series: np.ndarray) -> DiagnosticResult:
    if len(series) < 50:
        return DiagnosticResult("drift", 0.5, 0.0, "Insufficient data", False)
    
    returns = np.diff(series)
    vol = np.std(returns) + 1e-10
    drift = (series[-1] - series[0]) / (len(series) * vol)
    norm_drift = abs(drift) * np.sqrt(len(series))
    
    noise_score = 2 * (1 - stats.norm.cdf(norm_drift))
    
    return DiagnosticResult("drift", max(0, min(1, noise_score)), norm_drift, f"Drift={norm_drift:.2f}", noise_score > 0.5)


# =============================================================================
# GEOMETRY SIGNATURE AGENT
# =============================================================================

class GeometrySignatureAgent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def analyze(self, series) -> GeometryProfile:
        if isinstance(series, pd.Series):
            arr = series.dropna().values
        else:
            arr = np.array(series)
            arr = arr[~np.isnan(arr)]
        
        if len(arr) < 30:
            return GeometryProfile(noise_score=0.8, confidence=0.1, dominant_geometry=GeometryClass.PURE_NOISE)
        
        # Latent Flow (v3: S-curve + derivative + smoothed monotonicity)
        latent_diags = [
            diagnose_scurve(arr),
            diagnose_derivative_shape(arr),
            diagnose_smoothed_monotonicity(arr),
            diagnose_bounded_accumulation(arr),
        ]
        latent_score = np.mean([d.score for d in latent_diags])
        
        # Oscillator
        osc_diags = [
            diagnose_periodicity(arr),
            diagnose_long_memory(arr),
            diagnose_zero_crossings(arr),
        ]
        osc_score = np.mean([d.score for d in osc_diags])
        
        # Reflexive
        reflex_diags = [
            diagnose_fat_tails(arr),
            diagnose_volatility_clustering(arr),
            diagnose_tail_dependence(arr),
        ]
        reflex_score = np.mean([d.score for d in reflex_diags])
        
        # Noise
        noise_diags = [
            diagnose_spectral_flatness(arr),
            diagnose_null_dominance(arr),
            diagnose_randomness(arr),
            diagnose_drift(arr),
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
        recs = []
        
        for eng in ["pca", "correlation", "entropy", "wavelets"]:
            recs.append(EngineRecommendation(eng, 1.0 - profile.noise_score * 0.5, "Core engine"))
        
        if profile.latent_flow_score >= 0.7:
            recs.append(EngineRecommendation("sir", profile.latent_flow_score, f"Strong flow ({profile.latent_flow_score:.2f})", ["full_outbreak_cycle"]))
        elif profile.latent_flow_score >= 0.4:
            recs.append(EngineRecommendation("sir", profile.latent_flow_score * 0.5, f"Partial flow ({profile.latent_flow_score:.2f})", ["benchmark"]))
        
        if profile.oscillator_score >= 0.7:
            for r in recs:
                if r.engine == "wavelets":
                    r.weight = min(1.0, r.weight * 1.5)
            recs.append(EngineRecommendation("granger", profile.oscillator_score * 0.8, f"Oscillator ({profile.oscillator_score:.2f})", ["stationarity"]))
        
        if profile.reflexive_score >= 0.5:
            recs.append(EngineRecommendation("hmm", profile.reflexive_score, f"Reflexive ({profile.reflexive_score:.2f})", ["state_stability"]))
            recs.append(EngineRecommendation("dmd", profile.reflexive_score * 0.8, f"Reflexive ({profile.reflexive_score:.2f})", ["rank_stability"]))
        
        if profile.noise_score >= 0.6:
            for r in recs:
                r.weight *= (1 - profile.noise_score)
                r.reason += f" [NOISE={profile.noise_score:.2f}]"
        
        return sorted(recs, key=lambda r: r.weight, reverse=True)
    
    def _print_diagnostics(self, profile: GeometryProfile):
        print("=" * 60)
        print("GEOMETRY SIGNATURE ANALYSIS v3")
        print("=" * 60)
        
        for name, score, diags in [
            ("Latent Flow", profile.latent_flow_score, profile.latent_flow_diagnostics),
            ("Oscillator", profile.oscillator_score, profile.oscillator_diagnostics),
            ("Reflexive", profile.reflexive_score, profile.reflexive_diagnostics),
            ("Noise", profile.noise_score, profile.noise_diagnostics),
        ]:
            print(f"\n{name} Score: {score:.2f}")
            for d in diags:
                status = "✓" if d.passed else "✗"
                print(f"  {status} {d.name}: {d.score:.2f} - {d.description}")
        
        print(f"\n{'=' * 60}")
        print(f"DOMINANT: {profile.dominant_geometry.value}")
        print(f"HYBRID: {profile.is_hybrid} | CONFIDENCE: {profile.confidence:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("Geometry Signature Agent v3 - Synthetic Tests")
    print("=" * 70)
    
    agent = GeometrySignatureAgent(verbose=True)
    
    # Test 1: Logistic
    print("\n>>> TEST 1: Logistic growth (expect LATENT_FLOW)")
    t = np.linspace(0, 10, 200)
    logistic = 100 / (1 + np.exp(-1.5 * (t - 5))) + np.random.normal(0, 2, len(t))
    p1 = agent.analyze(logistic)
    
    # Test 2: Sine
    print("\n>>> TEST 2: Sine wave (expect COUPLED_OSCILLATOR)")
    sine = np.sin(np.linspace(0, 8*np.pi, 200)) + np.random.normal(0, 0.1, 200)
    p2 = agent.analyze(sine)
    
    # Test 3: GARCH
    print("\n>>> TEST 3: GARCH (expect REFLEXIVE_STOCHASTIC)")
    ret = np.random.standard_t(4, 200) * 0.02
    vol = np.ones(200)
    for i in range(1, 200):
        vol[i] = 0.9 * vol[i-1] + 0.1 * ret[i-1]**2
    garch = np.cumsum(ret * np.sqrt(vol))
    p3 = agent.analyze(garch)
    
    # Test 4: Noise
    print("\n>>> TEST 4: White noise (expect PURE_NOISE)")
    noise = np.random.normal(0, 1, 200)
    p4 = agent.analyze(noise)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Logistic", p1, "latent_flow"),
        ("Sine", p2, "coupled_oscillator"),
        ("GARCH", p3, "reflexive_stochastic"),
        ("Noise", p4, "pure_noise"),
    ]
    
    correct = 0
    for name, p, expected in results:
        actual = p.dominant_geometry.value
        ok = "✓" if actual == expected else "✗"
        if actual == expected:
            correct += 1
        print(f"{ok} {name:12}: {actual:25} (expected: {expected})")
    
    print(f"\nAccuracy: {correct}/4 ({correct/4*100:.0f}%)")