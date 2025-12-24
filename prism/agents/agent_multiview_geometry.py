"""
PRISM Multi-View Geometry Analysis

Different data transformations reveal different structural properties.
This module runs geometry signature analysis on multiple views and
reconciles them through an arbitration agent.

Philosophy:
    - Level view of SPY *is* latent_flow (trending accumulation)
    - Returns view of SPY *is* reflexive_stochastic (vol clustering, fat tails)
    - Both are TRUE - it's about which structural layer you're examining

Views:
    1. LEVEL        → Raw series (accumulation, trend)
    2. RETURNS      → First differences / log returns (dynamics)
    3. VOLATILITY   → Rolling std of returns (second-order structure)
    4. DEVIATION    → Distance from moving average (mean reversion)

View-Aware Scoring:
    v3 geometry agent was designed for LEVEL data. When applied to other views,
    some diagnostics become structurally inapplicable:
    
    RETURNS view:
        - Stationary by construction (no accumulation possible)
        - Latent_flow diagnostics (saturation, S-curve) → demoted 80%
        - Reflexive diagnostics (fat tails, vol clustering) → boosted 20%
        
    VOLATILITY view:
        - Second-order structure (vol of vol)
        - Latent_flow → demoted 60%
        - Reflexive (clustering, persistence) → boosted 30%
        
    DEVIATION view:
        - Mean-reverting by construction
        - Oscillator → boosted 40%
        - Latent_flow → demoted 70%

    This preserves v3's honest measurement while adding structural intelligence
    about what each transformation can reveal.

Domain-agnostic:
    Finance: level, returns, volatility, deviation
    Climate: level, anomaly, rate_of_change, deviation
    Epidemiology: cumulative, incidence, Rt, deviation

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
import numpy as np
import pandas as pd

from prism.config import get_geometry_config

logger = logging.getLogger(__name__)


# =============================================================================
# VIEW DEFINITIONS
# =============================================================================

class ViewType(Enum):
    """Canonical data transformations."""
    LEVEL = "level"               # Raw series
    RETURNS = "returns"           # First differences (pct_change or diff)
    VOLATILITY = "volatility"     # Rolling std of returns
    DEVIATION = "deviation"       # Distance from MA (z-score)
    ANOMALY = "anomaly"           # For climate: deviation from seasonal norm
    INCIDENCE = "incidence"       # For epi: new cases (diff of cumulative)


@dataclass
class ViewConfig:
    """Configuration for a single view transformation."""
    view_type: ViewType
    window: int = 21              # Rolling window for vol/deviation
    use_log: bool = False         # Log returns vs simple returns
    center: bool = True           # Center deviation around MA
    min_periods: int = 10         # Minimum periods for rolling calcs


# =============================================================================
# VIEW TRANSFORMER
# =============================================================================

class ViewTransformer:
    """
    Transforms raw time series into different structural views.
    
    Each transformation reveals different geometry:
    - Level: Shows accumulation, trend, saturation (latent_flow)
    - Returns: Shows dynamics, feedback, fat tails (reflexive)
    - Volatility: Shows second-order structure (reflexive)
    - Deviation: Shows mean reversion (oscillator)
    """
    
    @staticmethod
    def transform(
        series: np.ndarray,
        view_type: ViewType,
        config: Optional[ViewConfig] = None
    ) -> np.ndarray:
        """
        Transform series to specified view.
        
        Args:
            series: Raw time series (1D array)
            view_type: Type of transformation
            config: Optional configuration
            
        Returns:
            Transformed series (may be shorter due to differencing/rolling)
        """
        if config is None:
            config = ViewConfig(view_type=view_type)
        
        series = np.asarray(series).flatten()
        
        if view_type == ViewType.LEVEL:
            return series
        
        elif view_type == ViewType.RETURNS:
            return ViewTransformer._compute_returns(series, config.use_log)
        
        elif view_type == ViewType.VOLATILITY:
            returns = ViewTransformer._compute_returns(series, config.use_log)
            return ViewTransformer._rolling_std(returns, config.window, config.min_periods)
        
        elif view_type == ViewType.DEVIATION:
            return ViewTransformer._compute_deviation(series, config.window, config.min_periods)
        
        elif view_type == ViewType.ANOMALY:
            # For climate: deviation from rolling mean (same as deviation for now)
            return ViewTransformer._compute_deviation(series, config.window, config.min_periods)
        
        elif view_type == ViewType.INCIDENCE:
            # For epi: first difference (new cases)
            return np.diff(series)
        
        else:
            raise ValueError(f"Unknown view type: {view_type}")
    
    @staticmethod
    def _compute_returns(series: np.ndarray, use_log: bool = False) -> np.ndarray:
        """Compute returns (first differences or log returns)."""
        if len(series) < 2:
            return np.array([])
        
        if use_log:
            # Log returns: log(p_t / p_{t-1})
            with np.errstate(divide='ignore', invalid='ignore'):
                log_series = np.log(np.maximum(series, 1e-10))
            return np.diff(log_series)
        else:
            # Simple returns: (p_t - p_{t-1}) / p_{t-1}
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = np.diff(series) / np.maximum(np.abs(series[:-1]), 1e-10)
            # Clip extreme values
            returns = np.clip(returns, -10, 10)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            return returns
    
    @staticmethod
    def _rolling_std(series: np.ndarray, window: int, min_periods: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        if len(series) < min_periods:
            return np.array([])
        
        result = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            window_data = series[start:i+1]
            if len(window_data) >= min_periods:
                result.append(np.std(window_data, ddof=1))
            else:
                result.append(np.nan)
        
        result = np.array(result)
        # Remove leading NaNs
        first_valid = np.where(~np.isnan(result))[0]
        if len(first_valid) > 0:
            return result[first_valid[0]:]
        return np.array([])
    
    @staticmethod
    def _compute_deviation(series: np.ndarray, window: int, min_periods: int) -> np.ndarray:
        """Compute z-score deviation from rolling mean."""
        if len(series) < min_periods:
            return np.array([])
        
        result = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            window_data = series[start:i+1]
            if len(window_data) >= min_periods:
                mean = np.mean(window_data)
                std = np.std(window_data, ddof=1)
                if std > 1e-10:
                    result.append((series[i] - mean) / std)
                else:
                    result.append(0.0)
            else:
                result.append(np.nan)
        
        result = np.array(result)
        first_valid = np.where(~np.isnan(result))[0]
        if len(first_valid) > 0:
            return result[first_valid[0]:]
        return np.array([])


# =============================================================================
# RETURNS-SPECIFIC REFLEXIVE DETECTION
# =============================================================================

class ReturnsReflexiveDetector:
    """
    Detects reflexive structure specifically in returns data.
    
    v3's reflexive diagnostics were designed for level data.
    Returns data requires different tests:
    
    1. Fat tails: Excess kurtosis > 0 (normal = 0)
    2. Vol clustering: Autocorrelation of |returns| at lag 1-5
    3. Leverage effect: Negative correlation between r_t and |r_{t+1:t+5}|
    
    These are THE defining characteristics of reflexive financial returns.
    """
    
    @staticmethod
    def detect(returns: np.ndarray) -> Dict[str, float]:
        """
        Detect reflexive structure in returns data.
        
        Returns:
            Dict with scores for each diagnostic (0-1 scale)
        """
        if len(returns) < 50:
            return {"fat_tails": 0.0, "vol_clustering": 0.0, "leverage_effect": 0.0, "composite": 0.0}
        
        # Clean returns
        returns = np.asarray(returns).flatten()
        returns = returns[~np.isnan(returns)]
        returns = np.clip(returns, -10, 10)
        
        if len(returns) < 50:
            return {"fat_tails": 0.0, "vol_clustering": 0.0, "leverage_effect": 0.0, "composite": 0.0}
        
        # 1. Fat tails: excess kurtosis
        fat_tails = ReturnsReflexiveDetector._detect_fat_tails(returns)
        
        # 2. Volatility clustering: autocorrelation of |returns|
        vol_clustering = ReturnsReflexiveDetector._detect_vol_clustering(returns)
        
        # 3. Leverage effect: negative corr between returns and future vol
        leverage_effect = ReturnsReflexiveDetector._detect_leverage_effect(returns)
        
        # Composite score: weighted average
        # Vol clustering is the strongest signal for reflexivity
        composite = (
            0.25 * fat_tails +
            0.50 * vol_clustering +
            0.25 * leverage_effect
        )
        
        return {
            "fat_tails": fat_tails,
            "vol_clustering": vol_clustering,
            "leverage_effect": leverage_effect,
            "composite": composite,
        }
    
    @staticmethod
    def _detect_fat_tails(returns: np.ndarray) -> float:
        """
        Detect fat tails via excess kurtosis.
        
        Normal distribution has kurtosis = 3, excess kurtosis = 0.
        Financial returns typically have excess kurtosis of 3-10+.
        
        Score mapping:
            excess_kurt <= 0: 0.0 (thin or normal tails)
            excess_kurt = 1: 0.3
            excess_kurt = 3: 0.7
            excess_kurt >= 6: 1.0
        """
        from scipy.stats import kurtosis
        
        try:
            # Fisher=True gives excess kurtosis (normal = 0)
            excess_kurt = kurtosis(returns, fisher=True, nan_policy='omit')
            
            if np.isnan(excess_kurt) or excess_kurt <= 0:
                return 0.0
            
            # Map to 0-1 scale
            score = min(1.0, excess_kurt / 6.0)
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _detect_vol_clustering(returns: np.ndarray) -> float:
        """
        Detect volatility clustering via autocorrelation of |returns|.
        
        If volatility clusters, |r_t| should predict |r_{t+1}|.
        We check autocorrelation at lags 1-5.
        
        Score mapping:
            avg_autocorr <= 0: 0.0 (no clustering)
            avg_autocorr = 0.1: 0.5
            avg_autocorr >= 0.2: 1.0
        """
        try:
            abs_returns = np.abs(returns)
            n = len(abs_returns)
            
            # Compute autocorrelation at lags 1-5
            autocorrs = []
            for lag in range(1, min(6, n // 10)):
                if n - lag < 20:
                    continue
                corr = np.corrcoef(abs_returns[:-lag], abs_returns[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(corr)
            
            if not autocorrs:
                return 0.0
            
            avg_autocorr = np.mean(autocorrs)
            
            if avg_autocorr <= 0:
                return 0.0
            
            # Map to 0-1 scale
            score = min(1.0, avg_autocorr / 0.2)
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _detect_leverage_effect(returns: np.ndarray) -> float:
        """
        Detect leverage effect: negative returns → higher future volatility.
        
        This is a key asymmetry in financial markets.
        
        Score mapping:
            corr >= 0: 0.0 (no leverage effect)
            corr = -0.1: 0.5
            corr <= -0.2: 1.0
        """
        try:
            n = len(returns)
            if n < 30:
                return 0.0
            
            # Future volatility: rolling 5-day std
            window = min(5, n // 20)
            if window < 2:
                return 0.0
            
            future_vol = []
            for i in range(n - window):
                future_vol.append(np.std(returns[i+1:i+1+window]))
            
            future_vol = np.array(future_vol)
            current_returns = returns[:len(future_vol)]
            
            # Correlation between current return and future vol
            corr = np.corrcoef(current_returns, future_vol)[0, 1]
            
            if np.isnan(corr) or corr >= 0:
                return 0.0
            
            # Map to 0-1 scale (negative correlation = positive score)
            score = min(1.0, abs(corr) / 0.2)
            return float(score)
        except:
            return 0.0


# =============================================================================
# DEVIATION-SPECIFIC OSCILLATOR DETECTION
# =============================================================================

class DeviationOscillatorDetector:
    """
    Detects oscillator structure specifically in deviation/z-score data.
    
    Mean-reverting series should show:
    1. Negative autocorrelation at lag 1 (overshoots correct)
    2. Zero-crossings frequency above random
    3. Bounded range (doesn't drift)
    """
    
    @staticmethod
    def detect(deviation: np.ndarray) -> Dict[str, float]:
        """
        Detect oscillator structure in deviation data.
        
        Returns:
            Dict with scores for each diagnostic (0-1 scale)
        """
        if len(deviation) < 50:
            return {"mean_reversion": 0.0, "zero_crossings": 0.0, "bounded": 0.0, "composite": 0.0}
        
        deviation = np.asarray(deviation).flatten()
        deviation = deviation[~np.isnan(deviation)]
        
        if len(deviation) < 50:
            return {"mean_reversion": 0.0, "zero_crossings": 0.0, "bounded": 0.0, "composite": 0.0}
        
        # 1. Mean reversion: negative lag-1 autocorrelation
        mean_reversion = DeviationOscillatorDetector._detect_mean_reversion(deviation)
        
        # 2. Zero crossings: frequency of sign changes
        zero_crossings = DeviationOscillatorDetector._detect_zero_crossings(deviation)
        
        # 3. Bounded: stays within reasonable range
        bounded = DeviationOscillatorDetector._detect_bounded(deviation)
        
        composite = (
            0.40 * mean_reversion +
            0.35 * zero_crossings +
            0.25 * bounded
        )
        
        return {
            "mean_reversion": mean_reversion,
            "zero_crossings": zero_crossings,
            "bounded": bounded,
            "composite": composite,
        }
    
    @staticmethod
    def _detect_mean_reversion(deviation: np.ndarray) -> float:
        """Detect mean reversion via negative lag-1 autocorrelation."""
        try:
            n = len(deviation)
            if n < 20:
                return 0.0
            
            autocorr = np.corrcoef(deviation[:-1], deviation[1:])[0, 1]
            
            if np.isnan(autocorr):
                return 0.0
            
            # Mean reversion = negative autocorrelation
            # Random walk has autocorr ≈ 0
            # Strong mean reversion has autocorr ≈ -0.3 to -0.5
            if autocorr >= 0:
                return 0.0
            
            score = min(1.0, abs(autocorr) / 0.3)
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _detect_zero_crossings(deviation: np.ndarray) -> float:
        """Detect zero-crossing frequency."""
        try:
            n = len(deviation)
            if n < 20:
                return 0.0
            
            # Count sign changes
            signs = np.sign(deviation)
            crossings = np.sum(np.abs(np.diff(signs)) > 0)
            crossing_rate = crossings / (n - 1)
            
            # Random walk: ~0.5 crossing rate
            # Strong oscillator: higher crossing rate
            # Pure trend: low crossing rate
            
            if crossing_rate < 0.3:
                return 0.0
            
            # Score based on how much above random walk rate
            score = min(1.0, (crossing_rate - 0.3) / 0.3)
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _detect_bounded(deviation: np.ndarray) -> float:
        """Detect bounded range (no drift)."""
        try:
            # Check if range is bounded (no extreme outliers)
            q01 = np.percentile(deviation, 1)
            q99 = np.percentile(deviation, 99)
            
            range_width = q99 - q01
            
            # For z-scores, typical range is -3 to +3
            # Wider range suggests drift or fat tails
            if range_width <= 4:
                return 1.0
            elif range_width <= 6:
                return 0.7
            elif range_width <= 8:
                return 0.4
            else:
                return 0.2
        except:
            return 0.5


# =============================================================================
# VIEW GEOMETRY RESULT
# =============================================================================

@dataclass
class ViewGeometryResult:
    """Geometry result for a single view."""
    view_type: ViewType
    dominant_geometry: str
    confidence: float
    scores: Dict[str, float]  # latent, oscillator, reflexive, noise
    n_samples: int
    is_hybrid: bool = False
    
    def __str__(self):
        return f"{self.view_type.value}: {self.dominant_geometry} ({self.confidence:.2f})"


@dataclass
class MultiViewGeometryResult:
    """Complete multi-view geometry analysis."""
    indicator_id: str
    views: Dict[ViewType, ViewGeometryResult]
    consensus_geometry: str
    consensus_confidence: float
    rationale: str
    disagreement_score: float  # 0 = full agreement, 1 = complete disagreement
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"\n{self.indicator_id}:",
        ]
        for view_type, result in self.views.items():
            lines.append(f"  {view_type.value}_geometry: {result.dominant_geometry} ({result.confidence:.2f})")
        lines.append(f"  consensus: {self.consensus_geometry}")
        lines.append(f"  confidence: {self.consensus_confidence:.2f}")
        lines.append(f"  disagreement: {self.disagreement_score:.2f}")
        lines.append(f"  rationale: {self.rationale}")
        return "\n".join(lines)


# =============================================================================
# MULTI-VIEW GEOMETRY AGENT
# =============================================================================

class MultiViewGeometryAgent:
    """
    Runs geometry signature analysis on multiple views of the same series.
    
    View-Aware Scoring:
        Different views require different interpretation of v3 geometry scores.
        
        LEVEL view: All geometries compete fairly (v3 designed for this)
        RETURNS view: Stationary by construction - latent_flow inapplicable
        VOLATILITY view: Second-order structure - latent_flow less relevant
        DEVIATION view: Mean-reverting by construction - oscillator boosted
    
    Usage:
        from prism.agents.agent_geometry_signature import GeometrySignatureAgent
        
        agent = MultiViewGeometryAgent(base_agent=GeometrySignatureAgent())
        result = agent.analyze(series, indicator_id="SPY")
        print(result.summary())
    """
    
    # Default view configurations by data domain
    FINANCE_VIEWS = [ViewType.LEVEL, ViewType.RETURNS, ViewType.VOLATILITY]
    SPREAD_VIEWS = [ViewType.LEVEL, ViewType.DEVIATION]
    CLIMATE_VIEWS = [ViewType.LEVEL, ViewType.ANOMALY, ViewType.RETURNS]
    EPI_VIEWS = [ViewType.LEVEL, ViewType.INCIDENCE, ViewType.RETURNS]
    
    # View-aware score adjustments
    # These reflect structural properties of each transformation
    VIEW_SCORE_ADJUSTMENTS = {
        ViewType.LEVEL: {
            # Level view: v3 designed for this, all compete fairly
            "latent_flow": 1.0,
            "coupled_oscillator": 1.0,
            "reflexive_stochastic": 1.0,
            "pure_noise": 1.0,
        },
        ViewType.RETURNS: {
            # Returns are stationary by construction
            # Latent_flow diagnostics (saturation, S-curve) are structurally inapplicable
            # Oscillator patterns are harder to detect in differenced data
            # Reflexive (fat tails, vol clustering) and noise compete fairly
            "latent_flow": 0.2,           # Strongly demote - inapplicable
            "coupled_oscillator": 0.5,    # Demote - differencing obscures cycles
            "reflexive_stochastic": 1.2,  # Boost slightly - this is where reflexivity shows
            "pure_noise": 1.0,            # Fair baseline
        },
        ViewType.VOLATILITY: {
            # Volatility is second-order structure
            # Latent_flow less relevant (vol doesn't "accumulate" the same way)
            # Reflexive patterns (clustering, persistence) are primary signal
            "latent_flow": 0.4,           # Demote
            "coupled_oscillator": 0.7,    # Slightly demote
            "reflexive_stochastic": 1.3,  # Boost - vol clustering is reflexive signature
            "pure_noise": 1.0,            # Fair baseline
        },
        ViewType.DEVIATION: {
            # Deviation from MA is mean-reverting by construction
            # Oscillator patterns should dominate
            "latent_flow": 0.3,           # Demote - can't accumulate around mean
            "coupled_oscillator": 1.4,    # Boost - mean reversion is oscillatory
            "reflexive_stochastic": 0.8,  # Slight demote
            "pure_noise": 1.0,            # Fair baseline
        },
        ViewType.ANOMALY: {
            # Climate anomalies - similar to deviation
            "latent_flow": 0.5,
            "coupled_oscillator": 1.2,
            "reflexive_stochastic": 0.9,
            "pure_noise": 1.0,
        },
        ViewType.INCIDENCE: {
            # Epidemiology incidence (new cases)
            # Latent_flow still applicable (epidemic curves)
            # But reflexive patterns also relevant (behavioral feedback)
            "latent_flow": 1.0,
            "coupled_oscillator": 0.8,
            "reflexive_stochastic": 1.1,
            "pure_noise": 1.0,
        },
    }
    
    def __init__(self, base_agent=None, verbose: bool = False):
        """
        Initialize multi-view agent.

        Args:
            base_agent: GeometrySignatureAgent instance (v3)
            verbose: Print detailed output

        Parameters can be configured via config/geometry.yaml.
        """
        self.base_agent = base_agent
        self.verbose = verbose
        self.transformer = ViewTransformer()

        # Load config overrides for VIEW_SCORE_ADJUSTMENTS
        self._load_config_overrides()

    def _load_config_overrides(self) -> None:
        """Load configuration overrides from config/geometry.yaml."""
        try:
            cfg = get_geometry_config()

            # Override VIEW_SCORE_ADJUSTMENTS if present in config
            if "view_score_adjustments" in cfg:
                config_adjustments = cfg["view_score_adjustments"]

                # Map string view names to ViewType enums
                view_name_map = {
                    "level": ViewType.LEVEL,
                    "returns": ViewType.RETURNS,
                    "volatility": ViewType.VOLATILITY,
                    "deviation": ViewType.DEVIATION,
                    "anomaly": ViewType.ANOMALY,
                    "incidence": ViewType.INCIDENCE,
                }

                for view_name, adjustments in config_adjustments.items():
                    view_type = view_name_map.get(view_name.lower())
                    if view_type and adjustments:
                        self.VIEW_SCORE_ADJUSTMENTS[view_type] = adjustments

        except FileNotFoundError:
            logger.debug("geometry.yaml not found, using defaults")
    
    def analyze(
        self,
        series: np.ndarray,
        indicator_id: str = "unknown",
        views: Optional[List[ViewType]] = None,
        view_configs: Optional[Dict[ViewType, ViewConfig]] = None,
    ) -> MultiViewGeometryResult:
        """
        Analyze series across multiple views.
        
        Args:
            series: Raw time series
            indicator_id: Indicator identifier
            views: List of views to analyze (defaults to FINANCE_VIEWS)
            view_configs: Optional per-view configurations
            
        Returns:
            MultiViewGeometryResult with all views and consensus
        """
        if views is None:
            views = self.FINANCE_VIEWS
        
        if view_configs is None:
            view_configs = {}
        
        # Ensure base agent exists
        if self.base_agent is None:
            # Lazy import to avoid circular dependency
            from prism.agents.agent_geometry_signature import GeometrySignatureAgent
            self.base_agent = GeometrySignatureAgent(verbose=False)
        
        # Analyze each view
        view_results = {}
        for view_type in views:
            config = view_configs.get(view_type, ViewConfig(view_type=view_type))
            
            # Transform series
            transformed = self.transformer.transform(series, view_type, config)
            
            if len(transformed) < 50:
                if self.verbose:
                    print(f"  {view_type.value}: Skipped (insufficient data after transform)")
                continue
            
            # Run v3 geometry analysis
            profile = self.base_agent.analyze(transformed)
            
            # Apply view-aware score adjustments
            # v3 is designed for level data; other views need rebalancing
            adjustments = self.VIEW_SCORE_ADJUSTMENTS.get(view_type, {})
            
            adjusted_scores = {
                "latent_flow": profile.latent_flow_score * adjustments.get("latent_flow", 1.0),
                "coupled_oscillator": profile.oscillator_score * adjustments.get("coupled_oscillator", 1.0),
                "reflexive_stochastic": profile.reflexive_score * adjustments.get("reflexive_stochastic", 1.0),
                "pure_noise": profile.noise_score * adjustments.get("pure_noise", 1.0),
            }
            
            # Apply view-specific detectors that v3 can't handle
            if view_type == ViewType.RETURNS:
                # v3's reflexive diagnostics don't work well on returns
                # Use our returns-specific detector
                reflex_result = ReturnsReflexiveDetector.detect(transformed)
                
                # Blend v3's reflexive score with our detector
                # Our detector is more reliable for returns data
                v3_reflex = adjusted_scores["reflexive_stochastic"]
                our_reflex = reflex_result["composite"]
                
                # Weight our detector heavily (0.7) since v3 wasn't designed for returns
                blended_reflex = 0.3 * v3_reflex + 0.7 * our_reflex
                adjusted_scores["reflexive_stochastic"] = blended_reflex
                
                # If we detect strong reflexivity, reduce noise score
                if our_reflex > 0.5:
                    adjusted_scores["pure_noise"] *= (1.0 - our_reflex * 0.5)
                
                if self.verbose:
                    print(f"    [returns detector: fat_tails={reflex_result['fat_tails']:.2f}, "
                          f"vol_clust={reflex_result['vol_clustering']:.2f}, "
                          f"leverage={reflex_result['leverage_effect']:.2f}]")
            
            elif view_type == ViewType.DEVIATION:
                # v3's oscillator diagnostics don't work well on deviation
                # Use our deviation-specific detector
                osc_result = DeviationOscillatorDetector.detect(transformed)
                
                # Blend v3's oscillator score with our detector
                v3_osc = adjusted_scores["coupled_oscillator"]
                our_osc = osc_result["composite"]
                
                # Weight our detector heavily (0.7)
                blended_osc = 0.3 * v3_osc + 0.7 * our_osc
                adjusted_scores["coupled_oscillator"] = blended_osc
                
                # If we detect strong oscillation, reduce noise and latent scores
                if our_osc > 0.5:
                    adjusted_scores["pure_noise"] *= (1.0 - our_osc * 0.4)
                    adjusted_scores["latent_flow"] *= (1.0 - our_osc * 0.3)
                
                if self.verbose:
                    print(f"    [deviation detector: mean_rev={osc_result['mean_reversion']:.2f}, "
                          f"crossings={osc_result['zero_crossings']:.2f}, "
                          f"bounded={osc_result['bounded']:.2f}]")
            
            # Renormalize so max stays <= 1.0
            max_score = max(adjusted_scores.values())
            if max_score > 1.0:
                for k in adjusted_scores:
                    adjusted_scores[k] /= max_score
            
            # Determine adjusted dominant geometry
            adjusted_dominant = max(adjusted_scores, key=adjusted_scores.get)
            adjusted_confidence = adjusted_scores[adjusted_dominant]
            
            # Check for hybrid (multiple scores >= 0.5)
            high_scores = [g for g, s in adjusted_scores.items() 
                          if s >= 0.5 and g != "pure_noise"]
            is_hybrid = len(high_scores) > 1
            
            view_results[view_type] = ViewGeometryResult(
                view_type=view_type,
                dominant_geometry=adjusted_dominant,
                confidence=adjusted_confidence,
                scores=adjusted_scores,
                n_samples=len(transformed),
                is_hybrid=is_hybrid,
            )
            
            if self.verbose:
                raw_dom = profile.dominant_geometry.value if profile.dominant_geometry else "unknown"
                print(f"  {view_type.value}: {raw_dom} → {adjusted_dominant} ({adjusted_confidence:.2f})")
        
        # Arbitrate
        consensus, confidence, rationale, disagreement = self._arbitrate(view_results, indicator_id)
        
        return MultiViewGeometryResult(
            indicator_id=indicator_id,
            views=view_results,
            consensus_geometry=consensus,
            consensus_confidence=confidence,
            rationale=rationale,
            disagreement_score=disagreement,
        )
    
    def _arbitrate(
        self,
        view_results: Dict[ViewType, ViewGeometryResult],
        indicator_id: str
    ) -> Tuple[str, float, str, float]:
        """
        Arbitrate between views to determine consensus geometry.
        
        Returns:
            (consensus_geometry, confidence, rationale, disagreement_score)
        """
        if not view_results:
            return "unknown", 0.0, "No valid views", 1.0
        
        # Collect all geometries and their confidences
        geometry_votes: Dict[str, List[Tuple[ViewType, float]]] = {}
        for view_type, result in view_results.items():
            geom = result.dominant_geometry
            if geom not in geometry_votes:
                geometry_votes[geom] = []
            geometry_votes[geom].append((view_type, result.confidence))
        
        # Check for unanimous agreement
        if len(geometry_votes) == 1:
            consensus = list(geometry_votes.keys())[0]
            avg_conf = np.mean([v[1] for v in geometry_votes[consensus]])
            return consensus, avg_conf, "unanimous across all views", 0.0
        
        # Calculate weighted votes
        weighted_scores: Dict[str, float] = {}
        view_weights = {
            ViewType.LEVEL: 0.3,
            ViewType.RETURNS: 0.35,
            ViewType.VOLATILITY: 0.25,
            ViewType.DEVIATION: 0.3,
            ViewType.ANOMALY: 0.3,
            ViewType.INCIDENCE: 0.35,
        }
        
        for geom, votes in geometry_votes.items():
            score = 0.0
            for view_type, confidence in votes:
                weight = view_weights.get(view_type, 0.25)
                score += weight * confidence
            weighted_scores[geom] = score
        
        # Determine winner
        consensus = max(weighted_scores, key=weighted_scores.get)
        consensus_score = weighted_scores[consensus]
        
        # Calculate disagreement
        total_score = sum(weighted_scores.values())
        disagreement = 1.0 - (consensus_score / total_score) if total_score > 0 else 1.0
        
        # Build rationale
        rationale = self._build_rationale(view_results, consensus, geometry_votes)
        
        # Confidence is consensus score normalized, penalized by disagreement
        confidence = min(1.0, consensus_score) * (1 - disagreement * 0.3)
        
        return consensus, confidence, rationale, disagreement
    
    def _build_rationale(
        self,
        view_results: Dict[ViewType, ViewGeometryResult],
        consensus: str,
        geometry_votes: Dict[str, List[Tuple[ViewType, float]]]
    ) -> str:
        """Build human-readable rationale for consensus decision."""
        
        # Check for level vs dynamics disagreement
        level_result = view_results.get(ViewType.LEVEL)
        returns_result = view_results.get(ViewType.RETURNS)
        vol_result = view_results.get(ViewType.VOLATILITY)
        
        if level_result and returns_result:
            level_geom = level_result.dominant_geometry
            returns_geom = returns_result.dominant_geometry
            
            if level_geom == "latent_flow" and returns_geom == "reflexive_stochastic":
                return "reflexivity emerges in dynamics; trend in level is accumulation"
            
            if level_geom == "latent_flow" and returns_geom == "coupled_oscillator":
                return "oscillation in dynamics; trend masks cyclicality"
        
        if vol_result:
            vol_geom = vol_result.dominant_geometry
            if vol_geom == "reflexive_stochastic" and consensus == "reflexive_stochastic":
                return "reflexivity confirmed in second-order structure (volatility)"
        
        # Count supporting views
        supporting_views = [v.value for v, r in view_results.items() if r.dominant_geometry == consensus]
        if len(supporting_views) > 1:
            return f"majority agreement: {', '.join(supporting_views)}"
        
        return f"dominant in weighted voting; disagreement among views"


# =============================================================================
# ARBITRATION AGENT (Reconciles multiple indicators)
# =============================================================================

class ArbitrationAgent:
    """
    Reconciles geometry classifications across multiple indicators.
    
    Provides:
    - Cross-indicator consistency checks
    - Domain-level geometry summaries
    - Confidence propagation
    - Disagreement explanations
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[MultiViewGeometryResult] = []
    
    def add_result(self, result: MultiViewGeometryResult):
        """Add a multi-view result for arbitration."""
        self.results.append(result)
    
    def analyze_cohort(
        self,
        results: List[MultiViewGeometryResult]
    ) -> Dict:
        """
        Analyze a cohort of indicators.
        
        Returns:
            Summary dict with geometry distribution, confidence stats, etc.
        """
        self.results = results
        
        # Geometry distribution
        geometry_counts = {}
        for r in results:
            geom = r.consensus_geometry
            geometry_counts[geom] = geometry_counts.get(geom, 0) + 1
        
        # Confidence statistics
        confidences = [r.consensus_confidence for r in results]
        disagreements = [r.disagreement_score for r in results]
        
        # High-confidence classifications
        high_conf = [r for r in results if r.consensus_confidence > 0.7]
        
        # High-disagreement (uncertain) classifications
        high_disagree = [r for r in results if r.disagreement_score > 0.3]
        
        return {
            "n_indicators": len(results),
            "geometry_distribution": geometry_counts,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "avg_disagreement": np.mean(disagreements) if disagreements else 0,
            "high_confidence_count": len(high_conf),
            "high_disagreement_count": len(high_disagree),
            "high_disagreement_indicators": [r.indicator_id for r in high_disagree],
        }
    
    def explain_disagreements(self) -> str:
        """Generate explanation for high-disagreement indicators."""
        lines = ["HIGH DISAGREEMENT INDICATORS", "=" * 50]
        
        for r in self.results:
            if r.disagreement_score > 0.3:
                lines.append(f"\n{r.indicator_id} (disagreement: {r.disagreement_score:.2f}):")
                for view_type, view_result in r.views.items():
                    lines.append(f"  {view_type.value}: {view_result.dominant_geometry} ({view_result.confidence:.2f})")
                lines.append(f"  consensus: {r.consensus_geometry}")
                lines.append(f"  rationale: {r.rationale}")
        
        return "\n".join(lines)
    
    def generate_report(self) -> str:
        """Generate full arbitration report."""
        summary = self.analyze_cohort(self.results)
        
        lines = [
            "=" * 70,
            "MULTI-VIEW GEOMETRY ARBITRATION REPORT",
            "=" * 70,
            "",
            f"Total indicators: {summary['n_indicators']}",
            f"Average confidence: {summary['avg_confidence']:.2f}",
            f"Average disagreement: {summary['avg_disagreement']:.2f}",
            "",
            "GEOMETRY DISTRIBUTION:",
        ]
        
        for geom, count in sorted(summary['geometry_distribution'].items(), key=lambda x: -x[1]):
            pct = 100 * count / summary['n_indicators']
            lines.append(f"  {geom}: {count} ({pct:.0f}%)")
        
        lines.append("")
        lines.append(f"High confidence (>0.7): {summary['high_confidence_count']}")
        lines.append(f"High disagreement (>0.3): {summary['high_disagreement_count']}")
        
        if summary['high_disagreement_indicators']:
            lines.append(f"  → {', '.join(summary['high_disagreement_indicators'][:10])}")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_indicator_multiview(
    series: np.ndarray,
    indicator_id: str = "unknown",
    is_spread: bool = False,
    verbose: bool = False,
) -> MultiViewGeometryResult:
    """
    Convenience function for multi-view analysis.
    
    Args:
        series: Raw time series
        indicator_id: Indicator name
        is_spread: If True, use spread views (level + deviation)
        verbose: Print progress
    """
    agent = MultiViewGeometryAgent(verbose=verbose)
    
    if is_spread:
        views = MultiViewGeometryAgent.SPREAD_VIEWS
    else:
        views = MultiViewGeometryAgent.FINANCE_VIEWS
    
    return agent.analyze(series, indicator_id=indicator_id, views=views)


# =============================================================================
# TEST HARNESS
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    # Import v3 geometry agent
    from prism.agents.agent_geometry_signature import GeometrySignatureAgent
    
    print("=" * 70)
    print("Multi-View Geometry Analysis - View-Specific Detectors Demo")
    print("=" * 70)
    print()
    print("View-Specific Detectors:")
    print("  RETURNS: ReturnsReflexiveDetector (fat_tails, vol_clustering, leverage)")
    print("  DEVIATION: DeviationOscillatorDetector (mean_reversion, zero_crossings)")
    print()
    print("These detect patterns that v3 (designed for level data) cannot see.")
    print()
    
    np.random.seed(42)
    n = 500
    t = np.linspace(0, 10, n)
    
    # Test the returns detector directly
    print("=" * 70)
    print("TEST: ReturnsReflexiveDetector on synthetic data")
    print("=" * 70)
    
    # GARCH returns (should detect reflexive)
    garch_returns = np.zeros(n)
    sigma = np.zeros(n)
    sigma[0] = 0.02
    for i in range(1, n):
        sigma[i] = 0.0005 + 0.88 * sigma[i-1] + 0.10 * garch_returns[i-1]**2
        garch_returns[i] = np.random.normal(0, np.sqrt(sigma[i]))
    
    garch_result = ReturnsReflexiveDetector.detect(garch_returns)
    print(f"\nGARCH returns:")
    print(f"  fat_tails:     {garch_result['fat_tails']:.3f}")
    print(f"  vol_clustering: {garch_result['vol_clustering']:.3f}")
    print(f"  leverage_effect: {garch_result['leverage_effect']:.3f}")
    print(f"  COMPOSITE:     {garch_result['composite']:.3f}")
    
    # White noise returns (should NOT detect reflexive)
    noise_returns = np.random.randn(n) * 0.01
    noise_result = ReturnsReflexiveDetector.detect(noise_returns)
    print(f"\nWhite noise returns:")
    print(f"  fat_tails:     {noise_result['fat_tails']:.3f}")
    print(f"  vol_clustering: {noise_result['vol_clustering']:.3f}")
    print(f"  leverage_effect: {noise_result['leverage_effect']:.3f}")
    print(f"  COMPOSITE:     {noise_result['composite']:.3f}")
    
    # Test the deviation detector directly
    print("\n" + "=" * 70)
    print("TEST: DeviationOscillatorDetector on synthetic data")
    print("=" * 70)
    
    # Mean-reverting spread (should detect oscillator)
    spread = 0.5 * np.sin(0.5 * t) + 0.3 * np.sin(0.2 * t)
    spread += np.random.randn(n) * 0.1
    spread_zscore = (spread - np.mean(spread)) / np.std(spread)
    
    spread_result = DeviationOscillatorDetector.detect(spread_zscore)
    print(f"\nMean-reverting spread (z-score):")
    print(f"  mean_reversion: {spread_result['mean_reversion']:.3f}")
    print(f"  zero_crossings: {spread_result['zero_crossings']:.3f}")
    print(f"  bounded:        {spread_result['bounded']:.3f}")
    print(f"  COMPOSITE:      {spread_result['composite']:.3f}")
    
    # Random walk (should NOT detect oscillator)
    random_walk = np.cumsum(np.random.randn(n) * 0.1)
    rw_deviation = random_walk - pd.Series(random_walk).rolling(21).mean().values
    rw_deviation = rw_deviation[~np.isnan(rw_deviation)]
    
    rw_result = DeviationOscillatorDetector.detect(rw_deviation)
    print(f"\nRandom walk deviation:")
    print(f"  mean_reversion: {rw_result['mean_reversion']:.3f}")
    print(f"  zero_crossings: {rw_result['zero_crossings']:.3f}")
    print(f"  bounded:        {rw_result['bounded']:.3f}")
    print(f"  COMPOSITE:      {rw_result['composite']:.3f}")
    
    # Full multi-view tests
    print("\n" + "=" * 70)
    print("TEST: Full Multi-View Analysis")
    print("=" * 70)
    
    # Test 1: Simulated Equity (should show reflexive in returns/vol views)
    print("\n>>> Simulated Equity (GARCH-like)")
    print("    Expected: level=latent_flow, returns=REFLEXIVE, vol=REFLEXIVE")
    
    garch_level = 100 * np.cumprod(1 + garch_returns)
    
    agent = MultiViewGeometryAgent(verbose=True)
    result = agent.analyze(garch_level, indicator_id="SIM_EQUITY")
    print(result.summary())
    
    # Test 2: Yield Spread (should show oscillator in deviation view)
    print("\n>>> Simulated Yield Spread")
    print("    Expected: level=latent/mixed, deviation=OSCILLATOR")
    
    result = agent.analyze(spread, indicator_id="SIM_SPREAD",
                          views=[ViewType.LEVEL, ViewType.DEVIATION])
    print(result.summary())
    
    # Test 3: White Noise (should stay noise everywhere)
    print("\n>>> White Noise")
    print("    Expected: all views → pure_noise")
    
    noise_level = np.cumsum(np.random.randn(n) * 0.01)  # Random walk
    result = agent.analyze(noise_level, indicator_id="SIM_NOISE")
    print(result.summary())
    
    print("\n" + "=" * 70)
    print("The view-specific detectors identify patterns v3 cannot see in")
    print("transformed data, while respecting v3's honest measurements.")
    print("=" * 70)
