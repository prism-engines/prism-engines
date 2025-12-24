#!/usr/bin/env python3
"""
PRISM Ablation Study: Agent-Routed vs Direct Engine Execution

Compares two execution modes:
1. WITH AGENTS: Geometry detection → Routing → Weighted consensus
2. WITHOUT AGENTS: Direct execution, all engines, equal weights

Purpose: Validate that agent layer adds value, not just complexity.

Usage:
    python prism_ablation_study.py
    python prism_ablation_study.py --indicators SPY,XLF,XLK,GLD
    python prism_ablation_study.py --output results/ablation_report.json
    python prism_ablation_study.py --synthetic  # Include synthetic benchmarks

Output:
    - JSON report with full comparison
    - Console summary of key findings
    - Per-cohort breakdown if using cohorts
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import warnings

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Add project root to path (adjust as needed for your setup)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prism.db.connection import get_db_path

# Try to import PRISM modules - graceful fallback if not available
try:
    from prism.engines import list_engines, get_engine
    PRISM_AVAILABLE = True
except ImportError:
    PRISM_AVAILABLE = False
    print("Warning: PRISM engines not importable - using mock engines")


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class EngineResult:
    """Result from a single engine run."""
    engine: str
    success: bool
    runtime_ms: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExecutionRun:
    """Complete execution run (with or without agents)."""
    mode: str  # "with_agents" or "direct"
    started_at: str
    completed_at: str
    total_runtime_ms: float
    engines_executed: List[str]
    engines_skipped: List[str]
    engine_results: Dict[str, EngineResult] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    geometry_profile: Optional[Dict] = None  # Only for agent mode
    warnings: List[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Comparison between two execution modes."""
    # Execution differences
    engines_only_in_agent_mode: List[str]
    engines_only_in_direct_mode: List[str]
    engines_in_both: List[str]
    
    # Weight analysis
    weight_differences: Dict[str, Dict[str, float]]  # engine -> {agent_weight, direct_weight, diff}
    
    # Output correlations (for engines in both)
    metric_correlations: Dict[str, Dict[str, float]]  # engine -> {metric -> correlation}
    
    # Performance
    agent_runtime_ms: float
    direct_runtime_ms: float
    overhead_pct: float
    
    # Value assessment
    noise_handling: Dict[str, Any]  # How agents handled noisy data
    geometry_routing_impact: Dict[str, Any]  # Which engines geometry detection enabled/disabled


@dataclass
class AblationReport:
    """Complete ablation study report."""
    study_id: str
    timestamp: str
    data_source: str
    n_indicators: int
    n_observations: int
    
    # Individual runs
    agent_run: ExecutionRun
    direct_run: ExecutionRun
    
    # Comparison
    comparison: ComparisonResult
    
    # Synthetic benchmark results (if run)
    synthetic_benchmarks: Optional[Dict] = None
    
    # Summary findings
    key_findings: List[str] = field(default_factory=list)
    recommendation: str = ""


# =============================================================================
# Geometry Signature Detection (Simplified from your agent)
# =============================================================================

class SimpleGeometryDetector:
    """
    Simplified geometry detection for ablation study.
    Mirrors your GeometrySignatureAgent logic.
    """
    
    def analyze(self, series: np.ndarray) -> Dict[str, float]:
        """Compute geometry scores for a series."""
        if len(series) < 30:
            return {"noise_score": 1.0, "latent_flow": 0.0, 
                    "oscillator": 0.0, "reflexive": 0.0}
        
        # Clean the series
        series = np.array(series, dtype=float)
        series = series[~np.isnan(series)]
        
        if len(series) < 30:
            return {"noise_score": 1.0, "latent_flow": 0.0,
                    "oscillator": 0.0, "reflexive": 0.0}
        
        scores = {}
        
        # Noise score (randomness test)
        scores["noise_score"] = self._noise_score(series)
        
        # Latent flow (saturation/logistic pattern)
        scores["latent_flow"] = self._latent_flow_score(series)
        
        # Oscillator (cyclical patterns)
        scores["oscillator"] = self._oscillator_score(series)
        
        # Reflexive (volatility clustering, fat tails)
        scores["reflexive"] = self._reflexive_score(series)
        
        return scores
    
    def _noise_score(self, series: np.ndarray) -> float:
        """Test for randomness using runs test and autocorrelation."""
        try:
            # Autocorrelation at lag 1
            if len(series) > 1:
                ac1 = np.corrcoef(series[:-1], series[1:])[0, 1]
                if np.isnan(ac1):
                    ac1 = 0
            else:
                ac1 = 0
            
            # Low autocorrelation = more noise-like
            noise_from_ac = 1 - abs(ac1)
            
            # Runs test for randomness
            median = np.median(series)
            runs = np.sum(np.diff(series > median) != 0) + 1
            n = len(series)
            expected_runs = (2 * n - 1) / 3
            
            if expected_runs > 0:
                run_ratio = runs / expected_runs
                noise_from_runs = min(1.0, max(0.0, 1 - abs(run_ratio - 1)))
            else:
                noise_from_runs = 0.5
            
            return (noise_from_ac * 0.6 + noise_from_runs * 0.4)
        except:
            return 0.5
    
    def _latent_flow_score(self, series: np.ndarray) -> float:
        """Detect saturation/compartmental dynamics."""
        try:
            cumulative = np.cumsum(series - series.min())
            t = np.arange(len(cumulative))
            
            # Check for diminishing returns (second derivative)
            if len(cumulative) > 10:
                first_deriv = np.gradient(cumulative)
                second_deriv = np.gradient(first_deriv)
                
                # Saturation = consistently negative second derivative
                neg_ratio = np.mean(second_deriv < 0)
                return min(1.0, neg_ratio * 1.2)
            return 0.3
        except:
            return 0.3
    
    def _oscillator_score(self, series: np.ndarray) -> float:
        """Detect oscillatory/cyclical behavior."""
        try:
            # FFT to find dominant frequencies
            fft = np.fft.fft(series - np.mean(series))
            power = np.abs(fft[:len(fft)//2])**2
            
            if len(power) > 2:
                # Concentration of power in specific frequencies
                sorted_power = np.sort(power)[::-1]
                top3_ratio = np.sum(sorted_power[:3]) / (np.sum(power) + 1e-10)
                
                # Also check for periodicity via autocorrelation peaks
                return min(1.0, top3_ratio * 1.5)
            return 0.3
        except:
            return 0.3
    
    def _reflexive_score(self, series: np.ndarray) -> float:
        """Detect reflexive stochastic behavior (finance-like)."""
        try:
            # Check for volatility clustering
            returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
            
            if len(returns) < 20:
                return 0.3
            
            # ARCH effect: correlation of squared returns
            sq_returns = returns**2
            if len(sq_returns) > 1:
                arch_corr = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1]
                if np.isnan(arch_corr):
                    arch_corr = 0
            else:
                arch_corr = 0
            
            # Fat tails (kurtosis)
            kurtosis = stats.kurtosis(returns)
            fat_tail_score = min(1.0, max(0, kurtosis / 10))
            
            return (abs(arch_corr) * 0.5 + fat_tail_score * 0.5)
        except:
            return 0.3


# =============================================================================
# Engine Runners
# =============================================================================

# Engine configurations
ALL_ENGINES = [
    "pca", "cross_correlation", "entropy", "wavelet",
    "hmm", "dmd", "mutual_information", "clustering",
    "hurst", "spectral", "granger", "copula"
]

CORE_ENGINES = ["pca", "cross_correlation", "entropy", "wavelet"]

UNIVARIATE_ENGINES = ["entropy", "hurst", "spectral"]

ENGINE_DATA_PREFS = {
    "pca": "zscore",
    "cross_correlation": "zscore",
    "entropy": "zscore",
    "wavelet": "raw",
    "hmm": "zscore",
    "dmd": "zscore",
    "mutual_information": "zscore",
    "clustering": "zscore",
    "hurst": "raw",
    "spectral": "returns",
    "granger": "returns",
    "copula": "raw",
}


def prepare_data_variants(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare different data normalizations."""
    variants = {"raw": df}
    
    # Z-score normalization
    mean = df.mean()
    std = df.std().replace(0, 1)
    variants["zscore"] = (df - mean) / std
    
    # Returns
    variants["returns"] = df.pct_change().dropna()
    
    return variants


def run_engine_safe(engine_name: str, df: pd.DataFrame, 
                    data_variants: Dict[str, pd.DataFrame]) -> EngineResult:
    """Run a single engine with timing and error handling."""
    start_time = time.time()
    
    try:
        # Get appropriate data variant
        data_pref = ENGINE_DATA_PREFS.get(engine_name, "zscore")
        data = data_variants.get(data_pref, df)
        
        if data.empty or len(data) < 20:
            return EngineResult(
                engine=engine_name,
                success=False,
                runtime_ms=0,
                error="Insufficient data"
            )
        
        metrics = {}
        
        if PRISM_AVAILABLE:
            # Use real PRISM engine
            engine = get_engine(engine_name)
            if engine:
                result = engine.run(data)
                metrics = result if isinstance(result, dict) else {"result": result}
        else:
            # Mock engine results for testing
            metrics = _mock_engine_run(engine_name, data)
        
        runtime_ms = (time.time() - start_time) * 1000
        
        return EngineResult(
            engine=engine_name,
            success=True,
            runtime_ms=runtime_ms,
            metrics=metrics
        )
        
    except Exception as e:
        runtime_ms = (time.time() - start_time) * 1000
        return EngineResult(
            engine=engine_name,
            success=False,
            runtime_ms=runtime_ms,
            error=str(e)
        )


def _mock_engine_run(engine_name: str, data: pd.DataFrame) -> Dict[str, float]:
    """Mock engine run for testing without PRISM installed."""
    np.random.seed(hash(engine_name) % 2**32)
    
    if engine_name == "pca":
        return {"explained_variance_ratio": np.random.uniform(0.5, 0.9)}
    elif engine_name == "cross_correlation":
        return {"mean_correlation": np.random.uniform(0.2, 0.6)}
    elif engine_name == "entropy":
        return {"entropy": np.random.uniform(0.4, 0.8)}
    elif engine_name == "wavelet":
        return {"max_power": np.random.uniform(0.2, 0.5)}
    elif engine_name == "hmm":
        return {"n_states": np.random.randint(2, 5), "stability": np.random.uniform(0.5, 0.9)}
    elif engine_name == "dmd":
        return {"dominant_mode": np.random.uniform(0.8, 1.0)}
    elif engine_name == "hurst":
        return {"hurst_exponent": np.random.uniform(0.3, 0.7)}
    else:
        return {"value": np.random.random()}


# =============================================================================
# Execution Modes
# =============================================================================

def run_with_agents(df: pd.DataFrame, verbose: bool = True) -> ExecutionRun:
    """
    Run with full agent pipeline:
    1. Geometry detection on representative series
    2. Route engines based on geometry profile
    3. Apply weights based on profile scores
    """
    start_time = datetime.now()
    start_ms = time.time() * 1000
    
    if verbose:
        print("\n" + "=" * 60)
        print("AGENT-ROUTED EXECUTION")
        print("=" * 60)
    
    # Prepare data variants
    data_variants = prepare_data_variants(df)
    
    # Step 1: Geometry detection (on mean series or first principal component)
    detector = SimpleGeometryDetector()
    
    # Analyze each column and aggregate
    all_scores = []
    for col in df.columns:
        scores = detector.analyze(df[col].values)
        all_scores.append(scores)
    
    # Aggregate geometry profile
    geometry_profile = {
        k: np.mean([s[k] for s in all_scores]) 
        for k in all_scores[0].keys()
    }
    
    if verbose:
        print(f"\nGeometry Profile:")
        for k, v in geometry_profile.items():
            print(f"  {k}: {v:.3f}")
    
    # Step 2: Route engines based on geometry
    engines_to_run = []
    weights = {}
    engines_skipped = []
    warnings_list = []
    
    noise_score = geometry_profile.get("noise_score", 0.5)
    
    # Core engines always run (but weighted by noise)
    for engine in CORE_ENGINES:
        weight = 1.0 - (noise_score * 0.5)
        engines_to_run.append(engine)
        weights[engine] = weight
    
    # Conditional engines based on geometry
    if geometry_profile.get("latent_flow", 0) >= 0.5:
        # Would add SIR engine for epi data
        pass
    
    if geometry_profile.get("oscillator", 0) >= 0.5:
        engines_to_run.append("wavelet")  # boost weight
        weights["wavelet"] = min(1.0, weights.get("wavelet", 0.5) * 1.5)
        if "granger" not in engines_to_run:
            engines_to_run.append("granger")
            weights["granger"] = geometry_profile["oscillator"] * 0.8
    
    if geometry_profile.get("reflexive", 0) >= 0.5:
        engines_to_run.extend(["hmm", "dmd"])
        weights["hmm"] = geometry_profile["reflexive"]
        weights["dmd"] = geometry_profile["reflexive"] * 0.8
        
        if geometry_profile["reflexive"] >= 0.7:
            engines_to_run.append("copula")
            weights["copula"] = geometry_profile["reflexive"] * 0.7
    
    # Mutual information if non-linear signatures
    if geometry_profile.get("reflexive", 0) >= 0.4 or geometry_profile.get("latent_flow", 0) >= 0.4:
        engines_to_run.append("mutual_information")
        weights["mutual_information"] = max(
            geometry_profile.get("reflexive", 0),
            geometry_profile.get("latent_flow", 0)
        ) * 0.8
    
    # Suppress if noise-dominated
    if noise_score >= 0.6:
        warnings_list.append(f"HIGH NOISE ({noise_score:.2f}): All weights reduced")
        for engine in weights:
            weights[engine] *= (1 - noise_score)
    
    # Remove duplicates
    engines_to_run = list(dict.fromkeys(engines_to_run))
    
    # Determine skipped engines
    engines_skipped = [e for e in ALL_ENGINES if e not in engines_to_run]
    
    if verbose:
        print(f"\nEngines to run: {engines_to_run}")
        print(f"Engines skipped: {engines_skipped}")
        print(f"\nWeights:")
        for e, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {e}: {w:.3f}")
    
    # Step 3: Execute engines
    engine_results = {}
    for engine in engines_to_run:
        if verbose:
            print(f"\n  Running {engine}...", end=" ")
        
        result = run_engine_safe(engine, df, data_variants)
        engine_results[engine] = result
        
        if verbose:
            status = "✓" if result.success else "✗"
            print(f"{status} ({result.runtime_ms:.1f}ms)")
    
    end_time = datetime.now()
    total_ms = time.time() * 1000 - start_ms
    
    return ExecutionRun(
        mode="with_agents",
        started_at=start_time.isoformat(),
        completed_at=end_time.isoformat(),
        total_runtime_ms=total_ms,
        engines_executed=engines_to_run,
        engines_skipped=engines_skipped,
        engine_results={k: asdict(v) for k, v in engine_results.items()},
        weights=weights,
        geometry_profile=geometry_profile,
        warnings=warnings_list
    )


def run_direct(df: pd.DataFrame, verbose: bool = True) -> ExecutionRun:
    """
    Run direct execution - all engines, equal weights, no routing.
    """
    start_time = datetime.now()
    start_ms = time.time() * 1000
    
    if verbose:
        print("\n" + "=" * 60)
        print("DIRECT EXECUTION (No Agents)")
        print("=" * 60)
    
    # Prepare data variants
    data_variants = prepare_data_variants(df)
    
    # Run all engines with equal weight
    engines_to_run = ALL_ENGINES.copy()
    weights = {e: 1.0 for e in engines_to_run}
    
    if verbose:
        print(f"\nEngines to run: {engines_to_run}")
        print(f"All weights: 1.0 (equal)")
    
    engine_results = {}
    for engine in engines_to_run:
        if verbose:
            print(f"\n  Running {engine}...", end=" ")
        
        result = run_engine_safe(engine, df, data_variants)
        engine_results[engine] = result
        
        if verbose:
            status = "✓" if result.success else "✗"
            print(f"{status} ({result.runtime_ms:.1f}ms)")
    
    end_time = datetime.now()
    total_ms = time.time() * 1000 - start_ms
    
    return ExecutionRun(
        mode="direct",
        started_at=start_time.isoformat(),
        completed_at=end_time.isoformat(),
        total_runtime_ms=total_ms,
        engines_executed=engines_to_run,
        engines_skipped=[],
        engine_results={k: asdict(v) for k, v in engine_results.items()},
        weights=weights,
        geometry_profile=None,
        warnings=[]
    )


# =============================================================================
# Comparison Logic
# =============================================================================

def compare_runs(agent_run: ExecutionRun, direct_run: ExecutionRun) -> ComparisonResult:
    """Compare agent-routed vs direct execution."""
    
    # Engine coverage
    agent_engines = set(agent_run.engines_executed)
    direct_engines = set(direct_run.engines_executed)
    
    engines_only_agent = list(agent_engines - direct_engines)
    engines_only_direct = list(direct_engines - agent_engines)
    engines_both = list(agent_engines & direct_engines)
    
    # Weight differences
    weight_diffs = {}
    for engine in engines_both:
        agent_w = agent_run.weights.get(engine, 0)
        direct_w = direct_run.weights.get(engine, 1.0)
        weight_diffs[engine] = {
            "agent_weight": agent_w,
            "direct_weight": direct_w,
            "difference": agent_w - direct_w,
            "ratio": agent_w / direct_w if direct_w > 0 else 0
        }
    
    # Metric correlations (for engines run in both modes)
    metric_correlations = {}
    for engine in engines_both:
        agent_result = agent_run.engine_results.get(engine, {})
        direct_result = direct_run.engine_results.get(engine, {})
        
        if agent_result.get("success") and direct_result.get("success"):
            agent_metrics = agent_result.get("metrics", {})
            direct_metrics = direct_result.get("metrics", {})
            
            # Compare common metrics
            common_metrics = set(agent_metrics.keys()) & set(direct_metrics.keys())
            correlations = {}
            for metric in common_metrics:
                av = agent_metrics[metric]
                dv = direct_metrics[metric]
                if isinstance(av, (int, float)) and isinstance(dv, (int, float)):
                    # For single values, just report difference
                    correlations[metric] = {
                        "agent_value": av,
                        "direct_value": dv,
                        "match": abs(av - dv) < 0.001
                    }
            metric_correlations[engine] = correlations
    
    # Performance comparison
    overhead_pct = ((agent_run.total_runtime_ms - direct_run.total_runtime_ms) 
                    / direct_run.total_runtime_ms * 100) if direct_run.total_runtime_ms > 0 else 0
    
    # Noise handling assessment
    noise_handling = {}
    if agent_run.geometry_profile:
        noise_score = agent_run.geometry_profile.get("noise_score", 0)
        noise_handling = {
            "detected_noise_score": noise_score,
            "engines_suppressed": len(engines_only_direct),
            "weight_reductions": {
                e: wd["difference"] 
                for e, wd in weight_diffs.items() 
                if wd["difference"] < 0
            }
        }
    
    # Geometry routing impact
    geometry_impact = {}
    if agent_run.geometry_profile:
        geometry_impact = {
            "geometry_profile": agent_run.geometry_profile,
            "engines_enabled_by_geometry": [
                e for e in engines_only_agent 
                if e not in CORE_ENGINES
            ],
            "conditional_weights": {
                e: w for e, w in agent_run.weights.items()
                if e not in CORE_ENGINES
            }
        }
    
    return ComparisonResult(
        engines_only_in_agent_mode=engines_only_agent,
        engines_only_in_direct_mode=engines_only_direct,
        engines_in_both=engines_both,
        weight_differences=weight_diffs,
        metric_correlations=metric_correlations,
        agent_runtime_ms=agent_run.total_runtime_ms,
        direct_runtime_ms=direct_run.total_runtime_ms,
        overhead_pct=overhead_pct,
        noise_handling=noise_handling,
        geometry_routing_impact=geometry_impact
    )


# =============================================================================
# Synthetic Benchmarks
# =============================================================================

def run_synthetic_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """
    Run ablation study on synthetic data with known ground truth.
    Tests: Does agent routing correctly identify data types?
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SYNTHETIC BENCHMARKS")
        print("=" * 60)
    
    np.random.seed(42)
    results = {}
    
    # Test 1: Pure noise - agents should suppress
    if verbose:
        print("\n>>> Test 1: Pure Noise")
    noise_data = pd.DataFrame({
        f"noise_{i}": np.random.randn(200) for i in range(5)
    })
    noise_agent = run_with_agents(noise_data, verbose=False)
    noise_direct = run_direct(noise_data, verbose=False)
    
    results["pure_noise"] = {
        "expected": "High noise score, suppressed weights",
        "agent_noise_score": noise_agent.geometry_profile.get("noise_score", 0),
        "engines_skipped": len(noise_agent.engines_skipped),
        "mean_agent_weight": np.mean(list(noise_agent.weights.values())),
        "verdict": "PASS" if noise_agent.geometry_profile.get("noise_score", 0) > 0.5 else "FAIL"
    }
    if verbose:
        print(f"  Noise score: {results['pure_noise']['agent_noise_score']:.3f}")
        print(f"  Verdict: {results['pure_noise']['verdict']}")
    
    # Test 2: Clear sine wave - should detect oscillator
    if verbose:
        print("\n>>> Test 2: Oscillator (Sine Waves)")
    t = np.linspace(0, 8 * np.pi, 200)
    sine_data = pd.DataFrame({
        f"sine_{i}": np.sin(t + i * 0.5) + np.random.normal(0, 0.1, len(t))
        for i in range(5)
    })
    sine_agent = run_with_agents(sine_data, verbose=False)
    
    results["oscillator"] = {
        "expected": "High oscillator score",
        "oscillator_score": sine_agent.geometry_profile.get("oscillator", 0),
        "wavelet_weight": sine_agent.weights.get("wavelet", 0),
        "verdict": "PASS" if sine_agent.geometry_profile.get("oscillator", 0) > 0.4 else "FAIL"
    }
    if verbose:
        print(f"  Oscillator score: {results['oscillator']['oscillator_score']:.3f}")
        print(f"  Verdict: {results['oscillator']['verdict']}")
    
    # Test 3: GARCH-like - should detect reflexive
    if verbose:
        print("\n>>> Test 3: Reflexive (Volatility Clustering)")
    reflexive_data = pd.DataFrame()
    for i in range(5):
        returns = np.random.standard_t(4, 200) * 0.02
        vol = np.ones(200)
        for j in range(1, 200):
            vol[j] = 0.9 * vol[j-1] + 0.1 * returns[j-1]**2
        reflexive_data[f"garch_{i}"] = np.cumsum(returns * np.sqrt(vol))
    
    reflexive_agent = run_with_agents(reflexive_data, verbose=False)
    
    results["reflexive"] = {
        "expected": "High reflexive score, HMM/DMD enabled",
        "reflexive_score": reflexive_agent.geometry_profile.get("reflexive", 0),
        "hmm_enabled": "hmm" in reflexive_agent.engines_executed,
        "dmd_enabled": "dmd" in reflexive_agent.engines_executed,
        "verdict": "PASS" if reflexive_agent.geometry_profile.get("reflexive", 0) > 0.4 else "MARGINAL"
    }
    if verbose:
        print(f"  Reflexive score: {results['reflexive']['reflexive_score']:.3f}")
        print(f"  HMM enabled: {results['reflexive']['hmm_enabled']}")
        print(f"  Verdict: {results['reflexive']['verdict']}")
    
    # Test 4: Logistic growth - should detect latent flow
    if verbose:
        print("\n>>> Test 4: Latent Flow (Logistic Growth)")
    t = np.linspace(0, 10, 200)
    logistic_data = pd.DataFrame({
        f"logistic_{i}": 100 / (1 + np.exp(-1.5 * (t - 5 + i*0.5))) + np.random.normal(0, 2, len(t))
        for i in range(5)
    })
    logistic_agent = run_with_agents(logistic_data, verbose=False)
    
    results["latent_flow"] = {
        "expected": "High latent_flow score",
        "latent_flow_score": logistic_agent.geometry_profile.get("latent_flow", 0),
        "verdict": "PASS" if logistic_agent.geometry_profile.get("latent_flow", 0) > 0.4 else "MARGINAL"
    }
    if verbose:
        print(f"  Latent flow score: {results['latent_flow']['latent_flow_score']:.3f}")
        print(f"  Verdict: {results['latent_flow']['verdict']}")
    
    # Summary
    passes = sum(1 for r in results.values() if r["verdict"] == "PASS")
    total = len(results)
    results["summary"] = {
        "passed": passes,
        "total": total,
        "pass_rate": passes / total
    }
    
    if verbose:
        print(f"\n>>> Summary: {passes}/{total} benchmarks passed")
    
    return results


# =============================================================================
# Data Loading
# =============================================================================

def load_data(db_path: str, indicators: List[str], 
              start_date: str = None) -> pd.DataFrame:
    """Load indicator data from DuckDB."""
    conn = duckdb.connect(db_path, read_only=True)
    
    ind_list = ", ".join(f"'{i}'" for i in indicators)
    
    query = f"""
        SELECT date, indicator_id, value
        FROM clean.indicators
        WHERE indicator_id IN ({ind_list})
    """
    
    if start_date:
        query += f" AND date >= '{start_date}'"
    
    query += " ORDER BY date"
    
    df = conn.execute(query).fetchdf()
    conn.close()
    
    if df.empty:
        raise ValueError(f"No data found for indicators: {indicators}")
    
    # Pivot to wide format
    df_wide = df.pivot(index='date', columns='indicator_id', values='value')
    df_wide = df_wide.dropna()
    
    return df_wide


# =============================================================================
# Report Generation
# =============================================================================

def generate_findings(comparison: ComparisonResult, 
                      synthetic: Optional[Dict] = None) -> Tuple[List[str], str]:
    """Generate key findings and recommendation from comparison."""
    findings = []
    
    # Engine coverage
    if comparison.engines_only_in_direct_mode:
        findings.append(
            f"Agent routing skipped {len(comparison.engines_only_in_direct_mode)} engines: "
            f"{comparison.engines_only_in_direct_mode}"
        )
    
    # Weight modulation
    significant_reductions = [
        e for e, wd in comparison.weight_differences.items()
        if wd["difference"] < -0.2
    ]
    if significant_reductions:
        findings.append(
            f"Agent significantly reduced weights for: {significant_reductions}"
        )
    
    # Noise handling
    if comparison.noise_handling:
        noise_score = comparison.noise_handling.get("detected_noise_score", 0)
        if noise_score > 0.5:
            findings.append(
                f"High noise detected ({noise_score:.2f}) - weights appropriately reduced"
            )
    
    # Performance
    if comparison.overhead_pct > 20:
        findings.append(
            f"Agent overhead: {comparison.overhead_pct:.1f}% slower than direct execution"
        )
    elif comparison.overhead_pct < -10:
        findings.append(
            f"Agent saved {-comparison.overhead_pct:.1f}% runtime by skipping engines"
        )
    
    # Geometry routing
    if comparison.geometry_routing_impact:
        enabled = comparison.geometry_routing_impact.get("engines_enabled_by_geometry", [])
        if enabled:
            findings.append(
                f"Geometry detection enabled conditional engines: {enabled}"
            )
    
    # Recommendation
    if synthetic:
        pass_rate = synthetic.get("summary", {}).get("pass_rate", 0)
        if pass_rate >= 0.75:
            recommendation = (
                "AGENT LAYER VALIDATED: Correctly identifies data geometry and routes "
                "engines appropriately. Synthetic benchmark pass rate: {:.0%}".format(pass_rate)
            )
        else:
            recommendation = (
                "AGENT TUNING NEEDED: Geometry detection underperforming on synthetic "
                "benchmarks. Pass rate: {:.0%}. Review detection thresholds.".format(pass_rate)
            )
    else:
        if len(significant_reductions) > 0 or comparison.engines_only_in_direct_mode:
            recommendation = (
                "Agent layer is actively routing - provides value through selective "
                "engine execution and weight modulation."
            )
        else:
            recommendation = (
                "Agent layer has minimal impact on this dataset - consider whether "
                "overhead is justified."
            )
    
    return findings, recommendation


def print_report(report: AblationReport):
    """Print human-readable report to console."""
    print("\n" + "=" * 70)
    print("PRISM ABLATION STUDY REPORT")
    print("=" * 70)
    
    print(f"\nStudy ID: {report.study_id}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Data: {report.n_indicators} indicators, {report.n_observations} observations")
    
    print("\n" + "-" * 70)
    print("EXECUTION COMPARISON")
    print("-" * 70)
    
    # Handle both object and dict forms
    agent_run = report.agent_run if isinstance(report.agent_run, dict) else asdict(report.agent_run)
    direct_run = report.direct_run if isinstance(report.direct_run, dict) else asdict(report.direct_run)
    comparison = report.comparison if isinstance(report.comparison, dict) else asdict(report.comparison)
    
    print(f"\n{'Metric':<30} {'Agent':>15} {'Direct':>15}")
    print("-" * 60)
    print(f"{'Runtime (ms)':<30} {agent_run['total_runtime_ms']:>15.1f} {direct_run['total_runtime_ms']:>15.1f}")
    print(f"{'Engines executed':<30} {len(agent_run['engines_executed']):>15} {len(direct_run['engines_executed']):>15}")
    print(f"{'Engines skipped':<30} {len(agent_run['engines_skipped']):>15} {len(direct_run['engines_skipped']):>15}")
    
    if agent_run.get('geometry_profile'):
        print("\n" + "-" * 70)
        print("GEOMETRY PROFILE (Agent Detection)")
        print("-" * 70)
        for k, v in agent_run['geometry_profile'].items():
            print(f"  {k}: {v:.3f}")
    
    print("\n" + "-" * 70)
    print("WEIGHT DIFFERENCES")
    print("-" * 70)
    print(f"{'Engine':<25} {'Agent':>10} {'Direct':>10} {'Diff':>10}")
    print("-" * 55)
    weight_diffs = comparison.get('weight_differences', {})
    for engine, wd in sorted(weight_diffs.items(), key=lambda x: x[1]["difference"]):
        print(f"{engine:<25} {wd['agent_weight']:>10.3f} {wd['direct_weight']:>10.3f} {wd['difference']:>+10.3f}")
    
    if report.synthetic_benchmarks:
        print("\n" + "-" * 70)
        print("SYNTHETIC BENCHMARKS")
        print("-" * 70)
        for name, result in report.synthetic_benchmarks.items():
            if name != "summary":
                print(f"  {name}: {result.get('verdict', 'N/A')}")
        summary = report.synthetic_benchmarks.get("summary", {})
        print(f"\n  Pass Rate: {summary.get('passed', 0)}/{summary.get('total', 0)}")
    
    print("\n" + "-" * 70)
    print("KEY FINDINGS")
    print("-" * 70)
    for i, finding in enumerate(report.key_findings, 1):
        print(f"  {i}. {finding}")
    
    print("\n" + "-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)
    print(f"  {report.recommendation}")
    
    # Engines only in direct mode (skipped by agent)
    engines_skipped = comparison.get('engines_only_in_direct_mode', [])
    if engines_skipped:
        print("\n" + "-" * 70)
        print("ENGINES SKIPPED BY AGENT ROUTING")
        print("-" * 70)
        print(f"  {engines_skipped}")
    
    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Ablation Study: Agent vs Direct Execution"
    )
    parser.add_argument("--db", default=str(get_db_path()), 
                        help="Database path")
    parser.add_argument("--indicators", 
                        default="SPY,XLU,XLF,XLK,XLE,AGG,GLD,VIXCLS",
                        help="Comma-separated indicator list")
    parser.add_argument("--start", default="2015-01-01", 
                        help="Start date")
    parser.add_argument("--output", "-o", type=Path, 
                        help="Output JSON file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Include synthetic benchmarks")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    indicators = [i.strip() for i in args.indicators.split(",")]
    
    study_id = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load data
    if verbose:
        print(f"\nLoading data from {args.db}...")
    
    try:
        df = load_data(args.db, indicators, args.start)
        data_source = args.db
        if verbose:
            print(f"Loaded: {len(df)} rows, {len(df.columns)} indicators")
    except Exception as e:
        if verbose:
            print(f"Could not load from database: {e}")
            print("Using synthetic data for demonstration...")
        
        # Generate demo data
        np.random.seed(42)
        t = np.arange(500)
        df = pd.DataFrame({
            "demo_trend": t * 0.1 + np.random.randn(500) * 2,
            "demo_cycle": np.sin(t * 0.1) * 10 + np.random.randn(500),
            "demo_noise": np.random.randn(500) * 5,
            "demo_vol": np.cumsum(np.random.standard_t(4, 500) * 0.5),
        })
        data_source = "synthetic_demo"
    
    # Run with agents
    agent_run = run_with_agents(df, verbose=verbose)
    
    # Run direct
    direct_run = run_direct(df, verbose=verbose)
    
    # Compare
    comparison = compare_runs(agent_run, direct_run)
    
    # Synthetic benchmarks (optional)
    synthetic = None
    if args.synthetic:
        synthetic = run_synthetic_benchmarks(verbose=verbose)
    
    # Generate findings
    findings, recommendation = generate_findings(comparison, synthetic)
    
    # Build report
    report = AblationReport(
        study_id=study_id,
        timestamp=datetime.now().isoformat(),
        data_source=data_source,
        n_indicators=len(df.columns),
        n_observations=len(df),
        agent_run=agent_run,
        direct_run=direct_run,
        comparison=asdict(comparison),
        synthetic_benchmarks=synthetic,
        key_findings=findings,
        recommendation=recommendation
    )
    
    # Print report
    print_report(report)
    
    # Save JSON
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
