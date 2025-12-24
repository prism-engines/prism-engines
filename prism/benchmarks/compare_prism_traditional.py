"""
PRISM vs Traditional Methods Benchmark

Compares PRISM's multi-lens approach against standard quantitative methods.

Key Comparisons:
1. Regime Detection: PRISM geometry vs VIX thresholds / Markov switching
2. Correlation: Multi-scale wavelet vs rolling correlation
3. Lead/Lag: Transfer entropy network vs simple cross-correlation
4. Clustering: Dynamic PRISM clusters vs static sector groups

Metrics:
- Detection lead time (days early)
- False positive rate
- Out-of-sample accuracy
- Regime duration accuracy
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .traditional_methods import TraditionalAnalysis

logger = logging.getLogger(__name__)


# =============================================================================
# Known Events for Validation
# =============================================================================

KNOWN_EVENTS = {
    # Date, Event Type, Description
    "2008-09-15": ("crisis_start", "Lehman Brothers bankruptcy"),
    "2008-10-10": ("crisis_peak", "2008 crisis VIX peak"),
    "2009-03-09": ("crisis_end", "2008 crisis market bottom"),
    
    "2010-05-06": ("flash_crash", "Flash Crash"),
    
    "2011-08-05": ("correction", "US debt downgrade"),
    
    "2015-08-24": ("correction", "China devaluation selloff"),
    
    "2018-02-05": ("vol_spike", "Volmageddon"),
    "2018-12-24": ("correction", "Q4 2018 selloff bottom"),
    
    "2020-02-20": ("crisis_start", "COVID crash begins"),
    "2020-03-16": ("crisis_peak", "COVID VIX peak"),
    "2020-03-23": ("crisis_end", "COVID market bottom"),
    
    "2022-01-04": ("regime_change", "2022 rate hike regime begins"),
    "2022-10-12": ("correction_end", "2022 bear market bottom"),
    
    "2024-08-05": ("vol_spike", "Yen carry trade unwind"),
}


@dataclass
class BenchmarkResult:
    """Result of a single benchmark comparison."""
    event_date: date
    event_type: str
    traditional_detection: Optional[date]
    prism_detection: Optional[date]
    traditional_lead_days: int
    prism_lead_days: int
    prism_advantage: int
    
    def __str__(self):
        return (
            f"{self.event_date} ({self.event_type}): "
            f"Traditional={self.traditional_lead_days}d, "
            f"PRISM={self.prism_lead_days}d, "
            f"Advantage={self.prism_advantage}d"
        )


# =============================================================================
# Comparison Framework
# =============================================================================

class PRISMBenchmark:
    """
    Compare PRISM results against traditional methods.
    """
    
    def __init__(self):
        self.traditional = TraditionalAnalysis()
    
    def compare_regime_detection(
        self,
        traditional_regimes: pd.Series,
        prism_regimes: pd.Series,
        events: Optional[Dict[str, Tuple[str, str]]] = None,
        crisis_regime_traditional: int = 2,
        crisis_regime_prism: int = 2,
    ) -> Dict[str, Any]:
        """
        Compare regime detection timing.
        
        Args:
            traditional_regimes: Series of traditional regime labels
            prism_regimes: Series of PRISM regime labels
            events: Dict of event dates and types
            crisis_regime_traditional: Which regime number = crisis
            crisis_regime_prism: Which regime number = crisis
        
        Returns:
            Comparison metrics
        """
        if events is None:
            events = KNOWN_EVENTS
        
        results = []
        
        for event_date_str, (event_type, description) in events.items():
            event_date = pd.Timestamp(event_date_str)
            
            if event_date < traditional_regimes.index.min():
                continue
            if event_date < prism_regimes.index.min():
                continue
            if event_date > traditional_regimes.index.max():
                continue
            if event_date > prism_regimes.index.max():
                continue
            
            trad_detection = self._find_regime_entry(
                traditional_regimes,
                crisis_regime_traditional,
                event_date,
                lookback_days=60
            )
            
            prism_detection = self._find_regime_entry(
                prism_regimes,
                crisis_regime_prism,
                event_date,
                lookback_days=60
            )
            
            trad_lead = (event_date - trad_detection).days if trad_detection else 0
            prism_lead = (event_date - prism_detection).days if prism_detection else 0
            
            results.append(BenchmarkResult(
                event_date=event_date.date(),
                event_type=event_type,
                traditional_detection=trad_detection.date() if trad_detection else None,
                prism_detection=prism_detection.date() if prism_detection else None,
                traditional_lead_days=trad_lead,
                prism_lead_days=prism_lead,
                prism_advantage=prism_lead - trad_lead,
            ))
        
        advantages = [r.prism_advantage for r in results]
        
        return {
            "results": results,
            "avg_prism_advantage": np.mean(advantages) if advantages else 0,
            "events_prism_won": sum(1 for a in advantages if a > 0),
            "events_traditional_won": sum(1 for a in advantages if a < 0),
            "events_tied": sum(1 for a in advantages if a == 0),
            "max_prism_advantage": max(advantages) if advantages else 0,
        }
    
    def _find_regime_entry(
        self,
        regimes: pd.Series,
        target_regime: int,
        event_date: pd.Timestamp,
        lookback_days: int = 60,
    ) -> Optional[pd.Timestamp]:
        """
        Find when a regime was first entered before an event.
        """
        start_date = event_date - timedelta(days=lookback_days)
        
        mask = (regimes.index >= start_date) & (regimes.index <= event_date)
        window = regimes[mask]
        
        if len(window) == 0:
            return None
        
        is_target = window == target_regime
        regime_entries = is_target & (~is_target.shift(1).fillna(False))
        
        if regime_entries.any():
            return regime_entries[regime_entries].index[0]
        
        if is_target.iloc[0]:
            return window.index[0]
        
        return None
    
    def compare_correlation_detection(
        self,
        traditional_corr: pd.Series,
        prism_coherence: pd.Series,
        breakdown_dates: List[str],
        threshold_trad: float = 0.7,
        threshold_prism: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Compare correlation breakdown detection.
        """
        results = []
        
        for date_str in breakdown_dates:
            event_date = pd.Timestamp(date_str)
            
            trad_detection = self._find_threshold_cross(
                traditional_corr, threshold_trad, event_date, above=True
            )
            prism_detection = self._find_threshold_cross(
                prism_coherence, threshold_prism, event_date, above=True
            )
            
            trad_lead = (event_date - trad_detection).days if trad_detection else 0
            prism_lead = (event_date - prism_detection).days if prism_detection else 0
            
            results.append({
                "event_date": event_date.date(),
                "traditional_lead": trad_lead,
                "prism_lead": prism_lead,
                "prism_advantage": prism_lead - trad_lead,
            })
        
        return {
            "results": results,
            "avg_prism_advantage": np.mean([r["prism_advantage"] for r in results]) if results else 0,
        }
    
    def _find_threshold_cross(
        self,
        series: pd.Series,
        threshold: float,
        event_date: pd.Timestamp,
        lookback_days: int = 30,
        above: bool = True,
    ) -> Optional[pd.Timestamp]:
        """Find when series crossed threshold before event."""
        start_date = event_date - timedelta(days=lookback_days)
        mask = (series.index >= start_date) & (series.index <= event_date)
        window = series[mask]
        
        if len(window) == 0:
            return None
        
        if above:
            crosses = (window > threshold) & (window.shift(1) <= threshold)
        else:
            crosses = (window < threshold) & (window.shift(1) >= threshold)
        
        if crosses.any():
            return crosses[crosses].index[0]
        
        return None
    
    def compare_false_positives(
        self,
        traditional_signals: pd.Series,
        prism_signals: pd.Series,
        true_events: List[str],
        tolerance_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare false positive rates.
        """
        true_event_dates = [pd.Timestamp(d) for d in true_events]
        
        def count_false_positives(signals: pd.Series) -> int:
            signal_dates = signals[signals == 1].index
            false_positives = 0
            
            for signal_date in signal_dates:
                is_near_event = any(
                    abs((signal_date - event_date).days) <= tolerance_days
                    for event_date in true_event_dates
                )
                if not is_near_event:
                    false_positives += 1
            
            return false_positives
        
        trad_fp = count_false_positives(traditional_signals)
        prism_fp = count_false_positives(prism_signals)
        
        trad_total = int(traditional_signals.sum())
        prism_total = int(prism_signals.sum())
        
        return {
            "traditional_false_positives": trad_fp,
            "traditional_total_signals": trad_total,
            "traditional_fp_rate": trad_fp / trad_total if trad_total > 0 else 0,
            "prism_false_positives": prism_fp,
            "prism_total_signals": prism_total,
            "prism_fp_rate": prism_fp / prism_total if prism_total > 0 else 0,
        }
    
    def generate_comparison_report(
        self,
        traditional_results: Dict[str, Any],
        prism_results: Dict[str, Any],
        df: pd.DataFrame,
    ) -> str:
        """
        Generate human-readable comparison report.
        """
        lines = [
            "=" * 60,
            "PRISM vs TRADITIONAL METHODS COMPARISON",
            "=" * 60,
            "",
        ]
        
        lines.append(f"Analysis Period: {df.index.min().date()} to {df.index.max().date()}")
        lines.append(f"Indicators: {len(df.columns)}")
        lines.append("")
        
        lines.append("-" * 40)
        lines.append("TRADITIONAL METHODS SUMMARY")
        lines.append("-" * 40)
        
        if "vix_regime" in traditional_results:
            current = traditional_results["vix_regime"].iloc[-1]
            labels = {0: "Complacent", 1: "Normal", 2: "Fear", 3: "Panic"}
            lines.append(f"  VIX Regime: {labels.get(current, 'Unknown')}")
        
        if "rolling_correlation" in traditional_results:
            avg = traditional_results["rolling_correlation"]["average"].iloc[-1]
            lines.append(f"  Avg Correlation: {avg:.3f}")
        
        if "trend" in traditional_results:
            signal = traditional_results["trend"]["signal"].iloc[-1]
            lines.append(f"  Trend (50/200 MA): {'Bullish' if signal > 0 else 'Bearish'}")
        
        lines.append("")
        
        lines.append("-" * 40)
        lines.append("PRISM MULTI-LENS SUMMARY")
        lines.append("-" * 40)
        
        if "pca" in prism_results:
            pc1 = prism_results["pca"].get("variance_pc1", 0)
            lines.append(f"  PC1 Variance: {pc1:.1%}")
        
        if "clustering" in prism_results:
            n_clusters = prism_results["clustering"].get("n_clusters", 0)
            silhouette = prism_results["clustering"].get("silhouette_score", 0)
            lines.append(f"  Clusters: {n_clusters} (silhouette: {silhouette:.3f})")
        
        if "hurst" in prism_results:
            avg_h = prism_results["hurst"].get("avg_hurst", 0)
            lines.append(f"  Avg Hurst: {avg_h:.3f}")
        
        if "hmm" in prism_results:
            n_states = prism_results["hmm"].get("n_states", 0)
            lines.append(f"  HMM States: {n_states}")
        
        if "dmd" in prism_results:
            error = prism_results["dmd"].get("reconstruction_error", 0)
            lines.append(f"  DMD Reconstruction Error: {error:.3f}")
        
        lines.append("")
        lines.append("-" * 40)
        lines.append("KEY DIFFERENCES")
        lines.append("-" * 40)
        lines.append("  Traditional: Single metrics, binary thresholds")
        lines.append("  PRISM: 20 lenses, continuous geometry, consensus signals")
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# Automated Benchmark Runner
# =============================================================================

def run_full_benchmark(
    df: pd.DataFrame,
    prism_engine_results: Dict[str, Dict[str, Any]],
    vix: Optional[pd.Series] = None,
    yield_curve: Optional[pd.Series] = None,
    reference_price: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Run complete benchmark comparison.
    
    Args:
        df: Indicator data
        prism_engine_results: Dict of engine name -> metrics
        vix: VIX series (for traditional regime)
        yield_curve: 10Y-2Y spread
        reference_price: SPY or similar for trend
    
    Returns:
        Complete benchmark results
    """
    benchmark = PRISMBenchmark()
    
    traditional = benchmark.traditional.full_analysis(
        df=df,
        vix=vix,
        yield_curve=yield_curve,
        reference_price=reference_price,
    )
    
    report = benchmark.generate_comparison_report(
        traditional_results=traditional,
        prism_results=prism_engine_results,
        df=df,
    )
    
    return {
        "traditional": traditional,
        "prism": prism_engine_results,
        "report": report,
    }
