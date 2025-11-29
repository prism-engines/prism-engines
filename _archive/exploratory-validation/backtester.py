"""
Backtester for Historical Validation
=====================================

Tests VCF against known historical events:
- Did it detect regime shifts before/during major events?
- How much lead time did it provide?
- What's the false positive rate?

Usage:
    from backtester import HistoricalBacktester
    
    backtester = HistoricalBacktester()
    results = backtester.evaluate_regime_detection(lens, data)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


# =============================================================================
# KNOWN HISTORICAL EVENTS
# =============================================================================

MARKET_EVENTS = {
    # (start_date, end_date, event_name, event_type)
    'black_monday_1987': ('1987-10-19', '1987-10-19', 'Black Monday', 'crash'),
    'gulf_war_1990': ('1990-08-02', '1990-08-02', 'Gulf War Start', 'geopolitical'),
    'asian_crisis_1997': ('1997-07-02', '1997-10-27', 'Asian Financial Crisis', 'crisis'),
    'ltcm_1998': ('1998-08-17', '1998-09-23', 'LTCM Crisis', 'crisis'),
    'dotcom_peak_2000': ('2000-03-10', '2000-03-10', 'Dot-com Peak', 'bubble'),
    'dotcom_crash_2000': ('2000-03-13', '2000-04-14', 'Dot-com Crash', 'crash'),
    '911_2001': ('2001-09-11', '2001-09-21', '9/11 Attacks', 'geopolitical'),
    'enron_2001': ('2001-10-16', '2001-12-02', 'Enron Scandal', 'crisis'),
    'iraq_war_2003': ('2003-03-19', '2003-03-19', 'Iraq War Start', 'geopolitical'),
    'gfc_start_2007': ('2007-08-09', '2007-08-09', 'GFC BNP Paribas', 'crisis'),
    'bear_stearns_2008': ('2008-03-14', '2008-03-17', 'Bear Stearns Collapse', 'crisis'),
    'lehman_2008': ('2008-09-15', '2008-09-15', 'Lehman Brothers', 'crash'),
    'gfc_bottom_2009': ('2009-03-09', '2009-03-09', 'GFC Bottom', 'recovery'),
    'flash_crash_2010': ('2010-05-06', '2010-05-06', 'Flash Crash', 'crash'),
    'us_downgrade_2011': ('2011-08-05', '2011-08-08', 'US Debt Downgrade', 'crisis'),
    'taper_tantrum_2013': ('2013-05-22', '2013-06-24', 'Taper Tantrum', 'policy'),
    'china_crash_2015': ('2015-08-24', '2015-08-26', 'China Crash', 'crash'),
    'brexit_2016': ('2016-06-24', '2016-06-27', 'Brexit Vote', 'geopolitical'),
    'trump_election_2016': ('2016-11-08', '2016-11-09', 'Trump Election', 'political'),
    'volmageddon_2018': ('2018-02-05', '2018-02-09', 'Volmageddon', 'crash'),
    'q4_2018_selloff': ('2018-12-24', '2018-12-26', 'Q4 2018 Selloff', 'crash'),
    'covid_crash_2020': ('2020-02-24', '2020-03-23', 'COVID Crash', 'crash'),
    'covid_bottom_2020': ('2020-03-23', '2020-03-23', 'COVID Bottom', 'recovery'),
    'meme_stocks_2021': ('2021-01-27', '2021-01-29', 'GME Squeeze', 'anomaly'),
    'inflation_2022': ('2022-01-03', '2022-06-16', '2022 Bear Market', 'bear'),
    'svb_2023': ('2023-03-10', '2023-03-13', 'SVB Collapse', 'crisis'),
}

# Economic events (policy changes, etc.)
ECONOMIC_EVENTS = {
    'volcker_peak_1981': ('1981-06-01', '1981-06-30', 'Volcker Rate Peak', 'policy'),
    'plaza_accord_1985': ('1985-09-22', '1985-09-22', 'Plaza Accord', 'policy'),
    'greenspan_put_1998': ('1998-09-29', '1998-11-17', 'Greenspan Rate Cuts', 'policy'),
    'fed_hikes_2004': ('2004-06-30', '2004-06-30', 'Fed Tightening Begins', 'policy'),
    'qe1_2008': ('2008-11-25', '2008-11-25', 'QE1 Announced', 'policy'),
    'qe2_2010': ('2010-11-03', '2010-11-03', 'QE2 Announced', 'policy'),
    'twist_2011': ('2011-09-21', '2011-09-21', 'Operation Twist', 'policy'),
    'qe3_2012': ('2012-09-13', '2012-09-13', 'QE3 Announced', 'policy'),
    'taper_2013': ('2013-12-18', '2013-12-18', 'Taper Begins', 'policy'),
    'zirp_end_2015': ('2015-12-16', '2015-12-16', 'First Rate Hike', 'policy'),
    'covid_cut_2020': ('2020-03-15', '2020-03-15', 'Emergency Rate Cut', 'policy'),
    'rate_hikes_2022': ('2022-03-16', '2022-03-16', 'Fed Tightening 2022', 'policy'),
}


@dataclass
class EventDetectionResult:
    """Result of evaluating detection for a single event."""
    event_name: str
    event_date: pd.Timestamp
    event_type: str
    detected: bool
    lead_time_days: Optional[int]  # Days before event signal appeared
    signal_strength: Optional[float]
    detection_date: Optional[pd.Timestamp]
    
    def __repr__(self):
        if self.detected:
            return f"{self.event_name}: DETECTED ({self.lead_time_days}d lead)"
        return f"{self.event_name}: MISSED"


@dataclass
class BacktestResult:
    """Overall backtest results."""
    lens_name: str
    total_events: int
    detected_events: int
    detection_rate: float
    avg_lead_time: float
    false_positives: int
    false_positive_rate: float
    event_results: List[EventDetectionResult]
    
    def __repr__(self):
        return f"BacktestResult({self.lens_name}: {self.detection_rate:.1%} detection, {self.avg_lead_time:.1f}d avg lead)"


class HistoricalBacktester:
    """
    Backtests VCF lenses against known historical events.
    """
    
    def __init__(
        self,
        events: Dict[str, Tuple] = None,
        lookback_window: int = 30,  # Days before event to look for signals
        signal_threshold: float = 0.7,  # Threshold for regime change signal
    ):
        """
        Args:
            events: Dict of events (or uses MARKET_EVENTS by default)
            lookback_window: How many days before event to check for signals
            signal_threshold: Threshold above which regime change is detected
        """
        self.events = events or {**MARKET_EVENTS, **ECONOMIC_EVENTS}
        self.lookback_window = lookback_window
        self.signal_threshold = signal_threshold
    
    # =========================================================================
    # REGIME DETECTION EVALUATION
    # =========================================================================
    
    def evaluate_regime_detection(
        self,
        lens,
        data: pd.DataFrame,
        event_types: List[str] = None
    ) -> BacktestResult:
        """
        Evaluate lens's ability to detect regime changes around known events.
        
        Args:
            lens: Lens with analyze() method
            data: Historical data
            event_types: Filter to specific event types (e.g., ['crash', 'crisis'])
            
        Returns:
            BacktestResult with detection statistics
        """
        lens_name = type(lens).__name__
        
        # Run lens on full data
        try:
            result = lens.analyze(data)
        except Exception as e:
            return BacktestResult(
                lens_name=lens_name,
                total_events=0,
                detected_events=0,
                detection_rate=0.0,
                avg_lead_time=0.0,
                false_positives=0,
                false_positive_rate=0.0,
                event_results=[]
            )
        
        # Extract regime changes
        regime_changes = self._extract_regime_changes(result, data)
        
        # Filter events by type and date range
        valid_events = {}
        for event_id, (start_str, end_str, name, etype) in self.events.items():
            if event_types and etype not in event_types:
                continue
            
            event_date = pd.Timestamp(start_str)
            if event_date in data.index or (data.index[0] <= event_date <= data.index[-1]):
                valid_events[event_id] = (event_date, name, etype)
        
        # Evaluate each event
        event_results = []
        detected_count = 0
        lead_times = []
        
        for event_id, (event_date, name, etype) in valid_events.items():
            detection = self._check_event_detection(
                event_date, name, etype, regime_changes, data
            )
            event_results.append(detection)
            
            if detection.detected:
                detected_count += 1
                if detection.lead_time_days is not None:
                    lead_times.append(detection.lead_time_days)
        
        # Calculate false positives (regime changes not near any event)
        false_positives = self._count_false_positives(
            regime_changes, valid_events, data
        )
        
        total_signals = len(regime_changes)
        fp_rate = false_positives / total_signals if total_signals > 0 else 0.0
        
        return BacktestResult(
            lens_name=lens_name,
            total_events=len(valid_events),
            detected_events=detected_count,
            detection_rate=detected_count / len(valid_events) if valid_events else 0.0,
            avg_lead_time=np.mean(lead_times) if lead_times else 0.0,
            false_positives=false_positives,
            false_positive_rate=fp_rate,
            event_results=event_results
        )
    
    def _extract_regime_changes(
        self, 
        result: Dict, 
        data: pd.DataFrame
    ) -> List[Tuple[pd.Timestamp, float]]:
        """Extract regime change points from lens result."""
        changes = []
        
        # Try different result formats
        if 'regime_labels' in result:
            labels = np.array(result['regime_labels'])
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    if i < len(data):
                        changes.append((data.index[i], 1.0))
        
        elif 'states' in result:
            labels = np.array(result['states'])
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    if i < len(data):
                        changes.append((data.index[i], 1.0))
        
        elif 'regime_separation' in result:
            # Use regime separation as signal strength
            sep = result['regime_separation']
            if isinstance(sep, pd.Series):
                high_sep = sep[sep > sep.quantile(0.9)]
                for date in high_sep.index:
                    if date in data.index:
                        changes.append((date, high_sep[date]))
        
        elif 'coherence' in result:
            # Low coherence might indicate regime change
            coh = result['coherence']
            if isinstance(coh, pd.Series):
                low_coh = coh[coh < coh.quantile(0.1)]
                for date in low_coh.index:
                    changes.append((date, 1.0 - coh[date]))
        
        elif 'anomaly_rate' in result:
            # High anomaly rate might indicate regime change
            rate = result['anomaly_rate']
            if isinstance(rate, (dict, pd.Series)):
                if isinstance(rate, dict):
                    rate = pd.Series(rate)
                high_rate = rate[rate > rate.quantile(0.9)]
                for key in high_rate.index:
                    changes.append((pd.Timestamp(key), high_rate[key]))
        
        return sorted(changes, key=lambda x: x[0])
    
    def _check_event_detection(
        self,
        event_date: pd.Timestamp,
        event_name: str,
        event_type: str,
        regime_changes: List[Tuple[pd.Timestamp, float]],
        data: pd.DataFrame
    ) -> EventDetectionResult:
        """Check if a specific event was detected."""
        lookback_start = event_date - pd.Timedelta(days=self.lookback_window)
        lookafter_end = event_date + pd.Timedelta(days=5)  # Allow some lag
        
        # Find signals in the detection window
        signals_in_window = [
            (date, strength) for date, strength in regime_changes
            if lookback_start <= date <= lookafter_end
        ]
        
        if not signals_in_window:
            return EventDetectionResult(
                event_name=event_name,
                event_date=event_date,
                event_type=event_type,
                detected=False,
                lead_time_days=None,
                signal_strength=None,
                detection_date=None
            )
        
        # Find earliest signal
        earliest_signal = min(signals_in_window, key=lambda x: x[0])
        detection_date, signal_strength = earliest_signal
        
        lead_time = (event_date - detection_date).days
        
        return EventDetectionResult(
            event_name=event_name,
            event_date=event_date,
            event_type=event_type,
            detected=True,
            lead_time_days=lead_time,
            signal_strength=signal_strength,
            detection_date=detection_date
        )
    
    def _count_false_positives(
        self,
        regime_changes: List[Tuple[pd.Timestamp, float]],
        valid_events: Dict,
        data: pd.DataFrame
    ) -> int:
        """Count regime changes not associated with any known event."""
        event_dates = [event_date for event_date, _, _ in valid_events.values()]
        
        false_positives = 0
        for change_date, _ in regime_changes:
            is_near_event = False
            for event_date in event_dates:
                if abs((change_date - event_date).days) <= self.lookback_window:
                    is_near_event = True
                    break
            
            if not is_near_event:
                false_positives += 1
        
        return false_positives
    
    # =========================================================================
    # ROLLING BACKTEST
    # =========================================================================
    
    def rolling_backtest(
        self,
        lens,
        data: pd.DataFrame,
        train_window: int = 252,  # 1 year
        test_window: int = 63,    # 1 quarter
        step: int = 21            # 1 month
    ) -> Dict[str, Any]:
        """
        Rolling window backtest - trains on past, tests on future.
        
        Tests whether patterns detected in training period persist.
        """
        results = {
            'windows': [],
            'train_top_indicators': [],
            'test_top_indicators': [],
            'consistency_scores': [],
        }
        
        start_idx = 0
        while start_idx + train_window + test_window <= len(data):
            train_data = data.iloc[start_idx:start_idx + train_window]
            test_data = data.iloc[start_idx + train_window:start_idx + train_window + test_window]
            
            try:
                # Train period analysis
                train_result = lens.analyze(train_data)
                train_importance = self._get_importance(train_result)
                train_top = self._get_top_n(train_importance, 5)
                
                # Test period analysis
                test_result = lens.analyze(test_data)
                test_importance = self._get_importance(test_result)
                test_top = self._get_top_n(test_importance, 5)
                
                # Consistency: how many train top indicators are still top in test?
                overlap = len(set(train_top) & set(test_top))
                consistency = overlap / len(train_top) if train_top else 0
                
                results['windows'].append({
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                })
                results['train_top_indicators'].append(train_top)
                results['test_top_indicators'].append(test_top)
                results['consistency_scores'].append(consistency)
                
            except Exception:
                pass
            
            start_idx += step
        
        # Summary statistics
        if results['consistency_scores']:
            results['avg_consistency'] = np.mean(results['consistency_scores'])
            results['consistency_std'] = np.std(results['consistency_scores'])
        else:
            results['avg_consistency'] = 0
            results['consistency_std'] = 0
        
        return results
    
    def _get_importance(self, result: Dict) -> Dict[str, float]:
        """Extract importance as dict."""
        if 'importance' not in result:
            return {}
        
        imp = result['importance']
        if isinstance(imp, pd.Series):
            return imp.to_dict()
        elif isinstance(imp, dict):
            return imp
        return {}
    
    def _get_top_n(self, importance: Dict[str, float], n: int) -> List[str]:
        """Get top N indicators by importance."""
        return sorted(importance.keys(), key=lambda x: importance[x], reverse=True)[:n]
    
    # =========================================================================
    # OUT-OF-SAMPLE TEST
    # =========================================================================
    
    def out_of_sample_test(
        self,
        lens,
        data: pd.DataFrame,
        split_date: str = '2015-01-01'
    ) -> Dict[str, Any]:
        """
        Train on data before split_date, test on data after.
        
        Questions:
        - Do importance rankings persist out-of-sample?
        - Do detected relationships hold?
        """
        split = pd.Timestamp(split_date)
        
        train_data = data[data.index < split]
        test_data = data[data.index >= split]
        
        if len(train_data) < 100 or len(test_data) < 100:
            return {'error': 'Insufficient data for split'}
        
        # Analyze both periods
        train_result = lens.analyze(train_data)
        test_result = lens.analyze(test_data)
        
        train_importance = self._get_importance(train_result)
        test_importance = self._get_importance(test_result)
        
        # Compute rank correlation
        common_indicators = set(train_importance.keys()) & set(test_importance.keys())
        
        if len(common_indicators) < 3:
            return {'error': 'Too few common indicators'}
        
        train_ranks = {ind: i for i, ind in enumerate(sorted(
            common_indicators, key=lambda x: train_importance[x], reverse=True
        ))}
        test_ranks = {ind: i for i, ind in enumerate(sorted(
            common_indicators, key=lambda x: test_importance[x], reverse=True
        ))}
        
        # Spearman correlation
        train_r = np.array([train_ranks[ind] for ind in common_indicators])
        test_r = np.array([test_ranks[ind] for ind in common_indicators])
        
        from scipy.stats import spearmanr
        rank_corr, p_value = spearmanr(train_r, test_r)
        
        # Top-N overlap
        train_top5 = self._get_top_n(train_importance, 5)
        test_top5 = self._get_top_n(test_importance, 5)
        top5_overlap = len(set(train_top5) & set(test_top5)) / 5
        
        return {
            'train_period': (train_data.index[0], train_data.index[-1]),
            'test_period': (test_data.index[0], test_data.index[-1]),
            'rank_correlation': rank_corr,
            'rank_p_value': p_value,
            'top5_overlap': top5_overlap,
            'train_top5': train_top5,
            'test_top5': test_top5,
            'significant': p_value < 0.05,
        }
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_backtest_report(self, result: BacktestResult):
        """Print detailed backtest report."""
        print("\n" + "="*70)
        print(f"HISTORICAL BACKTEST: {result.lens_name}")
        print("="*70 + "\n")
        
        print(f"Detection Rate: {result.detection_rate:.1%} ({result.detected_events}/{result.total_events} events)")
        print(f"Average Lead Time: {result.avg_lead_time:.1f} days")
        print(f"False Positive Rate: {result.false_positive_rate:.1%} ({result.false_positives} signals)")
        
        print("\n" + "-"*40)
        print("EVENT DETAILS")
        print("-"*40)
        
        # Group by type
        by_type = {}
        for er in result.event_results:
            if er.event_type not in by_type:
                by_type[er.event_type] = []
            by_type[er.event_type].append(er)
        
        for event_type, events in sorted(by_type.items()):
            detected = sum(1 for e in events if e.detected)
            print(f"\n{event_type.upper()} ({detected}/{len(events)} detected)")
            
            for er in sorted(events, key=lambda x: x.event_date):
                status = "✓" if er.detected else "✗"
                lead = f"+{er.lead_time_days}d" if er.detected and er.lead_time_days else ""
                print(f"  {status} {er.event_date.strftime('%Y-%m-%d')} {er.event_name} {lead}")
        
        print("\n" + "="*70)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def quick_backtest(lens, data: pd.DataFrame) -> Dict[str, float]:
    """
    Quick backtest returning key metrics.
    
    Returns:
        Dict with detection_rate, avg_lead_time, false_positive_rate
    """
    backtester = HistoricalBacktester()
    result = backtester.evaluate_regime_detection(lens, data)
    
    return {
        'detection_rate': result.detection_rate,
        'avg_lead_time': result.avg_lead_time,
        'false_positive_rate': result.false_positive_rate,
    }


if __name__ == '__main__':
    print("Historical Backtester")
    print("="*50)
    print(f"Loaded {len(MARKET_EVENTS)} market events")
    print(f"Loaded {len(ECONOMIC_EVENTS)} economic events")
    print("\nEvent types:")
    
    types = set()
    for _, (_, _, _, etype) in {**MARKET_EVENTS, **ECONOMIC_EVENTS}.items():
        types.add(etype)
    
    for t in sorted(types):
        print(f"  • {t}")
    
    print("\nUsage:")
    print("  backtester = HistoricalBacktester()")
    print("  result = backtester.evaluate_regime_detection(lens, data)")
    print("  backtester.print_backtest_report(result)")
