"""
Stress Test Suite
=================

Test framework performance against historical crisis events.
Critical for establishing practical value and actuarial credibility.

Features:
- Run analysis over known crisis periods
- Compute detection lead time
- Track false positive rates
- Generate comprehensive stress test reports
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yaml
from pathlib import Path


@dataclass
class HistoricalEvent:
    """Definition of a historical crisis event."""
    id: str
    name: str
    event_date: datetime
    event_type: str
    severity: str  # 'moderate', 'severe', 'extreme'
    drawdown_pct: Optional[float] = None
    recovery_days: Optional[int] = None
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    notes: str = ""


@dataclass
class StressTestResult:
    """Result from stress testing one event."""
    event_id: str
    event_name: str
    event_date: datetime

    # Detection results
    signal_detected: bool
    detection_date: Optional[datetime] = None
    lead_time_days: Optional[int] = None  # Positive = early warning

    # Signal quality
    warning_signals: List[Dict] = field(default_factory=list)
    max_signal_strength: float = 0.0

    # Performance
    regime_before: Optional[str] = None
    regime_during: Optional[str] = None
    transition_detected: bool = False

    # False alarms
    false_alarms_prior_year: int = 0
    false_alarm_dates: List[datetime] = field(default_factory=list)

    # Impact analysis
    max_drawdown_actual: Optional[float] = None
    max_drawdown_if_acted: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_name': self.event_name,
            'event_date': self.event_date.isoformat() if self.event_date else None,
            'signal_detected': self.signal_detected,
            'detection_date': self.detection_date.isoformat() if self.detection_date else None,
            'lead_time_days': self.lead_time_days,
            'max_signal_strength': round(self.max_signal_strength, 4),
            'regime_before': self.regime_before,
            'regime_during': self.regime_during,
            'false_alarms_prior_year': self.false_alarms_prior_year,
            'max_drawdown_actual': self.max_drawdown_actual,
            'max_drawdown_if_acted': self.max_drawdown_if_acted
        }


@dataclass
class StressTestReport:
    """Comprehensive stress test report."""
    n_events: int
    events_tested: List[str]
    results: List[StressTestResult]

    # Aggregate metrics
    detection_rate: float  # Fraction of events detected
    average_lead_time: Optional[float]
    total_false_alarms: int
    false_alarm_rate: float  # Per year

    # Performance summary
    total_drawdown_avoided: float
    value_added: float  # Relative to buy-and-hold

    def summary(self) -> Dict[str, Any]:
        return {
            'n_events': self.n_events,
            'detection_rate': f"{self.detection_rate:.1%}",
            'average_lead_time_days': round(self.average_lead_time, 1) if self.average_lead_time else None,
            'total_false_alarms': self.total_false_alarms,
            'false_alarm_rate_per_year': round(self.false_alarm_rate, 2),
            'value_added': f"{self.value_added:.1%}"
        }


class HistoricalEventRegistry:
    """
    Registry of historical crisis events for stress testing.
    """

    DEFAULT_EVENTS = [
        {
            'id': 'crash_1987',
            'name': 'Black Monday',
            'event_date': '1987-10-19',
            'event_type': 'market_crash',
            'severity': 'extreme',
            'drawdown_pct': 22.6,
            'recovery_days': 451
        },
        {
            'id': 'ltcm_1998',
            'name': 'LTCM Crisis',
            'event_date': '1998-08-17',
            'event_type': 'liquidity_crisis',
            'severity': 'severe',
            'drawdown_pct': 19.3,
            'recovery_days': 58
        },
        {
            'id': 'dotcom_2000',
            'name': 'Dot-com Bubble',
            'date_start': '2000-03-10',
            'date_end': '2002-10-09',
            'event_type': 'bubble_burst',
            'severity': 'severe',
            'drawdown_pct': 49.1
        },
        {
            'id': 'gfc_2008',
            'name': 'Global Financial Crisis',
            'date_start': '2007-10-09',
            'date_end': '2009-03-09',
            'event_type': 'systemic_crisis',
            'severity': 'extreme',
            'drawdown_pct': 56.8
        },
        {
            'id': 'flash_crash_2010',
            'name': 'Flash Crash',
            'event_date': '2010-05-06',
            'event_type': 'market_crash',
            'severity': 'moderate',
            'drawdown_pct': 9.2,
            'recovery_days': 1
        },
        {
            'id': 'covid_2020',
            'name': 'COVID Crash',
            'event_date': '2020-02-19',
            'event_type': 'exogenous_shock',
            'severity': 'severe',
            'drawdown_pct': 33.9,
            'recovery_days': 148
        },
        {
            'id': 'rate_shock_2022',
            'name': 'Rate Shock',
            'date_start': '2022-01-03',
            'date_end': '2022-10-12',
            'event_type': 'monetary_tightening',
            'severity': 'moderate',
            'drawdown_pct': 25.4
        }
    ]

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize event registry.

        Args:
            registry_path: Path to YAML file with events (uses defaults if None)
        """
        self.events: Dict[str, HistoricalEvent] = {}

        if registry_path and Path(registry_path).exists():
            self._load_from_yaml(registry_path)
        else:
            self._load_defaults()

    def _load_defaults(self):
        """Load default events."""
        for event_data in self.DEFAULT_EVENTS:
            self._add_event(event_data)

    def _load_from_yaml(self, path: str):
        """Load events from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        for event_data in data.get('events', []):
            self._add_event(event_data)

    def _add_event(self, event_data: Dict):
        """Add event to registry."""
        # Parse dates
        event_date = event_data.get('event_date') or event_data.get('date_start')
        if isinstance(event_date, str):
            event_date = datetime.fromisoformat(event_date)

        date_start = event_data.get('date_start')
        if isinstance(date_start, str):
            date_start = datetime.fromisoformat(date_start)

        date_end = event_data.get('date_end')
        if isinstance(date_end, str):
            date_end = datetime.fromisoformat(date_end)

        event = HistoricalEvent(
            id=event_data['id'],
            name=event_data['name'],
            event_date=event_date,
            event_type=event_data.get('event_type', 'unknown'),
            severity=event_data.get('severity', 'moderate'),
            drawdown_pct=event_data.get('drawdown_pct'),
            recovery_days=event_data.get('recovery_days'),
            date_start=date_start,
            date_end=date_end,
            notes=event_data.get('notes', '')
        )

        self.events[event.id] = event

    def get_event(self, event_id: str) -> Optional[HistoricalEvent]:
        """Get event by ID."""
        return self.events.get(event_id)

    def get_events_in_range(self,
                            start_date: datetime,
                            end_date: datetime) -> List[HistoricalEvent]:
        """Get events within date range."""
        return [
            e for e in self.events.values()
            if start_date <= e.event_date <= end_date
        ]

    def list_events(self) -> List[str]:
        """List all event IDs."""
        return list(self.events.keys())


class StressTestRunner:
    """
    Run stress tests against historical events.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 signal_func: Callable[[pd.DataFrame], Dict],
                 event_registry: Optional[HistoricalEventRegistry] = None,
                 signal_threshold: float = 0.7):
        """
        Initialize stress test runner.

        Args:
            data: Historical data with DatetimeIndex
            signal_func: Function that produces warning signals from data
            event_registry: Registry of events (uses defaults if None)
            signal_threshold: Threshold for considering signal as "warning"
        """
        self.data = data
        self.signal_func = signal_func
        self.registry = event_registry or HistoricalEventRegistry()
        self.signal_threshold = signal_threshold

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

    def run_single_event(self,
                         event: HistoricalEvent,
                         lookback_days: int = 252,
                         lookahead_days: int = 63) -> StressTestResult:
        """
        Run stress test for a single event.

        Args:
            event: Historical event to test
            lookback_days: Days before event to analyze
            lookahead_days: Days after event to analyze

        Returns:
            StressTestResult
        """
        event_date = event.event_date

        # Define analysis window
        start_date = event_date - timedelta(days=lookback_days)
        end_date = event_date + timedelta(days=lookahead_days)

        # Get data for window
        window_data = self.data[
            (self.data.index >= start_date) &
            (self.data.index <= end_date)
        ]

        if len(window_data) == 0:
            return StressTestResult(
                event_id=event.id,
                event_name=event.name,
                event_date=event_date,
                signal_detected=False
            )

        # Run signal function on pre-event data
        pre_event_data = window_data[window_data.index < event_date]
        warning_signals = []
        detection_date = None
        max_signal = 0.0

        # Analyze rolling windows before event
        window_size = 21  # Trading days
        for i in range(window_size, len(pre_event_data)):
            window = pre_event_data.iloc[i-window_size:i]
            try:
                signals = self.signal_func(window)
                signal_strength = signals.get('signal_strength', 0)

                if signal_strength > max_signal:
                    max_signal = signal_strength

                if signal_strength >= self.signal_threshold:
                    signal_date = window.index[-1]
                    warning_signals.append({
                        'date': signal_date,
                        'strength': signal_strength,
                        'type': signals.get('signal_type', 'unknown')
                    })

                    if detection_date is None:
                        detection_date = signal_date
            except:
                pass

        # Compute lead time
        lead_time = None
        if detection_date is not None:
            lead_time = (event_date - detection_date).days

        # Count false alarms in prior year
        prior_year_start = event_date - timedelta(days=365)
        prior_year_end = event_date - timedelta(days=30)  # Exclude month before
        false_alarms = [
            s for s in warning_signals
            if prior_year_start <= s['date'] <= prior_year_end
        ]

        # Compute drawdown if signal was acted on
        if 'price' in window_data.columns or 'close' in window_data.columns:
            price_col = 'price' if 'price' in window_data.columns else 'close'
            prices = window_data[price_col]

            # Actual max drawdown
            peak = prices[:event_date].max()
            trough = prices[event_date:].min()
            actual_dd = (peak - trough) / peak if peak > 0 else 0

            # Drawdown if exited at signal
            if detection_date and detection_date in prices.index:
                exit_price = prices[detection_date]
                trough_after = prices[detection_date:].min()
                avoided_dd = (exit_price - trough_after) / exit_price if exit_price > 0 else 0
            else:
                avoided_dd = None
        else:
            actual_dd = event.drawdown_pct / 100 if event.drawdown_pct else None
            avoided_dd = None

        return StressTestResult(
            event_id=event.id,
            event_name=event.name,
            event_date=event_date,
            signal_detected=len(warning_signals) > 0,
            detection_date=detection_date,
            lead_time_days=lead_time,
            warning_signals=warning_signals,
            max_signal_strength=max_signal,
            false_alarms_prior_year=len(false_alarms),
            false_alarm_dates=[s['date'] for s in false_alarms],
            max_drawdown_actual=actual_dd,
            max_drawdown_if_acted=avoided_dd
        )

    def run_all_events(self, event_ids: Optional[List[str]] = None) -> StressTestReport:
        """
        Run stress tests for all (or specified) events.

        Args:
            event_ids: List of event IDs to test (all if None)

        Returns:
            StressTestReport
        """
        if event_ids is None:
            event_ids = self.registry.list_events()

        results = []
        data_start = self.data.index.min()
        data_end = self.data.index.max()

        for event_id in event_ids:
            event = self.registry.get_event(event_id)
            if event is None:
                continue

            # Check if event is within data range
            if event.event_date < data_start or event.event_date > data_end:
                continue

            result = self.run_single_event(event)
            results.append(result)

        # Compute aggregate metrics
        n_events = len(results)
        if n_events == 0:
            return StressTestReport(
                n_events=0,
                events_tested=[],
                results=[],
                detection_rate=0,
                average_lead_time=None,
                total_false_alarms=0,
                false_alarm_rate=0,
                total_drawdown_avoided=0,
                value_added=0
            )

        detections = [r for r in results if r.signal_detected]
        detection_rate = len(detections) / n_events

        lead_times = [r.lead_time_days for r in detections if r.lead_time_days is not None]
        avg_lead_time = np.mean(lead_times) if lead_times else None

        total_false_alarms = sum(r.false_alarms_prior_year for r in results)

        # Years of data analyzed
        years = (data_end - data_start).days / 365.25
        false_alarm_rate = total_false_alarms / years if years > 0 else 0

        # Value calculation
        dd_avoided = sum(
            (r.max_drawdown_actual or 0) - (r.max_drawdown_if_acted or r.max_drawdown_actual or 0)
            for r in results if r.signal_detected
        )

        return StressTestReport(
            n_events=n_events,
            events_tested=[r.event_id for r in results],
            results=results,
            detection_rate=detection_rate,
            average_lead_time=avg_lead_time,
            total_false_alarms=total_false_alarms,
            false_alarm_rate=false_alarm_rate,
            total_drawdown_avoided=dd_avoided,
            value_added=dd_avoided / n_events if n_events > 0 else 0
        )


def print_stress_test_report(report: StressTestReport):
    """Pretty-print stress test report."""
    print("=" * 70)
    print("STRESS TEST REPORT")
    print("=" * 70)
    print(f"Events tested: {report.n_events}")
    print(f"Detection rate: {report.detection_rate:.1%}")
    print(f"Average lead time: {report.average_lead_time:.1f} days" if report.average_lead_time else "N/A")
    print(f"False alarm rate: {report.false_alarm_rate:.2f} per year")
    print()

    print("INDIVIDUAL RESULTS:")
    print("-" * 70)

    for result in report.results:
        status = "DETECTED" if result.signal_detected else "MISSED"
        print(f"\n{result.event_name} ({result.event_id}): [{status}]")
        print(f"  Event date: {result.event_date.strftime('%Y-%m-%d')}")
        if result.signal_detected:
            print(f"  Detection date: {result.detection_date.strftime('%Y-%m-%d')}")
            print(f"  Lead time: {result.lead_time_days} days")
            print(f"  Signal strength: {result.max_signal_strength:.3f}")
        print(f"  False alarms prior year: {result.false_alarms_prior_year}")
        if result.max_drawdown_actual:
            print(f"  Actual drawdown: {result.max_drawdown_actual:.1%}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    summary = report.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 60)
    print("Stress Test Suite - Demo")
    print("=" * 60)

    # Create synthetic historical data
    np.random.seed(42)

    dates = pd.date_range('2000-01-01', '2023-12-31', freq='B')
    n = len(dates)

    # Simulate price with known crisis periods
    returns = np.random.randn(n) * 0.01

    # Add crisis periods
    crisis_periods = [
        ('2008-09-01', '2009-03-01', -0.03),  # GFC
        ('2020-02-20', '2020-03-23', -0.05),  # COVID
    ]

    for start, end, ret in crisis_periods:
        mask = (dates >= start) & (dates <= end)
        returns[mask] += ret

    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'price': prices,
        'returns': returns
    }, index=dates)

    # Simple signal function (volatility-based)
    def simple_signal(window_data):
        if 'returns' in window_data.columns:
            vol = window_data['returns'].std() * np.sqrt(252)
            signal_strength = min(vol / 0.3, 1.0)  # Normalize
        else:
            signal_strength = 0

        return {
            'signal_strength': signal_strength,
            'signal_type': 'volatility'
        }

    print("\nInitializing registry and runner...")
    registry = HistoricalEventRegistry()
    runner = StressTestRunner(data, simple_signal, registry, signal_threshold=0.5)

    print(f"Events in registry: {registry.list_events()}")
    print(f"Data range: {data.index.min()} to {data.index.max()}")

    print("\nRunning stress tests...")
    report = runner.run_all_events()

    print_stress_test_report(report)

    print("\nTest completed!")
