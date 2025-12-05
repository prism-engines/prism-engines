"""
Seismometer Calibration - Threshold tuning and backtesting

Tools for calibrating alert thresholds and validating detection performance
against known crisis events.
"""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Known market stress periods for backtesting
KNOWN_CRISIS_DATES = {
    # Format: 'name': (start_date, end_date)
    'gfc_peak': ('2008-09-15', '2008-11-30'),      # Lehman collapse
    'flash_crash': ('2010-05-06', '2010-05-10'),   # Flash crash
    'eu_debt': ('2011-07-01', '2011-10-31'),       # European debt crisis
    'taper_tantrum': ('2013-05-22', '2013-06-30'), # Taper tantrum
    'china_deval': ('2015-08-11', '2015-09-30'),   # China devaluation
    'brexit': ('2016-06-23', '2016-06-30'),        # Brexit vote
    'volmageddon': ('2018-02-02', '2018-02-12'),   # VIX spike
    'covid_crash': ('2020-02-20', '2020-03-23'),   # COVID market crash
    'svb_crisis': ('2023-03-08', '2023-03-15'),    # SVB collapse
}


def compute_alert_frequency(
    scores: pd.Series,
    threshold: float = 0.5
) -> float:
    """
    Compute how often stability drops below threshold.

    Useful for tuning thresholds to achieve desired alert frequency.

    Args:
        scores: Series of stability scores (higher = more stable)
        threshold: Stability threshold

    Returns:
        Fraction of observations below threshold
    """
    valid_scores = scores.dropna()

    if len(valid_scores) == 0:
        return 0.0

    below_threshold = (valid_scores < threshold).sum()

    return below_threshold / len(valid_scores)


def compute_alert_duration(
    scores: pd.Series,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute statistics about alert duration.

    Args:
        scores: Series of stability scores
        threshold: Alert threshold

    Returns:
        Dict with mean, median, max duration stats
    """
    valid_scores = scores.dropna()

    # Find alert periods (below threshold)
    in_alert = valid_scores < threshold

    # Group consecutive alert days
    alert_groups = (~in_alert).cumsum()
    alert_durations = []

    for group_id in alert_groups[in_alert].unique():
        mask = (alert_groups == group_id) & in_alert
        duration = mask.sum()
        if duration > 0:
            alert_durations.append(duration)

    if len(alert_durations) == 0:
        return {
            'mean_duration': 0.0,
            'median_duration': 0.0,
            'max_duration': 0.0,
            'n_alerts': 0,
        }

    return {
        'mean_duration': float(np.mean(alert_durations)),
        'median_duration': float(np.median(alert_durations)),
        'max_duration': float(np.max(alert_durations)),
        'n_alerts': len(alert_durations),
    }


def find_optimal_threshold(
    scores: pd.Series,
    crisis_dates: Optional[Dict[str, Tuple[str, str]]] = None,
    target_precision: float = 0.7,
    lead_days: int = 30,
    threshold_range: Tuple[float, float] = (0.3, 0.8),
    n_steps: int = 20
) -> Dict[str, Any]:
    """
    Find threshold that catches crises with acceptable false positive rate.

    Args:
        scores: Series of stability scores (higher = more stable)
        crisis_dates: Dict of crisis periods {name: (start, end)}
        target_precision: Desired precision (true alerts / total alerts)
        lead_days: Days before crisis to count as successful early warning
        threshold_range: Range of thresholds to search
        n_steps: Number of threshold steps to try

    Returns:
        Dict with optimal threshold and performance metrics
    """
    if crisis_dates is None:
        crisis_dates = KNOWN_CRISIS_DATES

    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    results = []

    for threshold in thresholds:
        metrics = backtest_alerts(
            scores,
            threshold,
            crisis_dates,
            lead_days=lead_days
        )

        if metrics.empty:
            continue

        # Compute precision: how many alerts were near actual crises
        n_crises_detected = metrics['detected'].sum()
        total_alert_days = (scores.dropna() < threshold).sum()

        # Estimate true positives (alert days within crisis or lead periods)
        true_positive_days = 0
        for _, row in metrics[metrics['detected']].iterrows():
            crisis_start = pd.to_datetime(row['crisis_start'])
            lead_start = crisis_start - pd.Timedelta(days=lead_days)
            crisis_end = pd.to_datetime(row['crisis_end'])

            mask = (scores.index >= lead_start) & (scores.index <= crisis_end)
            true_positive_days += (scores[mask] < threshold).sum()

        precision = true_positive_days / max(total_alert_days, 1)
        recall = n_crises_detected / len(metrics)

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * precision * recall / max(precision + recall, 0.001),
            'n_crises_detected': n_crises_detected,
            'total_crises': len(metrics),
            'alert_frequency': compute_alert_frequency(scores, threshold),
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return {
            'optimal_threshold': 0.5,
            'precision': 0.0,
            'recall': 0.0,
            'all_results': results_df,
        }

    # Find threshold closest to target precision
    results_df['precision_diff'] = abs(results_df['precision'] - target_precision)
    best_idx = results_df['precision_diff'].idxmin()
    best_result = results_df.loc[best_idx]

    return {
        'optimal_threshold': float(best_result['threshold']),
        'precision': float(best_result['precision']),
        'recall': float(best_result['recall']),
        'f1_score': float(best_result['f1_score']),
        'alert_frequency': float(best_result['alert_frequency']),
        'all_results': results_df,
    }


def backtest_alerts(
    scores: pd.Series,
    threshold: float,
    crisis_dates: Optional[Dict[str, Tuple[str, str]]] = None,
    lead_days: int = 30
) -> pd.DataFrame:
    """
    Analyze detection performance: did we alert before each crisis?

    Args:
        scores: Series of stability scores (higher = more stable)
        threshold: Alert threshold (alert when score < threshold)
        crisis_dates: Dict of crisis periods {name: (start, end)}
        lead_days: Days before crisis start to check for early warning

    Returns:
        DataFrame with detection results for each crisis
    """
    if crisis_dates is None:
        crisis_dates = KNOWN_CRISIS_DATES

    valid_scores = scores.dropna()

    if len(valid_scores) == 0:
        return pd.DataFrame()

    results = []

    for crisis_name, (start_date, end_date) in crisis_dates.items():
        crisis_start = pd.to_datetime(start_date)
        crisis_end = pd.to_datetime(end_date)
        lead_start = crisis_start - pd.Timedelta(days=lead_days)

        # Check if we have data for this period
        if crisis_start > valid_scores.index[-1]:
            continue
        if crisis_end < valid_scores.index[0]:
            continue

        # Look for alerts in lead period
        lead_mask = (valid_scores.index >= lead_start) & (valid_scores.index < crisis_start)
        lead_scores = valid_scores[lead_mask]

        if len(lead_scores) == 0:
            alerts_in_lead = False
            first_alert_date = None
            lead_days_before = None
            min_stability_in_lead = None
        else:
            alerts_in_lead = (lead_scores < threshold).any()
            if alerts_in_lead:
                alert_dates = lead_scores[lead_scores < threshold].index
                first_alert_date = alert_dates[0]
                lead_days_before = (crisis_start - first_alert_date).days
            else:
                first_alert_date = None
                lead_days_before = None
            min_stability_in_lead = lead_scores.min()

        # Look for alerts during crisis
        crisis_mask = (valid_scores.index >= crisis_start) & (valid_scores.index <= crisis_end)
        crisis_scores = valid_scores[crisis_mask]

        if len(crisis_scores) == 0:
            alerts_during = False
            min_stability_during = None
        else:
            alerts_during = (crisis_scores < threshold).any()
            min_stability_during = crisis_scores.min()

        results.append({
            'crisis_name': crisis_name,
            'crisis_start': start_date,
            'crisis_end': end_date,
            'detected': alerts_in_lead or alerts_during,
            'early_warning': alerts_in_lead,
            'first_alert_date': str(first_alert_date.date()) if first_alert_date else None,
            'lead_days_before': lead_days_before,
            'min_stability_in_lead': min_stability_in_lead,
            'alerts_during_crisis': alerts_during,
            'min_stability_during': min_stability_during,
        })

    return pd.DataFrame(results)


def analyze_false_positives(
    scores: pd.Series,
    threshold: float,
    crisis_dates: Optional[Dict[str, Tuple[str, str]]] = None,
    lead_days: int = 30
) -> pd.DataFrame:
    """
    Identify periods where alerts occurred but no crisis followed.

    Args:
        scores: Series of stability scores
        threshold: Alert threshold
        crisis_dates: Dict of crisis periods
        lead_days: Lead time window for valid alerts

    Returns:
        DataFrame with false positive alert periods
    """
    if crisis_dates is None:
        crisis_dates = KNOWN_CRISIS_DATES

    valid_scores = scores.dropna()

    # Build set of "protected" dates (crisis + lead periods)
    protected_dates = set()
    for start_date, end_date in crisis_dates.values():
        crisis_start = pd.to_datetime(start_date)
        crisis_end = pd.to_datetime(end_date)
        lead_start = crisis_start - pd.Timedelta(days=lead_days)

        date_range = pd.date_range(lead_start, crisis_end, freq='D')
        protected_dates.update(date_range)

    # Find alert dates
    alert_mask = valid_scores < threshold
    alert_dates = valid_scores[alert_mask].index

    # Filter to false positives (alerts not in protected periods)
    false_positive_dates = [d for d in alert_dates if d not in protected_dates]

    if len(false_positive_dates) == 0:
        return pd.DataFrame(columns=['start_date', 'end_date', 'duration_days', 'min_stability'])

    # Group consecutive false positive days
    fp_series = pd.Series(1, index=pd.DatetimeIndex(false_positive_dates))
    fp_series = fp_series.reindex(valid_scores.index, fill_value=0)

    # Find consecutive periods
    fp_groups = (fp_series == 0).cumsum()
    fp_periods = []

    for group_id in fp_groups[fp_series == 1].unique():
        mask = fp_groups == group_id
        period_dates = valid_scores.index[mask & (fp_series == 1)]

        if len(period_dates) == 0:
            continue

        fp_periods.append({
            'start_date': str(period_dates[0].date()),
            'end_date': str(period_dates[-1].date()),
            'duration_days': len(period_dates),
            'min_stability': valid_scores[period_dates].min(),
        })

    return pd.DataFrame(fp_periods)


def compute_detection_metrics(
    backtest_results: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute summary detection metrics from backtest results.

    Args:
        backtest_results: Output from backtest_alerts()

    Returns:
        Dict with detection rate, early warning rate, avg lead time
    """
    if backtest_results.empty:
        return {
            'detection_rate': 0.0,
            'early_warning_rate': 0.0,
            'avg_lead_days': 0.0,
            'n_crises': 0,
        }

    n_crises = len(backtest_results)
    n_detected = backtest_results['detected'].sum()
    n_early = backtest_results['early_warning'].sum()

    lead_days = backtest_results['lead_days_before'].dropna()
    avg_lead = lead_days.mean() if len(lead_days) > 0 else 0.0

    return {
        'detection_rate': n_detected / n_crises,
        'early_warning_rate': n_early / n_crises,
        'avg_lead_days': avg_lead,
        'n_crises': n_crises,
    }


def generate_calibration_report(
    scores: pd.Series,
    crisis_dates: Optional[Dict[str, Tuple[str, str]]] = None,
    thresholds: List[float] = None
) -> str:
    """
    Generate a text report of calibration analysis.

    Args:
        scores: Series of stability scores
        crisis_dates: Dict of crisis periods
        thresholds: List of thresholds to analyze

    Returns:
        Formatted report string
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    if crisis_dates is None:
        crisis_dates = KNOWN_CRISIS_DATES

    lines = [
        "=" * 60,
        "SEISMOMETER CALIBRATION REPORT",
        "=" * 60,
        f"Data range: {scores.index[0].date()} to {scores.index[-1].date()}",
        f"Total observations: {len(scores.dropna())}",
        "",
        "THRESHOLD ANALYSIS",
        "-" * 40,
    ]

    for threshold in thresholds:
        freq = compute_alert_frequency(scores, threshold)
        duration = compute_alert_duration(scores, threshold)
        backtest = backtest_alerts(scores, threshold, crisis_dates)
        metrics = compute_detection_metrics(backtest)

        lines.extend([
            f"\nThreshold: {threshold:.2f}",
            f"  Alert frequency: {freq:.1%}",
            f"  Avg alert duration: {duration['mean_duration']:.1f} days",
            f"  Detection rate: {metrics['detection_rate']:.1%}",
            f"  Early warning rate: {metrics['early_warning_rate']:.1%}",
            f"  Avg lead time: {metrics['avg_lead_days']:.1f} days",
        ])

    # Find optimal threshold
    optimal = find_optimal_threshold(scores, crisis_dates)

    lines.extend([
        "",
        "OPTIMAL THRESHOLD",
        "-" * 40,
        f"Threshold: {optimal['optimal_threshold']:.2f}",
        f"Precision: {optimal['precision']:.1%}",
        f"Recall: {optimal['recall']:.1%}",
        f"F1 Score: {optimal['f1_score']:.3f}",
        "",
        "CRISIS DETECTION DETAILS",
        "-" * 40,
    ])

    backtest = backtest_alerts(scores, optimal['optimal_threshold'], crisis_dates)
    for _, row in backtest.iterrows():
        status = "DETECTED" if row['detected'] else "MISSED"
        early = " (early warning)" if row['early_warning'] else ""
        lead = f" - {row['lead_days_before']:.0f} days ahead" if row['lead_days_before'] else ""
        lines.append(f"  {row['crisis_name']}: {status}{early}{lead}")

    lines.append("=" * 60)

    return "\n".join(lines)
