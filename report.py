"""
PRISM Engine - Report Generation
=================================

Generates human-readable summaries and HTML reports.
"""

import pandas as pd
from typing import Dict, Any
from pathlib import Path
from datetime import datetime


def generate_summary(results: Dict, coherence: Dict, args) -> str:
    """
    Generate a plain text summary.
    This is what you read to understand the results.
    """
    lines = []
    
    lines.append("=" * 60)
    lines.append("PRISM ENGINE - ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Run configuration
    lines.append("CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"  Mode: {'Quick' if args.quick else 'Full'}")
    if args.since:
        lines.append(f"  Date filter: {args.since}+")
    if args.focus:
        lines.append(f"  Focus category: {args.focus}")
    lines.append(f"  Weighting: {args.weights}")
    lines.append(f"  Windows analyzed: {results.get('n_windows', 0)}")
    lines.append(f"  Lenses used: {len(results.get('lenses_used', []))}")
    lines.append("")
    
    # Coherence results
    lines.append("COHERENCE INDEX")
    lines.append("-" * 40)
    lines.append(f"  Overall: {coherence['overall']:.3f}")
    lines.append(f"  Trend: {coherence['trend']}")
    
    if coherence['overall'] > 0.7:
        lines.append("  Interpretation: HIGH - Lenses strongly agree")
        lines.append("                  (In markets, this often precedes regime shifts)")
    elif coherence['overall'] > 0.4:
        lines.append("  Interpretation: MODERATE - Reasonable agreement")
    else:
        lines.append("  Interpretation: LOW - Lenses diverge")
        lines.append("                  (Often indicates healthy diversity)")
    
    if coherence.get('warning'):
        lines.append("")
        lines.append(f"  ⚠️  WARNING: {coherence['warning']}")
    lines.append("")
    
    # Top indicators
    lines.append("TOP INDICATORS (Consensus Ranking)")
    lines.append("-" * 40)
    
    rankings = results.get("rankings", pd.Series())
    if len(rankings) > 0:
        for i, (indicator, score) in enumerate(rankings.head(15).items(), 1):
            lines.append(f"  {i:2}. {indicator:<30} ({score:.3f})")
    else:
        lines.append("  No rankings available")
    lines.append("")
    
    # Lens agreement
    lines.append("LENS AGREEMENT")
    lines.append("-" * 40)
    agreement = results.get("lens_agreement")
    if agreement is not None and len(agreement) > 0:
        # Find most and least agreeing pairs
        mask = ~pd.np.eye(len(agreement), dtype=bool)
        values = agreement.values[mask]
        
        lines.append(f"  Mean agreement: {values.mean():.3f}")
        lines.append(f"  Min agreement: {values.min():.3f}")
        lines.append(f"  Max agreement: {values.max():.3f}")
    lines.append("")
    
    # Coherence over time
    if len(coherence.get('timeseries', [])) > 0:
        lines.append("COHERENCE OVER TIME")
        lines.append("-" * 40)
        ts = coherence['timeseries']
        lines.append(f"  Earliest: {ts.iloc[0]:.3f} ({ts.index[0].strftime('%Y-%m-%d')})")
        lines.append(f"  Latest: {ts.iloc[-1]:.3f} ({ts.index[-1].strftime('%Y-%m-%d')})")
        
        if len(ts) > 2:
            change = ts.iloc[-1] - ts.iloc[0]
            lines.append(f"  Change: {change:+.3f}")
    lines.append("")
    
    lines.append("=" * 60)
    lines.append("END OF SUMMARY")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def generate_html_report(results: Dict, coherence: Dict, output_path: Path):
    """
    Generate a visual HTML report.
    Opens nicely in a browser.
    """
    
    rankings = results.get("rankings", pd.Series())
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PRISM Engine Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            color: #e94560;
            border-bottom: 2px solid #e94560;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #0f3460;
            background: #e94560;
            padding: 10px;
            border-radius: 5px;
        }}
        .metric {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #e94560;
        }}
        .metric-label {{
            color: #888;
            font-size: 0.9em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #0f3460;
        }}
        th {{
            background: #0f3460;
            color: #e94560;
        }}
        tr:hover {{
            background: #1f2940;
        }}
        .warning {{
            background: #4a3000;
            border-left: 4px solid #ffa500;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .interpretation {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🔷 PRISM Engine Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Coherence Index</h2>
    <div class="grid">
        <div class="metric">
            <div class="metric-value">{coherence['overall']:.3f}</div>
            <div class="metric-label">Overall Coherence</div>
        </div>
        <div class="metric">
            <div class="metric-value">{coherence['trend']}</div>
            <div class="metric-label">Trend</div>
        </div>
        <div class="metric">
            <div class="metric-value">{results.get('n_windows', 0)}</div>
            <div class="metric-label">Windows Analyzed</div>
        </div>
    </div>
    
    <div class="interpretation">
        <strong>Interpretation:</strong><br>
        {_get_interpretation(coherence['overall'])}
    </div>
    
    {"<div class='warning'>⚠️ " + coherence['warning'] + "</div>" if coherence.get('warning') else ""}
    
    <h2>Top 15 Indicators</h2>
    <table>
        <tr><th>Rank</th><th>Indicator</th><th>Score</th></tr>
        {"".join(f"<tr><td>{i+1}</td><td>{ind}</td><td>{score:.3f}</td></tr>" 
                 for i, (ind, score) in enumerate(rankings.head(15).items()))}
    </table>
    
    <h2>Analysis Details</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Lenses Used</td><td>{len(results.get('lenses_used', []))}</td></tr>
        <tr><td>Total Indicators</td><td>{len(rankings)}</td></tr>
        <tr><td>Time Windows</td><td>{results.get('n_windows', 0)}</td></tr>
    </table>
    
</body>
</html>
"""
    
    output_path.write_text(html)


def _get_interpretation(coherence: float) -> str:
    """Get human interpretation of coherence value."""
    if coherence > 0.8:
        return "Very high coherence - all lenses strongly agree. In market contexts, this level of agreement often precedes significant regime changes. Worth investigating what's driving this alignment."
    elif coherence > 0.6:
        return "High coherence - lenses are notably aligned. Pay attention to which indicators are driving consensus."
    elif coherence > 0.4:
        return "Moderate coherence - reasonable agreement between lenses. This is typical for stable periods."
    elif coherence > 0.2:
        return "Low coherence - lenses see different patterns. Often indicates a diverse, healthy system without dominant single factors."
    else:
        return "Very low coherence - lenses strongly disagree. May indicate noisy data, transition period, or need to check data quality."
