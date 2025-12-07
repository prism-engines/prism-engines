#!/usr/bin/env python3
"""
PRISM Dashboard - Option 2: Full Analytics
===========================================

Comprehensive dashboard with:
- Signal status gauges
- Interactive charts (using Chart.js)
- Indicator rankings table
- Lens weights visualization
- Historical trends

Usage:
    python dashboard_full.py
    open ~/gdrive/prism_output/dashboard_full.html
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    import os
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

from datetime import datetime
import json
import pandas as pd

from output_config import OUTPUT_DIR, DATA_DIR
from sql_schema_extension import PrismDB, ensure_schema, table_exists, get_table_row_count

OUTPUT_PATH = OUTPUT_DIR / "dashboard_full.html"


def generate_full_dashboard():
    """
    Generate full analytics dashboard with graceful degradation.

    Handles missing data gracefully:
    - If analysis_signals table missing/empty: shows warning banner
    - If calibration incomplete: displays degraded mode notice
    - Always renders successfully without crashing
    """
    # Ensure schema exists first
    ensure_schema()

    db = PrismDB()

    # Check calibration status for degraded mode detection
    calibration_complete = True
    degraded_reasons = []

    # Check if analysis_signals table exists and has data
    if not table_exists('analysis_signals'):
        calibration_complete = False
        degraded_reasons.append("analysis_signals table missing")
    elif get_table_row_count('analysis_signals') == 0:
        calibration_complete = False
        degraded_reasons.append("no signals data available")

    # Check if calibration_lenses has data
    if not table_exists('calibration_lenses') or get_table_row_count('calibration_lenses') == 0:
        calibration_complete = False
        degraded_reasons.append("lens calibration not completed")

    # Get data with safe fallbacks
    try:
        signals = db.get_latest_signals() if calibration_complete else pd.DataFrame()
    except Exception:
        signals = pd.DataFrame()
        calibration_complete = False
        degraded_reasons.append("failed to load signals")

    try:
        calibration = db.get_calibration()
    except Exception:
        calibration = {}

    try:
        danger_history = db.get_danger_history(days=90) if calibration_complete else pd.DataFrame()
    except Exception:
        danger_history = pd.DataFrame()
    
    # Get rankings - try multiple locations
    rankings = pd.DataFrame()
    for rankings_path in [
        OUTPUT_DIR / "calibrated_analysis" / "rankings.csv",
        OUTPUT_DIR / "tuned_analysis" / "rankings.csv",
    ]:
        if rankings_path.exists():
            try:
                rankings = pd.read_csv(rankings_path, index_col=0)
                break
            except Exception:
                pass

    # Prepare chart data
    lens_weights = calibration.get('lens_weights') or {}
    lens_names = json.dumps(list(lens_weights.keys()) if lens_weights else [])
    lens_values = json.dumps(list(lens_weights.values()) if lens_weights else [])
    
    # Rankings for chart
    if not rankings.empty:
        top_20 = rankings.head(20)
        ranking_names = json.dumps(top_20.index.tolist())
        ranking_values = json.dumps(top_20['consensus_rank'].tolist())
    else:
        ranking_names = '[]'
        ranking_values = '[]'
    
    # Signal counts
    danger_count = len(signals[signals['status'] == 'DANGER']) if not signals.empty else 0
    warning_count = len(signals[signals['status'] == 'WARNING']) if not signals.empty else 0
    normal_count = len(signals[signals['status'] == 'NORMAL']) if not signals.empty else 0
    
    # Build signal cards
    signal_cards = ""
    if not signals.empty:
        for _, row in signals.iterrows():
            status = row['status']
            if status == 'DANGER':
                card_class = 'danger'
                icon = '🔴'
            elif status == 'WARNING':
                card_class = 'warning'
                icon = '🟡'
            else:
                card_class = 'normal'
                icon = '🟢'
            
            signal_cards += f"""
            <div class="signal-card {card_class}">
                <div class="signal-icon">{icon}</div>
                <div class="signal-info">
                    <div class="signal-name">{row['signal_name']}</div>
                    <div class="signal-indicator">{row['indicator']}</div>
                </div>
                <div class="signal-rank">#{row['rank']:.0f}</div>
            </div>
            """
    
    # Build rankings table (handle both consensus_rank and weighted_rank columns)
    ranking_rows = ""
    if not rankings.empty:
        rank_col = 'consensus_rank' if 'consensus_rank' in rankings.columns else 'weighted_rank' if 'weighted_rank' in rankings.columns else None
        for i, (ind, row) in enumerate(rankings.head(25).iterrows(), 1):
            tier = int(row.get('tier', 0)) if pd.notna(row.get('tier')) else '?'
            tier_class = f"tier-{tier}" if isinstance(tier, int) else ""
            rank_value = row[rank_col] if rank_col and pd.notna(row.get(rank_col)) else i
            ranking_rows += f"""
            <tr class="{tier_class}">
                <td>{i}</td>
                <td>{ind}</td>
                <td>{rank_value:.1f}</td>
                <td>T{tier}</td>
            </tr>
            """

    # Build degraded mode banner
    degraded_banner = ""
    if not calibration_complete:
        reasons_text = ", ".join(degraded_reasons) if degraded_reasons else "calibration incomplete"
        degraded_banner = f"""
        <div class="degraded-banner">
            <div class="degraded-icon">⚠️</div>
            <div class="degraded-text">
                <strong>Calibration Not Yet Completed</strong>
                <p>Reason: {reasons_text}</p>
                <p>Run the calibration pipeline to enable full dashboard features.</p>
            </div>
        </div>
        """
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid #2a2a4a;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 5px;
        }}
        .header .subtitle {{
            color: #888;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .metric-label {{
            color: #888;
            margin-top: 5px;
        }}
        .metric-card.danger .metric-value {{ color: #E74C3C; }}
        .metric-card.warning .metric-value {{ color: #F39C12; }}
        .metric-card.normal .metric-value {{ color: #27AE60; }}
        .metric-card.info .metric-value {{ color: #3498DB; }}
        
        .signals-section {{
            margin-bottom: 30px;
        }}
        .signals-section h2 {{
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .signals-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }}
        .signal-card {{
            display: flex;
            align-items: center;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255,255,255,0.05);
        }}
        .signal-card.danger {{ border-left: 4px solid #E74C3C; }}
        .signal-card.warning {{ border-left: 4px solid #F39C12; }}
        .signal-card.normal {{ border-left: 4px solid #27AE60; }}
        .signal-icon {{
            font-size: 1.5em;
            margin-right: 15px;
        }}
        .signal-info {{
            flex: 1;
        }}
        .signal-name {{
            font-weight: bold;
        }}
        .signal-indicator {{
            font-size: 0.85em;
            color: #888;
        }}
        .signal-rank {{
            font-size: 1.5em;
            font-weight: bold;
            color: #888;
        }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        @media (max-width: 900px) {{
            .charts-row, .metrics-row {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
        }}
        .chart-card h3 {{
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        
        .table-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .table-card h3 {{
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            color: #888;
            font-weight: normal;
        }}
        tr.tier-1 {{ background: rgba(39, 174, 96, 0.1); }}
        tr.tier-2 {{ background: rgba(52, 152, 219, 0.1); }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        .degraded-banner {{
            background: linear-gradient(135deg, #f39c12 0%, #e74c3c 100%);
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .degraded-icon {{
            font-size: 2.5em;
        }}
        .degraded-text strong {{
            font-size: 1.2em;
            display: block;
            margin-bottom: 5px;
        }}
        .degraded-text p {{
            margin: 2px 0;
            opacity: 0.9;
        }}
        .empty-state {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 PRISM Analytics</h1>
        <p class="subtitle">Portfolio Risk Intelligence & Signal Monitor</p>
    </div>

    <div class="container">
        {degraded_banner}
        <div class="metrics-row">
            <div class="metric-card danger">
                <div class="metric-value">{danger_count}</div>
                <div class="metric-label">Danger Signals</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-value">{warning_count}</div>
                <div class="metric-label">Warning Signals</div>
            </div>
            <div class="metric-card normal">
                <div class="metric-value">{normal_count}</div>
                <div class="metric-label">Normal</div>
            </div>
            <div class="metric-card info">
                <div class="metric-value">{len(calibration.get('active_indicators') or [])}</div>
                <div class="metric-label">Active Indicators</div>
            </div>
        </div>
        
        <div class="signals-section">
            <h2>🚦 Current Signals</h2>
            <div class="signals-grid">
                {signal_cards if signal_cards else '<div class="empty-state">No signals available. Run calibration to generate signals.</div>'}
            </div>
        </div>
        
        <div class="charts-row">
            <div class="chart-card">
                <h3>⚖️ Lens Weights</h3>
                <div class="chart-container">
                    <canvas id="lensChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h3>📊 Top Indicators by Rank</h3>
                <div class="chart-container">
                    <canvas id="rankChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="table-card">
            <h3>📈 Full Rankings</h3>
            {f'''<table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Indicator</th>
                        <th>Consensus Rank</th>
                        <th>Tier</th>
                    </tr>
                </thead>
                <tbody>
                    {ranking_rows}
                </tbody>
            </table>''' if ranking_rows else '<div class="empty-state">No rankings available. Run analysis to generate rankings.</div>'}
        </div>
        
        <div class="footer">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Calibration v{calibration.get('version', '?')} | 
            PRISM Engine
        </div>
    </div>
    
    <script>
        // Lens weights chart
        new Chart(document.getElementById('lensChart'), {{
            type: 'bar',
            data: {{
                labels: {lens_names},
                datasets: [{{
                    label: 'Weight',
                    data: {lens_values},
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    y: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#eee' }}
                    }}
                }}
            }}
        }});
        
        // Rankings chart
        new Chart(document.getElementById('rankChart'), {{
            type: 'bar',
            data: {{
                labels: {ranking_names},
                datasets: [{{
                    label: 'Rank',
                    data: {ranking_values},
                    backgroundColor: function(context) {{
                        const value = context.raw;
                        if (value <= 5) return 'rgba(231, 76, 60, 0.7)';
                        if (value <= 10) return 'rgba(243, 156, 18, 0.7)';
                        return 'rgba(39, 174, 96, 0.7)';
                    }},
                    borderWidth: 0
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{
                        reverse: true,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    y: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#eee', font: {{ size: 10 }} }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    OUTPUT_PATH.write_text(html)
    print(f"✅ Dashboard saved: {OUTPUT_PATH}")
    print(f"   Open in browser: file://{OUTPUT_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("📊 PRISM FULL ANALYTICS DASHBOARD")
    print("=" * 60)
    generate_full_dashboard()
