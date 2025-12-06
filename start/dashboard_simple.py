#!/usr/bin/env python3
"""
PRISM Dashboard - Option 1: Simple Status
==========================================

Clean, minimal dashboard showing:
- Current signal status (traffic light)
- Top 10 indicators
- Recent signal history

Generates a single HTML file you can open in any browser.

Usage:
    python dashboard_simple.py
    open ~/gdrive/prism_output/dashboard.html
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
import pandas as pd

from output_config import OUTPUT_DIR, DATA_DIR
from sql_schema_extension import PrismDB

OUTPUT_PATH = OUTPUT_DIR / "dashboard.html"


def generate_simple_dashboard():
    """Generate simple status dashboard."""
    
    db = PrismDB()
    
    # Get latest data
    signals = db.get_latest_signals()
    calibration = db.get_calibration()
    danger_history = db.get_danger_history(days=30)
    
    # Count by status
    danger_count = len(signals[signals['status'] == 'DANGER']) if not signals.empty else 0
    warning_count = len(signals[signals['status'] == 'WARNING']) if not signals.empty else 0
    
    # Overall status
    if danger_count > 0:
        overall_status = "DANGER"
        status_color = "#E74C3C"
        status_emoji = "🔴"
    elif warning_count > 0:
        overall_status = "WARNING"
        status_color = "#F39C12"
        status_emoji = "🟡"
    else:
        overall_status = "NORMAL"
        status_color = "#27AE60"
        status_emoji = "🟢"
    
    # Build signal rows
    signal_rows = ""
    if not signals.empty:
        for _, row in signals.iterrows():
            status = row['status']
            if status == 'DANGER':
                badge = '<span class="badge danger">DANGER</span>'
            elif status == 'WARNING':
                badge = '<span class="badge warning">WARNING</span>'
            else:
                badge = '<span class="badge normal">NORMAL</span>'
            
            signal_rows += f"""
            <tr>
                <td>{row['indicator']}</td>
                <td>{row['signal_name']}</td>
                <td>{row['rank']:.0f}</td>
                <td>{badge}</td>
            </tr>
            """
    
    # Build history rows
    history_rows = ""
    if not danger_history.empty:
        for _, row in danger_history.head(10).iterrows():
            history_rows += f"""
            <tr>
                <td>{row['run_date']}</td>
                <td>{row['indicator']}</td>
                <td>{row['rank']:.0f}</td>
            </tr>
            """
    else:
        history_rows = '<tr><td colspan="3">No DANGER signals in last 30 days</td></tr>'
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .status-card {{
            background: {status_color};
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .status-emoji {{
            font-size: 4em;
            margin-bottom: 10px;
        }}
        .status-text {{
            font-size: 2em;
            font-weight: bold;
        }}
        .status-detail {{
            margin-top: 10px;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .card {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
        }}
        .card h2 {{
            margin-bottom: 15px;
            font-size: 1.3em;
            color: #fff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #2a2a4a;
        }}
        th {{
            color: #888;
            font-weight: normal;
            font-size: 0.9em;
        }}
        .badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .badge.danger {{
            background: #E74C3C;
        }}
        .badge.warning {{
            background: #F39C12;
        }}
        .badge.normal {{
            background: #27AE60;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }}
        .calibration-info {{
            background: #0f3460;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }}
        .calibration-info h3 {{
            font-size: 1em;
            margin-bottom: 10px;
            color: #888;
        }}
        .lens-bar {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .lens-name {{
            width: 120px;
            font-size: 0.85em;
        }}
        .lens-weight {{
            flex: 1;
            background: #1a1a2e;
            height: 20px;
            border-radius: 4px;
            overflow: hidden;
        }}
        .lens-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 PRISM</h1>
        <p class="subtitle">Portfolio Risk Intelligence & Signal Monitor</p>
        
        <div class="status-card">
            <div class="status-emoji">{status_emoji}</div>
            <div class="status-text">{overall_status}</div>
            <div class="status-detail">
                {danger_count} danger signal(s) • {warning_count} warning signal(s)
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📊 Current Signals</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Indicator</th>
                            <th>Name</th>
                            <th>Rank</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {signal_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>📈 DANGER History (30 days)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Indicator</th>
                            <th>Rank</th>
                        </tr>
                    </thead>
                    <tbody>
                        {history_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="calibration-info">
            <h3>⚙️ Calibration: v{calibration.get('version', '?')} | {len(calibration.get('lens_weights', {}))} lenses | {len(calibration.get('active_indicators', []))} indicators</h3>
        </div>
        
        <div class="footer">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | PRISM Engine
        </div>
    </div>
</body>
</html>
"""
    
    OUTPUT_PATH.write_text(html)
    print(f"✅ Dashboard saved: {OUTPUT_PATH}")
    print(f"   Open in browser: file://{OUTPUT_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("📊 PRISM SIMPLE DASHBOARD")
    print("=" * 60)
    generate_simple_dashboard()
