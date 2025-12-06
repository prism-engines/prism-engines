#!/usr/bin/env python3
"""
PRISM Dashboard - Option 3: Terminal Style
==========================================

Minimal, terminal-inspired dashboard.
Fast to load, easy to read, no JavaScript.

Usage:
    python dashboard_terminal.py
    open ~/gdrive/prism_output/dashboard_term.html
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

OUTPUT_PATH = OUTPUT_DIR / "dashboard_term.html"


def generate_terminal_dashboard():
    """Generate terminal-style dashboard."""
    
    db = PrismDB()
    
    # Get data
    signals = db.get_latest_signals()
    calibration = db.get_calibration()
    
    # Rankings
    rankings_path = OUTPUT_DIR / "tuned_analysis" / "rankings.csv"
    if rankings_path.exists():
        rankings = pd.read_csv(rankings_path, index_col=0)
    else:
        rankings = pd.DataFrame()
    
    # Build status section
    danger_signals = signals[signals['status'] == 'DANGER'] if not signals.empty else pd.DataFrame()
    warning_signals = signals[signals['status'] == 'WARNING'] if not signals.empty else pd.DataFrame()
    
    status_lines = []
    
    if len(danger_signals) > 0:
        status_lines.append('<span class="red">██████████████████████████████████████████████████</span>')
        status_lines.append('<span class="red">█                    DANGER                      █</span>')
        status_lines.append('<span class="red">██████████████████████████████████████████████████</span>')
        status_lines.append('')
        for _, row in danger_signals.iterrows():
            status_lines.append(f'<span class="red">  🔴 {row["signal_name"]:20} │ {row["indicator"]:15} │ Rank {row["rank"]:.0f}</span>')
    elif len(warning_signals) > 0:
        status_lines.append('<span class="yellow">██████████████████████████████████████████████████</span>')
        status_lines.append('<span class="yellow">█                   WARNING                      █</span>')
        status_lines.append('<span class="yellow">██████████████████████████████████████████████████</span>')
    else:
        status_lines.append('<span class="green">██████████████████████████████████████████████████</span>')
        status_lines.append('<span class="green">█                    NORMAL                      █</span>')
        status_lines.append('<span class="green">██████████████████████████████████████████████████</span>')
    
    if len(warning_signals) > 0:
        status_lines.append('')
        for _, row in warning_signals.iterrows():
            status_lines.append(f'<span class="yellow">  🟡 {row["signal_name"]:20} │ {row["indicator"]:15} │ Rank {row["rank"]:.0f}</span>')
    
    status_section = '\n'.join(status_lines)
    
    # Build rankings section
    ranking_lines = []
    ranking_lines.append('┌─────┬──────────────────────────┬────────┬──────┐')
    ranking_lines.append('│  #  │ Indicator                │  Rank  │ Tier │')
    ranking_lines.append('├─────┼──────────────────────────┼────────┼──────┤')
    
    if not rankings.empty:
        for i, (ind, row) in enumerate(rankings.head(20).iterrows(), 1):
            tier = int(row.get('tier', 0)) if pd.notna(row.get('tier')) else '?'
            rank = row['consensus_rank']
            
            if rank <= 5:
                color = 'red'
            elif rank <= 10:
                color = 'yellow'
            else:
                color = 'white'
            
            line = f'│ {i:3} │ {ind:24} │ {rank:6.1f} │  T{tier}  │'
            ranking_lines.append(f'<span class="{color}">{line}</span>')
    
    ranking_lines.append('└─────┴──────────────────────────┴────────┴──────┘')
    rankings_section = '\n'.join(ranking_lines)
    
    # Build lens section
    lens_weights = calibration.get('lens_weights', {})
    lens_lines = []
    lens_lines.append('┌────────────────────┬────────┬──────────────────────────┐')
    lens_lines.append('│ Lens               │ Weight │ Bar                      │')
    lens_lines.append('├────────────────────┼────────┼──────────────────────────┤')
    
    for lens, weight in sorted(lens_weights.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(weight * 15)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        
        if weight >= 1.0:
            color = 'green'
        elif weight >= 0.7:
            color = 'yellow'
        else:
            color = 'red'
        
        line = f'│ {lens:18} │ {weight:6.2f} │ {bar} │'
        lens_lines.append(f'<span class="{color}">{line}</span>')
    
    lens_lines.append('└────────────────────┴────────┴──────────────────────────┘')
    lens_section = '\n'.join(lens_lines)
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM Terminal</title>
    <style>
        body {{
            background: #0a0a0a;
            color: #00ff00;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 14px;
            padding: 20px;
            line-height: 1.4;
        }}
        pre {{
            margin: 0;
            white-space: pre-wrap;
        }}
        .red {{ color: #ff4444; }}
        .yellow {{ color: #ffaa00; }}
        .green {{ color: #00ff00; }}
        .cyan {{ color: #00ffff; }}
        .white {{ color: #cccccc; }}
        .dim {{ color: #666666; }}
        .header {{
            color: #00ffff;
            margin-bottom: 20px;
        }}
        .section {{
            margin: 20px 0;
            padding: 10px;
            border: 1px solid #333;
        }}
        .section-title {{
            color: #00ffff;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
<pre>
<span class="header">
╔═══════════════════════════════════════════════════════════════════╗
║   ____  ____  ___ ____  __  __                                    ║
║  |  _ \|  _ \|_ _/ ___||  \/  |                                   ║
║  | |_) | |_) || |\___ \| |\/| |                                   ║
║  |  __/|  _ < | | ___) | |  | |                                   ║
║  |_|   |_| \_\___|____/|_|  |_|                                   ║
║                                                                   ║
║  Portfolio Risk Intelligence & Signal Monitor                     ║
╚═══════════════════════════════════════════════════════════════════╝
</span>

<span class="section-title">═══ SYSTEM STATUS ═══════════════════════════════════════════════════</span>

{status_section}

<span class="section-title">═══ INDICATOR RANKINGS ═══════════════════════════════════════════════</span>

{rankings_section}

<span class="section-title">═══ CALIBRATED LENS WEIGHTS ══════════════════════════════════════════</span>

{lens_section}

<span class="dim">
───────────────────────────────────────────────────────────────────────
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Calibration: v{calibration.get('version', '?')} │ {len(lens_weights)} lenses │ {len(calibration.get('active_indicators', []))} indicators
───────────────────────────────────────────────────────────────────────
</span>
</pre>
</body>
</html>
"""
    
    OUTPUT_PATH.write_text(html)
    print(f"✅ Dashboard saved: {OUTPUT_PATH}")
    print(f"   Open in browser: file://{OUTPUT_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("📊 PRISM TERMINAL DASHBOARD")
    print("=" * 60)
    generate_terminal_dashboard()
