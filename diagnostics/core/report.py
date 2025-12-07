"""
Diagnostic Report Generator

Creates formatted reports from diagnostic results in multiple formats:
- Console (colored text)
- JSON
- HTML
- Markdown
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from diagnostics.core.base import DiagnosticResult, DiagnosticStatus, DiagnosticCategory


class DiagnosticReporter:
    """
    Generates formatted reports from diagnostic results.

    Supports multiple output formats and can save to files.
    """

    def __init__(self, results: Dict = None):
        """
        Initialize reporter with results.

        Args:
            results: Dictionary with 'results', 'summary', and 'by_category' keys
        """
        self._results = results or {'results': [], 'summary': {}, 'by_category': {}}

    def set_results(self, results: Dict) -> None:
        """Update the results to report on."""
        self._results = results

    def to_json(self, indent: int = 2) -> str:
        """
        Generate JSON report.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self._results, indent=indent, default=str)

    def to_markdown(self) -> str:
        """
        Generate Markdown report.

        Returns:
            Markdown formatted string
        """
        lines = []
        summary = self._results.get('summary', {})
        results = self._results.get('results', [])
        by_category = self._results.get('by_category', {})

        # Header
        lines.append("# PRISM Diagnostics Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total | {summary.get('total', 0)} |")
        lines.append(f"| Passed | {summary.get('passed', 0)} |")
        lines.append(f"| Failed | {summary.get('failed', 0)} |")
        lines.append(f"| Warnings | {summary.get('warnings', 0)} |")
        lines.append(f"| Errors | {summary.get('errors', 0)} |")
        lines.append(f"| Skipped | {summary.get('skipped', 0)} |")
        lines.append(f"| Success Rate | {summary.get('success_rate', 0):.1f}% |")
        lines.append(f"| Duration | {summary.get('total_duration_ms', 0):.0f}ms |")
        lines.append("")

        # Status
        if summary.get('all_passed'):
            lines.append("> **Status:** All diagnostics passed!")
        else:
            lines.append("> **Status:** Some diagnostics failed. See details below.")
        lines.append("")

        # Results by Category
        lines.append("## Results by Category")
        lines.append("")

        for category, cat_results in by_category.items():
            lines.append(f"### {category.title()}")
            lines.append("")
            lines.append("| Diagnostic | Status | Duration | Message |")
            lines.append("|------------|--------|----------|---------|")

            for r in cat_results:
                status = r.get('status', 'unknown').upper()
                status_emoji = self._status_emoji(status)
                duration = f"{r.get('duration_ms', 0):.0f}ms"
                message = r.get('message', '')[:50]  # Truncate long messages
                lines.append(f"| {r.get('name', 'unknown')} | {status_emoji} {status} | {duration} | {message} |")

            lines.append("")

        # Failed Details
        failed = [r for r in results if r.get('status') in ('fail', 'error', 'timeout')]
        if failed:
            lines.append("## Failed Diagnostics Details")
            lines.append("")
            for r in failed:
                lines.append(f"### {r.get('name', 'unknown')}")
                lines.append("")
                lines.append(f"**Status:** {r.get('status', 'unknown').upper()}")
                lines.append(f"**Message:** {r.get('message', 'No message')}")
                lines.append("")

                if r.get('suggestions'):
                    lines.append("**Suggestions:**")
                    for s in r['suggestions']:
                        lines.append(f"- {s}")
                    lines.append("")

                if r.get('details'):
                    lines.append("**Details:**")
                    lines.append("```json")
                    lines.append(json.dumps(r['details'], indent=2))
                    lines.append("```")
                    lines.append("")

                if r.get('error_trace'):
                    lines.append("**Error Trace:**")
                    lines.append("```")
                    lines.append(r['error_trace'])
                    lines.append("```")
                    lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """
        Generate HTML report.

        Returns:
            HTML string
        """
        summary = self._results.get('summary', {})
        results = self._results.get('results', [])
        by_category = self._results.get('by_category', {})

        # Calculate status for overall badge
        if summary.get('all_passed'):
            overall_status = 'pass'
            overall_text = 'All Passed'
        elif summary.get('failed', 0) > 0 or summary.get('errors', 0) > 0:
            overall_status = 'fail'
            overall_text = 'Issues Found'
        else:
            overall_status = 'warn'
            overall_text = 'Warnings'

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM Diagnostics Report</title>
    <style>
        :root {{
            --pass-color: #22c55e;
            --fail-color: #ef4444;
            --warn-color: #f59e0b;
            --error-color: #dc2626;
            --skip-color: #6b7280;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; margin: 2rem 0 1rem; border-bottom: 2px solid var(--border-color); padding-bottom: 0.5rem; }}
        h3 {{ font-size: 1.25rem; margin: 1.5rem 0 0.75rem; }}
        .timestamp {{ color: #64748b; font-size: 0.875rem; margin-bottom: 2rem; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .summary-card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card .value {{ font-size: 2rem; font-weight: bold; }}
        .summary-card .label {{ color: #64748b; font-size: 0.875rem; }}
        .summary-card.pass .value {{ color: var(--pass-color); }}
        .summary-card.fail .value {{ color: var(--fail-color); }}
        .summary-card.warn .value {{ color: var(--warn-color); }}
        .status-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .status-badge.pass {{ background: #dcfce7; color: #166534; }}
        .status-badge.fail {{ background: #fee2e2; color: #991b1b; }}
        .status-badge.warn {{ background: #fef3c7; color: #92400e; }}
        .status-badge.error {{ background: #fee2e2; color: #991b1b; }}
        .status-badge.skip {{ background: #f1f5f9; color: #475569; }}
        .status-badge.timeout {{ background: #fef3c7; color: #92400e; }}
        .overall-status {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            font-size: 1.25rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
        }}
        .overall-status.pass {{ background: #dcfce7; color: #166534; }}
        .overall-status.fail {{ background: #fee2e2; color: #991b1b; }}
        .overall-status.warn {{ background: #fef3c7; color: #92400e; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover {{ background: #f8fafc; }}
        .message {{ max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .details-card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid var(--fail-color);
        }}
        .details-card h4 {{ margin-bottom: 0.5rem; }}
        .suggestions {{ margin: 1rem 0; }}
        .suggestions li {{ margin-left: 1.5rem; color: #64748b; }}
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.875rem;
        }}
        .category-section {{ margin-bottom: 2rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PRISM Diagnostics Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="overall-status {overall_status}">{overall_text}</div>

        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{summary.get('total', 0)}</div>
                <div class="label">Total</div>
            </div>
            <div class="summary-card pass">
                <div class="value">{summary.get('passed', 0)}</div>
                <div class="label">Passed</div>
            </div>
            <div class="summary-card fail">
                <div class="value">{summary.get('failed', 0)}</div>
                <div class="label">Failed</div>
            </div>
            <div class="summary-card warn">
                <div class="value">{summary.get('warnings', 0)}</div>
                <div class="label">Warnings</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get('errors', 0)}</div>
                <div class="label">Errors</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get('success_rate', 0):.1f}%</div>
                <div class="label">Success Rate</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get('total_duration_ms', 0):.0f}ms</div>
                <div class="label">Duration</div>
            </div>
        </div>

        <h2>Results by Category</h2>
'''

        # Add category sections
        for category, cat_results in by_category.items():
            html += f'''
        <div class="category-section">
            <h3>{category.title()}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Diagnostic</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody>
'''
            for r in cat_results:
                status = r.get('status', 'unknown')
                html += f'''
                    <tr>
                        <td>{r.get('name', 'unknown')}</td>
                        <td><span class="status-badge {status}">{status.upper()}</span></td>
                        <td>{r.get('duration_ms', 0):.0f}ms</td>
                        <td class="message">{r.get('message', '')}</td>
                    </tr>
'''
            html += '''
                </tbody>
            </table>
        </div>
'''

        # Add failed details
        failed = [r for r in results if r.get('status') in ('fail', 'error', 'timeout')]
        if failed:
            html += '''
        <h2>Failed Diagnostics Details</h2>
'''
            for r in failed:
                html += f'''
        <div class="details-card">
            <h4>{r.get('name', 'unknown')}</h4>
            <p><strong>Status:</strong> <span class="status-badge {r.get('status', 'unknown')}">{r.get('status', 'unknown').upper()}</span></p>
            <p><strong>Message:</strong> {r.get('message', 'No message')}</p>
'''
                if r.get('suggestions'):
                    html += '''
            <div class="suggestions">
                <strong>Suggestions:</strong>
                <ul>
'''
                    for s in r['suggestions']:
                        html += f'                    <li>{s}</li>\n'
                    html += '''
                </ul>
            </div>
'''
                if r.get('error_trace'):
                    html += f'''
            <div>
                <strong>Error Trace:</strong>
                <pre>{r['error_trace']}</pre>
            </div>
'''
                html += '''
        </div>
'''

        html += '''
    </div>
</body>
</html>
'''
        return html

    def save(self, path: str, format: str = 'auto') -> str:
        """
        Save report to file.

        Args:
            path: Output file path
            format: 'json', 'html', 'md', or 'auto' (detect from extension)

        Returns:
            Path to saved file
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == 'auto':
            ext = path.suffix.lower()
            format_map = {'.json': 'json', '.html': 'html', '.md': 'md', '.markdown': 'md'}
            format = format_map.get(ext, 'json')

        # Generate content
        if format == 'json':
            content = self.to_json()
        elif format == 'html':
            content = self.to_html()
        elif format == 'md':
            content = self.to_markdown()
        else:
            content = self.to_json()

        # Write file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        return str(path)

    def _status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        emojis = {
            'PASS': '✓',
            'FAIL': '✗',
            'WARN': '⚠',
            'ERROR': '⚠',
            'SKIP': '○',
            'TIMEOUT': '⏱',
        }
        return emojis.get(status.upper(), '?')
