"""
Report Generator - Create human-readable reports from PRISM results
"""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

# Get directory where this script lives
_SCRIPT_DIR = Path(__file__).parent.resolve()


class ReportGenerator:
    """Generate markdown and HTML reports from PRISM analysis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (_SCRIPT_DIR / "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown(self, results: Dict[str, Any], domain: str = "financial") -> str:
        """
        Generate markdown report from analysis results.

        Args:
            results: PRISM analysis results
            domain: 'financial' or 'climate'

        Returns:
            Markdown string
        """
        report = []
        report.append(f"# PRISM Analysis Report - {domain.title()}")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Indicators Analyzed:** {results.get('n_indicators', 'N/A')}")
        report.append(f"\n**Observations:** {results.get('n_observations', 'N/A')}")

        # Top Indicators
        report.append("\n\n## Top Consensus Indicators\n")
        top_indicators = results.get("top_indicators", [])
        if top_indicators:
            report.append("| Rank | Indicator | Confidence | Votes |")
            report.append("|------|-----------|------------|-------|")
            for ind in top_indicators:
                report.append(
                    f"| {ind['rank']} | {ind['indicator']} | "
                    f"{ind['confidence']:.2f} | {ind['votes']} |"
                )
        else:
            report.append("No top indicators available.")

        # Lens Agreement
        report.append("\n\n## Lens Agreement\n")
        lenses_run = results.get("lenses_run", [])
        report.append(f"Lenses used: {', '.join(lenses_run)}")

        # Key Findings
        report.append("\n\n## Key Findings\n")
        consensus = results.get("consensus", {})
        if "n_unanimous_top10" in consensus:
            report.append(
                f"- **{consensus['n_unanimous_top10']}** indicators ranked in top 10 by all lenses"
            )
        if "n_high_confidence" in consensus:
            report.append(
                f"- **{consensus['n_high_confidence']}** indicators with high confidence (>0.7)"
            )

        return "\n".join(report)

    def save_report(
        self,
        results: Dict[str, Any],
        domain: str = "financial",
        format: str = "markdown"
    ) -> Path:
        """
        Save report to file.

        Args:
            results: Analysis results
            domain: Domain name
            format: 'markdown' or 'html'

        Returns:
            Path to saved report
        """
        domain_dir = self.output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            content = self.generate_markdown(results, domain)
            path = domain_dir / "latest_report.md"
            with open(path, "w") as f:
                f.write(content)
        else:
            # Simple HTML wrapper
            md_content = self.generate_markdown(results, domain)
            html = f"<html><body><pre>{md_content}</pre></body></html>"
            path = domain_dir / "latest_report.html"
            with open(path, "w") as f:
                f.write(html)

        return path
