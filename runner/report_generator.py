"""
PRISM Report Generator
=======================

Generates HTML reports from analysis results using Jinja2 templates.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import webbrowser

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

logger = logging.getLogger(__name__)

# Template directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class ReportGenerator:
    """Generates HTML reports using Jinja2 templates."""

    # Map workflow types to templates
    TEMPLATE_MAP = {
        "regime_comparison": "reports/regime_comparison.html",
        "lens_validation": "reports/lens_validation.html",
        "temporal_analysis": "reports/generic.html",
        "daily_update": "reports/generic.html",
        "default": "reports/generic.html",
    }

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize the report generator.

        Args:
            templates_dir: Custom templates directory
        """
        if not JINJA2_AVAILABLE:
            raise ImportError(
                "Jinja2 is required for HTML reports. "
                "Install with: pip install jinja2"
            )

        self.templates_dir = templates_dir or TEMPLATES_DIR

        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml']),
        )

        # Add custom filters
        self.env.filters['round'] = lambda x, n=0: round(float(x), n) if x else 0

        logger.info(f"Report generator initialized with templates from: {self.templates_dir}")

    def get_template_for_workflow(self, workflow_type: str) -> str:
        """Get the template path for a workflow type."""
        return self.TEMPLATE_MAP.get(workflow_type, self.TEMPLATE_MAP["default"])

    def generate(
        self,
        results: Dict[str, Any],
        output_path: Optional[Path] = None,
        open_browser: bool = False
    ) -> Path:
        """
        Generate an HTML report from analysis results.

        Args:
            results: Analysis results dictionary
            output_path: Where to save the report (auto-generated if None)
            open_browser: Whether to open the report in browser

        Returns:
            Path to the generated report
        """
        # Determine workflow type
        workflow_type = results.get("workflow_type", results.get("workflow", "default"))
        template_name = self.get_template_for_workflow(workflow_type)

        logger.info(f"Generating report using template: {template_name}")

        # Load template
        try:
            template = self.env.get_template(template_name)
        except Exception as e:
            logger.warning(f"Template {template_name} not found, using generic")
            template = self.env.get_template("reports/generic.html")

        # Prepare context
        context = {
            **results,
            "generated_at": datetime.now().isoformat(),
        }

        # Render HTML
        html_content = template.render(**context)

        # Determine output path
        if output_path is None:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            output_path = output_dir / f"report_{workflow_type}_{timestamp}.html"

        # Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Report generated: {output_path}")

        # Open in browser if requested
        if open_browser:
            self.open_report(output_path)

        return output_path

    def open_report(self, report_path: Path) -> None:
        """Open a report in the default browser."""
        try:
            webbrowser.open(f"file://{report_path.absolute()}")
            logger.info(f"Opened report in browser: {report_path}")
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")

    def list_templates(self) -> Dict[str, str]:
        """List available report templates."""
        templates = {}
        reports_dir = self.templates_dir / "reports"

        if reports_dir.exists():
            for template_file in reports_dir.glob("*.html"):
                name = template_file.stem
                templates[name] = str(template_file.relative_to(self.templates_dir))

        return templates


def generate_report(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
    open_browser: bool = False
) -> Optional[Path]:
    """
    Convenience function to generate a report.

    Args:
        results: Analysis results
        output_path: Output file path
        open_browser: Open in browser after generation

    Returns:
        Path to report or None if Jinja2 not available
    """
    if not JINJA2_AVAILABLE:
        logger.warning("Jinja2 not available, skipping HTML report generation")
        return None

    try:
        generator = ReportGenerator()
        return generator.generate(results, output_path, open_browser)
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return None
