"""
Validation Report - Generate comprehensive validation reports
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class ValidationReport:
    """
    Generate comprehensive validation reports for PRISM analysis.

    Combines results from:
    - Permutation tests
    - Bootstrap analysis
    - Backtesting
    - Cross-lens validation
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = output_dir or Path("validation/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Any] = {}

    def add_permutation_results(self, results: Dict[str, Any]) -> None:
        """Add permutation test results."""
        self.results['permutation_tests'] = results

    def add_bootstrap_results(self, results: Dict[str, Any]) -> None:
        """Add bootstrap analysis results."""
        self.results['bootstrap_analysis'] = results

    def add_backtest_results(self, results: Dict[str, Any]) -> None:
        """Add backtesting results."""
        self.results['backtesting'] = results

    def add_cross_lens_results(self, results: Dict[str, Any]) -> None:
        """Add cross-lens validation results."""
        self.results['cross_lens_validation'] = results

    def compute_overall_score(self) -> Dict[str, Any]:
        """
        Compute overall validation score.

        Returns:
            Dictionary with overall scores and grades
        """
        scores = {}

        # Permutation test score (% of lenses with significant results)
        if 'permutation_tests' in self.results:
            perm = self.results['permutation_tests']
            if isinstance(perm, pd.DataFrame):
                scores['permutation'] = perm['significant'].mean()
            elif isinstance(perm, list):
                scores['permutation'] = np.mean([r.get('significant', False) for r in perm])

        # Bootstrap stability score
        if 'bootstrap_analysis' in self.results:
            boot = self.results['bootstrap_analysis']
            if 'individual_lens_stability' in boot:
                stabilities = [
                    v.get('mean_stability', 0)
                    for v in boot['individual_lens_stability'].values()
                ]
                scores['bootstrap_stability'] = np.mean(stabilities) if stabilities else 0

        # Backtest score (mean IC if available)
        if 'backtesting' in self.results:
            back = self.results['backtesting']
            if 'mean_ic' in back:
                # Normalize IC to 0-1 range (IC typically -0.5 to 0.5)
                scores['backtest'] = (back['mean_ic'] + 0.5) / 1.0

        # Cross-lens diversity score
        if 'cross_lens_validation' in self.results:
            cross = self.results['cross_lens_validation']
            if 'diversity_score' in cross:
                scores['diversity'] = cross['diversity_score']

        # Overall score (weighted average)
        weights = {
            'permutation': 0.3,
            'bootstrap_stability': 0.3,
            'backtest': 0.25,
            'diversity': 0.15
        }

        overall = 0
        total_weight = 0
        for key, weight in weights.items():
            if key in scores:
                overall += scores[key] * weight
                total_weight += weight

        if total_weight > 0:
            overall /= total_weight

        # Grade
        if overall >= 0.8:
            grade = 'A'
        elif overall >= 0.6:
            grade = 'B'
        elif overall >= 0.4:
            grade = 'C'
        elif overall >= 0.2:
            grade = 'D'
        else:
            grade = 'F'

        return {
            'component_scores': scores,
            'overall_score': overall,
            'grade': grade
        }

    def generate_markdown(self) -> str:
        """
        Generate markdown report.

        Returns:
            Markdown string
        """
        lines = []
        lines.append("# PRISM Validation Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Overall Score
        overall = self.compute_overall_score()
        lines.append("\n## Overall Validation Score")
        lines.append(f"\n**Grade: {overall['grade']}** (Score: {overall['overall_score']:.2f})")
        lines.append("\n### Component Scores")
        lines.append("| Component | Score |")
        lines.append("|-----------|-------|")
        for comp, score in overall['component_scores'].items():
            lines.append(f"| {comp.replace('_', ' ').title()} | {score:.3f} |")

        # Permutation Tests
        if 'permutation_tests' in self.results:
            lines.append("\n## Permutation Tests")
            perm = self.results['permutation_tests']
            if isinstance(perm, list):
                n_sig = sum(1 for r in perm if r.get('significant', False))
                lines.append(f"\n- **{n_sig}/{len(perm)}** lenses showed statistically significant rankings")
                lines.append("\n| Lens | p-value | Significant |")
                lines.append("|------|---------|-------------|")
                for r in perm:
                    sig = "✓" if r.get('significant', False) else "✗"
                    lines.append(f"| {r.get('lens', 'N/A')} | {r.get('p_value', 'N/A'):.4f} | {sig} |")

        # Bootstrap Analysis
        if 'bootstrap_analysis' in self.results:
            lines.append("\n## Bootstrap Stability Analysis")
            boot = self.results['bootstrap_analysis']
            if 'consensus_stability' in boot:
                cs = boot['consensus_stability']
                stable = cs.get('stable_consensus_indicators', [])
                lines.append(f"\n**Stable Consensus Indicators:** {len(stable)}")
                if stable:
                    lines.append(f"\nTop stable indicators: {', '.join(stable[:10])}")

        # Cross-Lens Validation
        if 'cross_lens_validation' in self.results:
            lines.append("\n## Cross-Lens Validation")
            cross = self.results['cross_lens_validation']

            lines.append(f"\n**Ensemble Diversity Score:** {cross.get('diversity_score', 0):.3f}")

            if 'stable_indicators' in cross:
                stable = cross['stable_indicators']
                stable_inds = stable.get('stable_indicators', [])
                lines.append(f"\n**Consistently Top-Ranked Indicators:** {len(stable_inds)}")
                if stable_inds:
                    lines.append(f"\n{', '.join(stable_inds[:10])}")

        # Recommendations
        lines.append("\n## Recommendations")
        recommendations = self._generate_recommendations(overall)
        for rec in recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def _generate_recommendations(self, overall: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recs = []
        scores = overall['component_scores']

        if scores.get('permutation', 1) < 0.5:
            recs.append("Consider increasing sample size - many lens results may not be statistically significant")

        if scores.get('bootstrap_stability', 1) < 0.5:
            recs.append("Rankings show high variability - focus on indicators that appear consistently across bootstrap samples")

        if scores.get('backtest', 0.5) < 0.4:
            recs.append("Historical predictive power is limited - consider using as descriptive tool rather than predictive")

        if scores.get('diversity', 1) < 0.3:
            recs.append("Lenses are highly correlated - consider using fewer lenses or adding diverse analytical approaches")

        if overall['grade'] in ['A', 'B']:
            recs.append("Validation results are strong - indicator rankings can be used with confidence")

        if not recs:
            recs.append("No specific recommendations - review individual test results for details")

        return recs

    def save_report(self, name: Optional[str] = None) -> Dict[str, Path]:
        """
        Save validation report in multiple formats.

        Args:
            name: Optional report name

        Returns:
            Dictionary with paths to saved files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"validation_report_{timestamp}"

        paths = {}

        # Save markdown
        md_path = self.output_dir / f"{name}.md"
        with open(md_path, 'w') as f:
            f.write(self.generate_markdown())
        paths['markdown'] = md_path

        # Save JSON (full results)
        json_path = self.output_dir / f"{name}.json"

        # Convert to JSON-serializable format
        def convert(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, Path):
                return str(obj)
            return obj

        serializable = json.loads(
            json.dumps(self.results, default=convert)
        )
        serializable['overall_score'] = self.compute_overall_score()

        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        paths['json'] = json_path

        logger.info(f"Validation report saved to {self.output_dir}")
        return paths


def run_full_validation(
    df: pd.DataFrame,
    lens_classes: List[type],
    output_dir: Optional[Path] = None,
    n_permutations: int = 500,
    n_bootstrap: int = 500
) -> ValidationReport:
    """
    Run full validation suite and generate report.

    Args:
        df: Input DataFrame
        lens_classes: List of lens classes to validate
        output_dir: Output directory for reports
        n_permutations: Number of permutations
        n_bootstrap: Number of bootstrap samples

    Returns:
        ValidationReport instance with all results
    """
    from .permutation_tests import run_permutation_tests
    from .bootstrap_analysis import run_bootstrap_analysis
    from .cross_lens_validator import CrossLensValidator

    report = ValidationReport(output_dir)

    logger.info("Running permutation tests...")
    perm_results = run_permutation_tests(df, lens_classes, n_permutations)
    report.add_permutation_results(perm_results.to_dict('records'))

    logger.info("Running bootstrap analysis...")
    boot_results = run_bootstrap_analysis(df, lens_classes, n_bootstrap)
    report.add_bootstrap_results(boot_results)

    logger.info("Running cross-lens validation...")
    cross_validator = CrossLensValidator()
    cross_results = cross_validator.validate_all(df, lens_classes)
    report.add_cross_lens_results(cross_results)

    return report
