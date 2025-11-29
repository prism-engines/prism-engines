"""
VCF Validation Report Generator
================================

Aggregates all validation tests into a comprehensive report.
This is the main entry point for validating VCF lenses.

Usage:
    from validation_report import VCFValidator
    
    validator = VCFValidator()
    report = validator.full_validation(lenses, data)
    validator.print_full_report(report)
    validator.export_report(report, 'vcf_validation_report.md')
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings

# Import all validation modules
from synthetic_data_generator import SyntheticDataGenerator, generate_all_test_cases
from lens_validator import LensValidator, ValidationResult
from permutation_tests import PermutationTester, PermutationResult
from bootstrap_analysis import BootstrapAnalyzer, BootstrapCI, RankStability
from backtester import HistoricalBacktester, BacktestResult
from cross_lens_validator import CrossLensValidator, ConsensusResult


@dataclass
class ValidationReport:
    """Complete validation report for VCF."""
    timestamp: str
    n_lenses: int
    n_indicators: int
    data_range: Tuple[str, str]
    
    # Individual test results
    synthetic_tests: Dict[str, List[ValidationResult]] = field(default_factory=dict)
    permutation_tests: Dict[str, List[PermutationResult]] = field(default_factory=dict)
    bootstrap_analysis: Dict[str, Dict] = field(default_factory=dict)
    backtest_results: Dict[str, BacktestResult] = field(default_factory=dict)
    cross_lens_agreement: Dict[str, Any] = field(default_factory=dict)
    
    # Summary scores
    synthetic_score: float = 0.0
    significance_score: float = 0.0
    stability_score: float = 0.0
    backtest_score: float = 0.0
    agreement_score: float = 0.0
    overall_score: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class VCFValidator:
    """
    Master validator that runs all validation tests and generates comprehensive reports.
    """
    
    def __init__(
        self,
        n_permutations: int = 100,
        n_bootstrap: int = 500,
        random_seed: int = 42
    ):
        """
        Args:
            n_permutations: Number of permutations for significance testing
            n_bootstrap: Number of bootstrap samples for confidence intervals
            random_seed: For reproducibility
        """
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        
        # Initialize sub-validators
        self.synthetic_gen = SyntheticDataGenerator(seed=random_seed)
        self.lens_validator = LensValidator()
        self.permutation_tester = PermutationTester(n_permutations=n_permutations, random_seed=random_seed)
        self.bootstrap_analyzer = BootstrapAnalyzer(n_bootstrap=n_bootstrap, random_seed=random_seed)
        self.backtester = HistoricalBacktester()
        self.cross_validator = CrossLensValidator()
    
    # =========================================================================
    # FULL VALIDATION
    # =========================================================================
    
    def full_validation(
        self,
        lenses: List,
        data: pd.DataFrame,
        run_backtest: bool = True,
        run_synthetic: bool = True,
        run_permutation: bool = True,
        run_bootstrap: bool = True,
        run_cross_lens: bool = True,
        verbose: bool = True
    ) -> ValidationReport:
        """
        Run complete validation suite.
        
        Args:
            lenses: List of lens instances
            data: Real market data
            run_*: Flags to enable/disable specific tests
            verbose: Print progress
            
        Returns:
            ValidationReport with all results
        """
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            n_lenses=len(lenses),
            n_indicators=len(data.columns),
            data_range=(str(data.index[0]), str(data.index[-1]))
        )
        
        # 1. Synthetic Data Tests
        if run_synthetic:
            if verbose:
                print("Running synthetic data tests...")
            report.synthetic_tests = self._run_synthetic_tests(lenses)
            report.synthetic_score = self._compute_synthetic_score(report.synthetic_tests)
        
        # 2. Permutation Tests
        if run_permutation:
            if verbose:
                print("Running permutation tests...")
            report.permutation_tests = self._run_permutation_tests(lenses, data)
            report.significance_score = self._compute_significance_score(report.permutation_tests)
        
        # 3. Bootstrap Analysis
        if run_bootstrap:
            if verbose:
                print("Running bootstrap analysis...")
            report.bootstrap_analysis = self._run_bootstrap_analysis(lenses, data)
            report.stability_score = self._compute_stability_score(report.bootstrap_analysis)
        
        # 4. Historical Backtest
        if run_backtest:
            if verbose:
                print("Running historical backtest...")
            report.backtest_results = self._run_backtests(lenses, data)
            report.backtest_score = self._compute_backtest_score(report.backtest_results)
        
        # 5. Cross-Lens Agreement
        if run_cross_lens:
            if verbose:
                print("Running cross-lens validation...")
            report.cross_lens_agreement = self._run_cross_lens(lenses, data)
            report.agreement_score = self._compute_agreement_score(report.cross_lens_agreement)
        
        # Compute overall score
        scores = [
            (report.synthetic_score, 0.25),
            (report.significance_score, 0.20),
            (report.stability_score, 0.20),
            (report.backtest_score, 0.20),
            (report.agreement_score, 0.15),
        ]
        report.overall_score = sum(score * weight for score, weight in scores if score > 0)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        report.warnings = self._generate_warnings(report)
        
        if verbose:
            print(f"Validation complete. Overall score: {report.overall_score:.2f}")
        
        return report
    
    # =========================================================================
    # INDIVIDUAL TEST RUNNERS
    # =========================================================================
    
    def _run_synthetic_tests(self, lenses: List) -> Dict[str, List[ValidationResult]]:
        """Run synthetic data tests for each lens."""
        results = {}
        
        for lens in lenses:
            name = type(lens).__name__
            try:
                lens_results = self.lens_validator.validate_lens_suite(lens, self.synthetic_gen)
                results[name] = lens_results
            except Exception as e:
                warnings.warn(f"Synthetic test failed for {name}: {e}")
                results[name] = []
        
        return results
    
    def _run_permutation_tests(self, lenses: List, data: pd.DataFrame) -> Dict[str, List[PermutationResult]]:
        """Run permutation tests for each lens."""
        results = {}
        
        for lens in lenses:
            name = type(lens).__name__
            try:
                lens_results = self.permutation_tester.test_lens(lens, data)
                results[name] = lens_results
            except Exception as e:
                warnings.warn(f"Permutation test failed for {name}: {e}")
                results[name] = []
        
        return results
    
    def _run_bootstrap_analysis(self, lenses: List, data: pd.DataFrame) -> Dict[str, Dict]:
        """Run bootstrap analysis for each lens."""
        results = {}
        
        for lens in lenses:
            name = type(lens).__name__
            try:
                ci = self.bootstrap_analyzer.importance_confidence_intervals(lens, data)
                ranks = self.bootstrap_analyzer.rank_stability(lens, data)
                results[name] = {
                    'confidence_intervals': ci,
                    'rank_stability': ranks,
                }
            except Exception as e:
                warnings.warn(f"Bootstrap analysis failed for {name}: {e}")
                results[name] = {}
        
        return results
    
    def _run_backtests(self, lenses: List, data: pd.DataFrame) -> Dict[str, BacktestResult]:
        """Run historical backtests for each lens."""
        results = {}
        
        for lens in lenses:
            name = type(lens).__name__
            try:
                result = self.backtester.evaluate_regime_detection(lens, data)
                results[name] = result
            except Exception as e:
                warnings.warn(f"Backtest failed for {name}: {e}")
        
        return results
    
    def _run_cross_lens(self, lenses: List, data: pd.DataFrame) -> Dict[str, Any]:
        """Run cross-lens validation."""
        try:
            return self.cross_validator.analyze_agreement(lenses, data)
        except Exception as e:
            warnings.warn(f"Cross-lens validation failed: {e}")
            return {}
    
    # =========================================================================
    # SCORE COMPUTATION
    # =========================================================================
    
    def _compute_synthetic_score(self, results: Dict[str, List[ValidationResult]]) -> float:
        """Compute aggregate score from synthetic tests."""
        all_scores = []
        for lens_results in results.values():
            for r in lens_results:
                all_scores.append(r.score)
        
        return np.mean(all_scores) if all_scores else 0.0
    
    def _compute_significance_score(self, results: Dict[str, List[PermutationResult]]) -> float:
        """Compute score based on statistical significance."""
        all_significant = []
        for lens_results in results.values():
            for r in lens_results:
                all_significant.append(float(r.significant))
        
        return np.mean(all_significant) if all_significant else 0.0
    
    def _compute_stability_score(self, results: Dict[str, Dict]) -> float:
        """Compute score based on bootstrap stability."""
        # Use average CI width (narrower = more stable = higher score)
        all_widths = []
        
        for lens_data in results.values():
            if 'confidence_intervals' in lens_data:
                for ci in lens_data['confidence_intervals'].values():
                    if hasattr(ci, 'ci_width'):
                        all_widths.append(ci.ci_width)
        
        if not all_widths:
            return 0.0
        
        # Convert width to score (smaller width = higher score)
        median_width = np.median(all_widths)
        # Score of 1.0 if width < 0.1, 0.0 if width > 1.0
        score = max(0, min(1, 1 - (median_width - 0.1) / 0.9))
        return score
    
    def _compute_backtest_score(self, results: Dict[str, BacktestResult]) -> float:
        """Compute score from historical backtests."""
        if not results:
            return 0.0
        
        scores = []
        for result in results.values():
            # Combine detection rate and false positive penalty
            score = result.detection_rate * (1 - result.false_positive_rate * 0.5)
            scores.append(score)
        
        return np.mean(scores)
    
    def _compute_agreement_score(self, results: Dict[str, Any]) -> float:
        """Compute score from cross-lens agreement."""
        if 'consensus' not in results:
            return 0.0
        
        return max(0, results['consensus'].overall_agreement)
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    
    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate actionable recommendations based on results."""
        recs = []
        
        if report.synthetic_score < 0.5:
            recs.append("⚠️ Synthetic tests show weak lens performance. Review lens implementations.")
        
        if report.significance_score < 0.5:
            recs.append("⚠️ Many results not statistically significant. Consider increasing data size or reviewing signal strength.")
        
        if report.stability_score < 0.5:
            recs.append("⚠️ Bootstrap shows unstable rankings. Results may not be robust.")
        
        if report.backtest_score < 0.3:
            recs.append("⚠️ Poor historical event detection. Regime detection may need tuning.")
        
        if report.agreement_score < 0.3:
            recs.append("⚠️ Low cross-lens agreement. Different lenses give very different answers.")
        
        if report.overall_score >= 0.7:
            recs.append("✓ Overall validation looks good. Framework appears scientifically sound.")
        
        return recs
    
    def _generate_warnings(self, report: ValidationReport) -> List[str]:
        """Generate warnings about potential issues."""
        warns = []
        
        # Check for failed tests
        for name, results in report.synthetic_tests.items():
            failed = sum(1 for r in results if not r.passed)
            if failed > len(results) / 2:
                warns.append(f"⚠️ {name}: Majority of synthetic tests failed")
        
        # Check for non-significant results
        for name, results in report.permutation_tests.items():
            non_sig = sum(1 for r in results if not r.significant)
            if non_sig == len(results) and len(results) > 0:
                warns.append(f"⚠️ {name}: No statistically significant results")
        
        return warns
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_full_report(self, report: ValidationReport):
        """Print comprehensive validation report."""
        print("\n" + "="*70)
        print("VCF VALIDATION REPORT")
        print("="*70)
        print(f"Generated: {report.timestamp}")
        print(f"Lenses: {report.n_lenses}, Indicators: {report.n_indicators}")
        print(f"Data range: {report.data_range[0]} to {report.data_range[1]}")
        
        # Summary scores
        print("\n" + "-"*40)
        print("SUMMARY SCORES")
        print("-"*40)
        print(f"  Synthetic Tests:     {report.synthetic_score:.2f}")
        print(f"  Statistical Sig.:    {report.significance_score:.2f}")
        print(f"  Stability:           {report.stability_score:.2f}")
        print(f"  Historical Backtest: {report.backtest_score:.2f}")
        print(f"  Cross-Lens Agreement:{report.agreement_score:.2f}")
        print(f"  ─────────────────────────────")
        print(f"  OVERALL SCORE:       {report.overall_score:.2f}")
        
        grade = "A" if report.overall_score >= 0.8 else "B" if report.overall_score >= 0.6 else "C" if report.overall_score >= 0.4 else "D"
        print(f"  GRADE:               {grade}")
        
        # Warnings
        if report.warnings:
            print("\n" + "-"*40)
            print("WARNINGS")
            print("-"*40)
            for w in report.warnings:
                print(f"  {w}")
        
        # Recommendations
        if report.recommendations:
            print("\n" + "-"*40)
            print("RECOMMENDATIONS")
            print("-"*40)
            for r in report.recommendations:
                print(f"  {r}")
        
        # Detailed results by category
        self._print_synthetic_details(report)
        self._print_permutation_details(report)
        self._print_bootstrap_details(report)
        self._print_backtest_details(report)
        self._print_agreement_details(report)
        
        print("\n" + "="*70)
        print("END OF REPORT")
        print("="*70 + "\n")
    
    def _print_synthetic_details(self, report: ValidationReport):
        """Print synthetic test details."""
        if not report.synthetic_tests:
            return
        
        print("\n" + "-"*40)
        print("SYNTHETIC DATA TESTS")
        print("-"*40)
        
        for lens_name, results in report.synthetic_tests.items():
            if not results:
                continue
            
            passed = sum(1 for r in results if r.passed)
            print(f"\n{lens_name}: {passed}/{len(results)} passed")
            for r in results:
                status = "✓" if r.passed else "✗"
                print(f"  {status} {r.test_name}: {r.score:.2f}")
    
    def _print_permutation_details(self, report: ValidationReport):
        """Print permutation test details."""
        if not report.permutation_tests:
            return
        
        print("\n" + "-"*40)
        print("STATISTICAL SIGNIFICANCE")
        print("-"*40)
        
        for lens_name, results in report.permutation_tests.items():
            if not results:
                continue
            
            sig = sum(1 for r in results if r.significant)
            print(f"\n{lens_name}: {sig}/{len(results)} significant")
            for r in results:
                stars = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
                print(f"  {r.metric_name}: p={r.p_value:.3f}{stars}")
    
    def _print_bootstrap_details(self, report: ValidationReport):
        """Print bootstrap analysis details."""
        if not report.bootstrap_analysis:
            return
        
        print("\n" + "-"*40)
        print("BOOTSTRAP STABILITY")
        print("-"*40)
        
        for lens_name, data in report.bootstrap_analysis.items():
            if not data:
                continue
            
            if 'rank_stability' in data:
                ranks = data['rank_stability']
                if ranks:
                    print(f"\n{lens_name} - Rank Stability:")
                    sorted_ranks = sorted(ranks.values(), key=lambda x: x.median_rank)[:5]
                    for rs in sorted_ranks:
                        print(f"  {rs.indicator}: median #{rs.median_rank:.0f} [{rs.rank_ci_lower}-{rs.rank_ci_upper}]")
    
    def _print_backtest_details(self, report: ValidationReport):
        """Print backtest details."""
        if not report.backtest_results:
            return
        
        print("\n" + "-"*40)
        print("HISTORICAL BACKTEST")
        print("-"*40)
        
        for lens_name, result in report.backtest_results.items():
            print(f"\n{lens_name}:")
            print(f"  Detection rate: {result.detection_rate:.1%}")
            print(f"  Avg lead time: {result.avg_lead_time:.1f} days")
            print(f"  False positive rate: {result.false_positive_rate:.1%}")
    
    def _print_agreement_details(self, report: ValidationReport):
        """Print cross-lens agreement details."""
        if not report.cross_lens_agreement:
            return
        
        print("\n" + "-"*40)
        print("CROSS-LENS AGREEMENT")
        print("-"*40)
        
        if 'consensus' in report.cross_lens_agreement:
            consensus = report.cross_lens_agreement['consensus']
            print(f"\nOverall agreement: {consensus.overall_agreement:.2f}")
            
            if consensus.top_consensus_indicators:
                print("\nConsensus top indicators:")
                for ind, avg_rank, n in consensus.top_consensus_indicators[:5]:
                    print(f"  {ind}: avg rank {avg_rank:.1f}")
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_report(self, report: ValidationReport, filepath: str):
        """Export report to markdown file."""
        lines = [
            "# VCF Validation Report",
            "",
            f"**Generated:** {report.timestamp}",
            f"**Lenses:** {report.n_lenses}",
            f"**Indicators:** {report.n_indicators}",
            f"**Data Range:** {report.data_range[0]} to {report.data_range[1]}",
            "",
            "## Summary Scores",
            "",
            "| Test | Score |",
            "|------|-------|",
            f"| Synthetic Tests | {report.synthetic_score:.2f} |",
            f"| Statistical Significance | {report.significance_score:.2f} |",
            f"| Stability | {report.stability_score:.2f} |",
            f"| Historical Backtest | {report.backtest_score:.2f} |",
            f"| Cross-Lens Agreement | {report.agreement_score:.2f} |",
            f"| **OVERALL** | **{report.overall_score:.2f}** |",
            "",
        ]
        
        if report.warnings:
            lines.extend([
                "## Warnings",
                "",
            ])
            for w in report.warnings:
                lines.append(f"- {w}")
            lines.append("")
        
        if report.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for r in report.recommendations:
                lines.append(f"- {r}")
            lines.append("")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Report exported to {filepath}")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def quick_validate(lenses: List, data: pd.DataFrame) -> float:
    """
    Quick validation returning just the overall score.
    """
    validator = VCFValidator(n_permutations=50, n_bootstrap=200)
    report = validator.full_validation(lenses, data, verbose=False)
    return report.overall_score


if __name__ == '__main__':
    print("VCF Validation Report Generator")
    print("="*50)
    print("\nUsage:")
    print("  from validation_report import VCFValidator")
    print("  ")
    print("  validator = VCFValidator()")
    print("  report = validator.full_validation(lenses, data)")
    print("  validator.print_full_report(report)")
    print("  validator.export_report(report, 'report.md')")
