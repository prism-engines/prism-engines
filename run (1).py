#!/usr/bin/env python3
"""
PRISM Engine
============

ONE COMMAND TO RUN EVERYTHING:

    python run.py              Full analysis
    python run.py --quick      Fast mode (fewer windows)
    python run.py --report     Generate HTML report
    python run.py --help       See all options

That's it. PRISM handles the rest.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# ============================================================
# SETUP
# ============================================================
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from engine.consolidate import load_and_consolidate
from engine.coherence import calculate_coherence
from engine.temporal import run_temporal_analysis
from engine.report import generate_summary, generate_html_report


def main():
    # ------------------------------------------------------------
    # PARSE ARGUMENTS (keep it simple)
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="PRISM Engine - Pattern Recognition through Integrated Signal Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                    Run full analysis
    python run.py --quick            Fast mode (fewer time windows)
    python run.py --since 2020       Analyze 2020 onward only
    python run.py --focus monetary   Focus on monetary indicators
    python run.py --report           Generate HTML report
    python run.py --weights equal    Override weighting method
        """
    )
    
    # Run modes
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: 3 windows instead of full history")
    parser.add_argument("--full", action="store_true", default=True,
                        help="Full analysis (default)")
    
    # Data filters
    parser.add_argument("--since", type=int, default=None,
                        help="Start year (e.g., --since 2020)")
    parser.add_argument("--focus", type=str, default=None,
                        choices=["monetary", "employment", "housing", "inflation", "sentiment", "real"],
                        help="Focus on one indicator category")
    
    # Weighting override
    parser.add_argument("--weights", type=str, default="auto",
                        choices=["auto", "equal", "variance", "entropy", "pca"],
                        help="Weighting method (default: auto-select best)")
    
    # Output options
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML report")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    
    args = parser.parse_args()
    
    # ------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------
    if not args.quiet:
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██████╗ ██╗███████╗███╗   ███╗                     ║
║   ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║                     ║
║   ██████╔╝██████╔╝██║███████╗██╔████╔██║                     ║
║   ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║                     ║
║   ██║     ██║  ██║██║███████║██║ ╚═╝ ██║                     ║
║   ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝                     ║
║                                                               ║
║   Pattern Recognition through Integrated Signal Methods       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """)
    
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Run started: {run_timestamp}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    if args.since:
        print(f"Date filter: {args.since} onward")
    if args.focus:
        print(f"Focus: {args.focus} indicators")
    print(f"Weighting: {args.weights}")
    print()
    
    # ------------------------------------------------------------
    # STEP 1: LOAD & CONSOLIDATE DATA
    # ------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading and Consolidating Data")
    print("=" * 60)
    
    data, categories = load_and_consolidate(
        data_dir=ROOT / "data" / "clean",
        weight_method=args.weights,
        focus_category=args.focus,
        since_year=args.since
    )
    
    print(f"  Loaded {len(data)} rows")
    print(f"  Categories: {list(categories.keys())}")
    print(f"  Total indicators: {sum(len(v) for v in categories.values())}")
    print()
    
    # ------------------------------------------------------------
    # STEP 2: RUN TEMPORAL ANALYSIS (all lenses, all windows)
    # ------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Running Analysis")
    print("=" * 60)
    
    n_windows = 3 if args.quick else None  # None = auto based on data
    
    results = run_temporal_analysis(
        data=data,
        categories=categories,
        n_windows=n_windows,
        weight_method=args.weights
    )
    
    print()
    
    # ------------------------------------------------------------
    # STEP 3: CALCULATE COHERENCE
    # ------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Calculating Coherence Index")
    print("=" * 60)
    
    coherence = calculate_coherence(results)
    
    print(f"  Overall Coherence: {coherence['overall']:.3f}")
    print(f"  Trend: {coherence['trend']}")
    if coherence.get('warning'):
        print(f"  ⚠️  {coherence['warning']}")
    print()
    
    # ------------------------------------------------------------
    # STEP 4: SAVE RESULTS
    # ------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Saving Results")
    print("=" * 60)
    
    # Always save to latest/
    latest_dir = ROOT / "results" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    # Also archive with timestamp
    archive_dir = ROOT / "results" / "archive" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    for out_dir in [latest_dir, archive_dir]:
        # Coherence over time
        coherence["timeseries"].to_csv(out_dir / "coherence.csv")
        
        # Rankings
        results["rankings"].to_csv(out_dir / "rankings.csv")
        
        # Summary
        summary_text = generate_summary(results, coherence, args)
        (out_dir / "summary.txt").write_text(summary_text)
    
    print(f"  Saved to: results/latest/")
    print(f"  Archived: results/archive/{archive_dir.name}/")
    print()
    
    # ------------------------------------------------------------
    # STEP 5: GENERATE REPORT (optional)
    # ------------------------------------------------------------
    if args.report:
        print("=" * 60)
        print("STEP 5: Generating Report")
        print("=" * 60)
        
        report_path = ROOT / "reports" / "latest_report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_html_report(results, coherence, report_path)
        print(f"  Report: {report_path}")
        print()
    
    # ------------------------------------------------------------
    # FINAL SUMMARY (always show this)
    # ------------------------------------------------------------
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " RESULTS SUMMARY ".center(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Coherence Index:  {coherence['overall']:>6.3f}".ljust(59) + "║")
    print(f"║  Trend:            {coherence['trend']:>6}".ljust(59) + "║")
    print(f"║  Windows Analyzed: {len(results['windows']):>6}".ljust(59) + "║")
    print("╟" + "─" * 58 + "╢")
    print("║  TOP 5 INDICATORS:".ljust(59) + "║")
    
    top5 = results["rankings"].head(5)
    for i, (indicator, score) in enumerate(top5.items(), 1):
        line = f"║    {i}. {indicator[:35]:<35} ({score:.3f})"
        print(line.ljust(59) + "║")
    
    print("╟" + "─" * 58 + "╢")
    
    # Interpretation
    if coherence['overall'] > 0.7:
        interp = "HIGH coherence - lenses strongly agree"
    elif coherence['overall'] > 0.4:
        interp = "MODERATE coherence - reasonable agreement"
    else:
        interp = "LOW coherence - lenses diverge (normal/healthy)"
    print(f"║  {interp:<56} ║")
    
    if coherence.get('warning'):
        print(f"║  ⚠️  {coherence['warning'][:52]:<52} ║")
    
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"Full results: results/latest/summary.txt")
    print()


if __name__ == "__main__":
    main()
