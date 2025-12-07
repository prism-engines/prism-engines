#!/usr/bin/env python3
"""
PRISM Demo Script
==================

Demonstrates the main capabilities of the PRISM system.

Usage:
    python demo.py              # Run full demo
    python demo.py --quick      # Quick demo
"""

import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    """Print a header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_discovery():
    """Demonstrate discovery capabilities."""
    print_header("DEMO: Discovery System")
    
    # Engines
    from engines import EngineLoader
    engine_loader = EngineLoader()
    engines = engine_loader.list_engines()
    
    print(f"\n📊 Found {len(engines)} engines:")
    for i, engine in enumerate(engines[:5], 1):
        print(f"   {i}. {engine}")
    if len(engines) > 5:
        print(f"   ... and {len(engines) - 5} more")
    
    # Workflows
    from workflows import WorkflowLoader
    workflow_loader = WorkflowLoader()
    workflows = workflow_loader.list_workflows()
    
    print(f"\n🔄 Found {len(workflows)} workflows:")
    for i, wf in enumerate(workflows[:5], 1):
        print(f"   {i}. {wf}")
    if len(workflows) > 5:
        print(f"   ... and {len(workflows) - 5} more")
    
    # Panels
    from panels import PanelLoader
    panel_loader = PanelLoader()
    panels = panel_loader.list_panels()
    
    print(f"\n📋 Found {len(panels)} panels:")
    for panel in panels:
        print(f"   • {panel}")


def demo_panel_loading():
    """Demonstrate panel loading."""
    print_header("DEMO: Panel Loading")
    
    from panels import PanelLoader
    loader = PanelLoader()
    
    # Load market panel
    print("\n📈 Loading Market Panel...")
    panel = loader.load_panel("market")
    
    if panel:
        print(f"   Name: {panel.name}")
        indicators = panel.get_indicators()
        print(f"   Indicators: {len(indicators)}")
        
        print("\n   Sample indicators:")
        for ind in indicators[:5]:
            if isinstance(ind, dict):
                name = ind.get('name', ind.get('key', 'unknown'))
                print(f"   • {name}")
            else:
                print(f"   • {ind}")


def demo_workflow_execution():
    """Demonstrate workflow execution."""
    print_header("DEMO: Workflow Execution")
    
    from runner import WorkflowExecutor
    executor = WorkflowExecutor()
    
    print("\n⚙️ Executing lens_validation workflow...")
    print("   Panel: market")
    print("   This validates all available engines.\n")
    
    start = time.time()
    results = executor.execute("market", "lens_validation")
    duration = time.time() - start
    
    print(f"   ✅ Status: {results.get('status', 'unknown')}")
    print(f"   ⏱️ Duration: {duration:.2f}s")
    print(f"   📊 Indicators: {results.get('indicators_analyzed', 'N/A')}")
    
    if results.get('lenses_run'):
        print(f"   🔍 Lenses: {', '.join(results['lenses_run'][:5])}")


def demo_regime_comparison():
    """Demonstrate regime comparison."""
    print_header("DEMO: Regime Comparison")
    
    from runner import WorkflowExecutor
    executor = WorkflowExecutor()
    
    print("\n📊 Comparing current market to historical regimes...")
    print("   Comparison periods: 2008, 2020")
    print()
    
    start = time.time()
    results = executor.execute(
        "market", 
        "regime_comparison",
        comparison_periods=[2008, 2020]
    )
    duration = time.time() - start
    
    print(f"   ✅ Status: {results.get('status', 'unknown')}")
    print(f"   ⏱️ Duration: {duration:.2f}s")
    
    if results.get('similarities'):
        print("\n   Similarity Scores:")
        for period, score in results['similarities'].items():
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"   {period}: [{bar}] {score:.1%}")
    
    if results.get('closest_match'):
        print(f"\n   🎯 Closest Match: {results['closest_match']}")


def demo_report_generation():
    """Demonstrate HTML report generation."""
    print_header("DEMO: HTML Report Generation")
    
    from runner import ReportGenerator
    
    generator = ReportGenerator()
    templates = generator.list_templates()
    
    print(f"\n📄 Available templates: {list(templates.keys())}")
    
    # Generate sample report
    print("\n   Generating sample regime comparison report...")
    
    sample_results = {
        "panel": "market",
        "workflow": "regime_comparison",
        "workflow_type": "regime_comparison",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "duration": 2.5,
        "indicators_analyzed": 33,
        "lenses_run": ["pca", "clustering", "granger"],
        "similarities": {"2008": 0.72, "2020": 0.85},
        "closest_match": "2020",
    }
    
    output_path = PROJECT_ROOT / "output" / "demo_report.html"
    report_path = generator.generate(sample_results, output_path)
    
    print(f"   ✅ Report generated: {report_path}")
    print(f"   📁 Size: {report_path.stat().st_size:,} bytes")


def demo_web_server():
    """Demonstrate web server capabilities."""
    print_header("DEMO: Web Server")
    
    from runner import create_app
    
    app = create_app()
    client = app.test_client()
    
    print("\n🌐 Testing web server endpoints...")
    
    # Health check
    response = client.get('/health')
    data = response.get_json()
    print(f"\n   /health: {data.get('status')}")
    print(f"      Panels: {data.get('panels')}")
    print(f"      Workflows: {data.get('workflows')}")
    
    # API endpoints
    response = client.get('/api/panels')
    panels = response.get_json()
    print(f"\n   /api/panels: {len(panels)} panels available")
    
    response = client.get('/api/workflows')
    workflows = response.get_json()
    print(f"   /api/workflows: {len(workflows)} workflows available")
    
    print("""
   To start the web server:
   $ python prism_run.py --web
   
   Then open: http://localhost:5000
""")


def main():
    """Run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PRISM Demo")
    parser.add_argument("--quick", action="store_true", help="Quick demo")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                      PRISM ENGINE DEMO                           ║
║                                                                  ║
║  Demonstrating multi-lens analysis system capabilities           ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Always show discovery
        demo_discovery()
        
        if not args.quick:
            demo_panel_loading()
            demo_workflow_execution()
            demo_regime_comparison()
            demo_report_generation()
        
        demo_web_server()
        
        print_header("DEMO COMPLETE")
        print("""
Next steps:
  • Run tests:     python test_all_phases.py
  • Start web:     python prism_run.py --web
  • CLI mode:      python prism_run.py
  • Direct mode:   python prism_run.py --panel market --workflow regime_comparison
""")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
