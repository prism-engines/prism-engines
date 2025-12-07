#!/usr/bin/env python3
"""
Phase 4 HTML Output System Verification
=========================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_jinja2_available():
    """Test Jinja2 is installed."""
    print("\n" + "=" * 50)
    print("TEST: Jinja2 Available")
    print("=" * 50)
    
    try:
        import jinja2
        print(f"✓ Jinja2 version: {jinja2.__version__}")
        return True
    except ImportError:
        print("❌ Jinja2 not installed")
        print("   Run: pip install jinja2")
        return False


def test_templates_exist():
    """Test template files exist."""
    print("\n" + "=" * 50)
    print("TEST: Templates Exist")
    print("=" * 50)
    
    templates_dir = PROJECT_ROOT / "templates"
    
    required = [
        "base.html",
        "reports/base_report.html",
        "reports/regime_comparison.html",
        "reports/lens_validation.html",
        "reports/generic.html",
    ]
    
    all_exist = True
    for template in required:
        path = templates_dir / template
        if path.exists():
            print(f"✓ {template}")
        else:
            print(f"❌ {template} MISSING")
            all_exist = False
    
    return all_exist


def test_report_generator_import():
    """Test report generator imports."""
    print("\n" + "=" * 50)
    print("TEST: Report Generator Import")
    print("=" * 50)
    
    try:
        from runner.report_generator import ReportGenerator, generate_report
        print("✓ ReportGenerator imported")
        print("✓ generate_report imported")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_report_generator_init():
    """Test report generator initialization."""
    print("\n" + "=" * 50)
    print("TEST: Report Generator Initialization")
    print("=" * 50)
    
    try:
        from runner.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        templates = generator.list_templates()
        
        print(f"✓ Generator initialized")
        print(f"✓ Found {len(templates)} templates: {list(templates.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generation():
    """Test actual report generation."""
    print("\n" + "=" * 50)
    print("TEST: Report Generation")
    print("=" * 50)
    
    try:
        from runner.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        
        # Test data
        results = {
            "panel": "market",
            "workflow": "regime_comparison",
            "workflow_type": "regime_comparison",
            "status": "completed",
            "timestamp": "2025-01-01T12:00:00",
            "duration": 1.5,
            "indicators_analyzed": 33,
            "lenses_run": ["pca", "clustering", "granger"],
            "similarities": {
                "2008": 0.72,
                "2020": 0.85,
                "2022": 0.61,
            },
            "closest_match": "2020",
            "comparison_periods": ["2008", "2020", "2022"],
        }
        
        # Generate report
        output_path = PROJECT_ROOT / "output" / "test_report.html"
        report_path = generator.generate(results, output_path)
        
        if report_path and report_path.exists():
            size = report_path.stat().st_size
            print(f"✓ Report generated: {report_path}")
            print(f"✓ File size: {size:,} bytes")
            
            # Check content
            content = report_path.read_text()
            checks = [
                ("<!DOCTYPE html>", "HTML doctype"),
                ("PRISM", "PRISM branding"),
                ("2020", "Closest match"),
                ("regime_comparison", "Workflow type"),
            ]
            
            for check, desc in checks:
                if check in content:
                    print(f"✓ Contains {desc}")
                else:
                    print(f"⚠ Missing {desc}")
            
            return True
        else:
            print("❌ Report not generated")
            return False
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_generates_html():
    """Test CLI generates HTML report."""
    print("\n" + "=" * 50)
    print("TEST: CLI Generates HTML")
    print("=" * 50)
    
    try:
        from runner import WorkflowExecutor, OutputManager
        
        executor = WorkflowExecutor()
        output = OutputManager()
        
        # Run a workflow
        results = executor.execute("market", "lens_validation")
        
        # Generate HTML
        run_dir = output.create_run_directory(prefix="test_cli")
        html_path = output.generate_html_report(results)
        
        if html_path and html_path.exists():
            print(f"✓ CLI workflow generated HTML: {html_path}")
            return True
        else:
            print("❌ HTML not generated from CLI")
            return False
            
    except Exception as e:
        print(f"❌ CLI HTML test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_templates():
    """Test different workflow templates."""
    print("\n" + "=" * 50)
    print("TEST: Multiple Templates")
    print("=" * 50)
    
    try:
        from runner.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        
        workflows = [
            ("regime_comparison", {"similarities": {"2020": 0.8}, "closest_match": "2020"}),
            ("lens_validation", {"engines_validated": 20, "engines_valid": 15, "validation_details": {}}),
            ("daily_update", {"update_date": "2025-01-01", "indicators_updated": 33}),
        ]
        
        all_ok = True
        for wf_type, extra_data in workflows:
            results = {
                "panel": "market",
                "workflow": wf_type,
                "workflow_type": wf_type,
                "status": "completed",
                "duration": 1.0,
                "indicators_analyzed": 33,
                **extra_data,
            }
            
            output_path = PROJECT_ROOT / "output" / f"test_{wf_type}.html"
            report_path = generator.generate(results, output_path)
            
            if report_path and report_path.exists():
                print(f"✓ {wf_type} template works")
            else:
                print(f"❌ {wf_type} template failed")
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"❌ Multiple template test failed: {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║             PRISM PHASE 4 VERIFICATION                       ║
    ║                                                              ║
    ║  Testing: HTML Output System                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        "jinja2": test_jinja2_available(),
        "templates": test_templates_exist(),
        "import": test_report_generator_import(),
        "init": test_report_generator_init(),
        "generation": test_report_generation(),
        "cli_html": test_cli_generates_html(),
        "multi_template": test_multiple_templates(),
    }
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 PHASE 4 (HTML OUTPUT) COMPLETE - ALL TESTS PASSED!")
    else:
        print("⚠️  PHASE 4 INCOMPLETE - SOME TESTS FAILED")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
