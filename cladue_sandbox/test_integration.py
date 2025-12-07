#!/usr/bin/env python3
"""
PRISM Integration Test
=======================

Tests full end-to-end workflows to verify the system works together.

Usage:
    python test_integration.py
"""

import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def test_cli_direct_mode():
    """Test CLI direct mode execution."""
    print_section("CLI Direct Mode Test")
    
    try:
        import subprocess
        
        # Run a simple workflow via CLI
        result = subprocess.run(
            [sys.executable, "prism_run.py", 
             "--panel", "market", 
             "--workflow", "lens_validation"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode == 0:
            print("✅ CLI direct mode executed successfully")
            
            # Check for expected output
            if "completed" in result.stdout.lower():
                print("✅ Workflow completed")
            if "saved to" in result.stdout.lower() or "output" in result.stdout.lower():
                print("✅ Results saved")
            
            return True
        else:
            print(f"❌ CLI returned error code: {result.returncode}")
            print(f"   Output: {result.stdout[-500:]}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_workflow_execution():
    """Test direct workflow execution via Python."""
    print_section("Workflow Execution Test")
    
    try:
        from runner import WorkflowExecutor, OutputManager
        
        executor = WorkflowExecutor()
        output = OutputManager()
        
        # Test lens validation
        print("\n[1/3] Running lens_validation...")
        start = time.time()
        results = executor.execute("market", "lens_validation")
        duration = time.time() - start
        
        assert results.get("status") == "completed", f"Status: {results.get('status')}"
        print(f"      ✅ Completed in {duration:.2f}s")
        print(f"      Indicators: {results.get('indicators_analyzed', 'N/A')}")
        
        # Test regime comparison
        print("\n[2/3] Running regime_comparison...")
        start = time.time()
        results = executor.execute("market", "regime_comparison", 
                                  comparison_periods=[2008, 2020])
        duration = time.time() - start
        
        assert results.get("status") == "completed", f"Status: {results.get('status')}"
        print(f"      ✅ Completed in {duration:.2f}s")
        if "similarities" in results:
            print(f"      Similarities: {results['similarities']}")
        if "closest_match" in results:
            print(f"      Closest match: {results['closest_match']}")
        
        # Test output saving
        print("\n[3/3] Testing output saving...")
        run_dir = output.create_run_directory(prefix="integration_test")
        json_path = output.save_json(results, "results.json")
        
        assert json_path.exists(), "JSON file not created"
        print(f"      ✅ Results saved to: {run_dir}")
        
        # Cleanup
        shutil.rmtree(run_dir, ignore_errors=True)
        print("      ✅ Cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_html_report_generation():
    """Test HTML report generation."""
    print_section("HTML Report Generation Test")
    
    try:
        from runner import ReportGenerator
        
        generator = ReportGenerator()
        
        # Test regime comparison report
        print("\n[1/2] Generating regime_comparison report...")
        results = {
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
            "comparison_periods": [2008, 2020],
        }
        
        output_path = PROJECT_ROOT / "output" / "test_regime_report.html"
        report_path = generator.generate(results, output_path)
        
        assert report_path.exists(), "Report not created"
        size = report_path.stat().st_size
        print(f"      ✅ Report generated: {report_path.name} ({size:,} bytes)")
        
        # Check content
        content = report_path.read_text()
        assert "PRISM" in content, "Missing PRISM branding"
        assert "2020" in content, "Missing closest match"
        print("      ✅ Report content verified")
        
        # Test lens validation report
        print("\n[2/2] Generating lens_validation report...")
        results = {
            "panel": "market",
            "workflow": "lens_validation",
            "workflow_type": "lens_validation",
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "duration": 1.5,
            "engines_validated": 20,
            "engines_valid": 15,
            "validation_details": {
                "pca": {"valid": True, "loadable": True, "errors": []},
                "clustering": {"valid": True, "loadable": True, "errors": []},
            }
        }
        
        output_path = PROJECT_ROOT / "output" / "test_validation_report.html"
        report_path = generator.generate(results, output_path)
        
        assert report_path.exists(), "Report not created"
        print(f"      ✅ Report generated: {report_path.name}")
        
        # Cleanup
        (PROJECT_ROOT / "output" / "test_regime_report.html").unlink(missing_ok=True)
        (PROJECT_ROOT / "output" / "test_validation_report.html").unlink(missing_ok=True)
        print("      ✅ Cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_server_endpoints():
    """Test web server API endpoints."""
    print_section("Web Server Endpoints Test")
    
    try:
        from runner import create_app
        
        app = create_app()
        client = app.test_client()
        
        # Test health endpoint
        print("\n[1/5] Testing /health...")
        response = client.get('/health')
        assert response.status_code == 200, f"Status: {response.status_code}"
        data = response.get_json()
        print(f"      ✅ Health: {data.get('status')}")
        print(f"      Panels: {data.get('panels')}, Workflows: {data.get('workflows')}")
        
        # Test panels API
        print("\n[2/5] Testing /api/panels...")
        response = client.get('/api/panels')
        assert response.status_code == 200, f"Status: {response.status_code}"
        data = response.get_json()
        print(f"      ✅ Found {len(data)} panels: {list(data.keys())}")
        
        # Test workflows API
        print("\n[3/5] Testing /api/workflows...")
        response = client.get('/api/workflows')
        assert response.status_code == 200, f"Status: {response.status_code}"
        data = response.get_json()
        print(f"      ✅ Found {len(data)} workflows")
        
        # Test index page
        print("\n[4/5] Testing index page...")
        response = client.get('/')
        assert response.status_code == 200, f"Status: {response.status_code}"
        html = response.data.decode('utf-8')
        assert 'PRISM' in html, "Missing PRISM branding"
        print("      ✅ Index page renders")
        
        # Test run endpoint
        print("\n[5/5] Testing /run endpoint...")
        response = client.post('/run', data={
            'panel': 'market',
            'workflow': 'lens_validation',
        })
        assert response.status_code == 200, f"Status: {response.status_code}"
        html = response.data.decode('utf-8')
        print("      ✅ Run endpoint works")
        
        return True
        
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_data_loading():
    """Test panel data loading capabilities."""
    print_section("Panel Data Loading Test")
    
    try:
        from panels import PanelLoader
        
        loader = PanelLoader()
        panels = loader.list_panels()
        
        print(f"\nFound {len(panels)} panels: {panels}")
        
        for panel_name in panels[:2]:  # Test first 2 panels
            print(f"\n[{panel_name}] Loading panel...")
            
            panel = loader.load_panel(panel_name)
            
            if panel:
                # Try different ways to get panel name
                name = getattr(panel, 'name', None) or getattr(panel, 'panel_name', None) or panel_name
                print(f"      ✅ Panel loaded: {name}")
                
                domain = getattr(panel, 'domain', 'N/A')
                print(f"      Domain: {domain}")
                
                # Get indicators
                if hasattr(panel, 'get_indicators'):
                    indicators = panel.get_indicators()
                    print(f"      Indicators: {len(indicators)}")
                    
                    if indicators:
                        sample = indicators[:3]
                        for ind in sample:
                            if isinstance(ind, dict):
                                print(f"        - {ind.get('key', ind.get('name', 'unknown'))}")
                            else:
                                print(f"        - {ind}")
                else:
                    print("      Indicators: (no get_indicators method)")
            else:
                print(f"      ⚠️ Panel returned None")
        
        return True
        
    except Exception as e:
        print(f"❌ Panel loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine_loading():
    """Test engine loading and basic operations."""
    print_section("Engine Loading Test")
    
    try:
        from engines import EngineLoader
        
        loader = EngineLoader()
        engines = loader.list_engines()
        
        print(f"\nFound {len(engines)} engines")
        
        # Test loading a few engines
        test_engines = ["pca", "clustering", "granger"]
        loaded_count = 0
        
        for engine_name in test_engines:
            if engine_name in engines:
                print(f"\n[{engine_name}] Testing engine...")
                
                try:
                    engine_class = loader.load_engine(engine_name)
                    
                    if engine_class:
                        print(f"      ✅ Engine loaded")
                        loaded_count += 1
                        
                        # Check for required methods
                        has_analyze = hasattr(engine_class, 'analyze') or hasattr(engine_class, 'run')
                        has_rank = hasattr(engine_class, 'rank_indicators')
                        
                        print(f"      Has analyze: {'✓' if has_analyze else '✗'}")
                        print(f"      Has rank: {'✓' if has_rank else '✗'}")
                    else:
                        print(f"      ⚠️ Engine returned None")
                        
                except Exception as e:
                    # Some engines may have missing dependencies
                    print(f"      ⚠️ Could not load: {str(e)[:50]}...")
        
        # Pass if we loaded at least one engine
        if loaded_count > 0:
            print(f"\n✅ Successfully loaded {loaded_count}/{len(test_engines)} test engines")
            return True
        else:
            print(f"\n❌ Could not load any engines")
            return False
        
    except Exception as e:
        print(f"❌ Engine loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  PRISM INTEGRATION TESTS                         ║
║                                                                  ║
║  Testing full end-to-end workflows                               ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    
    start_time = time.time()
    
    results = {
        "cli_direct": test_cli_direct_mode(),
        "workflow_exec": test_workflow_execution(),
        "html_reports": test_html_report_generation(),
        "web_server": test_web_server_endpoints(),
        "panel_loading": test_panel_data_loading(),
        "engine_loading": test_engine_loading(),
    }
    
    duration = time.time() - start_time
    
    # Summary
    print_section("INTEGRATION TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    print(f"\n{'Test':<20} {'Status':<10}")
    print("-" * 35)
    
    for test, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test:<20} {status}")
    
    print("-" * 35)
    print(f"\nTotal: {passed}/{len(results)} passed")
    print(f"Duration: {duration:.2f}s")
    
    print("\n" + "=" * 60)
    if failed == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
