#!/usr/bin/env python3
"""
PRISM Master Test Runner
=========================

Runs all phase verification tests and produces a summary report.

Usage:
    python test_all_phases.py           # Run all tests
    python test_all_phases.py --quick   # Quick smoke test only
    python test_all_phases.py --phase 3 # Run specific phase
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_test_script(script_name: str) -> dict:
    """Run a test script and capture results."""
    script_path = PROJECT_ROOT / script_name
    
    if not script_path.exists():
        return {
            "script": script_name,
            "status": "missing",
            "returncode": -1,
            "duration": 0,
            "output": f"Script not found: {script_path}"
        }
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT)
        )
        duration = time.time() - start
        
        return {
            "script": script_name,
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration": duration,
            "output": result.stdout + result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "script": script_name,
            "status": "timeout",
            "returncode": -1,
            "duration": 120,
            "output": "Test timed out after 120 seconds"
        }
    except Exception as e:
        return {
            "script": script_name,
            "status": "error",
            "returncode": -1,
            "duration": time.time() - start,
            "output": str(e)
        }


def print_banner(title: str):
    """Print a formatted banner."""
    width = 70
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║" + title.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")


def print_result(result: dict):
    """Print a single test result."""
    status_icons = {
        "passed": "✅",
        "failed": "❌",
        "missing": "⚠️",
        "timeout": "⏱️",
        "error": "💥"
    }
    
    icon = status_icons.get(result["status"], "❓")
    print(f"\n{icon} {result['script']}")
    print(f"   Status: {result['status'].upper()}")
    print(f"   Duration: {result['duration']:.2f}s")
    
    if result["status"] != "passed":
        # Show last few lines of output for failures
        lines = result["output"].strip().split("\n")
        if lines:
            print("   Output (last 5 lines):")
            for line in lines[-5:]:
                print(f"      {line}")


def run_all_phases():
    """Run all phase tests."""
    print_banner("PRISM MASTER TEST RUNNER")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    
    # Define test scripts for each phase
    phase_tests = {
        1: "test_phase1.py",
        2: "test_phase2_panels.py",
        3: "test_phase3_cli.py",
        4: "test_phase4_html.py",
        5: "test_phase5_web.py",
    }
    
    results = {}
    total_start = time.time()
    
    for phase, script in phase_tests.items():
        print_banner(f"PHASE {phase} TESTS")
        result = run_test_script(script)
        results[phase] = result
        print_result(result)
    
    total_duration = time.time() - total_start
    
    # Summary
    print_banner("TEST SUMMARY")
    
    passed = sum(1 for r in results.values() if r["status"] == "passed")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    other = len(results) - passed - failed
    
    print(f"\n{'Phase':<10} {'Status':<12} {'Duration':<12} {'Script'}")
    print("-" * 60)
    
    for phase, result in results.items():
        status_str = result["status"].upper()
        if result["status"] == "passed":
            status_str = f"✅ {status_str}"
        elif result["status"] == "failed":
            status_str = f"❌ {status_str}"
        else:
            status_str = f"⚠️ {status_str}"
        
        print(f"Phase {phase:<4} {status_str:<12} {result['duration']:>6.2f}s      {result['script']}")
    
    print("-" * 60)
    print(f"\nTotal: {passed} passed, {failed} failed, {other} other")
    print(f"Duration: {total_duration:.2f}s")
    
    # Final verdict
    print("\n" + "=" * 60)
    if passed == len(results):
        print("🎉 ALL PHASES PASSED!")
        return 0
    else:
        print(f"⚠️  {failed} PHASE(S) FAILED")
        return 1


def run_quick_smoke_test():
    """Run a quick smoke test of core functionality."""
    print_banner("QUICK SMOKE TEST")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Basic imports
    print("\n[1/6] Testing basic imports...")
    try:
        from engines import EngineLoader
        from workflows import WorkflowLoader
        from panels import PanelLoader
        print("      ✅ Core loaders import OK")
        tests_passed += 1
    except ImportError as e:
        print(f"      ❌ Import failed: {e}")
        tests_failed += 1
    
    # Test 2: Registry loading
    print("\n[2/6] Testing registry loading...")
    try:
        from engines import EngineLoader
        loader = EngineLoader()
        engines = loader.list_engines()
        assert len(engines) > 0, "No engines found"
        print(f"      ✅ Found {len(engines)} engines")
        tests_passed += 1
    except Exception as e:
        print(f"      ❌ Registry failed: {e}")
        tests_failed += 1
    
    # Test 3: Panel loading
    print("\n[3/6] Testing panel loading...")
    try:
        from panels import PanelLoader
        loader = PanelLoader()
        panels = loader.list_panels()
        assert len(panels) > 0, "No panels found"
        print(f"      ✅ Found {len(panels)} panels")
        tests_passed += 1
    except Exception as e:
        print(f"      ❌ Panel loading failed: {e}")
        tests_failed += 1
    
    # Test 4: Workflow executor
    print("\n[4/6] Testing workflow executor...")
    try:
        from runner import WorkflowExecutor
        executor = WorkflowExecutor()
        workflows = executor.list_workflows()
        assert len(workflows) > 0, "No workflows found"
        print(f"      ✅ Executor has {len(workflows)} workflows")
        tests_passed += 1
    except Exception as e:
        print(f"      ❌ Executor failed: {e}")
        tests_failed += 1
    
    # Test 5: Report generator
    print("\n[5/6] Testing report generator...")
    try:
        from runner import ReportGenerator
        generator = ReportGenerator()
        templates = generator.list_templates()
        assert len(templates) > 0, "No templates found"
        print(f"      ✅ Found {len(templates)} report templates")
        tests_passed += 1
    except Exception as e:
        print(f"      ❌ Report generator failed: {e}")
        tests_failed += 1
    
    # Test 6: Web server
    print("\n[6/6] Testing web server...")
    try:
        from runner import create_app
        app = create_app()
        client = app.test_client()
        response = client.get('/health')
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        print("      ✅ Web server health check OK")
        tests_passed += 1
    except Exception as e:
        print(f"      ❌ Web server failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"SMOKE TEST: {tests_passed}/{tests_passed + tests_failed} passed")
    
    if tests_failed == 0:
        print("✅ All smoke tests passed!")
        return 0
    else:
        print(f"❌ {tests_failed} test(s) failed")
        return 1


def run_single_phase(phase: int):
    """Run tests for a single phase."""
    phase_tests = {
        1: "test_phase1.py",
        2: "test_phase2_panels.py",
        3: "test_phase3_cli.py",
        4: "test_phase4_html.py",
        5: "test_phase5_web.py",
    }
    
    if phase not in phase_tests:
        print(f"❌ Invalid phase: {phase}")
        print(f"   Valid phases: {list(phase_tests.keys())}")
        return 1
    
    print_banner(f"PHASE {phase} TESTS")
    result = run_test_script(phase_tests[phase])
    print_result(result)
    
    # Show full output
    print("\n" + "=" * 50)
    print("FULL OUTPUT:")
    print("=" * 50)
    print(result["output"])
    
    return 0 if result["status"] == "passed" else 1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PRISM Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--phase", type=int, help="Run specific phase (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all phase tests (default)")
    
    args = parser.parse_args()
    
    if args.quick:
        return run_quick_smoke_test()
    elif args.phase:
        return run_single_phase(args.phase)
    else:
        return run_all_phases()


if __name__ == "__main__":
    sys.exit(main())
