#!/usr/bin/env python3
"""
Phase 2 Verification Script
============================

Run this after implementing Phase 2 to verify the HTML Runner works.

Usage:
    python test_phase2.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_flask_available():
    """Test Flask is installed."""
    print("\n" + "=" * 50)
    print("TEST: Flask Installation")
    print("=" * 50)

    try:
        from flask import Flask
        print("✓ Flask is installed")
        return True
    except ImportError:
        print("❌ Flask NOT installed")
        print("   Install with: pip install flask")
        return False


def test_runner_imports():
    """Test runner module imports."""
    print("\n" + "=" * 50)
    print("TEST: Runner Imports")
    print("=" * 50)

    try:
        from runner.prism_run import create_app, get_available_panels, get_available_workflows
        print("✓ prism_run imports OK")

        from runner.dispatcher import Dispatcher
        print("✓ dispatcher imports OK")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_flask_app_creation():
    """Test Flask app can be created."""
    print("\n" + "=" * 50)
    print("TEST: Flask App Creation")
    print("=" * 50)

    try:
        from runner.prism_run import create_app

        app = create_app()
        print(f"✓ Flask app created: {app.name}")

        # Check routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/run', '/reports', '/health']

        for route in expected_routes:
            if route in routes:
                print(f"✓ Route registered: {route}")
            else:
                print(f"❌ Route missing: {route}")
                return False

        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False


def test_data_loaders():
    """Test that data loaders return expected data."""
    print("\n" + "=" * 50)
    print("TEST: Data Loaders")
    print("=" * 50)

    try:
        from runner.prism_run import get_available_panels, get_available_workflows, get_weighting_methods

        # Test panels
        panels = get_available_panels()
        assert len(panels) >= 4, f"Expected >= 4 panels, got {len(panels)}"
        panel_keys = [p['key'] for p in panels]
        assert 'market' in panel_keys, "Missing 'market' panel"
        print(f"✓ Panels loaded: {panel_keys}")

        # Test workflows
        workflows = get_available_workflows()
        assert len(workflows) >= 3, f"Expected >= 3 workflows, got {len(workflows)}"
        workflow_keys = [w['key'] for w in workflows]
        assert 'regime_comparison' in workflow_keys, "Missing 'regime_comparison' workflow"
        print(f"✓ Workflows loaded: {len(workflows)} workflows")

        # Test weighting methods
        methods = get_weighting_methods()
        assert len(methods) >= 3, f"Expected >= 3 methods, got {len(methods)}"
        method_keys = [m['key'] for m in methods]
        assert 'hybrid' in method_keys, "Missing 'hybrid' method"
        print(f"✓ Weighting methods loaded: {method_keys}")

        return True
    except Exception as e:
        print(f"❌ Data loader failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dispatcher():
    """Test dispatcher can run analysis."""
    print("\n" + "=" * 50)
    print("TEST: Dispatcher")
    print("=" * 50)

    try:
        from runner.dispatcher import Dispatcher

        dispatcher = Dispatcher()
        print(f"✓ Dispatcher created")
        print(f"  Output dir: {dispatcher.output_dir}")

        # Run a simple workflow
        params = {
            'panel': 'market',
            'workflow': 'daily_update',
            'weighting': 'hybrid',
        }

        results = dispatcher.run(params)

        assert results.get('status') == 'success', f"Run failed: {results.get('error')}"
        print(f"✓ Analysis completed successfully")
        print(f"  Duration: {results.get('duration', 0):.2f}s")

        # Check report was created
        report_path = results.get('report_path')
        if report_path:
            report_file = Path(report_path)
            if report_file.exists():
                print(f"✓ Report created: {report_file.name}")
            else:
                print(f"⚠ Report path returned but file doesn't exist")

        return True
    except Exception as e:
        print(f"❌ Dispatcher failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_templates_exist():
    """Test that HTML templates exist."""
    print("\n" + "=" * 50)
    print("TEST: HTML Templates")
    print("=" * 50)

    template_dir = PROJECT_ROOT / "templates"

    required_templates = [
        "base.html",
        "runner/index.html",
        "runner/results.html",
        "runner/reports.html",
    ]

    all_exist = True
    for template in required_templates:
        template_path = template_dir / template
        if template_path.exists():
            size = template_path.stat().st_size
            print(f"✓ {template} ({size} bytes)")
        else:
            print(f"❌ {template} NOT FOUND")
            all_exist = False

    return all_exist


def test_flask_test_client():
    """Test Flask app with test client."""
    print("\n" + "=" * 50)
    print("TEST: Flask Test Client")
    print("=" * 50)

    try:
        from runner.prism_run import create_app

        app = create_app()
        app.config['TESTING'] = True

        with app.test_client() as client:
            # Test index page
            response = client.get('/')
            assert response.status_code == 200, f"Index returned {response.status_code}"
            assert b'PRISM' in response.data, "PRISM not in response"
            print(f"✓ GET / returns 200 OK")

            # Test health endpoint
            response = client.get('/health')
            assert response.status_code == 200, f"Health returned {response.status_code}"
            print(f"✓ GET /health returns 200 OK")

            # Test reports page
            response = client.get('/reports')
            assert response.status_code == 200, f"Reports returned {response.status_code}"
            print(f"✓ GET /reports returns 200 OK")

            # Test API endpoint
            response = client.get('/api/workflows/market')
            assert response.status_code == 200, f"API returned {response.status_code}"
            print(f"✓ GET /api/workflows/market returns 200 OK")

        return True
    except Exception as e:
        print(f"❌ Test client failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_run_endpoint():
    """Test the /run endpoint."""
    print("\n" + "=" * 50)
    print("TEST: Run Endpoint")
    print("=" * 50)

    try:
        from runner.prism_run import create_app

        app = create_app()
        app.config['TESTING'] = True

        with app.test_client() as client:
            # Submit analysis request
            response = client.post('/run', data={
                'panel': 'market',
                'workflow': 'daily_update',
                'weighting': 'hybrid',
                'timeframe': 'default',
            })

            assert response.status_code == 200, f"Run returned {response.status_code}"
            assert b'Analysis' in response.data or b'complete' in response.data.lower(), "Missing completion message"
            print(f"✓ POST /run returns 200 OK")
            print(f"✓ Response contains results page")

        return True
    except Exception as e:
        print(f"❌ Run endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 tests."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║             PRISM PHASE 2 VERIFICATION                       ║
    ║                                                              ║
    ║  Testing: HTML Runner (Flask Web Interface)                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    results = {
        "flask_available": test_flask_available(),
        "runner_imports": test_runner_imports(),
        "flask_app_creation": test_flask_app_creation(),
        "data_loaders": test_data_loaders(),
        "dispatcher": test_dispatcher(),
        "templates_exist": test_templates_exist(),
        "flask_test_client": test_flask_test_client(),
        "run_endpoint": test_run_endpoint(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 PHASE 2 VERIFICATION COMPLETE - ALL TESTS PASSED!")
        print("")
        print("To launch the HTML runner:")
        print("  python prism_run.py")
        print("")
        print("Or from runner module:")
        print("  python -m runner.prism_run")
    else:
        print("⚠️  PHASE 2 VERIFICATION INCOMPLETE - SOME TESTS FAILED")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
