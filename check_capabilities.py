#!/usr/bin/env python3
"""
PRISM Capability Checker
=========================

Quickly checks what capabilities are available and working.

Usage:
    python check_capabilities.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_import(module_path: str, items: list = None) -> dict:
    """Check if a module/items can be imported."""
    result = {"available": False, "items": {}, "error": None}
    
    try:
        module = __import__(module_path, fromlist=items or [])
        result["available"] = True
        
        if items:
            for item in items:
                try:
                    getattr(module, item)
                    result["items"][item] = True
                except AttributeError:
                    result["items"][item] = False
                    
    except ImportError as e:
        result["error"] = str(e)
    
    return result


def check_file_exists(path: str) -> bool:
    """Check if a file exists."""
    return (PROJECT_ROOT / path).exists()


def check_directory_exists(path: str) -> bool:
    """Check if a directory exists."""
    return (PROJECT_ROOT / path).is_dir()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                 PRISM CAPABILITY CHECKER                         ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ===== DEPENDENCIES =====
    print("\n📦 DEPENDENCIES")
    print("-" * 50)
    
    deps = {
        "yaml": "pyyaml",
        "pandas": "pandas",
        "numpy": "numpy",
        "jinja2": "jinja2",
        "flask": "flask",
    }
    
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (pip install {package})")
    
    # ===== CORE MODULES =====
    print("\n🔧 CORE MODULES")
    print("-" * 50)
    
    modules = [
        ("engines", ["EngineLoader"]),
        ("workflows", ["WorkflowLoader"]),
        ("panels", ["PanelLoader", "BasePanel"]),
        ("runner", ["CLIRunner", "WorkflowExecutor", "OutputManager"]),
        ("runner", ["ReportGenerator", "generate_report"]),
        ("runner", ["create_app", "run_server"]),
    ]
    
    for module_path, items in modules:
        result = check_import(module_path, items)
        if result["available"]:
            item_status = ", ".join(
                f"{'✓' if v else '✗'} {k}" 
                for k, v in result["items"].items()
            )
            print(f"  ✅ {module_path}: {item_status}")
        else:
            print(f"  ❌ {module_path}: {result['error']}")
    
    # ===== REGISTRIES =====
    print("\n📋 REGISTRIES")
    print("-" * 50)
    
    registries = [
        "engines/registry.yaml",
        "workflows/registry.yaml",
        "panels/market/registry.yaml",
        "panels/economy/registry.yaml",
        "panels/climate/registry.yaml",
        "panels/custom/registry.yaml",
    ]
    
    for reg in registries:
        if check_file_exists(reg):
            print(f"  ✅ {reg}")
        else:
            print(f"  ❌ {reg}")
    
    # ===== TEMPLATES =====
    print("\n📄 TEMPLATES")
    print("-" * 50)
    
    templates = [
        "templates/base.html",
        "templates/reports/base_report.html",
        "templates/reports/regime_comparison.html",
        "templates/reports/lens_validation.html",
        "templates/reports/generic.html",
        "templates/runner/index.html",
        "templates/runner/web_results.html",
        "templates/runner/error.html",
    ]
    
    for tmpl in templates:
        if check_file_exists(tmpl):
            print(f"  ✅ {tmpl}")
        else:
            print(f"  ❌ {tmpl}")
    
    # ===== DIRECTORIES =====
    print("\n📁 DIRECTORIES")
    print("-" * 50)
    
    directories = [
        "engines",
        "workflows", 
        "panels",
        "runner",
        "templates",
        "templates/reports",
        "templates/runner",
        "output",
    ]
    
    for d in directories:
        if check_directory_exists(d):
            print(f"  ✅ {d}/")
        else:
            print(f"  ❌ {d}/")
    
    # ===== COUNTS =====
    print("\n📊 RESOURCE COUNTS")
    print("-" * 50)
    
    try:
        from engines import EngineLoader
        loader = EngineLoader()
        engines = loader.list_engines()
        print(f"  Engines:   {len(engines)}")
    except:
        print(f"  Engines:   N/A")
    
    try:
        from workflows import WorkflowLoader
        loader = WorkflowLoader()
        workflows = loader.list_workflows()
        print(f"  Workflows: {len(workflows)}")
    except:
        print(f"  Workflows: N/A")
    
    try:
        from panels import PanelLoader
        loader = PanelLoader()
        panels = loader.list_panels()
        print(f"  Panels:    {len(panels)}")
    except:
        print(f"  Panels:    N/A")
    
    try:
        from runner import ReportGenerator
        gen = ReportGenerator()
        templates = gen.list_templates()
        print(f"  Templates: {len(templates)}")
    except:
        print(f"  Templates: N/A")
    
    # ===== ENTRY POINTS =====
    print("\n🚀 ENTRY POINTS")
    print("-" * 50)
    
    entry_points = [
        ("prism_run.py", "Main entry point"),
        ("prism_run.py --help", "Help available"),
        ("prism_run.py --list-panels", "List panels"),
        ("prism_run.py --list-workflows", "List workflows"),
        ("prism_run.py --list-engines", "List engines"),
        ("prism_run.py --web", "Web server"),
    ]
    
    for ep, desc in entry_points:
        if check_file_exists(ep.split()[0]):
            print(f"  ✅ python {ep:<35} # {desc}")
        else:
            print(f"  ❌ python {ep:<35} # {desc}")
    
    # ===== QUICK FUNCTIONALITY TEST =====
    print("\n⚡ QUICK FUNCTIONALITY TEST")
    print("-" * 50)
    
    # Test workflow execution
    try:
        from runner import WorkflowExecutor
        executor = WorkflowExecutor()
        result = executor.execute("market", "lens_validation")
        if result.get("status") == "completed":
            print(f"  ✅ Workflow execution works")
        else:
            print(f"  ⚠️ Workflow returned: {result.get('status')}")
    except Exception as e:
        print(f"  ❌ Workflow execution: {e}")
    
    # Test report generation
    try:
        from runner import ReportGenerator
        gen = ReportGenerator()
        test_results = {"panel": "test", "workflow": "test", "status": "ok"}
        output_path = PROJECT_ROOT / "output" / "_test_report.html"
        report = gen.generate(test_results, output_path)
        if report.exists():
            print(f"  ✅ Report generation works")
            report.unlink()  # Cleanup
        else:
            print(f"  ⚠️ Report file not created")
    except Exception as e:
        print(f"  ❌ Report generation: {e}")
    
    # Test web server
    try:
        from runner import create_app
        app = create_app()
        client = app.test_client()
        response = client.get('/health')
        if response.status_code == 200:
            print(f"  ✅ Web server works")
        else:
            print(f"  ⚠️ Web server returned: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Web server: {e}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 50)
    print("CAPABILITY CHECK COMPLETE")
    print("=" * 50)
    
    print("""
Usage examples:
  python prism_run.py                              # Interactive CLI
  python prism_run.py --panel market --workflow lens_validation
  python prism_run.py --web                        # Start web server
  python test_all_phases.py                        # Run all tests
  python test_integration.py                       # Integration tests
""")


if __name__ == "__main__":
    main()
