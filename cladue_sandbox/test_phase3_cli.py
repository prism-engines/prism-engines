#!/usr/bin/env python3
"""
Phase 3 CLI Runner Verification
================================
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_runner_imports():
    """Test runner package imports."""
    print("\n" + "=" * 50)
    print("TEST: Runner Imports")
    print("=" * 50)
    
    try:
        from runner import CLIRunner, WorkflowExecutor, OutputManager
        print("✓ CLIRunner imported")
        print("✓ WorkflowExecutor imported")
        print("✓ OutputManager imported")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_executor_initialization():
    """Test executor can access all loaders."""
    print("\n" + "=" * 50)
    print("TEST: Executor Initialization")
    print("=" * 50)
    
    try:
        from runner import WorkflowExecutor
        
        executor = WorkflowExecutor()
        
        # Test panel loader
        panels = executor.list_panels()
        print(f"✓ Panel loader: {len(panels)} panels")
        
        # Test workflow loader
        workflows = executor.list_workflows()
        print(f"✓ Workflow loader: {len(workflows)} workflows")
        
        # Test engine loader
        engines = executor.list_engines()
        print(f"✓ Engine loader: {len(engines)} engines")
        
        return True
    except Exception as e:
        print(f"❌ Executor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_manager():
    """Test output manager functionality."""
    print("\n" + "=" * 50)
    print("TEST: Output Manager")
    print("=" * 50)
    
    try:
        from runner import OutputManager
        
        output = OutputManager()
        
        # Test run directory creation
        run_dir = output.create_run_directory(prefix="test")
        print(f"✓ Created run directory: {run_dir}")
        
        # Test JSON save
        test_data = {"test": "data", "number": 42}
        json_path = output.save_json(test_data, "test.json")
        print(f"✓ Saved JSON: {json_path}")
        
        # Test text save
        text_path = output.save_text("Test content", "test.txt")
        print(f"✓ Saved text: {text_path}")
        
        # Cleanup
        import shutil
        shutil.rmtree(run_dir)
        print("✓ Cleanup complete")
        
        return True
    except Exception as e:
        print(f"❌ Output manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_execution():
    """Test workflow execution."""
    print("\n" + "=" * 50)
    print("TEST: Workflow Execution")
    print("=" * 50)
    
    try:
        from runner import WorkflowExecutor
        
        executor = WorkflowExecutor()
        
        # Execute a test workflow
        print("Executing: market / lens_validation")
        results = executor.execute(
            panel_key="market",
            workflow_key="lens_validation",
            parameters={}
        )
        
        status = results.get("status")
        print(f"  Status: {status}")
        print(f"  Duration: {results.get('duration', 0):.2f}s")
        
        if status == "completed":
            print("✓ Workflow execution completed")
            return True
        else:
            print(f"❌ Workflow failed: {results.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_list_commands():
    """Test CLI list commands."""
    print("\n" + "=" * 50)
    print("TEST: CLI List Commands")
    print("=" * 50)
    
    try:
        # Test --list-panels
        result = subprocess.run(
            [sys.executable, "prism_run.py", "--list-panels"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0 and "market" in result.stdout.lower():
            print("✓ --list-panels works")
        else:
            print(f"❌ --list-panels failed: {result.stderr}")
            return False
        
        # Test --list-workflows
        result = subprocess.run(
            [sys.executable, "prism_run.py", "--list-workflows"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0 and "regime" in result.stdout.lower():
            print("✓ --list-workflows works")
        else:
            print(f"❌ --list-workflows failed: {result.stderr}")
            return False
        
        # Test --list-engines
        result = subprocess.run(
            [sys.executable, "prism_run.py", "--list-engines"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0 and "pca" in result.stdout.lower():
            print("✓ --list-engines works")
        else:
            print(f"❌ --list-engines failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CLI list commands failed: {e}")
        return False


def test_cli_direct_mode():
    """Test CLI direct mode execution."""
    print("\n" + "=" * 50)
    print("TEST: CLI Direct Mode")
    print("=" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, "prism_run.py", 
             "--panel", "market", 
             "--workflow", "lens_validation"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✓ Direct mode execution completed")
            if "COMPLETED" in result.stdout or "completed" in result.stdout.lower():
                print("✓ Execution status: completed")
            return True
        else:
            print(f"❌ Direct mode failed with code {result.returncode}")
            print(f"   stderr: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CLI direct mode timed out")
        return False
    except Exception as e:
        print(f"❌ CLI direct mode failed: {e}")
        return False


def test_help_command():
    """Test help command."""
    print("\n" + "=" * 50)
    print("TEST: Help Command")
    print("=" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, "prism_run.py", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode == 0 and "panel" in result.stdout.lower():
            print("✓ Help command works")
            return True
        else:
            print(f"❌ Help command failed")
            return False
            
    except Exception as e:
        print(f"❌ Help command failed: {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║             PRISM PHASE 3 VERIFICATION                       ║
    ║                                                              ║
    ║  Testing: Unified CLI Runner                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        "imports": test_runner_imports(),
        "executor": test_executor_initialization(),
        "output_manager": test_output_manager(),
        "workflow_execution": test_workflow_execution(),
        "cli_list": test_cli_list_commands(),
        "cli_direct": test_cli_direct_mode(),
        "help": test_help_command(),
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
        print("🎉 PHASE 3 (CLI RUNNER) COMPLETE - ALL TESTS PASSED!")
    else:
        print("⚠️  PHASE 3 INCOMPLETE - SOME TESTS FAILED")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
