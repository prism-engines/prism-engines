#!/usr/bin/env python3
"""
Phase 1 Verification Script
============================

Run this after implementing Phase 1 to verify everything works.

Usage:
    python test_phase1.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_engine_loader():
    """Test engine loader functionality."""
    print("\n" + "=" * 50)
    print("TEST: Engine Loader")
    print("=" * 50)

    try:
        from engines import EngineLoader, list_engines, list_lenses

        # Test 1: List engines
        engines = list_engines()
        print(f"✓ list_engines() returned {len(engines)} engines")
        assert len(engines) >= 14, f"Expected >= 14 engines, got {len(engines)}"

        # Test 2: List lenses
        lenses = list_lenses()
        print(f"✓ list_lenses() returned {len(lenses)} lenses")
        assert len(lenses) >= 14, f"Expected >= 14 lenses, got {len(lenses)}"

        # Test 3: Engine loader instance
        loader = EngineLoader()

        # Test 4: Get engine info
        pca_info = loader.get_engine_info("pca")
        assert pca_info is not None, "PCA engine not found"
        assert pca_info.get("lens") == True, "PCA should be marked as lens"
        print(f"✓ get_engine_info('pca') works")

        # Test 5: Get engines by category
        geometry_engines = loader.get_engines_by_category("geometry")
        assert len(geometry_engines) > 0, "No geometry engines found"
        print(f"✓ get_engines_by_category('geometry') returned {len(geometry_engines)} engines")

        # Test 6: Get lens set
        full_set = loader.get_lens_set("full")
        quick_set = loader.get_lens_set("quick")
        assert len(full_set) > len(quick_set), "Full set should be larger than quick set"
        print(f"✓ get_lens_set() works (full={len(full_set)}, quick={len(quick_set)})")

        # Test 7: Get compatible engines
        compatible = loader.get_compatible_engines("market", "daily")
        assert len(compatible) > 0, "No compatible engines for market/daily"
        print(f"✓ get_compatible_engines('market', 'daily') returned {len(compatible)} engines")

        # Test 8: Validate engine
        validation = loader.validate_engine("pca")
        print(f"✓ validate_engine('pca'): valid={validation.get('valid')}")

        # Test 9: Summary
        summary = loader.get_engine_summary()
        assert len(summary) > 100, "Summary too short"
        print(f"✓ get_engine_summary() returned {len(summary)} chars")

        print("\n✅ ENGINE LOADER: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ ENGINE LOADER FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_loader():
    """Test workflow loader functionality."""
    print("\n" + "=" * 50)
    print("TEST: Workflow Loader")
    print("=" * 50)

    try:
        from workflows import WorkflowLoader, list_workflows

        # Test 1: List workflows
        workflows = list_workflows()
        print(f"✓ list_workflows() returned {len(workflows)} workflows")
        assert len(workflows) >= 3, f"Expected >= 3 workflows, got {len(workflows)}"

        # Test 2: Workflow loader instance
        loader = WorkflowLoader()

        # Test 3: Get workflow info
        regime_info = loader.get_workflow_info("regime_comparison")
        assert regime_info is not None, "regime_comparison not found"
        print(f"✓ get_workflow_info('regime_comparison') works")

        # Test 4: Get workflows by category
        regime_workflows = loader.get_workflows_by_category("regime")
        assert len(regime_workflows) > 0, "No regime workflows found"
        print(f"✓ get_workflows_by_category('regime') returned {len(regime_workflows)} workflows")

        # Test 5: Get compatible workflows
        compatible = loader.get_compatible_workflows("market")
        assert len(compatible) > 0, "No compatible workflows for market"
        print(f"✓ get_compatible_workflows('market') returned {len(compatible)} workflows")

        # Test 6: Get presets
        presets = loader.get_presets()
        assert len(presets) > 0, "No presets found"
        print(f"✓ get_presets() returned {len(presets)} presets")

        # Test 7: Get default parameters
        defaults = loader.get_default_parameters("regime_comparison")
        assert "comparison_periods" in defaults, "Missing comparison_periods default"
        print(f"✓ get_default_parameters() works")

        # Test 8: Summary
        summary = loader.get_workflow_summary()
        assert len(summary) > 50, "Summary too short"
        print(f"✓ get_workflow_summary() returned {len(summary)} chars")

        print("\n✅ WORKFLOW LOADER: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ WORKFLOW LOADER FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_existing_code():
    """Test that existing code still works."""
    print("\n" + "=" * 50)
    print("TEST: Existing Code Compatibility")
    print("=" * 50)

    errors = []

    # Test imports
    test_imports = [
        ("engine_core.lenses.pca_lens", "PCALens"),
        ("engine_core.lenses.correlation_lens", "CorrelationLens"),
        ("engine_core.lenses.clustering_lens", "ClusteringLens"),
        ("engine_core.orchestration.consensus", "ConsensusEngine"),
    ]

    for module_path, class_name in test_imports:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_path}.{class_name} imports OK")
        except ImportError as e:
            print(f"⚠ {module_path}.{class_name} import failed: {e}")
            errors.append((module_path, str(e)))
        except AttributeError as e:
            print(f"⚠ {module_path}.{class_name} not found: {e}")
            errors.append((module_path, str(e)))

    if errors:
        print(f"\n⚠️ EXISTING CODE: {len(errors)} imports failed (may be OK if modules don't exist yet)")
    else:
        print("\n✅ EXISTING CODE: ALL IMPORTS PASSED")

    return len(errors) == 0


def main():
    """Run all Phase 1 tests."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║             PRISM PHASE 1 VERIFICATION                       ║
    ║                                                              ║
    ║  Testing: Registry + Loader Infrastructure                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    results = {
        "engine_loader": test_engine_loader(),
        "workflow_loader": test_workflow_loader(),
        "existing_code": test_existing_code(),
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
        print("🎉 PHASE 1 VERIFICATION COMPLETE - ALL TESTS PASSED!")
    else:
        print("⚠️  PHASE 1 VERIFICATION INCOMPLETE - SOME TESTS FAILED")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
