#!/usr/bin/env python3
"""
Phase 2 Panel System Verification
==================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_panel_imports():
    """Test panel package imports."""
    print("\n" + "=" * 50)
    print("TEST: Panel Imports")
    print("=" * 50)
    
    try:
        from panels import list_panels, load_panel, get_panel_info, BasePanel
        print("✓ Panel package imports OK")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_discovery():
    """Test panel auto-discovery."""
    print("\n" + "=" * 50)
    print("TEST: Panel Discovery")
    print("=" * 50)
    
    try:
        from panels import list_panels, get_panel_info
        
        panels = list_panels()
        print(f"✓ Discovered {len(panels)} panels: {panels}")
        
        expected = ["market", "economy", "climate", "custom"]
        for p in expected:
            if p in panels:
                print(f"  ✓ {p} panel found")
            else:
                print(f"  ❌ {p} panel missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_loading():
    """Test loading individual panels."""
    print("\n" + "=" * 50)
    print("TEST: Panel Loading")
    print("=" * 50)
    
    try:
        from panels import load_panel
        
        # Test market panel
        market = load_panel("market")
        indicators = market.get_indicators()
        print(f"✓ Market panel: {len(indicators)} indicators")
        assert len(indicators) >= 30, f"Expected >= 30 market indicators, got {len(indicators)}"
        
        # Test economy panel
        economy = load_panel("economy")
        indicators = economy.get_indicators()
        print(f"✓ Economy panel: {len(indicators)} indicators")
        assert len(indicators) >= 20, f"Expected >= 20 economy indicators, got {len(indicators)}"
        
        # Test climate panel
        climate = load_panel("climate")
        indicators = climate.get_indicators()
        print(f"✓ Climate panel: {len(indicators)} indicators")
        
        # Test custom panel
        custom = load_panel("custom")
        indicators = custom.get_indicators()
        print(f"✓ Custom panel: {len(indicators)} indicators")
        
        return True
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_methods():
    """Test panel methods."""
    print("\n" + "=" * 50)
    print("TEST: Panel Methods")
    print("=" * 50)
    
    try:
        from panels import load_panel
        
        market = load_panel("market")
        
        # Test get_indicator_keys
        keys = market.get_indicator_keys()
        assert "spy" in keys, "Missing SPY indicator"
        print(f"✓ get_indicator_keys(): {len(keys)} keys")
        
        # Test get_categories
        categories = market.get_categories()
        assert "benchmark" in categories, "Missing benchmark category"
        print(f"✓ get_categories(): {categories}")
        
        # Test get_indicators_by_category
        benchmarks = market.get_indicators_by_category("benchmark")
        assert len(benchmarks) >= 3, "Should have at least 3 benchmarks"
        print(f"✓ get_indicators_by_category('benchmark'): {len(benchmarks)} indicators")
        
        # Test get_indicator_info
        spy_info = market.get_indicator_info("spy")
        assert spy_info is not None, "SPY info should exist"
        assert spy_info["symbol"] == "SPY", "SPY symbol should be SPY"
        print(f"✓ get_indicator_info('spy'): {spy_info['name']}")
        
        # Test get_metadata
        meta = market.get_metadata()
        assert meta["name"] == "market", "Panel name should be market"
        print(f"✓ get_metadata(): {meta['indicator_count']} indicators, {len(meta['categories'])} categories")
        
        return True
    except Exception as e:
        print(f"❌ Methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_validation():
    """Test panel validation."""
    print("\n" + "=" * 50)
    print("TEST: Panel Validation")
    print("=" * 50)
    
    try:
        from panels import PanelLoader
        
        loader = PanelLoader()
        results = loader.validate_all_panels()
        
        all_valid = True
        for panel, result in results.items():
            status = "✓" if result["valid"] else "❌"
            print(f"  {status} {panel}: valid={result['valid']}")
            if result.get("errors"):
                print(f"      Errors: {result['errors']}")
            if result.get("warnings"):
                print(f"      Warnings: {result['warnings']}")
            if not result["valid"]:
                all_valid = False
        
        return all_valid
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_summary():
    """Test panel summary generation."""
    print("\n" + "=" * 50)
    print("TEST: Panel Summary")
    print("=" * 50)
    
    try:
        from panels import PanelLoader
        
        loader = PanelLoader()
        summary = loader.get_panel_summary()
        print(summary)
        return True
    except Exception as e:
        print(f"❌ Summary failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║             PRISM PHASE 2 VERIFICATION                       ║
    ║                                                              ║
    ║  Testing: Panel Domain System                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        "imports": test_panel_imports(),
        "discovery": test_panel_discovery(),
        "loading": test_panel_loading(),
        "methods": test_panel_methods(),
        "validation": test_panel_validation(),
        "summary": test_panel_summary(),
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
        print("🎉 PHASE 2 (PANELS) COMPLETE - ALL TESTS PASSED!")
    else:
        print("⚠️  PHASE 2 INCOMPLETE - SOME TESTS FAILED")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
