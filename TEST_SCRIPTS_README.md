# PRISM Test Scripts Reference
## Verification & Testing Tools

---

## Quick Start

```bash
# Quick smoke test (30 seconds)
python test_all_phases.py --quick

# Full phase tests (2-3 minutes)
python test_all_phases.py

# Integration tests (1 minute)
python test_integration.py

# Capability check (instant)
python check_capabilities.py

# Demo (1-2 minutes)
python demo.py
```

---

## Test Scripts Overview

| Script | Purpose | Duration |
|--------|---------|----------|
| `test_all_phases.py` | Master test runner for all phases | ~25s |
| `test_integration.py` | End-to-end workflow tests | ~7s |
| `check_capabilities.py` | Quick capability verification | ~2s |
| `demo.py` | Feature demonstration | ~10s |

---

## 1. Master Test Runner (`test_all_phases.py`)

Runs all phase-specific test scripts and produces a summary.

```bash
# Run all phases
python test_all_phases.py

# Quick smoke test only
python test_all_phases.py --quick

# Run specific phase (1-5)
python test_all_phases.py --phase 3
```

**Output:**
```
╔════════════════════════════════════════════════════════════════════╗
║                      PRISM MASTER TEST RUNNER                      ║
╚════════════════════════════════════════════════════════════════════╝

Phase      Status       Duration     Script
------------------------------------------------------------
Phase 1    ✅ PASSED       4.30s      test_phase1.py
Phase 2    ✅ PASSED       0.97s      test_phase2_panels.py
Phase 3    ✅ PASSED      11.32s      test_phase3_cli.py
Phase 4    ✅ PASSED       3.39s      test_phase4_html.py
Phase 5    ✅ PASSED       2.95s      test_phase5_web.py

🎉 ALL PHASES PASSED!
```

---

## 2. Integration Tests (`test_integration.py`)

Tests full end-to-end workflows including:

- CLI direct mode execution
- Workflow execution via Python
- HTML report generation
- Web server endpoints
- Panel data loading
- Engine loading

```bash
python test_integration.py
```

**Tests:**
- `cli_direct` - CLI subprocess execution
- `workflow_exec` - Python workflow execution
- `html_reports` - Report generation
- `web_server` - Flask endpoint tests
- `panel_loading` - Panel loader tests
- `engine_loading` - Engine loader tests

---

## 3. Capability Checker (`check_capabilities.py`)

Quick verification of what's installed and working.

```bash
python check_capabilities.py
```

**Checks:**
- Dependencies (pyyaml, pandas, numpy, jinja2, flask)
- Core modules (engines, workflows, panels, runner)
- Registries (YAML files)
- Templates (HTML files)
- Directories
- Resource counts
- Entry points
- Quick functionality tests

---

## 4. Demo Script (`demo.py`)

Demonstrates main features of the system.

```bash
# Full demo
python demo.py

# Quick demo (discovery + web only)
python demo.py --quick
```

**Demonstrates:**
- Discovery system (engines, workflows, panels)
- Panel loading
- Workflow execution
- Regime comparison
- HTML report generation
- Web server capabilities

---

## 5. Phase-Specific Tests

### Phase 1: Registry & Loader Infrastructure
```bash
python test_phase1.py
```
Tests: Engine loader, workflow loader, existing code compatibility

### Phase 2: Panel Domain System
```bash
python test_phase2_panels.py
```
Tests: Panel loader, panel creation, indicator loading

### Phase 3: Unified CLI Runner
```bash
python test_phase3_cli.py
```
Tests: CLI runner, executor, output manager, direct mode

### Phase 4: HTML Output System
```bash
python test_phase4_html.py
```
Tests: Jinja2, templates, report generation

### Phase 5: HTML Runner UI (Flask)
```bash
python test_phase5_web.py
```
Tests: Flask app, endpoints, web UI

---

## Expected Results

All tests should pass:

```
test_all_phases.py:     5/5 phases ✅
test_integration.py:    6/6 tests ✅
check_capabilities.py:  All checks ✅
```

---

## Troubleshooting

### Missing Dependencies
```bash
pip install pyyaml pandas numpy jinja2 flask --break-system-packages
```

### Import Errors
Ensure you're in the project root:
```bash
cd /path/to/prism-engine-main
python test_all_phases.py
```

### Individual Test Failures
Run the specific phase test for details:
```bash
python test_all_phases.py --phase 3
```

---

## CI/CD Integration

For automated testing:

```bash
#!/bin/bash
# ci_test.sh

set -e

echo "Running PRISM tests..."

# Quick smoke test first
python test_all_phases.py --quick

# Full phase tests
python test_all_phases.py

# Integration tests
python test_integration.py

echo "All tests passed!"
```

---

## Test Output Files

Test runs may create files in `output/`:
- `test_*.html` - Test reports (cleaned up)
- `integration_test_*` - Integration test directories (cleaned up)

These are automatically cleaned up after successful tests.
