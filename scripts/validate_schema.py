#!/usr/bin/env python3
"""
PRISM Schema Validator - LANGUARD v2
====================================

Validates the actual database schema against LANGUARD v2 specification.
Detects mismatches between:
  - Proposed schema (LANGUARD v2)
  - Actual database tables/columns
  - Repo schema.sql
  - Code references in Python files

Usage:
    python scripts/validate_schema.py
    python scripts/validate_schema.py --fix  # Show recommended fixes
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data.sql.db_path import get_db_path
except ImportError:
    def get_db_path():
        return str(PROJECT_ROOT / "prism_data" / "prism.db")


# =============================================================================
# LANGUARD v2 CANONICAL SCHEMA (from GPT recommendation)
# =============================================================================

LANGUARD_V2_SCHEMA = {
    "indicators": {
        "columns": ["id", "indicator_name", "fred_code", "system", "category",
                    "description", "frequency", "source", "units", "metadata",
                    "created_at", "updated_at"],
        "key_column": "indicator_name",  # LANGUARD uses indicator_name
        "required": ["id", "indicator_name", "system"],
    },
    "indicator_values": {
        "columns": ["indicator_name", "date", "value", "quality_flag",
                    "provenance", "extra", "created_at"],
        "required": ["indicator_name", "date", "value"],
    },
    "timeseries": {
        "type": "TABLE",  # LANGUARD says timeseries should be a TABLE
        "columns": ["id", "indicator_id", "date", "value", "value_2",
                    "adjusted_value", "created_at", "updated_at"],
        "required": ["indicator_id", "date"],
        "note": "Legacy compatibility layer",
    },
    "systems": {
        "columns": ["system", "system_name"],
        "required": ["system"],
    },
}

# =============================================================================
# REPO SCHEMA.SQL (what's in data/sql/schema.sql)
# =============================================================================

REPO_SCHEMA = {
    "indicators": {
        "columns": ["id", "name", "system", "frequency", "source", "units",
                    "description", "metadata", "created_at", "updated_at"],
        "key_column": "name",  # REPO uses 'name', not 'indicator_name'
        "note": "MISSING: fred_code, category. USES 'name' not 'indicator_name'",
    },
    "timeseries": {
        "type": "VIEW",  # In repo schema.sql, timeseries is a VIEW
        "note": "VIEW mapping to indicator_values, not a table",
    },
    "systems": {
        "columns": ["system"],  # No system_name
        "note": "MISSING: system_name column",
    },
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def get_actual_schema(db_path: str) -> Dict[str, Any]:
    """Extract actual schema from SQLite database."""
    if not Path(db_path).exists():
        return {"error": "Database not found", "tables": {}, "views": {}}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    schema = {"tables": {}, "views": {}}

    # Get all tables and views
    objects = conn.execute(
        "SELECT name, type FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()

    for row in objects:
        name, obj_type = row["name"], row["type"]

        # Get column info
        columns = conn.execute(f"PRAGMA table_info([{name}])").fetchall()
        col_names = [c["name"] for c in columns]

        # Get row count
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
        except:
            count = 0

        target = schema["tables"] if obj_type == "table" else schema["views"]
        target[name] = {"columns": col_names, "row_count": count}

    conn.close()
    return schema


def print_section(title: str):
    """Print a section header."""
    print()
    print("─" * 70)
    print(f"  {title}")
    print("─" * 70)


def validate_indicator_column(actual: Dict) -> Tuple[str, List[str]]:
    """
    CRITICAL: Check if indicators table uses 'name' or 'indicator_name'.
    This is the root cause of most schema drift issues.
    """
    findings = []
    column_used = "unknown"

    if "indicators" not in actual["tables"]:
        return "missing", ["❌ indicators table NOT FOUND"]

    cols = actual["tables"]["indicators"]["columns"]

    has_name = "name" in cols
    has_indicator_name = "indicator_name" in cols

    if has_indicator_name and not has_name:
        column_used = "indicator_name"
        findings.append("✅ Uses 'indicator_name' (LANGUARD v2 compliant)")
    elif has_name and not has_indicator_name:
        column_used = "name"
        findings.append("⚠️  Uses 'name' (repo schema.sql style)")
        findings.append("   LANGUARD v2 expects 'indicator_name'")
    elif has_name and has_indicator_name:
        column_used = "both"
        findings.append("⚠️  Has BOTH 'name' AND 'indicator_name'")
    else:
        column_used = "neither"
        findings.append("❌ Has NEITHER 'name' nor 'indicator_name'")

    return column_used, findings


def validate_timeseries_type(actual: Dict) -> List[str]:
    """Check if timeseries is a table or view."""
    findings = []

    is_table = "timeseries" in actual["tables"]
    is_view = "timeseries" in actual["views"]

    if is_table:
        findings.append("📋 'timeseries' is a TABLE")
        cols = actual["tables"]["timeseries"]["columns"]
        count = actual["tables"]["timeseries"]["row_count"]
        findings.append(f"   Columns: {', '.join(cols)}")
        findings.append(f"   Row count: {count:,}")
        if "indicator_id" in cols:
            findings.append("   ✅ Uses indicator_id FK (LANGUARD v2 style)")
    elif is_view:
        findings.append("📋 'timeseries' is a VIEW (repo schema.sql style)")
        findings.append("   Maps to indicator_values")
        findings.append("   ⚠️  LANGUARD v2 expects a TABLE, not VIEW")
    else:
        findings.append("❌ 'timeseries' NOT FOUND (neither table nor view)")

    return findings


def validate_data_location(actual: Dict) -> List[str]:
    """Check where data is actually stored."""
    findings = []

    iv_count = actual["tables"].get("indicator_values", {}).get("row_count", 0)
    ts_count = actual["tables"].get("timeseries", {}).get("row_count", 0)

    findings.append(f"📊 indicator_values: {iv_count:,} rows")

    if "timeseries" in actual["tables"]:
        findings.append(f"📊 timeseries (table): {ts_count:,} rows")
    elif "timeseries" in actual["views"]:
        findings.append(f"📊 timeseries (view): mirrors indicator_values")

    # Detect mismatch
    if ts_count > 0 and iv_count == 0:
        findings.append("")
        findings.append("⚠️  DATA MISMATCH DETECTED:")
        findings.append("   Data is in 'timeseries' table")
        findings.append("   But 'indicator_values' is empty")
        findings.append("   → benchmark_suite.py may need to read from timeseries")

    return findings


def validate_missing_columns(actual: Dict) -> List[str]:
    """Check for missing LANGUARD v2 columns."""
    findings = []

    if "indicators" in actual["tables"]:
        actual_cols = set(actual["tables"]["indicators"]["columns"])
        languard_cols = set(LANGUARD_V2_SCHEMA["indicators"]["columns"])

        missing = languard_cols - actual_cols
        extra = actual_cols - languard_cols

        if missing:
            findings.append(f"⚠️  indicators missing LANGUARD columns: {missing}")
        if extra:
            findings.append(f"   indicators has extra columns: {extra}")

    if "systems" in actual["tables"]:
        actual_cols = set(actual["tables"]["systems"]["columns"])
        if "system_name" not in actual_cols:
            findings.append("⚠️  systems missing 'system_name' column (LANGUARD v2)")

    return findings


def scan_code_references() -> Dict[str, List[str]]:
    """Scan Python files for schema column references."""
    refs = {
        "uses_name": [],
        "uses_indicator_name": [],
        "reads_timeseries": [],
        "reads_indicator_values": [],
        "writes_timeseries": [],
        "writes_indicator_values": [],
    }

    key_files = [
        "workflows/benchmark_suite.py",
        "data/sql/load_benchmarks.py",
        "data/sql/db_connector.py",
        "data/sql/schema.sql",
    ]

    for rel_path in key_files:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text()

            # Check column references
            if "i.name" in content or "indicators.name" in content:
                refs["uses_name"].append(rel_path)
            if "indicator_name" in content:
                refs["uses_indicator_name"].append(rel_path)

            # Check table references
            if "FROM timeseries" in content or "JOIN timeseries" in content:
                refs["reads_timeseries"].append(rel_path)
            if "FROM indicator_values" in content:
                refs["reads_indicator_values"].append(rel_path)
            if "INTO timeseries" in content:
                refs["writes_timeseries"].append(rel_path)
            if "INTO indicator_values" in content:
                refs["writes_indicator_values"].append(rel_path)

        except Exception as e:
            pass

    return refs


def generate_alignment_report(col_used: str, actual: Dict, refs: Dict) -> List[str]:
    """Generate recommendations based on findings."""
    recommendations = []

    ts_is_table = "timeseries" in actual["tables"]
    ts_is_view = "timeseries" in actual["views"]
    iv_count = actual["tables"].get("indicator_values", {}).get("row_count", 0)
    ts_count = actual["tables"].get("timeseries", {}).get("row_count", 0)

    recommendations.append("ALIGNMENT OPTIONS:")
    recommendations.append("")

    if col_used == "name":
        recommendations.append("Option A: Align to REPO schema.sql (simpler)")
        recommendations.append("  - indicators uses 'name' column ✓")
        recommendations.append("  - Update load_benchmarks.py: indicator_name → name")
        recommendations.append("  - Update benchmark_suite.py: i.indicator_name → i.name")
        recommendations.append("")
        recommendations.append("Option B: Align to LANGUARD v2 (more work)")
        recommendations.append("  - Run migration to rename 'name' → 'indicator_name'")
        recommendations.append("  - Update schema.sql")
        recommendations.append("  - Update db_connector.py")

    elif col_used == "indicator_name":
        recommendations.append("Your DB follows LANGUARD v2 (indicator_name)")
        recommendations.append("  - Update schema.sql to match")
        recommendations.append("  - Ensure all code uses 'indicator_name'")

    recommendations.append("")

    if ts_count > 0 and iv_count == 0:
        recommendations.append("DATA TABLE FIX:")
        recommendations.append("  - Benchmark data is in 'timeseries' table")
        recommendations.append("  - benchmark_suite.py should read from 'timeseries'")
        recommendations.append("  - OR: Migrate data to 'indicator_values'")

    return recommendations


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run full schema validation."""
    show_fixes = "--fix" in sys.argv

    print("=" * 70)
    print("   PRISM SCHEMA VALIDATOR - LANGUARD v2")
    print("=" * 70)

    # Get database
    db_path = get_db_path()
    print(f"\nDatabase: {db_path}")

    if not Path(db_path).exists():
        print("\n❌ DATABASE NOT FOUND")
        print("   Run: python -c \"from data.sql.db_connector import init_database; init_database()\"")
        return 1

    actual = get_actual_schema(db_path)
    issues = []

    # 1. Critical column check
    print_section("1. CRITICAL: Indicator Key Column")
    col_used, col_findings = validate_indicator_column(actual)
    for f in col_findings:
        print(f"   {f}")
    if col_used not in ["indicator_name"]:
        issues.append("Indicator column mismatch")

    # 2. Timeseries type
    print_section("2. Timeseries: Table vs View")
    ts_findings = validate_timeseries_type(actual)
    for f in ts_findings:
        print(f"   {f}")

    # 3. Data location
    print_section("3. Data Location Check")
    data_findings = validate_data_location(actual)
    for f in data_findings:
        print(f"   {f}")

    # 4. Missing columns
    print_section("4. Missing LANGUARD v2 Columns")
    col_findings = validate_missing_columns(actual)
    if col_findings:
        for f in col_findings:
            print(f"   {f}")
    else:
        print("   ✅ No critical columns missing")

    # 5. Code references
    print_section("5. Code Schema References")
    refs = scan_code_references()
    print(f"   Files using 'name':            {refs['uses_name']}")
    print(f"   Files using 'indicator_name':  {refs['uses_indicator_name']}")
    print(f"   Files reading timeseries:      {refs['reads_timeseries']}")
    print(f"   Files reading indicator_values:{refs['reads_indicator_values']}")
    print(f"   Files writing timeseries:      {refs['writes_timeseries']}")
    print(f"   Files writing indicator_values:{refs['writes_indicator_values']}")

    # 6. Summary
    print_section("6. SCHEMA COMPARISON")
    print("""
   ┌─────────────────────┬──────────────────┬──────────────────┐
   │ Feature             │ LANGUARD v2      │ Repo schema.sql  │
   ├─────────────────────┼──────────────────┼──────────────────┤
   │ Indicator column    │ indicator_name   │ name             │
   │ timeseries          │ TABLE            │ VIEW             │
   │ systems.system_name │ Present          │ Missing          │
   │ indicators.fred_code│ Present          │ Missing          │
   │ indicators.category │ Present          │ Missing          │
   └─────────────────────┴──────────────────┴──────────────────┘
""")

    # Show fixes if requested
    if show_fixes:
        print_section("RECOMMENDED ACTIONS")
        recs = generate_alignment_report(col_used, actual, refs)
        for r in recs:
            print(f"   {r}")

    # Final summary
    print()
    print("=" * 70)
    if issues:
        print(f"   ⚠️  Found {len(issues)} alignment issue(s)")
        print("   Run with --fix for recommendations")
    else:
        print("   ✅ Schema validation complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
