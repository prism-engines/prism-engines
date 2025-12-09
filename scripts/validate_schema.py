#!/usr/bin/env python3
"""
PRISM Schema Validator - Schema v2.1
====================================

Validates the actual database schema against PRISM Schema v2.1 specification.
Detects mismatches between:
  - Schema v2.1 specification (canonical)
  - Actual database tables/columns
  - Code references in Python files

Schema v2.1 Key Requirements:
  - indicators table uses 'name' column (not indicator_name)
  - timeseries is a VIEW (maps to indicator_values)
  - indicators has fred_code, category columns
  - systems has system_name column

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
# PRISM SCHEMA v2.1 CANONICAL (Claude AI approved)
# =============================================================================

SCHEMA_V21 = {
    "indicators": {
        "columns": ["id", "name", "fred_code", "system", "category",
                    "frequency", "source", "units", "description", "metadata",
                    "created_at", "updated_at"],
        "key_column": "name",  # v2.1 uses 'name', NOT 'indicator_name'
        "required": ["id", "name", "system"],
    },
    "indicator_values": {
        "columns": ["indicator_name", "date", "value", "quality_flag",
                    "provenance", "extra", "created_at"],
        "required": ["indicator_name", "date", "value"],
    },
    "timeseries": {
        "type": "VIEW",  # v2.1: timeseries is a VIEW, not a table
        "note": "VIEW mapping to indicator_values for backward compat",
    },
    "systems": {
        "columns": ["system", "system_name"],
        "required": ["system"],
    },
    "calibration_lenses": {
        "required": ["lens_name"],
    },
    "calibration_indicators": {
        "required": ["indicator"],
    },
    "calibration_config": {
        "required": ["key"],
    },
    "analysis_rankings": {
        "required": ["run_date", "indicator"],
    },
    "analysis_signals": {
        "required": ["run_date", "indicator"],
    },
    "consensus_events": {
        "required": ["event_date"],
    },
    "consensus_history": {
        "required": ["date"],
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
    CRITICAL: Check if indicators table uses 'name' (v2.1 canonical).
    """
    findings = []
    column_used = "unknown"

    if "indicators" not in actual["tables"]:
        return "missing", ["❌ indicators table NOT FOUND"]

    cols = actual["tables"]["indicators"]["columns"]

    has_name = "name" in cols
    has_indicator_name = "indicator_name" in cols

    if has_name and not has_indicator_name:
        column_used = "name"
        findings.append("✅ Uses 'name' (Schema v2.1 compliant)")
    elif has_indicator_name and not has_name:
        column_used = "indicator_name"
        findings.append("⚠️  Uses 'indicator_name' (not v2.1 compliant)")
        findings.append("   Schema v2.1 expects 'name'")
    elif has_name and has_indicator_name:
        column_used = "both"
        findings.append("⚠️  Has BOTH 'name' AND 'indicator_name'")
    else:
        column_used = "neither"
        findings.append("❌ Has NEITHER 'name' nor 'indicator_name'")

    return column_used, findings


def validate_timeseries_type(actual: Dict) -> List[str]:
    """Check if timeseries is a view (v2.1 canonical)."""
    findings = []

    is_table = "timeseries" in actual["tables"]
    is_view = "timeseries" in actual["views"]

    if is_view:
        findings.append("✅ 'timeseries' is a VIEW (Schema v2.1 compliant)")
        findings.append("   Maps to indicator_values")
    elif is_table:
        findings.append("⚠️  'timeseries' is a TABLE (not v2.1 compliant)")
        cols = actual["tables"]["timeseries"]["columns"]
        count = actual["tables"]["timeseries"]["row_count"]
        findings.append(f"   Columns: {', '.join(cols)}")
        findings.append(f"   Row count: {count:,}")
        findings.append("   Schema v2.1 expects a VIEW, not TABLE")
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
    """Check for missing Schema v2.1 columns."""
    findings = []

    if "indicators" in actual["tables"]:
        actual_cols = set(actual["tables"]["indicators"]["columns"])
        v21_cols = set(SCHEMA_V21["indicators"]["columns"])

        missing = v21_cols - actual_cols
        extra = actual_cols - v21_cols

        if missing:
            findings.append(f"⚠️  indicators missing v2.1 columns: {missing}")
        if extra:
            findings.append(f"   indicators has extra columns: {extra}")

    if "systems" in actual["tables"]:
        actual_cols = set(actual["tables"]["systems"]["columns"])
        if "system_name" not in actual_cols:
            findings.append("⚠️  systems missing 'system_name' column (v2.1)")

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

    recommendations.append("SCHEMA v2.1 ALIGNMENT:")
    recommendations.append("")

    if col_used == "name":
        recommendations.append("✅ Indicator column is v2.1 compliant ('name')")
        recommendations.append("")
    elif col_used == "indicator_name":
        recommendations.append("⚠️  Your DB uses 'indicator_name' instead of 'name'")
        recommendations.append("  - Run: python scripts/migrate_to_v2.1.py")
        recommendations.append("  - Or rename column: indicator_name → name")
        recommendations.append("")
    else:
        recommendations.append(f"❌ Indicator column issue: {col_used}")
        recommendations.append("")

    if ts_is_view:
        recommendations.append("✅ timeseries is a VIEW (v2.1 compliant)")
    elif ts_is_table:
        recommendations.append("⚠️  timeseries is a TABLE (v2.1 expects VIEW)")
        recommendations.append("  - Create VIEW to map timeseries → indicator_values")
        recommendations.append("  - Or migrate data to indicator_values table")
    recommendations.append("")

    if ts_count > 0 and iv_count == 0:
        recommendations.append("DATA LOCATION:")
        recommendations.append("  - Data is in 'timeseries' table")
        recommendations.append("  - But 'indicator_values' is empty")
        recommendations.append("  - Migrate data or update queries to use timeseries")
        recommendations.append("")

    # Check for missing v2.1 columns
    ind_cols = actual["tables"].get("indicators", {}).get("columns", [])
    missing = []
    if "fred_code" not in ind_cols:
        missing.append("indicators.fred_code")
    if "category" not in ind_cols:
        missing.append("indicators.category")

    sys_cols = actual["tables"].get("systems", {}).get("columns", [])
    if "system_name" not in sys_cols:
        missing.append("systems.system_name")

    if missing:
        recommendations.append("MISSING v2.1 COLUMNS:")
        recommendations.append(f"  - {', '.join(missing)}")
        recommendations.append("  - Run: python scripts/migrate_to_v2.1.py")

    return recommendations


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run full schema validation."""
    show_fixes = "--fix" in sys.argv

    print("=" * 70)
    print("   PRISM SCHEMA VALIDATOR - v2.1")
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
    if col_used not in ["name"]:
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
    print_section("4. Missing Schema v2.1 Columns")
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
    print_section("6. SCHEMA v2.1 COMPLIANCE")
    print("""
   ┌─────────────────────┬──────────────────┬──────────────────┐
   │ Feature             │ Schema v2.1      │ Your Database    │
   ├─────────────────────┼──────────────────┼──────────────────┤
   │ Indicator column    │ name             │ """ + ("name ✓" if col_used == "name" else col_used + " ✗").ljust(16) + """ │
   │ timeseries type     │ VIEW             │ """ + ("VIEW ✓" if "timeseries" in actual["views"] else "TABLE ✗" if "timeseries" in actual["tables"] else "MISSING").ljust(16) + """ │
   │ systems.system_name │ Required         │ """ + ("Present ✓" if "system_name" in actual["tables"].get("systems", {}).get("columns", []) else "Missing ✗").ljust(16) + """ │
   │ indicators.fred_code│ Required         │ """ + ("Present ✓" if "fred_code" in actual["tables"].get("indicators", {}).get("columns", []) else "Missing ✗").ljust(16) + """ │
   │ indicators.category │ Required         │ """ + ("Present ✓" if "category" in actual["tables"].get("indicators", {}).get("columns", []) else "Missing ✗").ljust(16) + """ │
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
