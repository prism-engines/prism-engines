#!/usr/bin/env python3
"""
PRISM Family Registry Checker
=============================

Diagnostic tool to verify the indicator family registry.

Prints:
- All families with their members
- Canonical members per family
- Number of indicators per family
- Warnings for missing canonical
- Warnings for duplicate canonical
- History length comparisons

Usage:
    python start/check_families.py
    python start/check_families.py --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict

import yaml

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_families() -> Dict[str, Any]:
    """Load the families registry."""
    families_path = PROJECT_ROOT / "data" / "registry" / "families.yaml"

    if not families_path.exists():
        print(f"ERROR: Families registry not found: {families_path}")
        return {}

    with open(families_path, "r") as f:
        return yaml.safe_load(f)


def load_indicators() -> Dict[str, Any]:
    """Load the indicators registry."""
    indicators_path = PROJECT_ROOT / "data" / "registry" / "indicators.yaml"

    if not indicators_path.exists():
        print(f"ERROR: Indicators registry not found: {indicators_path}")
        return {}

    with open(indicators_path, "r") as f:
        data = yaml.safe_load(f)

    # Filter out non-dict entries
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def check_families(verbose: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive family registry check.

    Returns:
        Dict with check results and warnings
    """
    results = {
        "total_families": 0,
        "total_members": 0,
        "warnings": [],
        "errors": [],
        "families": {}
    }

    families_data = load_families()
    indicators_data = load_indicators()

    if not families_data:
        results["errors"].append("Failed to load families registry")
        return results

    families = families_data.get("families", {})
    results["total_families"] = len(families)

    # Track all canonical indicators across families
    all_canonicals: Set[str] = set()
    canonical_to_family: Dict[str, str] = {}

    # Track member occurrences
    member_to_families: Dict[str, List[str]] = defaultdict(list)

    print("\n" + "=" * 70)
    print("PRISM FAMILY REGISTRY CHECK")
    print("=" * 70)

    for family_id, family_config in families.items():
        family_info = {
            "canonical_name": family_config.get("canonical_name", "UNNAMED"),
            "members": [],
            "default_representation": None,
            "history_lengths": {},
            "warnings": []
        }

        members = family_config.get("members", {})
        rules = family_config.get("rules", {})
        default_rep = rules.get("default_representation")

        family_info["default_representation"] = default_rep
        family_info["member_count"] = len(members)
        results["total_members"] += len(members)

        # Check each member
        for member_name, member_config in members.items():
            family_info["members"].append(member_name)
            member_to_families[member_name].append(family_id)

            # Record history start
            history_start = member_config.get("history_start")
            if history_start:
                family_info["history_lengths"][member_name] = history_start

        # Check for missing default representation
        if not default_rep:
            warning = f"Family '{family_id}': missing default_representation"
            family_info["warnings"].append(warning)
            results["warnings"].append(warning)
        elif default_rep not in members:
            warning = f"Family '{family_id}': default_representation '{default_rep}' not in members"
            family_info["warnings"].append(warning)
            results["warnings"].append(warning)
        else:
            # Track canonical
            if default_rep in all_canonicals:
                existing_family = canonical_to_family.get(default_rep, "unknown")
                warning = f"DUPLICATE CANONICAL: '{default_rep}' is canonical for both '{existing_family}' and '{family_id}'"
                results["errors"].append(warning)
            else:
                all_canonicals.add(default_rep)
                canonical_to_family[default_rep] = family_id

        results["families"][family_id] = family_info

    # Check for indicators in multiple families
    for member, families_list in member_to_families.items():
        if len(families_list) > 1:
            warning = f"Member '{member}' appears in multiple families: {', '.join(families_list)}"
            results["warnings"].append(warning)

    # Print summary
    print(f"\nTotal families: {results['total_families']}")
    print(f"Total members: {results['total_members']}")

    # Print family details
    print("\n" + "-" * 70)
    print("FAMILIES")
    print("-" * 70)

    for family_id, family_info in results["families"].items():
        print(f"\n[{family_id}] {family_info['canonical_name']}")
        print(f"  Members: {family_info['member_count']}")
        print(f"  Default: {family_info['default_representation'] or 'NOT SET'}")

        if verbose:
            print(f"  Members list: {', '.join(family_info['members'])}")
            if family_info['history_lengths']:
                print("  History starts:")
                for member, start in family_info['history_lengths'].items():
                    print(f"    - {member}: {start}")

        if family_info['warnings']:
            for w in family_info['warnings']:
                print(f"  WARNING: {w}")

    # Print warnings
    if results["warnings"]:
        print("\n" + "-" * 70)
        print(f"WARNINGS ({len(results['warnings'])})")
        print("-" * 70)
        for w in results["warnings"]:
            print(f"  - {w}")

    # Print errors
    if results["errors"]:
        print("\n" + "-" * 70)
        print(f"ERRORS ({len(results['errors'])})")
        print("-" * 70)
        for e in results["errors"]:
            print(f"  - {e}")

    # Print summary
    print("\n" + "=" * 70)
    status = "PASS" if not results["errors"] else "FAIL"
    warning_count = len(results["warnings"])
    error_count = len(results["errors"])
    print(f"STATUS: {status} ({error_count} errors, {warning_count} warnings)")
    print("=" * 70)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check PRISM indicator family registry"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output including member lists"
    )
    args = parser.parse_args()

    results = check_families(verbose=args.verbose)

    # Return exit code based on errors
    if results["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
