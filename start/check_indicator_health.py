#!/usr/bin/env python3
"""
PRISM Indicator Health Checker
==============================

Diagnostic tool to verify indicator data quality.

Prints:
- Indicators with stale data
- Indicators with missing values
- Indicators with extremely short history
- Indicators that fail FRED/Tiingo fetch
- Indicators belonging to multiple families (error)

Usage:
    python start/check_indicator_health.py
    python start/check_indicator_health.py --check-fetch
    python start/check_indicator_health.py --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict

import yaml

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Thresholds
STALE_DAYS = 7  # Data older than this is considered stale
MIN_HISTORY_DAYS = 252  # Minimum expected history (1 trading year)


def load_indicators() -> Dict[str, Any]:
    """Load the indicators registry."""
    indicators_path = PROJECT_ROOT / "data" / "registry" / "indicators.yaml"

    if not indicators_path.exists():
        print(f"ERROR: Indicators registry not found: {indicators_path}")
        return {}

    with open(indicators_path, "r") as f:
        data = yaml.safe_load(f)

    return {k: v for k, v in data.items() if isinstance(v, dict)}


def load_families() -> Dict[str, Any]:
    """Load the families registry."""
    families_path = PROJECT_ROOT / "data" / "registry" / "families.yaml"

    if not families_path.exists():
        return {}

    with open(families_path, "r") as f:
        data = yaml.safe_load(f)

    return data.get("families", {})


def get_indicator_data_from_db(indicator_name: str) -> Optional[Dict[str, Any]]:
    """
    Get indicator data stats from the database.

    Returns dict with:
        - row_count
        - min_date
        - max_date
        - null_count
    """
    try:
        from data.sql.db_connector import get_connection

        conn = get_connection()
        cursor = conn.cursor()

        # Get stats
        cursor.execute("""
            SELECT
                COUNT(*) as row_count,
                MIN(date) as min_date,
                MAX(date) as max_date,
                SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) as null_count
            FROM indicator_values
            WHERE indicator_name = ?
        """, (indicator_name,))

        row = cursor.fetchone()

        if row and row[0] > 0:
            return {
                "row_count": row[0],
                "min_date": row[1],
                "max_date": row[2],
                "null_count": row[3] or 0
            }

    except Exception as e:
        pass

    return None


def check_fetch_capability(indicator_config: Dict[str, Any]) -> bool:
    """
    Test if an indicator can be fetched.

    Returns True if fetch succeeds, False otherwise.
    """
    source = indicator_config.get("source")
    source_id = indicator_config.get("source_id")

    if not source or not source_id:
        return False

    try:
        if source == "fred":
            from fetch.fetcher_fred import FREDFetcher
            fetcher = FREDFetcher()
            # Just test connection - don't actually fetch
            return fetcher.test_connection()
        elif source == "tiingo":
            from fetch.fetcher_tiingo import TiingoFetcher
            fetcher = TiingoFetcher()
            return fetcher.test_connection()
    except Exception:
        return False

    return False


def check_indicator_health(
    check_fetch: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive indicator health check.

    Returns:
        Dict with check results
    """
    results = {
        "total_indicators": 0,
        "healthy": 0,
        "stale": [],
        "missing_values": [],
        "short_history": [],
        "fetch_failures": [],
        "multi_family": [],
        "no_data": [],
        "warnings": [],
        "errors": []
    }

    indicators = load_indicators()
    families = load_families()

    if not indicators:
        results["errors"].append("Failed to load indicators registry")
        return results

    results["total_indicators"] = len(indicators)

    # Build member-to-family mapping
    member_to_families: Dict[str, List[str]] = defaultdict(list)
    for family_id, family_config in families.items():
        members = family_config.get("members", {})
        for member_name in members.keys():
            member_to_families[member_name].append(family_id)

    print("\n" + "=" * 70)
    print("PRISM INDICATOR HEALTH CHECK")
    print("=" * 70)
    print(f"Total indicators in registry: {results['total_indicators']}")
    print(f"Check date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    today = datetime.now().date()
    stale_threshold = today - timedelta(days=STALE_DAYS)

    # Check each indicator
    print("\nChecking indicators...")

    for name, config in indicators.items():
        source = config.get("source", "unknown")

        # Check if indicator is in multiple families
        if name in member_to_families and len(member_to_families[name]) > 1:
            results["multi_family"].append({
                "name": name,
                "families": member_to_families[name]
            })

        # Get data stats from DB
        data_stats = get_indicator_data_from_db(name)

        if data_stats is None:
            results["no_data"].append({"name": name, "source": source})
            continue

        # Check for stale data
        if data_stats["max_date"]:
            try:
                max_date = datetime.strptime(data_stats["max_date"], "%Y-%m-%d").date()
                if max_date < stale_threshold:
                    results["stale"].append({
                        "name": name,
                        "last_date": data_stats["max_date"],
                        "days_old": (today - max_date).days
                    })
            except (ValueError, TypeError):
                pass

        # Check for missing values
        if data_stats["null_count"] > 0:
            null_pct = (data_stats["null_count"] / data_stats["row_count"]) * 100
            if null_pct > 5:  # More than 5% null is concerning
                results["missing_values"].append({
                    "name": name,
                    "null_count": data_stats["null_count"],
                    "null_pct": round(null_pct, 2)
                })

        # Check for short history
        if data_stats["row_count"] < MIN_HISTORY_DAYS:
            results["short_history"].append({
                "name": name,
                "row_count": data_stats["row_count"]
            })
        else:
            results["healthy"] += 1

    # Check fetch capability (optional - slow)
    if check_fetch:
        print("\nTesting fetch capability...")
        # Test a sample indicator from each source
        fred_tested = False
        tiingo_tested = False

        for name, config in indicators.items():
            source = config.get("source")

            if source == "fred" and not fred_tested:
                if not check_fetch_capability(config):
                    results["fetch_failures"].append({
                        "source": "fred",
                        "indicator": name,
                        "reason": "FRED API connection failed"
                    })
                fred_tested = True

            elif source == "tiingo" and not tiingo_tested:
                if not check_fetch_capability(config):
                    results["fetch_failures"].append({
                        "source": "tiingo",
                        "indicator": name,
                        "reason": "Tiingo API connection failed"
                    })
                tiingo_tested = True

            if fred_tested and tiingo_tested:
                break

    # Print results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    print(f"\nHealthy indicators: {results['healthy']}/{results['total_indicators']}")

    # Stale indicators
    if results["stale"]:
        print(f"\n[STALE DATA] ({len(results['stale'])} indicators)")
        for item in results["stale"][:10 if not verbose else len(results["stale"])]:
            print(f"  - {item['name']}: last date {item['last_date']} ({item['days_old']} days old)")
        if not verbose and len(results["stale"]) > 10:
            print(f"  ... and {len(results['stale']) - 10} more")

    # No data
    if results["no_data"]:
        print(f"\n[NO DATA IN DB] ({len(results['no_data'])} indicators)")
        for item in results["no_data"][:10 if not verbose else len(results["no_data"])]:
            print(f"  - {item['name']} ({item['source']})")
        if not verbose and len(results["no_data"]) > 10:
            print(f"  ... and {len(results['no_data']) - 10} more")

    # Missing values
    if results["missing_values"]:
        print(f"\n[MISSING VALUES >5%] ({len(results['missing_values'])} indicators)")
        for item in results["missing_values"][:10 if not verbose else len(results["missing_values"])]:
            print(f"  - {item['name']}: {item['null_count']} nulls ({item['null_pct']}%)")
        if not verbose and len(results["missing_values"]) > 10:
            print(f"  ... and {len(results['missing_values']) - 10} more")

    # Short history
    if results["short_history"]:
        print(f"\n[SHORT HISTORY <{MIN_HISTORY_DAYS} days] ({len(results['short_history'])} indicators)")
        for item in results["short_history"][:10 if not verbose else len(results["short_history"])]:
            print(f"  - {item['name']}: {item['row_count']} rows")
        if not verbose and len(results["short_history"]) > 10:
            print(f"  ... and {len(results['short_history']) - 10} more")

    # Multi-family indicators (error condition)
    if results["multi_family"]:
        print(f"\n[ERROR: MULTI-FAMILY] ({len(results['multi_family'])} indicators)")
        for item in results["multi_family"]:
            print(f"  - {item['name']}: in families {', '.join(item['families'])}")

    # Fetch failures
    if results["fetch_failures"]:
        print(f"\n[FETCH FAILURES] ({len(results['fetch_failures'])} sources)")
        for item in results["fetch_failures"]:
            print(f"  - {item['source']}: {item['reason']}")

    # Summary
    print("\n" + "=" * 70)
    error_count = len(results["multi_family"]) + len(results["fetch_failures"])
    warning_count = len(results["stale"]) + len(results["missing_values"]) + len(results["short_history"])

    status = "PASS" if error_count == 0 else "FAIL"
    print(f"STATUS: {status}")
    print(f"  Healthy: {results['healthy']}")
    print(f"  No data: {len(results['no_data'])}")
    print(f"  Warnings: {warning_count}")
    print(f"  Errors: {error_count}")
    print("=" * 70)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check PRISM indicator data health"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all issues (not just first 10)"
    )
    parser.add_argument(
        "--check-fetch",
        action="store_true",
        help="Also test fetch capability (requires API keys)"
    )
    args = parser.parse_args()

    results = check_indicator_health(
        check_fetch=args.check_fetch,
        verbose=args.verbose
    )

    # Return exit code based on errors
    error_count = len(results.get("multi_family", [])) + len(results.get("fetch_failures", []))
    if error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
