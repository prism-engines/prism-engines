#!/usr/bin/env python3
"""
PRISM Clean Data Audit Report

Compares raw.indicators vs clean.indicators using the DataQualityAuditAgent.
Read-only analysis — no data modifications.

Usage:
    python scripts/audit_clean_report.py
    python scripts/audit_clean_report.py --indicators SPY,GLD,TLT
    python scripts/audit_clean_report.py --output reports/audit_2024.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.db.connection import get_db_path
from prism.agents import (
    AgentOrchestrator,
    DataQualityAuditAgent,
    GeometryDiagnosticAgent,
    EngineRoutingAgent,
    DiagnosticRegistry,
)


def load_raw_data(conn: duckdb.DuckDBPyConnection, indicators: list) -> pd.DataFrame:
    """Load data from raw.indicators."""
    ind_list = ", ".join(f"'{i}'" for i in indicators)
    
    query = f"""
        SELECT date, indicator_id, value
        FROM raw.indicators
        WHERE indicator_id IN ({ind_list})
        ORDER BY indicator_id, date
    """
    
    df = conn.execute(query).fetchdf()
    
    if df.empty:
        return pd.DataFrame()
    
    # Pivot to wide format
    df_wide = df.pivot(index='date', columns='indicator_id', values='value')
    return df_wide


def load_clean_data(conn: duckdb.DuckDBPyConnection, indicators: list) -> pd.DataFrame:
    """Load data from clean.indicators."""
    ind_list = ", ".join(f"'{i}'" for i in indicators)
    
    query = f"""
        SELECT date, indicator_id, value
        FROM clean.indicators
        WHERE indicator_id IN ({ind_list})
        ORDER BY indicator_id, date
    """
    
    df = conn.execute(query).fetchdf()
    
    if df.empty:
        return pd.DataFrame()
    
    # Pivot to wide format
    df_wide = df.pivot(index='date', columns='indicator_id', values='value')
    return df_wide


def get_available_indicators(conn: duckdb.DuckDBPyConnection) -> list:
    """Get list of indicators available in both raw and clean."""
    query = """
        SELECT DISTINCT r.indicator_id
        FROM raw.indicators r
        INNER JOIN clean.indicators c ON r.indicator_id = c.indicator_id
        ORDER BY r.indicator_id
    """
    
    result = conn.execute(query).fetchdf()
    return result['indicator_id'].tolist()


def compute_basic_stats(df: pd.DataFrame, label: str) -> dict:
    """Compute basic statistics for a dataset."""
    if df.empty:
        return {"label": label, "error": "No data"}
    
    stats = {
        "label": label,
        "n_indicators": len(df.columns),
        "n_rows": len(df),
        "date_range": {
            "start": str(df.index.min().date()) if hasattr(df.index.min(), 'date') else str(df.index.min()),
            "end": str(df.index.max().date()) if hasattr(df.index.max(), 'date') else str(df.index.max()),
        },
        "total_cells": df.size,
        "null_cells": int(df.isna().sum().sum()),
        "null_rate": float(df.isna().sum().sum() / df.size) if df.size > 0 else 0,
        "per_indicator": {}
    }
    
    for col in df.columns:
        series = df[col]
        stats["per_indicator"][col] = {
            "count": int(series.count()),
            "nulls": int(series.isna().sum()),
            "null_rate": float(series.isna().mean()),
            "mean": float(series.mean()) if series.count() > 0 else None,
            "std": float(series.std()) if series.count() > 1 else None,
            "min": float(series.min()) if series.count() > 0 else None,
            "max": float(series.max()) if series.count() > 0 else None,
        }
    
    return stats


def run_audit_agent(raw_df: pd.DataFrame, clean_df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run the DataQualityAuditAgent and return results."""
    
    if verbose:
        print("\n" + "-" * 50)
        print("RUNNING AUDIT AGENTS")
        print("-" * 50)
    
    # Set up orchestrator with audit agent
    orch = AgentOrchestrator()
    orch.register_agent(DataQualityAuditAgent(orch.registry))
    orch.register_agent(GeometryDiagnosticAgent(orch.registry))
    orch.register_agent(EngineRoutingAgent(orch.registry))
    
    # Prepare data for agent (handles both single series and multi-series)
    if len(raw_df.columns) == 1:
        raw_data = raw_df.iloc[:, 0].values
        clean_data = clean_df.iloc[:, 0].values
    else:
        # Multi-series: pass as dict
        raw_data = {col: raw_df[col].values for col in raw_df.columns}
        clean_data = {col: clean_df[col].values for col in clean_df.columns}
    
    if verbose:
        print(f"  [1/3] DataQualityAuditAgent... ", end="", flush=True)
    
    # Run pipeline
    result = orch.run_pipeline(
        raw_data=raw_data,
        cleaned_data=clean_data,
    )
    
    # Extract results with verbose output
    agent_results = {
        "quality": None,
        "geometry": None,
        "routing": None,
        "audit_trail": [],
    }
    
    if result.quality:
        agent_results["quality"] = {
            "safety_flag": result.quality.safety_flag,
            "missing_rate": result.quality.missing_rate,
            "variance_ratio": getattr(result.quality, 'variance_ratio', None),
            "distribution_shift": getattr(result.quality, 'distribution_shift', None),
            "warnings": getattr(result.quality, 'reasons', []),
        }
        if verbose:
            flag = result.quality.safety_flag.upper()
            symbol = "✓" if flag == "SAFE" else "✗"
            print(f"{symbol} {flag}")
            if result.quality.missing_rate > 0:
                print(f"      Missing rate: {result.quality.missing_rate:.1%}")
            if getattr(result.quality, 'reasons', []):
                for reason in result.quality.reasons:
                    print(f"      Warning: {reason}")
    
    if verbose:
        print(f"  [2/3] GeometryDiagnosticAgent... ", end="", flush=True)
    
    if result.geometry:
        agent_results["geometry"] = {
            "dominant_geometry": result.geometry.dominant_geometry.value if result.geometry.dominant_geometry else None,
            "confidence": result.geometry.confidence,
            "is_hybrid": result.geometry.is_hybrid,
            "scores": {
                "latent_flow": result.geometry.latent_flow_score,
                "oscillator": result.geometry.oscillator_score,
                "reflexive": result.geometry.reflexive_score,
                "noise": result.geometry.noise_score,
            }
        }
        if verbose:
            geom = result.geometry.dominant_geometry.value if result.geometry.dominant_geometry else "unknown"
            conf = result.geometry.confidence
            print(f"✓ {geom} (confidence: {conf:.2f})")
            print(f"      Scores: latent={result.geometry.latent_flow_score:.2f}, "
                  f"osc={result.geometry.oscillator_score:.2f}, "
                  f"reflex={result.geometry.reflexive_score:.2f}, "
                  f"noise={result.geometry.noise_score:.2f}")
    
    if verbose:
        print(f"  [3/3] EngineRoutingAgent... ", end="", flush=True)
    
    if result.routing:
        # Extract allowed/suppressed from engine_eligibility
        eligibility = result.routing.engine_eligibility
        allowed = [e for e, s in eligibility.items() if s == "allowed"]
        downweighted = [e for e, s in eligibility.items() if s == "downweighted"]
        suppressed = [e for e, s in eligibility.items() if s == "suppressed"]
        
        agent_results["routing"] = {
            "allowed_engines": allowed,
            "downweighted_engines": downweighted,
            "suppressed_engines": suppressed,
            "engine_weights": result.routing.weights,
            "suppression_reasons": result.routing.suppression_reasons,
        }
        if verbose:
            print(f"✓ {len(allowed)} allowed, {len(downweighted)} downweighted, {len(suppressed)} suppressed")
            if allowed:
                print(f"      Allowed: {', '.join(sorted(allowed)[:8])}")
            if downweighted:
                print(f"      Downweighted: {', '.join(sorted(downweighted)[:5])}")
    
    # Audit trail
    for entry in result.audit_trail:
        agent_results["audit_trail"].append({
            "timestamp": entry.timestamp,
            "agent": entry.agent_name,
            "decision": entry.decision,
            "reason": entry.reason,
        })
    
    if verbose:
        print("-" * 50)
    
    return agent_results


def generate_report(
    raw_stats: dict,
    clean_stats: dict,
    agent_results: dict,
    indicators: list,
) -> dict:
    """Generate the full audit report."""
    
    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "report_type": "clean_data_audit",
            "indicators_analyzed": indicators,
        },
        "summary": {
            "raw_null_rate": raw_stats.get("null_rate", 0),
            "clean_null_rate": clean_stats.get("null_rate", 0),
            "rows_raw": raw_stats.get("n_rows", 0),
            "rows_clean": clean_stats.get("n_rows", 0),
            "rows_delta": clean_stats.get("n_rows", 0) - raw_stats.get("n_rows", 0),
            "safety_flag": agent_results.get("quality", {}).get("safety_flag", "unknown"),
            "dominant_geometry": agent_results.get("geometry", {}).get("dominant_geometry", "unknown"),
        },
        "raw_data": raw_stats,
        "clean_data": clean_stats,
        "agent_analysis": agent_results,
        "comparison": {},
    }
    
    # Per-indicator comparison
    if raw_stats.get("per_indicator") and clean_stats.get("per_indicator"):
        for ind in indicators:
            raw_ind = raw_stats["per_indicator"].get(ind, {})
            clean_ind = clean_stats["per_indicator"].get(ind, {})
            
            report["comparison"][ind] = {
                "raw_count": raw_ind.get("count", 0),
                "clean_count": clean_ind.get("count", 0),
                "rows_dropped": raw_ind.get("count", 0) - clean_ind.get("count", 0),
                "raw_nulls": raw_ind.get("nulls", 0),
                "clean_nulls": clean_ind.get("nulls", 0),
                "nulls_removed": raw_ind.get("nulls", 0) - clean_ind.get("nulls", 0),
            }
    
    return report


def print_report(report: dict):
    """Print human-readable report to console."""
    
    print("\n" + "=" * 70)
    print("PRISM CLEAN DATA AUDIT REPORT")
    print("=" * 70)
    print(f"Generated: {report['meta']['generated_at']}")
    print(f"Indicators: {', '.join(report['meta']['indicators_analyzed'])}")
    
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    summary = report["summary"]
    print(f"  Safety Flag:      {summary['safety_flag'].upper()}")
    print(f"  Dominant Geometry: {summary['dominant_geometry']}")
    print(f"  Raw Null Rate:    {summary['raw_null_rate']:.4%}")
    print(f"  Clean Null Rate:  {summary['clean_null_rate']:.4%}")
    print(f"  Rows (raw):       {summary['rows_raw']}")
    print(f"  Rows (clean):     {summary['rows_clean']}")
    print(f"  Rows Delta:       {summary['rows_delta']:+d}")
    
    print("\n" + "-" * 70)
    print("PER-INDICATOR COMPARISON")
    print("-" * 70)
    print(f"  {'Indicator':<15} {'Raw':>8} {'Clean':>8} {'Dropped':>8} {'Nulls Fixed':>12}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    
    for ind, comp in report.get("comparison", {}).items():
        print(f"  {ind:<15} {comp['raw_count']:>8} {comp['clean_count']:>8} "
              f"{comp['rows_dropped']:>8} {comp['nulls_removed']:>12}")
    
    print("\n" + "-" * 70)
    print("GEOMETRY PROFILE")
    print("-" * 70)
    geo = report.get("agent_analysis", {}).get("geometry", {})
    if geo:
        scores = geo.get("scores", {})
        print(f"  Latent Flow:  {scores.get('latent_flow', 0):.3f}")
        print(f"  Oscillator:   {scores.get('oscillator', 0):.3f}")
        print(f"  Reflexive:    {scores.get('reflexive', 0):.3f}")
        print(f"  Noise:        {scores.get('noise', 0):.3f}")
        print(f"  Confidence:   {geo.get('confidence', 0):.3f}")
        print(f"  Hybrid:       {geo.get('is_hybrid', False)}")
    
    print("\n" + "-" * 70)
    print("ENGINE ROUTING")
    print("-" * 70)
    routing = report.get("agent_analysis", {}).get("routing", {})
    if routing:
        allowed = routing.get('allowed_engines', [])
        downweighted = routing.get('downweighted_engines', [])
        suppressed = routing.get('suppressed_engines', [])
        print(f"  Allowed:     {', '.join(allowed) if allowed else 'none'}")
        print(f"  Downweighted: {', '.join(downweighted) if downweighted else 'none'}")
        print(f"  Suppressed:  {len(suppressed)} engines")
    
    print("\n" + "-" * 70)
    print("AUDIT TRAIL")
    print("-" * 70)
    for entry in report.get("agent_analysis", {}).get("audit_trail", []):
        print(f"  [{entry['agent']}] {entry['decision']}")
        print(f"      Reason: {entry['reason']}")
    
    print("\n" + "=" * 70)
    

def main():
    parser = argparse.ArgumentParser(description="PRISM Clean Data Audit Report")
    parser.add_argument(
        "--db",
        default=str(get_db_path()),
        help="Path to DuckDB database"
    )
    parser.add_argument(
        "--indicators",
        default=None,
        help="Comma-separated list of indicators (default: all available)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output"
    )
    
    args = parser.parse_args()
    
    # Connect to DB
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    
    conn = duckdb.connect(str(db_path), read_only=True)
    
    # Get indicators
    if args.indicators:
        indicators = [i.strip() for i in args.indicators.split(",")]
    else:
        indicators = get_available_indicators(conn)
        if not indicators:
            print("Error: No indicators found in both raw and clean tables")
            sys.exit(1)
    
    if not args.quiet:
        print(f"Loading data for {len(indicators)} indicators...")
    
    # Load data
    raw_df = load_raw_data(conn, indicators)
    clean_df = load_clean_data(conn, indicators)
    
    if raw_df.empty:
        print("Error: No raw data found")
        sys.exit(1)
    
    if clean_df.empty:
        print("Error: No clean data found")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Raw data:   {len(raw_df)} rows x {len(raw_df.columns)} indicators")
        print(f"Clean data: {len(clean_df)} rows x {len(clean_df.columns)} indicators")
        print("Running audit agent...")
    
    # Compute stats
    raw_stats = compute_basic_stats(raw_df, "raw")
    clean_stats = compute_basic_stats(clean_df, "clean")
    
    # Align dataframes for agent (use intersection of dates)
    common_dates = raw_df.index.intersection(clean_df.index)
    raw_aligned = raw_df.loc[common_dates]
    clean_aligned = clean_df.loc[common_dates]
    
    # Run audit agent
    try:
        agent_results = run_audit_agent(raw_aligned, clean_aligned, verbose=not args.quiet)
    except Exception as e:
        agent_results = {"error": str(e)}
        if not args.quiet:
            print(f"Warning: Agent analysis failed: {e}")
    
    # Generate report
    report = generate_report(raw_stats, clean_stats, agent_results, indicators)
    
    # Output
    if not args.quiet:
        print_report(report)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {output_path}")
    
    conn.close()
    
    return report


if __name__ == "__main__":
    main()
