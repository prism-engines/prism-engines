#!/usr/bin/env python3
"""
PRISM Agent Pipeline Runner

Runs the 5-agent analysis pipeline:
    Agent 1: Data Quality Audit
    Agent 2: Geometry Diagnostic
    Agent 3: Engine Routing
    [Engines run here]
    Agent 4: Engine Stability Audit
    Agent 5: Blind Spot Detection

Usage:
    python scripts/run_agents.py --indicator SPY
    python scripts/run_agents.py --indicator SPY --window-years 5
    python scripts/run_agents.py --all --window-years 3
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prism.db.open import open_prism_db
from prism.agents.agent_foundation import (
    AgentOrchestrator,
    DiagnosticRegistry,
    PipelineResult,
)
from prism.agents.agent_data_quality import DataQualityAuditAgent
from prism.agents.agent_routing import GeometryDiagnosticAgent, EngineRoutingAgent
from prism.agents.agent_stability import EngineStabilityAuditAgent
from prism.agents.agent_blindspot import BlindSpotDetectionAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_indicator_data(indicator_id: str, window_years: float = 5.0):
    """
    Load raw and cleaned data for an indicator.

    Returns:
        tuple: (raw_data, cleaned_data, metadata)
    """
    conn = open_prism_db()

    # Get data from canonical table
    df = conn.execute("""
        SELECT date, value
        FROM data.indicators
        WHERE indicator_id = ?
        ORDER BY date
    """, [indicator_id]).fetchdf()

    conn.close()

    if df.empty:
        raise ValueError(f"No data found for indicator: {indicator_id}")

    # Apply window filter
    if window_years:
        cutoff = df['date'].max() - np.timedelta64(int(window_years * 365), 'D')
        df = df[df['date'] >= cutoff]

    raw_data = df['value'].values

    # Simple cleaning: forward-fill NaNs, remove outliers beyond 5 std
    cleaned_data = raw_data.copy()

    # Forward-fill NaNs
    mask = np.isnan(cleaned_data)
    if mask.any():
        idx = np.where(~mask, np.arange(len(cleaned_data)), 0)
        np.maximum.accumulate(idx, out=idx)
        cleaned_data = cleaned_data[idx]

    # Remove extreme outliers (clip to 5 std)
    mean, std = np.nanmean(cleaned_data), np.nanstd(cleaned_data)
    if std > 0:
        cleaned_data = np.clip(cleaned_data, mean - 5*std, mean + 5*std)

    metadata = {
        "indicator_id": indicator_id,
        "n_points": len(raw_data),
        "date_range": (str(df['date'].min()), str(df['date'].max())),
        "window_years": window_years,
    }

    return raw_data, cleaned_data, metadata


def create_orchestrator() -> AgentOrchestrator:
    """
    Create and configure the agent orchestrator with all 5 agents.
    """
    # Create shared registry
    registry = DiagnosticRegistry()

    # Create orchestrator
    orchestrator = AgentOrchestrator()

    # Register all 5 agents
    # Agent 1: Data Quality
    orchestrator.register_agent(DataQualityAuditAgent(registry))

    # Agent 2: Geometry Diagnostic
    orchestrator.register_agent(GeometryDiagnosticAgent(registry))

    # Agent 3: Engine Routing
    orchestrator.register_agent(EngineRoutingAgent(registry))

    # Agent 4: Engine Stability (runs after engines)
    orchestrator.register_agent(EngineStabilityAuditAgent(registry))

    # Agent 5: Blind Spot Detection
    orchestrator.register_agent(BlindSpotDetectionAgent(registry))

    # Set agent execution order
    orchestrator.set_pipeline([
        "data_quality",
        "geometry_diagnostic",
        "engine_routing",
        "engine_stability",
        "blind_spot_detection",
    ])

    logger.info(f"Registered agents: {orchestrator.list_agents()}")

    return orchestrator


def run_engines(cleaned_data: np.ndarray, routing_result) -> dict:
    """
    Run engines based on routing result.

    This is a simplified placeholder - in production, this would
    call the actual engine implementations.
    """
    engine_results = {}

    if routing_result is None:
        return engine_results

    # Get allowed engines from routing
    allowed_engines = [
        engine for engine, status in routing_result.engine_eligibility.items()
        if status == "allowed"
    ]

    logger.info(f"Running {len(allowed_engines)} engines: {allowed_engines}")

    # Placeholder - actual engine execution would go here
    # For now, just return empty results to demonstrate pipeline flow
    for engine in allowed_engines:
        engine_results[engine] = {
            "engine": engine,
            "status": "placeholder",
            "note": "Actual engine execution not implemented in this script",
        }

    return engine_results


def print_result(result: PipelineResult, metadata: dict):
    """Print pipeline results in a readable format."""
    print("\n" + "=" * 70)
    print(f"PRISM AGENT PIPELINE RESULTS")
    print(f"Indicator: {metadata['indicator_id']}")
    print(f"Window: {metadata['window_years']} years ({metadata['n_points']} points)")
    print("=" * 70)

    # Agent 1: Data Quality
    print("\n[Agent 1: Data Quality]")
    if result.quality:
        print(f"  Safety Flag: {result.quality.safety_flag}")
        if result.quality.reasons:
            for reason in result.quality.reasons:
                print(f"    - {reason}")
    else:
        print("  (not run)")

    # Agent 2: Geometry
    print("\n[Agent 2: Geometry Diagnostic]")
    if result.geometry:
        print(f"  Geometry Type: {getattr(result.geometry, 'geometry_type', 'unknown')}")
        print(f"  Confidence: {getattr(result.geometry, 'confidence', 0):.2f}")
    else:
        print("  (not run)")

    # Agent 3: Routing
    print("\n[Agent 3: Engine Routing]")
    if result.routing:
        allowed = [e for e, s in result.routing.engine_eligibility.items() if s == "allowed"]
        suppressed = [e for e, s in result.routing.engine_eligibility.items() if s == "suppressed"]
        print(f"  Allowed: {len(allowed)} engines")
        print(f"  Suppressed: {len(suppressed)} engines")
        if suppressed and result.routing.suppression_reasons:
            print("  Suppression reasons:")
            for engine, reason in list(result.routing.suppression_reasons.items())[:3]:
                print(f"    - {engine}: {reason}")
    else:
        print("  (not run)")

    # Agent 4: Stability
    print("\n[Agent 4: Engine Stability]")
    if result.stability:
        print(f"  Trustworthiness scores: {len(result.stability.trustworthiness_scores)} engines")
        if result.stability.artifact_flags:
            print(f"  Artifact flags: {result.stability.artifact_flags}")
    else:
        print("  (not run or no engines executed)")

    # Agent 5: Blind Spots
    print("\n[Agent 5: Blind Spot Detection]")
    if result.blindspots:
        if result.blindspots.blind_spots:
            print(f"  Blind spots detected: {len(result.blindspots.blind_spots)}")
            for spot in result.blindspots.blind_spots[:3]:
                print(f"    - {spot}")
        else:
            print("  No blind spots detected")

        if result.blindspots.escalation_recommendations:
            print("  Recommendations:")
            for rec in result.blindspots.escalation_recommendations[:3]:
                print(f"    - {rec}")
    else:
        print("  (not run)")

    # Warnings
    if result.warnings:
        print("\n[Warnings]")
        for warning in result.warnings:
            print(f"  - {warning}")

    # Consensus
    if result.consensus:
        print("\n[Consensus Report]")
        print(f"  Contributing engines: {result.consensus.n_contributing}")
        print(f"  Consensus confidence: {result.consensus.confidence:.2f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PRISM Agent Pipeline Runner")
    parser.add_argument("--indicator", "-i", type=str, help="Indicator ID to analyze")
    parser.add_argument("--all", action="store_true", help="Run on all indicators")
    parser.add_argument("--window-years", "-w", type=float, default=5.0, help="Window size in years")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max indicators when using --all")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create orchestrator
    orchestrator = create_orchestrator()

    # Get indicator(s) to process
    if args.all:
        conn = open_prism_db()
        indicators = conn.execute("""
            SELECT DISTINCT indicator_id
            FROM data.indicators
            LIMIT ?
        """, [args.limit]).fetchall()
        conn.close()
        indicator_ids = [row[0] for row in indicators]
    elif args.indicator:
        indicator_ids = [args.indicator]
    else:
        print("Error: Must specify --indicator or --all")
        sys.exit(1)

    # Process each indicator
    for indicator_id in indicator_ids:
        try:
            logger.info(f"Processing indicator: {indicator_id}")

            # Load data
            raw_data, cleaned_data, metadata = load_indicator_data(
                indicator_id,
                window_years=args.window_years
            )

            # Run pipeline
            result = orchestrator.run_pipeline(
                raw_data=raw_data,
                cleaned_data=cleaned_data,
                run_engines_fn=run_engines,
                window_size=len(cleaned_data),
            )

            # Print results
            print_result(result, metadata)

        except Exception as e:
            logger.error(f"Failed to process {indicator_id}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
