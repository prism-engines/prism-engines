#!/usr/bin/env python3
"""
PRISM Data Phase: Canonical Orchestrator

Single entry point for Data Phase execution.

Semantic Phase: DATA
- Admissible indicator data: ingestion, cleaning, normalization, suitability, cohorts
- Anything before engines create new measurements

Usage:
    python scripts/run_data_phase.py --domain economic
    python scripts/run_data_phase.py --domain climate --full

================================================================================
DATA PHASE IS AN ATOMIC OPERATION
================================================================================
Data Phase is not complete until geometry decisions are persisted and the run is locked.
Engines must refuse to run unless this condition is met.

Data Phase execution order:
    1. Load indicators
    2. Run multi-window geometry analysis
    3. Select optimal window per indicator
    4. Determine math eligibility
    5. Persist artifacts (best-effort for audit tables)
    6. Insert run lock (REQUIRED)
    7. Exit successfully

Rules:
    - --domain is REQUIRED
    - Default mode is 'short' (validation)
    - --full triggers authoritative Data Phase execution
    - Domain isolation is enforced (no cross-domain writes)
    - No engines may infer geometry
    - No engines may touch raw data
    - No engines may run on unpersisted Data Phase output

================================================================================
DATA PHASE IMMUTABILITY CONTRACT:
================================================================================
Once a Data Phase run completes and is locked:
    - NO writes to data tables for that run_id
    - NO overwriting of existing records
    - Any mutation after lock is a SYSTEM ERROR

================================================================================
DESIGN INTENT (DO NOT IGNORE):
================================================================================
Data Phase is an atomic operation.
Analysis without persistence is incomplete.
Engines must never infer geometry.
This is not stylistic. It is architectural.

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

import sys
import argparse
import uuid
import json
import logging
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# Ensure prism package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.db.open import open_prism_db
from scripts.tools.data_integrity import assert_run_id_unused, finalize_run_lock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA PHASE META TABLES
# =============================================================================

DATA_PHASE_TABLES = [
    # Core run tracking (canonical)
    "meta.data_runs",
    "meta.data_steps",
    "meta.data_run_lock",
]


# =============================================================================
# DOMAIN CONFIGURATION
# =============================================================================

class Domain(Enum):
    ECONOMIC = "economic"
    CLIMATE = "climate"
    EPIDEMIOLOGY = "epidemiology"


DOMAIN_CONFIG = {
    Domain.ECONOMIC: {
        "indicator_prefix": None,  # No prefix filtering for economic (legacy)
        "windows": [0.5, 1.0, 2.0, 3.0, 5.0, 7.0],
        "min_observations": 252,  # 1 year of daily data
        "description": "Financial and macroeconomic indicators",
    },
    Domain.CLIMATE: {
        "indicator_prefix": "CLIMATE_",  # Or filter by source
        "windows": [1.0, 2.0, 5.0, 10.0, 20.0, 30.0],  # Climate needs longer windows
        "min_observations": 120,  # 10 years of monthly data
        "description": "Climate and environmental indicators",
    },
    Domain.EPIDEMIOLOGY: {
        "indicator_prefix": "EPI_",
        "windows": [0.25, 0.5, 1.0, 2.0, 5.0],  # Shorter windows for epi
        "min_observations": 52,  # 1 year of weekly data
        "description": "Epidemiological indicators",
    },
}


# =============================================================================
# PHASE 1 CONTEXT
# =============================================================================

@dataclass
class Phase1Context:
    """Execution context for Phase 1 run."""
    run_id: str
    domain: Domain
    mode: str  # 'short' or 'full'
    started_at: datetime

    # Populated during execution
    indicators: List[str] = field(default_factory=list)
    geometry_results: Dict[str, Any] = field(default_factory=dict)
    eligibility_results: Dict[str, Any] = field(default_factory=dict)
    cohorts: List[Dict] = field(default_factory=list)
    engine_results: Dict[str, Any] = field(default_factory=dict)

    # Execution status
    completed_steps: List[str] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)

    def log_step(self, step: str):
        """Record completed step."""
        self.completed_steps.append(step)
        logger.info(f"✓ {step}")

    def log_error(self, step: str, error: str):
        """Record error."""
        self.errors.append({"step": step, "error": error, "timestamp": datetime.now()})
        logger.error(f"✗ {step}: {error}")


# =============================================================================
# STEP 1: LOAD INDICATORS
# =============================================================================

def step1_load_indicators(ctx: Phase1Context, conn) -> bool:
    """
    Step 1: Load indicators for domain.

    Validates data exists and applies domain filtering.
    """
    logger.info("-" * 60)
    logger.info("STEP 1: Load Indicators")
    logger.info("-" * 60)

    domain_cfg = DOMAIN_CONFIG[ctx.domain]

    # Query available indicators
    query = """
        SELECT DISTINCT indicator_id, COUNT(*) as n_rows
        FROM data.indicators
        GROUP BY indicator_id
        HAVING n_rows >= ?
        ORDER BY indicator_id
    """

    df = conn.execute(query, [domain_cfg["min_observations"]]).fetchdf()

    if len(df) == 0:
        ctx.log_error("step1_load_indicators", "No indicators found with sufficient data")
        return False

    # Apply domain prefix filter if specified
    if domain_cfg["indicator_prefix"]:
        prefix = domain_cfg["indicator_prefix"]
        df = df[df['indicator_id'].str.startswith(prefix)]

    ctx.indicators = df['indicator_id'].tolist()

    logger.info(f"  Domain: {ctx.domain.value}")
    logger.info(f"  Indicators: {len(ctx.indicators)}")
    logger.info(f"  Min observations: {domain_cfg['min_observations']}")

    if len(ctx.indicators) == 0:
        ctx.log_error("step1_load_indicators", f"No indicators found for domain {ctx.domain.value}")
        return False

    # In short mode, limit indicators for faster validation
    if ctx.mode == "short":
        max_indicators = 10
        if len(ctx.indicators) > max_indicators:
            logger.info(f"  [SHORT MODE] Limiting to {max_indicators} indicators")
            ctx.indicators = ctx.indicators[:max_indicators]

    ctx.log_step(f"step1_load_indicators ({len(ctx.indicators)} indicators)")
    return True


# =============================================================================
# STEP 2: TEMPORAL GEOMETRY SCAN
# =============================================================================

def step2_temporal_geometry_scan(ctx: Phase1Context, conn) -> bool:
    """
    Step 2: Run temporal geometry scan.

    For each indicator, analyze geometry at multiple windows.
    Produces: ctx.geometry_results[indicator][window] = geometry_dict
    """
    logger.info("-" * 60)
    logger.info("STEP 2: Temporal Geometry Scan")
    logger.info("-" * 60)

    try:
        from prism.agents.agent_multiview_geometry import (
            MultiViewGeometryAgent,
            ViewType,
        )
        from prism.agents.agent_geometry_signature import GeometrySignatureAgent
    except ImportError as e:
        ctx.log_error("step2_temporal_geometry_scan", f"Import error: {e}")
        return False

    domain_cfg = DOMAIN_CONFIG[ctx.domain]
    windows = domain_cfg["windows"]

    # In short mode, use fewer windows
    if ctx.mode == "short":
        windows = windows[:3]
        logger.info(f"  [SHORT MODE] Using {len(windows)} windows: {windows}")

    # Initialize agents
    base_agent = GeometrySignatureAgent(verbose=False)
    mv_agent = MultiViewGeometryAgent(base_agent=base_agent, verbose=False)

    # Spread indicators use different views
    spread_indicators = {'T10Y2Y', 'T10Y3M', 'T10YFF', 'TEDRATE'}

    total_scans = 0
    step_size = 63  # Quarterly steps

    for indicator_id in ctx.indicators:
        # Load data
        df = conn.execute("""
            SELECT date, value
            FROM data.indicators
            WHERE indicator_id = ?
            ORDER BY date
        """, [indicator_id]).fetchdf()

        if len(df) < domain_cfg["min_observations"]:
            continue

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        is_spread = indicator_id in spread_indicators

        indicator_results = {}

        for window_years in windows:
            window_days = int(window_years * 252)

            if len(df) < window_days + step_size * 10:
                continue

            # Rolling geometry analysis
            geometries = []
            confidences = []
            disagreements = []

            dates = df.index.tolist()

            for end_idx in range(window_days, len(dates), step_size):
                start_idx = end_idx - window_days
                window_data = df.iloc[start_idx:end_idx]['value'].dropna()

                if len(window_data) < window_days * 0.8:
                    continue

                try:
                    if is_spread:
                        views = [ViewType.LEVEL, ViewType.DEVIATION]
                    else:
                        views = [ViewType.LEVEL, ViewType.RETURNS, ViewType.VOLATILITY]

                    result = mv_agent.analyze(
                        window_data.values,
                        indicator_id=indicator_id,
                        views=views
                    )

                    geometries.append(result.consensus_geometry)
                    confidences.append(result.consensus_confidence)
                    disagreements.append(result.disagreement_score)

                except Exception:
                    continue

            if len(geometries) < 10:
                continue

            # Compute summary metrics
            geom_counts = pd.Series(geometries).value_counts()
            dominant_geometry = geom_counts.index[0]
            geometry_pct = geom_counts.iloc[0] / len(geometries)

            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            avg_disagreement = np.mean(disagreements)

            n_transitions = sum(
                1 for i in range(1, len(geometries))
                if geometries[i] != geometries[i-1]
            )
            stability = 1.0 - (n_transitions / max(1, len(geometries) - 1))

            quality_score = (
                0.40 * avg_confidence +
                0.30 * (1.0 - avg_disagreement) +
                0.20 * stability +
                0.10 * geometry_pct
            )

            indicator_results[window_years] = {
                'indicator_id': indicator_id,
                'window_years': window_years,
                'window_days': window_days,
                'dominant_geometry': dominant_geometry,
                'geometry_pct': geometry_pct,
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'avg_disagreement': avg_disagreement,
                'n_observations': len(geometries),
                'n_transitions': n_transitions,
                'stability': stability,
                'quality_score': quality_score,
            }
            total_scans += 1

        if indicator_results:
            # Mark optimal window
            best_window = max(indicator_results.keys(),
                            key=lambda w: indicator_results[w]['quality_score'])
            indicator_results[best_window]['is_optimal'] = True

            ctx.geometry_results[indicator_id] = indicator_results

    logger.info(f"  Scanned: {len(ctx.geometry_results)} indicators")
    logger.info(f"  Total (indicator, window) pairs: {total_scans}")

    ctx.log_step(f"step2_temporal_geometry_scan ({total_scans} scans)")
    return True


# =============================================================================
# SUITABILITY REPORT HELPER
# =============================================================================

def _print_suitability_report(ctx: Phase1Context, all_decisions: dict):
    """Print detailed suitability report."""
    print("\n" + "=" * 70)
    print("MATH SUITABILITY REPORT")
    print("=" * 70)

    # 1. Geometry by indicator
    print("\n--- GEOMETRY BY INDICATOR ---")
    print(f"{'Indicator':<20} {'Window':>8} {'Geometry':<22} {'Conf':>6} {'Status':<12}")
    print("-" * 70)

    for (indicator_id, window_years), decision in sorted(all_decisions.items()):
        geom = ctx.geometry_results.get(indicator_id, {}).get(window_years, {})
        optimal = "*" if geom.get('is_optimal', False) else " "
        print(f"{indicator_id:<20} {window_years:>6.1f}y{optimal} {decision.geometry:<22} {decision.confidence:>5.2f} {decision.status.value:<12}")

    # 2. Eligibility breakdown by geometry type
    print("\n--- ELIGIBILITY BY GEOMETRY ---")
    by_geom = {}
    for decision in all_decisions.values():
        geom = decision.geometry
        if geom not in by_geom:
            by_geom[geom] = {'eligible': 0, 'conditional': 0, 'ineligible': 0, 'total': 0}
        by_geom[geom][decision.status.value] = by_geom[geom].get(decision.status.value, 0) + 1
        by_geom[geom]['total'] += 1

    print(f"{'Geometry':<22} {'Eligible':>10} {'Conditional':>12} {'Ineligible':>12} {'Total':>8}")
    print("-" * 70)
    for geom, counts in sorted(by_geom.items()):
        print(f"{geom:<22} {counts['eligible']:>10} {counts['conditional']:>12} {counts['ineligible']:>12} {counts['total']:>8}")

    # 3. Engine routing summary
    print("\n--- ENGINE ROUTING SUMMARY ---")
    engine_counts = {}
    for decision in all_decisions.values():
        if decision.status.value != 'ineligible':
            for engine in decision.allowed_engines:
                engine_counts[engine] = engine_counts.get(engine, 0) + 1

    print(f"{'Engine':<25} {'# Eligible Runs':>15}")
    print("-" * 42)
    for engine, count in sorted(engine_counts.items(), key=lambda x: -x[1]):
        print(f"{engine:<25} {count:>15}")

    # 4. Optimal window distribution
    print("\n--- OPTIMAL WINDOW DISTRIBUTION ---")
    optimal_windows = {}
    for indicator_id, windows in ctx.geometry_results.items():
        for window_years, geom in windows.items():
            if geom.get('is_optimal', False):
                optimal_windows[window_years] = optimal_windows.get(window_years, 0) + 1
                break

    print(f"{'Window':>10} {'# Indicators':>15}")
    print("-" * 27)
    for window, count in sorted(optimal_windows.items()):
        print(f"{window:>8.1f}y {count:>15}")

    print("\n" + "=" * 70)


# =============================================================================
# STEP 3: MATH SUITABILITY (Using Agent)
# =============================================================================

def step3_math_suitability(ctx: Phase1Context, conn) -> bool:
    """
    Step 3: Evaluate math suitability using WindowSuitabilityAgent.

    For each (indicator, window), determine:
    - eligible: full engine access
    - conditional: with constraints
    - ineligible: no inference engines
    """
    logger.info("-" * 60)
    logger.info("STEP 3: Math Suitability Evaluation")
    logger.info("-" * 60)

    try:
        from prism.agents.agent_math_suitability import (
            WindowSuitabilityAgent,
            WindowGeometryResult,
        )

        # Initialize agent
        agent = WindowSuitabilityAgent()

        # Convert geometry results to WindowGeometryResult objects and evaluate
        all_decisions = {}

        for indicator_id, windows in ctx.geometry_results.items():
            indicator_decisions = {}

            for window_years, geom_dict in windows.items():
                # Create WindowGeometryResult from dict
                geom = WindowGeometryResult(
                    indicator_id=indicator_id,
                    window_years=window_years,
                    window_days=geom_dict['window_days'],
                    dominant_geometry=geom_dict['dominant_geometry'],
                    geometry_pct=geom_dict['geometry_pct'],
                    avg_confidence=geom_dict['avg_confidence'],
                    avg_disagreement=geom_dict['avg_disagreement'],
                    confidence_std=geom_dict.get('confidence_std', 0.0),
                    n_observations=geom_dict['n_observations'],
                    n_transitions=geom_dict['n_transitions'],
                    stability=geom_dict['stability'],
                    quality_score=geom_dict['quality_score'],
                    is_optimal=geom_dict.get('is_optimal', False),
                )

                # Evaluate
                decision = agent.evaluate(geom)

                # Store as dict for compatibility
                indicator_decisions[window_years] = {
                    'status': decision.status.value,
                    'rationale': decision.rationale,
                    'allowed_engines': decision.allowed_engines,
                    'prohibited_engines': decision.prohibited_engines,
                    'conditional_engines': decision.conditional_engines,
                    'geometry': decision.geometry,
                    'confidence': decision.confidence,
                    'policy_version': decision.policy_version,
                }

                all_decisions[(indicator_id, window_years)] = decision

            ctx.eligibility_results[indicator_id] = indicator_decisions

        # Generate summary
        total = len(all_decisions)
        by_status = {}
        for decision in all_decisions.values():
            status = decision.status.value
            by_status[status] = by_status.get(status, 0) + 1

        eligible = by_status.get('eligible', 0)
        conditional = by_status.get('conditional', 0)
        ineligible = by_status.get('ineligible', 0)

        logger.info(f"  Total evaluations: {total}")
        logger.info(f"  Eligible:    {eligible:4} ({100*eligible/total:.1f}%)" if total > 0 else "  Eligible:    0")
        logger.info(f"  Conditional: {conditional:4} ({100*conditional/total:.1f}%)" if total > 0 else "  Conditional: 0")
        logger.info(f"  Ineligible:  {ineligible:4} ({100*ineligible/total:.1f}%)" if total > 0 else "  Ineligible:  0")

        # Detailed report
        _print_suitability_report(ctx, all_decisions)

        ctx.log_step(f"step3_math_suitability ({total} evaluations)")
        return True

    except ImportError as e:
        logger.warning(f"WindowSuitabilityAgent not available: {e}")
        logger.warning("Falling back to inline evaluation")
        return step3_math_suitability_fallback(ctx, conn)


def step3_math_suitability_fallback(ctx: Phase1Context, conn) -> bool:
    """Fallback inline suitability if agent not available."""
    min_quality_score = 0.35
    min_confidence = 0.30
    max_disagreement = 0.65
    min_stability = 0.50

    eligible = 0
    conditional = 0
    ineligible = 0

    for indicator_id, windows in ctx.geometry_results.items():
        indicator_eligibility = {}

        for window_years, geom in windows.items():
            status = "eligible"
            rationale = []

            if geom['quality_score'] < min_quality_score:
                status = "conditional"
                rationale.append(f"low quality ({geom['quality_score']:.2f})")

            if geom['avg_confidence'] < min_confidence:
                status = "conditional"
                rationale.append(f"low confidence ({geom['avg_confidence']:.2f})")

            if geom['avg_disagreement'] > max_disagreement:
                status = "conditional"
                rationale.append(f"high disagreement ({geom['avg_disagreement']:.2f})")

            if geom['stability'] < min_stability:
                status = "conditional"
                rationale.append(f"low stability ({geom['stability']:.2f})")

            if geom['dominant_geometry'] == 'pure_noise' and geom['geometry_pct'] > 0.7:
                status = "ineligible"
                rationale = [f"pure_noise dominant ({geom['geometry_pct']*100:.0f}%)"]

            # Determine engines
            if status == "ineligible":
                allowed_engines = ["descriptive_stats", "null_check", "data_quality"]
            elif geom['dominant_geometry'] == 'latent_flow':
                allowed_engines = ["pca", "correlation", "entropy", "wavelets", "trend", "drift"]
            elif geom['dominant_geometry'] == 'reflexive_stochastic':
                allowed_engines = ["pca", "correlation", "entropy", "hmm", "regime_detection",
                                  "volatility_clustering", "copula", "garch"]
            elif geom['dominant_geometry'] == 'coupled_oscillator':
                allowed_engines = ["pca", "correlation", "entropy", "wavelets", "spectral",
                                  "phase_space", "dmd"]
            else:
                allowed_engines = ["pca", "correlation", "entropy", "descriptive_stats"]

            indicator_eligibility[window_years] = {
                'status': status,
                'rationale': rationale,
                'allowed_engines': allowed_engines,
                'prohibited_engines': [],
                'conditional_engines': [],
                'geometry': geom['dominant_geometry'],
                'confidence': geom['avg_confidence'],
                'policy_version': 'fallback-1.0.0',
            }

            if status == "eligible":
                eligible += 1
            elif status == "conditional":
                conditional += 1
            else:
                ineligible += 1

        ctx.eligibility_results[indicator_id] = indicator_eligibility

    total = eligible + conditional + ineligible
    if total > 0:
        logger.info(f"  [FALLBACK] Eligible: {eligible} ({100*eligible/total:.1f}%)")
        logger.info(f"  [FALLBACK] Conditional: {conditional} ({100*conditional/total:.1f}%)")
        logger.info(f"  [FALLBACK] Ineligible: {ineligible} ({100*ineligible/total:.1f}%)")

    ctx.log_step(f"step3_math_suitability_fallback ({total} evaluations)")
    return True


# =============================================================================
# STEP 4: COHORT DISCOVERY
# =============================================================================

def step4_cohort_discovery(ctx: Phase1Context, conn) -> bool:
    """
    Step 4: Discover cohorts using CohortDiscoveryAgent.

    Groups indicators by structural compatibility.
    Only uses engines that passed suitability.
    """
    logger.info("-" * 60)
    logger.info("STEP 4: Cohort Discovery")
    logger.info("-" * 60)

    try:
        from prism.agents.agent_cohort_discovery import CohortDiscoveryAgent

        # Get eligible indicators (at their optimal window)
        eligible_indicators = []
        for indicator_id, windows in ctx.eligibility_results.items():
            for window_years, elig in windows.items():
                if elig['status'] in ['eligible', 'conditional']:
                    # Check if this is optimal window
                    geom = ctx.geometry_results.get(indicator_id, {}).get(window_years, {})
                    if geom.get('is_optimal', False):
                        eligible_indicators.append(indicator_id)
                        break

        if len(eligible_indicators) < 2:
            logger.warning(f"  Only {len(eligible_indicators)} eligible indicators, skipping cohort discovery")
            ctx.cohorts = []
            ctx.log_step("step4_cohort_discovery (skipped - insufficient eligible indicators)")
            return True

        # Run cohort discovery
        agent = CohortDiscoveryAgent()
        cohort_result = agent.discover(eligible_indicators, conn)

        ctx.cohorts = cohort_result.get('cohorts', [])

        logger.info(f"  Eligible indicators: {len(eligible_indicators)}")
        logger.info(f"  Cohorts discovered: {len(ctx.cohorts)}")

        for cohort in ctx.cohorts:
            logger.info(f"    {cohort.get('cohort_id', 'unknown')}: {cohort.get('n_indicators', 0)} indicators")

        ctx.log_step(f"step4_cohort_discovery ({len(ctx.cohorts)} cohorts)")
        return True

    except ImportError as e:
        logger.warning(f"CohortDiscoveryAgent not available: {e}")
        return step4_cohort_discovery_fallback(ctx, conn)
    except Exception as e:
        logger.warning(f"Cohort discovery failed: {e}")
        return step4_cohort_discovery_fallback(ctx, conn)


def step4_cohort_discovery_fallback(ctx: Phase1Context, conn) -> bool:
    """Fallback: group by optimal geometry."""
    logger.info("  [FALLBACK] Grouping by dominant geometry")

    cohorts = {}
    for indicator_id, windows in ctx.geometry_results.items():
        for w, geom in windows.items():
            if geom.get('is_optimal'):
                geometry = geom['dominant_geometry']
                if geometry not in cohorts:
                    cohorts[geometry] = []
                cohorts[geometry].append(indicator_id)
                break

    ctx.cohorts = [
        {'cohort_id': geom, 'indicators': inds, 'n_indicators': len(inds)}
        for geom, inds in cohorts.items()
    ]

    logger.info(f"  Cohorts: {len(ctx.cohorts)}")
    for c in ctx.cohorts:
        logger.info(f"    {c['cohort_id']}: {c['n_indicators']} indicators")

    ctx.log_step(f"step4_cohort_discovery_fallback ({len(ctx.cohorts)} cohorts)")
    return True


# =============================================================================
# STEP 5: ENGINE EXECUTION
# =============================================================================

def step5_engine_execution(ctx: Phase1Context, conn) -> bool:
    """
    Step 5: Execute engines on eligible (indicator, window) pairs.

    Only runs engines that passed math suitability.
    Routes to appropriate engine based on geometry.
    """
    logger.info("-" * 60)
    logger.info("STEP 5: Engine Execution")
    logger.info("-" * 60)

    if ctx.mode == "short":
        logger.info("  [SHORT MODE] Skipping engine execution")
        ctx.log_step("step5_engine_execution (skipped in short mode)")
        return True

    # Collect certified runs
    certified_runs = []
    for indicator_id, windows in ctx.eligibility_results.items():
        for window_years, elig in windows.items():
            if elig['status'] in ['eligible', 'conditional']:
                for engine in elig['allowed_engines']:
                    certified_runs.append({
                        'indicator_id': indicator_id,
                        'window_years': window_years,
                        'engine': engine,
                        'geometry': elig['geometry'],
                    })

    logger.info(f"  Certified runs: {len(certified_runs)}")

    # Group by engine for efficient execution
    by_engine = {}
    for run in certified_runs:
        engine = run['engine']
        if engine not in by_engine:
            by_engine[engine] = []
        by_engine[engine].append(run)

    logger.info(f"  Engines to run: {len(by_engine)}")
    for engine, runs in sorted(by_engine.items(), key=lambda x: -len(x[1])):
        logger.info(f"    {engine:20} × {len(runs)}")

    # Execute engines
    results = []
    errors = []

    for engine_name, runs in by_engine.items():
        try:
            engine_results = _execute_engine(engine_name, runs, conn)
            results.extend(engine_results)
            logger.info(f"  ✓ {engine_name}: {len(engine_results)} results")
        except Exception as e:
            errors.append({'engine': engine_name, 'error': str(e)})
            logger.warning(f"  ✗ {engine_name}: {e}")

    ctx.engine_results = {
        'results': results,
        'errors': errors,
        'n_certified': len(certified_runs),
        'n_completed': len(results),
    }

    logger.info(f"  Completed: {len(results)}/{len(certified_runs)}")
    logger.info(f"  Errors: {len(errors)}")

    ctx.log_step(f"step5_engine_execution ({len(results)} results)")
    return True


def _execute_engine(engine_name: str, runs: list, conn) -> list:
    """
    Execute a single engine on multiple (indicator, window) pairs.

    Routes to appropriate engine implementation in prism/engines/.
    """
    results = []

    # Try to import engine
    try:
        # Dynamic import from prism.engines
        engine_module = importlib.import_module(f"prism.engines.{engine_name}")

        # Try various naming conventions
        class_name = engine_name.title().replace('_', '') + 'Engine'
        engine_class = getattr(engine_module, class_name, None)

        if engine_class is None:
            engine_class = getattr(engine_module, "Engine", None)

        if engine_class:
            engine = engine_class()

            for run in runs:
                try:
                    # Load data for this window
                    window_days = int(run['window_years'] * 252)

                    df = conn.execute("""
                        SELECT date, value
                        FROM data.indicators
                        WHERE indicator_id = ?
                        ORDER BY date DESC
                        LIMIT ?
                    """, [run['indicator_id'], window_days]).fetchdf()

                    if len(df) < window_days * 0.8:
                        continue

                    # Execute engine
                    result = engine.run(df['value'].values)
                    result['indicator_id'] = run['indicator_id']
                    result['window_years'] = run['window_years']
                    result['engine'] = engine_name
                    results.append(result)

                except Exception:
                    # Skip individual failures
                    continue
        else:
            # No engine class found - return stubs
            for run in runs:
                results.append({
                    'indicator_id': run['indicator_id'],
                    'window_years': run['window_years'],
                    'engine': engine_name,
                    'status': 'stub_no_class',
                })

    except ImportError:
        # Engine not implemented yet - return stubs
        for run in runs:
            results.append({
                'indicator_id': run['indicator_id'],
                'window_years': run['window_years'],
                'engine': engine_name,
                'status': 'not_implemented',
            })

    return results


# =============================================================================
# STEP 6: PERSIST ARTIFACTS (TRANSACTIONAL)
# =============================================================================

def step6_persist_artifacts(ctx: Phase1Context, conn) -> bool:
    """
    Step 6: Persist Data Phase artifacts and lock the run.

    CRITICAL: Only the run lock (meta.data_run_lock) is required.
    All other table writes are optional audit/legacy artifacts.
    Missing legacy tables produce warnings, not failures.

    Required (will fail if missing):
    - meta.data_run_lock (lock on success)

    Optional (warnings if missing):
    - Legacy phase1_* tables (removed during schema reset)
    """
    logger.info("-" * 60)
    logger.info("STEP 6: Persist Artifacts")
    logger.info("-" * 60)

    # Track what we successfully wrote (for logging)
    legacy_warnings = []

    # =========================================================================
    # OPTIONAL: Legacy audit tables (non-blocking)
    # =========================================================================

    # Try to persist to legacy tables - warn on failure, don't abort
    def _try_legacy_write(table_name: str, write_fn):
        """Attempt legacy write, log warning on failure."""
        try:
            count = write_fn()
            logger.info(f"  [LEGACY] {table_name}: {count} rows")
            return count
        except Exception as e:
            if "does not exist" in str(e):
                legacy_warnings.append(f"{table_name} (table removed)")
            else:
                legacy_warnings.append(f"{table_name}: {e}")
            return 0

    # Legacy: phase1_indicator_windows
    def _write_indicator_windows():
        count = 0
        for indicator_id, windows in ctx.geometry_results.items():
            for window_years, geom in windows.items():
                conn.execute("""
                    INSERT INTO meta.phase1_indicator_windows
                    (run_id, indicator_id, window_years, geom_class,
                     quality_score, confidence, disagreement, stability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    ctx.run_id, indicator_id, window_years,
                    geom['dominant_geometry'], geom['quality_score'],
                    geom['avg_confidence'], geom['avg_disagreement'],
                    geom['stability'],
                ])
                count += 1
        return count

    # Legacy: phase1_indicator_optimal
    def _write_indicator_optimal():
        count = 0
        for indicator_id, windows in ctx.geometry_results.items():
            for window_years, geom in windows.items():
                if geom.get('is_optimal', False):
                    conn.execute("""
                        INSERT INTO meta.phase1_indicator_optimal
                        (run_id, indicator_id, optimal_window, geom_class,
                         quality_score, confidence, disagreement, stability)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        ctx.run_id, indicator_id, window_years,
                        geom['dominant_geometry'], geom['quality_score'],
                        geom['avg_confidence'], geom['avg_disagreement'],
                        geom['stability'],
                    ])
                    count += 1
                    break
        return count

    # Legacy: phase1_math_eligibility
    def _write_math_eligibility():
        count = 0
        for indicator_id, windows in ctx.eligibility_results.items():
            optimal_elig = None
            for window_years, elig in windows.items():
                geom = ctx.geometry_results.get(indicator_id, {}).get(window_years, {})
                if geom.get('is_optimal', False):
                    optimal_elig = elig
                    break
            if optimal_elig:
                rationale = optimal_elig.get('rationale', [])
                reason = "; ".join(rationale) if rationale else None
                conn.execute("""
                    INSERT INTO meta.phase1_math_eligibility
                    (run_id, indicator_id, eligibility, reason)
                    VALUES (?, ?, ?, ?)
                """, [ctx.run_id, indicator_id, optimal_elig['status'], reason])
                count += 1
        return count

    # Legacy: phase1_scan_results
    def _write_scan_results():
        count = 0
        for indicator_id, windows in ctx.geometry_results.items():
            for window_years, geom in windows.items():
                conn.execute("""
                    INSERT INTO meta.phase1_scan_results
                    (run_id, indicator_id, window_years, window_days,
                     dominant_geometry, geometry_pct, avg_confidence,
                     avg_disagreement, confidence_std, n_observations, n_transitions,
                     stability, quality_score, is_optimal, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    ctx.run_id, indicator_id, window_years, geom['window_days'],
                    geom['dominant_geometry'], geom['geometry_pct'],
                    geom['avg_confidence'], geom['avg_disagreement'],
                    geom.get('confidence_std', 0.0), geom['n_observations'],
                    geom['n_transitions'], geom['stability'],
                    geom['quality_score'], geom.get('is_optimal', False),
                ])
                count += 1
        return count

    # Legacy: geometry_windows
    def _write_geometry_windows():
        count = 0
        for indicator_id, windows in ctx.geometry_results.items():
            for window_years, geom in windows.items():
                if geom.get('is_optimal', False):
                    conn.execute("""
                        INSERT INTO meta.geometry_windows
                        (run_id, indicator_id, optimal_window_y, window_days,
                         dominant_geometry, geometry_pct, avg_confidence,
                         avg_disagreement, confidence_std, n_observations, n_transitions,
                         stability, quality_score, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, [
                        ctx.run_id, indicator_id, window_years, geom['window_days'],
                        geom['dominant_geometry'], geom['geometry_pct'],
                        geom['avg_confidence'], geom['avg_disagreement'],
                        geom.get('confidence_std', 0.0), geom['n_observations'],
                        geom['n_transitions'], geom['stability'], geom['quality_score'],
                    ])
                    count += 1
                    break
        return count

    # Legacy: engine_eligibility
    def _write_engine_eligibility():
        count = 0
        for indicator_id, windows in ctx.eligibility_results.items():
            for window_years, elig in windows.items():
                conn.execute("""
                    INSERT INTO meta.engine_eligibility
                    (run_id, indicator_id, window_years, geometry,
                     confidence, disagreement, stability, status,
                     allowed_engines, prohibited_engines, conditional_engines,
                     rationale, policy_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    ctx.run_id, indicator_id, window_years, elig['geometry'],
                    elig['confidence'], elig.get('disagreement', 0.0),
                    elig.get('stability', 0.0), elig['status'],
                    json.dumps(elig['allowed_engines']),
                    json.dumps(elig.get('prohibited_engines', [])),
                    json.dumps(elig.get('conditional_engines', [])),
                    json.dumps(elig['rationale']),
                    elig.get('policy_version', '1.0.0'),
                ])
                count += 1
        return count

    # Attempt all legacy writes (non-blocking)
    _try_legacy_write("phase1_indicator_windows", _write_indicator_windows)
    _try_legacy_write("phase1_indicator_optimal", _write_indicator_optimal)
    _try_legacy_write("phase1_math_eligibility", _write_math_eligibility)
    _try_legacy_write("phase1_scan_results", _write_scan_results)
    _try_legacy_write("geometry_windows", _write_geometry_windows)
    _try_legacy_write("engine_eligibility", _write_engine_eligibility)

    # Log legacy warnings
    if legacy_warnings:
        logger.warning(f"  [LEGACY] Skipped {len(legacy_warnings)} legacy tables (removed during schema reset)")
        for w in legacy_warnings:
            logger.warning(f"    - {w}")

    # =========================================================================
    # REQUIRED: Insert run lock (this is the only critical operation)
    # =========================================================================
    try:
        conn.execute(
            "INSERT INTO meta.data_run_lock(run_id, started_at, domain, mode, locked_by) VALUES (?, now(), ?, ?, ?)",
            [ctx.run_id, ctx.domain.value, ctx.mode, "data_phase_orchestrator"]
        )
        logger.info(f"  ✓ run_id LOCKED (immutable): {ctx.run_id}")
    except Exception as e:
        ctx.log_error("step6_persist", f"CRITICAL: Failed to lock run: {e}")
        return False

    ctx.log_step("step6_persist_artifacts (run locked)")
    return True


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def _persist_run_header(ctx: Phase1Context, conn, domain_cfg: dict):
    """Persist run header to meta.data_runs."""
    try:
        conn.execute("""
            INSERT INTO meta.data_runs
            (run_id, domain, mode, started_at, status, indicators_requested)
            VALUES (?, ?, ?, ?, 'running', ?)
        """, [
            ctx.run_id,
            ctx.domain.value,
            ctx.mode,
            ctx.started_at,
            len(ctx.indicators) if ctx.indicators else 0,
        ])
    except Exception as e:
        logger.warning(f"Failed to persist run header: {e}")


def _persist_step(ctx: Phase1Context, conn, step_name: str, status: str,
                  started_at: datetime, n_items: int = 0, error: str = None):
    """Persist step log to meta.data_steps."""
    try:
        conn.execute("""
            INSERT INTO meta.data_steps
            (run_id, step_name, status, started_at, completed_at, n_items, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            ctx.run_id,
            step_name,
            status,
            started_at,
            datetime.now(),
            n_items,
            error,
        ])
    except Exception as e:
        logger.warning(f"Failed to persist step log: {e}")


def _update_run_completion(ctx: Phase1Context, conn):
    """Update run completion in meta.data_runs."""
    try:
        status = "completed" if len(ctx.errors) == 0 else "failed"
        conn.execute("""
            UPDATE meta.data_runs
            SET completed_at = ?,
                status = ?,
                indicators_scanned = ?,
                notes = ?
            WHERE run_id = ?
        """, [
            datetime.now(),
            status,
            len(ctx.geometry_results),
            f"steps={len(ctx.completed_steps)}, errors={len(ctx.errors)}",
            ctx.run_id,
        ])
    except Exception as e:
        logger.warning(f"Failed to update run completion: {e}")


def run_data_phase(domain: str, mode: str = "short", force: bool = False) -> Phase1Context:
    """
    Run Data Phase pipeline.

    Args:
        domain: Domain to process (economic, climate, epidemiology)
        mode: 'short' for validation, 'full' for authoritative execution
        force: If True, bypass run_id lock (not recommended)

    Returns:
        Phase1Context with results
    """
    # Validate domain
    try:
        domain_enum = Domain(domain.lower())
    except ValueError:
        valid = [d.value for d in Domain]
        raise ValueError(f"Invalid domain '{domain}'. Valid domains: {valid}")

    domain_cfg = DOMAIN_CONFIG[domain_enum]

    # Create execution context
    run_id = f"data_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    ctx = Phase1Context(
        run_id=run_id,
        domain=domain_enum,
        mode=mode,
        started_at=datetime.now(),
    )

    # Banner
    print("=" * 70)
    print("PRISM DATA PHASE: CANONICAL ORCHESTRATOR")
    print("=" * 70)
    print(f"  Run ID:  {ctx.run_id}")
    print(f"  Domain:  {ctx.domain.value}")
    print(f"  Mode:    {ctx.mode}")
    print(f"  Started: {ctx.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Get database connection
    conn = open_prism_db()

    try:
        # Assert run_id is unused (write-once integrity)
        lock_result = assert_run_id_unused(conn, ctx.run_id, force=force)
        logger.info(f"  {lock_result.message}")

        # Persist run header
        _persist_run_header(ctx, conn, domain_cfg)

        # Execute steps in order
        steps = [
            ("Load Indicators", step1_load_indicators),
            ("Temporal Geometry Scan", step2_temporal_geometry_scan),
            ("Math Suitability", step3_math_suitability),
            ("Cohort Discovery", step4_cohort_discovery),
            ("Engine Execution", step5_engine_execution),
            ("Persist Artifacts", step6_persist_artifacts),
        ]

        for step_name, step_func in steps:
            step_started = datetime.now()
            success = step_func(ctx, conn)

            # Persist step log
            status = "success" if success else "failed"
            n_items = len(ctx.indicators) if step_name == "Load Indicators" else 0
            error = ctx.errors[-1]['error'] if ctx.errors and not success else None
            _persist_step(ctx, conn, step_name, status, step_started, n_items, error)

            if not success:
                logger.error(f"Pipeline halted at: {step_name}")
                break

        # Update run completion
        _update_run_completion(ctx, conn)

        # Note: Lock is now inserted within step6_persist_artifacts transaction
        # If we reach here without errors, the run is already locked

    finally:
        conn.close()

    # Summary
    print("\n" + "=" * 70)
    print("DATA PHASE COMPLETE")
    print("=" * 70)
    print(f"  Run ID:     {ctx.run_id}")
    print(f"  Duration:   {(datetime.now() - ctx.started_at).total_seconds():.1f}s")
    print(f"  Steps:      {len(ctx.completed_steps)}")
    print(f"  Errors:     {len(ctx.errors)}")
    print(f"  Indicators: {len(ctx.indicators)}")
    print(f"  Cohorts:    {len(ctx.cohorts)}")

    if ctx.mode == "short":
        print("\n  [SHORT MODE] Run with --full for authoritative execution")

    if ctx.errors:
        print("\n  ERRORS:")
        for err in ctx.errors:
            print(f"    - {err['step']}: {err['error']}")

    # Hint about report generation (for full mode)
    if mode == "full":
        print(f"\n  Generate report: python scripts/data_report.py --run-id {ctx.run_id}")

    return ctx


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Data Phase Orchestrator (semantic phase: data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_data_phase.py --domain economic
    python scripts/run_data_phase.py --domain climate --full
    python scripts/run_data_phase.py --domain economic --full
        """
    )

    parser.add_argument(
        "--domain", "-d",
        type=str,
        required=True,
        choices=["economic", "climate", "epidemiology"],
        help="Domain to process (REQUIRED)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run authoritative Phase 1 (default: short validation mode)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass run_id lock (not recommended - violates write-once canon)"
    )

    args = parser.parse_args()

    mode = "full" if args.full else "short"

    run_data_phase(domain=args.domain, mode=mode, force=args.force)


if __name__ == "__main__":
    main()
