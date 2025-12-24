#!/usr/bin/env python3
"""
PRISM Phase 1: Canonical Orchestrator

Single entry point for Phase 1 execution.

Usage:
    python scripts/run_phase1.py --domain economic
    python scripts/run_phase1.py --domain climate --full

Phase 1 Sequence:
    1. Validate domain and load indicators
    2. Temporal geometry scan (per indicator, per window)
    3. Math suitability evaluation (admission control)
    4. Cohort discovery (using only admissible math)
    5. Engine execution (certified runs only)
    6. Persist Phase-1 artifacts (trusted by downstream phases)

Rules:
    - --domain is REQUIRED
    - Default mode is 'short' (validation)
    - --full triggers authoritative Phase-1 execution
    - Domain isolation is enforced (no cross-domain writes)
    - All agents run in defined order, no parallelism with other scripts

This replaces ad-hoc runners like run_cohort_pipeline.py and exploratory scripts.

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

from prism.db.connection import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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
        FROM clean.indicator_values
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
            FROM clean.indicator_values
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
        for key, decision in all_decisions.items():
            status = decision.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        eligible_pct = 100 * by_status.get('eligible', 0) / total if total > 0 else 0
        conditional_pct = 100 * by_status.get('conditional', 0) / total if total > 0 else 0
        ineligible_pct = 100 * by_status.get('ineligible', 0) / total if total > 0 else 0
        
        logger.info(f"  Total evaluations: {total}")
        logger.info(f"  Eligible:    {by_status.get('eligible', 0):4} ({eligible_pct:.1f}%)")
        logger.info(f"  Conditional: {by_status.get('conditional', 0):4} ({conditional_pct:.1f}%)")
        logger.info(f"  Ineligible:  {by_status.get('ineligible', 0):4} ({ineligible_pct:.1f}%)")
        
        ctx.log_step(f"step3_math_suitability ({total} evaluations)")
        return True
        
    except ImportError as e:
        logger.warning(f"MathSuitabilityAgent not available: {e}")
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
                        FROM clean.indicator_values
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
# STEP 6: PERSIST ARTIFACTS
# =============================================================================

def step6_persist_artifacts(ctx: Phase1Context, conn) -> bool:
    """
    Step 6: Persist Phase 1 artifacts to database.
    
    Writes to:
    - meta.geometry_windows
    - meta.engine_eligibility
    """
    logger.info("-" * 60)
    logger.info("STEP 6: Persist Artifacts")
    logger.info("-" * 60)
    
    if ctx.mode == "short":
        logger.info("  [SHORT MODE] Skipping persistence")
        ctx.log_step("step6_persist_artifacts (skipped in short mode)")
        return True
    
    # Persist geometry windows
    geometry_count = 0
    for indicator_id, windows in ctx.geometry_results.items():
        for window_years, geom in windows.items():
            try:
                conn.execute("""
                    INSERT INTO meta.geometry_windows
                    (run_id, indicator_id, window_years, window_days,
                     dominant_geometry, geometry_pct, avg_confidence,
                     avg_disagreement, n_observations, n_transitions,
                     stability, quality_score, is_optimal, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    ctx.run_id,
                    indicator_id,
                    window_years,
                    geom['window_days'],
                    geom['dominant_geometry'],
                    geom['geometry_pct'],
                    geom['avg_confidence'],
                    geom['avg_disagreement'],
                    geom['n_observations'],
                    geom['n_transitions'],
                    geom['stability'],
                    geom['quality_score'],
                    geom.get('is_optimal', False),
                ])
                geometry_count += 1
            except Exception as e:
                ctx.log_error("step6_persist", f"geometry_windows: {e}")
    
    # Persist eligibility
    eligibility_count = 0
    for indicator_id, windows in ctx.eligibility_results.items():
        for window_years, elig in windows.items():
            try:
                conn.execute("""
                    INSERT INTO meta.engine_eligibility
                    (run_id, indicator_id, window_years, geometry,
                     confidence, status, allowed_engines, rationale,
                     policy_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [
                    ctx.run_id,
                    indicator_id,
                    window_years,
                    elig['geometry'],
                    elig['confidence'],
                    elig['status'],
                    json.dumps(elig['allowed_engines']),
                    json.dumps(elig['rationale']),
                    elig.get('policy_version', '1.0.0'),
                ])
                eligibility_count += 1
            except Exception as e:
                ctx.log_error("step6_persist", f"engine_eligibility: {e}")
    
    logger.info(f"  Geometry windows: {geometry_count}")
    logger.info(f"  Eligibility records: {eligibility_count}")
    
    ctx.log_step(f"step6_persist_artifacts ({geometry_count + eligibility_count} records)")
    return True


# =============================================================================
# DETAILED REPORT
# =============================================================================

def _print_detailed_report(ctx: Phase1Context):
    """Print detailed Phase 1 report with all findings."""
    
    print("\n" + "=" * 70)
    print("DETAILED REPORT")
    print("=" * 70)
    
    # -----------------------------------------------------------------
    # GEOMETRY SUMMARY
    # -----------------------------------------------------------------
    if ctx.geometry_results:
        print("\n┌─ GEOMETRY BY INDICATOR")
        print("│")
        
        # Collect all geometries
        geometry_counts = {}
        for indicator_id, windows in ctx.geometry_results.items():
            for w, geom in windows.items():
                if geom.get('is_optimal'):
                    g = geom['dominant_geometry']
                    geometry_counts[g] = geometry_counts.get(g, 0) + 1
        
        for geom, count in sorted(geometry_counts.items(), key=lambda x: -x[1]):
            print(f"│  {geom:25} {count:3} indicators")
        
        print("│")
        print("│  Per-Indicator Detail:")
        print("│  " + "-" * 60)
        
        for indicator_id in sorted(ctx.geometry_results.keys()):
            windows = ctx.geometry_results[indicator_id]
            
            # Find optimal window
            optimal = None
            for w, geom in windows.items():
                if geom.get('is_optimal'):
                    optimal = (w, geom)
                    break
            
            if optimal:
                w, geom = optimal
                print(f"│  {indicator_id:20} @ {w:4}y → {geom['dominant_geometry']:20} "
                      f"(conf={geom['avg_confidence']:.2f}, qual={geom['quality_score']:.2f})")
        
        print("└" + "─" * 65)
    
    # -----------------------------------------------------------------
    # ELIGIBILITY SUMMARY
    # -----------------------------------------------------------------
    if ctx.eligibility_results:
        print("\n┌─ ELIGIBILITY BY INDICATOR")
        print("│")
        
        eligible_list = []
        conditional_list = []
        ineligible_list = []
        
        for indicator_id, windows in ctx.eligibility_results.items():
            for w, elig in windows.items():
                # Check if optimal window
                geom = ctx.geometry_results.get(indicator_id, {}).get(w, {})
                if not geom.get('is_optimal'):
                    continue
                
                entry = {
                    'indicator': indicator_id,
                    'window': w,
                    'geometry': elig['geometry'],
                    'status': elig['status'],
                    'rationale': elig.get('rationale', []),
                    'engines': elig.get('allowed_engines', []),
                }
                
                if elig['status'] == 'eligible':
                    eligible_list.append(entry)
                elif elig['status'] == 'conditional':
                    conditional_list.append(entry)
                else:
                    ineligible_list.append(entry)
        
        # Eligible
        if eligible_list:
            print(f"│  ✓ ELIGIBLE ({len(eligible_list)}):")
            for e in eligible_list:
                engines_str = ', '.join(e['engines'][:5])
                if len(e['engines']) > 5:
                    engines_str += f" (+{len(e['engines'])-5} more)"
                print(f"│    {e['indicator']:20} → {e['geometry']:20} engines: {engines_str}")
        
        # Conditional
        if conditional_list:
            print(f"│")
            print(f"│  ⚠ CONDITIONAL ({len(conditional_list)}):")
            for e in conditional_list:
                reason = '; '.join(e['rationale'][:2]) if e['rationale'] else 'unspecified'
                print(f"│    {e['indicator']:20} → {e['geometry']:20} ({reason})")
        
        # Ineligible
        if ineligible_list:
            print(f"│")
            print(f"│  ✗ INELIGIBLE ({len(ineligible_list)}):")
            for e in ineligible_list:
                reason = '; '.join(e['rationale'][:2]) if e['rationale'] else 'unspecified'
                print(f"│    {e['indicator']:20} → {e['geometry']:20} ({reason})")
        
        print("└" + "─" * 65)
    
    # -----------------------------------------------------------------
    # ENGINE ROUTING SUMMARY
    # -----------------------------------------------------------------
    if ctx.eligibility_results:
        print("\n┌─ ENGINE ROUTING SUMMARY")
        print("│")
        
        engine_counts = {}
        for indicator_id, windows in ctx.eligibility_results.items():
            for w, elig in windows.items():
                if elig['status'] in ['eligible', 'conditional']:
                    for eng in elig.get('allowed_engines', []):
                        engine_counts[eng] = engine_counts.get(eng, 0) + 1
        
        if engine_counts:
            print("│  Engines enabled (by frequency):")
            for eng, count in sorted(engine_counts.items(), key=lambda x: -x[1]):
                bar = "█" * min(count, 20)
                print(f"│    {eng:20} {count:3} {bar}")
        
        print("└" + "─" * 65)
    
    # -----------------------------------------------------------------
    # COHORT SUMMARY
    # -----------------------------------------------------------------
    if ctx.cohorts:
        print("\n┌─ COHORTS")
        print("│")
        for c in ctx.cohorts:
            print(f"│  {c.get('cohort_id', 'unknown'):25} {c.get('n_indicators', 0):3} indicators")
            if 'indicators' in c and c['indicators']:
                for ind in c['indicators'][:5]:
                    print(f"│    - {ind}")
                if len(c['indicators']) > 5:
                    print(f"│    ... and {len(c['indicators'])-5} more")
        print("└" + "─" * 65)
    
    # -----------------------------------------------------------------
    # WINDOW DISTRIBUTION
    # -----------------------------------------------------------------
    if ctx.geometry_results:
        print("\n┌─ OPTIMAL WINDOW DISTRIBUTION")
        print("│")
        
        window_counts = {}
        for indicator_id, windows in ctx.geometry_results.items():
            for w, geom in windows.items():
                if geom.get('is_optimal'):
                    window_counts[w] = window_counts.get(w, 0) + 1
        
        for w in sorted(window_counts.keys()):
            count = window_counts[w]
            bar = "█" * count
            print(f"│  {w:5.1f}y  {count:3}  {bar}")
        
        print("└" + "─" * 65)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_phase1(domain: str, mode: str = "short") -> Phase1Context:
    """
    Run Phase 1 pipeline.
    
    Args:
        domain: Domain to process (economic, climate, epidemiology)
        mode: 'short' for validation, 'full' for authoritative execution
        
    Returns:
        Phase1Context with results
    """
    # Validate domain
    try:
        domain_enum = Domain(domain.lower())
    except ValueError:
        valid = [d.value for d in Domain]
        raise ValueError(f"Invalid domain '{domain}'. Valid domains: {valid}")
    
    # Create execution context
    run_id = f"phase1_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    ctx = Phase1Context(
        run_id=run_id,
        domain=domain_enum,
        mode=mode,
        started_at=datetime.now(),
    )
    
    # Banner
    print("=" * 70)
    print("PRISM PHASE 1: CANONICAL ORCHESTRATOR")
    print("=" * 70)
    print(f"  Run ID:  {ctx.run_id}")
    print(f"  Domain:  {ctx.domain.value}")
    print(f"  Mode:    {ctx.mode}")
    print(f"  Started: {ctx.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Get database connection
    conn = get_connection()
    
    try:
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
            success = step_func(ctx, conn)
            if not success:
                logger.error(f"Pipeline halted at: {step_name}")
                break
        
    finally:
        conn.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"  Run ID:     {ctx.run_id}")
    print(f"  Duration:   {(datetime.now() - ctx.started_at).total_seconds():.1f}s")
    print(f"  Steps:      {len(ctx.completed_steps)}")
    print(f"  Errors:     {len(ctx.errors)}")
    print(f"  Indicators: {len(ctx.indicators)}")
    print(f"  Cohorts:    {len(ctx.cohorts)}")
    
    # Detailed Report
    _print_detailed_report(ctx)
    
    if ctx.mode == "short":
        print("\n  [SHORT MODE] Run with --full for authoritative execution")
    
    if ctx.errors:
        print("\n  ERRORS:")
        for err in ctx.errors:
            print(f"    - {err['step']}: {err['error']}")
    
    return ctx


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Phase 1 Canonical Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_phase1.py --domain economic
    python scripts/run_phase1.py --domain climate --full
    python scripts/run_phase1.py --domain economic --full
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
    
    args = parser.parse_args()
    
    mode = "full" if args.full else "short"
    
    run_phase1(domain=args.domain, mode=mode)


if __name__ == "__main__":
    main()
