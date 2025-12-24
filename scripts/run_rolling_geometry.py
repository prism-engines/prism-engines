#!/usr/bin/env python3
"""
PRISM Rolling Geometry Phase

Tracks how geometric structure evolves over time using rolling windows.
Detects regime changes, cluster births/deaths, and correlation shifts.

Output tables:
    structure.rolling_geometry     - Per-window geometry snapshots
    structure.cluster_evolution    - Cluster membership over time
    structure.regime_changes       - Detected structural shifts

Usage:
    python run_rolling_geometry.py --domain economic
    python run_rolling_geometry.py --domain seismology --full
    python run_rolling_geometry.py --domain economic --window 252 --step 21

Author: PRISM Project
Date: December 2024
"""

import sys
import os
import logging
import argparse
import uuid
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum

import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.db.open import open_prism_db
from prism.agents.agent_emergent_clusters import EmergentClusterDetector, ClusteringResult
from scripts.tools.prism_engine_gates import Domain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA
# =============================================================================

ROLLING_GEOMETRY_SCHEMA = """
-- Rolling geometry snapshots
CREATE TABLE IF NOT EXISTS structure.rolling_geometry (
    id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    n_indicators INTEGER NOT NULL,
    n_clusters INTEGER NOT NULL,
    n_singletons INTEGER NOT NULL,
    confidence DOUBLE,
    method_agreement DOUBLE,
    mean_correlation DOUBLE,
    min_correlation DOUBLE,
    max_correlation DOUBLE,
    correlation_spread DOUBLE,
    modularity_score DOUBLE,
    effective_dimension DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cluster membership evolution
CREATE TABLE IF NOT EXISTS structure.cluster_evolution (
    id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    window_end DATE NOT NULL,
    cluster_id INTEGER NOT NULL,
    indicator_id VARCHAR NOT NULL,
    internal_density DOUBLE,
    stability DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detected regime changes
CREATE TABLE IF NOT EXISTS structure.regime_changes (
    id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    change_date DATE NOT NULL,
    change_type VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    prev_value DOUBLE,
    new_value DOUBLE,
    magnitude DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_rolling_geometry_window
    ON structure.rolling_geometry(window_start, window_end);
CREATE INDEX IF NOT EXISTS idx_rolling_geometry_domain
    ON structure.rolling_geometry(domain);
CREATE INDEX IF NOT EXISTS idx_cluster_evolution_window
    ON structure.cluster_evolution(window_end);
CREATE INDEX IF NOT EXISTS idx_regime_changes_date
    ON structure.regime_changes(change_date);
"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WindowResult:
    """Result for a single time window."""
    window_start: datetime
    window_end: datetime
    n_indicators: int
    n_clusters: int
    cluster_members: Dict[int, List[str]]
    cluster_densities: Dict[int, float]
    cluster_stabilities: Dict[int, float]
    singletons: List[str]
    confidence: float
    method_agreement: float
    mean_correlation: float
    min_correlation: float
    max_correlation: float
    modularity_score: float


@dataclass
class RegimeChange:
    """A detected structural shift."""
    change_date: datetime
    change_type: str  # 'cluster_expansion', 'cluster_collapse', 'correlation_spike', 'correlation_drop'
    description: str
    prev_value: float
    new_value: float
    magnitude: float


@dataclass
class GeometryEvolution:
    """Complete rolling geometry analysis."""
    windows: List[WindowResult]
    regime_changes: List[RegimeChange]
    cluster_births: List[Tuple[datetime, List[str]]]
    cluster_deaths: List[Tuple[datetime, List[str]]]
    domain: str
    run_id: str


# =============================================================================
# DOMAIN CONFIGURATION
# =============================================================================

DOMAIN_CONFIG = {
    Domain.FINANCE: {
        'sources': ['tiingo', 'fred', 'yahoo'],
        'default_window': 252,  # 1 trading year
        'default_step': 21,     # 1 month
        'min_indicators': 5,
    },
    Domain.SEISMOLOGY: {
        'sources': ['usgs'],
        'default_window': 180,  # 6 months
        'default_step': 30,     # 1 month
        'min_indicators': 3,
    },
    Domain.CLIMATE: {
        'sources': ['noaa', 'nasa'],
        'default_window': 365,  # 1 year
        'default_step': 30,     # 1 month
        'min_indicators': 3,
    },
    Domain.EPIDEMIOLOGY: {
        'sources': ['delphi', 'owid', 'cdc', 'who'],
        'default_window': 90,   # 3 months
        'default_step': 7,      # 1 week
        'min_indicators': 3,
    },
}


# =============================================================================
# DATA ACCESS
# =============================================================================

def get_domain_indicators(
    conn: duckdb.DuckDBPyConnection,
    domain: Domain,
    max_indicators: Optional[int] = None
) -> List[str]:
    """Get indicators for a domain based on source."""
    config = DOMAIN_CONFIG.get(domain, DOMAIN_CONFIG[Domain.FINANCE])
    sources = config['sources']
    source_list = ','.join(f"'{s}'" for s in sources)

    query = f"""
    SELECT DISTINCT indicator_id
    FROM data.indicators
    WHERE source IN ({source_list})
      AND value IS NOT NULL
    ORDER BY indicator_id
    """

    if max_indicators:
        query += f" LIMIT {max_indicators}"

    result = conn.execute(query).fetchdf()
    return result['indicator_id'].tolist()


def get_indicator_data(
    conn: duckdb.DuckDBPyConnection,
    indicators: List[str],
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """Get indicator values for date range."""
    ind_list = ','.join(f"'{i}'" for i in indicators)

    query = f"""
    SELECT
        date,
        indicator_id,
        value
    FROM data.indicators
    WHERE indicator_id IN ({ind_list})
      AND date BETWEEN '{start_date}' AND '{end_date}'
      AND value IS NOT NULL
    ORDER BY date, indicator_id
    """

    return conn.execute(query).fetchdf()


def get_date_range(
    conn: duckdb.DuckDBPyConnection,
    indicators: List[str]
) -> Tuple[date, date]:
    """Get common date range for indicators."""
    ind_list = ','.join(f"'{i}'" for i in indicators)

    query = f"""
    SELECT
        MAX(min_date) as start_date,
        MIN(max_date) as end_date
    FROM (
        SELECT
            indicator_id,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM data.indicators
        WHERE indicator_id IN ({ind_list})
          AND value IS NOT NULL
        GROUP BY indicator_id
    )
    """

    result = conn.execute(query).fetchone()
    return result[0], result[1]


# =============================================================================
# CORRELATION COMPUTATION
# =============================================================================

def compute_correlation_matrix(
    conn: duckdb.DuckDBPyConnection,
    indicators: List[str],
    window_start: date,
    window_end: date,
    remove_pc1: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Compute correlation matrix for a window.
    Returns correlation matrix and stats dict.

    If remove_pc1=True, projects out PC1 before computing correlation.
    This removes the dominant market/systemic factor to reveal residual structure.
    """
    n = len(indicators)

    # Get data for window
    df = get_indicator_data(conn, indicators, window_start, window_end)

    if df.empty:
        return np.eye(n), {'mean': 0, 'min': 0, 'max': 0, 'spread': 0}

    # Pivot to wide format
    pivot = df.pivot(index='date', columns='indicator_id', values='value')

    # Compute returns (for price-like data) or use values directly
    # Check if values are price-like (positive, varying)
    is_price_like = (pivot > 0).all().all() and pivot.std().mean() > 0.01

    if is_price_like:
        # Use log returns
        returns = np.log(pivot / pivot.shift(1)).dropna()
    else:
        # Use values directly (counts, etc.)
        returns = pivot.dropna()

    if len(returns) < 10:
        return np.eye(n), {'mean': 0, 'min': 0, 'max': 0, 'spread': 0}

    # Only keep indicators we have data for
    available = [i for i in indicators if i in returns.columns]
    if len(available) < 2:
        return np.eye(n), {'mean': 0, 'min': 0, 'max': 0, 'spread': 0}

    # Optionally remove PC1 (dominant systemic factor)
    if remove_pc1 and len(available) >= 3:
        from sklearn.decomposition import PCA
        # Standardize
        data = returns[available].values
        data_std = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)
        # Fit PCA
        pca = PCA(n_components=min(len(available), len(data)))
        pca.fit(data_std)
        # Project out PC1: residual = data - (data . pc1) * pc1
        pc1 = pca.components_[0]  # First principal component
        projections = data_std @ pc1  # Project onto PC1
        pc1_contribution = np.outer(projections, pc1)  # Reconstruct PC1 component
        residuals = data_std - pc1_contribution  # Remove PC1
        # Replace returns with residuals for correlation
        returns = pd.DataFrame(residuals, columns=available, index=returns.index)

    # Compute correlation
    corr = returns[available].corr()

    # Build full matrix
    result = np.eye(n)
    id_to_idx = {ind: i for i, ind in enumerate(indicators)}

    for i, ind1 in enumerate(available):
        for j, ind2 in enumerate(available):
            idx1 = id_to_idx[ind1]
            idx2 = id_to_idx[ind2]
            val = corr.loc[ind1, ind2]
            if not np.isnan(val):
                result[idx1, idx2] = val

    # Compute stats
    upper_tri = result[np.triu_indices(n, k=1)]
    stats = {
        'mean': float(np.mean(upper_tri)),
        'min': float(np.min(upper_tri)),
        'max': float(np.max(upper_tri)),
        'spread': float(np.max(upper_tri) - np.min(upper_tri))
    }

    return result, stats


# =============================================================================
# ROLLING ANALYSIS
# =============================================================================

def analyze_window(
    conn: duckdb.DuckDBPyConnection,
    indicators: List[str],
    window_start: date,
    window_end: date,
    detector: EmergentClusterDetector,
    remove_pc1: bool = False
) -> WindowResult:
    """Analyze a single time window."""

    # Compute correlations
    corr_matrix, corr_stats = compute_correlation_matrix(
        conn, indicators, window_start, window_end, remove_pc1=remove_pc1
    )

    # Use absolute correlation as similarity for clustering
    similarity = np.abs(corr_matrix)

    # Detect clusters
    result = detector.detect_clusters(similarity, indicators)

    # Extract cluster info
    cluster_members = {}
    cluster_densities = {}
    cluster_stabilities = {}

    for cluster in result.clusters:
        cluster_members[cluster.cluster_id] = cluster.members
        cluster_densities[cluster.cluster_id] = cluster.internal_density
        cluster_stabilities[cluster.cluster_id] = cluster.stability

    # Convert dates to datetime for consistency
    ws = datetime.combine(window_start, datetime.min.time()) if isinstance(window_start, date) else window_start
    we = datetime.combine(window_end, datetime.min.time()) if isinstance(window_end, date) else window_end

    return WindowResult(
        window_start=ws,
        window_end=we,
        n_indicators=len(indicators),
        n_clusters=result.n_clusters,
        cluster_members=cluster_members,
        cluster_densities=cluster_densities,
        cluster_stabilities=cluster_stabilities,
        singletons=result.singletons,
        confidence=result.confidence,
        method_agreement=result.method_agreement,
        mean_correlation=corr_stats['mean'],
        min_correlation=corr_stats['min'],
        max_correlation=corr_stats['max'],
        modularity_score=result.modularity_score
    )


def detect_regime_changes(
    windows: List[WindowResult],
    cluster_threshold: int = 2,
    correlation_threshold: float = 0.15
) -> Tuple[List[RegimeChange], List[Tuple[datetime, List[str]]], List[Tuple[datetime, List[str]]]]:
    """Detect regime changes from window sequence."""

    regime_changes = []
    cluster_births = []
    cluster_deaths = []

    prev_clusters: Set[frozenset] = set()
    prev_n = 0
    prev_corr = 0.0

    for i, w in enumerate(windows):
        # Track cluster membership
        current_clusters = set()
        for members in w.cluster_members.values():
            current_clusters.add(frozenset(members))

        # Detect cluster births
        for cluster in current_clusters - prev_clusters:
            if len(cluster) > 1:
                cluster_births.append((w.window_end, list(cluster)))

        # Detect cluster deaths
        for cluster in prev_clusters - current_clusters:
            if len(cluster) > 1:
                cluster_deaths.append((w.window_end, list(cluster)))

        # Detect cluster count regime changes
        if i > 0 and abs(w.n_clusters - prev_n) >= cluster_threshold:
            direction = "expansion" if w.n_clusters > prev_n else "collapse"
            regime_changes.append(RegimeChange(
                change_date=w.window_end,
                change_type=f"cluster_{direction}",
                description=f"Cluster {direction}: {prev_n} -> {w.n_clusters}",
                prev_value=float(prev_n),
                new_value=float(w.n_clusters),
                magnitude=float(abs(w.n_clusters - prev_n))
            ))

        # Detect correlation regime changes
        if i > 0 and abs(w.mean_correlation - prev_corr) > correlation_threshold:
            direction = "spike" if w.mean_correlation > prev_corr else "drop"
            regime_changes.append(RegimeChange(
                change_date=w.window_end,
                change_type=f"correlation_{direction}",
                description=f"Correlation {direction}: {prev_corr:.2f} -> {w.mean_correlation:.2f}",
                prev_value=prev_corr,
                new_value=w.mean_correlation,
                magnitude=abs(w.mean_correlation - prev_corr)
            ))

        prev_clusters = current_clusters
        prev_n = w.n_clusters
        prev_corr = w.mean_correlation

    return regime_changes, cluster_births, cluster_deaths


def run_rolling_analysis(
    conn: duckdb.DuckDBPyConnection,
    domain: Domain,
    indicators: List[str],
    window_days: int,
    step_days: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    run_id: str = None,
    remove_pc1: bool = False
) -> GeometryEvolution:
    """Run complete rolling geometry analysis.

    If remove_pc1=True, projects out PC1 before correlation to reveal residual structure.
    """

    if run_id is None:
        run_id = f"rolling_{domain.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Get date range
    data_start, data_end = get_date_range(conn, indicators)

    if start_date is None:
        start_date = data_start
    if end_date is None:
        end_date = data_end

    logger.info(f"Rolling geometry analysis: {domain.value}")
    logger.info(f"  Indicators: {len(indicators)}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Window: {window_days} days, Step: {step_days} days")

    # Initialize detector
    detector = EmergentClusterDetector()

    # Generate windows
    windows = []
    current_end = start_date + timedelta(days=window_days)

    while current_end <= end_date:
        window_start = current_end - timedelta(days=window_days)

        try:
            result = analyze_window(
                conn, indicators, window_start, current_end, detector,
                remove_pc1=remove_pc1
            )
            windows.append(result)

            logger.info(
                f"  {current_end}: {result.n_clusters} clusters, "
                f"rho={result.mean_correlation:.3f}, conf={result.confidence:.1%}"
            )
        except Exception as e:
            logger.warning(f"  {current_end}: Failed - {e}")

        current_end += timedelta(days=step_days)

    if not windows:
        raise ValueError("No valid windows computed")

    # Detect regime changes
    regime_changes, cluster_births, cluster_deaths = detect_regime_changes(windows)

    logger.info(f"Analysis complete: {len(windows)} windows, {len(regime_changes)} regime changes")

    return GeometryEvolution(
        windows=windows,
        regime_changes=regime_changes,
        cluster_births=cluster_births,
        cluster_deaths=cluster_deaths,
        domain=domain.value,
        run_id=run_id
    )


# =============================================================================
# PERSISTENCE
# =============================================================================

def ensure_schema(conn: duckdb.DuckDBPyConnection):
    """Create schema if needed."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS structure")

    for statement in ROLLING_GEOMETRY_SCHEMA.split(';'):
        statement = statement.strip()
        if statement:
            try:
                conn.execute(statement)
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    logger.warning(f"Schema statement failed: {e}")


def persist_results(
    conn: duckdb.DuckDBPyConnection,
    evolution: GeometryEvolution
):
    """Persist rolling geometry results to database."""

    ensure_schema(conn)
    now = datetime.now()

    # Delete existing for this run
    conn.execute(
        "DELETE FROM structure.rolling_geometry WHERE run_id = ?",
        [evolution.run_id]
    )
    conn.execute(
        "DELETE FROM structure.cluster_evolution WHERE run_id = ?",
        [evolution.run_id]
    )
    conn.execute(
        "DELETE FROM structure.regime_changes WHERE run_id = ?",
        [evolution.run_id]
    )

    # Insert window results
    for w in evolution.windows:
        window_id = str(uuid.uuid4())

        conn.execute("""
            INSERT INTO structure.rolling_geometry
            (id, run_id, domain, window_start, window_end, n_indicators, n_clusters,
             n_singletons, confidence, method_agreement, mean_correlation,
             min_correlation, max_correlation, correlation_spread, modularity_score,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            window_id, evolution.run_id, evolution.domain,
            w.window_start.date(), w.window_end.date(),
            w.n_indicators, w.n_clusters, len(w.singletons),
            w.confidence, w.method_agreement, w.mean_correlation,
            w.min_correlation, w.max_correlation,
            w.max_correlation - w.min_correlation,
            w.modularity_score, now
        ])

        # Insert cluster memberships
        for cluster_id, members in w.cluster_members.items():
            density = w.cluster_densities.get(cluster_id, 0.0)
            stability = w.cluster_stabilities.get(cluster_id, 0.0)

            for indicator_id in members:
                conn.execute("""
                    INSERT INTO structure.cluster_evolution
                    (id, run_id, window_end, cluster_id, indicator_id,
                     internal_density, stability, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    str(uuid.uuid4()), evolution.run_id,
                    w.window_end.date(), cluster_id, indicator_id,
                    density, stability, now
                ])

    # Insert regime changes
    for rc in evolution.regime_changes:
        conn.execute("""
            INSERT INTO structure.regime_changes
            (id, run_id, change_date, change_type, description,
             prev_value, new_value, magnitude, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(uuid.uuid4()), evolution.run_id,
            rc.change_date.date(), rc.change_type, rc.description,
            rc.prev_value, rc.new_value, rc.magnitude, now
        ])

    logger.info(f"Persisted {len(evolution.windows)} windows to structure.rolling_geometry")
    logger.info(f"Persisted {len(evolution.regime_changes)} regime changes")


# =============================================================================
# REPORTING
# =============================================================================

def print_summary(evolution: GeometryEvolution):
    """Print analysis summary."""

    print()
    print("=" * 70)
    print(f"ROLLING GEOMETRY: {evolution.domain.upper()}")
    print("=" * 70)
    print()
    print(f"Windows analyzed: {len(evolution.windows)}")
    print(f"Time span: {evolution.windows[0].window_start.date()} to {evolution.windows[-1].window_end.date()}")
    print(f"Regime changes: {len(evolution.regime_changes)}")
    print()

    # Cluster count timeline
    print("CLUSTER COUNT OVER TIME:")
    for w in evolution.windows:
        bar = "█" * w.n_clusters + "░" * (5 - w.n_clusters)
        print(f"  {w.window_end.date()}: {bar} {w.n_clusters} clusters (ρ={w.mean_correlation:.2f})")

    # Regime changes
    if evolution.regime_changes:
        print()
        print("REGIME CHANGES:")
        for rc in evolution.regime_changes:
            print(f"  {rc.change_date.date()}: {rc.description}")

    # Cluster stability
    print()
    print("CLUSTER STABILITY:")

    # Track which indicators stay together
    cluster_history = {}
    for w in evolution.windows:
        for cluster_id, members in w.cluster_members.items():
            key = frozenset(members)
            if key not in cluster_history:
                cluster_history[key] = 0
            cluster_history[key] += 1

    # Most stable clusters
    stable_clusters = sorted(cluster_history.items(), key=lambda x: -x[1])[:5]
    for members, count in stable_clusters:
        pct = count / len(evolution.windows) * 100
        print(f"  {pct:.0f}% stable: {', '.join(sorted(members))}")

    # Correlation stats
    print()
    print("CORRELATION STATISTICS:")
    corrs = [w.mean_correlation for w in evolution.windows]
    print(f"  Mean: {np.mean(corrs):.3f}")
    print(f"  Min:  {np.min(corrs):.3f}")
    print(f"  Max:  {np.max(corrs):.3f}")
    print(f"  Std:  {np.std(corrs):.3f}")

    print()
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Rolling Geometry Phase'
    )
    parser.add_argument(
        '--domain', type=str, default='economic',
        choices=['economic', 'seismology', 'climate', 'epidemiology'],
        help='Data domain to analyze'
    )
    parser.add_argument(
        '--indicators', nargs='+',
        help='Specific indicators (default: all for domain)'
    )
    parser.add_argument(
        '--max-indicators', type=int,
        help='Limit number of indicators'
    )
    parser.add_argument(
        '--window', type=int,
        help='Window size in days (default: domain-specific)'
    )
    parser.add_argument(
        '--step', type=int,
        help='Step size in days (default: domain-specific)'
    )
    parser.add_argument(
        '--start', type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Full analysis (all indicators, fine step)'
    )
    parser.add_argument(
        '--skip-persist', action='store_true',
        help='Skip persisting to database'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output CSV file'
    )
    parser.add_argument(
        '--remove-pc1', action='store_true',
        help='Remove PC1 before correlation (reveals residual structure beyond dominant factor)'
    )

    args = parser.parse_args()

    # Map domain string to enum
    domain_map = {
        'economic': Domain.FINANCE,
        'seismology': Domain.SEISMOLOGY,
        'climate': Domain.CLIMATE,
        'epidemiology': Domain.EPIDEMIOLOGY
    }
    domain = domain_map[args.domain]
    config = DOMAIN_CONFIG[domain]

    # Determine parameters
    if args.full:
        logger.info("MODE: FULL ANALYSIS")
        max_indicators = args.max_indicators  # None = all
        window_days = args.window or config['default_window']
        step_days = args.step or 7  # Weekly for full
    else:
        logger.info("MODE: VALIDATION RUN (use --full for complete analysis)")
        max_indicators = args.max_indicators or 10
        window_days = args.window or config['default_window']
        step_days = args.step or config['default_step']

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date() if args.start else None
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else None

    with open_prism_db() as conn:
        # Get indicators
        if args.indicators:
            indicators = args.indicators
        else:
            indicators = get_domain_indicators(conn, domain, max_indicators)

        if len(indicators) < config['min_indicators']:
            logger.error(f"Need at least {config['min_indicators']} indicators, found {len(indicators)}")
            return 1

        logger.info(f"Indicators: {indicators[:10]}{'...' if len(indicators) > 10 else ''}")

        # Run analysis
        if args.remove_pc1:
            logger.info("MODE: REMOVING PC1 (residual correlations)")

        evolution = run_rolling_analysis(
            conn=conn,
            domain=domain,
            indicators=indicators,
            window_days=window_days,
            step_days=step_days,
            start_date=start_date,
            end_date=end_date,
            remove_pc1=args.remove_pc1
        )

        # Print summary
        print_summary(evolution)

        # Persist
        if not args.skip_persist:
            try:
                persist_results(conn, evolution)
            except Exception as e:
                logger.error(f"Failed to persist: {e}")

        # CSV output
        if args.output:
            rows = []
            for w in evolution.windows:
                rows.append({
                    'window_start': w.window_start.date(),
                    'window_end': w.window_end.date(),
                    'n_clusters': w.n_clusters,
                    'n_singletons': len(w.singletons),
                    'confidence': w.confidence,
                    'mean_correlation': w.mean_correlation,
                    'min_correlation': w.min_correlation,
                    'max_correlation': w.max_correlation,
                    'modularity': w.modularity_score
                })

            df = pd.DataFrame(rows)
            df.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
