#!/usr/bin/env python3
"""
PRISM Lens Geometry Analysis
============================
Analyze relationships between lenses - which agree, which are orthogonal,
which provide unique information.

This is meta-analysis: studying the lenses themselves, not the indicators.

Usage:
    python lens_geometry.py                    # Analyze most recent run
    python lens_geometry.py --run 5            # Analyze specific run
    python lens_geometry.py --compare-runs     # Compare lens behavior across runs
    python lens_geometry.py --save             # Save visualizations
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'start' else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# LENS WEIGHTING
# =============================================================================

def compute_cluster_weights(clusters: Dict[str, int]) -> Dict[str, float]:
    """
    Weight lenses by cluster membership.
    
    If 4 lenses cluster together, each gets weight 1/4.
    This ensures each *perspective* contributes equally, not each *method*.
    """
    # Count lenses per cluster
    cluster_sizes = {}
    for lens, cluster_id in clusters.items():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    
    # Weight = 1 / cluster_size
    weights = {}
    for lens, cluster_id in clusters.items():
        weights[lens] = 1.0 / cluster_sizes[cluster_id]
    
    return weights


def compute_independence_weights(corr: pd.DataFrame) -> Dict[str, float]:
    """
    Weight lenses by independence from others.
    
    Weight = 1 / (1 + mean_abs_correlation_with_others)
    
    Orthogonal lenses (low correlation) get high weight.
    Redundant lenses (high correlation) get low weight.
    """
    weights = {}
    lenses = corr.columns.tolist()
    
    for lens in lenses:
        # Mean absolute correlation with all OTHER lenses
        other_corrs = [abs(corr.loc[lens, other]) for other in lenses if other != lens]
        mean_corr = np.mean(other_corrs) if other_corrs else 0
        
        # Weight inversely proportional to correlation
        weights[lens] = 1.0 / (1.0 + mean_corr)
    
    return weights


def compute_uniqueness_weights(unique: Dict[str, List[str]], n_indicators: int) -> Dict[str, float]:
    """
    Weight lenses by unique contribution.
    
    Lenses that find things others miss get higher weight.
    """
    # Base weight for all
    weights = {lens: 0.5 for lens in unique.keys()}
    
    # Bonus for unique findings
    max_unique = max(len(v) for v in unique.values()) if unique else 1
    
    for lens, unique_indicators in unique.items():
        if unique_indicators:
            # Bonus proportional to unique findings
            bonus = 0.5 * (len(unique_indicators) / max(max_unique, 1))
            weights[lens] += bonus
    
    return weights


def compute_combined_weights(
    clusters: Dict[str, int],
    corr: pd.DataFrame,
    unique: Dict[str, List[str]],
    n_indicators: int,
    method: str = 'independence'
) -> Dict[str, float]:
    """
    Compute lens weights using specified method.
    
    Methods:
        'cluster': Weight by cluster membership
        'independence': Weight by orthogonality to other lenses
        'uniqueness': Weight by unique contributions
        'combined': Average of all three methods
    """
    if method == 'cluster':
        return compute_cluster_weights(clusters)
    elif method == 'independence':
        return compute_independence_weights(corr)
    elif method == 'uniqueness':
        return compute_uniqueness_weights(unique, n_indicators)
    elif method == 'combined':
        w1 = compute_cluster_weights(clusters)
        w2 = compute_independence_weights(corr)
        w3 = compute_uniqueness_weights(unique, n_indicators)
        
        # Normalize each to sum to 1, then average
        def normalize(w):
            total = sum(w.values())
            return {k: v/total for k, v in w.items()}
        
        w1, w2, w3 = normalize(w1), normalize(w2), normalize(w3)
        
        combined = {}
        for lens in w1.keys():
            combined[lens] = (w1[lens] + w2[lens] + w3[lens]) / 3
        
        # Renormalize
        total = sum(combined.values())
        return {k: v/total * len(combined) for k, v in combined.items()}
    else:
        raise ValueError(f"Unknown weighting method: {method}")


def print_lens_weights(weights: Dict[str, float], method: str):
    """Print lens weights."""
    print(f"\n⚖️  LENS WEIGHTS ({method})")
    print("=" * 70)
    print("   Higher weight = more independent/unique contribution")
    print("-" * 70)
    
    # Sort by weight descending
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    total = sum(weights.values())
    for lens, weight in sorted_weights:
        pct = weight / total * 100
        bar = "█" * int(pct / 2)
        print(f"   {lens:<18} {weight:.3f}  ({pct:5.1f}%) {bar}")


def save_lens_weights(run_id: int, weights: Dict[str, float], method: str) -> None:
    """Save lens weights to database."""
    import json
    from data.sql.db_connector import get_connection
    
    conn = get_connection()
    
    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lens_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            method TEXT,
            weights TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, method)
        )
    """)
    
    # Convert numpy types to Python native types for JSON
    clean_weights = {k: float(v) for k, v in weights.items()}
    
    conn.execute(
        "INSERT OR REPLACE INTO lens_weights (run_id, method, weights) VALUES (?, ?, ?)",
        (run_id, method, json.dumps(clean_weights))
    )
    
    conn.commit()
    conn.close()


# =============================================================================
# DATA LOADING
# =============================================================================

def get_lens_rankings_matrix(run_id: int) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load lens rankings as a matrix: indicators × lenses.
    
    Returns:
        DataFrame with indicators as rows, lenses as columns, scores as values
        List of indicator names
        List of lens names
    """
    from data.sql.db_connector import get_connection
    
    conn = get_connection()
    
    df = pd.read_sql(
        """
        SELECT indicator, lens_name, score
        FROM indicator_rankings
        WHERE run_id = ?
        """,
        conn,
        params=(run_id,)
    )
    conn.close()
    
    if df.empty:
        return pd.DataFrame(), [], []
    
    # Pivot to matrix form
    matrix = df.pivot(index='indicator', columns='lens_name', values='score')
    matrix = matrix.fillna(0)
    
    return matrix, list(matrix.index), list(matrix.columns)


def get_latest_run_id() -> Optional[int]:
    """Get the most recent run ID."""
    from data.sql.db_connector import get_connection
    
    conn = get_connection()
    row = conn.execute(
        "SELECT run_id FROM analysis_runs ORDER BY run_date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    
    return row[0] if row else None


# =============================================================================
# LENS CORRELATION ANALYSIS
# =============================================================================

def compute_lens_correlation(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation between lenses based on their indicator rankings.
    
    High correlation = lenses agree on what's important
    Low correlation = lenses see different things
    Negative correlation = lenses disagree
    """
    # Each column is a lens, each row is an indicator
    # Correlation between columns
    return matrix.corr(method='spearman')  # Rank correlation


def compute_lens_agreement(matrix: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Compute Jaccard similarity of top-K indicators between lenses.
    
    More interpretable than correlation for rankings.
    """
    lenses = matrix.columns.tolist()
    n_lenses = len(lenses)
    
    # Get top-K indicators for each lens
    top_sets = {}
    for lens in lenses:
        top_indicators = matrix[lens].nlargest(top_k).index.tolist()
        top_sets[lens] = set(top_indicators)
    
    # Compute pairwise Jaccard
    agreement = pd.DataFrame(index=lenses, columns=lenses, dtype=float)
    
    for i, l1 in enumerate(lenses):
        for j, l2 in enumerate(lenses):
            if i == j:
                agreement.loc[l1, l2] = 1.0
            else:
                intersection = len(top_sets[l1] & top_sets[l2])
                union = len(top_sets[l1] | top_sets[l2])
                agreement.loc[l1, l2] = intersection / union if union > 0 else 0
    
    return agreement


def compute_unique_contribution(matrix: pd.DataFrame, top_k: int = 10) -> Dict[str, List[str]]:
    """
    Find indicators that ONLY appear in one lens's top-K.
    These are the unique insights each lens provides.
    """
    lenses = matrix.columns.tolist()
    
    # Get top-K for each lens
    top_sets = {}
    for lens in lenses:
        top_sets[lens] = set(matrix[lens].nlargest(top_k).index.tolist())
    
    # All top indicators across all lenses
    all_top = set()
    for s in top_sets.values():
        all_top |= s
    
    # Find unique per lens
    unique = {}
    for lens in lenses:
        # Indicators in this lens's top-K but no other lens's top-K
        others = set()
        for other_lens, other_set in top_sets.items():
            if other_lens != lens:
                others |= other_set
        
        unique[lens] = list(top_sets[lens] - others)
    
    return unique


# =============================================================================
# LENS CLUSTERING / EMBEDDING
# =============================================================================

def compute_lens_embedding(matrix: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
    """
    Embed lenses in 2D space using PCA on their ranking vectors.
    
    Lenses that cluster together produce similar rankings.
    
    Returns:
        DataFrame with PC1, PC2 coordinates
        List of variance explained ratios
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Transpose: lenses as rows, indicators as features
    lens_vectors = matrix.T
    
    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(lens_vectors)
    
    # PCA to 2D
    n_components = min(2, len(lens_vectors))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)
    
    result = pd.DataFrame(
        coords,
        index=lens_vectors.index,
        columns=['PC1', 'PC2'] if coords.shape[1] >= 2 else ['PC1']
    )
    
    variance_explained = list(pca.explained_variance_ratio_)
    
    return result, variance_explained


def cluster_lenses(matrix: pd.DataFrame, n_clusters: int = 4) -> Dict[str, int]:
    """
    Cluster lenses by their ranking similarity.
    
    Returns dict of lens -> cluster_id
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    
    # Transpose: lenses as rows
    lens_vectors = matrix.T
    
    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(lens_vectors)
    
    # Cluster
    n_clusters = min(n_clusters, len(lens_vectors))
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(scaled)
    
    return dict(zip(lens_vectors.index, labels))


def compute_weighted_consensus(run_id: int, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Recompute consensus rankings using lens weights.
    
    Instead of simple average, weight each lens's contribution
    by its independence/uniqueness.
    """
    from data.sql.db_connector import get_connection
    
    conn = get_connection()
    
    # Get all rankings for this run
    df = pd.read_sql(
        """
        SELECT indicator, lens_name, score
        FROM indicator_rankings
        WHERE run_id = ?
        """,
        conn,
        params=(run_id,)
    )
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Pivot to matrix
    matrix = df.pivot(index='indicator', columns='lens_name', values='score')
    
    # Normalize each lens's scores to [0, 1]
    for col in matrix.columns:
        s = matrix[col]
        if s.max() != s.min():
            matrix[col] = (s - s.min()) / (s.max() - s.min())
        else:
            matrix[col] = 0.5
    
    # Apply weights
    weighted_scores = pd.Series(0.0, index=matrix.index)
    total_weight = 0.0
    
    for lens in matrix.columns:
        if lens in weights:
            w = weights[lens]
            weighted_scores += matrix[lens].fillna(0) * w
            total_weight += w
    
    if total_weight > 0:
        weighted_scores /= total_weight
    
    # Build result
    result = pd.DataFrame({
        'weighted_score': weighted_scores,
        'n_lenses': matrix.notna().sum(axis=1),
    })
    result = result.sort_values('weighted_score', ascending=False)
    result['rank'] = range(1, len(result) + 1)
    
    return result


def print_weighted_vs_unweighted(run_id: int, weights: Dict[str, float]):
    """Compare weighted vs unweighted consensus."""
    from data.sql.db_connector import get_connection
    
    # Get unweighted consensus
    conn = get_connection()
    unweighted = pd.read_sql(
        """
        SELECT indicator, mean_score, rank
        FROM consensus_rankings
        WHERE run_id = ?
        ORDER BY rank
        """,
        conn,
        params=(run_id,)
    )
    conn.close()
    
    # Compute weighted
    weighted = compute_weighted_consensus(run_id, weights)
    
    if unweighted.empty or weighted.empty:
        print("   No data to compare")
        return
    
    print("\n📊 WEIGHTED vs UNWEIGHTED CONSENSUS")
    print("=" * 70)
    print(f"   {'Rank':<6} {'Unweighted':<20} {'Weighted':<20} {'Change':<10}")
    print("-" * 70)
    
    # Compare top 15
    for i in range(min(15, len(unweighted))):
        uw_indicator = unweighted.iloc[i]['indicator']
        uw_rank = int(unweighted.iloc[i]['rank'])
        
        w_indicator = weighted.index[i]
        w_rank = i + 1
        
        # Find where unweighted indicator lands in weighted
        if uw_indicator in weighted.index:
            new_rank = weighted.loc[uw_indicator, 'rank']
            change = uw_rank - new_rank
            change_str = f"+{int(change)}" if change > 0 else str(int(change)) if change < 0 else "="
        else:
            change_str = "?"
        
        marker = "←" if uw_indicator != w_indicator else ""
        print(f"   {i+1:<6} {uw_indicator:<20} {w_indicator:<20} {change_str:<10} {marker}")


# =============================================================================
# LENS CATEGORIES (PRIOR KNOWLEDGE)
# =============================================================================

LENS_CATEGORIES = {
    'causality': ['granger', 'transfer_entropy', 'influence'],
    'structure': ['pca', 'clustering', 'network', 'tda'],
    'dynamics': ['regime', 'wavelet', 'decomposition', 'dmd'],
    'information': ['mutual_info', 'transfer_entropy'],
    'outliers': ['anomaly', 'magnitude'],
}

def get_lens_category(lens_name: str) -> str:
    """Get the theoretical category for a lens."""
    for category, lenses in LENS_CATEGORIES.items():
        if lens_name in lenses:
            return category
    return 'other'


# =============================================================================
# REPORTING
# =============================================================================

def print_correlation_matrix(corr: pd.DataFrame):
    """Print correlation matrix with highlighting."""
    print("\n📊 LENS CORRELATION MATRIX (Spearman)")
    print("=" * 70)
    print("   High correlation = lenses agree on rankings")
    print("   Low/negative = lenses see different structure")
    print("-" * 70)
    
    # Print header
    lenses = corr.columns.tolist()
    header = "         " + " ".join(f"{l[:6]:>7}" for l in lenses)
    print(header)
    
    # Print rows
    for lens in lenses:
        row = f"{lens[:8]:<8}"
        for other in lenses:
            val = corr.loc[lens, other]
            if lens == other:
                row += "    -  "
            elif val > 0.7:
                row += f"  {val:>5.2f}*"  # High agreement
            elif val < 0.3:
                row += f"  {val:>5.2f}°"  # Low agreement (interesting!)
            else:
                row += f"  {val:>5.2f} "
        print(row)
    
    print("\n   * = high agreement (>0.7)   ° = low agreement (<0.3)")


def print_agreement_matrix(agreement: pd.DataFrame, top_k: int):
    """Print Jaccard agreement on top-K."""
    print(f"\n🎯 TOP-{top_k} AGREEMENT (Jaccard Similarity)")
    print("=" * 70)
    print(f"   1.0 = identical top-{top_k}   0.0 = no overlap")
    print("-" * 70)
    
    lenses = agreement.columns.tolist()
    header = "         " + " ".join(f"{l[:6]:>7}" for l in lenses)
    print(header)
    
    for lens in lenses:
        row = f"{lens[:8]:<8}"
        for other in lenses:
            val = agreement.loc[lens, other]
            if lens == other:
                row += "    -  "
            elif val > 0.5:
                row += f"  {val:>5.2f}*"
            elif val < 0.2:
                row += f"  {val:>5.2f}°"
            else:
                row += f"  {val:>5.2f} "
        print(row)


def print_unique_contributions(unique: Dict[str, List[str]]):
    """Print what each lens uniquely detects."""
    print("\n🔍 UNIQUE CONTRIBUTIONS")
    print("=" * 70)
    print("   Indicators that ONLY appear in this lens's top-10:")
    print("-" * 70)
    
    for lens, indicators in sorted(unique.items()):
        if indicators:
            print(f"   {lens:<18} → {', '.join(indicators)}")
        else:
            print(f"   {lens:<18} → (no unique findings)")


def print_lens_clusters(clusters: Dict[str, int], embedding: pd.DataFrame):
    """Print lens clustering results."""
    print("\n🔬 LENS CLUSTERS")
    print("=" * 70)
    print("   Lenses that produce similar rankings cluster together:")
    print("-" * 70)
    
    # Group by cluster
    cluster_groups = {}
    for lens, cluster_id in clusters.items():
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(lens)
    
    for cluster_id in sorted(cluster_groups.keys()):
        lenses = cluster_groups[cluster_id]
        categories = [get_lens_category(l) for l in lenses]
        print(f"\n   Cluster {cluster_id}: {', '.join(lenses)}")
        print(f"            Categories: {', '.join(set(categories))}")


def print_embedding(embedding: pd.DataFrame, variance_explained: List[float]):
    """Print 2D embedding coordinates."""
    print("\n📍 LENS EMBEDDING (2D)")
    print("=" * 70)
    
    if len(variance_explained) >= 2:
        print(f"   PC1 explains {variance_explained[0]:.1%}, PC2 explains {variance_explained[1]:.1%}")
    elif len(variance_explained) == 1:
        print(f"   PC1 explains {variance_explained[0]:.1%}")
    
    print("-" * 70)
    print(f"   {'Lens':<18} {'PC1':>8} {'PC2':>8}  Category")
    print("-" * 70)
    
    for lens in embedding.index:
        pc1 = embedding.loc[lens, 'PC1']
        pc2 = embedding.loc[lens, 'PC2'] if 'PC2' in embedding.columns else 0
        cat = get_lens_category(lens)
        print(f"   {lens:<18} {pc1:>8.3f} {pc2:>8.3f}  {cat}")


def print_redundancy_analysis(corr: pd.DataFrame):
    """Identify potentially redundant lenses."""
    print("\n⚠️  REDUNDANCY ANALYSIS")
    print("=" * 70)
    print("   Lens pairs with correlation > 0.8 may be redundant:")
    print("-" * 70)
    
    found = False
    lenses = corr.columns.tolist()
    for i, l1 in enumerate(lenses):
        for l2 in lenses[i+1:]:
            val = corr.loc[l1, l2]
            if val > 0.8:
                found = True
                print(f"   {l1} ↔ {l2}: {val:.3f}")
    
    if not found:
        print("   No highly redundant pairs found - good lens diversity!")


def print_orthogonal_pairs(corr: pd.DataFrame):
    """Identify orthogonal lens pairs (provide independent info)."""
    print("\n✨ ORTHOGONAL PAIRS (Independent Information)")
    print("=" * 70)
    print("   Lens pairs with correlation < 0.2 see different things:")
    print("-" * 70)
    
    found = False
    lenses = corr.columns.tolist()
    for i, l1 in enumerate(lenses):
        for l2 in lenses[i+1:]:
            val = corr.loc[l1, l2]
            if val < 0.2:
                found = True
                cat1, cat2 = get_lens_category(l1), get_lens_category(l2)
                print(f"   {l1} ({cat1}) ↔ {l2} ({cat2}): {val:.3f}")
    
    if not found:
        print("   All lenses show some correlation - consider adding diverse methods")


# =============================================================================
# DATABASE STORAGE
# =============================================================================

def save_lens_geometry(
    run_id: int,
    correlation: pd.DataFrame,
    agreement: pd.DataFrame,
    embedding: pd.DataFrame,
    clusters: Dict[str, int],
    unique: Dict[str, List[str]]
) -> None:
    """Save lens geometry analysis to database."""
    import json
    from data.sql.db_connector import get_connection
    
    conn = get_connection()
    
    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lens_geometry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            correlation_matrix TEXT,
            agreement_matrix TEXT,
            embedding TEXT,
            clusters TEXT,
            unique_contributions TEXT,
            UNIQUE(run_id)
        )
    """)
    
    # Convert numpy types to Python native for JSON
    clean_clusters = {k: int(v) for k, v in clusters.items()}
    
    # Insert/replace
    conn.execute(
        """
        INSERT OR REPLACE INTO lens_geometry 
            (run_id, correlation_matrix, agreement_matrix, embedding, clusters, unique_contributions)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            correlation.to_json(),
            agreement.to_json(),
            embedding.to_json(),
            json.dumps(clean_clusters),
            json.dumps(unique)
        )
    )
    
    conn.commit()
    conn.close()


# =============================================================================
# VISUALIZATION (Optional - requires matplotlib)
# =============================================================================

def plot_lens_geometry(
    embedding: pd.DataFrame,
    clusters: Dict[str, int],
    output_path: Optional[Path] = None
):
    """Create 2D visualization of lens space."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib not available for plotting)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by cluster
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
    
    for lens in embedding.index:
        x = embedding.loc[lens, 'PC1']
        y = embedding.loc[lens, 'PC2'] if 'PC2' in embedding.columns else 0
        cluster = clusters.get(lens, 0)
        color = colors[cluster % len(colors)]
        
        ax.scatter(x, y, c=color, s=100, alpha=0.7)
        ax.annotate(lens, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PRISM Lens Geometry\n(Lenses that cluster together produce similar rankings)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"   📊 Saved plot: {output_path}")
    else:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def analyze_run(run_id: int, save: bool = False, plot: bool = False):
    """Run complete lens geometry analysis for a single run."""
    print(f"\n{'=' * 70}")
    print(f"🔬 LENS GEOMETRY ANALYSIS - Run {run_id}")
    print(f"{'=' * 70}")
    
    # Load data
    matrix, indicators, lenses = get_lens_rankings_matrix(run_id)
    
    if matrix.empty:
        print(f"   No ranking data found for run {run_id}")
        return
    
    print(f"\n   Loaded: {len(indicators)} indicators × {len(lenses)} lenses")
    
    # Compute metrics
    correlation = compute_lens_correlation(matrix)
    agreement = compute_lens_agreement(matrix, top_k=10)
    unique = compute_unique_contribution(matrix, top_k=10)
    embedding, variance_explained = compute_lens_embedding(matrix)
    clusters = cluster_lenses(matrix)
    
    # Compute weights
    weights_independence = compute_independence_weights(correlation)
    weights_cluster = compute_cluster_weights(clusters)
    weights_combined = compute_combined_weights(clusters, correlation, unique, len(indicators), 'combined')
    
    # Print reports
    print_correlation_matrix(correlation)
    print_agreement_matrix(agreement, top_k=10)
    print_unique_contributions(unique)
    print_lens_clusters(clusters, embedding)
    print_embedding(embedding, variance_explained)
    print_redundancy_analysis(correlation)
    print_orthogonal_pairs(correlation)
    
    # Print weights
    print_lens_weights(weights_independence, 'independence')
    print_lens_weights(weights_cluster, 'cluster')
    print_lens_weights(weights_combined, 'combined')
    
    # Compare weighted vs unweighted consensus
    print_weighted_vs_unweighted(run_id, weights_combined)
    
    # Save to DB
    if save:
        save_lens_geometry(run_id, correlation, agreement, embedding, clusters, unique)
        save_lens_weights(run_id, weights_independence, 'independence')
        save_lens_weights(run_id, weights_cluster, 'cluster')
        save_lens_weights(run_id, weights_combined, 'combined')
        print(f"\n   💾 Saved to database")
    
    # Plot
    if plot:
        output_path = SCRIPT_DIR / 'output' / f'lens_geometry_run{run_id}.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_lens_geometry(embedding, clusters, output_path)
    
    # Return weights for use by other scripts
    return {
        'independence': weights_independence,
        'cluster': weights_cluster,
        'combined': weights_combined,
    }


def main():
    parser = argparse.ArgumentParser(description='PRISM Lens Geometry Analysis')
    parser.add_argument('--run', '-r', type=int, help='Analyze specific run ID')
    parser.add_argument('--save', '-s', action='store_true', help='Save analysis to database')
    parser.add_argument('--plot', '-p', action='store_true', help='Generate visualization')
    parser.add_argument('--compare-runs', action='store_true', help='Compare lens behavior across runs')
    
    args = parser.parse_args()
    
    # Get run ID
    if args.run:
        run_id = args.run
    else:
        run_id = get_latest_run_id()
        if not run_id:
            print("No analysis runs found. Run analyze.py first.")
            return 1
    
    if args.compare_runs:
        print("Cross-run comparison not yet implemented")
        # TODO: Compare lens correlations across multiple runs
        return 0
    
    analyze_run(run_id, save=args.save, plot=args.plot)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
