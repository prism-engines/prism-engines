"""
Analysis module for PRISM Engine.

This module contains advisory and diagnostic tools:
- Hidden Variation Detector (HVD): Detects redundant indicators and divergences
"""

from .hidden_variation_detector import (
    build_hvd_report,
    print_hvd_summary,
    compute_similarity_matrix,
    detect_high_similarity_pairs,
    detect_clusters_kmeans,
    detect_clusters_correlation,
    detect_family_divergences,
    SimilarityResult,
    ClusterResult,
    FamilyDivergence,
    HVDReport,
)

__all__ = [
    'build_hvd_report',
    'print_hvd_summary',
    'compute_similarity_matrix',
    'detect_high_similarity_pairs',
    'detect_clusters_kmeans',
    'detect_clusters_correlation',
    'detect_family_divergences',
    'SimilarityResult',
    'ClusterResult',
    'FamilyDivergence',
    'HVDReport',
]
