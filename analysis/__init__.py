"""
Analysis module for PRISM Engine.

This module contains advisory and diagnostic tools:
- Hidden Variation Detector (HVD): Detects redundant indicators and divergences
- Phase 8: Stationarity, uncertainty quantification, multiple testing
- Phase 9: Surrogate data, null models, sensitivity analysis
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

# Phase 8: Statistical Foundation
from .stationarity import StationarityTester
from .uncertainty import UncertaintyEstimator
from .multiple_testing import bonferroni, holm, benjamini_hochberg, benjamini_yekutieli
from .cointegration import CointegrationAnalyzer
from .power_analysis import PowerAnalyzer

# Phase 9: Null Models & Significance
from .surrogates import SurrogateGenerator, SurrogateTest
from .null_models import NullModelGenerator
from .sensitivity import ParameterSensitivity, SpecificationCurve

__all__ = [
    # HVD
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
    # Phase 8
    'StationarityTester',
    'UncertaintyEstimator',
    'bonferroni',
    'holm',
    'benjamini_hochberg',
    'benjamini_yekutieli',
    'CointegrationAnalyzer',
    'PowerAnalyzer',
    # Phase 9
    'SurrogateGenerator',
    'SurrogateTest',
    'NullModelGenerator',
    'ParameterSensitivity',
    'SpecificationCurve',
]
