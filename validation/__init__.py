"""
PRISM Engine - Scientific Validation

Validation framework for lens accuracy and reliability.
"""

from .synthetic_data_generator import SyntheticDataGenerator
from .lens_validator import LensValidator
from .permutation_tests import PermutationTester, run_permutation_tests
from .bootstrap_analysis import BootstrapAnalyzer, run_bootstrap_analysis
from .backtester import Backtester
from .cross_lens_validator import CrossLensValidator
from .validation_report import ValidationReport, run_full_validation

__all__ = [
    'SyntheticDataGenerator',
    'LensValidator',
    'PermutationTester',
    'run_permutation_tests',
    'BootstrapAnalyzer',
    'run_bootstrap_analysis',
    'Backtester',
    'CrossLensValidator',
    'ValidationReport',
    'run_full_validation',
]
