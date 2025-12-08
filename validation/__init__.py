"""
PRISM Engine - Scientific Validation

Validation framework for lens accuracy and reliability.

Modules:
- Core validation: synthetic data, lens validation, permutation tests
- Phase 10: Historical stress tests, regime accuracy, tail analysis
- Phase 12: Cross-domain coherence testing
"""

from .synthetic_data_generator import SyntheticDataGenerator
from .lens_validator import LensValidator
from .permutation_tests import PermutationTester, run_permutation_tests
from .bootstrap_analysis import BootstrapAnalyzer, run_bootstrap_analysis
from .backtester import Backtester
from .cross_lens_validator import CrossLensValidator
from .validation_report import ValidationReport, run_full_validation

# Phase 10: Historical Validation
from .stress_tests import StressTestRunner, HistoricalEventRegistry
from .regime_accuracy import RegimeAccuracyAnalyzer
from .tail_analysis import TailAnalyzer
from .temporal_validation import TemporalValidator

# Phase 12: Cross-Domain Testing
from .cross_domain import CrossDomainAnalyzer, Domain, CrossDomainReport

__all__ = [
    # Core
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
    # Phase 10
    'StressTestRunner',
    'HistoricalEventRegistry',
    'RegimeAccuracyAnalyzer',
    'TailAnalyzer',
    'TemporalValidator',
    # Phase 12
    'CrossDomainAnalyzer',
    'Domain',
    'CrossDomainReport',
]
