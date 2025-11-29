"""
PRISM Engine - Scientific Validation

Validation framework for lens accuracy and reliability.
"""

from .synthetic_data_generator import SyntheticDataGenerator
from .lens_validator import LensValidator

__all__ = [
    'SyntheticDataGenerator',
    'LensValidator',
]
