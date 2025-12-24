"""
PRISM Cleaning Module

Cleans raw indicator data for engine consumption.

Usage:
    from prism.cleaning import clean_series, CleanResult
    
    result = clean_series(raw_df, frequency="daily")
    clean_df = result.data
    print(f"Method: {result.method}")
"""

from .cleaner import (
    clean_series,
    CleanResult,
    align_to_daily,
    validate_clean_data,
)

__all__ = [
    "clean_series",
    "CleanResult", 
    "align_to_daily",
    "validate_clean_data",
]
