# prism/clean/__init__.py
"""
PRISM Clean Layer (Hygiene Only)

This package is the canonical home for:
- Timestamp coercion/sorting
- Duplicate handling
- Numeric coercion
- Missingness policy application

NO transforms (no zscore/diff/log/rolling) are allowed here.
"""
from .cleaner import clean_series, CleanResult, align_to_daily, validate_clean_data
