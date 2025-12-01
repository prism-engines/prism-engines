"""
PRISM Engine
============

Pattern Recognition through Integrated Signal Methods

Usage:
    python run.py           # From project root
    
Modules:
    consolidate - Data loading and weighting
    temporal    - Time window analysis
    coherence   - Coherence Index calculation
    report      - Output generation
"""

from .consolidate import load_and_consolidate, build_consolidated_view
from .temporal import run_temporal_analysis, LENSES
from .coherence import calculate_coherence, interpret_coherence
from .report import generate_summary, generate_html_report

__version__ = "2.0.0"
__all__ = [
    "load_and_consolidate",
    "build_consolidated_view", 
    "run_temporal_analysis",
    "LENSES",
    "calculate_coherence",
    "interpret_coherence",
    "generate_summary",
    "generate_html_report"
]
