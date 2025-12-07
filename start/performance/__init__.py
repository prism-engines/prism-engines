"""
PRISM Performance Diagnostics Suite
====================================

Isolated performance diagnostic scripts for the PRISM Engine.
These scripts are read-only and never modify the database.

Scripts:
- check_fetch_performance.py: Benchmark fetch speed and consistency
- check_db_performance.py: Measure DB read/write performance & schema health
- check_geometry_engine.py: Validate geometry input before coherence/lenses
- check_coherence_engine.py: Validate wavelet coherence numerics
- check_pca_stability.py: Check PCA eigenvector drift
- hidden_variation_sanity_check.py: Run HVD on family clusters
- check_lens_performance.py: Ensure each lens produces sane outputs
- check_runtime_costs.py: Measure runtime of key engine segments

Run all diagnostics:
    python start/performance/run_all_diagnostics.py
"""

__version__ = "1.0.0"
