"""
PRISM - Progressive Structural Inference through Geometric Measurement

A framework for analyzing time-linear data within emergent geometric structures.

Architecture:
    - fetch: Data acquisition (raw data → DuckDB)
    - clean: Data cleaning (raw → cleaned)  [future]
    - normalize: Data normalization (cleaned → normalized)  [future]
    - engines: Mathematical methods (PCA, Granger, Hurst, etc.)  [future]
    - phases: Processing stages (Unbound → Structure → Bounded)  [future]
    
Quick Start:
    from prism.fetch import FetchRunner
    from prism.registry import RegistryLoader
    
    # Load indicators from registry
    registry = RegistryLoader()
    indicators = registry.to_fetch_list()
    
    # Fetch data
    runner = FetchRunner()
    summary = runner.run(indicators)
    summary.print_summary()
"""

__version__ = "0.1.0"
