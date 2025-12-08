"""
PRISM Engine Plugins
=====================

Drop-in directory for engine plugins.
Place Python files here to add new analysis engines.

Phase 7 Engines:
- HurstEngine: Long-term memory analysis via Hurst exponent
- SpectralCoherenceEngine: Frequency-domain coherence analysis
"""

# Import Phase 7 engines for easy access
try:
    from .hurst_engine import HurstEngine, compute_hurst
except ImportError:
    HurstEngine = None
    compute_hurst = None

try:
    from .spectral_coherence_engine import SpectralCoherenceEngine, compute_coherence
except ImportError:
    SpectralCoherenceEngine = None
    compute_coherence = None

__all__ = [
    'HurstEngine',
    'SpectralCoherenceEngine',
    'compute_hurst',
    'compute_coherence',
]
