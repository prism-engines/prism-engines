"""
Seismometer - ML-based Regime Instability Detection

Detects structural instability (coherence decay, divergence acceleration)
BEFORE regime breaks occur. Early warning via geometric deformation,
not crash prediction.

Usage:
    from engine_core.seismometer import Seismometer

    # Initialize and fit
    seis = Seismometer(conn=db_connection)
    seis.fit(end='2019-12-31')  # Baseline on pre-COVID period

    # Get current status
    status = seis.get_current_status()
    print(f"Stability: {status['stability_index']:.2f} ({status['alert_level']})")

    # Plot history
    seis.plot_stability_history(save_path='stability.png')

Alert Levels (stability_index):
    - 0.90-1.00: stable - Normal market behavior
    - 0.70-0.90: elevated - Increased noise
    - 0.50-0.70: pre_instability - Structural stress emerging
    - 0.30-0.50: divergence - Active structural breakdown
    - 0.00-0.30: high_risk - Imminent instability
"""

from .ensemble import Seismometer
from .clustering import ClusteringDriftDetector
from .autoencoder import ReconstructionErrorDetector
from .correlation_graph import CorrelationGraphDetector
from .base import BaseDetector

__all__ = [
    'Seismometer',
    'ClusteringDriftDetector',
    'ReconstructionErrorDetector',
    'CorrelationGraphDetector',
    'BaseDetector',
]

__version__ = '0.1.0'
