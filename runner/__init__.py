"""
PRISM Runner Package
=====================

Unified entry point for PRISM analysis with HTML interface.

Usage:
    # From command line
    python -m runner                    # Launch HTML UI
    python -m runner --cli              # CLI mode
    python -m runner --workflow regime  # Direct run

    # From code
    from runner import run_server, Dispatcher

    run_server()  # Launch HTML UI

    dispatcher = Dispatcher()
    results = dispatcher.run({"workflow": "regime_comparison"})
"""

from .dispatcher import Dispatcher, DispatchError

__all__ = [
    "Dispatcher",
    "DispatchError",
]
