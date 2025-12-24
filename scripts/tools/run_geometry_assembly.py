#!/usr/bin/env python3
"""
CLI runner for Geometry Assembly Agent

Usage:
    python scripts/run_geometry_assembly.py --run-id <run_id>
    python scripts/run_geometry_assembly.py --run-id <run_id> --dry-run
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prism.agents.geometry_assembly_agent import main

if __name__ == "__main__":
    main()
