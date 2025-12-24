#!/usr/bin/env python3
"""
Agent Audit Script - Run this to check and fix all agents in prism/agents/

Usage: python scripts/audit_agents.py
"""

import os
from pathlib import Path

AGENTS_DIR = Path("prism/agents")

# Expected agents
EXPECTED_AGENTS = [
    "agent_ablation.py",
    "agent_blindspot.py",
    "agent_cohort_discovery.py",
    "agent_data_quality.py",
    "agent_foundation.py",
    "agent_geometry_signature.py",
    "agent_multiview_geometry.py",
    "agent_routing.py",
    "agent_stability.py",
]

# Common import fixes needed
IMPORT_FIXES = {
    # Old import -> New import
    "from scripts.": "from prism.agents.",
    "from .agent_foundation": "from prism.agents.agent_foundation",
}

def audit_agents():
    print("=" * 60)
    print("PRISM Agent Audit")
    print("=" * 60)

    # Check agents exist
    print("\n1. Checking agent files exist:")
    for agent in EXPECTED_AGENTS:
        path = AGENTS_DIR / agent
        status = "✓" if path.exists() else "✗ MISSING"
        print(f"   {status} {agent}")

    # Check __init__.py exports
    print("\n2. Checking __init__.py:")
    init_path = AGENTS_DIR / "__init__.py"
    if init_path.exists():
        content = init_path.read_text()
        print(f"   ✓ exists ({len(content)} bytes)")
    else:
        print("   ✗ MISSING - creating...")
        create_init()

    # Scan for import issues
    print("\n3. Scanning for import issues:")
    issues = []
    for agent_file in AGENTS_DIR.glob("agent_*.py"):
        content = agent_file.read_text()
        for old, new in IMPORT_FIXES.items():
            if old in content:
                issues.append((agent_file.name, old, new))
                print(f"   ✗ {agent_file.name}: found '{old}'")

    if not issues:
        print("   ✓ No import issues found")

    # Check for Domain enum consistency
    print("\n4. Checking Domain enum:")
    routing_path = AGENTS_DIR / "agent_routing.py"
    if routing_path.exists():
        content = routing_path.read_text()
        if "class Domain" in content:
            print("   ✓ Domain enum exists in agent_routing.py")
        else:
            print("   ✗ Domain enum missing")

    print("\n" + "=" * 60)
    print("Audit complete")
    print("=" * 60)

def create_init():
    """Create proper __init__.py for agents package."""
    init_content = '''"""
PRISM Agents Package

All agents follow the naming convention: agent_<name>.py
"""

from prism.agents.agent_foundation import BaseAgent
from prism.agents.agent_geometry_signature import GeometrySignatureAgent
from prism.agents.agent_multiview_geometry import MultiViewGeometryAgent, ViewType
from prism.agents.agent_data_quality import DataQualityAgent
from prism.agents.agent_cohort_discovery import CohortDiscoveryAgent
from prism.agents.agent_routing import RoutingAgent
from prism.agents.agent_stability import StabilityAgent
from prism.agents.agent_blindspot import BlindspotAgent

__all__ = [
    "BaseAgent",
    "GeometrySignatureAgent",
    "MultiViewGeometryAgent",
    "ViewType",
    "DataQualityAgent",
    "CohortDiscoveryAgent",
    "RoutingAgent",
    "StabilityAgent",
    "BlindspotAgent",
]
'''
    init_path = AGENTS_DIR / "__init__.py"
    init_path.write_text(init_content)
    print("   ✓ Created __init__.py")

if __name__ == "__main__":
    audit_agents()
