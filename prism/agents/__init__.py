"""
PRISM Agent Layer

Two types of agents:
1. Domain Agents (data acquisition): ClimateAgent, EpidemiologyAgent, SeismologyAgent, etc.
2. Analysis Agents (decision making): DataQuality, Routing, Stability, etc.

Implements Engine Admission Control Spec with:
- Null Calibration (Spec Section B)
- Stability Scoring (Spec Section C)
- Weight Capping (Spec Section D)
- Consensus Reporting (Spec Section H)
"""

# Data acquisition agents
from .base import DomainAgent, IndicatorMeta, FetchResult
# from .orchestrator import Orchestrator  # TODO: Create orchestrator.py

# Analysis/decision agents - Foundation
from .agent_foundation import (
    # Audit
    AuditEntry,
    # Base class
    BaseAgent,
    # Spec Section B: Null Calibration
    NullCalibrationResult,
    # Spec Section C: Stability
    StabilityScore,
    # Spec Section D: Engine Contribution
    EngineContribution,
    # Spec Section H: Consensus Report
    ConsensusReport,
    # Agent Results
    DataQualityResult,
    RoutingResult,
    StabilityResult,
    BlindSpotResult,
    PipelineResult,
    # Infrastructure
    DiagnosticRegistry,
    AgentOrchestrator,
    load_agent_config,
    get_default_config,
)
from .agent_data_quality import DataQualityAuditAgent
from .agent_routing import GeometryDiagnosticAgent, EngineRoutingAgent
from .agent_stability import EngineStabilityAuditAgent
from .agent_blindspot import BlindSpotDetectionAgent
from .pv4_global_forcing_agent import PV4GlobalForcingAgent, PV4Result
from .agent_math_suitability import (
    WindowSuitabilityAgent,
    WindowGeometryResult,
    WindowEligibility,
    WindowSuitabilityPolicy,
    EligibilityStatus,
)
# from .indicator_contribution import (  # TODO: File missing
#     DecisionHint,
#     IndicatorContribution,
#     ContributionReport,
#     IndicatorContributionAgent,
# )

__all__ = [
    # Data acquisition
    "DomainAgent",
    "IndicatorMeta",
    "FetchResult",
    # "Orchestrator",  # TODO: Create orchestrator.py
    # Spec dataclasses
    "NullCalibrationResult",
    "StabilityScore",
    "EngineContribution",
    "ConsensusReport",
    # Analysis/decision
    "AuditEntry",
    "BaseAgent",
    "DataQualityResult",
    "RoutingResult",
    "StabilityResult",
    "BlindSpotResult",
    "PipelineResult",
    "DiagnosticRegistry",
    "AgentOrchestrator",
    "load_agent_config",
    "get_default_config",
    "DataQualityAuditAgent",
    "GeometryDiagnosticAgent",
    "EngineRoutingAgent",
    "EngineStabilityAuditAgent",
    "BlindSpotDetectionAgent",
    # PV4 Global Forcing
    "PV4GlobalForcingAgent",
    "PV4Result",
    # Math Suitability
    "WindowSuitabilityAgent",
    "WindowGeometryResult",
    "WindowEligibility",
    "WindowSuitabilityPolicy",
    "EligibilityStatus",
    # Indicator Contribution - TODO: File missing
    # "DecisionHint",
    # "IndicatorContribution",
    # "ContributionReport",
    # "IndicatorContributionAgent",
]
