"""
PRISM Semantic Guard

Runtime enforcement: PRISM core must not emit regime semantics.

PRISM outputs geometry, not regime labels. This guard enforces that constraint
at runtime by checking all outputs for forbidden interpretive tokens.

Usage:
    from prism.core import check_output_semantics, validate_hmm_output

    # After any engine/agent output
    check_output_semantics(output, context="my_engine")

    # Specifically for HMM
    validate_hmm_output(hmm_output)
"""

from typing import Any, Dict, List


class SemanticViolationError(ValueError):
    """Raised when forbidden semantics appear in PRISM core output."""
    pass


# =============================================================================
# Token Lists
# =============================================================================

# Forbidden tokens - these represent interpretive labels that should NOT appear
FORBIDDEN_TOKENS: List[str] = [
    # Regime semantics
    "regime", "regimes",

    # Market interpretations
    "bull", "bear",
    "crisis",
    "risk-on", "risk-off",

    # State interpretations
    "stress state", "calm state",
    "stressed", "normal state",

    # Economic labels
    "recession",
    "expansion",
    "bubble",
    "crash",
    "rally",
    "correction",
]


# Allowed tokens (for clarity) - these ARE permitted
ALLOWED_TOKENS: List[str] = [
    "state_id",           # numeric only
    "latent_state",       # as technical term
    "state_persistence",  # numeric metric
    "homeostatic",        # geometric concept
    "excursion",          # geometric concept
    "deformation",        # geometric concept
    "distance",           # geometric measurement
    "band",               # homeostatic band
    "trajectory",         # geometric path
]


# =============================================================================
# Validation Functions
# =============================================================================

def check_output_semantics(output: Any, context: str = "") -> None:
    """
    Hard-fail if forbidden semantics appear in output.

    Call after each agent/engine in pipeline.
    Raises ValueError if contaminated.

    Args:
        output: The output dict/object to check
        context: Context string for error message (e.g., engine name)

    Raises:
        SemanticViolationError: If forbidden token found in output
    """
    serialized = str(output).lower()

    for token in FORBIDDEN_TOKENS:
        if token in serialized:
            raise SemanticViolationError(
                f"SEMANTIC VIOLATION [{context}]: "
                f"Forbidden token '{token}' in output. "
                f"PRISM outputs geometry, not regime labels."
            )


def validate_hmm_output(output: Dict[str, Any]) -> None:
    """
    Ensure HMM outputs only unlabeled numeric state IDs.

    HMM should output:
        - state_id: integer (0, 1, 2, ...)
        - state_statistics: dict with numeric keys
        - transition_matrix: numeric array

    HMM should NOT output:
        - Labels like "bull", "bear", "calm", "stress"
        - String state identifiers like "hmm_state_bull"

    Args:
        output: HMM engine output dict

    Raises:
        SemanticViolationError: If semantic labels found
    """
    check_output_semantics(output, "HMM")

    if output is None:
        return

    # Additional check: state_id must be int, not string label
    if "state_id" in output:
        sid = output["state_id"]
        if isinstance(sid, str) and not sid.isdigit():
            raise SemanticViolationError(
                f"HMM state_id must be numeric, got: {sid}"
            )

    # Check state_statistics for semantic labels in keys
    state_stats = output.get("state_statistics", {})
    for key in state_stats.keys():
        key_lower = str(key).lower()
        for token in FORBIDDEN_TOKENS:
            if token in key_lower:
                raise SemanticViolationError(
                    f"SEMANTIC VIOLATION in HMM state_statistics: "
                    f"Key '{key}' contains forbidden token '{token}'. "
                    f"State keys must be numeric only (e.g., 'state_0')."
                )

    # Check for any labeled state_id values
    for key, val in state_stats.items():
        if isinstance(val, dict):
            state_id = val.get("state_id")
            if state_id is not None and not isinstance(state_id, (int, float)):
                state_str = str(state_id).lower()
                for token in FORBIDDEN_TOKENS:
                    if token in state_str:
                        raise SemanticViolationError(
                            f"SEMANTIC VIOLATION in HMM: "
                            f"state_id '{state_id}' contains semantic label. "
                            f"State IDs must be numeric only."
                        )


def validate_agent_output(output: Any, agent_name: str) -> None:
    """
    Validate any agent output for semantic violations.

    Args:
        output: Agent output to validate
        agent_name: Name of the agent for context

    Raises:
        SemanticViolationError: If semantic violations found
    """
    check_output_semantics(output, agent_name)


def validate_engine_output(output: Any, engine_name: str) -> None:
    """
    Validate any engine output for semantic violations.

    Args:
        output: Engine output to validate
        engine_name: Name of the engine for context

    Raises:
        SemanticViolationError: If semantic violations found
    """
    if engine_name.lower() == "hmm":
        validate_hmm_output(output)
    else:
        check_output_semantics(output, engine_name)


def is_allowed_token(token: str) -> bool:
    """Check if a token is in the allowed list."""
    return token.lower() in [t.lower() for t in ALLOWED_TOKENS]


def is_forbidden_token(token: str) -> bool:
    """Check if a token is in the forbidden list."""
    return token.lower() in [t.lower() for t in FORBIDDEN_TOKENS]
