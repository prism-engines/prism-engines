#!/usr/bin/env python3
"""
PRISM AI Interpretation CLI
============================

Command-line interface for generating AI interpretations of PRISM outputs.

Usage:
    python scripts/run_interpret.py window 42
    python scripts/run_interpret.py event 15
    python scripts/run_interpret.py regime 41 42
    python scripts/run_interpret.py indicator SPY
    python scripts/run_interpret.py ask "What drove the 2024 regime break?"
    python scripts/run_interpret.py compare finance climate

Options:
    --backend       AI backend (manual, claude, openai, ollama)
    --model         Model to use (e.g., claude-sonnet-4-20250514)
    --db            Path to temporal database
    --save          Save interpretation to database
    --no-save       Don't save interpretation
    --output        Output file path (optional)
    --prompt-only   Only show the prompt, don't call AI

Examples:
    # Generate window interpretation (manual mode)
    python scripts/run_interpret.py window 42

    # Use Claude API
    python scripts/run_interpret.py window 42 --backend claude

    # Save only the prompt to a file
    python scripts/run_interpret.py window 42 --prompt-only --output prompt.txt

    # Compare domains
    python scripts/run_interpret.py compare finance climate

    # Ask a question
    python scripts/run_interpret.py ask "Which indicators were most stable?"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from interpretation import PRISMInterpreter, FeedbackManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PRISM AI Interpretation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "command",
        choices=["window", "event", "regime", "indicator", "ask", "compare", "list", "feedback", "stats"],
        help="Interpretation command to run",
    )

    parser.add_argument(
        "args",
        nargs="*",
        help="Arguments for the command (e.g., window_id, indicator_name, question)",
    )

    parser.add_argument(
        "--backend",
        default="manual",
        choices=["manual", "claude", "openai", "ollama"],
        help="AI backend to use (default: manual)",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (defaults based on backend)",
    )

    parser.add_argument(
        "--db",
        default=None,
        help="Path to temporal database",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save interpretation to database (default: True)",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save interpretation to database",
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (optional)",
    )

    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Only show the prompt, don't call AI backend",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    return parser.parse_args()


def cmd_window(interpreter: PRISMInterpreter, args: list, prompt_only: bool = False) -> str:
    """Interpret a window."""
    if not args:
        return "Error: window_id required"

    try:
        window_id = int(args[0])
    except ValueError:
        return f"Error: Invalid window_id: {args[0]}"

    result = interpreter.interpret_window(window_id)

    if prompt_only:
        return f"=== PROMPT ===\n\n{result.prompt_used}"

    return result.interpretation


def cmd_event(interpreter: PRISMInterpreter, args: list, prompt_only: bool = False) -> str:
    """Interpret an event."""
    if not args:
        return "Error: event_id required"

    try:
        event_id = int(args[0])
    except ValueError:
        return f"Error: Invalid event_id: {args[0]}"

    result = interpreter.interpret_event(event_id)

    if prompt_only:
        return f"=== PROMPT ===\n\n{result.prompt_used}"

    return result.interpretation


def cmd_regime(interpreter: PRISMInterpreter, args: list, prompt_only: bool = False) -> str:
    """Interpret a regime break."""
    if len(args) < 2:
        return "Error: Two window IDs required (window_before_id, window_after_id)"

    try:
        window_before = int(args[0])
        window_after = int(args[1])
    except ValueError:
        return f"Error: Invalid window IDs: {args[0]}, {args[1]}"

    result = interpreter.interpret_regime_break(window_before, window_after)

    if prompt_only:
        return f"=== PROMPT ===\n\n{result.prompt_used}"

    return result.interpretation


def cmd_indicator(interpreter: PRISMInterpreter, args: list, prompt_only: bool = False) -> str:
    """Interpret an indicator."""
    if not args:
        return "Error: indicator_name required"

    indicator_name = args[0]
    result = interpreter.interpret_indicator(indicator_name)

    if prompt_only:
        return f"=== PROMPT ===\n\n{result.prompt_used}"

    return result.interpretation


def cmd_ask(interpreter: PRISMInterpreter, args: list, prompt_only: bool = False) -> str:
    """Answer a question."""
    if not args:
        return "Error: question required"

    question = " ".join(args)
    result = interpreter.answer_question(question)

    if prompt_only:
        return f"=== PROMPT ===\n\n{result.prompt_used}"

    return result.interpretation


def cmd_compare(interpreter: PRISMInterpreter, args: list, prompt_only: bool = False) -> str:
    """Compare domains."""
    if len(args) < 2:
        return "Error: At least two system names required"

    result = interpreter.compare_domains(args)

    if prompt_only:
        return f"=== PROMPT ===\n\n{result.prompt_used}"

    return result.interpretation


def cmd_list(interpreter: PRISMInterpreter, args: list, prompt_only: bool = False) -> str:
    """List interpretations."""
    interp_type = args[0] if args else None
    interpretations = interpreter.list_interpretations(
        interpretation_type=interp_type,
        limit=20,
    )

    if not interpretations:
        return "No interpretations found"

    lines = ["ID  | Type      | Target | Backend | Validated | Created"]
    lines.append("-" * 70)

    for i in interpretations:
        validated = "Yes" if i['validated'] else "No"
        lines.append(
            f"{i['id']:<4} | {i['interpretation_type']:<9} | "
            f"{i.get('target_id', 'N/A'):<6} | {i['backend']:<7} | "
            f"{validated:<9} | {i['created_at'][:16]}"
        )

    return "\n".join(lines)


def cmd_feedback(interpreter: PRISMInterpreter, args: list, db_path: Path) -> str:
    """Record feedback on an interpretation."""
    if len(args) < 2:
        return "Error: interpretation_id and feedback_type required"

    try:
        interp_id = int(args[0])
    except ValueError:
        return f"Error: Invalid interpretation_id: {args[0]}"

    feedback_type = args[1]
    notes = " ".join(args[2:]) if len(args) > 2 else None

    feedback_mgr = FeedbackManager(db_path=db_path)
    success = feedback_mgr.record_feedback(
        interpretation_id=interp_id,
        feedback_type=feedback_type,
        notes=notes,
    )

    if success:
        return f"Feedback recorded for interpretation {interp_id}"
    return "Failed to record feedback"


def cmd_stats(interpreter: PRISMInterpreter, args: list, db_path: Path) -> str:
    """Show feedback statistics."""
    feedback_mgr = FeedbackManager(db_path=db_path)
    stats = feedback_mgr.get_feedback_stats()

    if "error" in stats:
        return f"Error: {stats['error']}"

    lines = [
        "=== Interpretation Statistics ===",
        f"Total interpretations: {stats.get('total_interpretations', 0)}",
        f"Validated: {stats.get('validated_count', 0)}",
        f"Rejected: {stats.get('rejected_count', 0)}",
        f"Pending: {stats.get('pending_count', 0)}",
        "",
        "By type:",
    ]

    for type_name, type_stats in stats.get('by_type', {}).items():
        lines.append(
            f"  {type_name}: {type_stats['total']} total, "
            f"{type_stats['validated']} validated"
        )

    if 'validated_patterns' in stats:
        lines.append(f"\nValidated patterns: {stats['validated_patterns']}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()

    # Determine save setting
    save_interpretations = not args.no_save and args.save

    # Set up database path
    db_path = Path(args.db) if args.db else None

    # Initialize interpreter
    try:
        interpreter = PRISMInterpreter(
            backend=args.backend if not args.prompt_only else "manual",
            model=args.model,
            temporal_db_path=db_path,
            save_interpretations=save_interpretations and not args.prompt_only,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure the temporal database exists. Run temporal analysis first.")
        sys.exit(1)

    # Route to command handler
    try:
        if args.command == "window":
            output = cmd_window(interpreter, args.args, args.prompt_only)
        elif args.command == "event":
            output = cmd_event(interpreter, args.args, args.prompt_only)
        elif args.command == "regime":
            output = cmd_regime(interpreter, args.args, args.prompt_only)
        elif args.command == "indicator":
            output = cmd_indicator(interpreter, args.args, args.prompt_only)
        elif args.command == "ask":
            output = cmd_ask(interpreter, args.args, args.prompt_only)
        elif args.command == "compare":
            output = cmd_compare(interpreter, args.args, args.prompt_only)
        elif args.command == "list":
            output = cmd_list(interpreter, args.args, args.prompt_only)
        elif args.command == "feedback":
            output = cmd_feedback(interpreter, args.args, db_path)
        elif args.command == "stats":
            output = cmd_stats(interpreter, args.args, db_path)
        else:
            output = f"Unknown command: {args.command}"

    except ValueError as e:
        output = f"Error: {e}"
    except Exception as e:
        output = f"Error: {e}"
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()

    # Output result
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
        print(f"Output saved to: {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
