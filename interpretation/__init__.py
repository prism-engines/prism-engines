"""
PRISM AI Interpretation Layer
==============================

Bridges mathematical outputs and human-interpretable insights.

This module provides AI-powered interpretation of PRISM analysis results,
including:
- Window summaries and regime analysis
- Coherence event interpretation
- Cross-domain pattern recognition
- Natural language querying
- Human feedback loop for continuous improvement

Design Principles:
------------------
1. **Math as Ground Truth**: AI interpretations always reference underlying
   quantitative data. Never generate interpretations without supporting evidence.

2. **Human in the Loop**: AI generates hypotheses; humans validate.
   Validated interpretations become confirmed patterns.

3. **Domain Agnostic**: Same interpretation framework works across
   finance, climate, chemistry, and other domains.

4. **Reproducible**: All interpretations logged with input data,
   prompts, models, and human feedback.

Usage:
------

Manual Mode (generates prompts for you to submit):

    from interpretation import PRISMInterpreter

    interpreter = PRISMInterpreter(backend="manual")
    result = interpreter.interpret_window(window_id=42)
    print(result.prompt_used)  # Copy to Claude/GPT

API Mode (requires API key):

    import os
    from interpretation import PRISMInterpreter

    interpreter = PRISMInterpreter(
        backend="claude",
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )
    result = interpreter.interpret_window(window_id=42)
    print(result.interpretation)

Record Feedback:

    from interpretation import FeedbackManager

    feedback = FeedbackManager()
    feedback.record_feedback(
        interpretation_id=result.interpretation_id,
        feedback_type="validated",
        notes="Accurate analysis of the regime break"
    )

Natural Language Queries:

    result = interpreter.answer_question("What drove the 2024 regime break?")
    print(result.interpretation)

Components:
-----------
- `PRISMInterpreter`: Main interpretation engine with multiple AI backends
- `InterpretationResult`: Container for interpretations with metadata
- `AIContext`: Builds structured context from PRISM outputs
- `FeedbackManager`: Manages human feedback on interpretations
- `TEMPLATES`: Prompt templates for different interpretation tasks

Supported Backends:
-------------------
- `manual`: Returns formatted prompts for manual submission
- `claude`: Uses Anthropic's Claude API (requires anthropic package)
- `openai`: Uses OpenAI's API (requires openai package)
- `ollama`: Uses local Ollama instance (requires requests package)

Environment Variables:
----------------------
- `ANTHROPIC_API_KEY`: API key for Claude backend
- `OPENAI_API_KEY`: API key for OpenAI backend
- `OLLAMA_ENDPOINT`: Endpoint for Ollama (default: http://localhost:11434)
"""

from .ai_context import (
    AIContext,
    EventContext,
    IndicatorContext,
    RegimeBreakContext,
    WindowContext,
)
from .feedback import FeedbackManager, FeedbackRecord, ValidatedPattern
from .interpreter import InterpretationResult, PRISMInterpreter
from .prompt_templates import (
    TEMPLATES,
    TEMPLATE_METADATA,
    get_template,
    get_template_metadata,
    list_templates,
    render_template,
)

__all__ = [
    # Main interpreter
    "PRISMInterpreter",
    "InterpretationResult",
    # Context building
    "AIContext",
    "WindowContext",
    "IndicatorContext",
    "EventContext",
    "RegimeBreakContext",
    # Feedback
    "FeedbackManager",
    "FeedbackRecord",
    "ValidatedPattern",
    # Templates
    "TEMPLATES",
    "TEMPLATE_METADATA",
    "get_template",
    "get_template_metadata",
    "list_templates",
    "render_template",
]

__version__ = "0.1.0"
