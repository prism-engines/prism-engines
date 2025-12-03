"""
PRISM Prompt Templates
======================

Structured prompt templates for AI interpretation of PRISM outputs.

Each template is designed to:
1. Provide sufficient context for accurate interpretation
2. Ground all observations in quantitative data
3. Generate actionable, hypothesis-driven insights
4. Maintain domain-agnostic applicability

Templates use Python string formatting with named placeholders.
"""

from typing import Dict, Any

# Template metadata for documentation and validation
TEMPLATE_METADATA: Dict[str, Dict[str, Any]] = {
    "window_summary": {
        "description": "Summarize findings from a single analysis window",
        "required_fields": [
            "window_label", "start_date", "end_date", "top_indicators_table",
            "coherence_score", "agreeing_lenses", "disagreeing_lenses",
            "regime_transition_text"
        ],
        "output_format": "3-5 paragraphs",
    },
    "coherence_event": {
        "description": "Interpret a detected coherence event",
        "required_fields": [
            "event_date", "event_type", "coherence_score", "lens_list",
            "indicator_snapshot", "similar_events"
        ],
        "output_format": "2-3 paragraphs",
    },
    "regime_break": {
        "description": "Interpret a regime transition between windows",
        "required_fields": [
            "window_before_label", "before_start", "before_end",
            "window_after_label", "after_start", "after_end",
            "spearman_corr", "top_10_overlap", "rank_change_table",
            "historical_breaks"
        ],
        "output_format": "3-4 paragraphs",
    },
    "cross_domain_pattern": {
        "description": "Compare patterns across multiple systems/domains",
        "required_fields": [
            "systems_list", "correlation_matrix", "synchronized_events",
            "divergent_periods"
        ],
        "output_format": "3-4 paragraphs",
    },
    "indicator_deep_dive": {
        "description": "Deep analysis of a single indicator over time",
        "required_fields": [
            "indicator_name", "system", "frequency", "data_start", "data_end",
            "ranking_history_table", "high_importance_periods",
            "low_importance_periods", "lens_breakdown"
        ],
        "output_format": "4-5 paragraphs",
    },
    "natural_language_query": {
        "description": "Answer a natural language question about PRISM results",
        "required_fields": ["context_summary", "user_question"],
        "output_format": "Direct answer with citations",
    },
}


TEMPLATES: Dict[str, str] = {
    "window_summary": """You are analyzing outputs from PRISM, a mathematical framework that measures coherence across multiple analytical lenses. High coherence (lens agreement) signals significant system states; low coherence (disagreement) represents normal market noise or transitional periods.

## Analysis Window: {window_label}
Period: {start_date} to {end_date}

## Top Ranked Indicators (Most Influential)
{top_indicators_table}

## Lens Agreement Analysis
- Overall coherence score: {coherence_score:.2f}
- Lenses in agreement: {agreeing_lenses}
- Lenses in disagreement: {disagreeing_lenses}

## Regime Context
{regime_transition_text}

## Historical Comparison
{historical_comparison}

---

## Task
Provide a concise interpretation (3-5 paragraphs) that:

1. **State Assessment**: Explain what the top indicators suggest about the current system state. Which asset classes or economic factors are most influential? What does their prominence indicate?

2. **Lens Interpretation**: Interpret the lens agreement/disagreement pattern. High agreement suggests a clear signal; disagreement may indicate regime uncertainty or conflicting forces.

3. **Regime Context**: If there was a recent regime transition, contextualize what structural changes occurred. If stable, note the persistence of the current regime.

4. **Hypotheses**: Identify 1-2 specific hypotheses worth investigating further. These should be grounded in the data but suggest actionable next steps.

Ground all observations in the quantitative data provided above. Do not speculate beyond what the numbers support. Use phrases like "the data suggests" rather than making definitive claims.
""",

    "coherence_event": """PRISM detected a coherence event where multiple analytical lenses converged on similar conclusions about indicator importance.

## Event Details
- Date: {event_date}
- Event Type: {event_type}
- Coherence Score: {coherence_score:.2f} (range: 0-1, higher = more agreement)
- Participating Lenses: {lens_list}

## Indicator States at Event
The following shows how key indicators were ranked when this event occurred:
{indicator_snapshot}

## Historical Context
Previous similar events in the dataset:
{similar_events}

---

## Task
Provide an interpretation (2-3 paragraphs) that:

1. **Event Significance**: Explain what this lens convergence likely indicates about the system state. Why might multiple independent analytical methods agree at this point?

2. **Historical Comparison**: Compare this event to historical parallels if available. Are the indicator patterns similar? What happened after previous similar events?

3. **Forward Guidance**: Suggest what to monitor going forward. Which indicators or lens outputs would confirm or refute the signal this event represents?

Be specific about the indicators involved. Avoid vague statements like "market uncertainty" without grounding in the actual data.
""",

    "regime_break": """PRISM detected a potential regime break between consecutive analysis windows. A regime break indicates a structural shift in which indicators matter most.

## Transition Details
- From: {window_before_label} ({before_start} to {before_end})
- To: {window_after_label} ({after_start} to {after_end})

## Quantitative Evidence
- Spearman rank correlation between windows: {spearman_corr:.3f}
  (Values close to 1.0 = stable regime, values near 0 or negative = regime break)
- Top 10 indicator overlap: {top_10_overlap}%
- Number of indicators analyzed: {n_indicators}

## Indicators with Largest Rank Changes
{rank_change_table}

## Historical Regime Breaks
Previous regime breaks detected in this dataset:
{historical_breaks}

---

## Task
Provide an interpretation (3-4 paragraphs) that:

1. **Structural Change**: Explain what structural change this regime break represents. Which categories of indicators gained or lost importance? What does this shift suggest about the underlying dynamics?

2. **Key Drivers**: Identify which specific indicators drove the shift. Look at the rank change table - which movements are most significant and why?

3. **Potential Causes**: Suggest potential causes for this regime break, with appropriate uncertainty. Consider macroeconomic events, policy changes, or market structure shifts that might explain the timing.

4. **Implications**: What are the implications for analysis going forward? Should the interpretation of certain indicators change?

Be careful to distinguish correlation from causation. The data shows what changed, not necessarily why.
""",

    "cross_domain_pattern": """Comparing coherence patterns across multiple systems/domains in PRISM. This analysis identifies whether different analytical domains (e.g., finance, climate) show synchronized or divergent patterns.

## Systems Analyzed
{systems_list}

## Cross-Domain Correlation Matrix
Correlation of coherence scores between systems:
{correlation_matrix}

## Synchronized Events
Periods where multiple systems showed similar coherence patterns:
{synchronized_events}

## Divergent Periods
Periods where systems showed opposing or uncorrelated patterns:
{divergent_periods}

---

## Task
Provide an interpretation (3-4 paragraphs) that:

1. **Relationship Assessment**: Identify any meaningful relationships between these systems. Are they generally correlated, anti-correlated, or independent?

2. **Synchronization Analysis**: For synchronized events, hypothesize what might cause these systems to move together. Are there common external factors?

3. **Divergence Analysis**: For divergent periods, explore what might cause the systems to decouple. This can be as informative as synchronization.

4. **Cross-Domain Hypotheses**: Generate hypotheses about potential causal or correlational links. These should be testable and grounded in the data.

Note: Cross-domain analysis is exploratory. Apparent correlations may be spurious, especially with limited data. Frame findings as hypotheses, not conclusions.
""",

    "indicator_deep_dive": """Deep analysis of a single indicator's behavior across time in PRISM. This examines how the indicator's importance has varied and what factors influence its ranking.

## Indicator Profile
- Name: {indicator_name}
- System/Domain: {system}
- Data Frequency: {frequency}
- Data Range: {data_start} to {data_end}
- Total Windows Analyzed: {n_windows}

## Ranking History
How this indicator ranked across analysis windows:
{ranking_history_table}

## Periods of High Importance
Windows where this indicator ranked in top quartile:
{high_importance_periods}

## Periods of Low Importance
Windows where this indicator ranked in bottom quartile:
{low_importance_periods}

## Lens-Specific Views
How different lenses view this indicator:
{lens_breakdown}

---

## Task
Provide an interpretation (4-5 paragraphs) that:

1. **Role Analysis**: Explain this indicator's role in the system over time. Is it consistently important, or does its importance vary with conditions?

2. **Importance Drivers**: Identify what conditions make this indicator more or less important. Look at the timing of high/low importance periods - what was happening?

3. **Lens Perspective**: Analyze how different lenses view this indicator. If lenses disagree, what does that tell us about the indicator's nature?

4. **Relationships**: Based on when this indicator is important, suggest related indicators that might be worth monitoring together.

5. **Interpretation Guidance**: How should analysts interpret signals from this indicator? When is it most/least reliable?

Ground analysis in the specific numbers provided. Avoid generic statements that could apply to any indicator.
""",

    "natural_language_query": """You have access to PRISM analysis results. Answer the user's question based solely on the data provided. Do not use external knowledge.

## Available Data Summary
{context_summary}

## Detailed Context
{context_details}

## User Question
{user_question}

---

## Instructions
1. Answer the question directly and concisely
2. Cite specific numbers, dates, and indicators when relevant
3. If the data doesn't fully support an answer, clearly state what is and isn't known
4. If the question cannot be answered from the available data, say so explicitly
5. Suggest 1-2 follow-up queries that might provide additional insight

Format your response as:
**Answer**: [Your direct answer]

**Evidence**: [Specific data points supporting your answer]

**Limitations**: [What the data doesn't tell us, if applicable]

**Suggested Follow-ups**: [Related questions to explore]
""",

    "comparative_indicators": """Compare the behavior and importance of multiple indicators in PRISM.

## Indicators Being Compared
{indicator_list}

## Ranking Comparison Across Windows
{ranking_comparison_table}

## Correlation of Rankings
{ranking_correlation_matrix}

## Periods of Alignment
When these indicators moved together in the rankings:
{alignment_periods}

## Periods of Divergence
When these indicators moved in opposite directions:
{divergence_periods}

---

## Task
Provide an interpretation (3-4 paragraphs) that:

1. **Relationship Characterization**: Describe the relationship between these indicators in terms of their importance rankings. Are they substitutes, complements, or independent?

2. **Alignment Interpretation**: When they align, what does that typically indicate about market/system conditions?

3. **Divergence Interpretation**: When they diverge, what does that typically indicate? Which one tends to lead?

4. **Portfolio/Analysis Implications**: For analysts, what are the implications of these relationships?
""",

    "anomaly_interpretation": """PRISM's anomaly detection lens flagged unusual behavior in the following indicators.

## Anomaly Summary
- Detection Window: {window_label}
- Period: {start_date} to {end_date}
- Total Anomalies Detected: {n_anomalies}

## Flagged Indicators
{anomaly_table}

## Historical Anomaly Context
Previous anomalies for these indicators:
{historical_anomalies}

## System State at Detection
{system_state}

---

## Task
Provide an interpretation (2-3 paragraphs) that:

1. **Anomaly Assessment**: What type of unusual behavior was detected? Is this a single-indicator issue or a systemic pattern?

2. **Potential Causes**: What might cause these anomalies? Consider data issues, genuine market events, or structural changes.

3. **Response Guidance**: What should an analyst do with this information? Investigate further, adjust models, or note for future reference?

Distinguish between true anomalies (genuine unusual events) and potential data quality issues.
""",
}


def get_template(template_name: str) -> str:
    """
    Get a prompt template by name.

    Args:
        template_name: Name of the template (e.g., 'window_summary')

    Returns:
        Template string with placeholders

    Raises:
        KeyError: If template_name not found
    """
    if template_name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(f"Template '{template_name}' not found. Available: {available}")
    return TEMPLATES[template_name]


def get_template_metadata(template_name: str) -> Dict[str, Any]:
    """
    Get metadata for a template.

    Args:
        template_name: Name of the template

    Returns:
        Dictionary with description, required_fields, output_format

    Raises:
        KeyError: If template_name not found
    """
    if template_name not in TEMPLATE_METADATA:
        raise KeyError(f"Metadata for template '{template_name}' not found")
    return TEMPLATE_METADATA[template_name]


def render_template(template_name: str, **kwargs) -> str:
    """
    Render a template with provided values.

    Args:
        template_name: Name of the template
        **kwargs: Values to fill in the template

    Returns:
        Rendered template string

    Raises:
        KeyError: If template not found or required field missing
    """
    template = get_template(template_name)

    # Check for required fields if metadata exists
    if template_name in TEMPLATE_METADATA:
        required = TEMPLATE_METADATA[template_name].get("required_fields", [])
        missing = [f for f in required if f not in kwargs]
        if missing:
            raise KeyError(f"Missing required fields for '{template_name}': {missing}")

    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(f"Missing placeholder value: {e}")


def list_templates() -> Dict[str, str]:
    """
    List all available templates with their descriptions.

    Returns:
        Dictionary mapping template_name -> description
    """
    return {
        name: TEMPLATE_METADATA.get(name, {}).get("description", "No description")
        for name in TEMPLATES
    }
