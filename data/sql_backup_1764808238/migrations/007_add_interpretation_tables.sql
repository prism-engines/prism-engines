-- Migration 007: Add AI interpretation tables
-- ============================================
--
-- This migration adds tables for storing AI-generated interpretations
-- of PRISM analysis results, including human feedback for validation.

-- Store AI-generated interpretations
CREATE TABLE IF NOT EXISTS interpretations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interpretation_type TEXT NOT NULL,  -- 'window', 'event', 'regime', 'query', 'cross_domain', 'indicator'
    target_id INTEGER,                  -- window_id, event_id, etc. (NULL for queries)
    target_secondary_id INTEGER,        -- For regime breaks: window_after_id
    prompt_template TEXT NOT NULL,      -- Template name used
    prompt_rendered TEXT NOT NULL,      -- Full rendered prompt
    context_json TEXT NOT NULL,         -- JSON snapshot of input data
    interpretation TEXT NOT NULL,       -- AI-generated interpretation
    backend TEXT NOT NULL,              -- 'claude', 'openai', 'ollama', 'manual'
    model TEXT,                         -- Model identifier (e.g., 'claude-sonnet-4-20250514')
    created_at TEXT DEFAULT (datetime('now')),

    -- Human feedback fields
    validated INTEGER DEFAULT 0,        -- 0=pending, 1=validated, -1=rejected
    feedback_type TEXT,                 -- 'validated', 'rejected', 'refined'
    feedback_notes TEXT,                -- Human notes on the interpretation
    refined_interpretation TEXT,        -- Human-corrected version
    feedback_at TEXT                    -- When feedback was provided
);

CREATE INDEX IF NOT EXISTS idx_interpretations_type
    ON interpretations(interpretation_type);
CREATE INDEX IF NOT EXISTS idx_interpretations_target
    ON interpretations(target_id);
CREATE INDEX IF NOT EXISTS idx_interpretations_validated
    ON interpretations(validated);
CREATE INDEX IF NOT EXISTS idx_interpretations_backend
    ON interpretations(backend);
CREATE INDEX IF NOT EXISTS idx_interpretations_created
    ON interpretations(created_at);

-- Store validated patterns for reference
-- These are reusable interpretation templates extracted from validated interpretations
CREATE TABLE IF NOT EXISTS validated_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,         -- 'regime_break', 'coherence_spike', 'divergence', etc.
    pattern_name TEXT NOT NULL,         -- Human-readable name
    description TEXT NOT NULL,          -- Description of what this pattern means
    conditions_json TEXT NOT NULL,      -- JSON describing when pattern applies
    interpretation_template TEXT,       -- Reusable interpretation text template
    source_interpretation_id INTEGER,   -- Original interpretation this was extracted from
    examples_json TEXT,                 -- JSON array of example occurrences
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (source_interpretation_id) REFERENCES interpretations(id)
);

CREATE INDEX IF NOT EXISTS idx_validated_patterns_type
    ON validated_patterns(pattern_type);

-- Trigger to update updated_at on validated_patterns
CREATE TRIGGER IF NOT EXISTS update_validated_patterns_timestamp
    AFTER UPDATE ON validated_patterns
    FOR EACH ROW
    BEGIN
        UPDATE validated_patterns
        SET updated_at = datetime('now')
        WHERE id = OLD.id;
    END;

-- Store natural language queries and responses
-- Separate from interpretations for simpler querying
CREATE TABLE IF NOT EXISTS nl_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,             -- User's natural language question
    context_json TEXT NOT NULL,         -- JSON of data used to answer
    context_summary TEXT,               -- Brief summary of available context
    response TEXT NOT NULL,             -- AI-generated response
    backend TEXT NOT NULL,              -- AI backend used
    model TEXT,                         -- Model identifier
    helpful INTEGER,                    -- User rating: 1=helpful, 0=not helpful, NULL=no rating
    follow_up_questions TEXT,           -- JSON array of suggested follow-ups
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_nl_queries_helpful
    ON nl_queries(helpful);
CREATE INDEX IF NOT EXISTS idx_nl_queries_created
    ON nl_queries(created_at);

-- Store coherence events detected by the engine
-- This links AI interpretations to specific detected events
CREATE TABLE IF NOT EXISTS coherence_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date TEXT NOT NULL,           -- Date of the event
    event_type TEXT NOT NULL,           -- 'convergence', 'divergence', 'spike', 'regime_change'
    coherence_score REAL NOT NULL,      -- Coherence score at event
    window_id INTEGER,                  -- Associated analysis window
    participating_lenses TEXT,          -- JSON array of lens names involved
    indicator_snapshot TEXT,            -- JSON of indicator states at event
    severity TEXT,                      -- 'low', 'medium', 'high', 'critical'
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (window_id) REFERENCES windows(id)
);

CREATE INDEX IF NOT EXISTS idx_coherence_events_date
    ON coherence_events(event_date);
CREATE INDEX IF NOT EXISTS idx_coherence_events_type
    ON coherence_events(event_type);
CREATE INDEX IF NOT EXISTS idx_coherence_events_severity
    ON coherence_events(severity);
