PRAGMA foreign_keys = ON;

------------------------------------------------------------
-- SYSTEMS TABLE
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS systems (
    system TEXT PRIMARY KEY,
    description TEXT
);

------------------------------------------------------------
-- INDICATORS TABLE
-- Unified PRISM indicator registry (market, economy, benchmark, climate, etc.)
------------------------------------------------------------

CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    system TEXT NOT NULL DEFAULT 'market',
    frequency TEXT NOT NULL DEFAULT 'daily',
    source TEXT,
    units TEXT,
    description TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (system) REFERENCES systems(system)
);

CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);

------------------------------------------------------------
-- INDICATOR VALUES TABLE
-- Single time-series table for ALL indicators across all systems
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS indicator_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    value REAL,
    value_2 REAL,
    adjusted_value REAL,
    FOREIGN KEY(indicator_id) REFERENCES indicators(id),
    UNIQUE(indicator_id, date)
);

------------------------------------------------------------
-- DEFAULT SYSTEM SEED
------------------------------------------------------------
INSERT OR IGNORE INTO systems(system, description) VALUES
    ('market', 'Market price time series'),
    ('economy', 'Economic indicators'),
    ('benchmark', 'Synthetic benchmark datasets'),
    ('custom', 'User-defined or research datasets');

------------------------------------------------------------
-- FETCH LOG
------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_name TEXT,
    fred_code TEXT,
    timestamp TEXT,
    status TEXT,
    message TEXT
);
