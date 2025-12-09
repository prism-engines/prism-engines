-- PRISM Schema v2.1 - Core Database Schema
-- -----------------------------------------
-- This schema defines:
--   - systems (domain registry)
--   - indicators (metadata registry)
--   - indicator_values (UNIVERSAL VIEW)
--   - fetch_log
--   - metadata
-- Backward-compatible tables:
--   - market_prices
--   - econ_values
-- -----------------------------------------

---------------------------------------------
-- SYSTEMS TABLE
---------------------------------------------
CREATE TABLE IF NOT EXISTS systems (
    system TEXT PRIMARY KEY
);

---------------------------------------------
-- INDICATORS TABLE
---------------------------------------------
CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    system TEXT NOT NULL,
    frequency TEXT DEFAULT 'daily',
    source TEXT,
    description TEXT,
    metadata TEXT,
    FOREIGN KEY(system) REFERENCES systems(system)
);

---------------------------------------------
-- UNIVERSAL VALUE VIEW - indicator_values
-- DO NOT MODIFY THIS VIEW IN MIGRATIONS
---------------------------------------------
CREATE VIEW IF NOT EXISTS indicator_values AS
SELECT
    i.id AS indicator_id,
    i.name AS indicator_name,
    i.system AS system,
    v.date AS date,
    v.value AS value,
    v.source AS source
FROM timeseries v
JOIN indicators i ON v.indicator_id = i.id;

---------------------------------------------
-- BACKWARD-COMPATIBILITY TABLES
---------------------------------------------
CREATE TABLE IF NOT EXISTS market_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    date TEXT,
    value REAL
);

CREATE TABLE IF NOT EXISTS econ_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT,
    date TEXT,
    value REAL
);

---------------------------------------------
-- METADATA STORE
---------------------------------------------
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

---------------------------------------------
-- FETCH LOG
---------------------------------------------
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_id INTEGER,
    date TEXT,
    status TEXT,
    message TEXT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
