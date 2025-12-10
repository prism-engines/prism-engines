-- Migration 002: Schema v2 - Panel-Agnostic Data Migration
-- =========================================================
-- Migrates data from legacy tables to indicator_values
-- This migration is idempotent - safe to run multiple times

-- Migrate market_prices data to indicator_values
INSERT OR IGNORE INTO indicator_values (indicator_name, date, value, provenance)
SELECT ticker, date, value, 'migration:market_prices'
FROM market_prices
WHERE ticker IS NOT NULL AND date IS NOT NULL AND value IS NOT NULL;

-- Migrate econ_values data to indicator_values
INSERT OR IGNORE INTO indicator_values (indicator_name, date, value, provenance)
SELECT series_id, date, value, 'migration:econ_values'
FROM econ_values
WHERE series_id IS NOT NULL AND date IS NOT NULL AND value IS NOT NULL;

-- Record migration in metadata
INSERT OR REPLACE INTO metadata (key, value)
VALUES ('schema_version', 'v2');

INSERT OR REPLACE INTO metadata (key, value)
VALUES ('migration_002_completed', datetime('now'));
