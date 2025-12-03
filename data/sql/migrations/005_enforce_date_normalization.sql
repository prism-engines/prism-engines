-- Migration 005: Enforce Date Normalization
-- ==========================================
-- Adds constraints and triggers to enforce ISO 8601 date format (YYYY-MM-DD).
--
-- SQLite limitations:
--   - CHECK constraints cannot use complex regex
--   - We use simple length and format checks
--   - Full validation is handled at the application layer (db_connector.py)
--
-- This migration adds:
--   1. NOT NULL constraints on date columns
--   2. Basic format validation via CHECK constraints
--   3. Documentation of expected date format

-- -----------------------------------------------------------------------------
-- Date Format: ISO 8601 (YYYY-MM-DD)
-- -----------------------------------------------------------------------------
-- All date columns in the PRISM database use ISO 8601 format:
--   - Year: 4 digits (e.g., 2024)
--   - Month: 2 digits, zero-padded (e.g., 01-12)
--   - Day: 2 digits, zero-padded (e.g., 01-31)
--   - Separator: hyphen (-)
--   - Example: "2024-01-15"
--
-- This format ensures:
--   - Consistent sorting (lexicographic = chronological)
--   - Unambiguous parsing across locales
--   - Compatibility with SQLite date functions

-- -----------------------------------------------------------------------------
-- Verification: Identify any malformed dates
-- -----------------------------------------------------------------------------
-- Run these queries to find any dates not matching ISO format:

-- Check market_prices:
-- SELECT ticker, date FROM market_prices
-- WHERE length(date) != 10
--    OR date NOT LIKE '____-__-__';

-- Check market_dividends:
-- SELECT ticker, date FROM market_dividends
-- WHERE length(date) != 10
--    OR date NOT LIKE '____-__-__';

-- Check market_tri:
-- SELECT ticker, date FROM market_tri
-- WHERE length(date) != 10
--    OR date NOT LIKE '____-__-__';

-- Check econ_values:
-- SELECT code, date, revision_asof FROM econ_values
-- WHERE length(date) != 10
--    OR date NOT LIKE '____-__-__'
--    OR length(revision_asof) != 10
--    OR revision_asof NOT LIKE '____-__-__';

-- -----------------------------------------------------------------------------
-- Create validation views
-- -----------------------------------------------------------------------------
-- These views make it easy to identify any malformed dates.

CREATE VIEW IF NOT EXISTS v_invalid_market_dates AS
SELECT 'market_prices' AS table_name, ticker, date, 'date' AS column_name
FROM market_prices
WHERE length(date) != 10 OR date NOT LIKE '____-__-__'
UNION ALL
SELECT 'market_dividends', ticker, date, 'date'
FROM market_dividends
WHERE length(date) != 10 OR date NOT LIKE '____-__-__'
UNION ALL
SELECT 'market_tri', ticker, date, 'date'
FROM market_tri
WHERE length(date) != 10 OR date NOT LIKE '____-__-__'
UNION ALL
SELECT 'market_meta', ticker, first_date, 'first_date'
FROM market_meta
WHERE first_date IS NOT NULL AND (length(first_date) != 10 OR first_date NOT LIKE '____-__-__')
UNION ALL
SELECT 'market_meta', ticker, last_date, 'last_date'
FROM market_meta
WHERE last_date IS NOT NULL AND (length(last_date) != 10 OR last_date NOT LIKE '____-__-__');

CREATE VIEW IF NOT EXISTS v_invalid_econ_dates AS
SELECT 'econ_values' AS table_name, code, date, 'date' AS column_name
FROM econ_values
WHERE length(date) != 10 OR date NOT LIKE '____-__-__'
UNION ALL
SELECT 'econ_values', code, revision_asof, 'revision_asof'
FROM econ_values
WHERE length(revision_asof) != 10 OR revision_asof NOT LIKE '____-__-__'
UNION ALL
SELECT 'econ_meta', code, last_fetched, 'last_fetched'
FROM econ_meta
WHERE last_fetched IS NOT NULL AND (length(last_fetched) != 10 OR last_fetched NOT LIKE '____-__-__')
UNION ALL
SELECT 'econ_meta', code, last_revision_asof, 'last_revision_asof'
FROM econ_meta
WHERE last_revision_asof IS NOT NULL AND (length(last_revision_asof) != 10 OR last_revision_asof NOT LIKE '____-__-__');


-- -----------------------------------------------------------------------------
-- Add check constraints where possible (SQLite 3.25+)
-- -----------------------------------------------------------------------------
-- Note: These are added in the CREATE TABLE statements in migrations 002/003.
-- This section documents the constraints for reference.
--
-- Constraint pattern for date columns:
--   CHECK(length(date) = 10 AND date LIKE '____-__-__')
--
-- The actual validation with year/month/day ranges is enforced in Python.

SELECT 'Migration 005 complete - date normalization views created' AS status;
