-- PRISM Schema v2.1 - Migration 001 (Safe, Idempotent)

INSERT OR IGNORE INTO systems (system) VALUES
    ('market'),
    ('economic'),
    ('benchmark');

-- Add missing columns to indicators (safe if they already exist)
ALTER TABLE indicators ADD COLUMN description TEXT;
ALTER TABLE indicators ADD COLUMN metadata TEXT;

-- Deprecated cleanup (only drops old raw tables, not views)
DROP TABLE IF EXISTS market_prices_old;
DROP TABLE IF EXISTS econ_values_old;

-- NOTE:
-- DO NOT modify indicator_values (it is a VIEW)
-- DO NOT drop or recreate the view here
-- DO NOT attempt to migrate market_prices/econ_values; legacy loaders rely on them
