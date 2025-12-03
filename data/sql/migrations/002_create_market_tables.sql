-- Migration 002: Create Market Tables
-- ====================================
-- Creates tables for market data: prices, dividends, TRI, and metadata.
--
-- Tables:
--   - market_prices: Daily price data with optional derived columns
--   - market_dividends: Dividend payment records
--   - market_tri: Total Return Index values
--   - market_meta: Ticker metadata (date ranges, source info)

-- -----------------------------------------------------------------------------
-- market_prices Table
-- -----------------------------------------------------------------------------
-- Stores daily price data for securities with optional derived columns.

CREATE TABLE IF NOT EXISTS market_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,                    -- ISO format: YYYY-MM-DD
    price REAL NOT NULL,
    ret REAL,                              -- Daily return (optional)
    price_z REAL,                          -- Normalized price (optional)
    price_log REAL,                        -- Log price (optional)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_market_prices_ticker ON market_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_market_prices_date ON market_prices(date);
CREATE INDEX IF NOT EXISTS idx_market_prices_ticker_date ON market_prices(ticker, date);


-- -----------------------------------------------------------------------------
-- market_dividends Table
-- -----------------------------------------------------------------------------
-- Stores dividend payment records for securities.

CREATE TABLE IF NOT EXISTS market_dividends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,                    -- ISO format: YYYY-MM-DD
    dividend REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_market_dividends_ticker ON market_dividends(ticker);
CREATE INDEX IF NOT EXISTS idx_market_dividends_date ON market_dividends(date);


-- -----------------------------------------------------------------------------
-- market_tri Table
-- -----------------------------------------------------------------------------
-- Stores Total Return Index (TRI) values for securities.
-- TRI accounts for both price changes and reinvested dividends.

CREATE TABLE IF NOT EXISTS market_tri (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,                    -- ISO format: YYYY-MM-DD
    tri_value REAL NOT NULL,
    tri_z REAL,                            -- Normalized TRI (optional)
    tri_log REAL,                          -- Log TRI (optional)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_market_tri_ticker ON market_tri(ticker);
CREATE INDEX IF NOT EXISTS idx_market_tri_date ON market_tri(date);
CREATE INDEX IF NOT EXISTS idx_market_tri_ticker_date ON market_tri(ticker, date);


-- -----------------------------------------------------------------------------
-- market_meta Table
-- -----------------------------------------------------------------------------
-- Stores metadata about each ticker (date ranges, source, notes).

CREATE TABLE IF NOT EXISTS market_meta (
    ticker TEXT PRIMARY KEY,
    first_date TEXT,                       -- ISO format: YYYY-MM-DD
    last_date TEXT,                        -- ISO format: YYYY-MM-DD
    source TEXT,                           -- Data source (Yahoo Finance, etc.)
    notes TEXT,                            -- Additional notes
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- Triggers: Update timestamps
-- -----------------------------------------------------------------------------

CREATE TRIGGER IF NOT EXISTS update_market_prices_timestamp
    AFTER UPDATE ON market_prices
    FOR EACH ROW
BEGIN
    UPDATE market_prices SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS update_market_tri_timestamp
    AFTER UPDATE ON market_tri
    FOR EACH ROW
BEGIN
    UPDATE market_tri SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS update_market_meta_timestamp
    AFTER UPDATE ON market_meta
    FOR EACH ROW
BEGIN
    UPDATE market_meta SET updated_at = CURRENT_TIMESTAMP WHERE ticker = OLD.ticker;
END;
