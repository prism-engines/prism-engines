-- 2025_12_22_domains_schema.sql
-- Multi-domain data schema for orchestrator
--
-- Tables:
--   domains.indicators - Domain-specific indicator data
--   domains.metadata - Indicator metadata per domain


-- Domain data table
CREATE TABLE domains.indicators (
    date DATE,
    domain VARCHAR,
    indicator_id VARCHAR,
    value DOUBLE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, domain, indicator_id)
);

-- Domain metadata table
CREATE TABLE domains.metadata (
    indicator_id VARCHAR PRIMARY KEY,
    domain VARCHAR,
    name VARCHAR,
    source VARCHAR,
    category VARCHAR,
    frequency VARCHAR,
    description VARCHAR,
    first_date DATE,
    last_date DATE,
    row_count INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
