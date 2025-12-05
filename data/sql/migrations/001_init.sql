CREATE TABLE market_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    date TEXT,
    value REAL
);

CREATE TABLE econ_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT,
    date TEXT,
    value REAL
);

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

