"""
Sector Technical Indicators
===========================

Builds technical indicators for sectors and other assets.

Technical indicators include:
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD)
    - Volatility indicators (Bollinger Bands, ATR)
    - Trend indicators (ADX)
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def build_technical_indicators(
    registry: Dict[str, Any],
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, int]:
    """
    Build technical indicators for assets defined in the registry.

    Args:
        registry: The metric registry containing technical indicator definitions
        conn: SQLite database connection
        start_date: Start date for computation (YYYY-MM-DD)
        end_date: End date for computation (YYYY-MM-DD)

    Returns:
        Dictionary mapping technical indicator names to row counts written
    """
    results: Dict[str, int] = {}
    technical_defs = registry.get("technical", [])

    if not technical_defs:
        logger.info("No technical indicators defined in registry")
        return results

    cursor = conn.cursor()

    for tech in technical_defs:
        name = tech.get("name")
        indicator_type = tech.get("type", "sma")
        base_series = tech.get("base")
        params = tech.get("params", {})

        if not name or not base_series:
            continue

        try:
            # Load base series
            cursor.execute(
                """
                SELECT iv.date, iv.value
                FROM indicator_values iv
                JOIN indicators i ON iv.indicator_id = i.id
                WHERE i.name = ?
                ORDER BY iv.date
                """,
                (base_series,),
            )
            rows = cursor.fetchall()

            if not rows:
                logger.warning(f"No base data for technical {name}")
                results[name] = 0
                continue

            df = pd.DataFrame(rows, columns=["date", "value"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            series = df["value"]

            # Compute technical indicator
            output_series = _compute_technical(series, indicator_type, params)

            if output_series is None or output_series.empty:
                logger.warning(f"Empty result for technical {name}")
                results[name] = 0
                continue

            # Get or create indicator
            cursor.execute("SELECT id FROM indicators WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                indicator_id = row[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO indicators (name, system, frequency, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, "finance", "daily", "technical"),
                )
                indicator_id = cursor.lastrowid

            # Write values
            count = 0
            for date_val, value in output_series.items():
                if pd.isna(value):
                    continue
                date_str = date_val.strftime("%Y-%m-%d")
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
                    VALUES (?, ?, ?)
                    """,
                    (indicator_id, date_str, float(value)),
                )
                count += 1

            conn.commit()
            results[name] = count
            logger.info(f"Built technical {name}: {count} rows")

        except Exception as e:
            logger.error(f"Error building technical {name}: {e}")
            results[name] = 0

    return results


def _compute_technical(
    series: pd.Series,
    indicator_type: str,
    params: Dict[str, Any],
) -> Optional[pd.Series]:
    """
    Compute a technical indicator.

    Supported types:
        - sma: Simple Moving Average
        - ema: Exponential Moving Average
        - rsi: Relative Strength Index
        - macd: MACD line
        - bollinger_upper: Upper Bollinger Band
        - bollinger_lower: Lower Bollinger Band
        - volatility: Rolling standard deviation
    """
    indicator_type = indicator_type.lower()
    window = params.get("window", 20)

    if indicator_type == "sma":
        return series.rolling(window=window).mean()

    elif indicator_type == "ema":
        return series.ewm(span=window, adjust=False).mean()

    elif indicator_type == "rsi":
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    elif indicator_type == "macd":
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    elif indicator_type == "bollinger_upper":
        std_dev = params.get("std_dev", 2)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return sma + (std_dev * std)

    elif indicator_type == "bollinger_lower":
        std_dev = params.get("std_dev", 2)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return sma - (std_dev * std)

    elif indicator_type == "volatility":
        return series.rolling(window=window).std()

    elif indicator_type == "momentum":
        return series.pct_change(periods=window) * 100

    else:
        logger.warning(f"Unknown indicator type '{indicator_type}'")
        return None
