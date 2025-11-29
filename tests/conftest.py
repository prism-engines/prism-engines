"""
Pytest fixtures for PRISM tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_panel_small():
    """Small sample panel with 4 indicators for fast tests."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    np.random.seed(42)

    return pd.DataFrame({
        "date": dates,
        "indicator_a": np.cumsum(np.random.randn(100)) + 100,
        "indicator_b": np.cumsum(np.random.randn(100)) + 50,
        "indicator_c": np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + 100,
        "indicator_d": np.random.randn(100) * 5 + 75,
    })


@pytest.fixture
def sample_panel_medium():
    """Medium sample panel with 20 indicators."""
    dates = pd.date_range("2015-01-01", periods=500, freq="D")

    np.random.seed(42)

    data = {"date": dates}
    for i in range(20):
        data[f"ind_{i+1}"] = np.cumsum(np.random.randn(500) * 0.5) + 100 + i * 10

    return pd.DataFrame(data)


@pytest.fixture
def sample_panel_with_nan():
    """Sample panel with missing values for cleaning tests."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    np.random.seed(42)

    data = pd.DataFrame({
        "date": dates,
        "clean": np.random.randn(100) + 100,
        "sparse": np.random.randn(100) + 50,
        "gappy": np.random.randn(100) + 75,
    })

    # Add NaN patterns
    data.loc[10:15, "sparse"] = np.nan  # Small gap
    data.loc[50:70, "gappy"] = np.nan   # Large gap
    data.loc[[5, 25, 45, 65, 85], "sparse"] = np.nan  # Scattered

    return data


@pytest.fixture
def mock_fred_response():
    """Mock FRED API response."""
    dates = pd.date_range("2020-01-01", periods=50, freq="M")
    return pd.DataFrame({
        "date": dates,
        "value": np.random.randn(50) + 2.5
    })
