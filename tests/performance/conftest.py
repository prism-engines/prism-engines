"""
Pytest Configuration for Performance Tests
===========================================

Shared fixtures and configuration for the performance test suite.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def sample_panel_data():
    """
    Create sample panel data for testing.

    Returns:
        pd.DataFrame: Wide-format panel with date index and indicator columns
    """
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B")

    # Create correlated indicators for realistic testing
    base = np.random.randn(252)

    data = {
        "indicator_1": base + np.random.randn(252) * 0.2,
        "indicator_2": base * 0.8 + np.random.randn(252) * 0.3,
        "indicator_3": np.random.randn(252),  # Independent
        "indicator_4": -base * 0.6 + np.random.randn(252) * 0.4,  # Negative correlation
        "indicator_5": np.random.randn(252),  # Independent
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_lens_panel(sample_panel_data):
    """
    Create sample panel data in lens format (with 'date' column).

    Returns:
        pd.DataFrame: Panel with 'date' column and indicator columns
    """
    df = sample_panel_data.reset_index()
    df = df.rename(columns={"index": "date"})
    return df


@pytest.fixture
def large_sample_panel():
    """
    Create larger sample panel for performance testing.

    Returns:
        pd.DataFrame: Wide-format panel with 1000 rows
    """
    np.random.seed(42)
    dates = pd.date_range(start="2018-01-01", periods=1000, freq="B")

    data = {
        f"indicator_{i}": np.random.randn(1000)
        for i in range(20)
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_family_manager():
    """Create a mock FamilyManager for testing."""
    from unittest.mock import MagicMock

    mock_fm = MagicMock()
    mock_fm.list_families.return_value = [
        {
            "id": "test_family",
            "canonical_name": "Test Family",
            "members": {
                "test_ind_1": {"source": "test", "resolution": "daily"},
                "test_ind_2": {"source": "test", "resolution": "monthly"},
            }
        }
    ]

    return mock_fm


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    from unittest.mock import MagicMock

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    return mock_conn, mock_cursor


# Markers for test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_db: marks tests that require database"
    )
