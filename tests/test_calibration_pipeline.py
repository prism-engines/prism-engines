"""
Calibration Pipeline Diagnostics Tests
======================================

Verifies the calibration pipeline components work correctly:
1. Calibration panel loads (with fallback support)
2. SQL tables exist after schema extension
3. Dashboard renders without crashing
4. Run_calibrated completes in fallback mode

These tests ensure the pipeline never crashes, even with missing data.

Usage:
    pytest tests/test_calibration_pipeline.py -v
"""

import pytest
import sys
import tempfile
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

# Add start directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "start"))
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSQLSchemaExtension:
    """Test SQL schema auto-migration functionality."""

    def test_ensure_schema_creates_tables(self, tmp_path):
        """Verify ensure_schema creates all required tables."""
        from start.sql_schema_extension import SCHEMA_EXTENSION

        # Create temporary database
        db_path = tmp_path / "test_prism.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute schema extension
        for statement in SCHEMA_EXTENSION.split(';'):
            statement = statement.strip()
            if statement:
                cursor.execute(statement)

        conn.commit()

        # Check required tables exist
        required_tables = [
            'calibration_lenses',
            'calibration_indicators',
            'calibration_config',
            'analysis_rankings',
            'analysis_signals',
            'consensus_events',
            'consensus_history',
        ]

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}

        for table in required_tables:
            assert table in existing_tables, f"Table {table} not created"

        conn.close()

    def test_table_exists_function(self, tmp_path, monkeypatch):
        """Test table_exists helper function."""
        from start import sql_schema_extension

        # Create temporary database with one table
        db_path = tmp_path / "test_prism.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER)")
        conn.commit()
        conn.close()

        # Monkeypatch DB_PATH
        monkeypatch.setattr(sql_schema_extension, 'DB_PATH', db_path)

        # Test function
        assert sql_schema_extension.table_exists('test_table') is True
        assert sql_schema_extension.table_exists('nonexistent') is False

    def test_get_table_row_count(self, tmp_path, monkeypatch):
        """Test get_table_row_count helper function."""
        from start import sql_schema_extension

        # Create temporary database with data
        db_path = tmp_path / "test_prism.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER)")
        conn.execute("INSERT INTO test_table VALUES (1), (2), (3)")
        conn.commit()
        conn.close()

        # Monkeypatch DB_PATH
        monkeypatch.setattr(sql_schema_extension, 'DB_PATH', db_path)

        # Test function
        assert sql_schema_extension.get_table_row_count('test_table') == 3
        assert sql_schema_extension.get_table_row_count('nonexistent') == 0


class TestCalibrationPanelFallback:
    """Test calibration panel loading with fallbacks."""

    def test_synthetic_panel_generation(self):
        """Verify synthetic panel generates valid data."""
        from start.run_calibrated import _generate_synthetic_panel

        panel = _generate_synthetic_panel()

        # Check panel structure
        assert isinstance(panel, pd.DataFrame)
        assert len(panel) > 0
        assert len(panel.columns) >= 10  # Should have at least 10 indicators

        # Check index is datetime
        assert isinstance(panel.index, pd.DatetimeIndex)

        # Check no all-NaN columns
        assert not panel.isna().all().any()

        # Check reasonable value ranges for known columns
        if 'vix' in panel.columns:
            assert panel['vix'].min() >= 5
            assert panel['vix'].max() <= 100

    def test_synthetic_panel_is_reproducible(self):
        """Verify synthetic panel is reproducible with same seed."""
        from start.run_calibrated import _generate_synthetic_panel

        panel1 = _generate_synthetic_panel()
        panel2 = _generate_synthetic_panel()

        # Should be identical due to fixed seed
        pd.testing.assert_frame_equal(panel1, panel2)


class TestDashboardFallback:
    """Test dashboard renders without crashing."""

    def test_dashboard_imports(self):
        """Verify dashboard module imports without error."""
        from start import dashboard_full
        assert hasattr(dashboard_full, 'generate_full_dashboard')

    def test_dashboard_handles_empty_signals(self, tmp_path, monkeypatch):
        """Test dashboard renders with empty signals table."""
        from start import sql_schema_extension
        from start import dashboard_full

        # Create temporary database with schema but no data
        db_path = tmp_path / "test_prism.db"
        conn = sqlite3.connect(db_path)

        # Create tables
        for statement in sql_schema_extension.SCHEMA_EXTENSION.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    conn.execute(statement)
                except sqlite3.Error:
                    pass

        conn.commit()
        conn.close()

        # Monkeypatch paths
        monkeypatch.setattr(sql_schema_extension, 'DB_PATH', db_path)
        monkeypatch.setattr(sql_schema_extension, '_schema_ensured', False)

        # Also need to set output path
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        monkeypatch.setattr(dashboard_full, 'OUTPUT_PATH', output_dir / "dashboard_full.html")
        monkeypatch.setattr(dashboard_full, 'OUTPUT_DIR', output_dir)

        # Generate dashboard - should not raise
        dashboard_full.generate_full_dashboard()

        # Check output file was created
        assert (output_dir / "dashboard_full.html").exists()

        # Check it contains degraded mode banner
        html_content = (output_dir / "dashboard_full.html").read_text()
        assert "Calibration Not Yet Completed" in html_content


class TestCalibratedAnalysisFallback:
    """Test calibrated analysis runs with fallbacks."""

    def test_fallback_mode_with_empty_panel(self, tmp_path, monkeypatch, capsys):
        """Test analysis completes in fallback mode when panel is empty."""
        from start import run_calibrated
        from start import sql_schema_extension

        # Create temporary database with schema but no data
        db_path = tmp_path / "test_prism.db"
        conn = sqlite3.connect(db_path)

        for statement in sql_schema_extension.SCHEMA_EXTENSION.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    conn.execute(statement)
                except sqlite3.Error:
                    pass

        conn.commit()
        conn.close()

        # Create output directory
        output_dir = tmp_path / "output" / "calibrated_analysis"
        output_dir.mkdir(parents=True)

        # Monkeypatch paths
        monkeypatch.setattr(run_calibrated, 'DATA_DIR', tmp_path)
        monkeypatch.setattr(run_calibrated, 'OUTPUT_PATH', output_dir)
        monkeypatch.setattr(sql_schema_extension, 'DB_PATH', db_path)
        monkeypatch.setattr(sql_schema_extension, '_schema_ensured', False)

        # Run analysis - should complete without error
        rankings, signals = run_calibrated.run_calibrated_analysis()

        # Verify outputs
        assert isinstance(rankings, pd.DataFrame)
        assert isinstance(signals, pd.DataFrame)

        # Check files were created
        assert (output_dir / "rankings.csv").exists()
        assert (output_dir / "signals.csv").exists()

        # Check fallback mode was activated
        captured = capsys.readouterr()
        assert "FALLBACK MODE" in captured.out or "synthetic" in captured.out.lower()


class TestCalibrationLoader:
    """Test calibration loader functionality."""

    def test_get_lens_weights_default(self):
        """Test get_lens_weights returns defaults when no calibration."""
        from start.calibration_loader import get_lens_weights

        weights = get_lens_weights()

        # Should return dict with default equal weights
        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(isinstance(v, (int, float)) for v in weights.values())

    def test_apply_weights_to_rankings(self):
        """Test apply_weights_to_rankings function."""
        from start.calibration_loader import apply_weights_to_rankings

        # Create sample rankings
        rankings_df = pd.DataFrame({
            'pca': [1, 2, 3, 4, 5],
            'momentum': [5, 4, 3, 2, 1],
            'correlation': [2, 3, 4, 5, 1],
        }, index=['A', 'B', 'C', 'D', 'E'])

        result = apply_weights_to_rankings(rankings_df)

        # Should add weighted_rank and final_rank columns
        assert 'weighted_rank' in result.columns
        assert 'final_rank' in result.columns

        # Should be sorted by final_rank
        assert result['final_rank'].is_monotonic_increasing


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_with_synthetic_data(self, tmp_path, monkeypatch):
        """Test complete pipeline runs end-to-end with synthetic data."""
        from start import run_calibrated
        from start import sql_schema_extension

        # Create temporary database
        db_path = tmp_path / "test_prism.db"
        conn = sqlite3.connect(db_path)

        for statement in sql_schema_extension.SCHEMA_EXTENSION.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    conn.execute(statement)
                except sqlite3.Error:
                    pass

        conn.commit()
        conn.close()

        # Create output directory
        output_dir = tmp_path / "output" / "calibrated_analysis"
        output_dir.mkdir(parents=True)

        # Monkeypatch paths
        monkeypatch.setattr(run_calibrated, 'DATA_DIR', tmp_path)
        monkeypatch.setattr(run_calibrated, 'OUTPUT_PATH', output_dir)
        monkeypatch.setattr(sql_schema_extension, 'DB_PATH', db_path)
        monkeypatch.setattr(sql_schema_extension, '_schema_ensured', False)

        # Run analysis
        rankings, signals = run_calibrated.run_calibrated_analysis()

        # Verify all expected outputs exist
        expected_files = ['rankings.csv', 'signals.csv', 'summary.png']
        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Missing output file: {filename}"

        # Verify data quality
        assert len(rankings) > 0
        assert 'final_rank' in rankings.columns
        assert not rankings.index.duplicated().any()


# Run diagnostics if executed directly
if __name__ == "__main__":
    print("=" * 60)
    print("CALIBRATION PIPELINE DIAGNOSTICS")
    print("=" * 60)

    # Quick smoke tests
    print("\n1. Testing SQL Schema Extension...")
    try:
        from start.sql_schema_extension import ensure_schema, table_exists
        ensure_schema(verbose=True)
        print("   OK - Schema ensured")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n2. Testing Synthetic Panel Generation...")
    try:
        from start.run_calibrated import _generate_synthetic_panel
        panel = _generate_synthetic_panel()
        print(f"   OK - Generated {len(panel)} rows x {len(panel.columns)} columns")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n3. Testing Calibration Loader...")
    try:
        from start.calibration_loader import get_lens_weights, get_top_lenses
        weights = get_lens_weights()
        top = get_top_lenses(3)
        print(f"   OK - {len(weights)} lenses, top 3: {top}")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n4. Testing Dashboard Import...")
    try:
        from start.dashboard_full import generate_full_dashboard
        print("   OK - Dashboard module imports successfully")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n" + "=" * 60)
    print("Diagnostics complete. Run 'pytest tests/test_calibration_pipeline.py -v' for full tests.")
    print("=" * 60)
