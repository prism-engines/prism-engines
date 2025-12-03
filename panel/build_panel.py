"""
PRISM Panel Builder
===================

Main panel building logic that reads from the database and registries
to produce a unified, engine-ready panel.

Usage:
    from panel.build_panel import build_panel

    # Build with default settings
    df = build_panel()

    # Build with custom output path
    df = build_panel(output_path="data/panels/custom_panel.csv")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from panel.validators import validate_panel
from panel.transforms_market import align_market_series
from panel.transforms_econ import align_econ_series, forward_fill_to_daily

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project root (parent of panel directory)
PROJECT_ROOT = Path(__file__).parent.parent


def load_registry(registry_path: Path) -> dict:
    """
    Load a JSON registry file.

    Args:
        registry_path: Path to the registry JSON file

    Returns:
        Parsed registry dictionary

    Raises:
        FileNotFoundError: If registry file doesn't exist
        json.JSONDecodeError: If registry is not valid JSON
    """
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with open(registry_path) as f:
        return json.load(f)


def load_system_registry() -> dict:
    """
    Load the system registry.

    Returns:
        System registry dictionary
    """
    registry_path = PROJECT_ROOT / "data" / "registry" / "system_registry.json"
    return load_registry(registry_path)


def load_market_registry() -> dict:
    """
    Load the market registry.

    Returns:
        Market registry dictionary
    """
    system_reg = load_system_registry()
    market_path = PROJECT_ROOT / system_reg["registries"]["market"]
    return load_registry(market_path)


def load_economic_registry() -> dict:
    """
    Load the economic registry.

    Returns:
        Economic registry dictionary
    """
    system_reg = load_system_registry()
    econ_path = PROJECT_ROOT / system_reg["registries"]["economic"]
    return load_registry(econ_path)


def get_db_module():
    """
    Dynamically import the database module.

    Returns:
        The prism_db module
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from data.sql import prism_db
    return prism_db


def query_market_series(
    market_registry: dict,
    db_module,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Query market series from the database based on registry configuration.

    Args:
        market_registry: Market registry dictionary
        db_module: Database module with load_indicator function
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Dictionary mapping series names to DataFrames
    """
    series_dict = {}

    for ticker, config in market_registry.items():
        if not config.get("enabled", True):
            logger.debug(f"Skipping disabled market series: {ticker}")
            continue

        indicator_name = config.get("indicator_name", ticker)
        use_column = config.get("use_column", "value")

        try:
            df = db_module.load_indicator(
                name=indicator_name,
                start_date=start_date,
                end_date=end_date,
            )

            if df.empty:
                logger.warning(f"No data for market series: {indicator_name}")
                continue

            # Rename value column to ticker name
            df = df.rename(columns={"value": ticker})
            series_dict[ticker] = df

            logger.info(
                f"Loaded market series {ticker}: {len(df)} rows"
            )

        except ValueError as e:
            logger.warning(f"Could not load market series {indicator_name}: {e}")
        except Exception as e:
            logger.error(f"Error loading market series {indicator_name}: {e}")

    return series_dict


def query_economic_series(
    econ_registry: dict,
    db_module,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Query economic series from the database based on registry configuration.

    Args:
        econ_registry: Economic registry dictionary
        db_module: Database module with load_indicator function
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Dictionary mapping series names to DataFrames
    """
    series_dict = {}

    for code, config in econ_registry.items():
        if not config.get("enabled", True):
            logger.debug(f"Skipping disabled economic series: {code}")
            continue

        indicator_name = config.get("indicator_name", code.lower())
        use_column = config.get("use_column", "value")

        try:
            df = db_module.load_indicator(
                name=indicator_name,
                start_date=start_date,
                end_date=end_date,
            )

            if df.empty:
                logger.warning(f"No data for economic series: {indicator_name}")
                continue

            # Rename value column to indicator name
            df = df.rename(columns={"value": indicator_name})
            series_dict[indicator_name] = df

            logger.info(
                f"Loaded economic series {indicator_name}: {len(df)} rows"
            )

        except ValueError as e:
            logger.warning(f"Could not load economic series {indicator_name}: {e}")
        except Exception as e:
            logger.error(f"Error loading economic series {indicator_name}: {e}")

    return series_dict


def combine_series(
    market_series: dict[str, pd.DataFrame],
    econ_series: dict[str, pd.DataFrame],
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Combine market and economic series into a single panel DataFrame.

    Args:
        market_series: Dictionary of market series DataFrames
        econ_series: Dictionary of economic series DataFrames
        fill_method: Method for filling gaps ('ffill', 'none')

    Returns:
        Combined panel DataFrame with date index
    """
    all_series = {}

    # Add market series
    for name, df in market_series.items():
        if name in df.columns:
            all_series[name] = df[[name]]
        elif "value" in df.columns:
            all_series[name] = df[["value"]].rename(columns={"value": name})
        elif len(df.columns) == 1:
            all_series[name] = df.rename(columns={df.columns[0]: name})

    # Add economic series
    for name, df in econ_series.items():
        if name in df.columns:
            all_series[name] = df[[name]]
        elif "value" in df.columns:
            all_series[name] = df[["value"]].rename(columns={"value": name})
        elif len(df.columns) == 1:
            all_series[name] = df.rename(columns={df.columns[0]: name})

    if not all_series:
        logger.warning("No series to combine")
        return pd.DataFrame()

    # Outer join all series on date index
    combined = None
    for name, df in all_series.items():
        if combined is None:
            combined = df
        else:
            combined = combined.join(df, how="outer")

    # Sort by date
    combined = combined.sort_index()

    # Apply fill method
    if fill_method == "ffill":
        combined = combined.ffill()

    logger.info(
        f"Combined {len(all_series)} series into panel: "
        f"{len(combined)} rows x {len(combined.columns)} columns"
    )

    return combined


def build_panel(
    output_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fill_method: str = "ffill",
    validate: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the master panel from database and registry configuration.

    This is the main entry point for panel construction.

    Args:
        output_path: Path to save the panel CSV (default from system_registry)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        fill_method: Method for filling gaps ('ffill', 'none')
        validate: If True, run validation checks on the panel
        save: If True, save the panel to disk

    Returns:
        The constructed panel DataFrame

    Raises:
        FileNotFoundError: If registries are not found
        PanelValidationError: If validation fails and raise_on_error=True
    """
    logger.info("=" * 60)
    logger.info("Starting panel build")
    logger.info("=" * 60)

    # Load registries
    logger.info("Loading registries...")
    system_registry = load_system_registry()
    market_registry = load_market_registry()
    econ_registry = load_economic_registry()

    logger.info(
        f"Loaded registries: {len(market_registry)} market series, "
        f"{len(econ_registry)} economic series"
    )

    # Get database module
    db_module = get_db_module()

    # Query market series
    logger.info("Querying market series from database...")
    market_series = query_market_series(
        market_registry, db_module, start_date, end_date
    )
    logger.info(f"Retrieved {len(market_series)} market series")

    # Query economic series
    logger.info("Querying economic series from database...")
    econ_series = query_economic_series(
        econ_registry, db_module, start_date, end_date
    )
    logger.info(f"Retrieved {len(econ_series)} economic series")

    # Combine series
    logger.info("Combining series into panel...")
    panel = combine_series(market_series, econ_series, fill_method)

    if panel.empty:
        logger.warning("Panel is empty - no data was loaded from database")
        return panel

    # Validate panel
    if validate:
        logger.info("Validating panel...")
        min_rows = system_registry.get("panel", {}).get("min_rows", 100)
        max_nan_ratio = system_registry.get("panel", {}).get("max_nan_ratio", 0.5)

        validation_result = validate_panel(
            panel,
            min_rows=min_rows,
            max_nan_ratio=max_nan_ratio,
            raise_on_error=False,
        )

        if validation_result["is_valid"]:
            logger.info("Panel validation PASSED")
        else:
            logger.warning(
                f"Panel validation completed with {len(validation_result['errors'])} error(s)"
            )
            for error in validation_result["errors"]:
                logger.error(f"  - {error}")

        for warning in validation_result["warnings"]:
            logger.warning(f"  - {warning}")

    # Save panel
    if save:
        if output_path is None:
            output_path = system_registry.get("panel", {}).get(
                "master_panel_path", "data/panels/master_panel.csv"
            )

        output_file = PROJECT_ROOT / output_path

        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save with date as first column
        panel_to_save = panel.copy()
        panel_to_save.index.name = "date"
        panel_to_save.to_csv(output_file)

        logger.info(f"Panel saved to: {output_file}")
        logger.info(f"  Shape: {panel.shape[0]} rows x {panel.shape[1]} columns")

    logger.info("=" * 60)
    logger.info("Panel build complete")
    logger.info("=" * 60)

    return panel


def build_panel_from_csv(
    input_path: str,
    output_path: Optional[str] = None,
    validate: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build/validate a panel from an existing CSV file.

    Useful for re-validating or transforming existing panels.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save the panel (optional)
        validate: If True, run validation checks
        save: If True, save the panel to disk

    Returns:
        The loaded panel DataFrame
    """
    logger.info(f"Loading panel from: {input_path}")

    input_file = PROJECT_ROOT / input_path
    panel = pd.read_csv(input_file, index_col=0, parse_dates=True)

    logger.info(f"Loaded panel: {panel.shape[0]} rows x {panel.shape[1]} columns")

    if validate:
        system_registry = load_system_registry()
        min_rows = system_registry.get("panel", {}).get("min_rows", 100)
        max_nan_ratio = system_registry.get("panel", {}).get("max_nan_ratio", 0.5)

        validation_result = validate_panel(
            panel,
            min_rows=min_rows,
            max_nan_ratio=max_nan_ratio,
        )

        if validation_result["is_valid"]:
            logger.info("Panel validation PASSED")
        else:
            logger.warning("Panel validation completed with warnings/errors")

    if save and output_path:
        output_file = PROJECT_ROOT / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_file)
        logger.info(f"Panel saved to: {output_file}")

    return panel


if __name__ == "__main__":
    # Run panel build when executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Build PRISM panel")
    parser.add_argument(
        "--output", "-o",
        help="Output path for panel CSV",
        default=None,
    )
    parser.add_argument(
        "--start-date",
        help="Start date filter (YYYY-MM-DD)",
        default=None,
    )
    parser.add_argument(
        "--end-date",
        help="End date filter (YYYY-MM-DD)",
        default=None,
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to disk",
    )

    args = parser.parse_args()

    panel = build_panel(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        validate=not args.no_validate,
        save=not args.no_save,
    )

    print(f"\nPanel shape: {panel.shape}")
    print(f"Columns: {list(panel.columns)}")
    if not panel.empty:
        print(f"Date range: {panel.index.min()} to {panel.index.max()}")
