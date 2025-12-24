#!/usr/bin/env python
"""
Phase-selectable PRISM plotting (read-only).
"""

import argparse
import duckdb

from prism.db.config import CANONICAL_DB_PATH
from prism.visualization.io import (
    load_engine_series,
    load_geometry_series,
)
from prism.visualization.plots import (
    plot_indicator_engine,
    plot_engine_geometry,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(CANONICAL_DB_PATH), help="Path to DuckDB file")
    p.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--view", required=True, choices=["indicator_engine", "engine_geometry"])
    p.add_argument("--engine", required=True)
    p.add_argument("--indicator")
    p.add_argument("--bounded", choices=["bounded", "unbounded"], default="unbounded")
    p.add_argument("--columns", help="Comma-separated list of columns to plot")
    p.add_argument("--save", help="Optional path to save PNG" )

    args = p.parse_args()

    bounded_flag = args.bounded == "bounded"
    columns = args.columns.split(",") if args.columns else None

    conn = duckdb.connect(args.db, read_only=True)

    if args.view == "indicator_engine":
        if not args.indicator:
            raise ValueError("--indicator is required for indicator_engine view")
        df = load_engine_series(
            conn,
            phase=args.phase,
            engine=args.engine,
            indicator=args.indicator,
            bounded=bounded_flag,
        )
        title = f"Phase {args.phase} | {args.engine} | {args.indicator} | {args.bounded}"
        plot_indicator_engine(df, title=title, columns=columns, save_path=args.save)

    else:
        engine = None if args.engine == "system" else args.engine
        df = load_geometry_series(
            conn,
            phase=args.phase,
            engine=engine,
            bounded=bounded_flag,
        )
        title = f"Phase {args.phase} | Geometry | {args.engine} | {args.bounded}"
        plot_engine_geometry(df, title=title, columns=columns, save_path=args.save)


if __name__ == "__main__":
    main()
