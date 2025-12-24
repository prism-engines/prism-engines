#!/usr/bin/env python3
"""Data Phase Visualization (canon-first).

Primary:
- X = window_end
- Y = geometry (categorical)
- Overlay: confidence + disagreement (secondary axis)

Fallback:
- Run-level optimal window distribution
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from prism.db.open import open_prism_db

GEOM_ORDER = ["pure_noise", "latent_flow", "reflexive_stochastic", "coupled_oscillator"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--indicator", required=False)
    p.add_argument("--window-years", type=float, default=None)
    p.add_argument("--out", type=Path, default=None, help="Output path (default: reports/data/{run_id}_viz.png)")
    return p.parse_args()

def main():
    args = parse_args()
    con = open_prism_db()

    # Default output path includes run_id
    if args.out is None:
        out_dir = Path("reports/data")
        if args.indicator:
            args.out = out_dir / f"{args.run_id}_{args.indicator}_viz.png"
        else:
            args.out = out_dir / f"{args.run_id}_viz.png"

    if args.indicator:
        sql = """
        SELECT window_end, geometry, confidence, disagreement, window_years
        FROM meta.temporal_geometry_observations
        WHERE run_id = ? AND indicator_id = ?
        """
        params = [args.run_id, args.indicator]
        if args.window_years is not None:
            sql += " AND window_years = ?"
            params.append(args.window_years)
        sql += " ORDER BY window_end"
        df = con.execute(sql, params).fetchdf()

        if len(df) == 0:
            # fallback to scan summary if temporal obs not stored yet
            df2 = con.execute(
                """
                SELECT window_years, quality_score
                FROM meta.phase1_scan_results
                WHERE run_id = ? AND indicator_id = ?
                ORDER BY window_years
                """,
                [args.run_id, args.indicator],
            ).fetchdf()
            if len(df2) == 0:
                raise SystemExit("No data for indicator in this run_id.")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df2["window_years"], df2["quality_score"], marker="o")
            ax.set_title(f"Data Phase scan summary: {args.indicator} (run={args.run_id})")
            ax.set_xlabel("Window (years)")
            ax.set_ylabel("Quality score")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out, dpi=160)
            print(f"[wrote {args.out}]")
            return

        geom_to_y = {g:i for i,g in enumerate(GEOM_ORDER)}
        df["y"] = df["geometry"].map(geom_to_y).fillna(-1)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["window_end"], df["y"], marker="o", linewidth=1)
        ax.set_yticks(list(range(len(GEOM_ORDER))))
        ax.set_yticklabels(GEOM_ORDER)
        ax.set_title(f"Temporal Geometry Over Time: {args.indicator} | run={args.run_id}")
        ax.set_xlabel("Window end date")
        ax.set_ylabel("Geometry")
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(df["window_end"], df["confidence"], marker=".", linewidth=1)
        ax2.plot(df["window_end"], df["disagreement"], marker=".", linewidth=1)
        ax2.set_ylabel("Confidence / Disagreement")

        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=180)
        print(f"[wrote {args.out}]")
        return

    win = con.execute(
        """
        SELECT optimal_window_y AS window_years, COUNT(*) AS n
        FROM meta.geometry_windows
        WHERE run_id = ?
        GROUP BY 1
        ORDER BY 1
        """,
        [args.run_id],
    ).fetchdf()

    if len(win) == 0:
        raise SystemExit("No geometry_windows rows found for this run_id.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(win["window_years"].astype(str), win["n"])
    ax.set_title(f"Optimal window distribution (run={args.run_id})")
    ax.set_xlabel("Window (years)")
    ax.set_ylabel("# indicators")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"[wrote {args.out}]")

if __name__ == "__main__":
    main()
