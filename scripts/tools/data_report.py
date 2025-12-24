#!/usr/bin/env python3
"""Data Phase Integrity Report (Markdown)

Semantic Phase: DATA
- Admissible indicator data: ingestion, cleaning, normalization, suitability, cohorts

Goal:
- Print a long-form, human-auditable report to stdout
- Write a permanent copy to reports/data/<run_id>.md

This is intentionally verbose. Many reports over time is a feature, not a bug.
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import hashlib
import pandas as pd

from prism.db.open import open_prism_db

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("reports/data"))
    return p.parse_args()

def q(conn, sql, params=None):
    """Execute query, return empty DataFrame on error."""
    try:
        return conn.execute(sql, params or []).fetchdf()
    except Exception:
        return pd.DataFrame()

def main():
    args = parse_args()
    conn = open_prism_db()

    # Canonical tables
    run = q(conn, "SELECT * FROM meta.data_runs WHERE run_id = ?", [args.run_id])
    steps = q(conn, "SELECT * FROM meta.data_steps WHERE run_id = ? ORDER BY step_name", [args.run_id])
    lock = q(conn, "SELECT * FROM meta.data_run_lock WHERE run_id = ?", [args.run_id])

    # Legacy tables (optional - may not exist)
    windows = q(conn, "SELECT * FROM meta.geometry_windows WHERE run_id = ? ORDER BY indicator_id", [args.run_id])
    elig = q(conn, "SELECT * FROM meta.engine_eligibility WHERE run_id = ? ORDER BY indicator_id, window_years", [args.run_id])
    scans = q(conn, "SELECT * FROM meta.phase1_scan_results WHERE run_id = ? ORDER BY indicator_id, window_years", [args.run_id])

    lines = []
    lines.append(f"# PRISM Data Phase Integrity Report")
    lines.append("")
    lines.append(f"- Run ID: `{args.run_id}`")

    # Lock status (immutability indicator)
    if len(lock) == 1:
        lk = lock.iloc[0].to_dict()
        lines.append(f"- **Status: LOCKED (IMMUTABLE)** - locked at `{lk.get('started_at')}`")
    else:
        lines.append(f"- Status: UNLOCKED (run may be incomplete or failed)")

    if len(run) == 1:
        r = run.iloc[0].to_dict()
        lines.append(f"- Domain: `{r.get('domain')}`")
        lines.append(f"- Mode: `{r.get('mode')}`")
        lines.append(f"- Started: `{r.get('started_at')}`")
        lines.append(f"- Completed: `{r.get('completed_at')}`")
        lines.append(f"- Status: `{r.get('status')}`")
        lines.append(f"- Indicators requested: `{r.get('indicators_requested')}`")
        lines.append(f"- Indicators scanned: `{r.get('indicators_scanned')}`")
    else:
        lines.append("- (run header not found in meta.data_runs)")

    lines.append("\n---\n")
    lines.append("## Step Log")
    if len(steps) == 0:
        lines.append("No step records found.")
    else:
        for _, s in steps.iterrows():
            lines.append(f"- **{s['step_name']}**: `{s['status']}` (items={s.get('n_items')})")
            if s.get('error_message'):
                lines.append(f"  - error: `{s['error_message']}`")

    lines.append("\n---\n")
    lines.append("## Optimal Window Summary")
    if len(windows) == 0:
        lines.append("(Legacy table meta.geometry_windows not available)")
    else:
        dist = windows.groupby('optimal_window_y').size().reset_index(name='count').sort_values('count', ascending=False)
        lines.append("Window distribution:")
        for _, row in dist.iterrows():
            lines.append(f"- `{row['optimal_window_y']}y`: {int(row['count'])}")
        lines.append("")
        lines.append("Top 20 indicators by lowest quality_score:")
        worst = windows.sort_values('quality_score', ascending=True).head(20)
        for _, row in worst.iterrows():
            lines.append(f"- `{row['indicator_id']}`: window={row['optimal_window_y']}y geom={row['dominant_geometry']} q={row['quality_score']:.2f} conf={row['avg_confidence']:.2f} disagree={row['avg_disagreement']:.2f} stab={row['stability']:.2f}")

    lines.append("\n---\n")
    lines.append("## Math Suitability Summary (per indicator, per window)")
    if len(elig) == 0:
        lines.append("(Legacy table meta.engine_eligibility not available)")
    else:
        status_counts = elig.groupby('status').size().reset_index(name='count').sort_values('count', ascending=False)
        lines.append("Status distribution:")
        for _, row in status_counts.iterrows():
            lines.append(f"- `{row['status']}`: {int(row['count'])}")
        lines.append("")
        window_counts = elig.groupby(['window_years','status']).size().reset_index(name='count')
        lines.append("By window:")
        for w in sorted(window_counts['window_years'].unique()):
            subset = window_counts[window_counts['window_years']==w]
            parts = ", ".join([f"{r['status']}={int(r['count'])}" for _, r in subset.iterrows()])
            lines.append(f"- `{w}y`: {parts}")

    lines.append("\n---\n")
    lines.append("## Per-Indicator Detail (Optimal window + eligibility)")
    if len(windows) == 0:
        lines.append("No per-indicator detail available.")
    else:
        for _, row in windows.iterrows():
            ind = row['indicator_id']
            lines.append(f"\n### {ind}")
            lines.append(f"- Optimal window: `{row['optimal_window_y']}y` | geom=`{row['dominant_geometry']}` | q={row['quality_score']:.2f} | conf={row['avg_confidence']:.2f} | disagree={row['avg_disagreement']:.2f} | stab={row['stability']:.2f}")
            e = elig[elig['indicator_id']==ind]
            if len(e) == 0:
                lines.append("- Eligibility: (none recorded)")
            else:
                for _, er in e.iterrows():
                    lines.append(f"  - `{er['window_years']}y`: **{er['status']}** geom=`{er['geometry']}` conf={er['confidence']:.2f} disagree={er['disagreement']:.2f} stab={er['stability']:.2f}")

    if len(scans) > 0:
        lines.append("\n---\n")
        lines.append("## Full Scan Trace")
        lines.append("(Every indicator-window pair, including non-optimal windows.)")
        for ind in scans['indicator_id'].unique():
            lines.append(f"\n### {ind}")
            s = scans[scans['indicator_id']==ind]
            for _, r in s.iterrows():
                mark = "*" if bool(r.get('is_optimal')) else "-"
                lines.append(f"{mark} `{r['window_years']}y` geom=`{r['dominant_geometry']}` pct={r['geometry_pct']:.2f} q={r['quality_score']:.2f} conf={r['avg_confidence']:.2f} disagree={r['avg_disagreement']:.2f} trans={int(r['n_transitions'])} obs={int(r['n_observations'])}")

    md = "\n".join(lines) + "\n"
    sha = hashlib.sha256(md.encode("utf-8")).hexdigest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.run_id}.md"
    out_path.write_text(md, encoding="utf-8")

    # Close read-only connection before opening read-write
    conn.close()

    # Register report in DB (path + hash) - best effort
    try:
        conn_rw = open_prism_db()
        conn_rw.execute(
            "INSERT INTO meta.data_reports(run_id, report_path, sha256, report_markdown) VALUES (?, ?, ?, ?)",
            [args.run_id, str(out_path), sha, md]
        )
        conn_rw.close()
        print(f"[report registered in DB: meta.data_reports sha256={sha[:16]}...]")
    except Exception as e:
        print(f"[warning: could not register report in DB: {e}]")

    print(md)
    print(f"\n[report written to {out_path}]")

if __name__ == "__main__":
    main()
