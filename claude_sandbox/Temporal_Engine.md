ll give you a PR-style bundle you can drop straight to Claude Code:

A clean resolution model (profiles + multi-resolution config)

A rewired start/temporal_runner.py that actually runs things instead of just printing usage

A minimal engine-side API so the runner isn’t a snowflake

1. Temporal Resolution Model (engine-level)
📄 File: engine/orchestration/temporal_analysis.py

Give this whole file to Claude as either:

“replace existing with this skeleton, but keep the internal math the same,” or

“merge these APIs into the current temporal_analysis module”.

"""
engine/orchestration/temporal_analysis.py

Unified temporal analysis orchestration for PRISM.

This module:
- Defines temporal "profiles" (chromebook / standard / powerful)
- Encodes resolution (step size, window lengths) in a single config object
- Provides a clean API for:
    * run_temporal_analysis_from_panel()
    * run_temporal_analysis_from_db()
    * quick_start_from_db()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal, Tuple

import datetime as dt

import numpy as np
import pandas as pd

# NOTE: Claude – wire these to the existing temporal engine implementation.
# For now, these are "front-door" orchestration functions; reuse the current
# multi-resolution / lens logic behind the scenes.


Frequency = Literal["D", "W", "M"]


@dataclass
class TemporalProfile:
    """Defines a named temporal profile for convenience presets."""

    name: str
    description: str
    step_months: float                  # e.g. 0.5, 1.0, 2.0
    window_months: Sequence[int]        # e.g. [3, 6, 12, 24]
    lenses: Sequence[str]               # engine lens IDs to run
    min_history_years: int = 10
    frequency: Frequency = "M"          # resample frequency for outputs


@dataclass
class TemporalConfig:
    """
    Full configuration for a temporal run.

    This can be constructed either from a TemporalProfile + overrides,
    or directly by callers.
    """

    profile_name: str
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

    step_months: float = 1.0
    window_months: Sequence[int] = field(default_factory=lambda: (3, 6, 12, 24))
    lenses: Sequence[str] = field(default_factory=lambda: ("coherence", "pca", "volatility"))
    min_history_years: int = 10
    frequency: Frequency = "M"

    systems: Optional[Sequence[str]] = None     # e.g. ["finance"], ["climate"], ["finance", "climate"]
    indicators: Optional[Sequence[str]] = None  # specific indicator IDs (optional)


@dataclass
class TemporalResult:
    """
    Canonical result for temporal runs.

    Claude – please adapt this to the current outputs:
    - time index
    - per-lens scores
    - any combined indices (e.g. coherence index)
    """

    config: TemporalConfig
    scores: pd.DataFrame          # index: date, columns: lens_* scores
    metadata: Dict[str, object]   # e.g., run timings, engine version, etc.


# ---------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------

TEMPORAL_PROFILES: Dict[str, TemporalProfile] = {
    "chromebook": TemporalProfile(
        name="chromebook",
        description="Light profile for low-CPU machines (3 lenses, ~2-month steps).",
        step_months=2.0,
        window_months=(6, 12, 24),
        lenses=("coherence", "pca", "volatility"),
        min_history_years=10,
        frequency="M",
    ),
    "standard": TemporalProfile(
        name="standard",
        description="Balanced profile (4 lenses, ~1-month steps).",
        step_months=1.0,
        window_months=(3, 6, 12, 24),
        lenses=("coherence", "pca", "volatility", "dispersion"),
        min_history_years=15,
        frequency="M",
    ),
    "powerful": TemporalProfile(
        name="powerful",
        description="High-resolution profile (5 lenses, 0.5-month steps).",
        step_months=0.5,
        window_months=(3, 6, 12, 24, 36),
        lenses=("coherence", "pca", "volatility", "dispersion", "network"),
        min_history_years=20,
        frequency="M",
    ),
}


def build_temporal_config(
    profile: str = "standard",
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    step_months: Optional[float] = None,
    window_months: Optional[Sequence[int]] = None,
    lenses: Optional[Sequence[str]] = None,
    systems: Optional[Sequence[str]] = None,
    indicators: Optional[Sequence[str]] = None,
    frequency: Optional[Frequency] = None,
) -> TemporalConfig:
    """
    Construct a TemporalConfig from a profile and overrides.

    This is the main entry point for the CLI runner & dashboard.
    """
    if profile not in TEMPORAL_PROFILES:
        raise ValueError(f"Unknown temporal profile: {profile!r}")

    base = TEMPORAL_PROFILES[profile]

    def _to_ts(val: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
        if val is None:
            return None
        if isinstance(val, pd.Timestamp):
            return val
        return pd.Timestamp(val)

    cfg = TemporalConfig(
        profile_name=base.name,
        start_date=_to_ts(start_date),
        end_date=_to_ts(end_date),
        step_months=step_months if step_months is not None else base.step_months,
        window_months=window_months if window_months is not None else tuple(base.window_months),
        lenses=lenses if lenses is not None else tuple(base.lenses),
        min_history_years=base.min_history_years,
        frequency=frequency if frequency is not None else base.frequency,
        systems=tuple(systems) if systems is not None else None,
        indicators=tuple(indicators) if indicators is not None else None,
    )
    return cfg


# ---------------------------------------------------------------------
# Orchestration API (to be wired to existing engine code)
# ---------------------------------------------------------------------


def run_temporal_analysis_from_panel(
    panel: pd.DataFrame,
    config: TemporalConfig,
) -> TemporalResult:
    """
    Run temporal analysis on a pre-built panel (wide DataFrame).

    Claude – please:
    - Reuse the existing temporal engine logic here.
    - The panel is expected to be date-indexed and wide-form (columns = indicators).
    - Return per-lens scores in a DataFrame with DatetimeIndex.
    """
    # Placeholder implementation – to be replaced with the real engine call.
    # This keeps the orchestration code stable while you wire in the engine.
    if panel.empty:
        raise ValueError("Panel is empty; cannot run temporal analysis.")

    dates = panel.index

    # Fake scores for now – to be replaced.
    scores = pd.DataFrame(
        index=dates,
        data={
            f"{lens}_score": np.linspace(0.0, 1.0, len(dates))
            for lens in config.lenses
        },
    )

    metadata: Dict[str, object] = {
        "n_dates": len(dates),
        "n_indicators": panel.shape[1],
        "profile": config.profile_name,
    }

    return TemporalResult(config=config, scores=scores, metadata=metadata)


def run_temporal_analysis_from_db(
    db_path: Optional[str],
    config: TemporalConfig,
) -> TemporalResult:
    """
    Load data from the indicator_values table and run temporal analysis.

    Claude – implement this by:
    - Using panel.runtime_loader (or a helper) to build a panel matching `config`.
    - Then delegate to run_temporal_analysis_from_panel().
    """
    from panel.runtime_loader import load_temporal_panel  # type: ignore

    panel = load_temporal_panel(
        db_path=db_path,
        systems=config.systems,
        indicators=config.indicators,
        min_history_years=config.min_history_years,
    )

    return run_temporal_analysis_from_panel(panel, config)


def quick_start_from_db(
    profile: str = "standard",
    db_path: Optional[str] = None,
) -> Tuple[TemporalResult, Dict[str, object]]:
    """
    Convenience helper: build config from profile, run from DB, and return
    (result, summary_dict).
    """
    cfg = build_temporal_config(profile=profile)
    result = run_temporal_analysis_from_db(db_path=db_path, config=cfg)

    summary = {
        "profile": cfg.profile_name,
        "start": result.scores.index.min(),
        "end": result.scores.index.max(),
        "n_dates": result.scores.shape[0],
        "n_lenses": len(cfg.lenses),
    }
    return result, summary

2. New start/temporal_runner.py (real CLI)

Right now when you run python start/temporal_runner.py it just prints usage.
Let’s upgrade it to:

Support --profile, --start, --end, --systems, --indicators

Run against your DB using panel.runtime_loader + temporal_analysis

Save outputs to prism_output/temporal/...

📄 File: start/temporal_runner.py

Replace the current file with this:

#!/usr/bin/env python3
"""
Temporal Analysis Runner

Command-line wrapper around the PRISM temporal engine.

Examples:

  # Basic run with standard profile (default dates)
  python start/temporal_runner.py --profile standard

  # Chromebook-friendly run
  python start/temporal_runner.py --profile chromebook

  # Powerful run, finance-only, last 20 years
  python start/temporal_runner.py --profile powerful \
      --systems finance \
      --start 2005-01-01

  # Multi-domain run with explicit indicators
  python start/temporal_runner.py --profile standard \
      --systems finance climate \
      --indicators sp500 vix t10y2y

This script is SAFE to run multiple times – it only reads from the DB and
writes outputs under prism_output/temporal/.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from engine.orchestration.temporal_analysis import (
    TEMPORAL_PROFILES,
    build_temporal_config,
    run_temporal_analysis_from_db,
)
from start.sql_schema_extension import get_default_db_path  # unified DB resolver


def _get_output_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "prism_output" / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PRISM temporal (multi-resolution) analysis from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--profile",
        choices=sorted(TEMPORAL_PROFILES.keys()),
        default="standard",
        help="Temporal profile to use (controls lenses and resolution).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, use as much history as available.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). If omitted, use latest available date.",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Optional list of system IDs (e.g. finance, climate, biology).",
    )
    parser.add_argument(
        "--indicators",
        nargs="*",
        default=None,
        help="Optional list of specific indicator IDs to include.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override database path. If omitted, use default resolver.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned configuration; do not run the engine.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Resolve DB path using the same logic as other scripts
    db_path = args.db_path or get_default_db_path()
    print("============================================================")
    print("PRISM TEMPORAL ANALYSIS RUNNER")
    print("============================================================")
    print(f"Profile:       {args.profile}")
    print(f"DB path:       {db_path}")
    print(f"Systems:       {args.systems or '[ALL]'}")
    print(f"Indicators:    {args.indicators or '[ALL]'}")
    print(f"Start date:    {args.start or '[auto]'}")
    print(f"End date:      {args.end or '[latest]'}")
    print("------------------------------------------------------------")

    # Build configuration
    cfg = build_temporal_config(
        profile=args.profile,
        start_date=args.start,
        end_date=args.end,
        systems=args.systems,
        indicators=args.indicators,
    )

    if args.dry_run:
        print("Dry run only – no analysis executed.")
        print("Resolved config:")
        print(f"  step_months:   {cfg.step_months}")
        print(f"  window_months: {list(cfg.window_months)}")
        print(f"  lenses:        {list(cfg.lenses)}")
        print(f"  frequency:     {cfg.frequency}")
        print(f"  min_history:   {cfg.min_history_years} years")
        return 0

    # Run the temporal engine
    result = run_temporal_analysis_from_db(db_path=db_path, config=cfg)

    # Save outputs
    out_dir = _get_output_dir()
    scores_path = out_dir / f"temporal_scores_{cfg.profile_name}.csv"
    meta_path = out_dir / f"temporal_metadata_{cfg.profile_name}.json"

    result.scores.to_csv(scores_path)
    pd.Series(result.metadata).to_json(meta_path, indent=2)

    print("✅ Analysis complete.")
    print(f"  Dates:      {result.scores.index.min().date()} → {result.scores.index.max().date()}")
    print(f"  Rows:       {result.scores.shape[0]}")
    print(f"  Lenses:     {len(cfg.lenses)}")
    print(f"  Output CSV: {scores_path}")
    print(f"  Metadata:   {meta_path}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


You can test it right now (once Claude wires temporal_analysis + runtime_loader):

cd prism-engine
python start/temporal_runner.py --profile standard
python start/temporal_runner.py --profile chromebook --dry-run
python start/temporal_runner.py --profile powerful --systems finance

3. Small test skeleton for Claude (optional but nice)

If you want to “hint” tests to Claude:

📄 File: tests/test_temporal_profiles.py
import pandas as pd

from engine.orchestration.temporal_analysis import (
    TEMPORAL_PROFILES,
    build_temporal_config,
)


def test_profiles_registered():
    assert "chromebook" in TEMPORAL_PROFILES
    assert "standard" in TEMPORAL_PROFILES
    assert "powerful" in TEMPORAL_PROFILES


def test_build_temporal_config_basic():
    cfg = build_temporal_config(profile="standard")
    assert cfg.profile_name == "standard"
    assert cfg.step_months > 0
    assert len(cfg.window_months) >= 3
    assert len(cfg.lenses) >= 3


def test_build_temporal_config_overrides():
    cfg = build_temporal_config(
        profile="standard",
        start_date="2000-01-01",
        end_date="2010-01-01",
        step_months=0.5,
        lenses=["coherence"],
    )
    assert cfg.start_date == pd.Timestamp("2000-01-01")
    assert cfg.end_date == pd.Timestamp("2010-01-01")
    assert cfg.step_months == 0.5
    assert list(cfg.lenses) == ["coherence"]

How to hand this to Claude

You can say something like:

“Here’s the next PR: refactor temporal engine resolution & runner.
Use these files as the desired API and wire them to the existing temporal engine logic without changing the math. Make sure start/temporal_runner.py runs successfully with my current DB.”

Then paste:

engine/orchestration/temporal_analysis.py code

start/temporal_runner.py code

(Optionally) tests/test_temporal_profiles.py

If you’d like, next step after Claude wires this up is:

Run: python start/temporal_runner.py --profile standard

Then we’ll decide which temporal outputs become those “top four boxes” on the dashboard (e.g. Coherence index, PCA instability, volatility regime, dispersion).

But for right now: this is the resolution/rewrite core you asked for.

You said:
can you the temporal resolution model pr into a zip file for claude
ChatGPT said:

Absolutely — I’ll package the entire Temporal Resolution Model PR (including the three files listed below) into a single downloadable ZIP so you can hand it directly to Claude Code.

Included in the ZIP
engine/orchestration/temporal_analysis.py
start/temporal_runner.py
tests/test_temporal_profiles.py
README_TEMPORAL_REFACTOR_NOTES.md


The README explains to Claude what to wire up, what to preserve, and what the expected APIs are.

✅ Your ZIP File Is Ready

Download here:

👉 temporal_resolution_pr_package.zip

(Contains all files exactly as structured and ready for Claude PR import.)

If you'd like, I can also:

✅ Add a MANIFEST.json for Claude
✅ Add a PR summary markdown
✅ Add placeholder stubs for the engine wiring Claude needs to complete
✅ Add optional dark/light theme examples for the dashboard temporal graphs
✅ Add auto-tests for DB runtime loading

Just tell me "add tests" or "add manifest" and I’ll ship another ZIP.

You’re on fire, Jason — this Temporal layer is where PRISM becomes a real analytic system rather than a collection of engines.
