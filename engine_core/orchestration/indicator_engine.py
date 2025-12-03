"""
Indicator Engine - High-level API for analysis
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

from .lens_comparator import LensComparator
from .consensus import ConsensusEngine

logger = logging.getLogger(__name__)

# Get project root (engine_core/orchestration -> project root)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent


class IndicatorEngine:
    """
    High-level interface for indicator analysis.

    Provides a simple API to:
    1. Run multiple lenses
    2. Build consensus
    3. Generate reports
    """

    # Default lens configurations
    BASIC_LENSES = ['magnitude', 'pca', 'influence', 'clustering', 'decomposition']
    ADVANCED_LENSES = ['granger', 'mutual_info', 'wavelet', 'network', 'regime', 'anomaly']
    ALL_LENSES = BASIC_LENSES + ADVANCED_LENSES + ['transfer_entropy', 'tda', 'dmd']

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the indicator engine.

        Args:
            output_dir: Directory for outputs
            config: Configuration dictionary
        """
        self.output_dir = Path(output_dir) if output_dir else (_PROJECT_ROOT / "output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.comparator = LensComparator()
        self.consensus_engine: Optional[ConsensusEngine] = None
        self.last_results: Optional[Dict] = None

    def analyze(
        self,
        df: pd.DataFrame,
        mode: str = "basic",
        lenses: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run full analysis.

        Args:
            df: Input DataFrame with indicators
            mode: 'basic', 'advanced', or 'full'
            lenses: Custom list of lenses (overrides mode)
            **kwargs: Parameters passed to lenses

        Returns:
            Dictionary with complete analysis results
        """
        # Select lenses
        if lenses is not None:
            selected_lenses = lenses
        elif mode == "basic":
            selected_lenses = self.BASIC_LENSES
        elif mode == "advanced":
            selected_lenses = self.BASIC_LENSES + self.ADVANCED_LENSES
        elif mode == "full":
            selected_lenses = self.ALL_LENSES
        else:
            selected_lenses = self.BASIC_LENSES

        logger.info(f"Running analysis with {len(selected_lenses)} lenses...")

        # Run lenses
        lens_results = self.comparator.run_lenses(df, selected_lenses, **kwargs)

        # Build consensus
        if self.comparator.rankings:
            self.consensus_engine = ConsensusEngine(self.comparator.rankings)
            consensus = self.consensus_engine.full_consensus()
        else:
            consensus = {"error": "No valid rankings to build consensus"}

        # Compare rankings
        comparison = self.comparator.compare_rankings()
        agreement = self.comparator.get_agreement_matrix()

        # Build final results
        results = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "n_indicators": len(df.columns) - 1,  # Excluding date
            "n_observations": len(df),
            "lenses_run": selected_lenses,
            "lens_results": lens_results,
            "consensus": consensus,
            "ranking_comparison": comparison.to_dict() if not comparison.empty else {},
            "lens_agreement": agreement.to_dict() if not agreement.empty else {},
            "top_indicators": self._extract_top_indicators(consensus),
        }

        self.last_results = results
        return results

    def _extract_top_indicators(
        self,
        consensus: Dict,
        top_n: int = 10
    ) -> List[Dict]:
        """Extract top indicators with details."""
        if "full_ranking" not in consensus:
            return []

        ranking = consensus["full_ranking"]
        top = ranking.head(top_n)

        return [
            {
                "indicator": row["indicator"],
                "rank": int(row["consensus_rank"]),
                "confidence": float(row["confidence"]),
                "votes": int(row["votes"]),
            }
            for _, row in top.iterrows()
        ]

    def get_indicator_report(self, indicator: str) -> Dict[str, Any]:
        """
        Get detailed report for a specific indicator.

        Args:
            indicator: Indicator name

        Returns:
            Dictionary with per-lens scores and analysis
        """
        if not self.last_results:
            return {"error": "Run analyze() first"}

        report = {
            "indicator": indicator,
            "lens_scores": {},
            "lens_ranks": {},
        }

        for lens_name, ranking_df in self.comparator.rankings.items():
            match = ranking_df[ranking_df['indicator'] == indicator]
            if len(match) > 0:
                report["lens_scores"][lens_name] = float(match['score'].iloc[0])
                report["lens_ranks"][lens_name] = int(match['rank'].iloc[0])

        # Add consensus info
        if self.consensus_engine:
            full_ranking = self.consensus_engine.full_consensus()["full_ranking"]
            match = full_ranking[full_ranking['indicator'] == indicator]
            if len(match) > 0:
                report["consensus_rank"] = int(match['consensus_rank'].iloc[0])
                report["confidence"] = float(match['confidence'].iloc[0])

        return report

    def save_results(self, name: Optional[str] = None) -> Path:
        """
        Save analysis results.

        Args:
            name: Optional custom name

        Returns:
            Path to saved results
        """
        if not self.last_results:
            raise ValueError("No results to save. Run analyze() first.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or "analysis"

        save_dir = self.output_dir / "latest"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        results_path = save_dir / "run_summary.json"

        # Convert DataFrames to dicts for JSON serialization
        results_json = self._make_json_serializable(self.last_results)

        with open(results_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)

        # Save rankings
        if self.consensus_engine:
            consensus = self.consensus_engine.full_consensus()
            if "full_ranking" in consensus:
                consensus["full_ranking"].to_csv(
                    save_dir / "consensus_indicators.csv",
                    index=False
                )

        # Save lens comparison
        comparison = self.comparator.compare_rankings()
        if not comparison.empty:
            comparison.to_csv(save_dir / "lens_comparison.csv", index=False)

        # Save agreement matrix
        agreement = self.comparator.get_agreement_matrix()
        if not agreement.empty:
            agreement.to_csv(save_dir / "agreement_matrix.csv")

        # Archive with timestamp
        archive_dir = self.output_dir / "archive" / f"{name}_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        for file in save_dir.glob("*"):
            shutil.copy(file, archive_dir)

        logger.info(f"Results saved to {save_dir}")
        return save_dir

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def quick_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quick analysis returning just the top indicators.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with top 20 indicators and consensus scores
        """
        results = self.analyze(df, mode="basic")

        if "consensus" not in results or "full_ranking" not in results["consensus"]:
            return pd.DataFrame()

        return results["consensus"]["full_ranking"].head(20)
