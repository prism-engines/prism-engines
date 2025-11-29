"""
Lens Validator - Validate lens accuracy using synthetic data
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .synthetic_data_generator import SyntheticDataGenerator


class LensValidator:
    """
    Validate that lenses correctly identify known patterns.
    """

    def __init__(self, seed: int = 42):
        self.generator = SyntheticDataGenerator(seed)
        self.results: List[Dict] = []

    def validate_granger_lens(self, n_trials: int = 5) -> Dict[str, Any]:
        """
        Validate Granger lens identifies known causal relationships.

        Returns:
            Dictionary with validation results
        """
        from prism_engine.engine.lenses import GrangerLens

        successes = 0

        for _ in range(n_trials):
            # Generate data with known leader
            synth = self.generator.generate_with_known_leader(
                n_indicators=5,
                leader_idx=0
            )

            lens = GrangerLens()
            ranking = lens.rank_indicators(synth["data"])

            # Check if true leader is in top 2
            top_2 = ranking.head(2)["indicator"].tolist()
            if synth["ground_truth"]["leader"] in top_2:
                successes += 1

        accuracy = successes / n_trials

        result = {
            "lens": "granger",
            "test": "identify_leader",
            "n_trials": n_trials,
            "successes": successes,
            "accuracy": accuracy,
            "passed": accuracy >= 0.6
        }

        self.results.append(result)
        return result

    def validate_clustering_lens(self, n_trials: int = 5) -> Dict[str, Any]:
        """
        Validate Clustering lens identifies known clusters.

        Returns:
            Dictionary with validation results
        """
        from prism_engine.engine.lenses import ClusteringLens

        accuracies = []

        for _ in range(n_trials):
            synth = self.generator.generate_with_clusters(
                n_clusters=3,
                indicators_per_cluster=4
            )

            lens = ClusteringLens()
            result = lens.analyze(synth["data"], n_clusters=3)

            # Compare cluster assignments
            true_clusters = synth["ground_truth"]["cluster_assignments"]
            predicted_labels = result["indicator_labels"]

            # Compute adjusted Rand index (simplified)
            matches = 0
            total = 0
            for ind1 in true_clusters:
                for ind2 in true_clusters:
                    if ind1 < ind2:
                        same_true = true_clusters[ind1] == true_clusters[ind2]
                        same_pred = predicted_labels.get(ind1) == predicted_labels.get(ind2)
                        if same_true == same_pred:
                            matches += 1
                        total += 1

            accuracy = matches / total if total > 0 else 0
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)

        result = {
            "lens": "clustering",
            "test": "identify_clusters",
            "n_trials": n_trials,
            "mean_accuracy": mean_accuracy,
            "passed": mean_accuracy >= 0.7
        }

        self.results.append(result)
        return result

    def validate_regime_lens(self, n_trials: int = 5) -> Dict[str, Any]:
        """
        Validate Regime lens identifies known regime switches.

        Returns:
            Dictionary with validation results
        """
        from prism_engine.engine.lenses import RegimeSwitchingLens

        accuracies = []

        for _ in range(n_trials):
            synth = self.generator.generate_with_regimes(
                n_indicators=5,
                regime_lengths=[100, 150, 150]
            )

            lens = RegimeSwitchingLens()
            result = lens.analyze(synth["data"], n_regimes=3)

            # Compare regime assignments
            true_labels = synth["ground_truth"]["regime_labels"]
            pred_labels = result["regime_labels"]

            # Compute accuracy (allowing for label permutation)
            accuracy = self._regime_accuracy(true_labels, pred_labels)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)

        result = {
            "lens": "regime",
            "test": "identify_regimes",
            "n_trials": n_trials,
            "mean_accuracy": mean_accuracy,
            "passed": mean_accuracy >= 0.6
        }

        self.results.append(result)
        return result

    def _regime_accuracy(self, true_labels: List, pred_labels: List) -> float:
        """Compute regime accuracy with label permutation."""
        from itertools import permutations

        true_arr = np.array(true_labels)
        pred_arr = np.array(pred_labels)

        unique_labels = np.unique(pred_arr)
        best_accuracy = 0

        # Try all permutations of label mapping
        for perm in permutations(unique_labels):
            mapping = dict(zip(unique_labels, perm))
            mapped = np.array([mapping[p] for p in pred_arr])
            accuracy = np.mean(true_arr == mapped)
            best_accuracy = max(best_accuracy, accuracy)

        return best_accuracy

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests."""
        self.results = []

        granger_result = self.validate_granger_lens()
        clustering_result = self.validate_clustering_lens()
        regime_result = self.validate_regime_lens()

        n_passed = sum(1 for r in self.results if r["passed"])

        return {
            "n_tests": len(self.results),
            "n_passed": n_passed,
            "pass_rate": n_passed / len(self.results),
            "results": self.results
        }
