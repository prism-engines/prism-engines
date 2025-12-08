"""
Sensitivity Analysis Module
===========================

Analyze how robust results are to methodological choices.
Critical for establishing reliability of findings.

Methods:
- Parameter sensitivity: Vary one parameter, measure output change
- Bootstrap sensitivity: Sensitivity to data subsample
- Specification curve: Run all reasonable specifications
- Robustness checks: Systematic exploration of alternative approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from itertools import product


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""
    parameter_name: str
    parameter_values: List[Any]
    output_values: List[float]
    output_mean: float
    output_std: float
    coefficient_of_variation: float
    is_robust: bool  # CV < threshold
    sensitivity_score: float  # Normalized measure of sensitivity


@dataclass
class SpecificationResult:
    """Result from a single specification."""
    spec_id: int
    parameters: Dict[str, Any]
    estimate: float
    significant: bool
    p_value: Optional[float] = None


@dataclass
class SpecificationCurveResult:
    """Result from specification curve analysis."""
    n_specifications: int
    estimates: List[float]
    significant_count: int
    significant_fraction: float
    median_estimate: float
    estimate_range: Tuple[float, float]
    specifications: List[SpecificationResult]
    robust: bool  # Most specifications significant


class ParameterSensitivity:
    """
    Analyze sensitivity of results to parameter choices.
    """

    def __init__(self,
                 analysis_func: Callable,
                 default_params: Dict[str, Any]):
        """
        Initialize sensitivity analyzer.

        Args:
            analysis_func: Function that takes parameters and returns result
            default_params: Default parameter values
        """
        self.analysis_func = analysis_func
        self.default_params = default_params.copy()

    def one_at_a_time(self,
                      param_ranges: Dict[str, List[Any]],
                      output_key: str = 'estimate') -> Dict[str, SensitivityResult]:
        """
        One-at-a-time sensitivity analysis.

        Varies each parameter while holding others at default.

        Args:
            param_ranges: Dictionary of {param_name: [values_to_test]}
            output_key: Key in output dict to track

        Returns:
            Dictionary of sensitivity results per parameter
        """
        results = {}

        for param_name, values in param_ranges.items():
            outputs = []

            for value in values:
                params = self.default_params.copy()
                params[param_name] = value

                try:
                    result = self.analysis_func(**params)
                    if isinstance(result, dict):
                        output = result.get(output_key, result)
                    else:
                        output = result
                    outputs.append(float(output))
                except Exception as e:
                    outputs.append(np.nan)

            outputs = np.array(outputs)
            valid_outputs = outputs[~np.isnan(outputs)]

            if len(valid_outputs) > 0:
                mean_output = np.mean(valid_outputs)
                std_output = np.std(valid_outputs)
                cv = std_output / abs(mean_output) if mean_output != 0 else np.inf
            else:
                mean_output = np.nan
                std_output = np.nan
                cv = np.inf

            # Sensitivity score: normalized range
            if len(valid_outputs) > 1:
                output_range = np.max(valid_outputs) - np.min(valid_outputs)
                sensitivity_score = output_range / (abs(mean_output) + 1e-10)
            else:
                sensitivity_score = 0

            results[param_name] = SensitivityResult(
                parameter_name=param_name,
                parameter_values=list(values),
                output_values=outputs.tolist(),
                output_mean=mean_output,
                output_std=std_output,
                coefficient_of_variation=cv,
                is_robust=cv < 0.2,  # CV < 20% considered robust
                sensitivity_score=sensitivity_score
            )

        return results

    def compute_sensitivity_index(self,
                                   param_ranges: Dict[str, List[Any]],
                                   n_samples: int = 100) -> Dict[str, float]:
        """
        Compute Sobol-like sensitivity indices via sampling.

        Measures fraction of output variance attributable to each parameter.

        Args:
            param_ranges: Parameter ranges
            n_samples: Number of samples

        Returns:
            Dictionary of sensitivity indices
        """
        rng = np.random.RandomState(42)

        # Sample parameter space
        param_names = list(param_ranges.keys())
        samples = []

        for _ in range(n_samples):
            sample = {}
            for name, values in param_ranges.items():
                sample[name] = rng.choice(values)
            samples.append(sample)

        # Compute outputs
        outputs = []
        for sample in samples:
            params = self.default_params.copy()
            params.update(sample)
            try:
                result = self.analysis_func(**params)
                if isinstance(result, dict):
                    outputs.append(result.get('estimate', 0))
                else:
                    outputs.append(float(result))
            except:
                outputs.append(np.nan)

        outputs = np.array(outputs)
        valid_mask = ~np.isnan(outputs)
        outputs = outputs[valid_mask]
        samples = [s for i, s in enumerate(samples) if valid_mask[i]]

        if len(outputs) < 10:
            return {name: 0 for name in param_names}

        total_variance = np.var(outputs)
        if total_variance == 0:
            return {name: 0 for name in param_names}

        # Estimate main effect variance for each parameter
        indices = {}
        for name in param_names:
            # Group outputs by parameter value
            unique_values = list(set(s[name] for s in samples))
            group_means = []

            for val in unique_values:
                group_outputs = [o for o, s in zip(outputs, samples) if s[name] == val]
                if group_outputs:
                    group_means.append(np.mean(group_outputs))

            # Variance of conditional means
            if len(group_means) > 1:
                var_means = np.var(group_means)
                indices[name] = var_means / total_variance
            else:
                indices[name] = 0

        # Normalize to sum to 1
        total_index = sum(indices.values())
        if total_index > 0:
            indices = {k: v / total_index for k, v in indices.items()}

        return indices


class SpecificationCurve:
    """
    Specification curve analysis.

    Run analysis with all reasonable combinations of methodological choices
    to assess robustness.
    """

    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 analysis_func: Callable,
                 specification_options: Dict[str, List[Any]]):
        """
        Initialize specification curve analysis.

        Args:
            data: Data to analyze
            analysis_func: Function(data, **specs) -> {'estimate': float, 'p_value': float}
            specification_options: Dict of {option_name: [possible_values]}
        """
        self.data = data
        self.analysis_func = analysis_func
        self.specification_options = specification_options

    def run(self, alpha: float = 0.05) -> SpecificationCurveResult:
        """
        Run all specifications.

        Args:
            alpha: Significance threshold

        Returns:
            SpecificationCurveResult
        """
        # Generate all combinations
        option_names = list(self.specification_options.keys())
        option_values = list(self.specification_options.values())

        all_combinations = list(product(*option_values))

        specifications = []
        estimates = []
        significant_count = 0

        for i, combo in enumerate(all_combinations):
            params = dict(zip(option_names, combo))

            try:
                result = self.analysis_func(self.data, **params)

                estimate = result.get('estimate', 0)
                p_value = result.get('p_value', 1)
                significant = p_value < alpha if p_value is not None else False

                spec_result = SpecificationResult(
                    spec_id=i,
                    parameters=params,
                    estimate=estimate,
                    significant=significant,
                    p_value=p_value
                )

                specifications.append(spec_result)
                estimates.append(estimate)

                if significant:
                    significant_count += 1

            except Exception as e:
                # Skip failed specifications
                pass

        estimates = np.array(estimates)
        n_specs = len(estimates)

        if n_specs == 0:
            return SpecificationCurveResult(
                n_specifications=0,
                estimates=[],
                significant_count=0,
                significant_fraction=0,
                median_estimate=np.nan,
                estimate_range=(np.nan, np.nan),
                specifications=[],
                robust=False
            )

        return SpecificationCurveResult(
            n_specifications=n_specs,
            estimates=estimates.tolist(),
            significant_count=significant_count,
            significant_fraction=significant_count / n_specs,
            median_estimate=float(np.median(estimates)),
            estimate_range=(float(np.min(estimates)), float(np.max(estimates))),
            specifications=specifications,
            robust=significant_count / n_specs > 0.5  # >50% significant
        )

    def summary(self, result: SpecificationCurveResult) -> Dict[str, Any]:
        """Generate summary of specification curve."""
        estimates = np.array(result.estimates)

        return {
            'n_specifications': result.n_specifications,
            'median_estimate': result.median_estimate,
            'mean_estimate': float(np.mean(estimates)) if len(estimates) > 0 else np.nan,
            'std_estimate': float(np.std(estimates)) if len(estimates) > 0 else np.nan,
            'range': result.estimate_range,
            'significant_fraction': result.significant_fraction,
            'robust': result.robust,
            'interpretation': (
                "Results are ROBUST across specifications"
                if result.robust
                else "Results are SENSITIVE to specification choices"
            )
        }


class BootstrapSensitivity:
    """
    Assess sensitivity of results to data sampling.
    """

    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame],
                 analysis_func: Callable,
                 n_bootstrap: int = 500,
                 random_seed: Optional[int] = None):
        """
        Initialize bootstrap sensitivity analysis.

        Args:
            data: Original data
            analysis_func: Analysis function
            n_bootstrap: Number of bootstrap samples
            random_seed: For reproducibility
        """
        self.data = data
        self.analysis_func = analysis_func
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_seed)

    def run(self) -> Dict[str, Any]:
        """
        Run bootstrap sensitivity analysis.

        Returns:
            Dictionary with bootstrap results
        """
        if isinstance(self.data, pd.DataFrame):
            n = len(self.data)
            is_df = True
        else:
            n = len(self.data)
            is_df = False

        # Original estimate
        original_result = self.analysis_func(self.data)
        if isinstance(original_result, dict):
            original_estimate = original_result.get('estimate', 0)
        else:
            original_estimate = float(original_result)

        # Bootstrap estimates
        bootstrap_estimates = []

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = self.rng.randint(0, n, size=n)

            if is_df:
                sample = self.data.iloc[indices]
            else:
                sample = self.data[indices]

            try:
                result = self.analysis_func(sample)
                if isinstance(result, dict):
                    bootstrap_estimates.append(result.get('estimate', 0))
                else:
                    bootstrap_estimates.append(float(result))
            except:
                pass

        bootstrap_estimates = np.array(bootstrap_estimates)

        if len(bootstrap_estimates) == 0:
            return {
                'original_estimate': original_estimate,
                'bootstrap_mean': np.nan,
                'bootstrap_std': np.nan,
                'coefficient_of_variation': np.nan,
                'ci_95': (np.nan, np.nan),
                'robust': False
            }

        mean_est = np.mean(bootstrap_estimates)
        std_est = np.std(bootstrap_estimates)
        cv = std_est / abs(mean_est) if mean_est != 0 else np.inf

        return {
            'original_estimate': original_estimate,
            'bootstrap_mean': float(mean_est),
            'bootstrap_std': float(std_est),
            'coefficient_of_variation': float(cv),
            'ci_95': (
                float(np.percentile(bootstrap_estimates, 2.5)),
                float(np.percentile(bootstrap_estimates, 97.5))
            ),
            'robust': cv < 0.3,  # CV < 30%
            'bias': float(mean_est - original_estimate),
            'n_bootstrap': len(bootstrap_estimates)
        }


def run_sensitivity_suite(data: Union[np.ndarray, pd.DataFrame],
                          analysis_func: Callable,
                          param_ranges: Dict[str, List[Any]],
                          default_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run comprehensive sensitivity analysis.

    Args:
        data: Data to analyze
        analysis_func: Analysis function
        param_ranges: Parameter ranges to test
        default_params: Default parameter values

    Returns:
        Dictionary with all sensitivity results
    """
    if default_params is None:
        # Use first value of each range as default
        default_params = {k: v[0] for k, v in param_ranges.items()}

    # Wrap analysis function to include data
    def wrapped_func(**params):
        return analysis_func(data, **params)

    # Parameter sensitivity
    param_sens = ParameterSensitivity(wrapped_func, default_params)
    oat_results = param_sens.one_at_a_time(param_ranges)
    sensitivity_indices = param_sens.compute_sensitivity_index(param_ranges)

    # Specification curve
    spec_curve = SpecificationCurve(data, analysis_func, param_ranges)
    spec_result = spec_curve.run()
    spec_summary = spec_curve.summary(spec_result)

    # Bootstrap sensitivity
    boot_sens = BootstrapSensitivity(data, lambda d: analysis_func(d, **default_params))
    boot_result = boot_sens.run()

    # Overall robustness assessment
    param_robust = all(r.is_robust for r in oat_results.values())
    spec_robust = spec_result.robust
    boot_robust = boot_result.get('robust', False)

    return {
        'parameter_sensitivity': {
            name: {
                'values': result.output_values,
                'cv': result.coefficient_of_variation,
                'robust': result.is_robust
            }
            for name, result in oat_results.items()
        },
        'sensitivity_indices': sensitivity_indices,
        'specification_curve': spec_summary,
        'bootstrap_sensitivity': boot_result,
        'overall_robust': param_robust and spec_robust and boot_robust,
        'robustness_summary': {
            'parameter_robust': param_robust,
            'specification_robust': spec_robust,
            'bootstrap_robust': boot_robust
        }
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Sensitivity Analysis Module - Test")
    print("=" * 60)

    np.random.seed(42)

    # Create test data
    n = 200
    data = np.random.randn(n) * 2 + 5

    # Simple analysis function with parameters
    def analysis_func(data, window=20, method='mean'):
        if method == 'mean':
            result = np.mean(data[-window:])
        elif method == 'median':
            result = np.median(data[-window:])
        else:
            result = np.mean(data[-window:])
        return {'estimate': result, 'p_value': 0.01}

    print("\nParameter sensitivity (one-at-a-time):")
    print("-" * 60)

    default_params = {'window': 20, 'method': 'mean'}
    param_ranges = {
        'window': [10, 20, 30, 50, 100],
        'method': ['mean', 'median']
    }

    param_sens = ParameterSensitivity(
        lambda **p: analysis_func(data, **p),
        default_params
    )
    oat_results = param_sens.one_at_a_time(param_ranges)

    for name, result in oat_results.items():
        print(f"\n{name}:")
        print(f"  Values: {result.parameter_values}")
        print(f"  Outputs: {[round(v, 3) for v in result.output_values]}")
        print(f"  CV: {result.coefficient_of_variation:.3f}")
        print(f"  Robust: {result.is_robust}")

    print("\n" + "=" * 60)
    print("Specification curve:")
    print("-" * 60)

    spec_curve = SpecificationCurve(data, analysis_func, param_ranges)
    spec_result = spec_curve.run()
    spec_summary = spec_curve.summary(spec_result)

    print(f"Number of specifications: {spec_summary['n_specifications']}")
    print(f"Median estimate: {spec_summary['median_estimate']:.3f}")
    print(f"Range: {spec_summary['range']}")
    print(f"Significant fraction: {spec_summary['significant_fraction']:.1%}")
    print(f"Robust: {spec_summary['robust']}")

    print("\n" + "=" * 60)
    print("Bootstrap sensitivity:")
    print("-" * 60)

    boot_sens = BootstrapSensitivity(data, lambda d: analysis_func(d, **default_params))
    boot_result = boot_sens.run()

    print(f"Original estimate: {boot_result['original_estimate']:.3f}")
    print(f"Bootstrap mean: {boot_result['bootstrap_mean']:.3f}")
    print(f"Bootstrap std: {boot_result['bootstrap_std']:.3f}")
    print(f"95% CI: {boot_result['ci_95']}")
    print(f"Robust: {boot_result['robust']}")

    print("\n" + "=" * 60)
    print("Test completed!")
