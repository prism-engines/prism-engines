"""
PRISM Engines - Quick Start Example

This example demonstrates the basic usage of prism-engines.
"""

import prism_engines as prism

# Load and analyze data in one line
results = prism.run("sample_data.csv")

# Print the analysis report
results.print_report()

# Access individual engine results
pca_result = results["pca"]
print(f"\nGlobal Forcing Metric: {pca_result.metrics['global_forcing_metric']:.3f}")
print(f"Effective Dimension: {pca_result.metrics['effective_dimension']}")

# Check correlations
corr_result = results["correlation"]
print(f"\nMean Absolute Correlation: {corr_result.metrics['mean_abs_correlation']:.3f}")
max_corr = corr_result.metrics["max_correlation"]
print(f"Strongest Pair: {max_corr['pair']} = {max_corr['value']:.3f}")

# Check persistence (Hurst exponents)
hurst_result = results["hurst"]
print(f"\nMean Hurst Exponent: {hurst_result.metrics['mean_hurst']:.3f}")

# Generate visualization (requires matplotlib)
try:
    fig = results.plot()
    fig.savefig("analysis_plot.png", dpi=150)
    print("\nPlot saved to: analysis_plot.png")
except ImportError:
    print("\nInstall matplotlib for visualization: pip install matplotlib")

# Save full report
results.save("output", name="my_analysis")
