"""
PRISM Observatory - Geometric Analysis for Time Series

A consumer-friendly interface for prism-engines.
Deploy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PRISM Observatory",
    page_icon="🔬",
    layout="wide",
)

# Header
st.title("🔬 PRISM Observatory")
st.markdown("*Geometric analysis for time series data*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **PRISM** reveals hidden geometric structure in your data:

    - **Correlation** - How series move together
    - **PCA** - Dominant patterns & dimensionality
    - **Hurst** - Memory & persistence

    Upload a CSV to begin.
    """)

    st.markdown("---")
    st.markdown("**CSV Format**")
    st.code("date,series1,series2,...\n2024-01-01,100,200,...", language="text")

    st.markdown("---")
    st.markdown("[GitHub](https://github.com/prism-engines/prism-engines) · [Docs](#)")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="First column should be dates, remaining columns are numeric series"
    )

with col2:
    use_sample = st.checkbox("Use sample data instead")

# Load data
df = None

if use_sample:
    # Generate sample data
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    market = np.cumsum(np.random.randn(n) * 0.02)

    df = pd.DataFrame({
        'SPY': market + np.cumsum(np.random.randn(n) * 0.01),
        'QQQ': market * 1.2 + np.cumsum(np.random.randn(n) * 0.015),
        'IWM': market * 0.8 + np.cumsum(np.random.randn(n) * 0.02),
        'TLT': -market * 0.3 + np.cumsum(np.random.randn(n) * 0.01),
        'GLD': np.cumsum(np.random.randn(n) * 0.008),
    }, index=dates)

    st.success("Using sample market data (SPY, QQQ, IWM, TLT, GLD)")

elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1, how='all').ffill().bfill()
        st.success(f"Loaded {len(df)} rows × {len(df.columns)} series")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Run analysis
if df is not None and len(df.columns) >= 2:

    st.markdown("---")
    st.header("📊 Analysis Results")

    # Import prism_engines
    try:
        import prism_engines as prism
        results = prism.run(df)
    except ImportError:
        st.error("prism-engines not installed. Run: pip install prism-engines")
        st.stop()

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Correlation", "PCA", "Persistence"])

    # Tab 1: Summary
    with tab1:
        col1, col2, col3 = st.columns(3)

        pca_result = results["pca"]
        corr_result = results["correlation"]
        hurst_result = results["hurst"]

        with col1:
            st.metric(
                "Global Forcing",
                f"{pca_result.metrics['global_forcing_metric']:.1%}",
                help="How much variance is explained by the dominant mode (PC1)"
            )

        with col2:
            st.metric(
                "Effective Dimension",
                f"{pca_result.metrics['effective_dimension']}",
                help="Number of principal components needed for 90% variance"
            )

        with col3:
            mean_corr = corr_result.metrics['mean_abs_correlation']
            st.metric(
                "Mean Correlation",
                f"{mean_corr:.2f}",
                help="Average absolute correlation between all pairs"
            )

        # Interpretation
        st.markdown("### Interpretation")

        gfm = pca_result.metrics['global_forcing_metric']
        eff_dim = pca_result.metrics['effective_dimension']

        if gfm > 0.6:
            st.info(f"🌐 **Strong shared driver**: {gfm:.0%} of movement is coordinated across all series.")
        elif gfm > 0.4:
            st.info(f"📊 **Moderate structure**: {gfm:.0%} shared variance, with {eff_dim} distinct patterns.")
        else:
            st.info(f"🎲 **Diverse dynamics**: Only {gfm:.0%} shared variance. Series move independently.")

        mean_hurst = hurst_result.metrics.get('mean_hurst')
        if mean_hurst:
            if mean_hurst > 0.6:
                st.success(f"📈 **Persistent/trending**: Mean Hurst = {mean_hurst:.2f}")
            elif mean_hurst < 0.4:
                st.warning(f"↩️ **Mean-reverting**: Mean Hurst = {mean_hurst:.2f}")
            else:
                st.info(f"🎯 **Random walk**: Mean Hurst = {mean_hurst:.2f}")

    # Tab 2: Correlation
    with tab2:
        import matplotlib.pyplot as plt

        corr_matrix = np.array(corr_result.metrics['correlation_matrix'])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

        # Labels
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        ax.set_yticklabels(df.columns)

        # Colorbar
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('Correlation Matrix')

        plt.tight_layout()
        st.pyplot(fig)

        # Top correlations
        st.markdown("**Strongest Correlations**")
        max_corr = corr_result.metrics['max_correlation']
        min_corr = corr_result.metrics['min_correlation']
        st.write(f"- Highest: {max_corr['pair'][0]} ↔ {max_corr['pair'][1]}: **{max_corr['value']:.3f}**")
        st.write(f"- Lowest: {min_corr['pair'][0]} ↔ {min_corr['pair'][1]}: **{min_corr['value']:.3f}**")

    # Tab 3: PCA
    with tab3:
        var_ratio = pca_result.metrics['explained_variance_ratio']
        cumulative = pca_result.metrics['cumulative_variance']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Scree plot
        ax1.bar(range(1, len(var_ratio) + 1), var_ratio, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained')
        ax1.set_title('Scree Plot')
        ax1.set_xticks(range(1, len(var_ratio) + 1))

        # Cumulative
        ax2.plot(range(1, len(cumulative) + 1), cumulative, 'o-', color='steelblue')
        ax2.axhline(0.9, color='red', linestyle='--', label='90% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance')
        ax2.set_title('Cumulative Variance Explained')
        ax2.set_xticks(range(1, len(cumulative) + 1))
        ax2.legend()
        ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        st.pyplot(fig)

        # PC1 loadings
        st.markdown("**PC1 Loadings** (contribution to dominant mode)")
        loadings = pca_result.metrics['pc1_loadings']
        loading_df = pd.DataFrame({
            'Series': loadings.keys(),
            'Loading': loadings.values()
        }).sort_values('Loading', key=abs, ascending=False)
        st.dataframe(loading_df, hide_index=True)

    # Tab 4: Persistence (Hurst)
    with tab4:
        hurst_vals = hurst_result.metrics['hurst_exponents']
        classifications = hurst_result.metrics['persistence_classification']

        # Filter out None values
        valid_hurst = {k: v for k, v in hurst_vals.items() if v is not None}

        if valid_hurst:
            fig, ax = plt.subplots(figsize=(10, 6))

            colors = []
            for series in valid_hurst.keys():
                h = valid_hurst[series]
                if h < 0.4:
                    colors.append('salmon')
                elif h > 0.6:
                    colors.append('lightgreen')
                else:
                    colors.append('lightgray')

            bars = ax.barh(list(valid_hurst.keys()), list(valid_hurst.values()), color=colors)
            ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Random Walk (H=0.5)')
            ax.axvline(0.4, color='salmon', linestyle=':', alpha=0.7)
            ax.axvline(0.6, color='lightgreen', linestyle=':', alpha=0.7)
            ax.set_xlabel('Hurst Exponent')
            ax.set_title('Persistence Analysis')
            ax.set_xlim(0, 1)
            ax.legend()

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            **Legend:**
            - 🟢 Green (H > 0.6): Persistent/trending
            - ⚪ Gray (0.4 < H < 0.6): Random walk
            - 🔴 Red (H < 0.4): Mean-reverting
            """)
        else:
            st.warning("Insufficient data for Hurst analysis")

    # Download section
    st.markdown("---")
    st.header("📥 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        report_text = results.report()
        st.download_button(
            label="Download Report (TXT)",
            data=report_text,
            file_name="prism_report.txt",
            mime="text/plain"
        )

    with col2:
        # Save plot to bytes
        import io
        fig = results.plot()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="Download Charts (PNG)",
            data=buf,
            file_name="prism_charts.png",
            mime="image/png"
        )

elif df is not None:
    st.warning("Need at least 2 numeric columns for analysis.")

else:
    # Show placeholder
    st.info("👆 Upload a CSV file or check 'Use sample data' to begin analysis.")

    with st.expander("What kind of data works best?"):
        st.markdown("""
        PRISM works with any multivariate time series:

        - **Financial**: Stock prices, indices, rates
        - **Economic**: GDP, inflation, employment
        - **Scientific**: Sensor data, measurements
        - **Business**: Sales, metrics, KPIs

        **Requirements:**
        - First column: dates (YYYY-MM-DD)
        - Other columns: numeric values
        - At least 2 series, 10+ data points
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "PRISM Observatory · Powered by prism-engines · "
    "<a href='https://prismobservatory.com'>prismobservatory.com</a>"
    "</div>",
    unsafe_allow_html=True
)
