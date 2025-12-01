"""
PRISM Engine - Streamlit App
============================
A simple UI for running analysis and sending results to Claude.

To run:
    pip install streamlit anthropic pandas plotly
    streamlit run prism_app.py

Then open http://localhost:8501 in your browser.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Only import anthropic if user wants to use it
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PRISM Engine",
    page_icon="🔷",
    layout="wide"
)

# ============================================================
# SIDEBAR - Configuration
# ============================================================
st.sidebar.title("🔷 PRISM Engine")
st.sidebar.markdown("---")

# API Key input (stored in session, not saved anywhere)
st.sidebar.subheader("🔑 API Configuration")
api_key = st.sidebar.text_input(
    "Anthropic API Key", 
    type="password",
    help="Your key stays in your browser session only"
)

st.sidebar.markdown("---")

# Data selection
st.sidebar.subheader("📊 Data Selection")

indicators = st.sidebar.multiselect(
    "Select Indicators",
    options=[
        "S&P 500 (50d-200d MA)",
        "10-Year Treasury Yield", 
        "DXY (US Dollar Index)",
        "AGG (Bond ETF)",
        "VIX",
        "Gold",
        "Crude Oil"
    ],
    default=["S&P 500 (50d-200d MA)", "10-Year Treasury Yield", "DXY (US Dollar Index)", "AGG (Bond ETF)"]
)

date_range = st.sidebar.slider(
    "Date Range",
    min_value=1970,
    max_value=2025,
    value=(2020, 2025)
)

st.sidebar.markdown("---")

# Analysis options
st.sidebar.subheader("🔬 Analysis Options")

lenses = st.sidebar.multiselect(
    "Select Lenses",
    options=[
        "Correlation Matrix",
        "PCA",
        "Rolling Correlation",
        "Regime Detection",
        "Wavelet Analysis"
    ],
    default=["Correlation Matrix", "Rolling Correlation"]
)


# ============================================================
# MAIN CONTENT
# ============================================================
st.title("🔷 PRISM Engine")
st.markdown("*Multi-lens analysis of time series relationships*")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📈 Data & Charts", "🔬 Analysis", "🤖 Claude Summary"])


# ------------------------------------------------------------
# TAB 1: Data & Charts
# ------------------------------------------------------------
with tab1:
    st.header("Raw Data Preview")
    
    if not indicators:
        st.warning("Select at least one indicator in the sidebar")
    else:
        # Generate sample data (replace with your actual fetcher!)
        @st.cache_data
        def generate_sample_data(indicators, start_year, end_year):
            """
            REPLACE THIS with your actual data fetching logic!
            This just generates fake data for demo purposes.
            """
            dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
            np.random.seed(42)
            
            data = {"Date": dates}
            for i, ind in enumerate(indicators):
                # Generate correlated random walks
                base = np.cumsum(np.random.randn(len(dates)) * 0.5)
                noise = np.random.randn(len(dates)) * 0.2
                data[ind] = 100 + base + noise + i * 10
            
            return pd.DataFrame(data).set_index("Date")
        
        df = generate_sample_data(indicators, date_range[0], date_range[1])
        
        # Show data preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Time Series")
            fig = px.line(df, title="Selected Indicators Over Time")
            fig.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statistics")
            st.dataframe(df.describe().round(2))
        
        # Store in session state for other tabs
        st.session_state["data"] = df


# ------------------------------------------------------------
# TAB 2: Analysis
# ------------------------------------------------------------
with tab2:
    st.header("Lens Analysis")
    
    if "data" not in st.session_state:
        st.warning("Load data first in the Data & Charts tab")
    else:
        df = st.session_state["data"]
        
        # Correlation Matrix
        if "Correlation Matrix" in lenses:
            st.subheader("📊 Correlation Matrix")
            corr = df.corr()
            
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Store for Claude
            st.session_state["correlation"] = corr
        
        # Rolling Correlation
        if "Rolling Correlation" in lenses and len(df.columns) >= 2:
            st.subheader("📈 Rolling Correlation (60-day window)")
            
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Variable 1", df.columns, index=0)
            with col2:
                var2 = st.selectbox("Variable 2", df.columns, index=min(1, len(df.columns)-1))
            
            rolling_corr = df[var1].rolling(60).corr(df[var2])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode="lines",
                name=f"{var1} vs {var2}"
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title=f"Rolling 60-Day Correlation: {var1} vs {var2}",
                yaxis_title="Correlation",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.session_state["rolling_corr_summary"] = f"{var1} vs {var2}: mean={rolling_corr.mean():.3f}, current={rolling_corr.iloc[-1]:.3f}"
        
        # PCA placeholder
        if "PCA" in lenses:
            st.subheader("🔬 PCA Analysis")
            st.info("PCA lens coming soon - plug in your existing code here!")
        
        # Regime Detection placeholder  
        if "Regime Detection" in lenses:
            st.subheader("🎯 Regime Detection")
            st.info("Regime detection lens coming soon - plug in your existing code here!")


# ------------------------------------------------------------
# TAB 3: Claude Summary
# ------------------------------------------------------------
with tab3:
    st.header("🤖 Send to Claude for Analysis")
    
    if not ANTHROPIC_AVAILABLE:
        st.error("Install the anthropic package: `pip install anthropic`")
    elif not api_key:
        st.warning("Enter your Anthropic API key in the sidebar")
    elif "data" not in st.session_state:
        st.warning("Load data first in the Data & Charts tab")
    else:
        # Build context from analysis
        context_parts = []
        
        context_parts.append(f"**Indicators analyzed:** {', '.join(indicators)}")
        context_parts.append(f"**Date range:** {date_range[0]} to {date_range[1]}")
        context_parts.append(f"**Lenses applied:** {', '.join(lenses)}")
        
        if "correlation" in st.session_state:
            corr_str = st.session_state["correlation"].to_string()
            context_parts.append(f"\n**Correlation Matrix:**\n```\n{corr_str}\n```")
        
        if "rolling_corr_summary" in st.session_state:
            context_parts.append(f"\n**Rolling Correlation:** {st.session_state['rolling_corr_summary']}")
        
        analysis_context = "\n".join(context_parts)
        
        st.subheader("Analysis Context (what Claude will see)")
        st.text_area("Context", analysis_context, height=200, disabled=True)
        
        # Custom prompt
        user_prompt = st.text_area(
            "Your question for Claude",
            value="Please analyze these PRISM Engine results. What patterns do you see? Any regime changes or notable relationships?",
            height=100
        )
        
        # Send button
        if st.button("🚀 Send to Claude", type="primary"):
            with st.spinner("Claude is thinking..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    
                    full_prompt = f"""Here are the results from a PRISM Engine analysis:

{analysis_context}

User's question: {user_prompt}"""
                    
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1500,
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    
                    st.subheader("Claude's Analysis")
                    st.markdown(response.content[0].text)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("PRISM Engine v0.1 | Built with Streamlit")
