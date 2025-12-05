# PRISM Engine App

A Streamlit UI for running PRISM analysis and sending results to Claude.

## Quick Start

```bash
# 1. Create/activate your venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run prism_app.py
```

Then open http://localhost:8501 in your browser.

## What It Does

1. **Sidebar** - Configure your indicators, date range, and analysis lenses
2. **Data & Charts tab** - View your time series and basic stats
3. **Analysis tab** - Run correlation matrix, rolling correlation, etc.
4. **Claude Summary tab** - Send your results to Claude API for interpretation

## Customization Points

Look for these sections in `prism_app.py`:

- `generate_sample_data()` - Replace with your actual data fetcher
- The `lenses` section - Plug in your existing PRISM lens code
- The Claude prompt - Customize what context gets sent

## Getting an API Key

1. Go to https://console.anthropic.com
2. Create an account / sign in
3. Generate an API key
4. Paste it in the sidebar (it's never saved, just lives in your browser session)

## Next Steps

- [ ] Connect your real data fetchers
- [ ] Add your existing lens implementations
- [ ] Save/load configurations
- [ ] Export reports to PDF
