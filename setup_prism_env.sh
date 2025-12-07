#!/usr/bin/env bash
set -e

echo "============================================="
echo "     PRISM ENGINE – Environment Setup        "
echo "============================================="

VENV_DIR="$HOME/venvs/prism-mac-venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists. Updating..."
fi

echo "Activating venv..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing PRISM core dependencies..."
pip install pandas numpy scipy scikit-learn \
            matplotlib seaborn tqdm pyarrow \
            python-dateutil fredapi requests pyyaml

echo "Installing Tiingo dependencies..."
pip install requests

echo "Installing climate-ready libraries (future use only)..."
pip install cdsapi xarray netcdf4 cfgrib

ACTIVATE_FILE="$VENV_DIR/bin/activate"

echo "Injecting API KEYS into venv activate script..."
grep -qxF 'export FRED_API_KEY="3fd12c9d0fa4d7fd3c858b72251e3388"' "$ACTIVATE_FILE" || \
echo 'export FRED_API_KEY="3fd12c9d0fa4d7fd3c858b72251e3388"' >> "$ACTIVATE_FILE"

grep -qxF 'export TIINGO_API_KEY="39a87d4f616a7432dbc92533eeed14c238f6d159"' "$ACTIVATE_FILE" || \
echo 'export TIINGO_API_KEY="39a87d4f616a7432dbc92533eeed14c238f6d159"' >> "$ACTIVATE_FILE"

echo "Testing environment..."
python - <<EOF
import fredapi, cdsapi, xarray
print("PRISM environment OK")
EOF

echo "============================================="
echo "   PRISM ENGINE ENVIRONMENT READY            "
echo "============================================="
