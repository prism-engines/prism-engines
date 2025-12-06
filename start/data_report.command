#!/bin/bash
# Double-click launcher for data_report.py
# Auto-generated - do not edit

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "prism-mac-venv" ]; then
    source prism-mac-venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "========================================"
echo "Running data_report.py..."
echo "========================================"

python "start/data_report.py"

echo ""
echo "========================================"
echo "Done! Press any key to close."
read -n 1
