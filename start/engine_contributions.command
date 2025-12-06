#!/bin/bash
# Double-click launcher for engine_contributions.py
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
echo "Running engine_contributions.py..."
echo "========================================"

python "start/engine_contributions.py"

echo ""
echo "========================================"
echo "Done! Press any key to close."
read -n 1
