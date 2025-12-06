#!/bin/bash
# Double-click launcher for lens_contributions.py
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
echo "Running lens_contributions.py..."
echo "========================================"

python "start/lens_contributions.py"

echo ""
echo "========================================"
echo "Done! Press any key to close."
read -n 1
