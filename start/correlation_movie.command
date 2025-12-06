#!/bin/bash
# Double-click launcher for correlation_movie.py
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
echo "Running correlation_movie.py..."
echo "========================================"

python "start/correlation_movie.py"

echo ""
echo "========================================"
echo "Done! Press any key to close."
read -n 1
