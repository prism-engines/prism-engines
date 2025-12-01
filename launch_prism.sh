#!/bin/bash
# ============================================================
# PRISM Engine Launcher (Linux/Mac)
# ============================================================
# Double-click this file or run: ./launch_prism.sh
# 
# Make it executable first (one time):
#   chmod +x launch_prism.sh
# ============================================================

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  PRISM Engine Launcher"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Activate virtual environment
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "✓ Activating venv..."
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "  Python: $(which python)"
    echo "  Version: $(python --version)"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "✓ Activating .venv..."
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "  Python: $(which python)"
else
    echo "⚠ No venv found! Creating one..."
    python3 -m venv "$PROJECT_ROOT/venv"
    source "$PROJECT_ROOT/venv/bin/activate"
    
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
fi

echo ""
echo "========================================"
echo "  Ready! Choose an option:"
echo "========================================"
echo "  1) Open Python REPL"
echo "  2) Run Jupyter Notebook"
echo "  3) Run main.py"
echo "  4) Just open terminal here"
echo "  q) Quit"
echo ""

read -p "Choice: " choice

case $choice in
    1) python ;;
    2) jupyter notebook ;;
    3) python "$PROJECT_ROOT/main.py" ;;
    4) cd "$PROJECT_ROOT" && exec $SHELL ;;
    q) echo "Goodbye!" ;;
    *) echo "Opening terminal..." && cd "$PROJECT_ROOT" && exec $SHELL ;;
esac
