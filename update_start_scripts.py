#!/usr/bin/env python3
"""
Update All Start Scripts
========================

1. Replaces hardcoded paths with output_config imports
2. Adds double-click support (cd to project dir, activate venv)

Run from project root:
    python update_start_scripts.py
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
START_DIR = PROJECT_ROOT / "start"

# The new import block to add
NEW_IMPORTS = '''import os
import sys

# Double-click support: ensure we're in the right directory with right path
if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    sys.path.insert(0, str(Path(__file__).parent.parent))

from output_config import OUTPUT_DIR, DATA_DIR
'''

# Path replacements
REPLACEMENTS = [
    # DB paths
    (r'DB_PATH\s*=\s*Path\.home\(\)\s*/\s*["\']prism_data["\']\s*/\s*["\']prism\.db["\']', 
     'DB_PATH = DATA_DIR / "prism.db"'),
    (r'DB_PATH\s*=\s*Path\.home\(\)\s*/\s*["\']prism_data["\']\s*/\s*["\']climate\.db["\']',
     'DB_PATH = DATA_DIR / "climate.db"'),
    
    # Panel paths  
    (r'PANEL_PATH\s*=\s*Path\.home\(\)\s*/\s*["\']prism_data["\']\s*/\s*["\']climate_panel\.csv["\']',
     'PANEL_PATH = DATA_DIR / "climate_panel.csv"'),
    
    # Output dirs
    (r'OUTPUT_DIR\s*=\s*Path\.home\(\)\s*/\s*["\']prism-engine["\']\s*/\s*["\']output["\'].*',
     'OUTPUT_DIR = OUTPUT_DIR'),  # Uses the imported one
    (r'OUTPUT_DIR\s*=\s*Path\.home\(\)\s*/\s*["\']prism_data["\']\s*/\s*["\']climate_output["\']',
     'CLIMATE_OUTPUT_DIR = OUTPUT_DIR / "climate"'),
    (r'OUTPUT_DIR\s*=\s*PROJECT_ROOT\s*/\s*["\']output["\'].*',
     '# OUTPUT_DIR imported from output_config'),
     
    # Generic prism_data references
    (r'Path\.home\(\)\s*/\s*["\']prism_data["\']', 'DATA_DIR'),
    (r'Path\.home\(\)\s*/\s*["\']prism-engine["\']\s*/\s*["\']output["\']', 'OUTPUT_DIR'),
]


def update_script(script_path: Path) -> bool:
    """Update a single script. Returns True if changed."""
    
    content = script_path.read_text()
    original = content
    
    # Check if already updated
    if "from output_config import" in content:
        print(f"   ⏭️  Already updated: {script_path.name}")
        return False
    
    # Check if it uses paths we need to replace
    needs_update = any(
        "prism_data" in content or 
        "prism-engine/output" in content or
        'Path.home()' in content and 'prism' in content.lower()
        for _ in [1]
    )
    
    if not needs_update:
        print(f"   ⏭️  No paths to update: {script_path.name}")
        return False
    
    # Add the new imports after existing imports
    # Find the last import line
    lines = content.split('\n')
    last_import_idx = 0
    has_pathlib = False
    
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_idx = i
            if 'pathlib' in line:
                has_pathlib = True
    
    # Build new import section
    new_import_lines = []
    if not has_pathlib:
        new_import_lines.append("from pathlib import Path")
    
    # Add our imports after the last import
    import_block = """
# === PRISM PATH CONFIG ===
import os
import sys
# Double-click support: ensure correct directory and path
if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
from output_config import OUTPUT_DIR, DATA_DIR
# === END PATH CONFIG ===
"""
    
    # Insert after last import
    lines.insert(last_import_idx + 1, import_block)
    content = '\n'.join(lines)
    
    # Apply path replacements
    for pattern, replacement in REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    
    # Fix any double OUTPUT_DIR issues
    content = content.replace("OUTPUT_DIR = OUTPUT_DIR", "# OUTPUT_DIR imported from output_config")
    
    # Write back
    script_path.write_text(content)
    print(f"   ✅ Updated: {script_path.name}")
    return True


def create_launcher(script_path: Path):
    """Create a .command file for double-click launching on Mac."""
    
    launcher_path = script_path.with_suffix('.command')
    script_name = script_path.name
    
    launcher_content = f'''#!/bin/bash
# Double-click launcher for {script_name}
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
echo "Running {script_name}..."
echo "========================================"

python "start/{script_name}"

echo ""
echo "========================================"
echo "Done! Press any key to close."
read -n 1
'''
    
    launcher_path.write_text(launcher_content)
    launcher_path.chmod(0o755)  # Make executable
    print(f"   🖱️  Created launcher: {launcher_path.name}")


def main():
    print("=" * 60)
    print("🔧 UPDATING START SCRIPTS")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Start dir: {START_DIR}")
    
    # Get all Python scripts in start/
    scripts = sorted(START_DIR.glob("*.py"))
    
    print(f"\nFound {len(scripts)} scripts\n")
    
    updated_count = 0
    
    for script in scripts:
        if script.name.startswith('__'):
            continue
            
        changed = update_script(script)
        if changed:
            updated_count += 1
            create_launcher(script)
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE: Updated {updated_count} scripts")
    print("=" * 60)
    
    print("\nNext steps:")
    print("   1. Test a script: python start/overnight_lite.py")
    print("   2. Or double-click: start/overnight_lite.command")
    print("   3. Commit: git add -A && git commit -m 'Add cross-platform paths and launchers'")


if __name__ == "__main__":
    main()
