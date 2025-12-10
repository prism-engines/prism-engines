"""
Lens Inspector - Run this and paste the output back to Claude
"""

import os
import ast
from pathlib import Path

def find_project_root():
    """Find the PRISM project root."""
    # Try common locations
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.home() / 'prism-engine',
        Path.home() / 'Projects' / 'prism-engine',
    ]
    
    for path in candidates:
        if (path / 'engine_core').exists() or (path / 'lenses').exists():
            return path
    
    return Path.cwd()

def find_lens_folders(root: Path):
    """Find folders that might contain lenses."""
    lens_paths = []
    
    search_patterns = [
        'engine_core/lenses',
        'lenses',
        'src/lenses',
        'prism/lenses',
    ]
    
    for pattern in search_patterns:
        path = root / pattern
        if path.exists():
            lens_paths.append(path)
    
    # Also search for any folder named 'lenses'
    for path in root.rglob('lenses'):
        if path.is_dir() and 'venv' not in str(path) and '__pycache__' not in str(path):
            if path not in lens_paths:
                lens_paths.append(path)
    
    return lens_paths

def extract_class_info(filepath: Path):
    """Extract class and method info from a Python file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                classes.append({
                    'name': node.name,
                    'methods': methods
                })
        
        return classes, content[:2000]  # Return first 2000 chars too
        
    except Exception as e:
        return [], f"Error parsing: {e}"

def main():
    print("=" * 70)
    print("PRISM LENS INSPECTOR")
    print("=" * 70)
    
    root = find_project_root()
    print(f"\nProject root: {root}")
    
    # Find lens folders
    lens_folders = find_lens_folders(root)
    
    if not lens_folders:
        print("\n❌ No lens folders found!")
        print("Searched in:", root)
        print("\nTrying to list project structure...")
        for item in sorted(root.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                print(f"  📁 {item.name}/")
        return
    
    print(f"\n✓ Found {len(lens_folders)} lens folder(s):")
    
    for folder in lens_folders:
        print(f"\n{'='*70}")
        print(f"📁 {folder}")
        print("=" * 70)
        
        # List Python files
        py_files = list(folder.glob('*.py'))
        print(f"\nPython files ({len(py_files)}):")
        
        for py_file in sorted(py_files):
            if py_file.name.startswith('__'):
                continue
                
            print(f"\n  📄 {py_file.name}")
            classes, preview = extract_class_info(py_file)
            
            if classes:
                for cls in classes:
                    print(f"     class {cls['name']}")
                    for method in cls['methods']:
                        if not method.startswith('_') or method == '__init__':
                            print(f"       - {method}()")
    
    # Show one full lens file as example
    print("\n" + "=" * 70)
    print("SAMPLE LENS FILE CONTENT")
    print("=" * 70)
    
    for folder in lens_folders:
        for py_file in folder.glob('*.py'):
            if 'granger' in py_file.name.lower() or 'wavelet' in py_file.name.lower():
                print(f"\n--- {py_file.name} (first 100 lines) ---\n")
                try:
                    with open(py_file, 'r') as f:
                        lines = f.readlines()[:100]
                        print(''.join(lines))
                except Exception as e:
                    print(f"Error reading: {e}")
                break
        else:
            continue
        break
    else:
        # Just show the first lens file found
        for folder in lens_folders:
            for py_file in folder.glob('*.py'):
                if not py_file.name.startswith('__'):
                    print(f"\n--- {py_file.name} (first 100 lines) ---\n")
                    try:
                        with open(py_file, 'r') as f:
                            lines = f.readlines()[:100]
                            print(''.join(lines))
                    except Exception as e:
                        print(f"Error reading: {e}")
                    break
            break
    
    print("\n" + "=" * 70)
    print("END OF INSPECTION - Paste this output back to Claude")
    print("=" * 70)

if __name__ == '__main__':
    main()
