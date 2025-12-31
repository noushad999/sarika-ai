"""Create __init__.py files for all packages"""
from pathlib import Path

# All directories that need __init__.py
directories = [
    "ml",
    "ml/core",
    "ml/data",
    "ml/training",
    "ml/prompts",
    "ml/scripts",
    "ml/utils",
    "ml/evaluation",
]

print("Creating __init__.py files...")

for directory in directories:
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    init_file = dir_path / "__init__.py"
    init_file.touch()
    
    print(f"✓ {init_file}")

print("\n✅ All __init__.py files created!")
