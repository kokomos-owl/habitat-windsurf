#!/usr/bin/env python
"""
Debug script to understand Python module resolution path.
"""

import sys
import os
from pathlib import Path

# Print Python path
print("Python sys.path:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

# Try to import modules
print("\nTrying imports:")
try:
    import habitat_evolution
    print("✓ import habitat_evolution succeeded")
except ImportError as e:
    print(f"✗ import habitat_evolution failed: {e}")

try:
    import src.habitat_evolution
    print("✓ import src.habitat_evolution succeeded")
except ImportError as e:
    print(f"✗ import src.habitat_evolution failed: {e}")

# Print project structure
print("\nProject structure:")
project_root = Path(__file__).parent
print(f"Project root: {project_root}")
print("Top-level directories:")
for item in project_root.iterdir():
    if item.is_dir() and not item.name.startswith('.'):
        print(f"  {item.name}/")
