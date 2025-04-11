"""
Dependency finder for climate_e2e test.
This script analyzes the imports in the climate_e2e test to identify all required modules.
"""

import os
import sys
import importlib
import inspect
from modulefinder import ModuleFinder
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Target module to analyze
target_module = "tests.integration.climate_e2e.test_climate_e2e"

# Initialize ModuleFinder
finder = ModuleFinder()

# Run the finder on our target module
finder.run_script(str(project_root / "tests" / "integration" / "climate_e2e" / "test_climate_e2e.py"))

# Get all modules
modules = finder.modules.keys()

# Filter to only include project modules
project_modules = [m for m in modules if m.startswith("src.habitat_evolution") or m.startswith("tests.integration.climate_e2e")]

# Print the results
print("Dependencies for climate_e2e test:")
for module in sorted(project_modules):
    print(f"- {module}")

# Also identify data files used
print("\nData files:")
with open(str(project_root / "tests" / "integration" / "climate_e2e" / "conftest.py"), "r") as f:
    conftest = f.read()
    
# Extract data paths from climate_data_paths fixture
import re
data_paths = re.findall(r'"([^"]*\.json)"', conftest)
data_dirs = re.findall(r'"([^"]*climate_risk[^"]*)"', conftest)

for path in data_paths:
    print(f"- {path}")
for dir in data_dirs:
    print(f"- {dir}")
