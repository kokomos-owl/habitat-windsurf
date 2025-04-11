"""
Script to list all dependencies for the climate_e2e test.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def get_imports_from_file(file_path):
    """Extract import statements from a Python file."""
    imports = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('from src.habitat_evolution') or line.startswith('import src.habitat_evolution'):
                imports.append(line)
    return imports

def get_file_path_from_import(import_statement):
    """Convert an import statement to a file path."""
    if import_statement.startswith('from '):
        # Handle 'from x.y.z import a'
        parts = import_statement.split(' import ')[0].replace('from ', '').split('.')
    else:
        # Handle 'import x.y.z'
        parts = import_statement.replace('import ', '').split('.')
    
    # Convert to file path
    file_path = os.path.join(*parts)
    if not file_path.endswith('.py'):
        file_path += '.py'
    
    return file_path

def process_file(file_path, processed_files=None):
    """Process a file and its imports recursively."""
    if processed_files is None:
        processed_files = set()
    
    if file_path in processed_files:
        return []
    
    processed_files.add(file_path)
    
    try:
        imports = get_imports_from_file(file_path)
    except FileNotFoundError:
        return []
    
    files = [file_path]
    
    for import_statement in imports:
        import_file_path = get_file_path_from_import(import_statement)
        full_path = os.path.join(project_root, import_file_path)
        
        if os.path.exists(full_path):
            files.extend(process_file(full_path, processed_files))
    
    return files

# Start with the test file
test_file = os.path.join(project_root, 'tests', 'integration', 'climate_e2e', 'test_climate_e2e.py')
conftest_file = os.path.join(project_root, 'tests', 'integration', 'climate_e2e', 'conftest.py')
test_utils_file = os.path.join(project_root, 'tests', 'integration', 'climate_e2e', 'test_utils.py')

# Process all files
all_files = []
processed_files = set()

for file in [test_file, conftest_file, test_utils_file]:
    all_files.extend(process_file(file, processed_files))

# Remove duplicates and sort
all_files = sorted(set(all_files))

# Print the results
print("Files required for climate_e2e test:")
for file in all_files:
    print(f"- {os.path.relpath(file, project_root)}")

# Also identify data files
print("\nData files:")
with open(conftest_file, "r") as f:
    conftest = f.read()

import re
data_paths = re.findall(r'"([^"]*\.json)"', conftest)
data_dirs = re.findall(r'"([^"]*climate_risk[^"]*)"', conftest)

for path in data_paths:
    print(f"- {path}")
for dir in data_dirs:
    print(f"- {dir}")
