#!/usr/bin/env python3
"""
Debug script to trace import issues.

This script will help us understand the import issues by:
1. Showing the Python path
2. Checking if specific modules can be imported
3. Tracing the import process for problematic modules
"""

import sys
import os
import importlib
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def print_separator():
    print("\n" + "="*80 + "\n")

def check_import(module_name):
    """Try to import a module and report success or failure."""
    print(f"Attempting to import: {module_name}")
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        print(f"   Module file: {module.__file__}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        return False

def check_file_exists(path):
    """Check if a file exists and report."""
    if os.path.exists(path):
        print(f"✅ File exists: {path}")
        return True
    else:
        print(f"❌ File does not exist: {path}")
        return False

def main():
    # Print Python version
    print(f"Python version: {sys.version}")
    print_separator()
    
    # Print Python path
    print("Python path (sys.path):")
    for i, path in enumerate(sys.path):
        print(f"{i+1}. {path}")
    print_separator()
    
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        print(f"Adding project root to sys.path: {project_root}")
        sys.path.append(project_root)
    
    # Check if src directory exists
    src_dir = os.path.join(project_root, 'src')
    if os.path.exists(src_dir):
        print(f"✅ src directory exists: {src_dir}")
        if src_dir not in sys.path:
            print(f"Adding src directory to sys.path: {src_dir}")
            sys.path.append(src_dir)
    else:
        print(f"❌ src directory does not exist: {src_dir}")
    print_separator()
    
    # Check if key directories and files exist
    print("Checking if key directories and files exist:")
    paths_to_check = [
        os.path.join(project_root, 'src', 'habitat_evolution'),
        os.path.join(project_root, 'src', 'habitat_evolution', 'adaptive_core'),
        os.path.join(project_root, 'src', 'habitat_evolution', 'adaptive_core', 'persistence'),
        os.path.join(project_root, 'src', 'habitat_evolution', 'adaptive_core', 'persistence', 'interfaces'),
        os.path.join(project_root, 'src', 'habitat_evolution', 'adaptive_core', 'persistence', 'adapters'),
        os.path.join(project_root, 'src', 'habitat_evolution', 'adaptive_core', 'persistence', 'factory.py'),
        os.path.join(project_root, 'src', 'habitat_evolution', 'adaptive_core', 'persistence', 'interfaces', 'field_state_repository.py'),
        os.path.join(project_root, 'src', 'habitat_evolution', 'adaptive_core', 'persistence', 'adapters', 'field_state_repository_adapter.py'),
    ]
    
    for path in paths_to_check:
        check_file_exists(path)
    print_separator()
    
    # Try to import key modules
    print("Attempting to import key modules:")
    modules_to_check = [
        'src',
        'src.habitat_evolution',
        'src.habitat_evolution.adaptive_core',
        'src.habitat_evolution.adaptive_core.persistence',
        'src.habitat_evolution.adaptive_core.persistence.interfaces',
        'src.habitat_evolution.adaptive_core.persistence.adapters',
        'src.habitat_evolution.adaptive_core.persistence.factory',
        'src.habitat_evolution.adaptive_core.persistence.interfaces.field_state_repository',
        'src.habitat_evolution.adaptive_core.persistence.adapters.field_state_repository_adapter',
        'habitat_evolution',
        'habitat_evolution.adaptive_core',
        'habitat_evolution.adaptive_core.persistence',
        'habitat_evolution.adaptive_core.persistence.interfaces',
        'habitat_evolution.adaptive_core.persistence.adapters',
        'habitat_evolution.adaptive_core.persistence.factory',
        'habitat_evolution.adaptive_core.persistence.interfaces.field_state_repository',
        'habitat_evolution.adaptive_core.persistence.adapters.field_state_repository_adapter',
    ]
    
    for module in modules_to_check:
        check_import(module)
        print()  # Add a blank line between module checks
    
    print_separator()
    print("Debug complete!")

if __name__ == "__main__":
    main()
