"""
Load environment variables from test.env file.

This module provides functions to load environment variables from a test.env file,
which is useful for testing with API keys and other sensitive information.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_env_file(file_path):
    """
    Load environment variables from a file.
    
    Args:
        file_path: Path to the environment file
        
    Returns:
        Dict of loaded environment variables
    """
    env_vars = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                key, value = line.split('=', 1)
                env_vars[key] = value
                os.environ[key] = value
                logger.debug(f"Loaded environment variable: {key}")
                
        logger.info(f"Loaded {len(env_vars)} environment variables from {file_path}")
        return env_vars
    except Exception as e:
        logger.error(f"Error loading environment variables from {file_path}: {e}")
        return {}

def find_env_file(filename="test.env"):
    """
    Find the test.env file in the project directory structure.
    
    Args:
        filename: Name of the environment file to find
        
    Returns:
        Path to the environment file, or None if not found
    """
    # Start from the current directory
    current_dir = Path.cwd()
    
    # Look for the file in the current directory and its parents
    while current_dir.parent != current_dir:  # Stop at the root directory
        env_path = current_dir / filename
        if env_path.exists():
            return env_path
        current_dir = current_dir.parent
        
    # If we didn't find it, try the script's directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[2]  # Go up 3 levels from tests/integration/climate_e2e
    env_path = project_root / filename
    
    if env_path.exists():
        return env_path
        
    return None

def load_test_env():
    """
    Load environment variables from test.env file.
    
    Returns:
        Dict of loaded environment variables
    """
    env_path = find_env_file()
    if env_path:
        return load_env_file(env_path)
    else:
        logger.warning("test.env file not found")
        return {}
