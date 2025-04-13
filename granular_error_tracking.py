#!/usr/bin/env python
"""
Granular Error Tracking for Habitat Evolution

This script provides a detailed diagnostic approach to track errors in component initialization
and dependency chains in the Habitat Evolution system. It uses enhanced logging and
step-by-step verification to identify exactly where issues are occurring.
"""

import os
import sys
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("logs/granular_error_tracking.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("error_tracker")

# Import the system initializer
from src.habitat_evolution.infrastructure.initialization.system_initializer import (
    initialize_system,
    SystemInitializer
)

def setup_argparse():
    """Set up argument parsing for the script."""
    parser = argparse.ArgumentParser(description='Granular Error Tracking for Habitat Evolution')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--log-dir', default='logs', help='Directory to store log files')
    parser.add_argument('--report-dir', default='diagnostic_reports', help='Directory to store diagnostic reports')
    parser.add_argument('--config-file', default=None, help='Path to configuration file')
    
    return parser.parse_args()

def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    # Default configuration
    return {
        'arangodb': {
            'host': 'localhost',
            'port': 8529,
            'username': 'root',
            'password': 'habitat',
            'database': 'habitat_evolution'
        },
        'claude': {
            'api_key': 'mock_api_key_for_testing'
        }
    }

def main():
    """Run the granular error tracking."""
    args = setup_argparse()
    
    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"error_tracking_{timestamp}"
    
    # Configure logging
    log_file = os.path.join(args.log_dir, f"{run_id}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'))
    
    # Set log level based on verbose flag
    root_logger = logging.getLogger()
    if args.verbose:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)
    
    # Add file handler to root logger
    root_logger.addHandler(file_handler)
    
    logger.info(f"Starting granular error tracking: {run_id}")
    logger.info(f"Log file: {log_file}")
    
    # Load configuration
    config = load_config(args.config_file)
    logger.info("Configuration loaded")
    
    # Initialize the system
    logger.info("Initializing Habitat Evolution system...")
    success, components, errors = initialize_system(config)
    
    # Generate report
    report_file = os.path.join(args.report_dir, f"{run_id}_report.json")
    
    # Create the report
    report = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'components_initialized': list(components.keys()),
        'initialization_errors': errors
    }
    
    # Save the report
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Error tracking report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"GRANULAR ERROR TRACKING SUMMARY: {run_id}")
    print("="*50)
    
    if success:
        print("\nSystem initialization successful!")
        print(f"Components initialized: {len(components)}")
        for component in components:
            print(f"  - {component}")
    else:
        print("\nSystem initialization failed!")
        print(f"Components initialized: {len(components)} of {len(components) + len(errors)}")
        print("\nInitialization errors:")
        for component, error in errors.items():
            print(f"  - {component}: {error}")
    
    print("\n" + "="*50)
    print(f"Log file: {log_file}")
    print(f"Report file: {report_file}")
    print("="*50)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
