"""
Run Climate Processor

This script runs the HarmonicClimateProcessor to process climate risk data
through the harmonic I/O system without preset transformations and actants.

Usage:
    python -m habitat_evolution.climate_risk.run_climate_processor
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import argparse

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from habitat_evolution.climate_risk.harmonic_climate_processor import create_climate_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('climate_processor.log')
    ]
)

# Reset any existing loggers to use our format
for logger_name in logging.root.manager.loggerDict:
    logger_obj = logging.getLogger(logger_name)
    for handler in logger_obj.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)


def main():
    """Run the climate processor."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process climate risk data through harmonic I/O system')
    parser.add_argument('--data-dir', type=str, default='data/climate_risk',
                        help='Directory containing climate risk data')
    parser.add_argument('--output-dir', type=str, default='output/climate_risk',
                        help='Directory for output files')
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processor and I/O service
    logger.info(f"Creating climate processor for data directory: {args.data_dir}")
    processor, io_service = create_climate_processor(args.data_dir)
    
    try:
        # Process data
        logger.info("Starting data processing")
        metrics = processor.process_data()
        
        # Log metrics
        logger.info(f"Processing complete. Metrics: {metrics}")
        
        # Save metrics to output directory
        metrics_file = output_dir / f"processing_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
        
    finally:
        # Ensure I/O service is stopped
        logger.info("Stopping I/O service")
        io_service.stop()


if __name__ == "__main__":
    main()
