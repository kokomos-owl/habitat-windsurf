"""
Generate Climate Data for Visualization.

This script generates synthetic climate data for Massachusetts
and saves it to a CSV file for visualization.
"""

import logging
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_climate_data(region: str, output_dir: str = None):
    """
    Generate synthetic climate data for a region.
    
    Args:
        region: Region to generate data for
        output_dir: Directory to save the data to
    
    Returns:
        Path to the saved data file
    """
    # Create date range from 2000 to 2024
    dates = pd.date_range(start='2000-01-01', end='2024-01-01', freq='MS')
    
    # Generate temperature data with trend and seasonal components
    n = len(dates)
    
    # Base temperature varies by region
    if region.lower() in ['massachusetts', 'northeast']:
        base_temp = 10.0  # Celsius
        seasonal_amp = 15.0  # Seasonal amplitude
    elif region.lower() in ['florida', 'southeast']:
        base_temp = 22.0
        seasonal_amp = 8.0
    else:
        base_temp = 15.0
        seasonal_amp = 10.0
    
    # Add warming trend
    trend = np.linspace(0, 2.0, n)  # 2 degree warming over the period
    
    # Add seasonal component
    seasonal = seasonal_amp * np.sin(2 * np.pi * np.arange(n) / 12)
    
    # Add noise
    noise = np.random.normal(0, 1, n)
    
    # Combine components
    temps = base_temp + trend + seasonal + noise
    
    # Calculate anomalies from baseline
    baseline = base_temp + seasonal
    anomalies = temps - baseline
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature': temps,
        'anomaly': anomalies,
        'region': region
    })
    
    # Save to CSV
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data"
        )
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"climate_data_{region.lower()}_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)
    
    # Save data
    df.to_csv(file_path, index=False)
    logger.info(f"Generated climate data for {region} saved to {file_path}")
    
    return file_path


if __name__ == "__main__":
    # Generate data for Massachusetts
    data_file = generate_climate_data("Massachusetts")
    
    # Print instructions for visualization
    logger.info("\nTo visualize this data, run:")
    logger.info(f"python -m src.habitat_evolution.vector_tonic.visualization.climate_pattern_visualizer {data_file}")
