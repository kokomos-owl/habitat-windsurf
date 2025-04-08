"""
Climate Pattern Detection Demo for Habitat Evolution.

This script demonstrates the vector-tonic statistical analysis capabilities
by analyzing climate temperature data and detecting patterns.
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

from src.habitat_evolution.vector_tonic.data.climate_data_loader import ClimateTimeSeriesLoader
from src.habitat_evolution.vector_tonic.core.time_series_pattern_detector import TimeSeriesPatternDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_temperature_data(time_series, title=None):
    """Plot temperature data with anomalies."""
    plt.figure(figsize=(12, 6))
    
    # Convert timestamps to years for better visualization
    years = [int(ts[:4]) for ts in time_series.timestamps]
    
    # Plot raw values
    plt.plot(years, time_series.values, 'b-', label='Temperature')
    
    # Add a trend line
    z = np.polyfit(range(len(years)), time_series.values, 1)
    p = np.poly1d(z)
    plt.plot(years, p(range(len(years))), "r--", label=f'Trend (slope: {z[0]:.4f})')
    
    # Add anomalies as a bar chart
    plt.bar(years, time_series.anomalies, alpha=0.3, color='orange', label='Anomaly')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel(f'Temperature ({time_series.units})')
    plt.title(title or time_series.title)
    plt.legend()
    
    return plt


def visualize_patterns(time_series, patterns, title=None):
    """Visualize detected patterns in the time series."""
    plt.figure(figsize=(12, 8))
    
    # Convert timestamps to years for better visualization
    years = [int(ts[:4]) for ts in time_series.timestamps]
    
    # Plot the full time series
    plt.plot(years, time_series.anomalies, 'k-', alpha=0.5, label='Temperature Anomaly')
    
    # Plot each pattern with a different color
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    for i, pattern in enumerate(patterns):
        # Find the start and end indices
        start_year = int(pattern.start_time[:4])
        end_year = int(pattern.end_time[:4])
        
        # Find the indices in the years list
        start_idx = years.index(start_year) if start_year in years else 0
        end_idx = years.index(end_year) if end_year in years else len(years) - 1
        
        # Get the pattern values
        pattern_years = years[start_idx:end_idx+1]
        pattern_values = time_series.anomalies[start_idx:end_idx+1]
        
        # Choose color based on quality state
        if pattern.quality_state == "stable":
            color = colors[0]
            alpha = 0.9
            linewidth = 3
        elif pattern.quality_state == "emergent":
            color = colors[1]
            alpha = 0.7
            linewidth = 2
        else:  # hypothetical
            color = colors[2]
            alpha = 0.5
            linewidth = 1
        
        # Plot the pattern
        plt.plot(pattern_years, pattern_values, color=color, alpha=alpha, linewidth=linewidth,
                label=f'{pattern.quality_state.capitalize()} Pattern: {pattern.trend} ({start_year}-{end_year})')
        
        # Highlight the pattern area
        plt.fill_between(pattern_years, pattern_values, alpha=0.2, color=color)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel(f'Temperature Anomaly ({time_series.units})')
    plt.title(title or f'Patterns in {time_series.title}')
    plt.legend()
    
    return plt


def visualize_resonance(series1, series2, resonance_data):
    """Visualize resonance between two time series."""
    plt.figure(figsize=(12, 8))
    
    # Convert timestamps to years
    years1 = [int(ts[:4]) for ts in series1.timestamps]
    years2 = [int(ts[:4]) for ts in series2.timestamps]
    
    # Plot both series
    plt.plot(years1, series1.anomalies, 'b-', label=f'{series1.region} Anomaly')
    plt.plot(years2, series2.anomalies, 'r-', label=f'{series2.region} Anomaly')
    
    # Highlight periods of high resonance
    if resonance_data and 'lead_lag_relationship' in resonance_data:
        relationship = resonance_data['lead_lag_relationship']
        if relationship['relationship'] != 'synchronous':
            leading = series1 if relationship['leading_region'] == 'region1' else series2
            lagging = series2 if relationship['leading_region'] == 'region1' else series1
            lag = abs(relationship['lag'])
            
            plt.annotate(f"{leading.region} leads {lagging.region} by {lag} time steps",
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # Add correlation information
    if resonance_data and 'correlation' in resonance_data:
        corr = resonance_data['correlation']
        plt.annotate(f"Correlation: {corr:.3f}",
                    xy=(0.5, 0.9), xycoords='axes fraction',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3))
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (°F)')
    plt.title(f'Resonance Between {series1.region} and {series2.region} Temperature Patterns')
    plt.legend()
    
    return plt


def save_analysis_results(results, output_dir):
    """Save analysis results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full results
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save individual pattern files
    for region, region_results in results.items():
        if 'patterns' in region_results:
            pattern_file = os.path.join(output_dir, f'{region}_patterns.json')
            with open(pattern_file, 'w') as f:
                json.dump(region_results['patterns'], f, indent=2)
    
    logger.info(f"Analysis results saved to {output_dir}")


def main():
    """Run the climate pattern detection demo."""
    # Set up paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
    data_dir = os.path.join(base_dir, 'docs', 'time_series')
    output_dir = os.path.join(base_dir, 'docs', 'vector_tonic_results')
    
    logger.info(f"Using data directory: {data_dir}")
    logger.info(f"Output will be saved to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load climate data
    loader = ClimateTimeSeriesLoader(data_dir=data_dir)
    
    # Load Massachusetts and Northeast data
    ma_file = os.path.join(data_dir, 'MA_AvgTemp_91_24.json')
    ne_file = os.path.join(data_dir, 'NE_AvgTemp_91_24.json')
    
    ma_data = loader.load_temperature_anomaly_data(ma_file)
    ne_data = loader.load_temperature_anomaly_data(ne_file)
    
    logger.info(f"Loaded {len(ma_data.timestamps)} data points for {ma_data.region}")
    logger.info(f"Loaded {len(ne_data.timestamps)} data points for {ne_data.region}")
    
    # Plot the raw data
    ma_plot = plot_temperature_data(ma_data)
    ma_plot.savefig(os.path.join(output_dir, 'MA_temperature_data.png'))
    
    ne_plot = plot_temperature_data(ne_data)
    ne_plot.savefig(os.path.join(output_dir, 'NE_temperature_data.png'))
    
    # Create pattern detector
    detector = TimeSeriesPatternDetector(window_sizes=[3, 5, 7])
    
    # Detect patterns in Massachusetts data
    ma_patterns = detector.detect_patterns(ma_data.timestamps, ma_data.anomalies, threshold=0.4)
    logger.info(f"Detected {len(ma_patterns)} patterns in {ma_data.region} data")
    
    # Detect patterns in Northeast data
    ne_patterns = detector.detect_patterns(ne_data.timestamps, ne_data.anomalies, threshold=0.4)
    logger.info(f"Detected {len(ne_patterns)} patterns in {ne_data.region} data")
    
    # Visualize patterns
    ma_patterns_plot = visualize_patterns(ma_data, ma_patterns)
    ma_patterns_plot.savefig(os.path.join(output_dir, 'MA_patterns.png'))
    
    ne_patterns_plot = visualize_patterns(ne_data, ne_patterns)
    ne_patterns_plot.savefig(os.path.join(output_dir, 'NE_patterns.png'))
    
    # Detect resonance between regions
    resonance = detector.detect_resonance_between_series(ma_file, ne_file)
    logger.info(f"Resonance between regions: {resonance}")
    
    # Get detailed regional comparison
    comparison = loader.get_regional_comparison('MA', 'NE')
    logger.info(f"Regional comparison: {comparison}")
    
    # Visualize resonance
    resonance_plot = visualize_resonance(ma_data, ne_data, comparison)
    resonance_plot.savefig(os.path.join(output_dir, 'MA_NE_resonance.png'))
    
    # Compile results
    results = {
        "MA": {
            "metadata": {
                "title": ma_data.title,
                "units": ma_data.units,
                "base_period": ma_data.base_period,
                "region": ma_data.region
            },
            "patterns": [vars(p) for p in ma_patterns],
            "trend": detector._calculate_overall_trend(ma_data.anomalies)
        },
        "NE": {
            "metadata": {
                "title": ne_data.title,
                "units": ne_data.units,
                "base_period": ne_data.base_period,
                "region": ne_data.region
            },
            "patterns": [vars(p) for p in ne_patterns],
            "trend": detector._calculate_overall_trend(ne_data.anomalies)
        },
        "comparison": comparison,
        "resonance": resonance
    }
    
    # Save results
    save_analysis_results(results, output_dir)
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()
