"""
Climate Pattern Visualizer for Habitat Evolution.

This module provides visualization tools for climate patterns detected
by the vector-tonic pattern domain bridge. It creates interactive
visualizations of statistical patterns and their correlations with
semantic patterns.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ClimatePatternVisualizer:
    """
    Visualizes climate patterns and their correlations.
    """
    
    def __init__(self):
        """Initialize the climate pattern visualizer."""
        self.figure_size = (12, 8)
        self.dpi = 100
        self.color_map = {
            "warming_trend": "red",
            "cooling_trend": "blue",
            "stable_temperature": "green",
            "temperature_anomaly": "orange",
            "seasonal_variation": "purple"
        }
    
    def visualize_temperature_patterns(self, 
                                      data: pd.DataFrame, 
                                      patterns: List[Dict[str, Any]],
                                      title: Optional[str] = None):
        """
        Visualize temperature patterns in climate data.
        
        Args:
            data: DataFrame with climate data
            patterns: List of detected patterns
            title: Optional title for the plot
        """
        if 'temperature' not in data.columns or len(patterns) == 0:
            logger.warning("Cannot visualize: missing temperature data or patterns")
            return
        
        # Create figure
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot temperature data
        plt.plot(data['date'], data['temperature'], 'b-', alpha=0.7, label='Temperature')
        
        # Plot trend lines for each pattern
        for pattern in patterns:
            if pattern['type'] in ['warming_trend', 'cooling_trend', 'stable_temperature']:
                # Get linear trend
                slope = pattern['metadata'].get('slope', 0)
                intercept = pattern['metadata'].get('intercept', 0)
                
                x = np.arange(len(data))
                trend_line = slope * x + intercept
                
                # Plot trend line
                color = self.color_map.get(pattern['type'], 'gray')
                plt.plot(data['date'], trend_line, f'--', color=color,
                         label=f"{pattern['type']} ({pattern['region']})")
                
                # Add annotation
                mid_point = len(data) // 2
                plt.annotate(
                    f"{pattern['type']}\nMagnitude: {pattern['magnitude']:.2f}",
                    xy=(data['date'].iloc[mid_point], trend_line[mid_point]),
                    xytext=(20, 20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2')
                )
            
            elif pattern['type'] == 'temperature_anomaly':
                # Highlight anomalies
                z_scores = (data['temperature'] - data['temperature'].mean()) / data['temperature'].std()
                anomaly_mask = np.abs(z_scores) > 2.0
                
                plt.scatter(
                    data.loc[anomaly_mask, 'date'],
                    data.loc[anomaly_mask, 'temperature'],
                    color='red',
                    s=50,
                    alpha=0.7,
                    label='Temperature Anomalies'
                )
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title(title or f'Climate Patterns for {data["region"].iloc[0]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def visualize_co_patterns(self, 
                             co_patterns: List[Dict[str, Any]],
                             statistical_patterns: Dict[str, Dict[str, Any]],
                             semantic_patterns: Dict[str, Dict[str, Any]]):
        """
        Visualize correlations between statistical and semantic patterns.
        
        Args:
            co_patterns: List of co-patterns
            statistical_patterns: Dictionary of statistical patterns by ID
            semantic_patterns: Dictionary of semantic patterns by ID
        """
        if not co_patterns:
            logger.warning("No co-patterns to visualize")
            return
        
        # Create figure
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Create a correlation matrix visualization
        pattern_count = len(co_patterns)
        correlation_matrix = np.zeros((pattern_count, 3))  # [index, correlation, type]
        
        labels = []
        
        for i, co_pattern in enumerate(co_patterns):
            # Extract pattern IDs
            stat_id = co_pattern.get("statistical_pattern_id")
            sem_id = co_pattern.get("semantic_pattern_id")
            
            # Get pattern details
            stat_pattern = statistical_patterns.get(stat_id, {})
            sem_pattern = semantic_patterns.get(sem_id, {})
            
            # Create label
            stat_label = f"{stat_pattern.get('type', 'Unknown')} ({stat_pattern.get('region', 'Unknown')})"
            sem_text = sem_pattern.get('text', 'Unknown')
            sem_label = sem_text[:50] + "..." if len(sem_text) > 50 else sem_text
            
            labels.append(f"{i+1}. {stat_label} ↔ {sem_label}")
            
            # Store correlation data
            correlation_matrix[i, 0] = i
            correlation_matrix[i, 1] = co_pattern.get('correlation_strength', 0.0)
            correlation_matrix[i, 2] = 0 if co_pattern.get('correlation_type') == 'temporal' else 1
        
        # Sort by correlation strength
        sorted_indices = np.argsort(correlation_matrix[:, 1])[::-1]
        correlation_matrix = correlation_matrix[sorted_indices]
        labels = [labels[int(i)] for i in sorted_indices]
        
        # Plot correlation bars
        colors = ['blue', 'green']
        bar_positions = np.arange(pattern_count)
        
        plt.barh(
            bar_positions,
            correlation_matrix[:, 1],
            color=[colors[int(t)] for t in correlation_matrix[:, 2]],
            alpha=0.7
        )
        
        # Add labels
        plt.yticks(bar_positions, labels)
        plt.xlabel('Correlation Strength')
        plt.title('Cross-Domain Pattern Correlations')
        
        # Add legend
        plt.legend(['Temporal Correlation', 'Semantic Correlation'])
        
        # Show plot
        plt.tight_layout()
        plt.show()


def visualize_climate_patterns(data_file: str, patterns_file: str = None):
    """
    Standalone function to visualize climate patterns from data files.
    
    Args:
        data_file: Path to climate data CSV file
        patterns_file: Optional path to patterns JSON file
    """
    try:
        # Load data
        data = pd.read_csv(data_file)
        
        # Convert date column if needed
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        # Create visualizer
        visualizer = ClimatePatternVisualizer()
        
        # Visualize data
        visualizer.visualize_temperature_patterns(
            data=data,
            patterns=[],  # No patterns for simple visualization
            title=f"Climate Data from {data_file}"
        )
        
        logger.info(f"Visualized climate data from {data_file}")
        
    except Exception as e:
        logger.error(f"Error visualizing climate data: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        patterns_file = sys.argv[2] if len(sys.argv) > 2 else None
        visualize_climate_patterns(data_file, patterns_file)
    else:
        logger.info("Usage: python climate_pattern_visualizer.py <data_file> [patterns_file]")
