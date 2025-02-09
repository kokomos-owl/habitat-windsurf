"""Process climate risk metrics for Martha's Vineyard towns."""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple

class TownMetricsProcessor:
    """Process and aggregate climate risk metrics for Martha's Vineyard towns."""
    
    def __init__(self):
        """Initialize the processor."""
        self.data_dir = Path(__file__).parent.parent / 'maps' / 'data'
        self.geojson_path = self.data_dir / 'marthas_vineyard.geojson'
        
        # Load town boundaries
        self.town_boundaries = gpd.read_file(self.geojson_path)
        
        # Initialize metrics storage
        self.metrics = {
            'coherence': {},
            'cross_pattern_flow': {},
            'emergence_rate': {},
            'social_support': {}
        }
    
    def process_climate_metrics(self, metrics_data: Dict[str, float]) -> None:
        """Process climate risk metrics for each town.
        
        Args:
            metrics_data: Dictionary containing metric values for each town
        """
        # Here we'll integrate with the actual climate risk metrics
        # For now, using placeholder data
        for town in self.town_boundaries['TOWN'].unique():
            for metric in self.metrics.keys():
                # This will be replaced with actual metric calculations
                self.metrics[metric][town] = metrics_data.get(
                    f"{town}_{metric}", 
                    0.0
                )
    
    def get_metric_range(self, metric: str) -> Tuple[float, float]:
        """Get the min and max values for a given metric.
        
        Args:
            metric: Name of the metric
            
        Returns:
            Tuple of (min_value, max_value)
        """
        values = list(self.metrics[metric].values())
        return min(values), max(values)
    
    def get_metric_values(self) -> pd.DataFrame:
        """Get all metric values as a DataFrame.
        
        Returns:
            DataFrame with towns as index and metrics as columns
        """
        return pd.DataFrame(self.metrics)
    
    def merge_with_geometry(self) -> gpd.GeoDataFrame:
        """Merge metric values with town geometries.
        
        Returns:
            GeoDataFrame with metrics and geometries
        """
        metrics_df = self.get_metric_values()
        return self.town_boundaries.merge(
            metrics_df,
            left_on='TOWN',
            right_index=True
        )
