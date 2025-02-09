"""Process climate risk metrics for Martha's Vineyard towns."""

from typing import Dict, List, Tuple

class TownMetricsProcessor:
    """Process and aggregate climate risk metrics for Martha's Vineyard towns."""
    
    def __init__(self):
        """Initialize the processor."""
        # Town coordinates (centroids)
        self.town_coordinates = {
            'Aquinnah': [-70.8260, 41.3474],
            'Chilmark': [-70.7574, 41.3432],
            'Edgartown': [-70.5133, 41.3896],
            'Oak Bluffs': [-70.5595, 41.4546],
            'Tisbury': [-70.6134, 41.4532],
            'West Tisbury': [-70.6784, 41.3815]
        }
        
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
        # Process metrics for each town
        for town in self.town_coordinates.keys():
            for metric in self.metrics.keys():
                # Get metric value with fallback to 0.0
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
    
    def get_town_coordinates(self) -> List[Dict[str, any]]:
        """Get coordinates for each town.
        
        Returns:
            List of dictionaries containing town coordinates
        """
        return [
            {
                'name': town,
                'coordinates': coords,
                'metrics': {
                    metric: self.metrics[metric][town]
                    for metric in self.metrics.keys()
                }
            }
            for town, coords in self.town_coordinates.items()
        ]
