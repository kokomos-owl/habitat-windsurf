# src/habitat_evolution/vector_tonic/field_state/multi_scale_analyzer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class MultiScaleAnalyzer:
    """
    Analyzes time series data at multiple temporal scales.
    """
    
    def __init__(self, scales: List[int] = None):
        """
        Initialize the multi-scale analyzer.
        
        Args:
            scales: List of window sizes for analysis (in days)
        """
        self.scales = scales or [30, 90, 365]
    
    def analyze(self, data: pd.DataFrame, value_column: str = 'temperature') -> Dict[str, Any]:
        """
        Analyze time series data at multiple scales.
        
        Args:
            data: DataFrame with time series data
            value_column: Column containing values to analyze
            
        Returns:
            Multi-scale analysis results
        """
        if value_column not in data.columns or len(data) < 2:
            return {"scale_results": {}, "cross_scale_patterns": []}
            
        # Results for each scale
        scale_results = {}
        
        # Analyze at each scale
        for scale in self.scales:
            if len(data) < scale:
                continue
                
            # Create rolling window
            rolling = data[value_column].rolling(window=min(scale, len(data)), min_periods=1)
            
            # Calculate statistics at this scale
            scale_mean = rolling.mean()
            scale_std = rolling.std()
            
            # Calculate trend at this scale
            valid_indices = ~np.isnan(scale_mean)
            if sum(valid_indices) > 2:
                x = np.arange(len(scale_mean))[valid_indices]
                y = scale_mean.values[valid_indices]
                trend = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
            else:
                trend = 0
                
            scale_results[scale] = {
                "trend": trend,
                "scale_magnitude": abs(trend) * (scale / 30)  # Normalize by month
            }
            
        # Find cross-scale patterns
        cross_scale = []
        
        # Check for consistent trends across scales
        trends = {scale: results["trend"] for scale, results in scale_results.items()}
        
        if trends and all(trend > 0.01 for trend in trends.values()):
            cross_scale.append({
                "id": "multi_scale_warming",
                "type": "consistent_warming",
                "scales": list(trends.keys()),
                "magnitude": float(np.mean(list(trends.values()))),
                "confidence": 0.8
            })
        elif trends and all(trend < -0.01 for trend in trends.values()):
            cross_scale.append({
                "id": "multi_scale_cooling",
                "type": "consistent_cooling",
                "scales": list(trends.keys()),
                "magnitude": float(abs(np.mean(list(trends.values())))),
                "confidence": 0.8
            })
            
        return {
            "scale_results": scale_results,
            "cross_scale_patterns": cross_scale
        }