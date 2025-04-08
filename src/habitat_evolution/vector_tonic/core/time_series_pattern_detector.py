"""
Time Series Pattern Detector for Habitat Evolution.

This module provides the TimeSeriesPatternDetector class that applies
vector-tonic methods to detect patterns in time-series data. It implements
sliding window analysis to identify patterns across different temporal scales.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import json
from datetime import datetime

from src.habitat_evolution.vector_tonic.core.vector_tonic_field import VectorTonicField

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPattern:
    """
    Represents a pattern detected in time-series data.
    
    Attributes:
        id: Unique identifier for the pattern
        start_time: Start time of the pattern
        end_time: End time of the pattern
        values: The time-series values within the pattern
        trend: The overall trend of the pattern (increasing, decreasing, stable)
        magnitude: The magnitude of the pattern
        quality_state: The quality state of the pattern (hypothetical, emergent, stable)
        metadata: Additional metadata about the pattern
    """
    id: str
    start_time: str
    end_time: str
    values: List[float]
    trend: str
    magnitude: float
    quality_state: str
    metadata: Dict[str, Any]


class TimeSeriesPatternDetector:
    """
    Detects patterns in time-series data using vector-tonic methods.
    
    This class implements sliding window analysis to detect patterns
    across different temporal scales. It converts time-series data into
    vector-tonic fields and applies field equations to identify patterns.
    """
    
    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize a new time series pattern detector.
        
        Args:
            window_sizes: List of window sizes to use for sliding window analysis
                          If None, default window sizes will be used
        """
        self.window_sizes = window_sizes or [3, 5, 7]
        self.fields = {}
        
    def load_time_series(self, file_path: str) -> Dict[str, Any]:
        """
        Load time-series data from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing time-series data
            
        Returns:
            The loaded time-series data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    
    def preprocess_time_series(self, time_series_data: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        """
        Preprocess time-series data for pattern detection.
        
        Args:
            time_series_data: The time-series data to preprocess
            
        Returns:
            A tuple of (timestamps, values)
        """
        # Extract timestamps and values from the data
        timestamps = []
        values = []
        
        # Handle the specific format of our climate data
        if "data" in time_series_data:
            for timestamp, data_point in time_series_data["data"].items():
                timestamps.append(timestamp)
                
                # Use anomaly if available, otherwise use value
                if "anomaly" in data_point:
                    values.append(data_point["anomaly"])
                else:
                    values.append(data_point["value"])
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values), key=lambda x: x[0])
        timestamps = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]
        
        return timestamps, values
    
    def create_vector_tonic_field(self, timestamps: List[str], values: List[float], 
                                 window_size: int) -> VectorTonicField:
        """
        Create a vector-tonic field from time-series data using a sliding window.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            window_size: Size of the sliding window
            
        Returns:
            A vector-tonic field representing the time-series data
        """
        # Determine the dimensions of the field
        # For time-series data, we'll use a 2D field where:
        # - First dimension is time
        # - Second dimension is the value space
        
        # Number of windows we can create
        num_windows = len(values) - window_size + 1
        if num_windows <= 0:
            raise ValueError(f"Window size {window_size} is too large for the data with {len(values)} points")
        
        # Create a field with dimensions (num_windows, window_size)
        field = VectorTonicField(dimensions=(num_windows, window_size))
        
        # Populate the field with values
        for i in range(num_windows):
            window_values = values[i:i+window_size]
            for j, value in enumerate(window_values):
                field.set_value((i, j), value)
        
        # Calculate field properties
        field.calculate_gradients()
        field.calculate_potential()
        
        # Store the field with its metadata
        field_id = f"field_{window_size}_{len(timestamps)}"
        self.fields[field_id] = {
            "field": field,
            "window_size": window_size,
            "timestamps": timestamps,
            "values": values
        }
        
        return field
    
    def detect_patterns(self, timestamps: List[str], values: List[float], 
                       threshold: float = 0.5) -> List[TimeSeriesPattern]:
        """
        Detect patterns in time-series data using vector-tonic fields.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            threshold: Threshold for pattern detection
            
        Returns:
            A list of detected patterns
        """
        patterns = []
        
        # Create fields for each window size and detect patterns
        for window_size in self.window_sizes:
            # Skip if window size is too large
            if window_size >= len(values):
                logger.warning(f"Skipping window size {window_size} as it's too large for data with {len(values)} points")
                continue
                
            # Create a vector-tonic field
            field = self.create_vector_tonic_field(timestamps, values, window_size)
            
            # Detect patterns in the field
            field_patterns = field.detect_patterns(threshold=threshold)
            
            # Convert field patterns to time-series patterns
            for fp in field_patterns:
                # Get the center of the pattern
                center_i, center_j = fp["center"]
                
                # Calculate the start and end indices in the original time-series
                start_idx = center_i
                end_idx = start_idx + window_size
                
                # Ensure indices are within bounds
                start_idx = max(0, start_idx)
                end_idx = min(len(timestamps), end_idx)
                
                # Extract pattern values
                pattern_values = values[start_idx:end_idx]
                
                # Calculate trend
                if len(pattern_values) >= 2:
                    first_val = pattern_values[0]
                    last_val = pattern_values[-1]
                    if last_val > first_val * 1.05:  # 5% increase
                        trend = "increasing"
                    elif last_val < first_val * 0.95:  # 5% decrease
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:
                    trend = "unknown"
                
                # Calculate magnitude (max absolute deviation from mean)
                mean_val = np.mean(pattern_values)
                magnitude = max(abs(val - mean_val) for val in pattern_values)
                
                # Create a time-series pattern
                ts_pattern = TimeSeriesPattern(
                    id=fp["id"],
                    start_time=timestamps[start_idx],
                    end_time=timestamps[end_idx-1] if end_idx > 0 else timestamps[0],
                    values=pattern_values,
                    trend=trend,
                    magnitude=magnitude,
                    quality_state=fp["quality_state"],
                    metadata={
                        "window_size": window_size,
                        "potential": fp["potential"],
                        "field_center": fp["center"]
                    }
                )
                
                patterns.append(ts_pattern)
        
        return patterns
    
    def detect_resonance_between_series(self, series1_path: str, series2_path: str, 
                                      window_size: int = 5) -> Dict[str, float]:
        """
        Detect resonance between two time-series datasets.
        
        Args:
            series1_path: Path to the first time-series JSON file
            series2_path: Path to the second time-series JSON file
            window_size: Window size to use for analysis
            
        Returns:
            A dictionary mapping resonance IDs to strength values
        """
        # Load and preprocess both time-series
        series1_data = self.load_time_series(series1_path)
        series2_data = self.load_time_series(series2_path)
        
        timestamps1, values1 = self.preprocess_time_series(series1_data)
        timestamps2, values2 = self.preprocess_time_series(series2_data)
        
        # Ensure both series have the same length
        min_length = min(len(values1), len(values2))
        values1 = values1[:min_length]
        values2 = values2[:min_length]
        timestamps1 = timestamps1[:min_length]
        timestamps2 = timestamps2[:min_length]
        
        # Create vector-tonic fields for both series
        field1 = self.create_vector_tonic_field(timestamps1, values1, window_size)
        field2 = self.create_vector_tonic_field(timestamps2, values2, window_size)
        
        # Detect resonance between the fields
        resonance = field1.detect_resonance(field2)
        
        return resonance
    
    def analyze_time_series(self, file_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Perform a complete analysis of a time-series dataset.
        
        Args:
            file_path: Path to the time-series JSON file
            threshold: Threshold for pattern detection
            
        Returns:
            Analysis results including patterns and metadata
        """
        # Load and preprocess the time-series
        data = self.load_time_series(file_path)
        timestamps, values = self.preprocess_time_series(data)
        
        # Detect patterns
        patterns = self.detect_patterns(timestamps, values, threshold)
        
        # Calculate overall statistics
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "trend": self._calculate_overall_trend(values)
        }
        
        # Organize results
        results = {
            "metadata": {
                "title": data.get("description", {}).get("title", "Unknown"),
                "units": data.get("description", {}).get("units", "Unknown"),
                "base_period": data.get("description", {}).get("base_period", "Unknown"),
                "analysis_time": datetime.now().isoformat()
            },
            "statistics": stats,
            "patterns": [self._pattern_to_dict(p) for p in patterns],
            "data_points": len(values)
        }
        
        return results
    
    def _calculate_overall_trend(self, values: List[float]) -> str:
        """
        Calculate the overall trend of a time-series.
        
        Args:
            values: List of time-series values
            
        Returns:
            Trend description: "increasing", "decreasing", or "stable"
        """
        if len(values) < 2:
            return "unknown"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Determine trend based on slope
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _pattern_to_dict(self, pattern: TimeSeriesPattern) -> Dict[str, Any]:
        """
        Convert a TimeSeriesPattern to a dictionary.
        
        Args:
            pattern: The pattern to convert
            
        Returns:
            Dictionary representation of the pattern
        """
        return {
            "id": pattern.id,
            "start_time": pattern.start_time,
            "end_time": pattern.end_time,
            "values": pattern.values,
            "trend": pattern.trend,
            "magnitude": pattern.magnitude,
            "quality_state": pattern.quality_state,
            "metadata": pattern.metadata
        }
