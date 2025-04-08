"""
Climate Time Series Data Loader for Habitat Evolution.

This module provides specialized loaders for climate time series data,
with specific handling for temperature anomaly datasets and other
climate-related time series formats.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClimateTimeSeries:
    """
    Represents a climate time series dataset.
    
    Attributes:
        title: Title of the dataset
        units: Units of measurement
        base_period: Base period for anomaly calculations
        timestamps: List of timestamps
        values: List of raw values
        anomalies: List of anomaly values
        region: Geographic region of the data
    """
    title: str
    units: str
    base_period: str
    timestamps: List[str]
    values: List[float]
    anomalies: List[float]
    region: str


class ClimateTimeSeriesLoader:
    """
    Specialized loader for climate time series data.
    
    This class provides methods for loading and preprocessing climate
    time series data, with specific handling for temperature anomaly
    datasets and other climate-related formats.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize a new climate time series loader.
        
        Args:
            data_dir: Directory containing climate data files
                     If None, defaults to docs/time_series
        """
        self.data_dir = data_dir or os.path.join("docs", "time_series")
        
    def load_temperature_anomaly_data(self, file_path: str) -> ClimateTimeSeries:
        """
        Load temperature anomaly data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            A ClimateTimeSeries object containing the data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata
        description = data.get("description", {})
        title = description.get("title", "Unknown")
        units = description.get("units", "Unknown")
        base_period = description.get("base_period", "Unknown")
        
        # Extract region from title
        region = title.split(" ")[0] if " " in title else "Unknown"
        
        # Extract timestamps, values, and anomalies
        timestamps = []
        values = []
        anomalies = []
        
        for timestamp, point_data in data.get("data", {}).items():
            timestamps.append(timestamp)
            values.append(point_data.get("value", 0.0))
            anomalies.append(point_data.get("anomaly", 0.0))
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values, anomalies), key=lambda x: x[0])
        timestamps = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]
        anomalies = [item[2] for item in sorted_data]
        
        return ClimateTimeSeries(
            title=title,
            units=units,
            base_period=base_period,
            timestamps=timestamps,
            values=values,
            anomalies=anomalies,
            region=region
        )
    
    def load_all_temperature_data(self) -> Dict[str, ClimateTimeSeries]:
        """
        Load all temperature data files in the data directory.
        
        Returns:
            A dictionary mapping file names to ClimateTimeSeries objects
        """
        result = {}
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".json") and "Temp" in file_name:
                file_path = os.path.join(self.data_dir, file_name)
                try:
                    time_series = self.load_temperature_anomaly_data(file_path)
                    result[file_name] = time_series
                except Exception as e:
                    logger.error(f"Error loading {file_name}: {str(e)}")
        
        return result
    
    def get_regional_comparison(self, region1: str, region2: str) -> Dict[str, Any]:
        """
        Compare temperature data between two regions.
        
        Args:
            region1: First region code (e.g., "MA", "NE")
            region2: Second region code
            
        Returns:
            Comparison results including correlation and differences
        """
        # Find files for the specified regions
        region1_file = None
        region2_file = None
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".json"):
                if region1 in file_name:
                    region1_file = os.path.join(self.data_dir, file_name)
                if region2 in file_name:
                    region2_file = os.path.join(self.data_dir, file_name)
        
        if not region1_file or not region2_file:
            missing = []
            if not region1_file:
                missing.append(region1)
            if not region2_file:
                missing.append(region2)
            raise ValueError(f"Data for regions {', '.join(missing)} not found")
        
        # Load data for both regions
        ts1 = self.load_temperature_anomaly_data(region1_file)
        ts2 = self.load_temperature_anomaly_data(region2_file)
        
        # Ensure data points align
        common_timestamps = set(ts1.timestamps).intersection(set(ts2.timestamps))
        
        # Filter data to common timestamps
        idx1 = [ts1.timestamps.index(ts) for ts in common_timestamps if ts in ts1.timestamps]
        idx2 = [ts2.timestamps.index(ts) for ts in common_timestamps if ts in ts2.timestamps]
        
        aligned_timestamps = sorted(common_timestamps)
        aligned_anomalies1 = [ts1.anomalies[i] for i in idx1]
        aligned_anomalies2 = [ts2.anomalies[i] for i in idx2]
        
        # Calculate correlation
        correlation = np.corrcoef(aligned_anomalies1, aligned_anomalies2)[0, 1]
        
        # Calculate differences
        differences = [a1 - a2 for a1, a2 in zip(aligned_anomalies1, aligned_anomalies2)]
        
        # Calculate trends
        trend1 = self._calculate_trend(aligned_anomalies1)
        trend2 = self._calculate_trend(aligned_anomalies2)
        
        # Determine if one region leads the other
        lead_lag = self._analyze_lead_lag(aligned_timestamps, aligned_anomalies1, aligned_anomalies2)
        
        return {
            "region1": region1,
            "region2": region2,
            "correlation": float(correlation),
            "mean_difference": float(np.mean(differences)),
            "max_difference": float(np.max(differences)),
            "min_difference": float(np.min(differences)),
            "trend_region1": trend1,
            "trend_region2": trend2,
            "lead_lag_relationship": lead_lag,
            "common_data_points": len(aligned_timestamps)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """
        Calculate trend information for a series of values.
        
        Args:
            values: List of values
            
        Returns:
            Trend information including direction and magnitude
        """
        if len(values) < 2:
            return {"direction": "unknown", "magnitude": 0.0}
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Determine direction
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"
        
        # Calculate predicted values
        predicted = [slope * i + intercept for i in range(len(values))]
        
        # Calculate R-squared
        ss_total = sum((v - np.mean(values))**2 for v in values)
        ss_residual = sum((v - p)**2 for v, p in zip(values, predicted))
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        return {
            "direction": direction,
            "magnitude": float(abs(slope)),
            "slope": float(slope),
            "r_squared": float(r_squared)
        }
    
    def _analyze_lead_lag(self, timestamps: List[str], series1: List[float], 
                         series2: List[float]) -> Dict[str, Any]:
        """
        Analyze if one time series leads or lags another.
        
        Args:
            timestamps: List of timestamps
            series1: First time series values
            series2: Second time series values
            
        Returns:
            Lead-lag analysis results
        """
        # Convert to numpy arrays for easier manipulation
        s1 = np.array(series1)
        s2 = np.array(series2)
        
        # Calculate cross-correlation
        max_lag = min(10, len(s1) // 4)  # Limit to reasonable lag values
        correlations = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # s1 lags behind s2
                corr = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
            elif lag > 0:
                # s1 leads s2
                corr = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
            else:
                # No lag
                corr = np.corrcoef(s1, s2)[0, 1]
            
            correlations.append((lag, corr))
        
        # Find lag with maximum correlation
        max_corr_lag, max_corr = max(correlations, key=lambda x: x[1])
        
        # Determine lead-lag relationship
        if abs(max_corr_lag) <= 1:  # Within 1 time step
            relationship = "synchronous"
            leading_region = "neither"
        elif max_corr_lag < 0:
            relationship = "lagging"
            leading_region = "region2"
        else:
            relationship = "leading"
            leading_region = "region1"
        
        return {
            "relationship": relationship,
            "leading_region": leading_region,
            "lag": max_corr_lag,
            "max_correlation": float(max_corr)
        }
