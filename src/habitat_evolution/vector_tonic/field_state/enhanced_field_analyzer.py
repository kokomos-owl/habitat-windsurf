"""
Enhanced field state analyzer for climate time series data.

This module provides an enhanced field state analyzer that uses real NOAA time series data
to detect complex patterns and relationships in climate data.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats, signal

class EnhancedFieldStateAnalyzer:
    """
    An enhanced field-state analyzer for climate time series data.
    """
    
    def __init__(self, dimensions: int = 5, data_dir: Optional[Path] = None):
        """
        Initialize the enhanced field state analyzer.
        
        Args:
            dimensions: Number of dimensions in the field space
            data_dir: Directory containing climate data files
        """
        self.dimensions = dimensions
        self.pattern_positions = {}
        self.data_dir = data_dir or Path(__file__).parents[5] / "docs" / "untitled folder"
        self.ma_data = None
        self.ne_data = None
        self._load_climate_data()
    
    def _load_climate_data(self):
        """Load the NOAA climate data."""
        ma_path = self.data_dir / "MA_AvgTemp_91_24.json"
        ne_path = self.data_dir / "NE_AvgTemp_91_24.json"
        
        try:
            with open(ma_path, "r") as f:
                self.ma_data = json.load(f)
            
            with open(ne_path, "r") as f:
                self.ne_data = json.load(f)
                
            print(f"Loaded NOAA climate data: MA and NE temperature data from 1991-2024")
        except Exception as e:
            print(f"Error loading NOAA climate data: {e}")
    
    def _convert_json_to_dataframe(self, json_data: Dict) -> pd.DataFrame:
        """
        Convert JSON climate data to pandas DataFrame.
        
        Args:
            json_data: JSON climate data
            
        Returns:
            DataFrame with climate data
        """
        if not json_data:
            return pd.DataFrame()
            
        data = json_data.get("data", {})
        
        # Extract data points
        dates = []
        values = []
        anomalies = []
        
        for date_str, point in data.items():
            # Convert YYYYMM format to datetime
            year = int(date_str[:4])
            month = int(date_str[4:])
            date = pd.Timestamp(year=year, month=month, day=1)
            
            dates.append(date)
            values.append(point.get("value", 0))
            anomalies.append(point.get("anomaly", 0))
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "temperature": values,
            "anomaly": anomalies
        })
        
        # Sort by date
        df = df.sort_values("date")
        
        return df
    
    def analyze_time_series(self, region: str = "MA") -> Dict[str, Any]:
        """
        Analyze time series data using enhanced field-state principles.
        
        Args:
            region: Region to analyze (MA or NE)
            
        Returns:
            Analysis results with patterns and their field positions
        """
        # Select data based on region
        json_data = self.ma_data if region == "MA" else self.ne_data
        if not json_data:
            return {"patterns": [], "field_properties": {}}
        
        # Convert to DataFrame
        df = self._convert_json_to_dataframe(json_data)
        if df.empty:
            return {"patterns": [], "field_properties": {}}
        
        # Calculate field properties
        temps = df["temperature"].values
        anomalies = df["anomaly"].values
        dates = df["date"].values
        
        # 1. Linear trend
        x = np.arange(len(temps))
        trend_model = np.polyfit(x, temps, 1)
        trend = trend_model[0]  # Slope
        
        # 2. Volatility (temperature variability)
        volatility = np.std(temps)
        
        # 3. Recent acceleration (comparing recent trend to overall trend)
        recent_period = min(10, len(temps) // 3)  # Last ~10 years or 1/3 of data
        if len(temps) > recent_period:
            recent_trend = np.polyfit(x[-recent_period:], temps[-recent_period:], 1)[0]
            acceleration = recent_trend - trend
        else:
            acceleration = 0
        
        # 4. Anomaly intensity (magnitude of recent anomalies)
        recent_anomalies = anomalies[-recent_period:] if len(anomalies) > recent_period else anomalies
        anomaly_intensity = np.mean(np.abs(recent_anomalies))
        
        # 5. Seasonal pattern strength
        if len(temps) >= 24:  # Need at least 2 years of data
            # Detrend the data
            detrended = signal.detrend(temps)
            # Calculate autocorrelation
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Take second half
            # Normalize
            autocorr = autocorr / autocorr[0]
            # Look for peaks around 12 months
            if len(autocorr) > 12:
                seasonal_strength = autocorr[12]  # Correlation at 12-month lag
            else:
                seasonal_strength = 0
        else:
            seasonal_strength = 0
        
        # Create patterns based on field properties
        patterns = []
        
        # Pattern 1: Temperature Trend
        if abs(trend) > 0.01:
            pattern = {
                "id": f"{region.lower()}_temp_trend",
                "type": "warming_trend" if trend > 0 else "cooling_trend",
                "magnitude": abs(trend),
                "position": [trend, volatility, acceleration, anomaly_intensity, seasonal_strength],
                "confidence": min(0.5 + abs(trend) * 10, 0.95),  # Higher confidence for stronger trends
                "metadata": {
                    "region": region,
                    "data_source": "NOAA",
                    "time_period": f"{df['date'].min().year}-{df['date'].max().year}"
                }
            }
            patterns.append(pattern)
        
        # Pattern 2: Acceleration Pattern
        if abs(acceleration) > 0.005:
            pattern = {
                "id": f"{region.lower()}_temp_acceleration",
                "type": "accelerating_warming" if acceleration > 0 else "decelerating_warming",
                "magnitude": abs(acceleration),
                "position": [trend, volatility, acceleration, anomaly_intensity, seasonal_strength],
                "confidence": min(0.5 + abs(acceleration) * 50, 0.9),  # Higher confidence for stronger acceleration
                "metadata": {
                    "region": region,
                    "data_source": "NOAA",
                    "time_period": f"{df['date'].min().year}-{df['date'].max().year}"
                }
            }
            patterns.append(pattern)
        
        # Pattern 3: Anomaly Intensity Pattern
        if anomaly_intensity > 0.3:  # Significant anomalies
            pattern = {
                "id": f"{region.lower()}_anomaly_intensity",
                "type": "increasing_anomalies" if np.mean(recent_anomalies) > 0 else "decreasing_anomalies",
                "magnitude": anomaly_intensity,
                "position": [trend, volatility, acceleration, anomaly_intensity, seasonal_strength],
                "confidence": min(0.5 + anomaly_intensity, 0.9),
                "metadata": {
                    "region": region,
                    "data_source": "NOAA",
                    "time_period": f"{df['date'].min().year}-{df['date'].max().year}"
                }
            }
            patterns.append(pattern)
        
        # Pattern 4: Seasonal Pattern
        if seasonal_strength > 0.3:  # Significant seasonal pattern
            pattern = {
                "id": f"{region.lower()}_seasonal_pattern",
                "type": "strong_seasonal_cycle",
                "magnitude": seasonal_strength,
                "position": [trend, volatility, acceleration, anomaly_intensity, seasonal_strength],
                "confidence": min(0.5 + seasonal_strength, 0.9),
                "metadata": {
                    "region": region,
                    "data_source": "NOAA",
                    "time_period": f"{df['date'].min().year}-{df['date'].max().year}"
                }
            }
            patterns.append(pattern)
        
        # Pattern 5: Volatility Pattern
        if volatility > np.mean(temps) * 0.05:  # Significant volatility
            pattern = {
                "id": f"{region.lower()}_volatility_pattern",
                "type": "high_temperature_volatility",
                "magnitude": volatility,
                "position": [trend, volatility, acceleration, anomaly_intensity, seasonal_strength],
                "confidence": min(0.5 + volatility / np.mean(temps), 0.9),
                "metadata": {
                    "region": region,
                    "data_source": "NOAA",
                    "time_period": f"{df['date'].min().year}-{df['date'].max().year}"
                }
            }
            patterns.append(pattern)
        
        # Calculate extreme events
        if len(temps) > 20:
            # Calculate 90th and 10th percentiles
            p90 = np.percentile(temps, 90)
            p10 = np.percentile(temps, 10)
            
            # Count extreme events in recent period
            extreme_high = np.sum(temps[-recent_period:] > p90)
            extreme_low = np.sum(temps[-recent_period:] < p10)
            
            # Expected number of extreme events in recent period
            expected = recent_period * 0.1
            
            # Pattern 6: Extreme High Temperature Events
            if extreme_high > expected * 1.5:  # 50% more than expected
                pattern = {
                    "id": f"{region.lower()}_extreme_high_temp",
                    "type": "increasing_extreme_high_temps",
                    "magnitude": extreme_high / expected,
                    "position": [trend, volatility, acceleration, anomaly_intensity, seasonal_strength],
                    "confidence": min(0.5 + (extreme_high - expected) / expected, 0.9),
                    "metadata": {
                        "region": region,
                        "data_source": "NOAA",
                        "time_period": f"{df['date'].min().year}-{df['date'].max().year}"
                    }
                }
                patterns.append(pattern)
            
            # Pattern 7: Extreme Low Temperature Events
            if extreme_low > expected * 1.5:  # 50% more than expected
                pattern = {
                    "id": f"{region.lower()}_extreme_low_temp",
                    "type": "increasing_extreme_low_temps",
                    "magnitude": extreme_low / expected,
                    "position": [trend, volatility, acceleration, anomaly_intensity, seasonal_strength],
                    "confidence": min(0.5 + (extreme_low - expected) / expected, 0.9),
                    "metadata": {
                        "region": region,
                        "data_source": "NOAA",
                        "time_period": f"{df['date'].min().year}-{df['date'].max().year}"
                    }
                }
                patterns.append(pattern)
        
        return {
            "patterns": patterns,
            "field_properties": {
                "trend": trend,
                "volatility": volatility,
                "acceleration": acceleration,
                "anomaly_intensity": anomaly_intensity,
                "seasonal_strength": seasonal_strength,
                "region": region,
                "data_source": "NOAA",
                "time_period": f"{df['date'].min().year}-{df['date'].max().year}"
            }
        }
    
    def analyze_regional_differences(self) -> Dict[str, Any]:
        """
        Analyze differences between MA and NE temperature patterns.
        
        Returns:
            Analysis of regional differences
        """
        ma_results = self.analyze_time_series("MA")
        ne_results = self.analyze_time_series("NE")
        
        ma_props = ma_results["field_properties"]
        ne_props = ne_results["field_properties"]
        
        # Calculate differences
        diff_trend = ma_props.get("trend", 0) - ne_props.get("trend", 0)
        diff_volatility = ma_props.get("volatility", 0) - ne_props.get("volatility", 0)
        diff_acceleration = ma_props.get("acceleration", 0) - ne_props.get("acceleration", 0)
        diff_anomaly = ma_props.get("anomaly_intensity", 0) - ne_props.get("anomaly_intensity", 0)
        diff_seasonal = ma_props.get("seasonal_strength", 0) - ne_props.get("seasonal_strength", 0)
        
        # Create regional difference patterns
        patterns = []
        
        # Pattern: Regional Trend Difference
        if abs(diff_trend) > 0.005:
            pattern = {
                "id": "ma_ne_trend_difference",
                "type": "regional_warming_difference",
                "magnitude": abs(diff_trend),
                "position": [diff_trend, diff_volatility, diff_acceleration, diff_anomaly, diff_seasonal],
                "confidence": min(0.5 + abs(diff_trend) * 20, 0.9),
                "metadata": {
                    "regions": ["MA", "NE"],
                    "data_source": "NOAA",
                    "description": f"{'MA' if diff_trend > 0 else 'NE'} is warming faster than {'NE' if diff_trend > 0 else 'MA'}"
                }
            }
            patterns.append(pattern)
        
        # Pattern: Regional Volatility Difference
        if abs(diff_volatility) > 0.1:
            pattern = {
                "id": "ma_ne_volatility_difference",
                "type": "regional_volatility_difference",
                "magnitude": abs(diff_volatility),
                "position": [diff_trend, diff_volatility, diff_acceleration, diff_anomaly, diff_seasonal],
                "confidence": min(0.5 + abs(diff_volatility), 0.9),
                "metadata": {
                    "regions": ["MA", "NE"],
                    "data_source": "NOAA",
                    "description": f"{'MA' if diff_volatility > 0 else 'NE'} has higher temperature volatility than {'NE' if diff_volatility > 0 else 'MA'}"
                }
            }
            patterns.append(pattern)
        
        # Pattern: Regional Acceleration Difference
        if abs(diff_acceleration) > 0.002:
            pattern = {
                "id": "ma_ne_acceleration_difference",
                "type": "regional_acceleration_difference",
                "magnitude": abs(diff_acceleration),
                "position": [diff_trend, diff_volatility, diff_acceleration, diff_anomaly, diff_seasonal],
                "confidence": min(0.5 + abs(diff_acceleration) * 100, 0.9),
                "metadata": {
                    "regions": ["MA", "NE"],
                    "data_source": "NOAA",
                    "description": f"{'MA' if diff_acceleration > 0 else 'NE'} is experiencing faster acceleration in warming than {'NE' if diff_acceleration > 0 else 'MA'}"
                }
            }
            patterns.append(pattern)
        
        # Pattern: Regional Anomaly Difference
        if abs(diff_anomaly) > 0.1:
            pattern = {
                "id": "ma_ne_anomaly_difference",
                "type": "regional_anomaly_difference",
                "magnitude": abs(diff_anomaly),
                "position": [diff_trend, diff_volatility, diff_acceleration, diff_anomaly, diff_seasonal],
                "confidence": min(0.5 + abs(diff_anomaly) * 2, 0.9),
                "metadata": {
                    "regions": ["MA", "NE"],
                    "data_source": "NOAA",
                    "description": f"{'MA' if diff_anomaly > 0 else 'NE'} is experiencing stronger temperature anomalies than {'NE' if diff_anomaly > 0 else 'MA'}"
                }
            }
            patterns.append(pattern)
        
        return {
            "patterns": patterns,
            "field_properties": {
                "diff_trend": diff_trend,
                "diff_volatility": diff_volatility,
                "diff_acceleration": diff_acceleration,
                "diff_anomaly": diff_anomaly,
                "diff_seasonal": diff_seasonal,
                "regions": ["MA", "NE"],
                "data_source": "NOAA"
            }
        }
    
    def detect_pattern_relationships(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect relationships between patterns.
        
        Args:
            patterns: List of patterns
            
        Returns:
            List of pattern relationships
        """
        relationships = []
        
        # Compare each pair of patterns
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                # Skip self-relationships
                if i == j:
                    continue
                
                # Get pattern positions
                pos1 = pattern1.get("position", [0, 0, 0, 0, 0])
                pos2 = pattern2.get("position", [0, 0, 0, 0, 0])
                
                # Calculate distance in field space
                distance = np.sqrt(np.sum([(a - b) ** 2 for a, b in zip(pos1, pos2)]))
                
                # Calculate correlation (dot product of normalized vectors)
                norm1 = np.sqrt(np.sum([a ** 2 for a in pos1]))
                norm2 = np.sqrt(np.sum([a ** 2 for a in pos2]))
                
                if norm1 > 0 and norm2 > 0:
                    correlation = np.sum([a * b for a, b in zip(pos1, pos2)]) / (norm1 * norm2)
                else:
                    correlation = 0
                
                # Determine relationship type
                if correlation > 0.8:
                    rel_type = "reinforcing"
                    strength = correlation
                elif correlation < -0.8:
                    rel_type = "opposing"
                    strength = abs(correlation)
                elif distance < 0.5:
                    rel_type = "proximal"
                    strength = 1 - distance
                elif abs(correlation) > 0.5:
                    rel_type = "correlated" if correlation > 0 else "inversely_correlated"
                    strength = abs(correlation)
                else:
                    # No significant relationship
                    continue
                
                # Create relationship
                relationship = {
                    "id": f"{pattern1['id']}_{pattern2['id']}",
                    "pattern1_id": pattern1["id"],
                    "pattern2_id": pattern2["id"],
                    "type": rel_type,
                    "strength": strength,
                    "distance": distance,
                    "correlation": correlation,
                    "description": f"{rel_type.capitalize()} relationship between {pattern1['type']} and {pattern2['type']}"
                }
                
                relationships.append(relationship)
        
        return relationships
    
    def analyze_all(self) -> Dict[str, Any]:
        """
        Run a complete analysis of all available climate data.
        
        Returns:
            Complete analysis results
        """
        # Analyze MA data
        ma_results = self.analyze_time_series("MA")
        ma_patterns = ma_results["patterns"]
        
        # Analyze NE data
        ne_results = self.analyze_time_series("NE")
        ne_patterns = ne_results["patterns"]
        
        # Analyze regional differences
        diff_results = self.analyze_regional_differences()
        diff_patterns = diff_results["patterns"]
        
        # Combine all patterns
        all_patterns = ma_patterns + ne_patterns + diff_patterns
        
        # Detect relationships
        relationships = self.detect_pattern_relationships(all_patterns)
        
        return {
            "ma_results": ma_results,
            "ne_results": ne_results,
            "diff_results": diff_results,
            "all_patterns": all_patterns,
            "relationships": relationships,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_source": "NOAA",
                "regions": ["MA", "NE"]
            }
        }
    
    def visualize_field_space(self, patterns: List[Dict[str, Any]], dimensions: List[int] = [0, 1, 2],
                             output_path: Optional[str] = None):
        """
        Visualize patterns in field space.
        
        Args:
            patterns: List of patterns
            dimensions: Dimensions to visualize (indices)
            output_path: Path to save visualization
        """
        if len(patterns) < 2:
            print("Not enough patterns to visualize")
            return
        
        # Extract positions for selected dimensions
        positions = []
        for pattern in patterns:
            pos = pattern.get("position", [0] * self.dimensions)
            # Ensure position has enough dimensions
            if len(pos) < max(dimensions) + 1:
                pos = pos + [0] * (max(dimensions) + 1 - len(pos))
            positions.append([pos[d] for d in dimensions])
        
        # Convert to numpy array
        positions = np.array(positions)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # 3D plot if 3 dimensions selected
        if len(dimensions) == 3:
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=[p.get("magnitude", 0.5) for p in patterns],
                               s=[p.get("confidence", 0.5) * 100 for p in patterns],
                               alpha=0.7, cmap='viridis')
            
            # Add labels
            for i, pattern in enumerate(patterns):
                ax.text(positions[i, 0], positions[i, 1], positions[i, 2],
                      pattern.get("type", "").replace("_", " "),
                      fontsize=8)
            
            # Set labels
            dim_names = ["Trend", "Volatility", "Acceleration", "Anomaly Intensity", "Seasonal Strength"]
            ax.set_xlabel(dim_names[dimensions[0]] if dimensions[0] < len(dim_names) else f"Dimension {dimensions[0]}")
            ax.set_ylabel(dim_names[dimensions[1]] if dimensions[1] < len(dim_names) else f"Dimension {dimensions[1]}")
            ax.set_zlabel(dim_names[dimensions[2]] if dimensions[2] < len(dim_names) else f"Dimension {dimensions[2]}")
            
        # 2D plot if 2 dimensions selected
        elif len(dimensions) == 2:
            ax = fig.add_subplot(111)
            
            # Plot points
            scatter = ax.scatter(positions[:, 0], positions[:, 1],
                               c=[p.get("magnitude", 0.5) for p in patterns],
                               s=[p.get("confidence", 0.5) * 100 for p in patterns],
                               alpha=0.7, cmap='viridis')
            
            # Add labels
            for i, pattern in enumerate(patterns):
                ax.text(positions[i, 0], positions[i, 1],
                      pattern.get("type", "").replace("_", " "),
                      fontsize=8)
            
            # Set labels
            dim_names = ["Trend", "Volatility", "Acceleration", "Anomaly Intensity", "Seasonal Strength"]
            ax.set_xlabel(dim_names[dimensions[0]] if dimensions[0] < len(dim_names) else f"Dimension {dimensions[0]}")
            ax.set_ylabel(dim_names[dimensions[1]] if dimensions[1] < len(dim_names) else f"Dimension {dimensions[1]}")
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Pattern Magnitude")
        
        # Set title
        plt.title("Climate Patterns in Field Space")
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            print(f"Saved field space visualization to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()
    
    def visualize_relationships(self, patterns: List[Dict[str, Any]], relationships: List[Dict[str, Any]],
                               output_path: Optional[str] = None):
        """
        Visualize pattern relationships as a network.
        
        Args:
            patterns: List of patterns
            relationships: List of pattern relationships
            output_path: Path to save visualization
        """
        if len(patterns) < 2 or len(relationships) < 1:
            print("Not enough patterns or relationships to visualize")
            return
        
        # Create pattern lookup
        pattern_lookup = {p["id"]: p for p in patterns}
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create positions (using first two dimensions of field space)
        positions = {}
        for pattern in patterns:
            pos = pattern.get("position", [0] * self.dimensions)
            if len(pos) >= 2:
                positions[pattern["id"]] = (pos[0], pos[1])
            else:
                # Random position if no field position
                positions[pattern["id"]] = (np.random.random(), np.random.random())
        
        # Plot nodes
        for pattern_id, pos in positions.items():
            pattern = pattern_lookup.get(pattern_id, {})
            plt.scatter(pos[0], pos[1], s=pattern.get("confidence", 0.5) * 100,
                      c=[pattern.get("magnitude", 0.5)], cmap='viridis',
                      alpha=0.7)
            plt.text(pos[0], pos[1], pattern.get("type", "").replace("_", " "),
                   fontsize=8)
        
        # Plot edges
        for rel in relationships:
            p1_id = rel.get("pattern1_id", "")
            p2_id = rel.get("pattern2_id", "")
            
            if p1_id in positions and p2_id in positions:
                p1_pos = positions[p1_id]
                p2_pos = positions[p2_id]
                
                # Determine line style based on relationship type
                rel_type = rel.get("type", "")
                if rel_type == "reinforcing":
                    linestyle = "-"
                    color = "green"
                elif rel_type == "opposing":
                    linestyle = "--"
                    color = "red"
                elif rel_type == "proximal":
                    linestyle = ":"
                    color = "blue"
                elif rel_type == "correlated":
                    linestyle = "-."
                    color = "purple"
                elif rel_type == "inversely_correlated":
                    linestyle = "-."
                    color = "orange"
                else:
                    linestyle = ":"
                    color = "gray"
                
                # Draw line with width proportional to strength
                plt.plot([p1_pos[0], p2_pos[0]], [p1_pos[1], p2_pos[1]],
                       linestyle=linestyle, color=color,
                       linewidth=rel.get("strength", 0.5) * 2,
                       alpha=0.7)
        
        # Set labels and title
        plt.xlabel("Trend")
        plt.ylabel("Volatility")
        plt.title("Climate Pattern Relationships")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="green", linestyle="-", label="Reinforcing"),
            Line2D([0], [0], color="red", linestyle="--", label="Opposing"),
            Line2D([0], [0], color="blue", linestyle=":", label="Proximal"),
            Line2D([0], [0], color="purple", linestyle="-.", label="Correlated"),
            Line2D([0], [0], color="orange", linestyle="-.", label="Inversely Correlated")
        ]
        plt.legend(handles=legend_elements)
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            print(f"Saved relationship visualization to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()
