"""
Real Climate Pattern Visualizer

This module provides visualization tools for real climate data patterns
and their correlations with climate risk documents.
"""

import json
import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealClimateVisualizer:
    """
    Visualizer for real climate data patterns and their correlations.
    """
    
    def __init__(self, 
                climate_data_dir: str = None,
                climate_risk_dir: str = None,
                output_dir: str = None):
        """
        Initialize the climate visualizer.
        
        Args:
            climate_data_dir: Directory containing climate data JSON files
            climate_risk_dir: Directory containing climate risk documents
            output_dir: Directory to save visualization outputs
        """
        # Set default directories if not provided
        self.climate_data_dir = climate_data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))),
            "docs", "untitled folder"
        )
        
        self.climate_risk_dir = climate_risk_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))),
            "data", "climate_risk"
        )
        
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))),
            "output"
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Climate data directory: {self.climate_data_dir}")
        logger.info(f"Climate risk directory: {self.climate_risk_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_temperature_data(self, region: str) -> pd.DataFrame:
        """
        Load temperature data for a specific region from JSON files.
        
        Args:
            region: Region to load data for (e.g., "MA" for Massachusetts,
                  "NE" for Northeast)
            
        Returns:
            DataFrame with temperature data
        """
        # Map region to filename pattern
        region_map = {
            "massachusetts": "MA_AvgTemp",
            "ma": "MA_AvgTemp",
            "northeast": "NE_AvgTemp",
            "ne": "NE_AvgTemp"
        }
        
        region_key = region.lower()
        if region_key not in region_map:
            raise ValueError(f"Unsupported region: {region}. Supported regions: {list(region_map.keys())}")
        
        file_pattern = region_map[region_key]
        
        # Find matching files
        matching_files = []
        for filename in os.listdir(self.climate_data_dir):
            if file_pattern in filename and filename.endswith(".json"):
                matching_files.append(os.path.join(self.climate_data_dir, filename))
        
        if not matching_files:
            raise FileNotFoundError(f"No temperature data files found for region: {region}")
        
        # Use the most recent file if multiple matches
        data_file = sorted(matching_files)[-1]
        logger.info(f"Loading temperature data from: {data_file}")
        
        # Load JSON data
        with open(data_file, 'r') as f:
            json_data = json.load(f)
        
        # Extract metadata
        description = json_data.get("description", {})
        title = description.get("title", "")
        units = description.get("units", "")
        base_period = description.get("base_period", "")
        
        # Extract time series data
        data_points = json_data.get("data", {})
        
        # Convert to DataFrame
        records = []
        for date_str, values in data_points.items():
            year = int(date_str[:4])
            month = int(date_str[4:6])
            date = datetime(year, month, 1)
            
            record = {
                "date": date,
                "temperature": values.get("value"),
                "anomaly": values.get("anomaly"),
                "region": region
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add metadata as attributes
        df.attrs["title"] = title
        df.attrs["units"] = units
        df.attrs["base_period"] = base_period
        
        return df
    
    def visualize_temperature_trends(self, 
                                    regions: List[str] = None,
                                    save_fig: bool = True,
                                    show_fig: bool = True) -> Optional[Figure]:
        """
        Visualize temperature trends for multiple regions.
        
        Args:
            regions: List of regions to visualize
            save_fig: Whether to save the figure to a file
            show_fig: Whether to display the figure
            
        Returns:
            Matplotlib figure if show_fig is True, None otherwise
        """
        if regions is None:
            regions = ["MA", "NE"]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot each region
        for region in regions:
            try:
                df = self.load_temperature_data(region)
                
                # Sort by date
                df = df.sort_values("date")
                
                # Plot temperature
                plt.plot(df["date"], df["temperature"], 
                         marker='o', markersize=4, linestyle='-', 
                         label=f"{region} Temperature")
                
                # Calculate trend using linear regression
                x = np.arange(len(df))
                slope, intercept = np.polyfit(df["date"].astype(np.int64) // 10**9, 
                                             df["temperature"], 1)
                
                # Plot trend line
                trend_line = slope * (df["date"].astype(np.int64) // 10**9) + intercept
                plt.plot(df["date"], trend_line, '--', 
                         label=f"{region} Trend (slope: {slope:.6f})")
                
                logger.info(f"Region {region}: Trend slope = {slope:.6f} degrees/second")
                
                # Calculate warming rate per decade
                seconds_per_decade = 10 * 365.25 * 24 * 60 * 60
                warming_per_decade = slope * seconds_per_decade
                logger.info(f"Region {region}: Warming rate = {warming_per_decade:.2f} degrees/decade")
                
            except Exception as e:
                logger.error(f"Error processing {region} data: {e}")
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Temperature (°F)')
        plt.title('Temperature Trends by Region')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key events
        plt.annotate('2024: Highest recorded\ntemperatures', 
                    xy=(datetime(2024, 7, 1), 50.8), 
                    xytext=(datetime(2022, 7, 1), 51.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        # Save figure if requested
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temperature_trends_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved temperature trends visualization to {filepath}")
        
        # Show figure if requested
        if show_fig:
            plt.tight_layout()
            plt.show()
            return plt.gcf()
        else:
            plt.close()
            return None
    
    def visualize_temperature_anomalies(self, 
                                       regions: List[str] = None,
                                       save_fig: bool = True,
                                       show_fig: bool = True) -> Optional[Figure]:
        """
        Visualize temperature anomalies for multiple regions.
        
        Args:
            regions: List of regions to visualize
            save_fig: Whether to save the figure to a file
            show_fig: Whether to display the figure
            
        Returns:
            Matplotlib figure if show_fig is True, None otherwise
        """
        if regions is None:
            regions = ["MA", "NE"]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot each region
        for region in regions:
            try:
                df = self.load_temperature_data(region)
                
                # Sort by date
                df = df.sort_values("date")
                
                # Plot anomaly
                plt.plot(df["date"], df["anomaly"], 
                         marker='o', markersize=4, linestyle='-', 
                         label=f"{region} Anomaly")
                
                # Calculate trend using linear regression
                x = np.arange(len(df))
                slope, intercept = np.polyfit(df["date"].astype(np.int64) // 10**9, 
                                             df["anomaly"], 1)
                
                # Plot trend line
                trend_line = slope * (df["date"].astype(np.int64) // 10**9) + intercept
                plt.plot(df["date"], trend_line, '--', 
                         label=f"{region} Trend (slope: {slope:.6f})")
                
                logger.info(f"Region {region}: Anomaly trend slope = {slope:.6f} degrees/second")
                
                # Calculate warming rate per decade
                seconds_per_decade = 10 * 365.25 * 24 * 60 * 60
                warming_per_decade = slope * seconds_per_decade
                logger.info(f"Region {region}: Anomaly rate = {warming_per_decade:.2f} degrees/decade")
                
            except Exception as e:
                logger.error(f"Error processing {region} data: {e}")
        
        # Add zero line
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Temperature Anomaly (°F)')
        plt.title('Temperature Anomalies by Region')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key events
        plt.annotate('2024: Highest recorded\nanomalies', 
                    xy=(datetime(2024, 7, 1), 2.0), 
                    xytext=(datetime(2022, 7, 1), 2.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        # Save figure if requested
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temperature_anomalies_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved temperature anomalies visualization to {filepath}")
        
        # Show figure if requested
        if show_fig:
            plt.tight_layout()
            plt.show()
            return plt.gcf()
        else:
            plt.close()
            return None
    
    def visualize_decadal_changes(self, 
                                 regions: List[str] = None,
                                 save_fig: bool = True,
                                 show_fig: bool = True) -> Optional[Figure]:
        """
        Visualize decadal temperature changes for multiple regions.
        
        Args:
            regions: List of regions to visualize
            save_fig: Whether to save the figure to a file
            show_fig: Whether to display the figure
            
        Returns:
            Matplotlib figure if show_fig is True, None otherwise
        """
        if regions is None:
            regions = ["MA", "NE"]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Define decades
        decades = [
            ("1990s", 1990, 1999),
            ("2000s", 2000, 2009),
            ("2010s", 2010, 2019),
            ("2020s", 2020, 2029)
        ]
        
        # Store data for bar chart
        decade_data = {region: [] for region in regions}
        decade_labels = []
        
        # Process each region
        for region in regions:
            try:
                df = self.load_temperature_data(region)
                
                # Calculate average temperature for each decade
                for decade_name, start_year, end_year in decades:
                    decade_mask = (df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)
                    if decade_mask.any():
                        avg_temp = df.loc[decade_mask, "temperature"].mean()
                        decade_data[region].append(avg_temp)
                        
                        if decade_name not in decade_labels:
                            decade_labels.append(decade_name)
                
            except Exception as e:
                logger.error(f"Error processing {region} data: {e}")
        
        # Set up bar positions
        bar_width = 0.35
        x = np.arange(len(decade_labels))
        
        # Plot bars for each region
        for i, region in enumerate(regions):
            plt.bar(x + i*bar_width, decade_data[region], 
                   width=bar_width, label=region)
        
        # Add labels and title
        plt.xlabel('Decade')
        plt.ylabel('Average Temperature (°F)')
        plt.title('Decadal Temperature Changes by Region')
        plt.xticks(x + bar_width/2, decade_labels)
        plt.legend()
        
        # Add value labels on bars
        for i, region in enumerate(regions):
            for j, value in enumerate(decade_data[region]):
                plt.text(j + i*bar_width, value + 0.1, f'{value:.1f}', 
                        ha='center', va='bottom')
        
        # Save figure if requested
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"decadal_changes_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved decadal changes visualization to {filepath}")
        
        # Show figure if requested
        if show_fig:
            plt.tight_layout()
            plt.show()
            return plt.gcf()
        else:
            plt.close()
            return None
    
    def extract_climate_risk_keywords(self, region: Optional[str] = None) -> Dict[str, int]:
        """
        Extract and count climate risk keywords from documents.
        
        Args:
            region: Optional region to filter documents by
            
        Returns:
            Dictionary mapping keywords to their occurrence counts
        """
        # Define climate-related keywords to search for
        keywords = [
            "sea level rise", "flooding", "erosion", "temperature", 
            "warming", "climate change", "extreme weather", "precipitation",
            "drought", "heat wave", "coastal", "storm surge", "vulnerability",
            "adaptation", "mitigation", "resilience"
        ]
        
        # Initialize counts
        keyword_counts = {keyword: 0 for keyword in keywords}
        
        # Get all text files in the climate risk directory
        for filename in os.listdir(self.climate_risk_dir):
            if not filename.endswith(".txt"):
                continue
            
            file_path = os.path.join(self.climate_risk_dir, filename)
            
            # Read document content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract document region
            doc_region = self._extract_region_from_document(content, filename)
            
            # Filter by region if specified
            if region and region.lower() not in doc_region.lower():
                continue
            
            # Count keyword occurrences
            for keyword in keywords:
                count = content.lower().count(keyword.lower())
                keyword_counts[keyword] += count
        
        # Remove keywords with zero occurrences
        keyword_counts = {k: v for k, v in keyword_counts.items() if v > 0}
        
        return keyword_counts
    
    def _extract_region_from_document(self, content: str, filename: str) -> str:
        """
        Extract region information from document content or filename.
        
        Args:
            content: Document content
            filename: Document filename
            
        Returns:
            Region string
        """
        # Try to extract from content
        region_match = re.search(r"Region:\s*([^,\n]+)", content)
        if region_match:
            return region_match.group(1).strip()
        
        # Try to extract from first few lines
        first_lines = content.split("\n")[:5]
        for line in first_lines:
            if "cape cod" in line.lower():
                return "Cape Cod, Massachusetts"
            if "massachusetts" in line.lower() or "ma" in line.lower():
                return "Massachusetts"
            if "northeast" in line.lower() or "new england" in line.lower():
                return "Northeast"
        
        # Extract from filename
        if "cape_code" in filename.lower() or "cape_cod" in filename.lower():
            return "Cape Cod, Massachusetts"
        if "marthas_vineyard" in filename.lower() or "vineyard" in filename.lower():
            return "Martha's Vineyard, Massachusetts"
        if "boston" in filename.lower():
            return "Boston, Massachusetts"
        if "nantucket" in filename.lower():
            return "Nantucket, Massachusetts"
        if "plum_island" in filename.lower():
            return "Plum Island, Massachusetts"
        
        # Default
        return "Massachusetts"
    
    def visualize_climate_risk_keywords(self, 
                                       region: Optional[str] = None,
                                       save_fig: bool = True,
                                       show_fig: bool = True) -> Optional[Figure]:
        """
        Visualize climate risk keywords from documents.
        
        Args:
            region: Optional region to filter documents by
            save_fig: Whether to save the figure to a file
            show_fig: Whether to display the figure
            
        Returns:
            Matplotlib figure if show_fig is True, None otherwise
        """
        # Extract keywords
        keyword_counts = self.extract_climate_risk_keywords(region)
        
        if not keyword_counts:
            logger.warning("No keywords found in climate risk documents")
            return None
        
        # Sort keywords by count
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [k for k, v in sorted_keywords]
        counts = [v for k, v in sorted_keywords]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(keywords))
        plt.barh(y_pos, counts, align='center', alpha=0.7)
        plt.yticks(y_pos, keywords)
        
        # Add labels and title
        plt.xlabel('Occurrence Count')
        plt.title('Climate Risk Keywords in Documents')
        
        # Add count labels
        for i, count in enumerate(counts):
            plt.text(count + 0.1, i, str(count), va='center')
        
        # Save figure if requested
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            region_str = region.lower() if region else "all"
            filename = f"climate_risk_keywords_{region_str}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved climate risk keywords visualization to {filepath}")
        
        # Show figure if requested
        if show_fig:
            plt.tight_layout()
            plt.show()
            return plt.gcf()
        else:
            plt.close()
            return None
    
    def visualize_temperature_vs_risk(self, 
                                     region: str = "MA",
                                     save_fig: bool = True,
                                     show_fig: bool = True) -> Optional[Figure]:
        """
        Visualize temperature trends alongside climate risk keywords.
        
        Args:
            region: Region to visualize
            save_fig: Whether to save the figure to a file
            show_fig: Whether to display the figure
            
        Returns:
            Matplotlib figure if show_fig is True, None otherwise
        """
        try:
            # Load temperature data
            df = self.load_temperature_data(region)
            
            # Sort by date
            df = df.sort_values("date")
            
            # Extract climate risk keywords
            keyword_counts = self.extract_climate_risk_keywords(region)
            
            if not keyword_counts:
                logger.warning("No keywords found in climate risk documents")
                return None
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot temperature on top subplot
            ax1.plot(df["date"], df["temperature"], 
                    marker='o', markersize=4, linestyle='-', 
                    label=f"{region} Temperature")
            
            # Calculate trend using linear regression
            slope, intercept = np.polyfit(df["date"].astype(np.int64) // 10**9, 
                                         df["temperature"], 1)
            
            # Plot trend line
            trend_line = slope * (df["date"].astype(np.int64) // 10**9) + intercept
            ax1.plot(df["date"], trend_line, '--', 
                    label=f"Trend (slope: {slope:.6f})")
            
            # Add labels and title to top subplot
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Temperature (°F)')
            ax1.set_title(f'Temperature Trend for {region}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Sort keywords by count
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            keywords = [k for k, v in sorted_keywords[:10]]  # Top 10 keywords
            counts = [v for k, v in sorted_keywords[:10]]
            
            # Create horizontal bar chart on bottom subplot
            y_pos = np.arange(len(keywords))
            ax2.barh(y_pos, counts, align='center', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(keywords)
            
            # Add labels and title to bottom subplot
            ax2.set_xlabel('Occurrence Count')
            ax2.set_title('Top Climate Risk Keywords')
            
            # Add count labels
            for i, count in enumerate(counts):
                ax2.text(count + 0.1, i, str(count), va='center')
            
            # Save figure if requested
            if save_fig:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"temperature_vs_risk_{region.lower()}_{timestamp}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Saved temperature vs risk visualization to {filepath}")
            
            # Show figure if requested
            if show_fig:
                plt.tight_layout()
                plt.show()
                return fig
            else:
                plt.close()
                return None
            
        except Exception as e:
            logger.error(f"Error creating temperature vs risk visualization: {e}")
            return None


if __name__ == "__main__":
    # Create visualizer
    visualizer = RealClimateVisualizer()
    
    # Visualize temperature trends
    visualizer.visualize_temperature_trends()
    
    # Visualize temperature anomalies
    visualizer.visualize_temperature_anomalies()
    
    # Visualize decadal changes
    visualizer.visualize_decadal_changes()
    
    # Visualize climate risk keywords
    visualizer.visualize_climate_risk_keywords()
    
    # Visualize temperature vs risk
    visualizer.visualize_temperature_vs_risk("MA")
    visualizer.visualize_temperature_vs_risk("NE")
