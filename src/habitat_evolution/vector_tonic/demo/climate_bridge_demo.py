"""
Climate Pattern Bridge Demo for Habitat Evolution.

This script demonstrates the integration of the vector-tonic statistical pattern
domain with the semantic pattern domain using Habitat's event bus. It shows how
to detect patterns in climate time series data and correlate them with semantic
patterns extracted from climate risk documents.
"""

import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from datetime import datetime

from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.core.services.time_provider import TimeProvider
from src.habitat_evolution.vector_tonic.bridge.pattern_domain_bridge import PatternDomainBridge
from src.habitat_evolution.vector_tonic.bridge.events import (
    create_statistical_pattern_detected_event,
    create_statistical_pattern_quality_changed_event
)
from src.habitat_evolution.vector_tonic.data.climate_data_loader import ClimateDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClimatePatternDetector:
    """
    Detects statistical patterns in climate time series data.
    """
    
    def __init__(self, event_bus: LocalEventBus):
        """
        Initialize the climate pattern detector.
        
        Args:
            event_bus: Habitat's event bus for publishing events
        """
        self.event_bus = event_bus
        self.data_loader = ClimateDataLoader()
        self.detected_patterns = {}
    
    def load_climate_data(self, region: str) -> pd.DataFrame:
        """
        Load climate data for a specific region.
        
        Args:
            region: Region to load data for
            
        Returns:
            DataFrame with climate data
        """
        # Use the ClimateDataLoader to get data
        # For demo purposes, we'll generate synthetic data if needed
        try:
            data = self.data_loader.load_data(region)
            logger.info(f"Loaded climate data for {region}")
            return data
        except Exception as e:
            logger.warning(f"Error loading data for {region}: {e}")
            logger.info(f"Generating synthetic data for {region}")
            return self._generate_synthetic_data(region)
    
    def _generate_synthetic_data(self, region: str) -> pd.DataFrame:
        """
        Generate synthetic climate data for demo purposes.
        
        Args:
            region: Region to generate data for
            
        Returns:
            DataFrame with synthetic climate data
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
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'temperature': temps,
            'region': region
        })
        
        return df
    
    def detect_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect statistical patterns in climate data.
        
        Args:
            data: DataFrame with climate data
            
        Returns:
            List of detected patterns
        """
        patterns = []
        region = data['region'].iloc[0] if 'region' in data.columns else 'Unknown'
        
        # Extract time series
        if 'temperature' in data.columns:
            # Detect temperature trends
            temp_patterns = self._detect_temperature_patterns(data, region)
            patterns.extend(temp_patterns)
        
        # Publish patterns to event bus
        for pattern in patterns:
            event = create_statistical_pattern_detected_event(
                pattern_id=pattern['id'],
                pattern_data=pattern
            )
            self.event_bus.publish(event)
            logger.info(f"Published pattern: {pattern['type']} in {pattern['region']}")
            
            # Store pattern
            self.detected_patterns[pattern['id']] = pattern
        
        return patterns
    
    def _detect_temperature_patterns(self, data: pd.DataFrame, region: str) -> List[Dict[str, Any]]:
        """
        Detect temperature patterns in climate data.
        
        Args:
            data: DataFrame with climate data
            region: Region the data is for
            
        Returns:
            List of temperature patterns
        """
        patterns = []
        
        # Extract temperature data
        temps = data['temperature'].values
        dates = data['date'].values
        
        # Calculate trend using linear regression
        x = np.arange(len(temps))
        slope, intercept = np.polyfit(x, temps, 1)
        
        # Determine trend direction and magnitude
        if slope > 0.01:  # Warming trend
            trend_type = "warming_trend"
            trend_direction = "increasing"
            magnitude = min(1.0, slope * 10)  # Scale magnitude to 0-1
        elif slope < -0.01:  # Cooling trend
            trend_type = "cooling_trend"
            trend_direction = "decreasing"
            magnitude = min(1.0, abs(slope) * 10)
        else:  # Stable
            trend_type = "stable_temperature"
            trend_direction = "stable"
            magnitude = 0.3
        
        # Create pattern
        pattern_id = f"stat_pattern_{uuid.uuid4().hex[:8]}"
        start_date = pd.to_datetime(dates[0]).strftime('%Y%m')
        end_date = pd.to_datetime(dates[-1]).strftime('%Y%m')
        
        # Add keywords for semantic matching
        keywords = []
        if trend_type == "warming_trend":
            keywords = ["warming", "temperature increase", "heat", "climate change"]
        elif trend_type == "cooling_trend":
            keywords = ["cooling", "temperature decrease", "cold"]
        else:
            keywords = ["stable", "consistent", "unchanged"]
        
        # Add region-specific keywords
        keywords.append(region.lower())
        
        pattern = {
            "id": pattern_id,
            "type": trend_type,
            "region": region,
            "trend": trend_direction,
            "start_time": start_date,
            "end_time": end_date,
            "magnitude": magnitude,
            "confidence": 0.8,
            "quality_state": "emergent",
            "keywords": keywords,
            "metadata": {
                "slope": slope,
                "intercept": intercept,
                "detection_method": "linear_regression"
            }
        }
        
        patterns.append(pattern)
        
        # Check for temperature anomalies
        z_scores = (temps - np.mean(temps)) / np.std(temps)
        anomaly_indices = np.where(np.abs(z_scores) > 2.0)[0]
        
        if len(anomaly_indices) > len(temps) * 0.05:  # If more than 5% are anomalies
            anomaly_pattern_id = f"stat_pattern_{uuid.uuid4().hex[:8]}"
            
            # Find the period with most anomalies
            if len(anomaly_indices) > 0:
                anomaly_start = pd.to_datetime(dates[anomaly_indices[0]]).strftime('%Y%m')
                anomaly_end = pd.to_datetime(dates[anomaly_indices[-1]]).strftime('%Y%m')
            else:
                anomaly_start = start_date
                anomaly_end = end_date
            
            anomaly_pattern = {
                "id": anomaly_pattern_id,
                "type": "temperature_anomaly",
                "region": region,
                "trend": "increasing" if np.mean(z_scores[anomaly_indices]) > 0 else "decreasing",
                "start_time": anomaly_start,
                "end_time": anomaly_end,
                "magnitude": min(1.0, np.mean(np.abs(z_scores[anomaly_indices])) / 3.0),
                "confidence": 0.7,
                "quality_state": "emergent",
                "keywords": ["anomaly", "unusual temperature", "extreme", region.lower()],
                "metadata": {
                    "mean_z_score": float(np.mean(np.abs(z_scores[anomaly_indices]))),
                    "anomaly_count": len(anomaly_indices),
                    "detection_method": "z_score"
                }
            }
            
            patterns.append(anomaly_pattern)
        
        return patterns
    
    def visualize_patterns(self, data: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """
        Visualize detected patterns in climate data.
        
        Args:
            data: DataFrame with climate data
            patterns: List of detected patterns
        """
        if 'temperature' not in data.columns or len(patterns) == 0:
            logger.warning("Cannot visualize: missing temperature data or patterns")
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
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
                plt.plot(data['date'], trend_line, 'r--', 
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
        plt.title(f'Climate Patterns for {data["region"].iloc[0]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show plot
        plt.tight_layout()
        plt.show()


class ClimateRiskDocumentProcessor:
    """
    Processes climate risk documents and extracts semantic patterns.
    """
    
    def __init__(self, event_bus: LocalEventBus):
        """
        Initialize the climate risk document processor.
        
        Args:
            event_bus: Habitat's event bus for publishing events
        """
        self.event_bus = event_bus
        self.documents = []
        self.extracted_patterns = {}
    
    def load_documents(self):
        """Load sample climate risk documents for demo purposes."""
        # In a real system, this would load actual documents from a database
        # For demo purposes, we'll create some sample documents
        self.documents = [
            {
                "id": "doc001",
                "title": "Climate Risk Assessment for Massachusetts",
                "content": """
                Massachusetts has experienced significant warming since 2010 through 2023, 
                with average temperatures rising faster than the global average. This trend 
                is projected to continue, with potential impacts on agriculture, public health, 
                and coastal infrastructure. Extreme weather events have become more frequent, 
                with increased precipitation intensity and more frequent heat waves.
                """,
                "source": "Massachusetts Climate Change Adaptation Report",
                "date": "2024-01-15"
            },
            {
                "id": "doc002",
                "title": "Northeast Regional Climate Trends",
                "content": """
                Temperature anomalies in the Northeast region have increased dramatically by 2024,
                with winter temperatures showing the greatest change. The frequency of extreme 
                precipitation events has increased by 55% since 1958. Sea levels along the 
                Northeast coast have risen by more than a foot since 1900, increasing the risk 
                of coastal flooding during storms.
                """,
                "source": "Northeast Regional Climate Center",
                "date": "2023-11-30"
            },
            {
                "id": "doc003",
                "title": "Coastal Flooding Risk in Massachusetts",
                "content": """
                Coastal flooding has become more frequent in Massachusetts since 2015, 
                affecting communities from Boston to Cape Cod. Rising sea levels combined 
                with more intense storms have led to increased erosion and property damage. 
                The Massachusetts coast has experienced a 15% increase in high-tide flooding 
                events compared to the previous decade.
                """,
                "source": "Massachusetts Office of Coastal Zone Management",
                "date": "2023-08-22"
            }
        ]
        
        logger.info(f"Loaded {len(self.documents)} climate risk documents")
        return self.documents
    
    def extract_patterns(self) -> List[Dict[str, Any]]:
        """
        Extract semantic patterns from climate risk documents.
        
        Returns:
            List of extracted semantic patterns
        """
        patterns = []
        
        for doc in self.documents:
            # Extract sentences that might contain patterns
            sentences = [s.strip() for s in doc["content"].split(".") if len(s.strip()) > 10]
            
            for sentence in sentences:
                # Simple heuristic: sentences with temporal markers likely contain patterns
                temporal_markers = self._extract_temporal_markers(sentence)
                
                if temporal_markers:
                    pattern_id = f"sem_pattern_{uuid.uuid4().hex[:8]}"
                    
                    # Determine quality state based on language
                    quality_state = "stable"
                    if any(word in sentence.lower() for word in ["projected", "expected", "future", "potential"]):
                        quality_state = "hypothetical"
                    elif any(word in sentence.lower() for word in ["increasing", "rising", "growing"]):
                        quality_state = "emergent"
                    
                    # Create pattern
                    pattern = {
                        "id": pattern_id,
                        "text": sentence,
                        "quality_state": quality_state,
                        "confidence": 0.8,
                        "source": doc["source"],
                        "temporal_markers": temporal_markers,
                        "metadata": {
                            "document_id": doc["id"],
                            "extraction_date": datetime.now().strftime("%Y-%m-%d")
                        }
                    }
                    
                    patterns.append(pattern)
                    self.extracted_patterns[pattern_id] = pattern
                    
                    # Publish pattern to event bus
                    event_data = {
                        "pattern_id": pattern["id"],
                        "pattern_text": pattern["text"],
                        "quality_state": pattern["quality_state"],
                        "confidence": pattern["confidence"],
                        "temporal_markers": pattern["temporal_markers"],
                        "metadata": pattern["metadata"]
                    }
                    
                    event = Event.create(
                        type="semantic_pattern_detected",
                        data=event_data,
                        source=pattern["source"]
                    )
                    
                    self.event_bus.publish(event)
                    logger.info(f"Published semantic pattern: {pattern_id}")
        
        return patterns
    
    def _extract_temporal_markers(self, text: str) -> List[Dict[str, str]]:
        """
        Extract temporal markers from text.
        
        Args:
            text: Text to extract markers from
            
        Returns:
            List of temporal markers
        """
        markers = []
        text_lower = text.lower()
        
        # Check for year mentions
        for year in range(2000, 2025):
            year_str = str(year)
            if year_str in text_lower:
                markers.append({"time": f"{year}01", "text": year_str})
        
        # Check for decade mentions
        if "since 2000" in text_lower:
            markers.append({"time": "200001", "text": "since 2000"})
        if "since 2010" in text_lower:
            markers.append({"time": "201001", "text": "since 2010"})
        if "since 2020" in text_lower:
            markers.append({"time": "202001", "text": "since 2020"})
            
        # Check for specific time periods
        if "through 2023" in text_lower:
            markers.append({"time": "202312", "text": "through 2023"})
        if "by 2024" in text_lower:
            markers.append({"time": "202401", "text": "by 2024"})
        if "since 2015" in text_lower:
            markers.append({"time": "201501", "text": "since 2015"})
        
        return markers


def run_climate_bridge_demo():
    """Run the climate pattern bridge demo."""
    logger.info("Starting Climate Pattern Bridge Demo...")
    
    # Initialize components
    event_bus = LocalEventBus()
    time_provider = TimeProvider()
    pattern_bridge = PatternDomainBridge(event_bus, time_provider)
    
    # Initialize pattern detectors
    climate_detector = ClimatePatternDetector(event_bus)
    document_processor = ClimateRiskDocumentProcessor(event_bus)
    
    # Load climate data for Massachusetts
    data = climate_detector.load_climate_data("Massachusetts")
    
    # Detect statistical patterns
    statistical_patterns = climate_detector.detect_patterns(data)
    logger.info(f"Detected {len(statistical_patterns)} statistical patterns")
    
    # Load and process climate risk documents
    document_processor.load_documents()
    semantic_patterns = document_processor.extract_patterns()
    logger.info(f"Extracted {len(semantic_patterns)} semantic patterns")
    
    # Get co-patterns
    co_patterns = pattern_bridge.get_co_patterns()
    logger.info(f"Detected {len(co_patterns)} co-patterns")
    
    # Print co-patterns
    for i, co_pattern in enumerate(co_patterns):
        logger.info(f"Co-Pattern {i+1}:")
        logger.info(f"  ID: {co_pattern.get('id', 'Unknown')}")
        
        # Get related patterns
        stat_pattern_id = co_pattern.get("statistical_pattern_id")
        sem_pattern_id = co_pattern.get("semantic_pattern_id")
        
        stat_pattern = climate_detector.detected_patterns.get(stat_pattern_id, {})
        sem_pattern = document_processor.extracted_patterns.get(sem_pattern_id, {})
        
        logger.info(f"  Statistical: {stat_pattern.get('type', 'Unknown')} in {stat_pattern.get('region', 'Unknown')}")
        logger.info(f"  Semantic: {sem_pattern.get('text', 'Unknown')[:100]}...")
        logger.info(f"  Correlation: {co_pattern.get('correlation_strength', 0.0):.2f} ({co_pattern.get('correlation_type', 'Unknown')})")
        logger.info(f"  Quality: {co_pattern.get('quality_state', 'Unknown')}")
        logger.info("")
    
    # Visualize patterns
    try:
        climate_detector.visualize_patterns(data, statistical_patterns)
        logger.info("Visualization displayed successfully")
    except Exception as e:
        logger.error(f"Error displaying visualization: {e}")
    
    logger.info("Climate Pattern Bridge Demo completed")


if __name__ == "__main__":
    run_climate_bridge_demo()
