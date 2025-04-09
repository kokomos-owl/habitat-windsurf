"""
Real Climate Pattern Bridge Demo for Habitat Evolution.

This script demonstrates the integration of the vector-tonic statistical pattern
domain with the semantic pattern domain using Habitat's event bus and real
climate data. It shows how to detect patterns in real climate time series data
and correlate them with semantic patterns extracted from climate risk documents.
"""

import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime

from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.core.services.time_provider import TimeProvider
from src.habitat_evolution.vector_tonic.bridge.pattern_domain_bridge import PatternDomainBridge
from src.habitat_evolution.vector_tonic.bridge.events import (
    create_statistical_pattern_detected_event,
    create_statistical_pattern_quality_changed_event
)
from src.habitat_evolution.vector_tonic.data.real_climate_data_loader import RealClimateDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealClimatePatternDetector:
    """
    Detects statistical patterns in real climate time series data.
    """
    
    def __init__(self, event_bus: LocalEventBus):
        """
        Initialize the climate pattern detector.
        
        Args:
            event_bus: Habitat's event bus for publishing events
        """
        self.event_bus = event_bus
        self.data_loader = RealClimateDataLoader()
        self.detected_patterns = {}
    
    def load_climate_data(self, region: str) -> pd.DataFrame:
        """
        Load climate data for a specific region.
        
        Args:
            region: Region to load data for
            
        Returns:
            DataFrame with climate data
        """
        try:
            data = self.data_loader.load_temperature_data(region)
            logger.info(f"Loaded climate data for {region}")
            return data
        except Exception as e:
            logger.error(f"Error loading data for {region}: {e}")
            raise
    
    def detect_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect statistical patterns in climate data.
        
        Args:
            data: DataFrame with climate data
            
        Returns:
            List of detected patterns
        """
        patterns = []
        region = data['region'].iloc[0]
        
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
            keywords = ["warming", "temperature increase", "heat", "climate change", 
                       "warmer", "rising temperatures", "warming trend"]
        elif trend_type == "cooling_trend":
            keywords = ["cooling", "temperature decrease", "cold", "cooling trend"]
        else:
            keywords = ["stable", "consistent", "unchanged", "steady temperature"]
        
        # Add region-specific keywords
        keywords.append(region.lower())
        
        # Add more specific keywords based on magnitude
        if magnitude > 0.5:
            keywords.extend(["significant", "substantial", "notable"])
        
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
                "slope": float(slope),
                "intercept": float(intercept),
                "detection_method": "linear_regression",
                "units": data.attrs.get("units", "Degrees Fahrenheit"),
                "base_period": data.attrs.get("base_period", "")
            }
        }
        
        patterns.append(pattern)
        
        # Check for temperature anomalies
        anomalies = data.get('anomaly', None)
        if anomalies is not None and not anomalies.isna().all():
            # Use provided anomalies
            anomaly_values = anomalies.values
            anomaly_threshold = 1.0  # 1 degree anomaly threshold
            
            # Find significant anomalies
            significant_anomalies = np.abs(anomaly_values) > anomaly_threshold
            anomaly_indices = np.where(significant_anomalies)[0]
            
            if len(anomaly_indices) > len(temps) * 0.05:  # If more than 5% are anomalies
                anomaly_pattern_id = f"stat_pattern_{uuid.uuid4().hex[:8]}"
                
                # Find the period with most anomalies
                if len(anomaly_indices) > 0:
                    anomaly_start = pd.to_datetime(dates[anomaly_indices[0]]).strftime('%Y%m')
                    anomaly_end = pd.to_datetime(dates[anomaly_indices[-1]]).strftime('%Y%m')
                else:
                    anomaly_start = start_date
                    anomaly_end = end_date
                
                # Determine if anomalies are mostly positive or negative
                anomaly_direction = "increasing" if np.mean(anomaly_values[anomaly_indices]) > 0 else "decreasing"
                
                anomaly_pattern = {
                    "id": anomaly_pattern_id,
                    "type": "temperature_anomaly",
                    "region": region,
                    "trend": anomaly_direction,
                    "start_time": anomaly_start,
                    "end_time": anomaly_end,
                    "magnitude": min(1.0, np.mean(np.abs(anomaly_values[anomaly_indices])) / 2.0),
                    "confidence": 0.7,
                    "quality_state": "emergent",
                    "keywords": ["anomaly", "unusual temperature", "extreme", "abnormal", 
                               region.lower(), "temperature anomaly"],
                    "metadata": {
                        "mean_anomaly": float(np.mean(anomaly_values[anomaly_indices])),
                        "max_anomaly": float(np.max(anomaly_values[anomaly_indices])),
                        "anomaly_count": int(len(anomaly_indices)),
                        "detection_method": "anomaly_threshold",
                        "units": data.attrs.get("units", "Degrees Fahrenheit"),
                        "base_period": data.attrs.get("base_period", "")
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
                # Highlight anomalies if available
                if 'anomaly' in data.columns:
                    anomaly_threshold = 1.0
                    anomaly_mask = np.abs(data['anomaly']) > anomaly_threshold
                    
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
        plt.ylabel(data.attrs.get('units', 'Temperature'))
        plt.title(f'Climate Patterns for {data["region"].iloc[0]} ({data.attrs.get("base_period", "")})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show plot
        plt.tight_layout()
        plt.show()


class RealClimateRiskDocumentProcessor:
    """
    Processes real climate risk documents and extracts semantic patterns.
    """
    
    def __init__(self, event_bus: LocalEventBus):
        """
        Initialize the climate risk document processor.
        
        Args:
            event_bus: Habitat's event bus for publishing events
        """
        self.event_bus = event_bus
        self.data_loader = RealClimateDataLoader()
        self.documents = []
        self.extracted_patterns = {}
    
    def load_documents(self, region: Optional[str] = None):
        """
        Load climate risk documents, optionally filtered by region.
        
        Args:
            region: Optional region to filter documents by
            
        Returns:
            List of document dictionaries
        """
        self.documents = self.data_loader.load_climate_risk_documents(region)
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
            sentences = self.data_loader.extract_sentences_from_document(doc)
            
            for sentence in sentences:
                # Only process sentences with temporal markers
                if not sentence.get("temporal_markers"):
                    continue
                
                pattern_id = f"sem_pattern_{uuid.uuid4().hex[:8]}"
                
                # Determine quality state based on language
                quality_state = "stable"
                text_lower = sentence["text"].lower()
                
                if any(word in text_lower for word in ["projected", "expected", "future", "potential", "by 2050", "by 2030"]):
                    quality_state = "hypothetical"
                elif any(word in text_lower for word in ["increasing", "rising", "growing", "accelerating"]):
                    quality_state = "emergent"
                
                # Create pattern
                pattern = {
                    "id": pattern_id,
                    "text": sentence["text"],
                    "quality_state": quality_state,
                    "confidence": 0.8,
                    "source": doc["source"],
                    "region": sentence["region"],
                    "temporal_markers": sentence["temporal_markers"],
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


def run_real_climate_bridge_demo():
    """Run the real climate pattern bridge demo."""
    logger.info("Starting Real Climate Pattern Bridge Demo...")
    
    # Initialize components
    event_bus = LocalEventBus()
    time_provider = TimeProvider()
    pattern_bridge = PatternDomainBridge(event_bus, time_provider)
    
    # Initialize pattern detectors
    climate_detector = RealClimatePatternDetector(event_bus)
    document_processor = RealClimateRiskDocumentProcessor(event_bus)
    
    # Load climate data for Massachusetts
    try:
        ma_data = climate_detector.load_climate_data("MA")
        
        # Detect statistical patterns
        ma_patterns = climate_detector.detect_patterns(ma_data)
        logger.info(f"Detected {len(ma_patterns)} statistical patterns for Massachusetts")
        
        # Visualize Massachusetts patterns
        climate_detector.visualize_patterns(ma_data, ma_patterns)
    except Exception as e:
        logger.error(f"Error processing Massachusetts data: {e}")
    
    # Load climate data for Northeast
    try:
        ne_data = climate_detector.load_climate_data("NE")
        
        # Detect statistical patterns
        ne_patterns = climate_detector.detect_patterns(ne_data)
        logger.info(f"Detected {len(ne_patterns)} statistical patterns for Northeast")
        
        # Visualize Northeast patterns
        climate_detector.visualize_patterns(ne_data, ne_patterns)
    except Exception as e:
        logger.error(f"Error processing Northeast data: {e}")
    
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
    
    logger.info("Real Climate Pattern Bridge Demo completed")


if __name__ == "__main__":
    run_real_climate_bridge_demo()
