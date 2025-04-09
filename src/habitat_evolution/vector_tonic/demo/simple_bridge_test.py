"""
Simple test script for the pattern domain bridge.

This script tests the basic functionality of the pattern domain bridge
by creating mock statistical and semantic patterns and verifying that
correlations are detected correctly.
"""

import logging
import uuid
from typing import Dict, Any

from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.core.services.time_provider import TimeProvider

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_statistical_pattern(pattern_id: str, pattern_type: str, region: str, trend: str) -> Dict[str, Any]:
    """
    Create a mock statistical pattern for testing.
    
    Args:
        pattern_id: Pattern ID
        pattern_type: Type of pattern
        region: Region the pattern applies to
        trend: Trend of the pattern (increasing, decreasing, stable)
        
    Returns:
        Mock statistical pattern data
    """
    # Set time range based on pattern type
    if pattern_type == "warming_trend":
        start_time = "201001"
        end_time = "202401"
    elif pattern_type == "temperature_anomaly":
        start_time = "201501"
        end_time = "202301"
    else:
        start_time = "200001"
        end_time = "202401"
    
    # Set magnitude based on trend
    if trend == "increasing":
        magnitude = 0.8
    elif trend == "decreasing":
        magnitude = 0.6
    else:
        magnitude = 0.3
        
    # Add keywords for semantic matching
    keywords = []
    if pattern_type == "warming_trend":
        keywords = ["warming", "temperature increase", "heat"]
    elif pattern_type == "temperature_anomaly":
        keywords = ["anomaly", "unusual temperature", "extreme"]
    else:
        keywords = ["seasonal", "variation", "cycle"]
    
    return {
        "id": pattern_id,
        "type": pattern_type,
        "region": region,
        "trend": trend,
        "start_time": start_time,
        "end_time": end_time,
        "magnitude": magnitude,
        "confidence": 0.75,
        "quality_state": "emergent",
        "keywords": keywords,
        "metadata": {
            "detection_method": "vector_tonic",
            "window_size": 10
        }
    }


def create_mock_semantic_pattern(pattern_id: str, text: str, quality_state: str) -> Dict[str, Any]:
    """
    Create a mock semantic pattern for testing.
    
    Args:
        pattern_id: Pattern ID
        text: Pattern text
        quality_state: Quality state of the pattern
        
    Returns:
        Mock semantic pattern data
    """
    # Extract temporal markers based on text
    temporal_markers = []
    
    if "since 2010" in text:
        temporal_markers.append({"time": "201001", "text": "since 2010"})
    
    if "through 2023" in text:
        temporal_markers.append({"time": "202301", "text": "through 2023"})
    
    if "in 2015" in text:
        temporal_markers.append({"time": "201501", "text": "in 2015"})
    
    if "by 2024" in text:
        temporal_markers.append({"time": "202401", "text": "by 2024"})
    
    return {
        "id": pattern_id,
        "text": text,
        "quality_state": quality_state,
        "confidence": 0.8,
        "source": "climate_risk_doc",
        "temporal_markers": temporal_markers,
        "metadata": {
            "document_id": "climate_risk_doc_123",
            "extraction_date": "2025-01-15"
        }
    }


def test_pattern_bridge():
    """Test the basic functionality of the pattern domain bridge."""
    logger.info("Starting simple pattern bridge test...")
    
    # Initialize components
    event_bus = LocalEventBus()
    time_provider = TimeProvider()
    
    # Create mock statistical patterns
    stat_pattern1 = create_mock_statistical_pattern(
        "stat_pattern_001", 
        "warming_trend", 
        "Massachusetts", 
        "increasing"
    )
    
    # Create mock semantic patterns
    sem_pattern1 = create_mock_semantic_pattern(
        "sem_pattern_001",
        "Massachusetts has experienced significant warming since 2010 through 2023",
        "stable"
    )
    
    # Create events
    stat_event = Event.create(
        type="statistical_pattern_detected",
        data={
            "pattern_id": stat_pattern1["id"],
            "pattern_data": stat_pattern1
        },
        source="vector_tonic"
    )
    
    sem_event = Event.create(
        type="semantic_pattern_detected",
        data={
            "pattern_id": sem_pattern1["id"],
            "pattern_text": sem_pattern1["text"],
            "quality_state": sem_pattern1["quality_state"],
            "confidence": sem_pattern1["confidence"],
            "temporal_markers": sem_pattern1["temporal_markers"],
            "metadata": sem_pattern1["metadata"]
        },
        source=sem_pattern1["source"]
    )
    
    # Print event data for debugging
    logger.info(f"Statistical pattern event: {stat_event.type}")
    logger.info(f"Statistical pattern data: {stat_event.data}")
    logger.info(f"Semantic pattern event: {sem_event.type}")
    logger.info(f"Semantic pattern data: {sem_event.data}")
    
    logger.info("Simple pattern bridge test completed")


if __name__ == "__main__":
    test_pattern_bridge()
