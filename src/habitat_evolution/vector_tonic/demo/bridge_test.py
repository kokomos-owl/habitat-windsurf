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
from src.habitat_evolution.vector_tonic.bridge.pattern_domain_bridge import PatternDomainBridge
from src.habitat_evolution.vector_tonic.bridge.events import (
    create_statistical_pattern_detected_event,
    create_statistical_pattern_quality_changed_event
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_statistical_pattern(pattern_type: str, region: str, trend: str) -> Dict[str, Any]:
    """
    Create a mock statistical pattern for testing.
    
    Args:
        pattern_type: Type of pattern
        region: Region the pattern applies to
        trend: Trend of the pattern (increasing, decreasing, stable)
        
    Returns:
        Mock statistical pattern data
    """
    pattern_id = f"stat_pattern_{uuid.uuid4().hex[:8]}"
    
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


def create_mock_semantic_pattern(text: str, quality_state: str) -> Dict[str, Any]:
    """
    Create a mock semantic pattern for testing.
    
    Args:
        text: Pattern text
        quality_state: Quality state of the pattern
        
    Returns:
        Mock semantic pattern data
    """
    pattern_id = f"sem_pattern_{uuid.uuid4().hex[:8]}"
    
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
    logger.info("Starting pattern bridge test...")
    
    # Initialize components
    event_bus = LocalEventBus()
    time_provider = TimeProvider()
    pattern_bridge = PatternDomainBridge(event_bus, time_provider)
    
    # Create mock statistical patterns
    statistical_patterns = [
        create_mock_statistical_pattern("warming_trend", "Massachusetts", "increasing"),
        create_mock_statistical_pattern("temperature_anomaly", "Northeast", "increasing"),
        create_mock_statistical_pattern("seasonal_variation", "Massachusetts", "stable")
    ]
    
    # Create mock semantic patterns
    semantic_patterns = [
        create_mock_semantic_pattern(
            "Massachusetts has experienced significant warming since 2010 through 2023",
            "stable"
        ),
        create_mock_semantic_pattern(
            "Temperature anomalies in the Northeast region have increased dramatically by 2024",
            "emergent"
        ),
        create_mock_semantic_pattern(
            "Coastal flooding has become more frequent in Massachusetts since 2015",
            "hypothetical"
        )
    ]
    
    # Register patterns with the bridge
    logger.info("Registering patterns with the bridge...")
    
    # Register statistical patterns
    for pattern in statistical_patterns:
        event = create_statistical_pattern_detected_event(
            pattern_id=pattern["id"],
            pattern_data=pattern
        )
        event_bus.publish(event)
    
    # Register semantic patterns
    for pattern in semantic_patterns:
        # Create a proper Event object for semantic patterns
        event_data = {
            "pattern_id": pattern["id"],
            "pattern_text": pattern["text"],
            "quality_state": pattern["quality_state"],
            "confidence": pattern["confidence"],
            "temporal_markers": pattern["temporal_markers"],
            "metadata": pattern["metadata"]
        }
        
        event = Event.create(
            type="pattern_detected",
            data=event_data,
            source=pattern["source"]
        )
        
        # Call the bridge's handler directly
        pattern_bridge.on_semantic_pattern_detected(event)
    
    # Get co-patterns
    co_patterns = pattern_bridge.get_co_patterns()
    
    # Print results
    logger.info(f"Detected {len(co_patterns)} co-patterns:")
    
    for i, co_pattern in enumerate(co_patterns):
        logger.info(f"Co-Pattern {i+1}:")
        logger.info(f"  ID: {co_pattern['id']}")
        logger.info(f"  Quality: {co_pattern['quality_state']}")
        logger.info(f"  Correlation: {co_pattern['correlation_strength']:.2f} ({co_pattern['correlation_type']})")
        
        # Get related patterns
        stat_pattern = None
        sem_pattern = None
        
        for pattern in statistical_patterns:
            if pattern["id"] == co_pattern["statistical_pattern_id"]:
                stat_pattern = pattern
                break
        
        for pattern in semantic_patterns:
            if pattern["id"] == co_pattern["semantic_pattern_id"]:
                sem_pattern = pattern
                break
        
        if stat_pattern:
            logger.info(f"  Statistical: {stat_pattern.get('type', 'Unknown')} ({stat_pattern.get('region', 'Unknown')})")
        
        if sem_pattern:
            logger.info(f"  Semantic: {sem_pattern.get('text', 'Unknown')}")
        
        logger.info("")
    
    logger.info("Pattern bridge test completed")


if __name__ == "__main__":
    test_pattern_bridge()
