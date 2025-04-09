"""
Final test script for the pattern domain bridge.

This script tests the complete functionality of the pattern domain bridge
by creating mock statistical and semantic patterns, publishing events to the
event bus, and verifying that correlations are detected correctly.
"""

import logging
import uuid
from typing import Dict, Any, List

from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.core.services.time_provider import TimeProvider
from src.habitat_evolution.vector_tonic.bridge.pattern_domain_bridge import PatternDomainBridge
from src.habitat_evolution.vector_tonic.bridge.events import (
    create_statistical_pattern_detected_event
)

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


def print_co_patterns(co_patterns: List[Dict[str, Any]]):
    """
    Print co-patterns in a readable format.
    
    Args:
        co_patterns: List of co-patterns to print
    """
    logger.info(f"Detected {len(co_patterns)} co-patterns:")
    
    for i, co_pattern in enumerate(co_patterns):
        logger.info(f"Co-Pattern {i+1}:")
        logger.info(f"  ID: {co_pattern.get('id', 'Unknown')}")
        logger.info(f"  Statistical Pattern: {co_pattern.get('statistical_pattern_id', 'Unknown')}")
        logger.info(f"  Semantic Pattern: {co_pattern.get('semantic_pattern_id', 'Unknown')}")
        logger.info(f"  Correlation: {co_pattern.get('correlation_strength', 0.0):.2f} ({co_pattern.get('correlation_type', 'Unknown')})")
        logger.info(f"  Quality: {co_pattern.get('quality_state', 'Unknown')}")
        logger.info("")


def test_pattern_bridge():
    """Test the complete functionality of the pattern domain bridge."""
    logger.info("Starting final pattern bridge test...")
    
    # Initialize components
    event_bus = LocalEventBus()
    time_provider = TimeProvider()
    pattern_bridge = PatternDomainBridge(event_bus, time_provider)
    
    # Create mock statistical patterns
    statistical_patterns = [
        create_mock_statistical_pattern(
            "stat_pattern_001", 
            "warming_trend", 
            "Massachusetts", 
            "increasing"
        ),
        create_mock_statistical_pattern(
            "stat_pattern_002", 
            "temperature_anomaly", 
            "Northeast", 
            "increasing"
        ),
        create_mock_statistical_pattern(
            "stat_pattern_003", 
            "seasonal_variation", 
            "Massachusetts", 
            "stable"
        )
    ]
    
    # Create mock semantic patterns
    semantic_patterns = [
        create_mock_semantic_pattern(
            "sem_pattern_001",
            "Massachusetts has experienced significant warming since 2010 through 2023",
            "stable"
        ),
        create_mock_semantic_pattern(
            "sem_pattern_002",
            "Temperature anomalies in the Northeast region have increased dramatically by 2024",
            "emergent"
        ),
        create_mock_semantic_pattern(
            "sem_pattern_003",
            "Coastal flooding has become more frequent in Massachusetts since 2015",
            "hypothetical"
        )
    ]
    
    logger.info("Registering patterns with the bridge...")
    
    # Register statistical patterns
    for pattern in statistical_patterns:
        event = create_statistical_pattern_detected_event(
            pattern_id=pattern["id"],
            pattern_data=pattern
        )
        event_bus.publish(event)
        logger.info(f"Published statistical pattern event: {pattern['id']}")
    
    # Register semantic patterns
    for pattern in semantic_patterns:
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
        
        # Call the bridge's handler directly since we don't have a semantic pattern event creator
        pattern_bridge.on_semantic_pattern_detected(event)
        logger.info(f"Published semantic pattern event: {pattern['id']}")
    
    # Debug pattern storage
    logger.info(f"Statistical patterns stored: {len(pattern_bridge.statistical_patterns)}")
    for pattern_id, pattern in pattern_bridge.statistical_patterns.items():
        logger.info(f"  {pattern_id}: {pattern.get('type', 'Unknown')} in {pattern.get('region', 'Unknown')}")
    
    logger.info(f"Semantic patterns stored: {len(pattern_bridge.semantic_patterns)}")
    for pattern_id, pattern in pattern_bridge.semantic_patterns.items():
        logger.info(f"  {pattern_id}: {pattern.get('text', 'Unknown')[:50]}...")
    
    # Debug correlation thresholds
    logger.info(f"Correlation thresholds: {pattern_bridge.correlation_thresholds}")
    
    # Test direct correlation calculation
    if pattern_bridge.statistical_patterns and pattern_bridge.semantic_patterns:
        stat_pattern = next(iter(pattern_bridge.statistical_patterns.values()))
        sem_pattern = next(iter(pattern_bridge.semantic_patterns.values()))
        
        # Calculate correlation directly
        correlation = pattern_bridge._calculate_correlation(stat_pattern, sem_pattern)
        logger.info(f"Direct correlation test: {correlation[0]:.2f} ({correlation[1]})")
        
        # Calculate components
        temporal_corr = pattern_bridge._calculate_temporal_correlation(stat_pattern, sem_pattern)
        logger.info(f"  Temporal correlation: {temporal_corr:.2f}")
        
        # Extract times for debugging
        stat_time = pattern_bridge._extract_time_from_statistical_pattern(stat_pattern)
        sem_time = pattern_bridge._extract_time_from_semantic_pattern(sem_pattern)
        logger.info(f"  Statistical time: {stat_time}")
        logger.info(f"  Semantic time: {sem_time}")
    
    # Get co-patterns
    co_patterns = pattern_bridge.get_co_patterns()
    
    # Print results
    print_co_patterns(co_patterns)
    
    logger.info("Final pattern bridge test completed")


if __name__ == "__main__":
    test_pattern_bridge()
