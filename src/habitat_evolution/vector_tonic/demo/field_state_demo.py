import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from ...infrastructure.services.pattern_evolution_service import PatternEvolutionService
from ...infrastructure.services.event_service import EventService
from ...infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
from ...infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from ..data.climate_data_loader import ClimateDataLoader
from ..field_state.simple_field_analyzer import SimpleFieldStateAnalyzer
from ..field_state.multi_scale_analyzer import MultiScaleAnalyzer
from ..bridge.field_pattern_bridge import FieldPatternBridge

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_field_state_demo(use_mock_services: bool = False) -> Dict[str, Any]:
    """
    Run a demonstration of the field state analysis.
    
    Args:
        use_mock_services: Whether to use mock services for testing
        
    Returns:
        Demo results
    """
    logger.info("Starting Field State Analysis Demo...")
    
    # Initialize services
    if use_mock_services:
        # Use mock services for testing
        from unittest.mock import MagicMock
        pattern_evolution_service = MagicMock()
        pattern_evolution_service.create_pattern.return_value = {
            "id": "mock_pattern_id",
            "type": "climate_warming_trend",
            "quality_state": "emergent"
        }
    else:
        # Use real services
        arangodb_connection = ArangoDBConnection(
            host="localhost",
            port=8529,
            username="root",
            password="habitat",
            database_name="habitat_evolution"
        )
        
        event_service = EventService()
        bidirectional_flow_service = BidirectionalFlowService(event_service)
        
        pattern_evolution_service = PatternEvolutionService(
            event_service=event_service,
            bidirectional_flow_service=bidirectional_flow_service,
            arangodb_connection=arangodb_connection
        )
        
        # Initialize pattern evolution service
        pattern_evolution_service.initialize()
    
    # Initialize components
    climate_loader = ClimateDataLoader()
    field_analyzer = SimpleFieldStateAnalyzer()
    multi_scale_analyzer = MultiScaleAnalyzer()
    field_bridge = FieldPatternBridge(pattern_evolution_service)
    
    # Load climate data
    logger.info("Loading climate data...")
    try:
        data = climate_loader.load_data("Massachusetts")
    except Exception as e:
        logger.error(f"Error loading climate data: {e}")
        # Generate synthetic data for demo
        dates = pd.date_range(start='2020-01-01', periods=120, freq='D')
        base = np.linspace(10, 15, 120)  # 5 degree warming over 120 days
        seasonal = 3 * np.sin(2 * np.pi * np.arange(120) / 30)
        temps = base + seasonal
        data = pd.DataFrame({
            'date': dates, 
            'temperature': temps,
            'region': 'Massachusetts'
        })
    
    # Analyze with field state
    logger.info("Analyzing with field state...")
    field_results = field_analyzer.analyze_time_series(data)
    
    # Analyze at multiple scales
    logger.info("Performing multi-scale analysis...")
    scale_results = multi_scale_analyzer.analyze(data)
    
    # Bridge to pattern evolution
    logger.info("Bridging to pattern evolution...")
    bridge_results = field_bridge.process_time_series(
        data,
        metadata={"region": "Massachusetts", "source": "field_state_demo"}
    )
    
    # Print results
    logger.info("\nField State Analysis Results:")
    logger.info(f"Found {len(field_results['patterns'])} field patterns")
    for pattern in field_results["patterns"]:
        logger.info(f"  - {pattern['type']} (magnitude: {pattern['magnitude']:.2f})")
    
    logger.info("\nMulti-Scale Analysis Results:")
    logger.info(f"Found {len(scale_results['cross_scale_patterns'])} cross-scale patterns")
    for pattern in scale_results["cross_scale_patterns"]:
        logger.info(f"  - {pattern['type']} (magnitude: {pattern['magnitude']:.2f})")
    
    logger.info("\nPattern Evolution Bridge Results:")
    logger.info(f"Created {len(bridge_results['patterns'])} patterns in evolution service")
    
    logger.info("\nField State Demo completed")
    
    return {
        "field_results": field_results,
        "scale_results": scale_results,
        "bridge_results": bridge_results
    }

if __name__ == "__main__":
    from unittest.mock import MagicMock
    run_field_state_demo()
