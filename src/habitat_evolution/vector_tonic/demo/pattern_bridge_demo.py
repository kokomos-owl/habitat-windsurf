"""
Pattern Bridge Demo for Habitat Evolution.

This script demonstrates the bridge between statistical and semantic patterns
using climate time series data. It shows how statistical patterns detected
in climate data can be correlated with semantic patterns extracted from
climate risk documents, creating co-patterns that represent the same
underlying phenomena across different domains.
"""

import os
import json
import uuid
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from src.habitat_evolution.core.services.event_bus import EventBus
from src.habitat_evolution.core.services.time_provider import TimeProvider
from src.habitat_evolution.vector_tonic.data.climate_data_loader import ClimateTimeSeriesLoader
from src.habitat_evolution.vector_tonic.core.time_series_pattern_detector import TimeSeriesPatternDetector
from src.habitat_evolution.vector_tonic.bridge.pattern_domain_bridge import PatternDomainBridge
from src.habitat_evolution.vector_tonic.bridge.events import (
    StatisticalPatternDetectedEvent,
    StatisticalPatternQualityChangedEvent
)
from src.habitat_evolution.vector_tonic.visualization.pattern_correlation_visualizer import PatternCorrelationVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)


class PatternBridgeDemo:
    """
    Demo for the pattern bridge module.
    
    This class demonstrates how to use the pattern bridge to connect
    statistical patterns detected in climate time series data with
    semantic patterns extracted from climate risk documents.
    """
    
    def __init__(self, 
                data_dir: str = "/Users/prphillips/Documents/GitHub/habitat_alpha/docs/time_series",
                output_dir: str = "/Users/prphillips/Documents/GitHub/habitat_alpha/docs/demo_output"):
        """
        Initialize the pattern bridge demo.
        
        Args:
            data_dir: Directory containing climate time series data
            output_dir: Directory for saving demo outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.time_provider = TimeProvider()
        self.event_bus = EventBus()
        self.climate_loader = ClimateTimeSeriesLoader()
        self.pattern_detector = TimeSeriesPatternDetector()
        self.pattern_bridge = PatternDomainBridge(self.event_bus, self.time_provider)
        self.visualizer = PatternCorrelationVisualizer()
        
        # Storage for patterns and time series data
        self.ma_time_series = None
        self.ne_time_series = None
        self.statistical_patterns = []
        self.semantic_patterns = []
        self.co_patterns = []
        
        logger.info("Pattern Bridge Demo initialized")
    
    def load_climate_data(self):
        """Load climate time series data."""
        logger.info("Loading climate time series data...")
        
        # Load Massachusetts temperature data
        ma_file_path = os.path.join(self.data_dir, "MA_AvgTemp_91_24.json")
        self.ma_time_series = self.climate_loader.load_temperature_data(ma_file_path)
        
        # Load Northeast temperature data
        ne_file_path = os.path.join(self.data_dir, "NE_AvgTemp_91_24.json")
        self.ne_time_series = self.climate_loader.load_temperature_data(ne_file_path)
        
        logger.info(f"Loaded MA time series with {len(self.ma_time_series.data)} data points")
        logger.info(f"Loaded NE time series with {len(self.ne_time_series.data)} data points")
    
    def detect_statistical_patterns(self):
        """Detect statistical patterns in climate time series data."""
        logger.info("Detecting statistical patterns...")
        
        # Detect patterns in Massachusetts data
        ma_patterns = self.pattern_detector.detect_patterns(
            self.ma_time_series,
            window_size=10,
            step_size=5,
            min_confidence=0.6
        )
        
        # Detect patterns in Northeast data
        ne_patterns = self.pattern_detector.detect_patterns(
            self.ne_time_series,
            window_size=10,
            step_size=5,
            min_confidence=0.6
        )
        
        # Add region information to patterns
        for pattern in ma_patterns:
            pattern["region"] = "Massachusetts"
            pattern["id"] = f"stat_pattern_{uuid.uuid4().hex[:8]}"
            
            # Publish pattern detected event
            self._publish_statistical_pattern_event(pattern)
        
        for pattern in ne_patterns:
            pattern["region"] = "Northeast"
            pattern["id"] = f"stat_pattern_{uuid.uuid4().hex[:8]}"
            
            # Publish pattern detected event
            self._publish_statistical_pattern_event(pattern)
        
        # Store patterns
        self.statistical_patterns = ma_patterns + ne_patterns
        
        logger.info(f"Detected {len(ma_patterns)} patterns in MA data")
        logger.info(f"Detected {len(ne_patterns)} patterns in NE data")
    
    def create_mock_semantic_patterns(self):
        """
        Create mock semantic patterns for demonstration purposes.
        
        In a real implementation, these would come from Habitat's
        semantic pattern extraction system.
        """
        logger.info("Creating mock semantic patterns...")
        
        # Create mock semantic patterns related to climate change
        semantic_patterns = [
            {
                "id": f"sem_pattern_{uuid.uuid4().hex[:8]}",
                "text": "Rising temperatures in Massachusetts are increasing wildfire risks",
                "quality_state": "emergent",
                "confidence": 0.75,
                "source": "climate_risk_doc_123",
                "temporal_markers": [
                    {"time": "201501", "text": "since 2015"},
                    {"time": "202001", "text": "through 2020"}
                ],
                "metadata": {
                    "document_id": "climate_risk_doc_123",
                    "page": 15,
                    "extraction_date": "2025-01-15"
                }
            },
            {
                "id": f"sem_pattern_{uuid.uuid4().hex[:8]}",
                "text": "Northeast region experiencing accelerated warming in the past decade",
                "quality_state": "stable",
                "confidence": 0.85,
                "source": "climate_risk_doc_456",
                "temporal_markers": [
                    {"time": "201401", "text": "since 2014"},
                    {"time": "202401", "text": "through present day (2024)"}
                ],
                "metadata": {
                    "document_id": "climate_risk_doc_456",
                    "page": 8,
                    "extraction_date": "2025-02-20"
                }
            },
            {
                "id": f"sem_pattern_{uuid.uuid4().hex[:8]}",
                "text": "Temperature anomalies in Massachusetts correlate with increased coastal flooding events",
                "quality_state": "hypothetical",
                "confidence": 0.65,
                "source": "climate_risk_doc_789",
                "temporal_markers": [
                    {"time": "200001", "text": "beginning in 2000"},
                    {"time": "202001", "text": "through 2020"}
                ],
                "metadata": {
                    "document_id": "climate_risk_doc_789",
                    "page": 22,
                    "extraction_date": "2025-03-05"
                }
            },
            {
                "id": f"sem_pattern_{uuid.uuid4().hex[:8]}",
                "text": "Consistent warming trend observed across the Northeast since 1991",
                "quality_state": "stable",
                "confidence": 0.9,
                "source": "climate_risk_doc_101",
                "temporal_markers": [
                    {"time": "199101", "text": "since 1991"},
                    {"time": "202401", "text": "through 2024"}
                ],
                "metadata": {
                    "document_id": "climate_risk_doc_101",
                    "page": 5,
                    "extraction_date": "2025-01-10"
                }
            },
            {
                "id": f"sem_pattern_{uuid.uuid4().hex[:8]}",
                "text": "Significant temperature increases in Massachusetts summers since 2010",
                "quality_state": "emergent",
                "confidence": 0.8,
                "source": "climate_risk_doc_202",
                "temporal_markers": [
                    {"time": "201001", "text": "since 2010"},
                    {"time": "202301", "text": "through 2023"}
                ],
                "metadata": {
                    "document_id": "climate_risk_doc_202",
                    "page": 17,
                    "extraction_date": "2025-02-15"
                }
            }
        ]
        
        # Store patterns
        self.semantic_patterns = semantic_patterns
        
        # Register patterns with the bridge
        for pattern in semantic_patterns:
            self._register_semantic_pattern_with_bridge(pattern)
        
        logger.info(f"Created {len(semantic_patterns)} mock semantic patterns")
    
    def _publish_statistical_pattern_event(self, pattern: Dict[str, Any]):
        """
        Publish a statistical pattern detected event.
        
        Args:
            pattern: The detected statistical pattern
        """
        event = StatisticalPatternDetectedEvent(
            pattern_id=pattern["id"],
            pattern_data=pattern
        )
        
        self.event_bus.publish(event)
    
    def _register_semantic_pattern_with_bridge(self, pattern: Dict[str, Any]):
        """
        Register a semantic pattern with the bridge.
        
        Args:
            pattern: The semantic pattern
        """
        # Create a simple event-like object
        class MockEvent:
            pass
        
        event = MockEvent()
        event.pattern_id = pattern["id"]
        event.pattern_text = pattern["text"]
        event.quality_state = pattern["quality_state"]
        event.confidence = pattern["confidence"]
        event.source = pattern["source"]
        event.temporal_markers = pattern["temporal_markers"]
        event.metadata = pattern["metadata"]
        
        # Call the bridge's handler directly
        self.pattern_bridge.on_semantic_pattern_detected(event)
    
    def get_co_patterns(self):
        """Get co-patterns from the bridge."""
        logger.info("Retrieving co-patterns...")
        
        # Get all co-patterns
        self.co_patterns = self.pattern_bridge.get_co_patterns()
        
        logger.info(f"Retrieved {len(self.co_patterns)} co-patterns")
        
        # Print co-pattern information
        for i, co_pattern in enumerate(self.co_patterns):
            logger.info(f"Co-Pattern {i+1}:")
            logger.info(f"  ID: {co_pattern['id']}")
            logger.info(f"  Quality: {co_pattern['quality_state']}")
            logger.info(f"  Correlation: {co_pattern['correlation_strength']:.2f} ({co_pattern['correlation_type']})")
            
            # Get related patterns
            stat_pattern = None
            sem_pattern = None
            
            for pattern in self.statistical_patterns:
                if pattern["id"] == co_pattern["statistical_pattern_id"]:
                    stat_pattern = pattern
                    break
            
            for pattern in self.semantic_patterns:
                if pattern["id"] == co_pattern["semantic_pattern_id"]:
                    sem_pattern = pattern
                    break
            
            if stat_pattern:
                logger.info(f"  Statistical: {stat_pattern.get('type', 'Unknown')} ({stat_pattern.get('region', 'Unknown')})")
            
            if sem_pattern:
                logger.info(f"  Semantic: {sem_pattern.get('text', 'Unknown')}")
            
            logger.info("")
    
    def visualize_results(self):
        """Create visualizations of the results."""
        logger.info("Creating visualizations...")
        
        # 1. Time series with patterns for Massachusetts
        ma_fig = self.visualize_time_series_with_patterns(
            self.ma_time_series.to_dict(),
            [p for p in self.statistical_patterns if p.get("region") == "Massachusetts"],
            "Massachusetts Temperature Patterns (1991-2024)"
        )
        
        # 2. Time series with patterns for Northeast
        ne_fig = self.visualize_time_series_with_patterns(
            self.ne_time_series.to_dict(),
            [p for p in self.statistical_patterns if p.get("region") == "Northeast"],
            "Northeast Temperature Patterns (1991-2024)"
        )
        
        # 3. Pattern correlation heatmap
        heatmap_fig = self.visualize_correlation_heatmap()
        
        # 4. Pattern network
        network_fig = self.visualize_pattern_network()
        
        # 5. Sliding window view (recent period)
        sliding_fig = self.visualize_sliding_window("201501", "202401")
        
        # Save figures
        ma_fig.savefig(os.path.join(self.output_dir, "ma_temperature_patterns.png"), dpi=300)
        ne_fig.savefig(os.path.join(self.output_dir, "ne_temperature_patterns.png"), dpi=300)
        heatmap_fig.savefig(os.path.join(self.output_dir, "pattern_correlation_heatmap.png"), dpi=300)
        network_fig.savefig(os.path.join(self.output_dir, "pattern_network.png"), dpi=300)
        sliding_fig.savefig(os.path.join(self.output_dir, "sliding_window_view.png"), dpi=300)
        
        logger.info(f"Saved visualizations to {self.output_dir}")
    
    def visualize_time_series_with_patterns(self, 
                                          time_series_data: Dict[str, Any],
                                          patterns: List[Dict[str, Any]],
                                          title: str) -> plt.Figure:
        """
        Visualize time series data with detected patterns.
        
        Args:
            time_series_data: Time series data dictionary
            patterns: List of detected patterns
            title: Title for the visualization
            
        Returns:
            Matplotlib figure
        """
        return self.visualizer.plot_time_series_with_patterns(
            time_series_data,
            patterns,
            title=title,
            show_anomalies=True
        )
    
    def visualize_correlation_heatmap(self) -> plt.Figure:
        """
        Visualize pattern correlations as a heatmap.
        
        Returns:
            Matplotlib figure
        """
        return self.visualizer.plot_pattern_correlation_heatmap(
            self.statistical_patterns,
            self.semantic_patterns,
            self.co_patterns,
            title="Statistical-Semantic Pattern Correlation Heatmap"
        )
    
    def visualize_pattern_network(self) -> plt.Figure:
        """
        Visualize pattern correlations as a network.
        
        Returns:
            Matplotlib figure
        """
        return self.visualizer.plot_pattern_network(
            self.statistical_patterns,
            self.semantic_patterns,
            self.co_patterns,
            title="Pattern Correlation Network",
            min_correlation=0.4
        )
    
    def visualize_sliding_window(self, 
                               window_start: str,
                               window_end: str) -> plt.Figure:
        """
        Visualize patterns and correlations in a sliding window.
        
        Args:
            window_start: Start time for the window (YYYYMM format)
            window_end: End time for the window (YYYYMM format)
            
        Returns:
            Matplotlib figure
        """
        # Use Massachusetts data for the sliding window view
        return self.visualizer.plot_sliding_window_view(
            self.ma_time_series.to_dict(),
            self.statistical_patterns,
            self.semantic_patterns,
            self.co_patterns,
            window_start=window_start,
            window_end=window_end,
            title=f"Climate Pattern Evolution ({window_start[:4]}-{window_end[:4]})"
        )
    
    def save_results_to_json(self):
        """Save results to JSON files for further analysis."""
        logger.info("Saving results to JSON...")
        
        # Save statistical patterns
        with open(os.path.join(self.output_dir, "statistical_patterns.json"), "w") as f:
            json.dump(self.statistical_patterns, f, cls=CustomJSONEncoder, indent=2)
        
        # Save semantic patterns
        with open(os.path.join(self.output_dir, "semantic_patterns.json"), "w") as f:
            json.dump(self.semantic_patterns, f, cls=CustomJSONEncoder, indent=2)
        
        # Save co-patterns
        with open(os.path.join(self.output_dir, "co_patterns.json"), "w") as f:
            json.dump(self.co_patterns, f, cls=CustomJSONEncoder, indent=2)
        
        logger.info(f"Saved results to {self.output_dir}")
    
    def run_demo(self):
        """Run the complete demo."""
        logger.info("Starting Pattern Bridge Demo...")
        
        # Step 1: Load climate data
        self.load_climate_data()
        
        # Step 2: Detect statistical patterns
        self.detect_statistical_patterns()
        
        # Step 3: Create mock semantic patterns
        self.create_mock_semantic_patterns()
        
        # Step 4: Get co-patterns
        self.get_co_patterns()
        
        # Step 5: Visualize results
        self.visualize_results()
        
        # Step 6: Save results to JSON
        self.save_results_to_json()
        
        logger.info("Pattern Bridge Demo completed successfully")


if __name__ == "__main__":
    # Run the demo
    demo = PatternBridgeDemo()
    demo.run_demo()
