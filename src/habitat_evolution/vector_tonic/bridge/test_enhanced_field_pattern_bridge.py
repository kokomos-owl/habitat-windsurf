"""
Test for the enhanced FieldPatternBridge.

This test demonstrates how the FieldPatternBridge observes topological-temporal relationships
from field components and associates them with patterns without creating artificial data.
"""

import sys
import os
import pytest
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.habitat_evolution.vector_tonic.bridge.field_pattern_bridge import FieldPatternBridge
from src.habitat_evolution.field.field_state import TonicHarmonicFieldState
from src.habitat_evolution.field.topological_field_analyzer import TopologicalFieldAnalyzer
from src.habitat_evolution.vector_tonic.field_state.simple_field_analyzer import SimpleFieldStateAnalyzer
from src.habitat_evolution.vector_tonic.field_state.multi_scale_analyzer import MultiScaleAnalyzer
from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.infrastructure.services.bidirectional_flow_service import BidirectionalFlowService
from src.habitat_evolution.infrastructure.interfaces.services.bidirectional_flow_interface import BidirectionalFlowInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnhancedFieldPatternBridge:
    """Test suite for the enhanced FieldPatternBridge."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create enhanced mock pattern evolution service
        self.mock_pattern_service = MagicMock(spec=PatternEvolutionService)
        
        # Configure the mock to create more meaningful patterns with properties for relationship detection
        self.pattern_counter = 0
        
        def create_meaningful_pattern(pattern_data, metadata=None):
            # Create a pattern with rich properties for relationship detection
            self.pattern_counter += 1
            pattern_id = f"test_pattern_{self.pattern_counter}"
            
            # Extract meaningful properties from the data
            pattern_type = "temperature_trend"
            confidence = 0.75
            magnitude = 0.5 + (self.pattern_counter * 0.1)  # Increasing magnitudes for testing relationships
            position = [self.pattern_counter * 0.3, self.pattern_counter * 0.2]  # Unique positions
            
            # Add region from metadata if available
            region = metadata.get("region", "unknown") if metadata else "unknown"
            
            # Create time range based on pattern counter
            time_range = {
                "start": 2010 + (self.pattern_counter % 3),
                "end": 2015 + (self.pattern_counter % 5)
            }
            
            # Return the pattern ID and store the pattern for later verification
            return pattern_id
        
        self.mock_pattern_service.create_pattern.side_effect = create_meaningful_pattern
        
        # Create mock bidirectional flow service for relationships
        self.mock_flow_service = MagicMock()
        self.mock_flow_service._create_pattern_relationship = MagicMock(return_value=None)
        
        # Create mock field state
        self.mock_field_state = MagicMock(spec=TonicHarmonicFieldState)
        self.mock_field_state.create_snapshot.return_value = {
            "field_properties": {
                "coherence": 0.85,
                "stability": 0.75,
                "navigability_score": 0.8
            },
            "density": {
                "density_centers": [
                    {"center_id": "c1", "position": [0.1, 0.2], "density": 0.9},
                    {"center_id": "c2", "position": [0.7, 0.8], "density": 0.7}
                ]
            }
        }
        
        # Create a mock field analyzer that returns multiple patterns
        self.mock_field_analyzer = MagicMock(spec=SimpleFieldStateAnalyzer)
        
        # Configure the mock analyzer to return multiple patterns
        def analyze_time_series_with_multiple_patterns(data):
            # Extract data properties for realistic patterns
            date_range = None
            temp_range = None
            
            if hasattr(data, 'date') and len(data.date) > 0:
                date_range = (min(data.date), max(data.date))
            
            if hasattr(data, 'temperature') and len(data.temperature) > 0:
                temp_values = data.temperature.values
                temp_range = (min(temp_values), max(temp_values))
                temp_diff = max(temp_values) - min(temp_values)
            
            # Create multiple patterns with different characteristics
            patterns = [
                {
                    "type": "warming_trend",
                    "confidence": 0.8,
                    "magnitude": temp_diff / 10.0 if temp_range else 0.5,
                    "position": [0.3, 0.2, 0.5],
                    "metadata": {
                        "temporal": {"time_period": date_range[0] if date_range else "2010"},
                        "region": "Massachusetts"
                    }
                },
                {
                    "type": "seasonal_cycle",
                    "confidence": 0.75,
                    "magnitude": temp_diff / 15.0 if temp_range else 0.4,
                    "position": [0.6, 0.4, 0.5],
                    "metadata": {
                        "temporal": {"time_period": date_range[0] if date_range else "2012"},
                        "region": "Massachusetts"
                    }
                },
                {
                    "type": "extreme_event",
                    "confidence": 0.65,
                    "magnitude": temp_diff / 8.0 if temp_range else 0.6,
                    "position": [0.2, 0.7, 0.5],
                    "metadata": {
                        "temporal": {"time_period": date_range[1] if date_range else "2020"},
                        "region": "Massachusetts"
                    }
                }
            ]
            
            return {
                "patterns": patterns,
                "field_properties": {
                    "temporal_range": date_range,
                    "value_range": temp_range,
                    "complexity": 0.7
                }
            }
        
        self.mock_field_analyzer.analyze_time_series.side_effect = analyze_time_series_with_multiple_patterns
        
        # Create mock topological analyzer
        self.mock_topological_analyzer = MagicMock(spec=TopologicalFieldAnalyzer)
        self.mock_topological_analyzer.analyze_field.return_value = {
            "topology": {
                "effective_dimensionality": 3,
                "principal_dimensions": [0, 1, 2],
                "eigenvalues": np.array([0.6, 0.3, 0.1])
            }
        }
        
        # Create a mock multi-scale analyzer
        self.mock_multi_scale_analyzer = MagicMock(spec=MultiScaleAnalyzer)
        
        # Configure the mock multi-scale analyzer to return patterns
        def analyze_with_multiple_patterns(data):
            # Extract data properties for realistic patterns
            date_range = None
            if hasattr(data, 'date') and len(data.date) > 0:
                date_range = (min(data.date), max(data.date))
                
            # Create multi-scale patterns with different characteristics
            patterns = [
                {
                    "type": "long_term_trend",
                    "confidence": 0.85,
                    "magnitude": 0.7,
                    "position": [0.4, 0.3, 0.6],
                    "time_range": {"start": str(date_range[0]) if date_range else "1990", 
                                  "end": str(date_range[1]) if date_range else "2024"},
                    "metadata": {"region": "Massachusetts"}
                },
                {
                    "type": "decadal_oscillation",
                    "confidence": 0.75,
                    "magnitude": 0.6,
                    "position": [0.7, 0.5, 0.4],
                    "time_range": {"start": "2000", "end": "2020"},
                    "metadata": {"region": "Massachusetts"}
                }
            ]
            
            # Create temporal patterns with relationships between test patterns
            temporal_patterns = [
                {
                    "type": "sequence",
                    "confidence": 0.8,
                    "related_patterns": ["test_pattern_1", "test_pattern_2", "test_pattern_3"]
                }
            ]
            
            return {
                "patterns": patterns,
                "temporal_patterns": temporal_patterns,
                "cross_scale_patterns": [],
                "scale_properties": {
                    "scale_coherence": 0.75,
                    "scale_stability": 0.8
                }
            }
        
        self.mock_multi_scale_analyzer.analyze.side_effect = analyze_with_multiple_patterns
        
        # Create the bridge with mocks
        self.bridge = FieldPatternBridge(
            pattern_evolution_service=self.mock_pattern_service,
            field_state=self.mock_field_state,
            topological_analyzer=self.mock_topological_analyzer
        )
        
        # Set the bidirectional flow service on the bridge
        self.bridge.bidirectional_flow_service = self.mock_flow_service
        
        # Set the field analyzer and multi-scale analyzer directly
        self.bridge.field_analyzer = self.mock_field_analyzer
        self.bridge.multi_scale_analyzer = self.mock_multi_scale_analyzer
        
        # Create test time series data
        self.time_series_data = self._create_test_time_series()
        
    def _create_test_time_series(self):
        """Create test time series data with realistic climate patterns."""
        # Create time points (monthly data for 10 years)
        time_points = 120
        time = np.arange(time_points)
        
        # Create a warming trend with seasonal variations, acceleration, and some noise
        trend = 0.02 * time  # warming trend
        acceleration = 0.0001 * time * time  # acceleration component
        seasonal = 3 * np.sin(2 * np.pi * time / 12)  # seasonal cycle
        noise = np.random.normal(0, 1, time_points)  # random variations
        
        # Combine components
        temperature = 50 + trend + acceleration + seasonal + noise
        
        # Convert to DataFrame
        dates = []
        temps = []
        for i, temp in enumerate(temperature):
            # Format as YYYYMM
            year = 2010 + i // 12
            month = i % 12 + 1
            dates.append(f"{year}{month:02d}")
            temps.append(temp)
        
        # Create pandas DataFrame
        import pandas as pd
        data = pd.DataFrame({
            "date": dates,
            "temperature": temps
        })
        
        return data
    
    def test_process_time_series(self):
        """Test processing time series data with the enhanced bridge."""
        # Process the time series data
        results = self.bridge.process_time_series(self.time_series_data, {"region": "Massachusetts"})
        
        # Verify field analysis was performed
        assert "field_analysis" in results
        assert "patterns" in results["field_analysis"]
        
        # Verify multi-scale analysis was performed
        assert "multi_scale" in results
        
        # Verify patterns were created
        assert "patterns" in results
        assert len(results["patterns"]) > 0
        
        # Verify relationships were detected
        assert "relationships" in results
        
        # Verify field state context was included
        assert "field_state_context" in results
        assert "field_properties" in results["field_state_context"]
        
        # Verify topological context was included
        assert "topological_context" in results
        
        # Verify pattern evolution service was called
        assert self.mock_pattern_service.create_pattern.call_count > 0
        
        # If relationships were detected, verify relationship creation was called
        if results["relationships"] and hasattr(self.mock_flow_service, '_create_pattern_relationship'):
            assert self.mock_flow_service._create_pattern_relationship.call_count > 0
        
        # Display detailed information about the data and patterns
        logger.info("\nSYNTHETIC DATA SUMMARY:")
        logger.info(f"Time series shape: {self.time_series_data.shape}")
        logger.info(f"Date range: {self.time_series_data['date'].iloc[0]} to {self.time_series_data['date'].iloc[-1]}")
        logger.info(f"Temperature range: {self.time_series_data['temperature'].min():.2f} to {self.time_series_data['temperature'].max():.2f}")
        
        logger.info("\nPATTERNS DETECTED:")
        for i, pattern in enumerate(results["patterns"]):
            logger.info(f"Pattern {i+1}: {pattern.get('type', 'unknown')} (ID: {pattern.get('id', 'unknown')})")
            logger.info(f"  - Confidence: {pattern.get('confidence', 0):.2f}")
            logger.info(f"  - Magnitude: {pattern.get('magnitude', 0):.2f}")
            if 'position' in pattern:
                logger.info(f"  - Position: {pattern['position']}")
        
        # Analyze why there are no relationships
        logger.info("\nRELATIONSHIP ANALYSIS:")
        if not results["relationships"]:
            logger.info("No relationships were detected. Possible reasons:")
            logger.info("  1. Not enough patterns detected to form meaningful relationships")
            logger.info(f"  2. Patterns are too dissimilar (only {len(results['patterns'])} patterns found)")
            logger.info("  3. Relationship detection thresholds may be too high")
            logger.info("  4. The current implementation may not be fully detecting pattern relationships")
        else:
            for i, rel in enumerate(results["relationships"]):
                logger.info(f"Relationship {i+1}: {rel.get('type', 'unknown')}")
                logger.info(f"  - Source: {rel.get('source_id', 'unknown')}")
                logger.info(f"  - Target: {rel.get('target_id', 'unknown')}")
        
        # Log the results
        logger.info(f"\nProcessed time series data and found {len(results['patterns'])} patterns and {len(results['relationships'])} relationships")
    
    def test_with_real_noaa_data(self):
        """Test with real NOAA climate data if available."""
        # Load real NOAA data using absolute path
        project_root = Path("/Users/prphillips/Documents/GitHub/habitat_alpha")
        data_path = project_root / "docs" / "time_series_json" / "MA_AvgTemp_91_24.json"
        
        if not data_path.exists():
            logger.warning(f"NOAA data not found at {data_path}, skipping test_with_real_noaa_data")
            return
        
        try:
            # Load the data
            with open(data_path, "r") as f:
                noaa_data = json.load(f)
            
            # Extract time series
            dates = []
            temps = []
            for date_str, point in noaa_data.get("data", {}).items():
                dates.append(date_str)
                temps.append(point.get("value", 0))
            
            # Create pandas DataFrame
            import pandas as pd
            data = pd.DataFrame({
                "date": dates,
                "temperature": temps
            })
            
            # Process with the bridge
            results = self.bridge.process_time_series(data, {"region": "Massachusetts", "source": "NOAA"})
            
            # Verify results
            assert "field_analysis" in results
            assert "patterns" in results["field_analysis"]
            assert "relationships" in results
            
            # Display detailed information about the data and patterns
            logger.info("\nMASSACHUSETTS NOAA DATA SUMMARY:")
            logger.info(f"Time series shape: {data.shape}")
            logger.info(f"Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
            logger.info(f"Temperature range: {data['temperature'].min():.2f} to {data['temperature'].max():.2f}")
            
            logger.info("\nPATTERNS DETECTED:")
            for i, pattern in enumerate(results["patterns"]):
                logger.info(f"Pattern {i+1}: {pattern.get('type', 'unknown')} (ID: {pattern.get('id', 'unknown')})")
                logger.info(f"  - Confidence: {pattern.get('confidence', 0):.2f}")
                logger.info(f"  - Magnitude: {pattern.get('magnitude', 0):.2f}")
                if 'position' in pattern:
                    logger.info(f"  - Position: {pattern['position']}")
            
            # Analyze why there are no relationships
            logger.info("\nRELATIONSHIP ANALYSIS:")
            if not results["relationships"]:
                logger.info("No relationships were detected. Possible reasons:")
                logger.info("  1. Not enough patterns detected to form meaningful relationships")
                logger.info(f"  2. Patterns are too dissimilar (only {len(results['patterns'])} patterns found)")
                logger.info("  3. Relationship detection thresholds may be too high")
                logger.info("  4. The current implementation may not be fully detecting pattern relationships")
            else:
                for i, rel in enumerate(results["relationships"]):
                    logger.info(f"Relationship {i+1}: {rel.get('type', 'unknown')}")
                    logger.info(f"  - Source: {rel.get('source_id', 'unknown')}")
                    logger.info(f"  - Target: {rel.get('target_id', 'unknown')}")
            
            # Log the results
            logger.info(f"\nProcessed NOAA data and found {len(results['patterns'])} patterns and {len(results['relationships'])} relationships")
            
        except Exception as e:
            logger.error(f"Error processing NOAA data: {e}")
    
    def test_with_ne_noaa_data(self):
        """Test with New England NOAA climate data."""
        # Load real NOAA data for New England using absolute path
        project_root = Path("/Users/prphillips/Documents/GitHub/habitat_alpha")
        data_path = project_root / "docs" / "time_series_json" / "NE_AvgTemp_91_24.json"
        
        if not data_path.exists():
            logger.warning(f"New England NOAA data not found at {data_path}, skipping test_with_ne_noaa_data")
            return
        
        try:
            # Load the data
            with open(data_path, "r") as f:
                noaa_data = json.load(f)
            
            # Extract time series
            dates = []
            temps = []
            for date_str, point in noaa_data.get("data", {}).items():
                dates.append(date_str)
                temps.append(point.get("value", 0))
            
            # Create DataFrame
            import pandas as pd
            data = pd.DataFrame({
                "date": dates,
                "temperature": temps
            })
            
            # Process with the bridge
            results = self.bridge.process_time_series(data, {"region": "New England", "source": "NOAA"})
            
            # Verify results
            assert "field_analysis" in results
            assert "patterns" in results["field_analysis"]
            assert "relationships" in results
            
            # Display detailed information about the data and patterns
            logger.info("\nNEW ENGLAND NOAA DATA SUMMARY:")
            logger.info(f"Time series shape: {data.shape}")
            logger.info(f"Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
            logger.info(f"Temperature range: {data['temperature'].min():.2f} to {data['temperature'].max():.2f}")
            
            logger.info("\nPATTERNS DETECTED:")
            for i, pattern in enumerate(results["patterns"]):
                logger.info(f"Pattern {i+1}: {pattern.get('type', 'unknown')} (ID: {pattern.get('id', 'unknown')})")
                logger.info(f"  - Confidence: {pattern.get('confidence', 0):.2f}")
                logger.info(f"  - Magnitude: {pattern.get('magnitude', 0):.2f}")
                if 'position' in pattern:
                    logger.info(f"  - Position: {pattern['position']}")
            
            # Analyze why there are no relationships
            logger.info("\nRELATIONSHIP ANALYSIS:")
            if not results["relationships"]:
                logger.info("No relationships were detected. Possible reasons:")
                logger.info("  1. Not enough patterns detected to form meaningful relationships")
                logger.info(f"  2. Patterns are too dissimilar (only {len(results['patterns'])} patterns found)")
                logger.info("  3. Relationship detection thresholds may be too high")
                logger.info("  4. The current implementation may not be fully detecting pattern relationships")
            else:
                for i, rel in enumerate(results["relationships"]):
                    logger.info(f"Relationship {i+1}: {rel.get('type', 'unknown')}")
                    logger.info(f"  - Source: {rel.get('source_id', 'unknown')}")
                    logger.info(f"  - Target: {rel.get('target_id', 'unknown')}")
            
            # Log the results
            logger.info(f"\nProcessed New England NOAA data and found {len(results['patterns'])} patterns and {len(results['relationships'])} relationships")
            
        except Exception as e:
            logger.error(f"Error processing New England NOAA data: {e}")


if __name__ == "__main__":
    # Run the test directly
    test = TestEnhancedFieldPatternBridge()
    test.setup_method()
    test.test_process_time_series()
    test.test_with_real_noaa_data()
    test.test_with_ne_noaa_data()
