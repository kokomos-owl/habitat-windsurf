import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.habitat_evolution.vector_tonic.bridge.field_pattern_bridge import FieldPatternBridge

class TestFieldPatternBridge(unittest.TestCase):
    
    def setUp(self):
        # Create mock pattern evolution service
        self.mock_pattern_evolution_service = MagicMock()
        self.mock_pattern_evolution_service.create_pattern.return_value = {
            "id": "test_pattern_id",
            "type": "climate_warming_trend",
            "quality_state": "emergent"
        }
        
        # Create bridge with mock service
        self.bridge = FieldPatternBridge(self.mock_pattern_evolution_service)
        
    def test_process_time_series(self):
        # Create test data with warming trend
        dates = pd.date_range(start='2020-01-01', periods=12, freq='MS')
        temps = np.linspace(10, 15, 12)  # 5 degree warming over 12 months
        data = pd.DataFrame({'date': dates, 'temperature': temps})
        
        # Process with bridge
        results = self.bridge.process_time_series(data, {"region": "Massachusetts"})
        
        # Verify results
        self.assertIn("field_analysis", results)
        self.assertIn("multi_scale", results)
        self.assertIn("patterns", results)
        
        # Verify pattern evolution service was called
        self.mock_pattern_evolution_service.create_pattern.assert_called()
        
    def test_convert_to_evolution_pattern(self):
        # Test pattern conversion
        source_pattern = {
            "id": "test_pattern",
            "type": "warming_trend",
            "magnitude": 0.5,
            "position": [0.5, 0.3, 0.2],
            "confidence": 0.8
        }
        
        # Convert pattern
        result = self.bridge._convert_to_evolution_pattern(
            source_pattern, 
            "field_state",
            {"region": "Massachusetts"}
        )
        
        # Verify conversion
        self.assertEqual(result["id"], "test_pattern")
        self.assertEqual(result["type"], "climate_warming_trend")
        self.assertEqual(result["magnitude"], 0.5)
        self.assertEqual(result["field_position"], [0.5, 0.3, 0.2])
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["quality_state"], "emergent")
