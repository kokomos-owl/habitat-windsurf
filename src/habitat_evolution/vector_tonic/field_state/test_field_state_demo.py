# tests/vector_tonic/demo/test_field_state_demo.py
import unittest
from unittest.mock import patch, MagicMock
from src.habitat_evolution.vector_tonic.demo.field_state_demo import run_field_state_demo

class TestFieldStateDemo(unittest.TestCase):
    
    @patch('src.habitat_evolution.vector_tonic.demo.field_state_demo.SimpleFieldStateAnalyzer')
    @patch('src.habitat_evolution.vector_tonic.demo.field_state_demo.MultiScaleAnalyzer')
    @patch('src.habitat_evolution.vector_tonic.demo.field_state_demo.FieldPatternBridge')
    def test_run_field_state_demo(self, mock_bridge_class, mock_multi_scale_class, mock_field_analyzer_class):
        # Set up mocks
        mock_field_analyzer = MagicMock()
        mock_field_analyzer.analyze_time_series.return_value = {
            "patterns": [{"type": "warming_trend", "magnitude": 0.5}],
            "field_properties": {"trend": 0.5}
        }
        mock_field_analyzer_class.return_value = mock_field_analyzer
        
        mock_multi_scale = MagicMock()
        mock_multi_scale.analyze.return_value = {
            "scale_results": {},
            "cross_scale_patterns": [{"type": "consistent_warming", "magnitude": 0.4}]
        }
        mock_multi_scale_class.return_value = mock_multi_scale
        
        mock_bridge = MagicMock()
        mock_bridge.process_time_series.return_value = {
            "field_analysis": {},
            "multi_scale": {},
            "patterns": [{"id": "test_pattern"}]
        }
        mock_bridge_class.return_value = mock_bridge
        
        # Run demo with mock services
        results = run_field_state_demo(use_mock_services=True)
        
        # Verify results
        self.assertIn("field_results", results)
        self.assertIn("scale_results", results)
        self.assertIn("bridge_results", results)
        
        # Verify mocks were called
        mock_field_analyzer.analyze_time_series.assert_called()
        mock_multi_scale.analyze.assert_called()
        mock_bridge.process_time_series.assert_called()