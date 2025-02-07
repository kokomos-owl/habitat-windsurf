"""
Tests for the HabitatFlow core management system.
"""

import pytest
from datetime import datetime
from src.core.flow.habitat_flow import (
    HabitatFlow,
    FlowState,
    FlowType
)

@pytest.fixture
def flow_manager():
    """Create a flow manager instance for testing."""
    return HabitatFlow()

@pytest.fixture
def sample_content():
    """Sample content for testing flow processing."""
    return {
        'text': 'Temperature increase of 2.5°C impacts coastal regions',
        'metadata': {
            'source': 'climate_data',
            'timestamp': datetime.now().isoformat()
        }
    }

class TestFlowState:
    """Test suite for FlowState functionality."""
    
    def test_flow_state_validation(self):
        """Test flow state validation logic."""
        # Valid state
        valid_state = FlowState(
            strength=0.8,
            coherence=0.7,
            emergence_potential=0.6
        )
        assert valid_state.is_valid() is True
        
        # Invalid state
        invalid_state = FlowState(
            strength=0.2,
            coherence=0.2,
            emergence_potential=0.2
        )
        assert invalid_state.is_valid() is False

    def test_flow_state_serialization(self):
        """Test flow state serialization."""
        state = FlowState(
            strength=0.8,
            coherence=0.7,
            emergence_potential=0.6
        )
        state_dict = state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert all(k in state_dict for k in [
            'strength', 'coherence', 'emergence_potential',
            'temporal_context', 'last_updated'
        ])

class TestHabitatFlow:
    """Test suite for HabitatFlow functionality."""
    
    @pytest.mark.asyncio
    async def test_process_flow(self, flow_manager, sample_content):
        """Test processing content through flow system."""
        result = await flow_manager.process_flow(sample_content)
        
        assert isinstance(result, dict)
        assert all(k in result for k in [
            'flow_id', 'state', 'pattern_state', 'is_valid'
        ])
        
        # Verify flow tracking
        flow_id = result['flow_id']
        assert flow_id in flow_manager.active_flows
        
        # Verify history recording
        assert len(flow_manager.flow_history) > 0
        assert flow_manager.flow_history[-1]['flow_id'] == flow_id

    def test_emergence_calculation(self, flow_manager):
        """Test emergence potential calculation."""
        pattern_result = {
            'flow_velocity': 0.8,
            'flow_direction': 3.14159,  # π radians
            'confidence': 0.7
        }
        
        emergence = flow_manager._calculate_emergence(pattern_result)
        assert 0.0 <= emergence <= 1.0
        
        # Test with missing values
        incomplete_result = {'flow_velocity': 0.8}
        emergence = flow_manager._calculate_emergence(incomplete_result)
        assert 0.0 <= emergence <= 1.0

    def test_coherence_calculation(self, flow_manager, mocker):
        """Test coherence calculation."""
        # Mock PatternState
        pattern_state = mocker.Mock()
        pattern_state.pattern = "test_pattern"
        pattern_state.confidence = 0.8
        
        # Mock get_pattern_relationships
        mock_relationships = [("related1", 0.7), ("related2", 0.6)]
        mocker.patch.object(
            flow_manager.pattern_evolution,
            'get_pattern_relationships',
            return_value=mock_relationships
        )
        
        coherence = flow_manager._calculate_coherence(pattern_state)
        assert 0.0 <= coherence <= 1.0
        
        # Test with no relationships
        mocker.patch.object(
            flow_manager.pattern_evolution,
            'get_pattern_relationships',
            return_value=[]
        )
        coherence = flow_manager._calculate_coherence(pattern_state)
        assert coherence == 0.0

    def test_flow_history_maintenance(self, flow_manager):
        """Test flow history size maintenance."""
        # Create more than 1000 history entries
        for _ in range(1100):
            flow_manager._record_flow_history(
                'test_flow',
                FlowState(strength=0.8, coherence=0.7)
            )
            
        assert len(flow_manager.flow_history) == 1000
