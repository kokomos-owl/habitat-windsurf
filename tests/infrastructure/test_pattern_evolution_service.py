"""
Tests for the PatternEvolutionService with AdaptiveID integration.

This module tests the functionality of the PatternEvolutionService,
focusing on the integration with AdaptiveID for versioning, relationship tracking,
and context management.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from typing import Dict, List, Any

from src.habitat_evolution.infrastructure.services.pattern_evolution_service import PatternEvolutionService
from src.habitat_evolution.adaptive_core.models.pattern import Pattern
from src.habitat_evolution.infrastructure.adapters.pattern_adaptive_id_adapter import PatternAdaptiveIDAdapter


@pytest.fixture
def mock_event_service():
    """Create a mock event service for testing."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_bidirectional_flow_service():
    """Create a mock bidirectional flow service for testing."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_arangodb_connection():
    """Create a mock ArangoDB connection for testing."""
    mock = MagicMock()
    # Setup common mock behaviors
    mock.collection_exists.return_value = False
    mock.graph_exists.return_value = False
    return mock


@pytest.fixture
def pattern_evolution_service(mock_event_service, mock_bidirectional_flow_service, mock_arangodb_connection):
    """Create a PatternEvolutionService instance for testing."""
    service = PatternEvolutionService(
        event_service=mock_event_service,
        bidirectional_flow_service=mock_bidirectional_flow_service,
        arangodb_connection=mock_arangodb_connection
    )
    return service


@pytest.fixture
def sample_pattern():
    """Create a sample pattern for testing."""
    pattern_id = str(uuid.uuid4())
    return {
        "id": pattern_id,
        "_key": pattern_id,
        "base_concept": "test_concept",
        "creator_id": "test_user",
        "weight": 1.0,
        "confidence": 0.7,
        "uncertainty": 0.3,
        "coherence": 0.8,
        "phase_stability": 0.6,
        "signal_strength": 0.75,
        "quality_state": "hypothetical",
        "properties": {"key1": "value1", "key2": "value2"},
        "metrics": {"usage_count": 5, "feedback_count": 2},
        "quality": {
            "score": 0.7,
            "usage_count": 5,
            "feedback_count": 2,
            "last_used": datetime.now().isoformat(),
            "last_feedback": datetime.now().isoformat()
        },
        "timestamp": datetime.now().isoformat()
    }


class TestPatternEvolutionService:
    """Tests for the PatternEvolutionService."""

    def test_initialize_creates_collections(self, pattern_evolution_service, mock_arangodb_connection):
        """Test that initialize creates the necessary collections."""
        # Arrange
        
        # Act
        pattern_evolution_service.initialize()
        
        # Assert
        assert pattern_evolution_service.running is True
        assert mock_arangodb_connection.create_collection.call_count == 5
        mock_arangodb_connection.create_collection.assert_any_call("patterns")
        mock_arangodb_connection.create_collection.assert_any_call("pattern_quality_transitions")
        mock_arangodb_connection.create_collection.assert_any_call("pattern_usage")
        mock_arangodb_connection.create_collection.assert_any_call("pattern_feedback")
        mock_arangodb_connection.create_collection.assert_any_call("pattern_relationships", is_edge=True)
        mock_arangodb_connection.create_graph.assert_called_once()

    def test_initialize_sets_up_event_subscriptions(self, pattern_evolution_service, mock_event_service):
        """Test that initialize sets up the necessary event subscriptions."""
        # Arrange
        
        # Act
        pattern_evolution_service.initialize()
        
        # Assert
        assert mock_event_service.subscribe.call_count == 5
        mock_event_service.subscribe.assert_any_call("pattern.created", pattern_evolution_service._handle_pattern_event)
        mock_event_service.subscribe.assert_any_call("pattern.updated", pattern_evolution_service._handle_pattern_event)
        mock_event_service.subscribe.assert_any_call("pattern.deleted", pattern_evolution_service._handle_pattern_event)
        mock_event_service.subscribe.assert_any_call("pattern.usage", pattern_evolution_service._handle_pattern_usage_event)
        mock_event_service.subscribe.assert_any_call("pattern.feedback", pattern_evolution_service._handle_pattern_feedback_event)

    def test_handle_pattern_event_creates_pattern(self, pattern_evolution_service, sample_pattern):
        """Test that _handle_pattern_event creates a pattern when the event type is 'created'."""
        # Arrange
        event_data = {
            "type": "created",
            "pattern": sample_pattern
        }
        
        # Mock the _create_pattern method
        pattern_evolution_service._create_pattern = MagicMock()
        
        # Act
        pattern_evolution_service._handle_pattern_event(event_data)
        
        # Assert
        pattern_evolution_service._create_pattern.assert_called_once_with(sample_pattern)

    def test_handle_pattern_event_updates_pattern(self, pattern_evolution_service, sample_pattern):
        """Test that _handle_pattern_event updates a pattern when the event type is 'updated'."""
        # Arrange
        event_data = {
            "type": "updated",
            "pattern": sample_pattern
        }
        
        # Mock the _update_pattern method
        pattern_evolution_service._update_pattern = MagicMock()
        
        # Act
        pattern_evolution_service._handle_pattern_event(event_data)
        
        # Assert
        pattern_evolution_service._update_pattern.assert_called_once_with(sample_pattern["id"], sample_pattern)

    def test_track_pattern_usage_with_adaptive_id(self, pattern_evolution_service, sample_pattern, mock_arangodb_connection):
        """Test that track_pattern_usage correctly uses AdaptiveID for versioning and context tracking."""
        # Arrange
        pattern_id = sample_pattern["id"]
        context = {"query": "test query", "user_id": "test_user"}
        
        # Mock _get_pattern to return the sample pattern
        pattern_evolution_service._get_pattern = MagicMock(return_value=sample_pattern)
        
        # Mock the PatternAdaptiveIDAdapter
        mock_adapter = MagicMock()
        mock_adapter.get_pattern.return_value = Pattern(
            id=sample_pattern["id"],
            base_concept=sample_pattern["base_concept"],
            creator_id=sample_pattern["creator_id"],
            weight=sample_pattern["weight"],
            confidence=sample_pattern["confidence"],
            uncertainty=sample_pattern["uncertainty"],
            coherence=sample_pattern["coherence"],
            phase_stability=sample_pattern["phase_stability"],
            signal_strength=sample_pattern["signal_strength"]
        )
        mock_adapter.get_pattern.return_value.metrics = {"usage_count": 6}
        mock_adapter.get_pattern.return_value.to_dict = MagicMock(return_value={
            "id": sample_pattern["id"],
            "base_concept": sample_pattern["base_concept"],
            "confidence": sample_pattern["confidence"],
            "metrics": {"usage_count": 6},
            "quality": {
                "score": 0.7,
                "usage_count": 6,
                "feedback_count": 2,
                "last_used": datetime.now().isoformat(),
                "last_feedback": sample_pattern["quality"]["last_feedback"]
            }
        })
        
        # Patch the PatternAdaptiveIDAdapter constructor
        with patch('src.habitat_evolution.infrastructure.services.pattern_evolution_service.PatternAdaptiveIDAdapter', return_value=mock_adapter):
            # Mock other methods
            pattern_evolution_service._update_pattern = MagicMock()
            pattern_evolution_service._check_quality_transition = MagicMock()
            
            # Act
            pattern_evolution_service.track_pattern_usage(pattern_id, context)
            
            # Assert
            # Check that the adapter methods were called correctly
            mock_adapter.update_temporal_context.assert_called_once()
            mock_adapter.create_version.assert_called_once()
            mock_adapter.get_pattern.assert_called_once()
            
            # Check that the document was created in ArangoDB
            mock_arangodb_connection.create_document.assert_called_once()
            
            # Check that _check_quality_transition was called
            pattern_evolution_service._check_quality_transition.assert_called_once()
            
            # Check that _update_pattern was called
            pattern_evolution_service._update_pattern.assert_called_once()
            
            # Check that the pattern was published
            pattern_evolution_service.bidirectional_flow_service.publish_pattern.assert_called_once()

    def test_track_pattern_feedback_with_adaptive_id(self, pattern_evolution_service, sample_pattern, mock_arangodb_connection):
        """Test that track_pattern_feedback correctly uses AdaptiveID for versioning and context tracking."""
        # Arrange
        pattern_id = sample_pattern["id"]
        feedback = {"rating": 4, "comment": "Good pattern", "user_id": "test_user"}
        
        # Mock _get_pattern to return the sample pattern
        pattern_evolution_service._get_pattern = MagicMock(return_value=sample_pattern)
        
        # Mock the PatternAdaptiveIDAdapter
        mock_adapter = MagicMock()
        mock_adapter.get_pattern.return_value = Pattern(
            id=sample_pattern["id"],
            base_concept=sample_pattern["base_concept"],
            creator_id=sample_pattern["creator_id"],
            weight=sample_pattern["weight"],
            confidence=sample_pattern["confidence"],
            uncertainty=sample_pattern["uncertainty"],
            coherence=sample_pattern["coherence"],
            phase_stability=sample_pattern["phase_stability"],
            signal_strength=sample_pattern["signal_strength"]
        )
        mock_adapter.get_pattern.return_value.metrics = {"feedback_count": 3}
        mock_adapter.get_pattern.return_value.to_dict = MagicMock(return_value={
            "id": sample_pattern["id"],
            "base_concept": sample_pattern["base_concept"],
            "confidence": sample_pattern["confidence"],
            "metrics": {"feedback_count": 3},
            "quality": {
                "score": 0.7,
                "usage_count": 5,
                "feedback_count": 3,
                "last_used": sample_pattern["quality"]["last_used"],
                "last_feedback": datetime.now().isoformat()
            }
        })
        
        # Patch the PatternAdaptiveIDAdapter constructor
        with patch('src.habitat_evolution.infrastructure.services.pattern_evolution_service.PatternAdaptiveIDAdapter', return_value=mock_adapter):
            # Mock other methods
            pattern_evolution_service._update_pattern = MagicMock()
            pattern_evolution_service._check_quality_transition = MagicMock()
            
            # Act
            pattern_evolution_service.track_pattern_feedback(pattern_id, feedback)
            
            # Assert
            # Check that the adapter methods were called correctly
            mock_adapter.update_temporal_context.assert_called_once()
            mock_adapter.create_version.assert_called_once()
            mock_adapter.get_pattern.assert_called_once()
            
            # Check that the document was created in ArangoDB
            mock_arangodb_connection.create_document.assert_called_once()
            
            # Check that _check_quality_transition was called
            pattern_evolution_service._check_quality_transition.assert_called_once()
            
            # Check that _update_pattern was called
            pattern_evolution_service._update_pattern.assert_called_once()
            
            # Check that the pattern was published
            pattern_evolution_service.bidirectional_flow_service.publish_pattern.assert_called_once()

    def test_get_pattern_evolution_with_adaptive_id(self, pattern_evolution_service, sample_pattern, mock_arangodb_connection):
        """Test that get_pattern_evolution correctly retrieves the evolution history using AdaptiveID."""
        # Arrange
        pattern_id = sample_pattern["id"]
        
        # Mock _get_pattern to return the sample pattern
        pattern_evolution_service._get_pattern = MagicMock(return_value=sample_pattern)
        
        # Mock ArangoDB query results
        quality_transitions = [
            {
                "pattern_id": pattern_id,
                "old_state": None,
                "new_state": "hypothetical",
                "reason": "creation",
                "timestamp": (datetime.now().replace(hour=1)).isoformat()
            },
            {
                "pattern_id": pattern_id,
                "old_state": "hypothetical",
                "new_state": "candidate",
                "reason": "Met criteria for candidate pattern",
                "timestamp": (datetime.now().replace(hour=2)).isoformat()
            }
        ]
        
        usage_history = [
            {
                "pattern_id": pattern_id,
                "context": {"query": "test query 1"},
                "timestamp": (datetime.now().replace(hour=1, minute=30)).isoformat()
            },
            {
                "pattern_id": pattern_id,
                "context": {"query": "test query 2"},
                "timestamp": (datetime.now().replace(hour=1, minute=45)).isoformat()
            }
        ]
        
        feedback_history = [
            {
                "pattern_id": pattern_id,
                "feedback": {"rating": 4},
                "timestamp": (datetime.now().replace(hour=1, minute=40)).isoformat()
            }
        ]
        
        # Setup mock ArangoDB connection to return these results
        mock_arangodb_connection.execute_aql.side_effect = [
            quality_transitions,
            usage_history,
            feedback_history
        ]
        
        # Mock the PatternAdaptiveIDAdapter
        mock_adapter = MagicMock()
        mock_adapter.adaptive_id.id = "adaptive_id_123"
        mock_adapter.adaptive_id.metadata = {
            "version_count": 4,
            "created_at": (datetime.now().replace(hour=1)).isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        # Mock version history from AdaptiveID
        version_history = [
            MagicMock(
                timestamp=(datetime.now().replace(hour=1)).isoformat(),
                version_id="v1",
                origin="creation",
                data={"base_concept": "test_concept"}
            ),
            MagicMock(
                timestamp=(datetime.now().replace(hour=1, minute=30)).isoformat(),
                version_id="v2",
                origin="pattern_usage",
                data={"usage_count": 1}
            ),
            MagicMock(
                timestamp=(datetime.now().replace(hour=1, minute=40)).isoformat(),
                version_id="v3",
                origin="pattern_feedback",
                data={"feedback_count": 1}
            ),
            MagicMock(
                timestamp=(datetime.now().replace(hour=2)).isoformat(),
                version_id="v4",
                origin="quality_update",
                data={"quality_state": "candidate"}
            )
        ]
        mock_adapter.get_version_history.return_value = version_history
        
        # Patch the PatternAdaptiveIDAdapter constructor
        with patch('src.habitat_evolution.infrastructure.services.pattern_evolution_service.PatternAdaptiveIDAdapter', return_value=mock_adapter):
            # Act
            result = pattern_evolution_service.get_pattern_evolution(pattern_id)
            
            # Assert
            # Check that the adapter methods were called correctly
            mock_adapter.get_version_history.assert_called_once()
            
            # Check that ArangoDB queries were executed
            assert mock_arangodb_connection.execute_aql.call_count == 3
            
            # Check the result structure
            assert result["pattern_id"] == pattern_id
            assert result["current_state"] == sample_pattern["quality_state"]
            assert len(result["timeline"]) == 9  # 2 quality transitions + 2 usages + 1 feedback + 4 versions
            assert result["adaptive_id"]["id"] == "adaptive_id_123"
            assert result["adaptive_id"]["version_count"] == 4
            assert result["status"] == "success"

    def test_update_pattern_quality_with_adaptive_id(self, pattern_evolution_service, sample_pattern):
        """Test that update_pattern_quality correctly uses AdaptiveID for versioning."""
        # Arrange
        pattern_id = sample_pattern["id"]
        quality_metrics = {
            "score": 0.8,
            "confidence_factor": 0.9
        }
        
        # Mock _get_pattern to return the sample pattern
        pattern_evolution_service._get_pattern = MagicMock(return_value=sample_pattern)
        
        # Mock the PatternAdaptiveIDAdapter
        mock_adapter = MagicMock()
        
        # Patch the PatternAdaptiveIDAdapter constructor
        with patch('src.habitat_evolution.infrastructure.services.pattern_evolution_service.PatternAdaptiveIDAdapter', return_value=mock_adapter):
            # Mock other methods
            pattern_evolution_service._update_pattern = MagicMock()
            pattern_evolution_service._check_quality_transition = MagicMock()
            
            # Act
            pattern_evolution_service.update_pattern_quality(pattern_id, quality_metrics)
            
            # Assert
            # Check that the adapter methods were called correctly
            mock_adapter.create_version.assert_called_once()
            
            # Check that _check_quality_transition was called
            pattern_evolution_service._check_quality_transition.assert_called_once()
            
            # Check that _update_pattern was called
            pattern_evolution_service._update_pattern.assert_called_once()
            
            # Check that the pattern was published
            pattern_evolution_service.bidirectional_flow_service.publish_pattern.assert_called_once()

    def test_check_quality_transition_hypothetical_to_candidate(self, pattern_evolution_service):
        """Test that _check_quality_transition correctly transitions from hypothetical to candidate."""
        # Arrange
        pattern = {
            "id": "test_pattern",
            "quality_state": "hypothetical",
            "confidence": 0.7,
            "quality": {
                "usage_count": 5,
                "feedback_count": 2
            }
        }
        
        # Mock _track_quality_transition
        pattern_evolution_service._track_quality_transition = MagicMock()
        
        # Act
        pattern_evolution_service._check_quality_transition(pattern)
        
        # Assert
        assert pattern["quality_state"] == "candidate"
        pattern_evolution_service._track_quality_transition.assert_called_once_with(
            "test_pattern",
            "hypothetical",
            "candidate",
            "Met criteria for candidate pattern"
        )

    def test_check_quality_transition_no_change(self, pattern_evolution_service):
        """Test that _check_quality_transition doesn't change state when criteria aren't met."""
        # Arrange
        pattern = {
            "id": "test_pattern",
            "quality_state": "hypothetical",
            "confidence": 0.5,  # Below threshold
            "quality": {
                "usage_count": 5,
                "feedback_count": 2
            }
        }
        
        # Mock _track_quality_transition
        pattern_evolution_service._track_quality_transition = MagicMock()
        
        # Act
        pattern_evolution_service._check_quality_transition(pattern)
        
        # Assert
        assert pattern["quality_state"] == "hypothetical"
        pattern_evolution_service._track_quality_transition.assert_not_called()

    def test_identify_emerging_patterns(self, pattern_evolution_service, mock_arangodb_connection):
        """Test that identify_emerging_patterns correctly identifies emerging patterns."""
        # Arrange
        threshold = 0.7
        
        # Mock ArangoDB query results
        patterns = [
            {
                "id": "pattern1",
                "quality_state": "hypothetical",
                "confidence": 0.8,
                "quality": {
                    "usage_count": 3,
                    "feedback_count": 2
                }
            },
            {
                "id": "pattern2",
                "quality_state": "hypothetical",
                "confidence": 0.75,
                "quality": {
                    "usage_count": 1,  # Below threshold
                    "feedback_count": 1
                }
            },
            {
                "id": "pattern3",
                "quality_state": "hypothetical",
                "confidence": 0.9,
                "quality": {
                    "usage_count": 4,
                    "feedback_count": 3
                }
            }
        ]
        
        # Setup mock ArangoDB connection to return these results
        mock_arangodb_connection.execute_aql.return_value = patterns
        
        # Act
        result = pattern_evolution_service.identify_emerging_patterns(threshold)
        
        # Assert
        assert len(result) == 2
        assert result[0]["id"] == "pattern1"
        assert result[1]["id"] == "pattern3"
        # pattern2 should be excluded due to low usage count
