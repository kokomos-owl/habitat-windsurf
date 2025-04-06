"""
Simplified test for bidirectional flow between PatternAwareRAG and vector-tonic system.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import uuid

# Import fix for Habitat Evolution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import with adapter architecture
from src.habitat_evolution.core.services.event_bus import LocalEventBus as EventManagementService, Event
from src.habitat_evolution.adaptive_core.models.pattern import Pattern
from src.habitat_evolution.adaptive_core.emergence.interfaces.learning_window_observer import LearningWindowState
from typing import Dict, Any, List, Optional

# Create a simplified version of BidirectionalFlowManager for testing
class StateChangeBuffer:
    """Buffer for managing state changes to prevent feedback loops."""
    
    def __init__(self, max_size: int = 100, expiry_seconds: int = 60):
        """Initialize the state change buffer.
        
        Args:
            max_size: Maximum number of changes to track
            expiry_seconds: Time in seconds before a change is considered expired
        """
        self.changes = {}
        self.max_size = max_size
        self.expiry_seconds = expiry_seconds
    
    def add_change(self, entity_id: str, change_type: str, data: Any) -> None:
        """Add a change to the buffer."""
        key = f"{change_type}:{entity_id}"
        self.changes[key] = {
            "timestamp": datetime.now(),
            "data": data
        }
    
    def has_recent_change(self, entity_id: str, change_type: str) -> bool:
        """Check if an entity has changed recently."""
        key = f"{change_type}:{entity_id}"
        return key in self.changes

class BidirectionalFlowManager:
    """Simplified manager for bidirectional flow between PatternAwareRAG and vector-tonic system."""
    
    def __init__(self, event_bus, semantic_potential_calculator=None):
        """Initialize the bidirectional flow manager."""
        self.event_bus = event_bus
        self.state_change_buffer = StateChangeBuffer()
        self.semantic_potential_calculator = semantic_potential_calculator
        self.coherence_threshold = 0.3
        self.constructive_dissonance_allowance = 0.1
        self.current_window_states = {}
        self.correlated_prompts = {}
    
    def is_coherent_change(self, entity_id: str, data: Any) -> bool:
        """Determine if a change is coherent and should be processed."""
        # Check if this is a duplicate/recent change
        if self.state_change_buffer.has_recent_change(entity_id, "pattern"):
            return False
            
        # Add to buffer to prevent feedback loops
        self.state_change_buffer.add_change(entity_id, "pattern", data)
        
        # Check coherence threshold with constructive dissonance allowance
        if hasattr(data, 'coherence'):
            # If pattern has constructive dissonance potential, apply allowance
            constructive_dissonance = getattr(data, 'constructive_dissonance', 0.0)
            effective_threshold = self.coherence_threshold - (constructive_dissonance * self.constructive_dissonance_allowance)
            
            if data.coherence < effective_threshold:
                return False
            
        return True
    
    def is_coherent_field_change(self, field_id: str, state: Dict[str, Any]) -> bool:
        """Determine if a field change is coherent and should be processed."""
        # Check if this is a duplicate/recent change
        if self.state_change_buffer.has_recent_change(field_id, "field"):
            return False
            
        # Add to buffer to prevent feedback loops
        self.state_change_buffer.add_change(field_id, "field", state)
        
        # Check coherence threshold with constructive dissonance allowance
        coherence = state.get("coherence", 0.0)
        constructive_dissonance = state.get("constructive_dissonance", 0.0)
        effective_threshold = self.coherence_threshold - (constructive_dissonance * self.constructive_dissonance_allowance)
        
        if coherence < effective_threshold:
            return False
            
        return True
        
    def update_window_state(self, window_id: str, old_state: str, new_state: str) -> None:
        """Update the window state and perform appropriate actions."""
        self.current_window_states[window_id] = new_state
        
        # Generate correlated prompts based on window state transitions
        if old_state == LearningWindowState.CLOSED.value and new_state == LearningWindowState.OPENING.value:
            self.correlated_prompts[window_id] = "Preparing to learn new patterns..."
        elif old_state == LearningWindowState.OPENING.value and new_state == LearningWindowState.OPEN.value:
            self.correlated_prompts[window_id] = "Actively learning and evolving patterns..."
        elif old_state == LearningWindowState.OPEN.value and new_state == LearningWindowState.CLOSING.value:
            self.correlated_prompts[window_id] = "Consolidating learned patterns..."
        elif old_state == LearningWindowState.CLOSING.value and new_state == LearningWindowState.CLOSED.value:
            self.correlated_prompts[window_id] = "Pattern learning complete, ready for retrieval."
            
    def get_correlated_prompt(self, window_id: str) -> str:
        """Get the correlated prompt for a window."""
        return self.correlated_prompts.get(window_id, "")

# Configure colorful logging
try:
    import colorlog
    
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )
    
    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Success for test assertions
    def log_success(message):
        logger.info(f"\033[1;32m✓ SUCCESS: {message}\033[0m")
        
    # Failure for test assertions (though assertions will raise exceptions)
    def log_failure(message):
        logger.error(f"\033[1;31m✗ FAILED: {message}\033[0m")
        
    # Test section headers
    def log_test_start(test_name):
        logger.info(f"\033[1;36m{'=' * 20} STARTING TEST: {test_name} {'=' * 20}\033[0m")
        
    def log_test_end(test_name):
        logger.info(f"\033[1;36m{'=' * 20} COMPLETED TEST: {test_name} {'=' * 20}\033[0m")
        
    # Pattern and event details
    def log_pattern(pattern):
        logger.info(f"\033[1;34mPATTERN: {pattern.id} - {pattern.base_concept} (coherence: {pattern.coherence})\033[0m")
        logger.debug(f"  Content: {pattern.properties.get('content', 'No content')}")
        
    def log_event(event):
        logger.info(f"\033[1;35mEVENT: {event.type} from {event.source}\033[0m")
        logger.debug(f"  Data: {event.data}")
        
except ImportError:
    # Fallback to standard logging if colorlog is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Plain text versions of the logging functions
    def log_success(message):
        logger.info(f"SUCCESS: {message}")
        
    def log_failure(message):
        logger.error(f"FAILED: {message}")
        
    def log_test_start(test_name):
        logger.info(f"{'=' * 20} STARTING TEST: {test_name} {'=' * 20}")
        
    def log_test_end(test_name):
        logger.info(f"{'=' * 20} COMPLETED TEST: {test_name} {'=' * 20}")
        
    def log_pattern(pattern):
        logger.info(f"PATTERN: {pattern.id} - {pattern.base_concept} (coherence: {pattern.coherence})")
        logger.debug(f"  Content: {pattern.properties.get('content', 'No content')}")
        
    def log_event(event):
        logger.info(f"EVENT: {event.type} from {event.source}")
        logger.debug(f"  Data: {event.data}")

# Test class for bidirectional flow
class BidirectionalFlowTest:
    """Test class for bidirectional flow between PatternAwareRAG and vector-tonic system."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize the test environment.
        
        Args:
            data_dir: Directory containing test data
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "data" / "climate_risk"
        
        # Create shared event bus
        self.event_bus = EventManagementService()
        
        # Create bidirectional flow manager
        self.bidirectional_flow = BidirectionalFlowManager(
            event_bus=self.event_bus,
            semantic_potential_calculator=None
        )
        
        # Initialize event tracking
        self.detected_patterns = []
        self.field_state_updates = []
        self.window_state_changes = []
    
    async def setup(self):
        """Set up the test environment."""
        logger.info("Setting up test environment...")
        
        # Subscribe to events for testing
        self.event_bus.subscribe("pattern.detected", self._on_pattern_detected)
        self.event_bus.subscribe("field.state.updated", self._on_field_state_updated)
        self.event_bus.subscribe("window.state.changed", self._on_window_state_changed)
    
    def _on_pattern_detected(self, event: Event):
        """Handle pattern detected events."""
        event_data = event.data
        log_event(event)
        pattern = event_data.get('pattern')
        if hasattr(pattern, 'id') and hasattr(pattern, 'base_concept'):
            log_pattern(pattern)
        else:
            logger.info(f"Pattern detected: {event_data.get('pattern_id')} (direction: {event_data.get('direction')})")
        self.detected_patterns.append(event_data)
    
    def _on_field_state_updated(self, event: Event):
        """Handle field state updated events."""
        event_data = event.data
        log_event(event)
        field_id = event_data.get('field_id')
        state = event_data.get('state', {})
        logger.info(f"Field state updated: {field_id} (coherence: {state.get('coherence')}, direction: {event_data.get('direction')})")
        self.field_state_updates.append(event_data)
    
    def _on_window_state_changed(self, event: Event):
        """Handle window state changed events."""
        event_data = event.data
        log_event(event)
        window_id = event_data.get('window_id')
        old_state = event_data.get('old_state')
        new_state = event_data.get('new_state')
        logger.info(f"Window state transition: {window_id} {old_state} → {new_state} (direction: {event_data.get('direction')})")
        self.window_state_changes.append(event_data)
        
    async def load_climate_risk_document(self) -> str:
        """Load climate risk document."""
        logger.info("Loading climate risk document...")
        climate_risk_path = self.data_dir / "climate_risk_marthas_vineyard.txt"
        
        with open(climate_risk_path, "r") as f:
            climate_risk_doc = f.read()
            
        return climate_risk_doc
    
    async def test_ingestion_direction(self):
        """Test ingestion direction flow (RAG → vector-tonic)."""
        log_test_start("INGESTION DIRECTION FLOW")
        
        # Create a pattern
        pattern_id = str(uuid.uuid4())
        pattern = Pattern(
            id=pattern_id,
            base_concept="flood risk",
            creator_id="rag_system",
            weight=1.0,
            confidence=0.9,
            uncertainty=0.1,
            coherence=0.88,
            phase_stability=0.75,
            signal_strength=0.82,
            properties={
                "content": "Martha's Vineyard faces increasing flood risks with the historical 100-year rainfall event becoming 5 times more likely by late-century."
            },
            metrics={
                "coherence": 0.88,
                "density": 0.82,
                "stability": 0.75
            },
            state="EMERGING"
        )
        
        # Publish pattern in ingestion direction
        event = Event.create(
            type="pattern.detected",
            data={
                "pattern_id": pattern_id,
                "pattern": pattern,
                "direction": "ingestion"
            },
            source="rag"
        )
        self.event_bus.publish(event)
        
        # Wait for events to propagate
        await asyncio.sleep(0.5)
        
        # Check if pattern was detected
        assert len(self.detected_patterns) > 0, "No patterns were detected"
        detected = self.detected_patterns[0]
        assert detected.get("direction") == "ingestion", "Pattern direction should be ingestion"
        assert detected.get("pattern_id") == pattern_id, "Pattern ID mismatch"
        
        log_success("Ingestion direction test passed!")
        log_test_end("INGESTION DIRECTION FLOW")
    
    async def test_retrieval_direction(self):
        """Test retrieval direction flow (vector-tonic → RAG)."""
        log_test_start("RETRIEVAL DIRECTION FLOW")
        
        # Reset event tracking
        self.field_state_updates = []
        
        # Publish field state update in retrieval direction
        event = Event.create(
            type="field.state.updated",
            data={
                "field_id": "climate_risk_field",
                "state": {
                    "coherence": 0.85,
                    "density": 0.78,
                    "stability": 0.72,
                    "cross_paths": ["climate_adaptation", "economic_impact"],
                    "back_pressure": 0.45
                },
                "direction": "retrieval"
            },
            source="vector_tonic"
        )
        self.event_bus.publish(event)
        
        # Wait for events to propagate
        await asyncio.sleep(0.5)
        
        # Check if field state update was processed
        assert len(self.field_state_updates) > 0, "No field state updates were processed"
        update = self.field_state_updates[0]
        assert update.get("direction") == "retrieval", "Field state update direction should be retrieval"
        assert update.get("field_id") == "climate_risk_field", "Field ID mismatch"
        
        log_success("Retrieval direction test passed!")
        log_test_end("RETRIEVAL DIRECTION FLOW")
    
    async def test_coherence_based_filtering(self):
        """Test coherence-based filtering in bidirectional flow."""
        log_test_start("COHERENCE-BASED FILTERING")
        
        # Test 1: Pattern in buffer should be filtered
        pattern_id = str(uuid.uuid4())
        pattern = Pattern(
            id=pattern_id,
            base_concept="drought analysis",
            creator_id="vector_tonic_system",
            weight=0.6,
            confidence=0.5,
            uncertainty=0.4,
            coherence=0.35,
            phase_stability=0.38,
            signal_strength=0.42,
            properties={
                "content": "Drought occurrence was analyzed using a water balance model based on precipitation and atmospheric evaporative demand."
            },
            metrics={
                "coherence": 0.35,
                "density": 0.42,
                "stability": 0.38
            },
            state="EMERGING"
        )
        
        # Add pattern to buffer to simulate recent change
        self.bidirectional_flow.state_change_buffer.add_change(pattern_id, "pattern", pattern)
        
        # Check if pattern would be filtered due to being in buffer
        filtered = self.bidirectional_flow.is_coherent_change(pattern_id, pattern)
        assert not filtered, "Pattern should be filtered due to being in buffer"
        
        # Test 2: High coherence pattern should pass filtering
        high_coherence_pattern_id = str(uuid.uuid4())
        high_coherence_pattern = Pattern(
            id=high_coherence_pattern_id,
            base_concept="wildfire danger",
            creator_id="vector_tonic_system",
            weight=1.0,
            confidence=0.95,
            uncertainty=0.05,
            coherence=0.92,  # High coherence
            phase_stability=0.78,
            signal_strength=0.85,
            properties={
                "content": "Martha's Vineyard can expect an average additional eight wildfire danger days per year, representing a 44% increase over the historical annual average of 18 days."
            },
            metrics={
                "coherence": 0.92,
                "density": 0.85,
                "stability": 0.78
            },
            state="EMERGING"
        )
        
        # Check if high coherence pattern would pass filtering
        passed = self.bidirectional_flow.is_coherent_change(high_coherence_pattern_id, high_coherence_pattern)
        assert passed, "High coherence pattern should pass filtering"
        
        # Test 3: Pattern exactly at coherence threshold
        threshold_pattern_id = str(uuid.uuid4())
        threshold_pattern = Pattern(
            id=threshold_pattern_id,
            base_concept="sea level rise",
            creator_id="vector_tonic_system",
            weight=0.7,
            confidence=0.6,
            uncertainty=0.3,
            coherence=0.3,  # Exactly at threshold
            phase_stability=0.4,
            signal_strength=0.5,
            properties={
                "content": "Sea level rise projections for Martha's Vineyard indicate a potential increase of 2.5 feet by 2050."
            },
            metrics={
                "coherence": 0.3,
                "density": 0.4,
                "stability": 0.5
            },
            state="EMERGING"
        )
        
        # Check if threshold pattern would pass filtering
        threshold_passed = self.bidirectional_flow.is_coherent_change(threshold_pattern_id, threshold_pattern)
        assert threshold_passed, "Pattern at exact coherence threshold should pass filtering"
        
        log_success("Coherence-based filtering test passed!")
        log_test_end("COHERENCE-BASED FILTERING")
    
    async def test_window_state_cycle(self):
        """Test complete window state cycle."""
        log_test_start("WINDOW STATE CYCLE")
        
        window_id = "climate_risk_window"
        
        # Test all window state transitions
        state_transitions = [
            (LearningWindowState.CLOSED.value, LearningWindowState.OPENING.value),
            (LearningWindowState.OPENING.value, LearningWindowState.OPEN.value),
            (LearningWindowState.OPEN.value, LearningWindowState.CLOSING.value),
            (LearningWindowState.CLOSING.value, LearningWindowState.CLOSED.value)
        ]
        
        for old_state, new_state in state_transitions:
            # Reset event tracking
            self.window_state_changes = []
            
            # Publish window state change
            event = Event.create(
                type="window.state.changed",
                data={
                    "window_id": window_id,
                    "old_state": old_state,
                    "new_state": new_state,
                    "direction": "retrieval"
                },
                source="vector_tonic"
            )
            self.event_bus.publish(event)
            
            # Update bidirectional flow manager's window state
            self.bidirectional_flow.update_window_state(window_id, old_state, new_state)
            
            # Wait for events to propagate
            await asyncio.sleep(0.5)
            
            # Check if window state change was processed
            assert len(self.window_state_changes) > 0, f"No window state changes were processed for {old_state} -> {new_state}"
            change = self.window_state_changes[0]
            assert change.get("window_id") == window_id, "Window ID mismatch"
            assert change.get("old_state") == old_state, "Old window state mismatch"
            assert change.get("new_state") == new_state, "New window state mismatch"
            
            # Check if correlated prompt was generated
            prompt = self.bidirectional_flow.get_correlated_prompt(window_id)
            assert prompt, f"No correlated prompt generated for {old_state} -> {new_state}"
            logger.info(f"\033[1;33mWindow state transition {old_state} → {new_state}\033[0m")
            logger.info(f"\033[1;32mCorrelated prompt: \"{prompt}\"\033[0m")
        
        log_success("Window state cycle test passed!")
        log_test_end("WINDOW STATE CYCLE")
    
    async def test_constructive_dissonance(self):
        """Test constructive dissonance allowance in filtering."""
        log_test_start("CONSTRUCTIVE DISSONANCE ALLOWANCE")
        
        # Create a pattern with low coherence but high constructive dissonance
        pattern_id = str(uuid.uuid4())
        pattern = Pattern(
            id=pattern_id,
            base_concept="climate adaptation strategies",
            creator_id="vector_tonic_system",
            weight=0.5,
            confidence=0.4,
            uncertainty=0.6,
            coherence=0.25,  # Below threshold
            phase_stability=0.3,
            signal_strength=0.4,
            properties={
                "content": "Novel climate adaptation strategies combining traditional knowledge with advanced technology show promise for coastal communities.",
                "constructive_dissonance": 0.8  # High constructive dissonance
            },
            metrics={
                "coherence": 0.25,
                "density": 0.3,
                "stability": 0.4,
                "constructive_dissonance": 0.8
            },
            state="EMERGING"
        )
        
        # Add constructive_dissonance attribute dynamically
        setattr(pattern, 'constructive_dissonance', 0.8)
        
        # Check if pattern with high constructive dissonance passes despite low coherence
        passed = self.bidirectional_flow.is_coherent_change(pattern_id, pattern)
        assert passed, "Pattern with high constructive dissonance should pass despite low coherence"
        
        log_success("Constructive dissonance test passed!")
        log_test_end("CONSTRUCTIVE DISSONANCE ALLOWANCE")
    
    async def test_correlated_prompt_validation(self):
        """Test correlated prompt generation and validation."""
        log_test_start("CORRELATED PROMPT VALIDATION")
        
        window_id = "climate_risk_window"
        
        # Test prompt generation for CLOSED -> OPENING transition
        self.bidirectional_flow.update_window_state(
            window_id, 
            LearningWindowState.CLOSED.value, 
            LearningWindowState.OPENING.value
        )
        
        prompt = self.bidirectional_flow.get_correlated_prompt(window_id)
        assert prompt, "No correlated prompt generated for CLOSED -> OPENING transition"
        assert "Preparing" in prompt, "Prompt should indicate preparation phase"
        
        # Test prompt generation for OPENING -> OPEN transition
        self.bidirectional_flow.update_window_state(
            window_id, 
            LearningWindowState.OPENING.value, 
            LearningWindowState.OPEN.value
        )
        
        prompt = self.bidirectional_flow.get_correlated_prompt(window_id)
        assert prompt, "No correlated prompt generated for OPENING -> OPEN transition"
        assert "learning" in prompt.lower(), "Prompt should indicate active learning phase"
        
        log_success("Correlated prompt validation test passed!")
        log_test_end("CORRELATED PROMPT VALIDATION")
    
    async def test_feedback_loop_prevention(self):
        """Test feedback loop prevention mechanisms."""
        log_test_start("FEEDBACK LOOP PREVENTION")
        
        # Create a pattern
        pattern_id = str(uuid.uuid4())
        pattern = Pattern(
            id=pattern_id,
            base_concept="temperature increase",
            creator_id="rag_system",
            weight=0.8,
            confidence=0.7,
            uncertainty=0.2,
            coherence=0.8,
            phase_stability=0.7,
            signal_strength=0.75,
            properties={
                "content": "Martha's Vineyard is projected to experience a 5°F increase in average temperature by 2050."
            },
            metrics={
                "coherence": 0.8,
                "density": 0.75,
                "stability": 0.7
            },
            state="EMERGING"
        )
        
        # First attempt should pass
        first_pass = self.bidirectional_flow.is_coherent_change(pattern_id, pattern)
        assert first_pass, "First attempt should pass filtering"
        
        # Second attempt with same pattern should be filtered (prevent feedback loop)
        second_pass = self.bidirectional_flow.is_coherent_change(pattern_id, pattern)
        assert not second_pass, "Second attempt with same pattern should be filtered"
        
        # Create a field state
        field_id = "climate_temperature_field"
        field_state = {
            "coherence": 0.85,
            "density": 0.8,
            "stability": 0.75
        }
        
        # First attempt should pass
        first_field_pass = self.bidirectional_flow.is_coherent_field_change(field_id, field_state)
        assert first_field_pass, "First field state change should pass filtering"
        
        # Second attempt with same field state should be filtered
        second_field_pass = self.bidirectional_flow.is_coherent_field_change(field_id, field_state)
        assert not second_field_pass, "Second field state change should be filtered"
        
        log_success("Feedback loop prevention test passed!")
        log_test_end("FEEDBACK LOOP PREVENTION")
    
    async def run_all_tests(self):
        """Run all tests."""
        await self.setup()
        await self.test_ingestion_direction()
        await self.test_retrieval_direction()
        await self.test_coherence_based_filtering()
        await self.test_window_state_cycle()
        await self.test_constructive_dissonance()
        await self.test_correlated_prompt_validation()
        await self.test_feedback_loop_prevention()
        logger.info("\033[1;32m✅ ALL TESTS COMPLETED SUCCESSFULLY! ✅\033[0m")


# Run the test
if __name__ == "__main__":
    test = BidirectionalFlowTest()
    asyncio.run(test.run_all_tests())
