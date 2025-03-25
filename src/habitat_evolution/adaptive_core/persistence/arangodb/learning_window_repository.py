"""
Learning Window Repository for ArangoDB.

Handles persistence of Learning Windows and their states to ArangoDB.
Implements the temporal framework for pattern evolution tracking.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import uuid
from datetime import datetime

from .base_repository import ArangoDBBaseRepository
from .connection import ArangoDBConnectionManager
from ...learning.window import LearningWindow, WindowState

logger = logging.getLogger(__name__)

class LearningWindowRepository(ArangoDBBaseRepository):
    """
    Repository for persisting LearningWindow objects to ArangoDB.
    
    Handles the storage and retrieval of learning windows, their states,
    and the patterns they contain. Supports temporal queries and window
    coordination.
    """
    
    def __init__(self):
        """Initialize the LearningWindow repository."""
        super().__init__()
        self.collection_name = "LearningWindow"
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Edge collections we'll be working with
        self.window_contains_pattern_collection = "WindowContainsPattern"
        
        # Document collections for related entities
        self.temporal_pattern_collection = "TemporalPattern"
    
    def _dict_to_entity(self, data: Dict[str, Any]) -> LearningWindow:
        """
        Convert a dictionary to a LearningWindow object.
        
        Args:
            data: Dictionary containing LearningWindow data
            
        Returns:
            LearningWindow object
        """
        window = LearningWindow(
            window_id=data.get("window_id", str(uuid.uuid4())),
            stability_threshold=data.get("stability_threshold", 0.7),
            max_size=data.get("max_size", 100)
        )
        
        # Set additional properties
        window.state = WindowState(data.get("state", WindowState.CLOSED.value))
        window.start_time = data.get("start_time")
        window.end_time = data.get("end_time")
        
        return window
    
    def save(self, window: LearningWindow) -> str:
        """
        Save a LearningWindow to ArangoDB.
        
        This method saves the window and its state to the database.
        
        Args:
            window: LearningWindow object to save
            
        Returns:
            ID of the saved window
        """
        # Convert window to dict
        window_dict = {
            "window_id": window.window_id,
            "state": window.state.value,
            "stability_threshold": window.stability_threshold,
            "max_size": window.max_size,
            "start_time": window.start_time,
            "end_time": window.end_time,
            "current_size": len(window.observations) if hasattr(window, "observations") else 0,
            "updated_at": datetime.now().isoformat()
        }
        
        # Check if window already exists
        existing_window = self.find_by_window_id(window.window_id)
        
        if existing_window:
            # Update existing window
            self.update(existing_window.window_id, window)
            return existing_window.window_id
        else:
            # Create new window
            window_dict["created_at"] = datetime.now().isoformat()
            window_dict["_key"] = window.window_id
            
            result = self.db.collection(self.collection_name).insert(window_dict, return_new=True)
            return result["_key"]
    
    def save_window_state_transition(self, window: LearningWindow, old_state: WindowState) -> None:
        """
        Save a window state transition to ArangoDB.
        
        This method records when a learning window changes state, which is
        important for tracking the temporal evolution of patterns.
        
        Args:
            window: LearningWindow that changed state
            old_state: Previous state of the window
        """
        # Create a state transition record
        transition = {
            "_key": str(uuid.uuid4()),
            "window_id": window.window_id,
            "old_state": old_state.value,
            "new_state": window.state.value,
            "timestamp": datetime.now().isoformat(),
            "observation_count": len(window.observations) if hasattr(window, "observations") else 0
        }
        
        # We'll use a separate collection for state transitions
        if not self.db.has_collection("WindowStateTransition"):
            self.db.create_collection("WindowStateTransition")
        
        self.db.collection("WindowStateTransition").insert(transition)
        
        # Update the window record with the new state
        self.save(window)
    
    def save_pattern(self, window: LearningWindow, pattern: Dict[str, Any]) -> str:
        """
        Save a pattern detected in a learning window.
        
        Args:
            window: LearningWindow that detected the pattern
            pattern: Dictionary containing pattern information
            
        Returns:
            ID of the saved pattern
        """
        # Ensure pattern has required fields
        if "id" not in pattern:
            pattern["id"] = str(uuid.uuid4())
        
        pattern["_key"] = pattern["id"]
        pattern["detected_at"] = datetime.now().isoformat()
        pattern["window_id"] = window.window_id
        
        # Check if pattern already exists
        existing_pattern = self._find_pattern_by_id(pattern["id"])
        
        pattern_id = None
        
        if existing_pattern:
            # Update existing pattern
            pattern_id = existing_pattern["_key"]
            self.db.collection(self.temporal_pattern_collection).update(pattern_id, pattern)
        else:
            # Create new pattern
            result = self.db.collection(self.temporal_pattern_collection).insert(pattern, return_new=True)
            pattern_id = result["_key"]
            
            # Create edge from Window to Pattern
            self._create_window_pattern_edge(window.window_id, pattern_id)
        
        return pattern_id
    
    def _create_window_pattern_edge(self, window_id: str, pattern_id: str) -> str:
        """
        Create an edge from a LearningWindow to a TemporalPattern.
        
        Args:
            window_id: ID of the LearningWindow
            pattern_id: ID of the TemporalPattern
            
        Returns:
            ID of the created edge
        """
        # Create the edge
        edge = {
            "_from": f"{self.collection_name}/{window_id}",
            "_to": f"{self.temporal_pattern_collection}/{pattern_id}",
            "created_at": datetime.now().isoformat()
        }
        
        result = self.db.collection(self.window_contains_pattern_collection).insert(edge, return_new=True)
        return result["_key"]
    
    def _find_pattern_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a TemporalPattern by its ID.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Dictionary containing the pattern, or None if not found
        """
        try:
            return self.db.collection(self.temporal_pattern_collection).get(pattern_id)
        except:
            return None
    
    def find_by_window_id(self, window_id: str) -> Optional[LearningWindow]:
        """
        Find a LearningWindow by its ID.
        
        Args:
            window_id: ID of the window
            
        Returns:
            LearningWindow object, or None if not found
        """
        try:
            window_dict = self.db.collection(self.collection_name).get(window_id)
            if not window_dict:
                return None
            
            return self._dict_to_entity(window_dict)
        except:
            return None
    
    def find_windows_by_time_range(self, start_time: str, end_time: str) -> List[LearningWindow]:
        """
        Find learning windows within a time range.
        
        Args:
            start_time: Start of the time range (ISO format)
            end_time: End of the time range (ISO format)
            
        Returns:
            List of LearningWindow objects
        """
        query = """
        FOR w IN LearningWindow
        FILTER w.start_time >= @start_time AND w.end_time <= @end_time
        SORT w.start_time
        RETURN w
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "start_time": start_time,
            "end_time": end_time
        })
        
        windows = []
        for doc in cursor:
            windows.append(self._dict_to_entity(doc))
        
        return windows
    
    def find_overlapping_windows(self, window: LearningWindow) -> List[LearningWindow]:
        """
        Find learning windows that overlap with the given window.
        
        This is useful for window coordination and field awareness.
        
        Args:
            window: LearningWindow to find overlaps for
            
        Returns:
            List of overlapping LearningWindow objects
        """
        query = """
        FOR w IN LearningWindow
        FILTER w.window_id != @window_id
        FILTER (
            (w.start_time <= @start_time AND w.end_time >= @start_time) OR
            (w.start_time <= @end_time AND w.end_time >= @end_time) OR
            (w.start_time >= @start_time AND w.end_time <= @end_time)
        )
        SORT w.start_time
        RETURN w
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "window_id": window.window_id,
            "start_time": window.start_time,
            "end_time": window.end_time
        })
        
        windows = []
        for doc in cursor:
            windows.append(self._dict_to_entity(doc))
        
        return windows
    
    def get_window_patterns(self, window_id: str) -> List[Dict[str, Any]]:
        """
        Get all patterns detected in a learning window.
        
        Args:
            window_id: ID of the learning window
            
        Returns:
            List of pattern dictionaries
        """
        query = """
        FOR v, e IN 1..1 OUTBOUND @window_id WindowContainsPattern
        SORT v.detected_at
        RETURN v
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "window_id": f"{self.collection_name}/{window_id}"
        })
        
        return list(cursor)
    
    def get_pattern_evolution(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """
        Get pattern evolution within a time range.
        
        This method uses the specialized PATTERN_EVOLUTION traversal function
        to retrieve how patterns evolve over time.
        
        Args:
            start_time: Start of the time range (ISO format)
            end_time: End of the time range (ISO format)
            
        Returns:
            List of pattern evolution chains
        """
        # Use the specialized PATTERN_EVOLUTION traversal function
        query = """
        LET traversal_function = DOCUMENT("AQLTraversals/PATTERN_EVOLUTION")
        LET evolutions = EVAL(traversal_function.code)(@start_time, @end_time)
        
        RETURN evolutions
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "start_time": start_time,
            "end_time": end_time
        })
        
        return list(cursor)
    
    def calculate_narrative_coherence(self, window_id: str) -> float:
        """
        Calculate narrative coherence for a learning window.
        
        Narrative coherence measures how well the patterns in a window
        form a coherent story or narrative structure.
        
        Args:
            window_id: ID of the learning window
            
        Returns:
            Narrative coherence score (0.0 to 1.0)
        """
        # Get patterns in the window
        patterns = self.get_window_patterns(window_id)
        
        if not patterns:
            return 0.0
        
        # Calculate average stability as a simple coherence metric
        total_stability = sum(p.get("stability", 0.0) for p in patterns)
        avg_stability = total_stability / len(patterns)
        
        # Get actant journeys that intersect with this window
        query = """
        FOR p IN TemporalPattern
        FILTER p.window_id == @window_id
        
        // Find actants involved in these patterns
        LET actants = (
            FOR actant IN p.actants || []
            RETURN actant
        )
        
        // Get journeys for these actants
        LET journeys = (
            FOR actant IN actants
            FOR j IN ActantJourney
            FILTER j.actant_name == actant
            RETURN j
        )
        
        RETURN UNIQUE(journeys)
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "window_id": window_id
        })
        
        journeys = list(cursor)
        
        # Calculate coherence based on journey continuity
        journey_factor = min(1.0, len(journeys) / 5.0)  # Normalize to 0.0-1.0
        
        # Combine metrics for overall coherence
        coherence = (avg_stability * 0.7) + (journey_factor * 0.3)
        
        return coherence
    
    def calculate_character_development(self, window_id: str) -> Dict[str, float]:
        """
        Calculate character development metrics for a learning window.
        
        Character development measures how actants (characters) evolve
        within the window, particularly through role shifts and transitions.
        
        Args:
            window_id: ID of the learning window
            
        Returns:
            Dictionary mapping actant names to development scores
        """
        # Get the window
        window = self.find_by_window_id(window_id)
        if not window:
            return {}
        
        # Get actants involved in this window
        query = """
        FOR p IN TemporalPattern
        FILTER p.window_id == @window_id
        
        // Find actants involved in these patterns
        LET pattern_actants = (
            FOR actant IN p.actants || []
            RETURN actant
        )
        
        // Combine all unique actants
        RETURN UNIQUE(pattern_actants)
        """
        
        cursor = self.db.aql.execute(query, bind_vars={
            "window_id": window_id
        })
        
        actants = list(cursor)
        
        # Calculate development metrics for each actant
        development_scores = {}
        
        for actant in actants:
            # Count role shifts within this window's time range
            query = """
            FOR rs IN RoleShift
            FILTER rs.actant_name == @actant_name
            FILTER rs.timestamp >= @start_time AND rs.timestamp <= @end_time
            RETURN rs
            """
            
            cursor = self.db.aql.execute(query, bind_vars={
                "actant_name": actant,
                "start_time": window.start_time,
                "end_time": window.end_time
            })
            
            role_shifts = list(cursor)
            
            # Count domain transitions within this window's time range
            query = """
            FOR dt IN DomainTransition
            FILTER dt.actant_name == @actant_name
            FILTER dt.timestamp >= @start_time AND dt.timestamp <= @end_time
            RETURN dt
            """
            
            cursor = self.db.aql.execute(query, bind_vars={
                "actant_name": actant,
                "start_time": window.start_time,
                "end_time": window.end_time
            })
            
            domain_transitions = list(cursor)
            
            # Calculate development score based on role shifts and domain transitions
            role_shift_factor = min(1.0, len(role_shifts) / 3.0)  # Normalize to 0.0-1.0
            transition_factor = min(1.0, len(domain_transitions) / 5.0)  # Normalize to 0.0-1.0
            
            # Combine metrics for overall development score
            development_score = (role_shift_factor * 0.6) + (transition_factor * 0.4)
            
            development_scores[actant] = development_score
        
        return development_scores
