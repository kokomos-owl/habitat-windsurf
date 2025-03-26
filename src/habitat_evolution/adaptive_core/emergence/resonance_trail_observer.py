"""
Resonance Trail Observer

This module implements the ResonanceTrailObserver class, which observes the natural
formation of resonance trails in semantic space as patterns move and evolve.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import uuid
import logging
import math
import numpy as np

from ..id.adaptive_id import AdaptiveID
from ...field.field_state import TonicHarmonicFieldState


class ResonanceTrailObserver:
    """
    Observes the natural formation of resonance trails in semantic space.
    
    This class tracks how patterns move through semantic space, allowing trails
    to form naturally without imposing predefined structures. It enables the
    system to detect and leverage emergent pathways in the semantic landscape.
    """
    
    def __init__(self, field_state: TonicHarmonicFieldState, persistence_factor: float = 0.9):
        """
        Initialize a resonance trail observer.
        
        Args:
            field_state: The field state to observe
            persistence_factor: Factor controlling trail persistence (0-1)
        """
        self.field_state = field_state
        self.persistence_factor = persistence_factor
        self.observed_movements = []
        self.trail_map = {}  # Maps trail keys to trail data
        self.trail_history = []
        
        # Create an AdaptiveID for this observer
        self.adaptive_id = AdaptiveID(
            base_concept="resonance_trail_observer",
            creator_id="system"
        )
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def observe_pattern_movement(self, pattern_id: str, old_position: Tuple, 
                                new_position: Tuple, timestamp: str = None) -> Dict[str, Any]:
        """
        Record pattern movement without imposing structure.
        
        Args:
            pattern_id: ID of the pattern that moved
            old_position: Previous position in semantic space
            new_position: New position in semantic space
            timestamp: Timestamp of the movement (defaults to now)
            
        Returns:
            Movement data
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Create movement record
        movement = {
            "id": str(uuid.uuid4()),
            "pattern_id": pattern_id,
            "old_position": old_position,
            "new_position": new_position,
            "timestamp": timestamp,
            "distance": self._calculate_distance(old_position, new_position)
        }
        
        # Add to observed movements
        self.observed_movements.append(movement)
        
        # Create trail key
        trail_key = self._get_trail_key(old_position, new_position)
        
        # Update trail map
        if trail_key not in self.trail_map:
            self.trail_map[trail_key] = {
                "intensity": 0.0,
                "created_at": timestamp,
                "last_updated": timestamp,
                "pattern_ids": set(),
                "movements": []
            }
        
        self.trail_map[trail_key]["intensity"] += 1.0
        self.trail_map[trail_key]["pattern_ids"].add(pattern_id)
        self.trail_map[trail_key]["last_updated"] = timestamp
        self.trail_map[trail_key]["movements"].append(movement["id"])
        
        # Add to trail history
        self.trail_history.append({
            "timestamp": timestamp,
            "pattern_id": pattern_id,
            "old_position": old_position,
            "new_position": new_position,
            "trail_key": trail_key
        })
        
        # Update the AdaptiveID with this movement
        movement_key = f"pattern_movement_{pattern_id}_{timestamp}"
        movement_data = {
            "pattern_id": pattern_id,
            "old_position": str(old_position),
            "new_position": str(new_position),
            "timestamp": timestamp,
            "trail_key": trail_key
        }
        self.adaptive_id.update_temporal_context(movement_key, movement_data, "resonance_observation")
        
        # Register with field state to participate in field analysis
        if hasattr(self.field_state, 'register_observer'):
            self.adaptive_id.register_with_field_observer(self.field_state)
        
        return movement
    
    def _get_trail_key(self, pos1: Tuple, pos2: Tuple) -> str:
        """
        Generate a unique key for a trail between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Trail key
        """
        # Sort positions to ensure consistent keys regardless of direction
        sorted_positions = sorted([str(pos1), str(pos2)])
        return f"{sorted_positions[0]}_{sorted_positions[1]}"
    
    def _calculate_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """
        Calculate distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        # Convert to numpy arrays for easier calculation
        p1 = np.array(pos1)
        p2 = np.array(pos2)
        
        # Calculate Euclidean distance
        return np.linalg.norm(p2 - p1)
    
    def decay_trails(self) -> None:
        """Apply decay to all trails based on persistence factor."""
        current_time = datetime.now()
        
        for trail_key, trail_data in self.trail_map.items():
            # Calculate time-based decay
            last_updated = datetime.fromisoformat(trail_data["last_updated"])
            hours_since = (current_time - last_updated).total_seconds() / 3600
            
            # Apply decay
            decay_factor = self.persistence_factor ** hours_since
            trail_data["intensity"] *= decay_factor
            
            # Update AdaptiveID with decay information
            decay_key = f"trail_decay_{trail_key}_{current_time.isoformat()}"
            decay_data = {
                "trail_key": trail_key,
                "old_intensity": trail_data["intensity"] / decay_factor,
                "new_intensity": trail_data["intensity"],
                "decay_factor": decay_factor,
                "hours_since_update": hours_since,
                "timestamp": current_time.isoformat()
            }
            self.adaptive_id.update_temporal_context(decay_key, decay_data, "trail_decay")
    
    def get_trail_influence(self, position: Tuple) -> Dict[str, Any]:
        """
        Get the influence of nearby trails on a position.
        
        Args:
            position: Position to check
            
        Returns:
            Trail influence data
        """
        influence_data = {
            "position": position,
            "timestamp": datetime.now().isoformat(),
            "total_influence": 0.0,
            "contributing_trails": []
        }
        
        for trail_key, trail_data in self.trail_map.items():
            # Extract positions from trail key
            trail_positions = trail_key.split("_")
            pos1 = eval(trail_positions[0])  # Convert string to tuple
            pos2 = eval(trail_positions[1])
            
            # Calculate distance to trail (point-to-line distance)
            distance = self._point_to_line_distance(position, pos1, pos2)
            
            # Apply distance-based decay
            proximity_factor = math.exp(-distance * 2)
            
            # Apply time-based decay
            current_time = datetime.now()
            last_updated = datetime.fromisoformat(trail_data["last_updated"])
            hours_since = (current_time - last_updated).total_seconds() / 3600
            time_factor = self.persistence_factor ** hours_since
            
            # Calculate influence
            trail_influence = trail_data["intensity"] * proximity_factor * time_factor
            
            # Add to total if significant
            if trail_influence > 0.01:
                influence_data["total_influence"] += trail_influence
                influence_data["contributing_trails"].append({
                    "trail_key": trail_key,
                    "influence": trail_influence,
                    "distance": distance,
                    "intensity": trail_data["intensity"],
                    "proximity_factor": proximity_factor,
                    "time_factor": time_factor
                })
        
        # Sort contributing trails by influence
        influence_data["contributing_trails"].sort(key=lambda x: x["influence"], reverse=True)
        
        return influence_data
    
    def _point_to_line_distance(self, point: Tuple, line_start: Tuple, line_end: Tuple) -> float:
        """
        Calculate the distance from a point to a line segment.
        
        Args:
            point: The point
            line_start: Start of the line segment
            line_end: End of the line segment
            
        Returns:
            Distance from point to line segment
        """
        # Convert to numpy arrays
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)
        
        # Handle case where line_start and line_end are the same point
        if np.array_equal(a, b):
            return np.linalg.norm(p - a)
        
        # Calculate projection
        line_vec = b - a
        point_vec = p - a
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        projection = np.dot(point_vec, line_unitvec)
        
        # Handle cases where projection is outside the line segment
        if projection <= 0:
            return np.linalg.norm(p - a)
        elif projection >= line_len:
            return np.linalg.norm(p - b)
        else:
            # Calculate perpendicular distance
            projection_vec = line_unitvec * projection
            closest_point = a + projection_vec
            return np.linalg.norm(p - closest_point)
    
    def detect_emergent_pathways(self, threshold: float = 1.5) -> List[Dict[str, Any]]:
        """
        Detect emergent pathways from trail patterns.
        
        Args:
            threshold: Minimum intensity threshold for pathway detection
            
        Returns:
            List of detected pathways
        """
        pathways = []
        
        # Decay trails first to ensure current intensities
        self.decay_trails()
        
        # Find connected trails that form pathways
        visited_trails = set()
        
        for trail_key, trail_data in self.trail_map.items():
            # Skip if already part of a pathway or below threshold
            if trail_key in visited_trails or trail_data["intensity"] < threshold:
                continue
            
            # Start a new pathway
            pathway = {
                "id": f"pathway_{len(pathways)}",
                "trails": [trail_key],
                "intensity": trail_data["intensity"],
                "pattern_ids": set(trail_data["pattern_ids"]),
                "created_at": trail_data["created_at"],
                "last_updated": trail_data["last_updated"]
            }
            
            # Mark as visited
            visited_trails.add(trail_key)
            
            # Extract positions
            trail_positions = trail_key.split("_")
            pos1 = eval(trail_positions[0])
            pos2 = eval(trail_positions[1])
            
            # Find connected trails
            self._extend_pathway(pathway, pos1, pos2, visited_trails, threshold)
            
            # Only add if pathway has multiple trails
            if len(pathway["trails"]) > 1:
                # Calculate average intensity
                pathway["average_intensity"] = sum(self.trail_map[t]["intensity"] for t in pathway["trails"]) / len(pathway["trails"])
                
                # Update the AdaptiveID with this pathway
                self.adaptive_id.update_context({
                    "pathway_detected": pathway["id"],
                    "pathway_data": {
                        "id": pathway["id"],
                        "trail_count": len(pathway["trails"]),
                        "pattern_count": len(pathway["pattern_ids"]),
                        "average_intensity": pathway["average_intensity"],
                        "timestamp": datetime.now().isoformat()
                    },
                    "timestamp": datetime.now().isoformat()
                })
                
                pathways.append(pathway)
        
        return pathways
    
    def _extend_pathway(self, pathway: Dict[str, Any], pos1: Tuple, pos2: Tuple, 
                       visited_trails: Set[str], threshold: float) -> None:
        """
        Recursively extend a pathway by finding connected trails.
        
        Args:
            pathway: The pathway to extend
            pos1: First endpoint of the current trail
            pos2: Second endpoint of the current trail
            visited_trails: Set of already visited trails
            threshold: Minimum intensity threshold
        """
        # Check all trails for connections to pos1 or pos2
        for trail_key, trail_data in self.trail_map.items():
            # Skip if already visited or below threshold
            if trail_key in visited_trails or trail_data["intensity"] < threshold:
                continue
            
            # Extract positions
            trail_positions = trail_key.split("_")
            t_pos1 = eval(trail_positions[0])
            t_pos2 = eval(trail_positions[1])
            
            # Check for connections
            if self._positions_match(pos1, t_pos1) or self._positions_match(pos1, t_pos2) or \
               self._positions_match(pos2, t_pos1) or self._positions_match(pos2, t_pos2):
                
                # Add to pathway
                pathway["trails"].append(trail_key)
                pathway["intensity"] += trail_data["intensity"]
                pathway["pattern_ids"].update(trail_data["pattern_ids"])
                
                # Update timestamps if needed
                created_at = datetime.fromisoformat(trail_data["created_at"])
                pathway_created = datetime.fromisoformat(pathway["created_at"])
                if created_at < pathway_created:
                    pathway["created_at"] = trail_data["created_at"]
                
                last_updated = datetime.fromisoformat(trail_data["last_updated"])
                pathway_updated = datetime.fromisoformat(pathway["last_updated"])
                if last_updated > pathway_updated:
                    pathway["last_updated"] = trail_data["last_updated"]
                
                # Mark as visited
                visited_trails.add(trail_key)
                
                # Continue extending from the new endpoints
                if self._positions_match(pos1, t_pos1):
                    self._extend_pathway(pathway, pos2, t_pos2, visited_trails, threshold)
                elif self._positions_match(pos1, t_pos2):
                    self._extend_pathway(pathway, pos2, t_pos1, visited_trails, threshold)
                elif self._positions_match(pos2, t_pos1):
                    self._extend_pathway(pathway, pos1, t_pos2, visited_trails, threshold)
                elif self._positions_match(pos2, t_pos2):
                    self._extend_pathway(pathway, pos1, t_pos1, visited_trails, threshold)
    
    def _positions_match(self, pos1: Tuple, pos2: Tuple, tolerance: float = 1e-6) -> bool:
        """
        Check if two positions match within a tolerance.
        
        Args:
            pos1: First position
            pos2: Second position
            tolerance: Distance tolerance
            
        Returns:
            True if positions match, False otherwise
        """
        return self._calculate_distance(pos1, pos2) < tolerance
    
    def register_with_learning_window(self, learning_window) -> None:
        """
        Register this observer with a learning window.
        
        Args:
            learning_window: The learning window to register with
        """
        self.adaptive_id.register_with_learning_window(learning_window)
