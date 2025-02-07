"""Pattern evolution visualization hooks.

This module provides hooks for visualizing pattern evolution,
allowing us to track and replay pattern formation and state
transitions through both time and state-space.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime, timedelta
import json
from collections import defaultdict

@dataclass
class EvolutionFrame:
    """Captures a moment in pattern evolution."""
    timestamp: datetime
    patterns: Dict[str, Dict[str, Any]]
    relationships: Dict[str, List[Dict[str, Any]]]
    metrics: Dict[str, float]
    state_space: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to serializable dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'patterns': self.patterns,
            'relationships': self.relationships,
            'metrics': self.metrics,
            'state_space': self.state_space
        }

@dataclass
class EvolutionTimeline:
    """Maintains timeline of pattern evolution."""
    frames: List[EvolutionFrame] = field(default_factory=list)
    frame_index: Dict[datetime, int] = field(default_factory=dict)
    
    def add_frame(self, frame: EvolutionFrame) -> None:
        """Add a new evolution frame."""
        self.frames.append(frame)
        self.frame_index[frame.timestamp] = len(self.frames) - 1
    
    def get_frame(self, timestamp: datetime) -> Optional[EvolutionFrame]:
        """Get closest frame to timestamp."""
        if not self.frames:
            return None
            
        # Find closest timestamp
        closest = min(self.frame_index.keys(),
                     key=lambda x: abs(x - timestamp))
        return self.frames[self.frame_index[closest]]
    
    def get_frame_range(self, 
                       start: datetime,
                       end: datetime) -> Iterator[EvolutionFrame]:
        """Get frames within a time range."""
        for frame in self.frames:
            if start <= frame.timestamp <= end:
                yield frame

@dataclass
class EvolutionVisualizer:
    """Manages pattern evolution visualization."""
    timeline: EvolutionTimeline = field(default_factory=EvolutionTimeline)
    
    def capture_evolution_state(self,
                              patterns: Dict[str, Dict[str, Any]],
                              relationships: Dict[str, List[Dict[str, Any]]],
                              metrics: Dict[str, float],
                              state_space: Dict[str, Any]) -> None:
        """Capture current evolution state."""
        frame = EvolutionFrame(
            timestamp=datetime.utcnow(),
            patterns=patterns,
            relationships=relationships,
            metrics=metrics,
            state_space=state_space
        )
        self.timeline.add_frame(frame)
    
    def get_evolution_replay(self,
                           start: datetime,
                           end: datetime,
                           include_metrics: bool = True) -> List[Dict[str, Any]]:
        """Get evolution replay data for visualization."""
        frames = []
        for frame in self.timeline.get_frame_range(start, end):
            frame_data = {
                'timestamp': frame.timestamp.isoformat(),
                'patterns': frame.patterns,
                'relationships': frame.relationships
            }
            if include_metrics:
                frame_data.update({
                    'metrics': frame.metrics,
                    'state_space': frame.state_space
                })
            frames.append(frame_data)
        return frames
    
    def get_pattern_trajectory(self,
                             pattern_id: str,
                             start: datetime,
                             end: datetime) -> List[Dict[str, Any]]:
        """Get evolution trajectory for a specific pattern."""
        trajectory = []
        for frame in self.timeline.get_frame_range(start, end):
            if pattern_id in frame.patterns:
                state = frame.patterns[pattern_id]
                relationships = frame.relationships.get(pattern_id, [])
                
                trajectory.append({
                    'timestamp': frame.timestamp.isoformat(),
                    'state': state,
                    'relationships': relationships,
                    'metrics': {
                        k: v for k, v in frame.metrics.items()
                        if k.startswith(f'pattern_{pattern_id}')
                    }
                })
        return trajectory
    
    def export_timeline(self, filepath: str) -> None:
        """Export evolution timeline to file."""
        with open(filepath, 'w') as f:
            json.dump({
                'frames': [
                    frame.to_dict() for frame in self.timeline.frames
                ]
            }, f, indent=2)
    
    def import_timeline(self, filepath: str) -> None:
        """Import evolution timeline from file."""
        with open(filepath) as f:
            data = json.load(f)
            for frame_data in data['frames']:
                frame = EvolutionFrame(
                    timestamp=datetime.fromisoformat(frame_data['timestamp']),
                    patterns=frame_data['patterns'],
                    relationships=frame_data['relationships'],
                    metrics=frame_data['metrics'],
                    state_space=frame_data['state_space']
                )
                self.timeline.add_frame(frame)
