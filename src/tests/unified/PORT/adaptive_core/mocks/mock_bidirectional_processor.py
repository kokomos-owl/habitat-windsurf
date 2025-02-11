"""
Mock bidirectional processor for testing purposes.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from .mock_coherence import CoherenceChecker, CoherenceThresholds
from .mock_observable import MockObservable
from .mock_adaptive_id import MockAdaptiveID

class MockBidirectionalProcessor:
    """Mock bidirectional processor for testing."""
    
    def __init__(self):
        """Initialize processor with mock components."""
        self.coherence_checker = CoherenceChecker()
        self.observable = MockObservable()
        self.processing_history = []
        
    async def process_structure(self, data: Dict[str, Any], adaptive_id: Optional[MockAdaptiveID] = None) -> Dict[str, Any]:
        """Process structural aspects of data."""
        result = {
            "structure_score": 0.8,
            "structure_analysis": {
                "completeness": 0.85,
                "consistency": 0.75,
                "validity": 0.80
            }
        }
        
        # Update adaptive context
        if adaptive_id:
            adaptive_id.update_state({
                "type": "structure",
                "data": data,
                "result": result
            })
        
        # Track processing
        self.processing_history.append({
            "type": "structure",
            "data": data,
            "result": result
        })
        
        return result
        
    async def process_meaning(self, data: Dict[str, Any], adaptive_id: Optional[MockAdaptiveID] = None) -> Dict[str, Any]:
        """Process semantic meaning of data."""
        result = {
            "meaning_score": 0.85,
            "semantic_analysis": {
                "relevance": 0.9,
                "coherence": 0.8,
                "context_alignment": 0.85
            }
        }
        
        # Update adaptive context
        if adaptive_id:
            adaptive_id.update_state({
                "type": "meaning",
                "data": data,
                "result": result
            })
        
        # Track processing
        self.processing_history.append({
            "type": "meaning",
            "data": data,
            "result": result
        })
        
        return result
        
    async def process_bidirectional(self, data: Dict[str, Any], adaptive_id: Optional[MockAdaptiveID] = None) -> Dict[str, Any]:
        """Process both structure and meaning with bidirectional influence."""
        structure_result = await self.process_structure(data, adaptive_id)
        meaning_result = await self.process_meaning(data, adaptive_id)
        
        # Combine results
        combined_result = {
            "structure": structure_result,
            "meaning": meaning_result,
            "overall_score": (structure_result["structure_score"] + meaning_result["meaning_score"]) / 2
        }
        
        # Update visualization
        self.observable.update_cell("bidirectional_result", combined_result)
        
        return combined_result
        
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get history of processing operations."""
        return self.processing_history
