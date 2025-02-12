"""
Document processing adapter for pattern extraction.
"""
from typing import Dict, Any, List
import numpy as np
from dataclasses import dataclass

from ...core.pattern.types import Pattern, FieldState, FieldGradients

@dataclass
class DocumentFeatures:
    """Features extracted from document."""
    coherence: float
    energy: float
    density: float
    turbulence: float
    patterns: List[Dict[str, float]]

class DocumentProcessor:
    """Processes documents for pattern extraction."""
    
    def __init__(self, coherence_threshold: float = 0.3):
        self.coherence_threshold = coherence_threshold
    
    def process_document(self, content: str) -> DocumentFeatures:
        """Process document content for features.
        
        Args:
            content: Document content
            
        Returns:
            Extracted features
        """
        # Extract basic metrics
        words = content.split()
        sentences = content.split('.')
        
        # Calculate coherence from sentence structure
        coherence = self._calculate_coherence(sentences)
        
        # Calculate energy from word dynamics
        energy = self._calculate_energy(words)
        
        # Calculate density
        density = len(words) / max(len(sentences), 1)
        
        # Calculate turbulence from variance
        turbulence = self._calculate_turbulence(sentences)
        
        # Extract potential patterns
        patterns = self._extract_patterns(sentences)
        
        return DocumentFeatures(
            coherence=coherence,
            energy=energy,
            density=density,
            turbulence=turbulence,
            patterns=patterns
        )
    
    def to_field_state(self, features: DocumentFeatures) -> FieldState:
        """Convert document features to field state.
        
        Args:
            features: Extracted features
            
        Returns:
            Field state representation
        """
        return FieldState(
            gradients=FieldGradients(
                coherence=features.coherence,
                energy=features.energy,
                density=features.density,
                turbulence=features.turbulence
            ),
            patterns=[],  # Patterns added separately
            timestamp=0.0  # Set by caller
        )
    
    def _calculate_coherence(self, sentences: List[str]) -> float:
        """Calculate coherence from sentence structure."""
        if not sentences:
            return 0.0
            
        # Use sentence length variance as coherence measure
        lengths = [len(s.split()) for s in sentences]
        variance = np.var(lengths) if len(lengths) > 1 else 0
        
        # Normalize variance to 0-1 range
        return max(0.0, min(1.0, 1.0 - (variance / 100.0)))
    
    def _calculate_energy(self, words: List[str]) -> float:
        """Calculate energy from word dynamics."""
        if not words:
            return 0.0
            
        # Use word length and frequency as energy measure
        avg_length = sum(len(w) for w in words) / len(words)
        unique_ratio = len(set(words)) / len(words)
        
        # Combine metrics
        return max(0.0, min(1.0, (avg_length / 10.0) * unique_ratio))
    
    def _calculate_turbulence(self, sentences: List[str]) -> float:
        """Calculate turbulence from sentence variance."""
        if not sentences:
            return 0.0
            
        # Use changes in sentence structure as turbulence
        structures = [len(s.split(',')) for s in sentences]
        variance = np.var(structures) if len(structures) > 1 else 0
        
        return max(0.0, min(1.0, variance / 10.0))
    
    def _extract_patterns(self, sentences: List[str]) -> List[Dict[str, float]]:
        """Extract potential patterns from sentences."""
        patterns = []
        
        for i, sentence in enumerate(sentences):
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Calculate sentence metrics
            words = sentence.split()
            if not words:
                continue
                
            coherence = self._calculate_coherence([sentence])
            energy = self._calculate_energy(words)
            
            # Only keep strong patterns
            if coherence >= self.coherence_threshold:
                patterns.append({
                    'coherence': coherence,
                    'energy': energy,
                    'position': i / len(sentences)
                })
        
        return patterns
