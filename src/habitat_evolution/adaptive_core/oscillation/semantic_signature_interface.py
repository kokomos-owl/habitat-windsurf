"""
Semantic Signature Interface

This module provides interfaces for translating oscillatory signatures into
human-understandable semantic concepts and narratives. It bridges the mathematical
representation of oscillatory properties with meaningful semantic interpretations
that users can understand and work with.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
import uuid
from datetime import datetime
import logging
import numpy as np
from collections import deque

from ..id.adaptive_id import AdaptiveID
from ..id.oscillatory_adaptive_id import OscillatoryAdaptiveID
from .oscillatory_signature import OscillatorySignature
from .signature_service import OscillatorySignatureService
from .adaptive_id_signature_service import AdaptiveIDSignatureService

logger = logging.getLogger(__name__)

class SignatureSemanticTranslator:
    """
    Translates oscillatory signatures into human-understandable semantic concepts.
    
    This class provides methods to convert the mathematical properties of oscillatory
    signatures (frequency, amplitude, phase, harmonics) into semantic concepts that
    humans can understand and work with, such as change rate, impact, alignment,
    and complexity.
    """
    
    @staticmethod
    def translate_signature_to_semantic_concepts(signature: Union[OscillatorySignature, OscillatoryAdaptiveID]) -> Dict[str, str]:
        """
        Translate oscillatory signature to human-understandable semantic concepts.
        
        Args:
            signature: An oscillatory signature or oscillatory adaptive ID
            
        Returns:
            Dictionary mapping semantic concept names to descriptions
        """
        semantic_concepts = {}
        
        # Extract oscillatory properties based on input type
        if isinstance(signature, OscillatorySignature):
            frequency = signature.fundamental_frequency
            amplitude = signature.fundamental_amplitude
            phase = signature.fundamental_phase
            harmonics = signature.harmonics
            energy = signature.energy_level
        elif isinstance(signature, OscillatoryAdaptiveID):
            frequency = signature.oscillatory_properties["fundamental_frequency"]
            amplitude = signature.oscillatory_properties["fundamental_amplitude"]
            phase = signature.oscillatory_properties["fundamental_phase"]
            harmonics = signature.oscillatory_properties["harmonics"]
            energy = signature.oscillatory_properties["energy_level"]
        else:
            raise TypeError("Input must be OscillatorySignature or OscillatoryAdaptiveID")
        
        # Translate frequency to concept of change rate
        if frequency < 0.1:
            semantic_concepts["change_rate"] = "very stable, changes slowly over long periods"
        elif frequency < 0.3:
            semantic_concepts["change_rate"] = "moderately stable, evolves gradually"
        elif frequency < 0.6:
            semantic_concepts["change_rate"] = "dynamic, changes regularly with context"
        else:
            semantic_concepts["change_rate"] = "highly volatile, changes rapidly and frequently"
        
        # Translate amplitude to concept of impact
        if amplitude < 0.3:
            semantic_concepts["impact"] = "subtle influence, operates in the background"
        elif amplitude < 0.7:
            semantic_concepts["impact"] = "moderate influence, noticeable but not dominant"
        else:
            semantic_concepts["impact"] = "strong influence, central to current context"
        
        # Translate phase to concept of alignment
        # We'll use a simplified approach based on where in the cycle the entity is
        cycle_position = (phase / (2 * np.pi)) * 100  # Convert to percentage through cycle
        
        if cycle_position < 25:
            semantic_concepts["cycle_position"] = "emerging, beginning to manifest"
        elif cycle_position < 50:
            semantic_concepts["cycle_position"] = "growing, increasing in relevance"
        elif cycle_position < 75:
            semantic_concepts["cycle_position"] = "peaking, at maximum relevance"
        else:
            semantic_concepts["cycle_position"] = "receding, decreasing in immediate relevance"
        
        # Translate harmonic pattern to concept of complexity
        if isinstance(harmonics, list):
            harmonic_count = len(harmonics)
        else:
            harmonic_count = 0
            
        if harmonic_count <= 1:
            semantic_concepts["complexity"] = "simple, straightforward concept"
        elif harmonic_count <= 3:
            semantic_concepts["complexity"] = "moderately complex, has several aspects"
        else:
            semantic_concepts["complexity"] = "complex, multifaceted concept with many dimensions"
        
        # Translate energy level to concept of activity
        if energy < 0.3:
            semantic_concepts["activity"] = "dormant, currently inactive or minimally active"
        elif energy < 0.7:
            semantic_concepts["activity"] = "active, currently engaged in the semantic field"
        else:
            semantic_concepts["activity"] = "highly active, intensely engaged in current context"
        
        # Determine overall semantic significance
        significance = (amplitude * 0.4) + (energy * 0.4) + (harmonic_count * 0.05) + (frequency * 0.15)
        
        if significance < 0.3:
            semantic_concepts["significance"] = "peripheral, not central to current understanding"
        elif significance < 0.6:
            semantic_concepts["significance"] = "relevant, contributes to current understanding"
        else:
            semantic_concepts["significance"] = "essential, critical to current understanding"
        
        return semantic_concepts

    @staticmethod
    def translate_resonance_to_semantic_concept(resonance: float) -> str:
        """
        Translate resonance value to a semantic concept.
        
        Args:
            resonance: Resonance value between 0 and 1
            
        Returns:
            Semantic description of the resonance
        """
        if resonance < 0.3:
            return "weak connection, minimal semantic relationship"
        elif resonance < 0.5:
            return "modest connection, some semantic overlap"
        elif resonance < 0.7:
            return "moderate connection, clear semantic relationship"
        elif resonance < 0.9:
            return "strong connection, significant semantic relationship"
        else:
            return "profound connection, deeply intertwined concepts"
    
    @staticmethod
    def translate_interference_to_semantic_concept(interference_value: float, interference_type: str) -> str:
        """
        Translate interference pattern to a semantic concept.
        
        Args:
            interference_value: Interference value between -1 and 1
            interference_type: Type of interference (constructive, destructive, neutral)
            
        Returns:
            Semantic description of the interference
        """
        if interference_type == "constructive" and interference_value > 0.7:
            return "mutually reinforcing, concepts amplify each other"
        elif interference_type == "constructive":
            return "complementary, concepts support each other"
        elif interference_type == "destructive" and interference_value < -0.7:
            return "conflicting, concepts contradict or undermine each other"
        elif interference_type == "destructive":
            return "competing, concepts partially contradict each other"
        else:
            return "independent, concepts coexist without strong interaction"


class SignatureNarrativeGenerator:
    """
    Generates narrative explanations of oscillatory signatures.
    
    This class provides methods to create human-readable narratives that explain
    the behavior, relationships, and predicted evolution of entities based on
    their oscillatory signatures.
    """
    
    def __init__(self, 
                signature_service: Optional[Union[OscillatorySignatureService, AdaptiveIDSignatureService]] = None,
                translator: Optional[SignatureSemanticTranslator] = None):
        """
        Initialize the narrative generator.
        
        Args:
            signature_service: Service for accessing signatures
            translator: Translator for converting signatures to semantic concepts
        """
        self.signature_service = signature_service
        self.translator = translator or SignatureSemanticTranslator()
    
    def generate_signature_narrative(self, 
                                   signature: Union[OscillatorySignature, OscillatoryAdaptiveID], 
                                   entity_name: str) -> str:
        """
        Generate a narrative explanation of an entity's signature.
        
        Args:
            signature: The entity's oscillatory signature
            entity_name: Name of the entity
            
        Returns:
            Narrative explanation of the entity's behavior
        """
        # Translate signature to semantic concepts
        concepts = self.translator.translate_signature_to_semantic_concepts(signature)
        
        # Generate basic narrative
        narrative = f"The concept of '{entity_name}' is currently {concepts['change_rate']}. "
        narrative += f"It has {concepts['impact']} in the current context and is {concepts['complexity']}. "
        narrative += f"It is currently {concepts['cycle_position']} and {concepts['activity']}. "
        narrative += f"Overall, it is {concepts['significance']}.\n\n"
        
        # Add information about relationships with other concepts if available
        if hasattr(signature, 'related_signatures') and signature.related_signatures:
            narrative += "Relationships with other concepts:\n"
            for related_name, resonance in signature.related_signatures:
                resonance_description = self.translator.translate_resonance_to_semantic_concept(resonance)
                narrative += f"- {related_name}: {resonance_description}\n"
            narrative += "\n"
        
        # Add predictive insights based on oscillatory pattern
        narrative += "Based on its oscillatory pattern:\n"
        
        # Extract oscillatory properties based on input type
        if isinstance(signature, OscillatorySignature):
            frequency = signature.fundamental_frequency
            amplitude = signature.fundamental_amplitude
            phase = signature.fundamental_phase
            energy = signature.energy_level
        elif isinstance(signature, OscillatoryAdaptiveID):
            frequency = signature.oscillatory_properties["fundamental_frequency"]
            amplitude = signature.oscillatory_properties["fundamental_amplitude"]
            phase = signature.oscillatory_properties["fundamental_phase"]
            energy = signature.oscillatory_properties["energy_level"]
        
        # Generate predictions based on oscillatory properties
        if frequency > 0.5 and amplitude > 0.7:
            narrative += "- This concept is likely to become increasingly important in the near future.\n"
        elif frequency > 0.5 and amplitude < 0.3:
            narrative += "- This concept may briefly emerge but is unlikely to sustain importance.\n"
        elif frequency < 0.2 and amplitude > 0.7:
            narrative += "- This concept represents a slow-moving but powerful trend with lasting importance.\n"
        
        if phase < np.pi/2:
            narrative += "- This concept is in its early stages and will continue to develop.\n"
        elif phase < np.pi:
            narrative += "- This concept is approaching its peak influence.\n"
        elif phase < 3*np.pi/2:
            narrative += "- This concept is beginning to decline in immediate relevance.\n"
        else:
            narrative += "- This concept is completing its current cycle and may soon begin a new phase.\n"
        
        if energy < 0.3:
            narrative += "- This concept may become less relevant unless new related information emerges.\n"
        elif energy > 0.8:
            narrative += "- This concept is highly energized and likely to remain active.\n"
        
        return narrative
    
    def generate_relationship_narrative(self, 
                                      signature1: Union[OscillatorySignature, OscillatoryAdaptiveID],
                                      signature2: Union[OscillatorySignature, OscillatoryAdaptiveID],
                                      entity_name1: str,
                                      entity_name2: str) -> str:
        """
        Generate a narrative explanation of the relationship between two entities.
        
        Args:
            signature1: First entity's oscillatory signature
            signature2: Second entity's oscillatory signature
            entity_name1: Name of the first entity
            entity_name2: Name of the second entity
            
        Returns:
            Narrative explanation of the relationship
        """
        # Calculate resonance and interference
        if isinstance(signature1, OscillatorySignature) and isinstance(signature2, OscillatorySignature):
            resonance = signature1.calculate_resonance(signature2)
            interference_value, interference_type = signature1.calculate_interference(signature2)
        elif isinstance(signature1, OscillatoryAdaptiveID) and isinstance(signature2, OscillatoryAdaptiveID):
            resonance = signature1.calculate_resonance(signature2)
            interference_value, interference_type = signature1.calculate_interference(signature2)
        else:
            raise TypeError("Both signatures must be of the same type")
        
        # Translate to semantic concepts
        resonance_description = self.translator.translate_resonance_to_semantic_concept(resonance)
        interference_description = self.translator.translate_interference_to_semantic_concept(
            interference_value, interference_type)
        
        # Generate narrative
        narrative = f"Relationship between '{entity_name1}' and '{entity_name2}':\n\n"
        narrative += f"These concepts have a {resonance_description}. "
        narrative += f"Their interaction is {interference_description}.\n\n"
        
        # Add more detailed analysis
        if resonance > 0.7 and interference_value > 0.7:
            narrative += "These concepts strongly reinforce each other, creating a synergistic relationship. "
            narrative += "When one becomes more relevant, the other is likely to follow. "
            narrative += "They form a coherent semantic cluster that should be considered together.\n"
        elif resonance > 0.7 and interference_value < -0.7:
            narrative += "These concepts are closely related but in tension with each other. "
            narrative += "They represent opposing perspectives on a shared domain. "
            narrative += "This tension may be productive, generating new insights through dialectical interaction.\n"
        elif resonance < 0.3 and interference_value > 0.7:
            narrative += "These concepts are distinct but complementary. "
            narrative += "Despite operating in different semantic domains, they support each other. "
            narrative += "This suggests an unexpected connection that may yield novel insights.\n"
        elif resonance < 0.3 and interference_value < -0.7:
            narrative += "These concepts are distinct and conflicting. "
            narrative += "They represent fundamentally different perspectives that are difficult to reconcile. "
            narrative += "This conflict may indicate a deeper conceptual divide that needs resolution.\n"
        
        # Extract oscillatory properties
        if isinstance(signature1, OscillatorySignature):
            freq1 = signature1.fundamental_frequency
            freq2 = signature2.fundamental_frequency
            phase1 = signature1.fundamental_phase
            phase2 = signature2.fundamental_phase
        else:
            freq1 = signature1.oscillatory_properties["fundamental_frequency"]
            freq2 = signature2.oscillatory_properties["fundamental_frequency"]
            phase1 = signature1.oscillatory_properties["fundamental_phase"]
            phase2 = signature2.oscillatory_properties["fundamental_phase"]
        
        # Add temporal relationship analysis
        freq_ratio = min(freq1, freq2) / max(freq1, freq2) if max(freq1, freq2) > 0 else 0
        phase_diff = abs((phase1 - phase2) % (2 * np.pi))
        
        narrative += "\nTemporal relationship:\n"
        
        if abs(freq_ratio - 1.0) < 0.1:
            narrative += "- These concepts operate on similar timescales, evolving in parallel.\n"
        elif freq_ratio < 0.5:
            narrative += "- These concepts operate on different timescales. "
            if freq1 < freq2:
                narrative += f"'{entity_name1}' represents a slower, more enduring pattern, while "
                narrative += f"'{entity_name2}' represents a faster, more transient pattern.\n"
            else:
                narrative += f"'{entity_name2}' represents a slower, more enduring pattern, while "
                narrative += f"'{entity_name1}' represents a faster, more transient pattern.\n"
        
        if phase_diff < np.pi/4 or phase_diff > 7*np.pi/4:
            narrative += "- These concepts are currently in phase, reinforcing each other.\n"
        elif phase_diff > 3*np.pi/4 and phase_diff < 5*np.pi/4:
            narrative += "- These concepts are currently out of phase, potentially counteracting each other.\n"
        elif phase_diff < np.pi:
            narrative += f"- '{entity_name1}' is leading '{entity_name2}' in their cycle, "
            narrative += "suggesting it may influence or precede the other.\n"
        else:
            narrative += f"- '{entity_name2}' is leading '{entity_name1}' in their cycle, "
            narrative += "suggesting it may influence or precede the other.\n"
        
        return narrative
    
    def generate_prediction_narrative(self, 
                                    prediction: Dict[str, Any], 
                                    entity_name: str) -> str:
        """
        Generate a narrative explanation of a predicted future state.
        
        Args:
            prediction: Dictionary with prediction data
            entity_name: Name of the entity
            
        Returns:
            Narrative explanation of the predicted future
        """
        narrative = f"Predicted future for '{entity_name}':\n\n"
        
        # Extract current and predicted states
        current = prediction["current_state"]
        predicted = prediction["predicted_state"]
        time_delta = prediction["time_delta"]
        
        # Determine time frame description
        if time_delta < 5:
            time_frame = "near future"
        elif time_delta < 20:
            time_frame = "medium term"
        else:
            time_frame = "long term"
        
        # Calculate changes
        freq_change = predicted["fundamental_frequency"] - current["fundamental_frequency"]
        amp_change = predicted["fundamental_amplitude"] - current["fundamental_amplitude"]
        energy_change = predicted["energy_level"] - current["energy_level"]
        
        # Generate narrative based on changes
        narrative += f"In the {time_frame}, this concept is expected to undergo the following changes:\n"
        
        if abs(freq_change) < 0.05:
            narrative += "- Its rate of change will remain relatively stable.\n"
        elif freq_change > 0:
            narrative += "- It will begin to change more rapidly, becoming more dynamic.\n"
        else:
            narrative += "- It will begin to change more slowly, becoming more stable.\n"
        
        if abs(amp_change) < 0.05:
            narrative += "- Its impact and influence will remain at similar levels.\n"
        elif amp_change > 0:
            narrative += "- Its impact and influence will increase, becoming more prominent.\n"
        else:
            narrative += "- Its impact and influence will decrease, becoming less prominent.\n"
        
        if abs(energy_change) < 0.05:
            narrative += "- Its activity level will remain relatively unchanged.\n"
        elif energy_change > 0:
            narrative += "- Its activity level will increase, becoming more engaged in the semantic field.\n"
        else:
            narrative += "- Its activity level will decrease, becoming less active in the semantic field.\n"
        
        # Overall prediction
        overall_change = (abs(freq_change) + abs(amp_change) + abs(energy_change)) / 3
        
        narrative += "\nOverall prediction:\n"
        
        if overall_change < 0.1:
            narrative += "This concept is expected to remain relatively stable in the coming period.\n"
        elif overall_change < 0.3:
            narrative += "This concept is expected to undergo moderate changes in the coming period.\n"
        else:
            narrative += "This concept is expected to undergo significant changes in the coming period.\n"
        
        if amp_change > 0.2 and energy_change > 0.2:
            narrative += "It is likely to become increasingly important and central to understanding.\n"
        elif amp_change < -0.2 and energy_change < -0.2:
            narrative += "It is likely to become less important and may fade from immediate relevance.\n"
        
        return narrative


class SemanticSignatureInterface:
    """
    Interface for querying oscillatory signatures in semantic terms.
    
    This class provides methods for users to interact with oscillatory signatures
    using natural language queries and receive semantic explanations of entity
    behavior, relationships, and predictions.
    """
    
    def __init__(self, 
                signature_service: Union[OscillatorySignatureService, AdaptiveIDSignatureService],
                narrative_generator: Optional[SignatureNarrativeGenerator] = None):
        """
        Initialize the semantic interface.
        
        Args:
            signature_service: Service for accessing signatures
            narrative_generator: Generator for creating narratives
        """
        self.signature_service = signature_service
        self.narrative_generator = narrative_generator or SignatureNarrativeGenerator(signature_service)
    
    def explain_concept_behavior(self, concept_name: str) -> str:
        """
        Explain how a concept is behaving in semantic terms.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            Semantic explanation of the concept's behavior
        """
        # Get the signature
        if isinstance(self.signature_service, OscillatorySignatureService):
            signature = self.signature_service.get_signature(concept_name)
        else:  # AdaptiveIDSignatureService
            signature = self.signature_service.get_oscillatory_adaptive_id(concept_name)
        
        if not signature:
            return f"The concept '{concept_name}' is not currently active in the system."
        
        # Generate narrative
        return self.narrative_generator.generate_signature_narrative(signature, concept_name)
    
    def explain_concept_relationship(self, concept_name1: str, concept_name2: str) -> str:
        """
        Explain the relationship between two concepts in semantic terms.
        
        Args:
            concept_name1: Name of the first concept
            concept_name2: Name of the second concept
            
        Returns:
            Semantic explanation of the relationship
        """
        # Get the signatures
        if isinstance(self.signature_service, OscillatorySignatureService):
            signature1 = self.signature_service.get_signature(concept_name1)
            signature2 = self.signature_service.get_signature(concept_name2)
        else:  # AdaptiveIDSignatureService
            signature1 = self.signature_service.get_oscillatory_adaptive_id(concept_name1)
            signature2 = self.signature_service.get_oscillatory_adaptive_id(concept_name2)
        
        if not signature1:
            return f"The concept '{concept_name1}' is not currently active in the system."
        
        if not signature2:
            return f"The concept '{concept_name2}' is not currently active in the system."
        
        # Generate relationship narrative
        return self.narrative_generator.generate_relationship_narrative(
            signature1, signature2, concept_name1, concept_name2)
    
    def find_related_concepts(self, concept_name: str, relationship_type: str = "resonant") -> str:
        """
        Find concepts related to the given concept in a specific way.
        
        Args:
            concept_name: Name of the concept
            relationship_type: Type of relationship to look for
            
        Returns:
            Semantic explanation of related concepts
        """
        # Get related entities
        if relationship_type == "resonant":
            related = self.signature_service.find_resonant_entities(concept_name, threshold=0.7)
            
            if not related:
                return f"No concepts with strong resonance to '{concept_name}' were found."
            
            # Generate narrative
            narrative = f"Concepts resonating with '{concept_name}':\n\n"
            
            for entity_id, resonance in related:
                resonance_description = SignatureSemanticTranslator.translate_resonance_to_semantic_concept(resonance)
                narrative += f"- {entity_id}: {resonance_description} (resonance: {resonance:.2f})\n"
            
            return narrative
        
        return f"Relationship type '{relationship_type}' is not supported."
    
    def predict_concept_evolution(self, concept_name: str, time_frame: str = "near_future") -> str:
        """
        Predict how a concept will evolve in the specified time frame.
        
        Args:
            concept_name: Name of the concept
            time_frame: Time frame for prediction
            
        Returns:
            Semantic prediction of the concept's evolution
        """
        # Map time frame to time delta
        time_deltas = {"near_future": 5.0, "medium_term": 20.0, "long_term": 50.0}
        
        if time_frame not in time_deltas:
            return f"Time frame '{time_frame}' is not supported. Use 'near_future', 'medium_term', or 'long_term'."
        
        # Get prediction
        prediction = self.signature_service.predict_entity_state(concept_name, time_deltas[time_frame])
        
        if not prediction:
            return f"The concept '{concept_name}' is not currently active in the system."
        
        # Generate prediction narrative
        return self.narrative_generator.generate_prediction_narrative(prediction, concept_name)
    
    def summarize_field_state(self) -> str:
        """
        Provide a semantic summary of the current field state.
        
        Returns:
            Semantic summary of the field state
        """
        # This would require access to field state information
        # For now, we'll return a placeholder
        return "Field state summarization is not yet implemented."
    
    def explain_concept_in_context(self, concept_name: str, context_name: str) -> str:
        """
        Explain how a concept behaves in a specific context.
        
        Args:
            concept_name: Name of the concept
            context_name: Name of the context
            
        Returns:
            Semantic explanation of the concept in context
        """
        # This would require context-specific signature information
        # For now, we'll return a placeholder
        return f"Explanation of '{concept_name}' in context '{context_name}' is not yet implemented."
