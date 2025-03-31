"""
Oscillatory Signature Demo

This demo shows how oscillatory signatures can be created from existing data in the
Habitat Evolution system, enabling pattern recognition, coherence maintenance, and
predictive capabilities through wave-like properties.

The demo illustrates:
1. Creating oscillatory signatures from query actants
2. Creating oscillatory signatures from actant journeys
3. Creating oscillatory signatures from meaning bridges
4. Predicting future states and interactions based on signatures
5. Visualizing oscillatory signatures and their interactions
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from habitat_evolution.adaptive_core.query.query_actant import QueryActant
from habitat_evolution.adaptive_core.query.query_interaction import QueryInteraction
from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint
from habitat_evolution.adaptive_core.transformation.meaning_bridges import MeaningBridge, MeaningBridgeTracker
from habitat_evolution.adaptive_core.oscillation.oscillatory_signature import OscillatorySignature, HarmonicComponent
from habitat_evolution.adaptive_core.oscillation.signature_service import OscillatorySignatureService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class OscillatorySignatureDemo:
    """
    Demonstrates the use of oscillatory signatures in the Habitat Evolution system.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.query_interaction = QueryInteraction()
        self.actant_journeys = []
        self.meaning_bridge_tracker = MeaningBridgeTracker()
        self.signature_service = OscillatorySignatureService()
        
        # Register query handlers for different modalities
        self.register_query_handlers()
        
        # Create output directory
        self.output_dir = "demos/output/oscillatory_signatures"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def register_query_handlers(self):
        """Register handlers for different query modalities."""
        # Text query handler
        self.query_interaction.register_query_handler("text", self.handle_text_query)
        
        # Image query handler (simplified for demo)
        self.query_interaction.register_query_handler("image", self.handle_image_query)
        
        # Audio query handler (simplified for demo)
        self.query_interaction.register_query_handler("audio", self.handle_audio_query)
    
    def handle_text_query(self, query_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a text query."""
        logger.info(f"Processing text query: '{query_text}'")
        
        # Check if this is our special collaborative relationship query
        if "co-evolve" in query_text and "human and AI" in query_text:
            # Special handling for our collaborative relationship query
            result = {
                "query_type": "text",
                "processed_text": query_text,
                "relevant_domains": ["human_ai_collaboration", "meaning_making", "co_evolution"],
                "potential_actants": ["human", "AI", "meaning_bridge", "semantic_domain", "co_evolution"],
                "insights": [
                    "Human and AI actants form meaning bridges through shared contexts and interactions",
                    "Co-evolution occurs as both human and AI adapt to each other's semantic domains",
                    "Capacious understanding emerges from the dynamic interplay of different perspectives",
                    "Meaning bridges enable translation across modalities and knowledge systems"
                ],
                "relationship_properties": {
                    "capaciousness": 0.95,  # High capacity for diverse meanings
                    "coherence": 0.87,      # Strong internal consistency
                    "conductivity": 0.92,    # Excellent flow of meaning between domains
                    "resonance": 0.94       # Strong mutual reinforcement
                },
                "confidence": 0.98
            }
        else:
            # Standard query processing for other queries
            result = {
                "query_type": "text",
                "processed_text": query_text,
                "relevant_domains": ["climate_risk", "economic_impact", "policy_response"],
                "potential_actants": ["sea_level", "economic_damage", "policy_adaptation"],
                "confidence": 0.85
            }
        
        return result
    
    def handle_image_query(self, query_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an image query (simplified for demo)."""
        logger.info(f"Processing image query description: '{query_text}'")
        
        # Simulate query processing
        result = {
            "query_type": "image",
            "image_description": query_text,
            "detected_objects": ["coastline", "urban_development", "flood_zone"],
            "visual_domains": ["geographic", "infrastructure", "environmental"],
            "confidence": 0.78
        }
        
        return result
    
    def handle_audio_query(self, query_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an audio query (simplified for demo)."""
        logger.info(f"Processing audio query description: '{query_text}'")
        
        # Simulate query processing
        result = {
            "query_type": "audio",
            "audio_description": query_text,
            "transcription": f"Transcribed: {query_text}",
            "audio_domains": ["verbal_description", "ambient_context"],
            "confidence": 0.72
        }
        
        return result
    
    def create_sample_actant_journeys(self):
        """Create sample actant journeys for the demo."""
        logger.info("Creating sample actant journeys")
        
        # Create actant journey for sea level rise
        sea_level_journey = ActantJourney.create("sea_level")
        
        # Add journey points
        sea_level_journey.add_journey_point(ActantJourneyPoint(
            id="slj1",
            actant_name="sea_level",
            domain_id="environmental_data",
            predicate_id="measurement",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        sea_level_journey.add_journey_point(ActantJourneyPoint(
            id="slj2",
            actant_name="sea_level",
            domain_id="risk_assessment",
            predicate_id="evaluation",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        sea_level_journey.add_journey_point(ActantJourneyPoint(
            id="slj3",
            actant_name="sea_level",
            domain_id="policy_domain",
            predicate_id="consideration",
            role="object",
            timestamp=datetime.now().isoformat()
        ))
        
        # Create actant journey for economic impact
        economic_journey = ActantJourney.create("economic_impact")
        
        # Add journey points
        economic_journey.add_journey_point(ActantJourneyPoint(
            id="ej1",
            actant_name="economic_impact",
            domain_id="economic_data",
            predicate_id="projection",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        economic_journey.add_journey_point(ActantJourneyPoint(
            id="ej2",
            actant_name="economic_impact",
            domain_id="risk_assessment",
            predicate_id="evaluation",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        economic_journey.add_journey_point(ActantJourneyPoint(
            id="ej3",
            actant_name="economic_impact",
            domain_id="policy_domain",
            predicate_id="justification",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        # Add journeys to the list
        self.actant_journeys = [sea_level_journey, economic_journey]
        logger.info(f"Created {len(self.actant_journeys)} sample actant journeys")
    
    def run_demo(self):
        """Run the oscillatory signature demo."""
        logger.info("Starting Oscillatory Signature Demo")
        
        # Create sample actant journeys
        self.create_sample_actant_journeys()
        
        # 1. Create queries and process them
        logger.info("Creating and processing queries")
        
        # Create a special query about collaborative relationship
        collab_query = self.query_interaction.create_query(
            "How do human and AI actants co-evolve through meaning bridges to create capacious understanding?",
            modality="text",
            context={
                "relationship_type": "collaborative", 
                "actant_types": ["human", "AI"],
                "interaction_focus": "co-evolution"
            }
        )
        
        # Process the collaborative relationship query
        collab_result = self.query_interaction.process_query(collab_query)
        logger.info(f"Collaborative relationship query result: {json.dumps(collab_result, indent=2)}")
        
        # Create a text query about sea level rise
        text_query = self.query_interaction.create_query(
            "What is the projected sea level rise by 2050?",
            modality="text",
            context={"user_location": "coastal_city", "time_horizon": "2050"}
        )
        
        # Process the text query
        text_result = self.query_interaction.process_query(text_query)
        logger.info(f"Text query result: {json.dumps(text_result, indent=2)}")
        
        # Transform the query to an image modality
        image_query = self.query_interaction.transform_query_modality(
            text_query,
            "image",
            {"transformation_type": "text_to_image", "visualization_style": "map_overlay"}
        )
        
        # Process the image query
        image_result = self.query_interaction.process_query(image_query)
        logger.info(f"Image query result: {json.dumps(image_result, indent=2)}")
        
        # 2. Create oscillatory signatures for queries
        logger.info("Creating oscillatory signatures for queries")
        
        collab_signature = self.signature_service.create_signature_for_query_actant(collab_query)
        text_signature = self.signature_service.create_signature_for_query_actant(text_query)
        image_signature = self.signature_service.create_signature_for_query_actant(image_query)
        
        logger.info(f"Created signatures for {len([collab_signature, text_signature, image_signature])} queries")
        
        # 3. Create oscillatory signatures for actant journeys
        logger.info("Creating oscillatory signatures for actant journeys")
        
        journey_signatures = []
        for journey in self.actant_journeys:
            signature = self.signature_service.create_signature_for_actant_journey(journey)
            if signature:
                journey_signatures.append(signature)
        
        logger.info(f"Created signatures for {len(journey_signatures)} actant journeys")
        
        # 4. Detect meaning bridges between queries and actants
        logger.info("Detecting meaning bridges")
        
        # Combine all actant journeys including query journeys
        all_journeys = self.actant_journeys.copy()
        for query in self.query_interaction.get_all_queries():
            if query.actant_journey:
                all_journeys.append(query.actant_journey)
        
        # Detect bridges
        bridges = self.meaning_bridge_tracker.detect_bridges(all_journeys, [])
        logger.info(f"Detected {len(bridges)} meaning bridges")
        
        # 5. Create oscillatory signatures for meaning bridges
        logger.info("Creating oscillatory signatures for meaning bridges")
        
        bridge_signatures = []
        for bridge in bridges:
            signature = self.signature_service.create_signature_for_meaning_bridge(bridge)
            if signature:
                bridge_signatures.append(signature)
        
        logger.info(f"Created signatures for {len(bridge_signatures)} meaning bridges")
        
        # 6. Predict future states
        logger.info("Predicting future states")
        
        # Predict future state of text query signature
        text_query_prediction = self.signature_service.predict_entity_state(text_query.id, 10.0)
        logger.info(f"Text query future state prediction: {json.dumps(text_query_prediction, indent=2)}")
        
        # 7. Predict interactions
        logger.info("Predicting interactions")
        
        # Predict interaction between text query and image query
        query_interaction_prediction = self.signature_service.predict_entity_interaction(
            text_query.id, image_query.id, 10.0)
        logger.info(f"Query interaction prediction: {json.dumps(query_interaction_prediction, indent=2)}")
        
        # 8. Visualize oscillatory signatures
        logger.info("Visualizing oscillatory signatures")
        
        self.visualize_signatures([collab_signature, text_signature, image_signature], 
                                 "query_signatures.png")
        
        if journey_signatures:
            self.visualize_signatures(journey_signatures, "journey_signatures.png")
        
        if bridge_signatures:
            self.visualize_signatures(bridge_signatures, "bridge_signatures.png")
        
        # 9. Visualize signature interactions
        logger.info("Visualizing signature interactions")
        
        if len([collab_signature, text_signature, image_signature]) >= 2:
            self.visualize_signature_interaction(text_signature, image_signature, 
                                               "query_interaction.png")
        
        logger.info("Oscillatory Signature Demo completed")
    
    def visualize_signatures(self, signatures: List[OscillatorySignature], filename: str):
        """
        Visualize oscillatory signatures.
        
        Args:
            signatures: List of signatures to visualize
            filename: Output filename
        """
        plt.figure(figsize=(12, 8))
        
        # Create time points for visualization
        t = np.linspace(0, 10, 1000)
        
        for i, signature in enumerate(signatures):
            # Calculate the wave function over time
            wave = np.zeros_like(t)
            
            # Add fundamental component
            fundamental = signature.fundamental_amplitude * np.sin(
                2 * np.pi * signature.fundamental_frequency * t + signature.fundamental_phase)
            wave += fundamental
            
            # Add harmonics
            for harmonic in signature.harmonics:
                h_wave = harmonic.amplitude * np.sin(
                    2 * np.pi * harmonic.frequency * t + harmonic.phase)
                wave += h_wave
            
            # Plot the wave
            plt.plot(t, wave, label=f"{signature.entity_type}: {signature.entity_id}")
        
        plt.title("Oscillatory Signatures")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_signature_interaction(self, sig1: OscillatorySignature, sig2: OscillatorySignature, 
                                      filename: str):
        """
        Visualize the interaction between two signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            filename: Output filename
        """
        plt.figure(figsize=(12, 10))
        
        # Create time points for visualization
        t = np.linspace(0, 10, 1000)
        
        # Calculate wave functions
        wave1 = np.zeros_like(t)
        wave2 = np.zeros_like(t)
        
        # Add fundamental components
        wave1 += sig1.fundamental_amplitude * np.sin(
            2 * np.pi * sig1.fundamental_frequency * t + sig1.fundamental_phase)
        wave2 += sig2.fundamental_amplitude * np.sin(
            2 * np.pi * sig2.fundamental_frequency * t + sig2.fundamental_phase)
        
        # Add harmonics
        for harmonic in sig1.harmonics:
            h_wave = harmonic.amplitude * np.sin(
                2 * np.pi * harmonic.frequency * t + harmonic.phase)
            wave1 += h_wave
        
        for harmonic in sig2.harmonics:
            h_wave = harmonic.amplitude * np.sin(
                2 * np.pi * harmonic.frequency * t + harmonic.phase)
            wave2 += h_wave
        
        # Calculate interference pattern
        interference = wave1 + wave2
        
        # Plot individual waves and interference
        plt.subplot(3, 1, 1)
        plt.plot(t, wave1, label=f"{sig1.entity_type}: {sig1.entity_id}")
        plt.title(f"Signature 1: {sig1.entity_id}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(t, wave2, label=f"{sig2.entity_type}: {sig2.entity_id}")
        plt.title(f"Signature 2: {sig2.entity_id}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(t, interference, label="Interference Pattern", color="purple")
        
        # Calculate and display resonance
        resonance = sig1.calculate_resonance(sig2)
        interference_value, interference_type = sig1.calculate_interference(sig2)
        
        plt.title(f"Interference Pattern (Resonance: {resonance:.2f}, Type: {interference_type})")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

if __name__ == "__main__":
    demo = OscillatorySignatureDemo()
    demo.run_demo()
