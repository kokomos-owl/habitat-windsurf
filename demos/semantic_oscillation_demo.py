"""
Semantic Oscillation Demo

This demo shows how oscillatory signatures integrated with AdaptiveID can be
translated into meaningful semantic concepts and narratives for human users.

The demo illustrates:
1. Creating oscillatory adaptive IDs for entities
2. Generating semantic explanations of entity behavior
3. Explaining relationships between entities in semantic terms
4. Predicting future states with semantic narratives
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

from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from habitat_evolution.adaptive_core.id.oscillatory_adaptive_id import OscillatoryAdaptiveID
from habitat_evolution.adaptive_core.query.query_actant import QueryActant
from habitat_evolution.adaptive_core.query.query_interaction import QueryInteraction
from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint
from habitat_evolution.adaptive_core.transformation.meaning_bridges import MeaningBridge, MeaningBridgeTracker
from habitat_evolution.adaptive_core.oscillation.adaptive_id_signature_service import AdaptiveIDSignatureService
from habitat_evolution.adaptive_core.oscillation.semantic_signature_interface import (
    SignatureSemanticTranslator, 
    SignatureNarrativeGenerator,
    SemanticSignatureInterface
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class SemanticOscillationDemo:
    """
    Demonstrates the semantic interpretation of oscillatory signatures.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.signature_service = AdaptiveIDSignatureService()
        self.semantic_interface = SemanticSignatureInterface(self.signature_service)
        self.query_interaction = QueryInteraction()
        
        # Register query handlers for different modalities
        self.register_query_handlers()
        
        # Create output directory
        self.output_dir = "demos/output/semantic_oscillations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def register_query_handlers(self):
        """Register handlers for different query modalities."""
        # Text query handler
        self.query_interaction.register_query_handler("text", self.handle_text_query)
    
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
    
    def create_sample_entities(self):
        """Create sample entities with oscillatory adaptive IDs."""
        logger.info("Creating sample entities with oscillatory adaptive IDs")
        
        # Create climate change entity
        climate_change = OscillatoryAdaptiveID(
            base_concept="climate change",
            creator_id="demo",
            weight=1.0,
            confidence=0.95,
            uncertainty=0.05,
            fundamental_frequency=0.1,  # Slow-changing concept
            fundamental_amplitude=0.9,  # High impact
            fundamental_phase=np.pi/4,  # Early in its cycle
            harmonics=[
                {
                    "frequency": 0.2,
                    "amplitude": 0.7,
                    "phase": np.pi/6
                },
                {
                    "frequency": 0.3,
                    "amplitude": 0.5,
                    "phase": np.pi/3
                }
            ]
        )
        
        # Create sea level rise entity
        sea_level_rise = OscillatoryAdaptiveID(
            base_concept="sea level rise",
            creator_id="demo",
            weight=0.8,
            confidence=0.9,
            uncertainty=0.1,
            fundamental_frequency=0.15,  # Slightly faster than climate change
            fundamental_amplitude=0.7,   # Moderate to high impact
            fundamental_phase=np.pi/3,   # Early-mid in its cycle
            harmonics=[
                {
                    "frequency": 0.3,
                    "amplitude": 0.6,
                    "phase": np.pi/4
                }
            ]
        )
        
        # Create economic impact entity
        economic_impact = OscillatoryAdaptiveID(
            base_concept="economic impact",
            creator_id="demo",
            weight=0.7,
            confidence=0.8,
            uncertainty=0.2,
            fundamental_frequency=0.3,   # Faster-changing concept
            fundamental_amplitude=0.8,   # High impact
            fundamental_phase=np.pi/2,   # Mid-cycle
            harmonics=[
                {
                    "frequency": 0.6,
                    "amplitude": 0.5,
                    "phase": np.pi/3
                },
                {
                    "frequency": 0.9,
                    "amplitude": 0.3,
                    "phase": np.pi/2
                }
            ]
        )
        
        # Create policy response entity
        policy_response = OscillatoryAdaptiveID(
            base_concept="policy response",
            creator_id="demo",
            weight=0.6,
            confidence=0.7,
            uncertainty=0.3,
            fundamental_frequency=0.5,   # Fast-changing concept
            fundamental_amplitude=0.6,   # Moderate impact
            fundamental_phase=np.pi/6,   # Early in its cycle
            harmonics=[
                {
                    "frequency": 1.0,
                    "amplitude": 0.4,
                    "phase": np.pi/4
                }
            ]
        )
        
        # Create human-AI collaboration entity
        human_ai_collab = OscillatoryAdaptiveID(
            base_concept="human-AI collaboration",
            creator_id="demo",
            weight=0.9,
            confidence=0.85,
            uncertainty=0.15,
            fundamental_frequency=0.25,  # Moderate-changing concept
            fundamental_amplitude=0.85,  # High impact
            fundamental_phase=np.pi/4,   # Early in its cycle
            harmonics=[
                {
                    "frequency": 0.5,
                    "amplitude": 0.7,
                    "phase": np.pi/3
                },
                {
                    "frequency": 0.75,
                    "amplitude": 0.5,
                    "phase": np.pi/2
                },
                {
                    "frequency": 1.0,
                    "amplitude": 0.3,
                    "phase": 2*np.pi/3
                }
            ]
        )
        
        # Add to cache
        self.signature_service.adaptive_id_cache["climate_change"] = climate_change
        self.signature_service.adaptive_id_cache["sea_level_rise"] = sea_level_rise
        self.signature_service.adaptive_id_cache["economic_impact"] = economic_impact
        self.signature_service.adaptive_id_cache["policy_response"] = policy_response
        self.signature_service.adaptive_id_cache["human_ai_collaboration"] = human_ai_collab
        
        logger.info("Created sample entities with oscillatory adaptive IDs")
        
        return {
            "climate_change": climate_change,
            "sea_level_rise": sea_level_rise,
            "economic_impact": economic_impact,
            "policy_response": policy_response,
            "human_ai_collaboration": human_ai_collab
        }
    
    def create_query_with_oscillatory_id(self):
        """Create a query with an oscillatory adaptive ID."""
        logger.info("Creating query with oscillatory adaptive ID")
        
        # Create a query about climate change
        query = self.query_interaction.create_query(
            "What are the economic impacts of sea level rise due to climate change?",
            modality="text",
            context={"focus": "economic", "timeframe": "2050"}
        )
        
        # Process the query
        query_result = self.query_interaction.process_query(query)
        logger.info(f"Query result: {json.dumps(query_result, indent=2)}")
        
        # Create oscillatory adaptive ID for the query
        oscillatory_id = self.signature_service.create_oscillatory_adaptive_id_for_query(query)
        
        logger.info(f"Created oscillatory adaptive ID for query: {query.id}")
        
        return query
    
    def run_demo(self):
        """Run the semantic oscillation demo."""
        logger.info("Starting Semantic Oscillation Demo")
        
        # 1. Create sample entities with oscillatory adaptive IDs
        entities = self.create_sample_entities()
        
        # 2. Create a query with an oscillatory adaptive ID
        query = self.create_query_with_oscillatory_id()
        
        # 3. Generate semantic explanations of entity behavior
        logger.info("Generating semantic explanations of entity behavior")
        
        climate_explanation = self.semantic_interface.explain_concept_behavior("climate_change")
        sea_level_explanation = self.semantic_interface.explain_concept_behavior("sea_level_rise")
        economic_explanation = self.semantic_interface.explain_concept_behavior("economic_impact")
        
        # Save explanations to files
        self.save_to_file("climate_change_explanation.md", climate_explanation)
        self.save_to_file("sea_level_rise_explanation.md", sea_level_explanation)
        self.save_to_file("economic_impact_explanation.md", economic_explanation)
        
        # 4. Explain relationships between entities
        logger.info("Explaining relationships between entities")
        
        climate_sea_relationship = self.semantic_interface.explain_concept_relationship(
            "climate_change", "sea_level_rise")
        sea_economic_relationship = self.semantic_interface.explain_concept_relationship(
            "sea_level_rise", "economic_impact")
        
        # Save relationship explanations to files
        self.save_to_file("climate_sea_relationship.md", climate_sea_relationship)
        self.save_to_file("sea_economic_relationship.md", sea_economic_relationship)
        
        # 5. Predict future states
        logger.info("Predicting future states")
        
        climate_prediction = self.semantic_interface.predict_concept_evolution(
            "climate_change", "medium_term")
        policy_prediction = self.semantic_interface.predict_concept_evolution(
            "policy_response", "near_future")
        
        # Save predictions to files
        self.save_to_file("climate_change_prediction.md", climate_prediction)
        self.save_to_file("policy_response_prediction.md", policy_prediction)
        
        # 6. Find related concepts
        logger.info("Finding related concepts")
        
        related_to_climate = self.semantic_interface.find_related_concepts("climate_change")
        
        # Save related concepts to file
        self.save_to_file("related_to_climate.md", related_to_climate)
        
        # 7. Update oscillatory states and show changes
        logger.info("Updating oscillatory states and showing changes")
        
        # Update climate change state
        self.signature_service.update_oscillatory_state(
            "climate_change",
            time_delta=5.0,
            amplitude_change=0.1,  # Increasing impact
            energy_change=0.05     # Slightly increasing energy
        )
        
        # Update sea level rise state
        self.signature_service.update_oscillatory_state(
            "sea_level_rise",
            time_delta=5.0,
            amplitude_change=0.15,  # Significantly increasing impact
            energy_change=0.1       # Increasing energy
        )
        
        # Get updated explanations
        updated_climate_explanation = self.semantic_interface.explain_concept_behavior("climate_change")
        updated_sea_level_explanation = self.semantic_interface.explain_concept_behavior("sea_level_rise")
        
        # Save updated explanations to files
        self.save_to_file("climate_change_explanation_updated.md", updated_climate_explanation)
        self.save_to_file("sea_level_rise_explanation_updated.md", updated_sea_level_explanation)
        
        # 8. Show how query's oscillatory ID relates to concepts
        logger.info("Showing how query's oscillatory ID relates to concepts")
        
        query_climate_relationship = self.semantic_interface.explain_concept_relationship(
            query.id, "climate_change")
        
        # Save query relationship to file
        self.save_to_file("query_climate_relationship.md", query_climate_relationship)
        
        logger.info("Semantic Oscillation Demo completed")
        logger.info(f"Results saved to {self.output_dir}")
    
    def save_to_file(self, filename: str, content: str):
        """Save content to a file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Saved {filepath}")

if __name__ == "__main__":
    demo = SemanticOscillationDemo()
    demo.run_demo()
