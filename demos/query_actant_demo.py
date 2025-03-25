"""
Query Actant Demo

This demo shows how queries can be treated as first-class actants in the Habitat Evolution system,
allowing them to participate in semantic relationships and transformations across modalities
and AI systems.

The demo illustrates:
1. Creating queries as actants with their own identity and journey
2. Processing queries through different modalities
3. Detecting meaning bridges between queries and other actants
4. Creating narratives that describe query journeys and relationships
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.habitat_evolution.adaptive_core.query.query_actant import QueryActant
from src.habitat_evolution.adaptive_core.query.query_interaction import QueryInteraction
from src.habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint
from src.habitat_evolution.adaptive_core.transformation.meaning_bridges import MeaningBridge, MeaningBridgeTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class QueryActantDemo:
    """
    Demonstrates the use of queries as first-class actants in the Habitat Evolution system.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.query_interaction = QueryInteraction()
        self.actant_journeys = []
        self.meaning_bridge_tracker = MeaningBridgeTracker()
        
        # Register query handlers for different modalities
        self.register_query_handlers()
        
        # Create output directory
        self.output_dir = "demos/output/query_actants"
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
        
        # Simulate query processing
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
        
        # Create actant journey for policy response
        policy_journey = ActantJourney.create("policy_response")
        
        # Add journey points
        policy_journey.add_journey_point(ActantJourneyPoint(
            id="pj1",
            actant_name="policy_response",
            domain_id="policy_domain",
            predicate_id="formulation",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        policy_journey.add_journey_point(ActantJourneyPoint(
            id="pj2",
            actant_name="policy_response",
            domain_id="implementation",
            predicate_id="execution",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        policy_journey.add_journey_point(ActantJourneyPoint(
            id="pj3",
            actant_name="policy_response",
            domain_id="evaluation",
            predicate_id="assessment",
            role="subject",
            timestamp=datetime.now().isoformat()
        ))
        
        # Add journeys to the list
        self.actant_journeys = [sea_level_journey, economic_journey, policy_journey]
        logger.info(f"Created {len(self.actant_journeys)} sample actant journeys")
    
    def run_demo(self):
        """Run the query actant demo."""
        logger.info("Starting Query Actant Demo")
        
        # Create sample actant journeys
        self.create_sample_actant_journeys()
        
        # 1. Create a text query about sea level rise
        logger.info("Creating initial text query")
        text_query = self.query_interaction.create_query(
            "What is the projected sea level rise by 2050?",
            modality="text",
            context={"user_location": "coastal_city", "time_horizon": "2050"}
        )
        
        # Process the text query
        text_result = self.query_interaction.process_query(text_query)
        logger.info(f"Text query result: {json.dumps(text_result, indent=2)}")
        
        # 2. Transform the query to an image modality
        logger.info("Transforming query to image modality")
        image_query = self.query_interaction.transform_query_modality(
            text_query,
            "image",
            {"transformation_type": "text_to_image", "visualization_style": "map_overlay"}
        )
        
        # Process the image query
        image_result = self.query_interaction.process_query(image_query)
        logger.info(f"Image query result: {json.dumps(image_result, indent=2)}")
        
        # 3. Evolve the query based on new information
        logger.info("Evolving the query based on new information")
        evolved_query = self.query_interaction.evolve_query(
            text_query,
            "What economic impacts will result from a 2-meter sea level rise by 2050?",
            {"evolution_reason": "refining_scope", "additional_context": "economic_focus"}
        )
        
        # Process the evolved query
        evolved_result = self.query_interaction.process_query(evolved_query)
        logger.info(f"Evolved query result: {json.dumps(evolved_result, indent=2)}")
        
        # 4. Transform the evolved query to an audio modality
        logger.info("Transforming evolved query to audio modality")
        audio_query = self.query_interaction.transform_query_modality(
            evolved_query,
            "audio",
            {"transformation_type": "text_to_speech", "voice_style": "professional"}
        )
        
        # Process the audio query
        audio_result = self.query_interaction.process_query(audio_query)
        logger.info(f"Audio query result: {json.dumps(audio_result, indent=2)}")
        
        # 5. Detect meaning bridges between queries and other actants
        logger.info("Detecting meaning bridges")
        
        # Combine all actant journeys including query journeys
        all_journeys = self.actant_journeys.copy()
        for query in self.query_interaction.get_all_queries():
            if query.actant_journey:
                all_journeys.append(query.actant_journey)
        
        # Detect bridges
        bridges = self.meaning_bridge_tracker.detect_bridges(all_journeys, [])
        logger.info(f"Detected {len(bridges)} meaning bridges")
        
        # 6. Create query narratives
        logger.info("Creating query narratives")
        
        # Create narrative for the original text query
        text_query_narrative = self.query_interaction.create_query_narrative(text_query, bridges)
        
        # Create narrative for the evolved query
        evolved_query_narrative = self.query_interaction.create_query_narrative(evolved_query, bridges)
        
        # Save narratives to files
        self.save_query_narratives(text_query, text_query_narrative)
        self.save_query_narratives(evolved_query, evolved_query_narrative)
        
        # 7. Visualize query journeys and bridges
        logger.info("Visualizing query journeys and bridges")
        self.visualize_query_network(all_journeys, bridges)
        
        logger.info("Query Actant Demo completed")
    
    def save_query_narratives(self, query: QueryActant, narrative: str):
        """Save query narratives to files."""
        # Create a safe filename from the query ID
        filename = f"query_narrative_{query.id}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the narrative
        with open(filepath, "w") as f:
            f.write(narrative)
        
        logger.info(f"Saved query narrative to {filepath}")
    
    def visualize_query_network(self, actant_journeys: List[ActantJourney], bridges: List[MeaningBridge]):
        """Visualize the network of queries, actants, and bridges."""
        # Create a graph
        G = nx.Graph()
        
        # Add nodes for actants
        actant_ids = set()
        for journey in actant_journeys:
            actant_ids.add(journey.actant_id)
            
            # Determine if this is a query actant
            is_query = journey.actant_id.startswith("query_")
            
            # Add node with appropriate attributes
            G.add_node(
                journey.actant_id,
                type="query" if is_query else "actant",
                size=300 if is_query else 200,
                color="red" if is_query else "blue"
            )
        
        # Add edges for bridges
        for bridge in bridges:
            if bridge.source_actant_id in actant_ids and bridge.target_actant_id in actant_ids:
                G.add_edge(
                    bridge.source_actant_id,
                    bridge.target_actant_id,
                    weight=bridge.propensity,
                    type=bridge.bridge_type
                )
        
        # Create the visualization
        plt.figure(figsize=(12, 10))
        
        # Get node positions
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_types = nx.get_node_attributes(G, 'type')
        node_sizes = nx.get_node_attributes(G, 'size')
        node_colors = nx.get_node_attributes(G, 'color')
        
        # Draw query nodes
        query_nodes = [node for node, type_val in node_types.items() if type_val == "query"]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=query_nodes,
            node_size=[node_sizes[node] for node in query_nodes],
            node_color="red",
            alpha=0.8,
            label="Queries"
        )
        
        # Draw actant nodes
        actant_nodes = [node for node, type_val in node_types.items() if type_val == "actant"]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=actant_nodes,
            node_size=[node_sizes[node] for node in actant_nodes],
            node_color="blue",
            alpha=0.8,
            label="Actants"
        )
        
        # Draw edges with different colors based on bridge type
        edge_types = nx.get_edge_attributes(G, 'type')
        edge_weights = nx.get_edge_attributes(G, 'weight')
        
        # Group edges by type
        edges_by_type = {}
        for (u, v), edge_type in edge_types.items():
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append((u, v))
        
        # Draw edges for each type
        colors = {"co-occurrence": "green", "sequence": "purple", "domain_crossing": "orange"}
        for edge_type, edges in edges_by_type.items():
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                width=[edge_weights.get((u, v), 1) * 2 for u, v in edges],
                alpha=0.7,
                edge_color=colors.get(edge_type, "gray"),
                label=f"{edge_type.title()} Bridges"
            )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        
        # Add legend
        plt.legend()
        
        # Remove axis
        plt.axis("off")
        
        # Add title
        plt.title("Query Actants and Meaning Bridges Network", fontsize=16)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, "query_network.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved query network visualization to {output_path}")

if __name__ == "__main__":
    demo = QueryActantDemo()
    demo.run_demo()
