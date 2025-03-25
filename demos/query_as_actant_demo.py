#!/usr/bin/env python3
"""
Query as Actant Demo

This script demonstrates how queries can be treated as first-class actants in the
HarmonicIO dataflow system. It shows how queries can participate in the semantic
ecosystem, evolving and transforming as they interact with other actants and domains.

The demo illustrates the modality-agnostic nature of the Habitat Pattern Language
framework, where semantic patterns can be preserved across different representations.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from habitat_evolution.adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from habitat_evolution.climate_risk.harmonic_climate_processor import create_climate_processor
from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint, DomainTransition
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('query_as_actant_demo.log')
    ]
)

logger = logging.getLogger(__name__)


class QueryAsActantDemo:
    """
    Demonstrates how queries can function as first-class actants.
    
    This class shows how queries can participate in the semantic ecosystem,
    evolving and transforming as they interact with other actants and domains.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the Query as Actant demo.
        
        Args:
            data_dir: Directory containing climate risk data
        """
        self.data_dir = Path(data_dir)
        
        # Create HarmonicIO service and climate processor
        logger.info("Initializing HarmonicIO service and climate processor")
        self.processor, self.io_service = create_climate_processor(str(self.data_dir))
        
        # Initialize visualization data
        self.visualization_data = {
            "nodes": [],
            "edges": [],
            "query_paths": {}
        }
        
        # Initialize query actants
        self.query_actants = []
        
    def run_demo(self):
        """
        Run the Query as Actant demo.
        
        This method processes climate risk data, creates query actants,
        tracks their journeys through the system, and visualizes the results.
        """
        logger.info("Starting Query as Actant demo")
        
        # Start the HarmonicIO service
        self.io_service.start()
        logger.info("HarmonicIO service started")
        
        # Process climate risk data
        logger.info("Processing climate risk data")
        metrics = self.processor.process_data()
        logger.info(f"Processing metrics: {json.dumps(metrics, indent=2)}")
        
        # Allow time for async operations to complete
        logger.info("Waiting for async operations to complete")
        time.sleep(2)
        
        # Create query actants
        self.create_query_actants()
        
        # Process queries through the system
        self.process_query_actants()
        
        # Prepare visualization data
        self.prepare_visualization_data()
        
        # Create visualization
        self.create_query_journey_visualization()
        
        # Create query transformation narrative
        self.create_query_transformation_narrative()
        
        # Stop the HarmonicIO service
        self.io_service.stop()
        logger.info("HarmonicIO service stopped")
        
        logger.info("Query as Actant demo completed")
        
    def create_query_actants(self):
        """
        Create query actants that will journey through the system.
        
        This method creates query actants with AdaptiveIDs, allowing them
        to function as first-class entities in the semantic ecosystem.
        """
        logger.info("Creating query actants")
        
        # Define sample queries
        queries = [
            {
                "id": "query_001",
                "text": "What are the economic impacts of sea level rise?",
                "focus": "economic_impact",
                "context": "climate_risk",
                "confidence": 0.8
            },
            {
                "id": "query_002",
                "text": "What policy responses address coastal flooding?",
                "focus": "policy_response",
                "context": "adaptation",
                "confidence": 0.7
            },
            {
                "id": "query_003",
                "text": "How do sea level rise and economic impacts influence policy?",
                "focus": "integrated_analysis",
                "context": "decision_making",
                "confidence": 0.6
            }
        ]
        
        # Create actant journeys for each query
        for query in queries:
            # Create adaptive ID for query
            adaptive_id = AdaptiveID(
                base_concept=query["text"],
                creator_id="query_as_actant_demo",
                weight=1.0,
                confidence=query["confidence"],
                uncertainty=0.3
            )
            
            # Add initial temporal context
            adaptive_id.update_temporal_context(
                "creation_time",
                datetime.now().isoformat(),
                "initialization"
            )
            
            # Add query context
            adaptive_id.update_temporal_context(
                "query_focus",
                query["focus"],
                "focus_assignment"
            )
            
            adaptive_id.update_temporal_context(
                "query_context",
                query["context"],
                "context_assignment"
            )
            
            # Create actant journey for query
            journey = ActantJourney.create(actant_name=query["id"])
            journey.adaptive_id = adaptive_id
            
            # Add to query actants list
            self.query_actants.append({
                "query": query,
                "journey": journey
            })
            
        logger.info(f"Created {len(self.query_actants)} query actants")
        
    def process_query_actants(self):
        """
        Process query actants through the system.
        
        This method simulates the journey of query actants through different
        domains, creating journey points and domain transitions.
        """
        logger.info("Processing query actants through the system")
        
        # Get discovered domains from processor
        domains = self.processor.discovered_domains
        logger.info(f"Found {len(domains)} domains: {domains}")
        
        # For each query actant, create a journey through relevant domains
        for query_actant in self.query_actants:
            query = query_actant["query"]
            journey = query_actant["journey"]
            
            # Determine relevant domains based on query focus
            relevant_domains = []
            for domain in domains:
                if query["focus"] in domain or "risk" in domain:
                    relevant_domains.append(domain)
                    
            if not relevant_domains:
                relevant_domains = list(domains)
                
            logger.info(f"Query {query['id']} will journey through domains: {relevant_domains}")
            
            # Create journey points for each relevant domain
            for i, domain in enumerate(relevant_domains):
                # Determine role based on domain and query focus
                if "risk" in domain.lower():
                    role = "investigator"
                elif "economic" in domain.lower():
                    role = "analyst"
                elif "policy" in domain.lower():
                    role = "advisor"
                else:
                    role = "explorer"
                
                # Create journey point
                point = ActantJourneyPoint.create(
                    actant_name=query["id"],
                    domain_id=domain,
                    predicate_id=f"query_predicate_{i}",
                    role=role,
                    timestamp=datetime.now().isoformat()
                )
                
                # Add to journey
                journey.add_journey_point(point)
                
                # Update adaptive ID with domain context
                journey.adaptive_id.update_temporal_context(
                    f"domain_visit_{domain}",
                    {"role": role, "timestamp": datetime.now().isoformat()},
                    "domain_interaction"
                )
                
                # Simulate pattern evolution by adjusting confidence and uncertainty
                if i > 0:
                    # Queries become more refined as they journey through domains
                    journey.adaptive_id.confidence += 0.05
                    journey.adaptive_id.uncertainty -= 0.05
                    
                    # Ensure values stay in valid range
                    journey.adaptive_id.confidence = min(0.95, journey.adaptive_id.confidence)
                    journey.adaptive_id.uncertainty = max(0.05, journey.adaptive_id.uncertainty)
            
            # Create domain transitions between consecutive domains
            for i in range(len(relevant_domains) - 1):
                source_domain = relevant_domains[i]
                target_domain = relevant_domains[i + 1]
                
                # Determine source and target roles
                if "risk" in source_domain.lower():
                    source_role = "investigator"
                elif "economic" in source_domain.lower():
                    source_role = "analyst"
                elif "policy" in source_domain.lower():
                    source_role = "advisor"
                else:
                    source_role = "explorer"
                    
                if "risk" in target_domain.lower():
                    target_role = "investigator"
                elif "economic" in target_domain.lower():
                    target_role = "analyst"
                elif "policy" in target_domain.lower():
                    target_role = "advisor"
                else:
                    target_role = "explorer"
                
                # Create transition
                transition = DomainTransition.create(
                    actant_name=query["id"],
                    source_domain_id=source_domain,
                    target_domain_id=target_domain,
                    source_predicate_id=f"query_predicate_{i}",
                    target_predicate_id=f"query_predicate_{i+1}",
                    source_role=source_role,
                    target_role=target_role,
                    timestamp=datetime.now().isoformat()
                )
                
                # Add to journey
                journey.add_domain_transition(transition)
                
        logger.info("Processed query actants through the system")
        
    def prepare_visualization_data(self):
        """
        Prepare data for query journey visualization.
        """
        logger.info("Preparing visualization data")
        
        # Add domains as nodes
        domains = set()
        for query_actant in self.query_actants:
            journey = query_actant["journey"]
            for point in journey.journey_points:
                domains.add(point.domain_id)
                
        for domain in domains:
            self.visualization_data["nodes"].append({
                "id": domain,
                "type": "domain",
                "label": domain
            })
        
        # Add queries as nodes
        for query_actant in self.query_actants:
            query = query_actant["query"]
            journey = query_actant["journey"]
            
            self.visualization_data["nodes"].append({
                "id": query["id"],
                "type": "query",
                "label": query["id"],
                "text": query["text"]
            })
            
            # Track query paths
            self.visualization_data["query_paths"][query["id"]] = []
            
            # Add domain transitions as edges
            for transition in journey.domain_transitions:
                edge = {
                    "source": transition.source_domain_id,
                    "target": transition.target_domain_id,
                    "query": query["id"],
                    "source_role": transition.source_role,
                    "target_role": transition.target_role
                }
                self.visualization_data["edges"].append(edge)
                
                # Add to query path
                self.visualization_data["query_paths"][query["id"]].append(
                    (transition.source_domain_id, transition.target_domain_id)
                )
        
        logger.info(f"Prepared visualization data with {len(self.visualization_data['nodes'])} nodes and {len(self.visualization_data['edges'])} edges")
        
    def create_query_journey_visualization(self):
        """
        Create a visualization of query journeys through the system.
        
        This method creates a network graph visualization showing how queries
        journey through different domains, functioning as first-class actants.
        """
        logger.info("Creating query journey visualization")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.visualization_data["nodes"]:
            if node["type"] == "domain":
                G.add_node(node["id"], label=node["label"], node_type=node["type"])
        
        # Add edges
        for edge in self.visualization_data["edges"]:
            G.add_edge(edge["source"], edge["target"], 
                      query=edge["query"],
                      source_role=edge["source_role"],
                      target_role=edge["target_role"])
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_size=2000, 
                              node_color="lightgreen", 
                              alpha=0.8)
        
        # Draw edges with different colors for different queries
        query_colors = {
            "query_001": "red",
            "query_002": "blue",
            "query_003": "purple"
        }
        
        for query_id, color in query_colors.items():
            query_edges = [(u, v) for u, v, d in G.edges(data=True) if d["query"] == query_id]
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=query_edges,
                                  width=2, 
                                  alpha=0.7, 
                                  edge_color=color,
                                  arrows=True,
                                  arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=query_id) 
                          for query_id, color in query_colors.items()]
        plt.legend(handles=legend_elements, loc="upper right")
        
        # Add title and adjust layout
        plt.title("Query as Actant: Journey Visualization", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        
        # Save figure
        plt.savefig("query_journey_visualization.png", dpi=300, bbox_inches="tight")
        logger.info("Saved query journey visualization to query_journey_visualization.png")
        
    def create_query_transformation_narrative(self):
        """
        Create a narrative of how queries transform as they journey through the system.
        
        This method creates a markdown document describing how queries function as
        first-class actants, evolving and transforming as they interact with domains.
        """
        logger.info("Creating query transformation narrative")
        
        narrative = []
        narrative.append("# Query as Actant: Transformation Narrative")
        narrative.append("\nThis document describes how queries function as first-class actants in the semantic ecosystem, evolving and transforming as they journey through different domains.\n")
        
        # Add section on the modality-agnostic nature of the framework
        narrative.append("## Modality-Agnostic Knowledge Representation")
        narrative.append("\nThe Habitat Pattern Language framework enables a modality-agnostic approach to knowledge representation, where semantic patterns can be preserved across different representations. Queries, as actants, demonstrate this capability by maintaining their semantic identity while transforming across domains.\n")
        
        # Add section on queries as creative relationships
        narrative.append("## Queries as Creative Relationships")
        narrative.append("\nQueries represent creative relationships between habitat, human, and intelligent agents. They are not merely information retrieval mechanisms but active participants in the semantic ecosystem, capable of evolution and transformation.\n")
        
        # Add individual query narratives
        narrative.append("## Query Transformation Stories")
        
        for query_actant in self.query_actants:
            query = query_actant["query"]
            journey = query_actant["journey"]
            
            # Get adaptive ID data
            adaptive_id_dict = journey.adaptive_id.to_dict()
            
            # Add query header
            narrative.append(f"\n### {query['id']}: {query['text']}")
            
            # Add initial query context
            narrative.append(f"\n**Initial Context:** {query['context']}")
            narrative.append(f"\n**Focus:** {query['focus']}")
            narrative.append(f"\n**Initial Confidence:** {query['confidence']}")
            
            # Add pattern propensities if available
            if "pattern_propensities" in adaptive_id_dict:
                propensities = adaptive_id_dict["pattern_propensities"]
                narrative.append("\n#### Pattern Propensities")
                narrative.append(f"\n- Coherence: {propensities.get('coherence', 'N/A')}")
                narrative.append(f"\n- Capaciousness: {propensities.get('capaciousness', 'N/A')}")
                
                if "directionality" in propensities:
                    narrative.append("\n#### Directionality")
                    for direction, value in propensities["directionality"].items():
                        narrative.append(f"\n- {direction}: {value}")
            
            # Add journey narrative
            narrative.append("\n#### Transformation Journey")
            
            if journey.journey_points:
                # Sort journey points by timestamp
                sorted_points = sorted(journey.journey_points, 
                                      key=lambda p: datetime.fromisoformat(p.timestamp))
                
                # Create narrative of the journey
                for i, point in enumerate(sorted_points):
                    if i == 0:
                        narrative.append(f"\nThe query begins in the {point.domain_id} domain as a {point.role}.")
                    else:
                        narrative.append(f"\nThen, it moves to the {point.domain_id} domain as a {point.role}.")
            
            # Add role shifts
            role_shifts = journey.get_role_shifts()
            if role_shifts:
                narrative.append("\n#### Role Transformations")
                for shift in role_shifts:
                    narrative.append(f"\n- Transformed from {shift.source_role} to {shift.target_role} when moving from {shift.source_domain_id} to {shift.target_domain_id}.")
            
            # Add semantic transformation insights
            narrative.append("\n#### Semantic Evolution")
            
            # Final confidence and uncertainty
            narrative.append(f"\n**Final Confidence:** {journey.adaptive_id.confidence:.2f}")
            narrative.append(f"\n**Final Uncertainty:** {journey.adaptive_id.uncertainty:.2f}")
            
            # Evolution narrative
            if journey.adaptive_id.confidence > query["confidence"]:
                narrative.append(f"\nAs the query journeyed through domains, its confidence increased by {(journey.adaptive_id.confidence - query['confidence']):.2f}, indicating that the semantic ecosystem provided clarity and refinement to the original query.")
            else:
                narrative.append(f"\nAs the query journeyed through domains, its confidence remained stable, suggesting that the semantic ecosystem maintained the integrity of the original query intent.")
                
            if journey.adaptive_id.uncertainty < 0.3:  # Initial uncertainty was 0.3
                narrative.append(f"\nThe query's uncertainty decreased as it interacted with domain knowledge, resulting in a more focused and precise semantic representation.")
            
            # Add separator
            narrative.append("\n---\n")
        
        # Add conclusion
        narrative.append("## Conclusion: Queries as First-Class Actants")
        narrative.append("\nThis demonstration shows how queries can function as first-class actants in the semantic ecosystem, with their own identities, journeys, and transformations. By treating queries as actants, we enable a more dynamic and interactive knowledge representation system, where queries are not just static information retrieval mechanisms but active participants in the co-evolution of semantic patterns.\n")
        narrative.append("\nThe modality-agnostic nature of the Habitat Pattern Language framework allows these query actants to maintain their semantic identity across different representations and domains, demonstrating the power of pattern-based knowledge representation in creating a more flexible and adaptive semantic ecosystem.\n")
        
        # Write narrative to file
        with open("query_transformation_narrative.md", "w") as f:
            f.write("\n".join(narrative))
            
        logger.info("Saved query transformation narrative to query_transformation_narrative.md")


def main():
    """Run the Query as Actant demo."""
    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "climate_risk"
    
    # Create and run demo
    demo = QueryAsActantDemo(str(data_dir))
    demo.run_demo()


if __name__ == "__main__":
    main()
