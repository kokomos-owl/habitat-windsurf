"""
Visualize Actant Journeys

This script visualizes the actant journeys created by the climate risk processor,
demonstrating how the enhanced ActantJourney integration with AdaptiveID is working.
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
import matplotlib.pyplot as plt
import networkx as nx

# Add src to path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from habitat_evolution.adaptive_core.transformation.actant_journey_tracker import ActantJourney, ActantJourneyTracker, ActantJourneyPoint, DomainTransition
from habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID
from habitat_evolution.climate_risk.harmonic_climate_processor import create_climate_processor


def create_journey_network_graph(journeys: List[ActantJourney], domains: List[str], output_dir: str) -> None:
    """
    Create a network graph visualization of actant journeys across domains.
    
    Args:
        journeys: List of actant journeys to visualize
        domains: List of domain names
        output_dir: Directory to save the visualization
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add domain nodes
    for domain in domains:
        G.add_node(domain, type='domain')
    
    # Add actant nodes and edges
    for journey in journeys:
        actant_name = journey.actant_name
        G.add_node(actant_name, type='actant')
        
        # Add edges for domain transitions
        for transition in journey.domain_transitions:
            source = transition.source_domain_id
            target = transition.target_domain_id
            # Add edge between domains
            if not G.has_edge(source, target):
                G.add_edge(source, target, weight=1, actants=[])
            else:
                G[source][target]['weight'] += 1
            
            # Track which actants are involved in this transition
            G[source][target]['actants'].append(actant_name)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Position nodes using a spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw domain nodes (larger, blue)
    domain_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'domain']
    nx.draw_networkx_nodes(G, pos, nodelist=domain_nodes, node_size=700, node_color='skyblue', alpha=0.8)
    
    # Draw actant nodes (smaller, green)
    actant_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'actant']
    nx.draw_networkx_nodes(G, pos, nodelist=actant_nodes, node_size=300, node_color='lightgreen', alpha=0.6)
    
    # Draw edges with varying thickness based on weight
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            width = 1 + data['weight'] * 0.5
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7, 
                                 edge_color='gray', arrows=True, arrowsize=15)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title('Climate Risk Actant Journey Network')
    plt.axis('off')
    
    # Save the figure
    output_path = os.path.join(output_dir, 'actant_journey_network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Network graph saved to {output_path}")


def create_pattern_propensity_visualization(journeys: List[ActantJourney], output_dir: str) -> None:
    """
    Create visualizations of pattern propensity for actant journeys.
    
    Args:
        journeys: List of actant journeys to visualize
        output_dir: Directory to save the visualization
    """
    # Collect pattern propensity data
    coherence_values = []
    capaciousness_values = []
    directionality_counts = {}
    
    for journey in journeys:
        # Get pattern propensity from adaptive ID
        pattern_propensity = journey.adaptive_id.get_temporal_context("pattern_propensity")
        if pattern_propensity:
            # Extract coherence and capaciousness
            coherence_values.append((journey.actant_name, pattern_propensity.get('coherence', 0)))
            capaciousness_values.append((journey.actant_name, pattern_propensity.get('capaciousness', 0)))
            
            # Extract directionality
            directionality = pattern_propensity.get('directionality', {})
            for direction, value in directionality.items():
                if direction not in directionality_counts:
                    directionality_counts[direction] = 0
                directionality_counts[direction] += value
    
    # Sort by values
    coherence_values.sort(key=lambda x: x[1], reverse=True)
    capaciousness_values.sort(key=lambda x: x[1], reverse=True)
    
    # Create coherence and capaciousness bar chart
    plt.figure(figsize=(12, 8))
    
    # Plot coherence
    names = [item[0][:15] for item in coherence_values]  # Truncate long names
    values = [item[1] for item in coherence_values]
    plt.bar(names, values, color='blue', alpha=0.7, label='Coherence')
    
    plt.title('Pattern Coherence by Actant')
    plt.xlabel('Actant')
    plt.ylabel('Coherence Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'pattern_coherence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create capaciousness bar chart
    plt.figure(figsize=(12, 8))
    
    names = [item[0][:15] for item in capaciousness_values]  # Truncate long names
    values = [item[1] for item in capaciousness_values]
    plt.bar(names, values, color='green', alpha=0.7, label='Capaciousness')
    
    plt.title('Pattern Capaciousness by Actant')
    plt.xlabel('Actant')
    plt.ylabel('Capaciousness Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'pattern_capaciousness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create directionality pie chart if we have data
    if directionality_counts:
        plt.figure(figsize=(10, 10))
        
        labels = []
        sizes = []
        for direction, count in sorted(directionality_counts.items(), key=lambda x: x[1], reverse=True):
            # Shorten the direction name for display
            short_label = direction.replace('_to_', '→')
            if len(short_label) > 25:
                short_label = short_label[:22] + '...'
            labels.append(short_label)
            sizes.append(count)
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
               shadow=True, explode=[0.05] * len(sizes))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Directionality of Pattern Transformations')
        
        # Save the figure
        output_path = os.path.join(output_dir, 'pattern_directionality.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Pattern propensity visualizations saved to {output_dir}")


def save_journey_data(journeys: List[ActantJourney], output_dir: str) -> None:
    """
    Save journey data as JSON for further analysis.
    
    Args:
        journeys: List of actant journeys to save
        output_dir: Directory to save the data
    """
    journey_data = []
    
    for journey in journeys:
        # Convert journey to dictionary
        journey_dict = journey.to_dict()
        
        # Add additional information from AdaptiveID
        if journey.adaptive_id:
            journey_dict['pattern_propensity'] = journey.adaptive_id.get_temporal_context("pattern_propensity")
            journey_dict['journey_state'] = journey.adaptive_id.get_temporal_context("journey_state")
        
        journey_data.append(journey_dict)
    
    # Save as JSON
    output_path = os.path.join(output_dir, 'actant_journeys.json')
    with open(output_path, 'w') as f:
        json.dump(journey_data, f, indent=2)
    
    logger.info(f"Journey data saved to {output_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_actant_journeys(data_dir: str, output_dir: str = None):
    """
    Visualize actant journeys created from climate risk data.
    
    Args:
        data_dir: Directory containing climate risk data
    """
    logger.info(f"Creating climate processor for data directory: {data_dir}")
    processor, io_service = create_climate_processor(data_dir)
    
    try:
        # Process data
        logger.info("Processing climate risk data")
        metrics = processor.process_data()
        logger.info(f"Processing complete. Metrics: {metrics}")
        
        # Get actant journeys from the processor
        actant_journeys = processor.get_actant_journeys()
        logger.info(f"Retrieved {len(actant_journeys)} actant journeys from processor")
        
        if not actant_journeys:
            logger.warning("No actant journeys found. Creating sample journeys for visualization...")
            # Create sample journeys for demonstration if none exist
            actant_journeys = create_sample_journeys()
            
        # Extract unique domains from journeys
        domains = set()
        for journey in actant_journeys:
            for point in journey.journey_points:
                domains.add(point.domain_id)
            for transition in journey.domain_transitions:
                domains.add(transition.source_domain_id)
                domains.add(transition.target_domain_id)
        
        logger.info(f"Found {len(domains)} unique domains in actant journeys")
        
        # Create visualizations
        logger.info("Creating network graph visualization...")
        create_journey_network_graph(actant_journeys, list(domains), output_dir)
        
        logger.info("Creating pattern propensity visualizations...")
        create_pattern_propensity_visualization(actant_journeys, output_dir)
        
        logger.info("Saving journey data as JSON...")
        save_journey_data(actant_journeys, output_dir)
        
        # If we have actant journeys from the processor or sample data, we don't need to create more
        if actant_journeys:
            logger.info(f"Using {len(actant_journeys)} existing actant journeys")
            return actant_journeys
            
        # If we reach here, we need to create actant journeys
        logger.info("Creating actant journeys from entities")
        actant_journeys = []
        
        for entity_id in processor.discovered_entities:
            # Create journey for entity
            journey = ActantJourney.create(entity_id)
            
            # Initialize AdaptiveID
            journey.initialize_adaptive_id()
            
            # Add journey points for each domain where entity appears
            for domain_id in processor.discovered_domains:
                # Check if entity appears in this domain
                # In a real implementation, we would check the actual data
                # For this example, we'll add all entities to all domains
                journey_point = ActantJourneyPoint.create(
                    actant_name=entity_id,
                    domain_id=domain_id,
                    predicate_id=str(uuid.uuid4()),
                    role="subject",  # Default role
                    timestamp=datetime.now().isoformat(),
                    confidence=0.8
                )
                journey.add_journey_point(journey_point)
            
            # Add domain transitions
            domains = list(processor.discovered_domains)
            for i in range(len(domains) - 1):
                source_domain = domains[i]
                target_domain = domains[i + 1]
                
                # Create and add domain transition using the create method
                source_predicate_id = str(uuid.uuid4())
                target_predicate_id = str(uuid.uuid4())
                transition = DomainTransition.create(
                    actant_name=entity_id,
                    source_domain_id=source_domain,
                    target_domain_id=target_domain,
                    source_predicate_id=source_predicate_id,
                    target_predicate_id=target_predicate_id,
                    source_role="subject",  # Default role
                    target_role="object",   # Default role
                    timestamp=datetime.now().isoformat()
                )
                journey.add_domain_transition(transition)
            
            # Add role shifts for demonstration
            journey.add_role_shift(
                source_role="subject",
                target_role="object",
                predicate_id=str(uuid.uuid4()),
                timestamp=metrics["start_time"]
            )
            
            # Add journey to the tracker's dictionary
            journey_tracker.actant_journeys[journey.actant_name] = journey
            actant_journeys.append(journey)
        
        # Print journey information
        logger.info(f"Created {len(actant_journeys)} actant journeys")
        
        for journey in actant_journeys:
            journey_dict = journey.to_dict()
            logger.info(f"Journey for {journey.actant_name}:")
            logger.info(f"  - ID: {journey.id}")
            logger.info(f"  - Journey points: {len(journey.journey_points)}")
            logger.info(f"  - Domain transitions: {len(journey.domain_transitions)}")
            
            # Print AdaptiveID information
            if journey.adaptive_id:
                logger.info(f"  - AdaptiveID: {journey.adaptive_id.id}")
                logger.info(f"  - Base concept: {journey.adaptive_id.base_concept}")
                logger.info(f"  - Confidence: {journey.adaptive_id.confidence}")
                
                # Print pattern propensity if available
                pattern_propensity = journey.adaptive_id.get_temporal_context("pattern_propensity")
                if pattern_propensity:
                    logger.info(f"  - Pattern propensity: {pattern_propensity}")
                
                # Print journey state if available
                journey_state = journey.adaptive_id.get_temporal_context("journey_state")
                if journey_state:
                    logger.info(f"  - Journey state: {journey_state}")
            
            logger.info("")
        
        # Demonstrate pattern propagation across learning windows
        logger.info("Demonstrating pattern propagation across learning windows")
        
        # Create a mock learning window
        class MockLearningWindow:
            def __init__(self, name):
                self.name = name
                self.registered_ids = []
                self.state_changes = []
            
            def register_adaptive_id(self, adaptive_id):
                self.registered_ids.append(adaptive_id)
                logger.info(f"Learning window {self.name} registered AdaptiveID {adaptive_id.id}")
            
            def notify_state_change(self, adaptive_id, change_type, old_value, new_value, origin):
                self.state_changes.append({
                    "adaptive_id": adaptive_id.id,
                    "change_type": change_type,
                    "origin": origin
                })
                logger.info(f"Learning window {self.name} notified of state change: {change_type} from {origin}")
                
            def record_state_change(self, adaptive_id, change_type, old_value, new_value, origin):
                # This method is required by the AdaptiveID system
                # Handle both string IDs and AdaptiveID objects
                id_value = adaptive_id if isinstance(adaptive_id, str) else adaptive_id.id
                self.state_changes.append({
                    "adaptive_id": id_value,
                    "change_type": change_type,
                    "origin": origin
                })
                logger.info(f"Learning window {self.name} recorded state change: {change_type} from {origin}")
        
        # Create learning windows
        learning_windows = [
            MockLearningWindow("Window1"),
            MockLearningWindow("Window2")
        ]
        
        # Register journeys with learning windows
        for journey in actant_journeys:
            for window in learning_windows:
                journey.register_with_learning_window(window)
        
        # Make changes to journeys to demonstrate notifications
        logger.info("Making changes to journeys to demonstrate notifications")
        
        for journey in actant_journeys:
            # Add another domain transition
            # Create a domain transition object
            transition = DomainTransition.create(
                actant_name=journey.actant_name,
                source_domain_id=domains[-1],
                target_domain_id=domains[0],
                source_predicate_id=str(uuid.uuid4()),
                target_predicate_id=str(uuid.uuid4()),
                source_role="subject",
                target_role="object",
                timestamp=metrics["end_time"]
            )
            
            # Add the transition to the journey
            journey.add_domain_transition(transition)
            
            # Add another role shift
            journey.add_role_shift(
                source_role="object",
                target_role="subject",
                predicate_id=str(uuid.uuid4()),
                timestamp=metrics["end_time"]
            )
        
        # Print notification statistics
        for i, window in enumerate(learning_windows):
            logger.info(f"Learning window {window.name}:")
            logger.info(f"  - Registered AdaptiveIDs: {len(window.registered_ids)}")
            logger.info(f"  - Received state changes: {len(window.state_changes)}")
            
            # Count change types
            change_types = {}
            for change in window.state_changes:
                change_type = change["change_type"]
                change_types[change_type] = change_types.get(change_type, 0) + 1
            
            logger.info(f"  - Change types: {change_types}")
            logger.info("")
        
        # Create visualizations
        logger.info("Creating visualizations")
        
        # 1. Create journey network graph
        create_journey_network_graph(actant_journeys, domains, output_dir)
        
        # 2. Create pattern propensity visualization
        create_pattern_propensity_visualization(actant_journeys, output_dir)
        
        # 3. Save journey data as JSON
        save_journey_data(actant_journeys, output_dir)
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    finally:
        # Ensure I/O service is stopped
        logger.info("Stopping I/O service")
        io_service.stop()


def create_sample_journeys() -> List[ActantJourney]:
    """Create sample actant journeys for demonstration purposes."""
    logger.info("Creating sample actant journeys for demonstration")
    
    # Create domains
    domains = ["climate_risk", "economic_impact", "social_adaptation", "policy_response", "infrastructure"]
    
    # Create sample journeys
    journeys = []
    
    # Sample actants
    actants = [
        "sea_level_rise", "coastal_erosion", "storm_surge", "rainfall_pattern", 
        "drought_condition", "temperature_increase", "ecosystem_change"
    ]
    
    for actant_name in actants:
        # Create a new journey using the create class method
        journey = ActantJourney.create(actant_name=actant_name)
        
        # Add journey points (3-5 per journey)
        num_points = random.randint(3, 5)
        visited_domains = random.sample(domains, num_points)
        
        for i, domain in enumerate(visited_domains):
            # Create roles based on domain
            if domain == "climate_risk":
                role = "hazard"
            elif domain == "economic_impact":
                role = "stressor"
            elif domain == "social_adaptation":
                role = "catalyst"
            elif domain == "policy_response":
                role = "target"
            else:
                role = "vulnerability"
                
            # Create timestamp (earlier points have earlier timestamps)
            timestamp = datetime.now().isoformat()
            
            # Create journey point
            point = ActantJourneyPoint.create(
                actant_name=actant_name,
                domain_id=domain,
                predicate_id=f"predicate_{i}",
                role=role,
                timestamp=timestamp
            )
            
            # Add to journey
            journey.add_journey_point(point)
            
            # Update pattern propensity
            if journey.adaptive_id:
                # Random values for demonstration
                pattern_propensity = {
                    "coherence": random.uniform(0.1, 1.0),
                    "capaciousness": random.uniform(0.1, 1.0),
                    "directionality": {
                        f"{domains[0]}_to_{domains[1]}": random.uniform(0.1, 0.5),
                        f"{domains[1]}_to_{domains[2]}": random.uniform(0.1, 0.5),
                        f"{domains[2]}_to_{domains[3]}": random.uniform(0.1, 0.5),
                    }
                }
                journey.adaptive_id.update_temporal_context(
                    "pattern_propensity",
                    pattern_propensity,
                    "initialization"
                )
        
        # Add domain transitions (connecting the journey points)
        for i in range(len(visited_domains) - 1):
            source_domain = visited_domains[i]
            target_domain = visited_domains[i + 1]
            
            # Create source and target roles
            if source_domain == "climate_risk":
                source_role = "hazard"
            else:
                source_role = "stressor"
                
            if target_domain == "policy_response":
                target_role = "target"
            else:
                target_role = "vulnerability"
            
            # Create transition
            transition = DomainTransition.create(
                actant_name=actant_name,
                source_domain_id=source_domain,
                target_domain_id=target_domain,
                source_predicate_id=f"predicate_{i}",
                target_predicate_id=f"predicate_{i+1}",
                source_role=source_role,
                target_role=target_role,
                timestamp=datetime.now().isoformat()
            )
            
            # Add to journey
            journey.add_domain_transition(transition)
        
        journeys.append(journey)
    
    return journeys


if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description='Visualize actant journeys from climate risk data')
    parser.add_argument('--data-dir', type=str, default='data/climate_risk',
                        help='Directory containing climate risk data')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save visualization outputs')
    
    args = parser.parse_args()
    
    # Resolve data directory path
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        # Make relative to src directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), data_dir)
    
    # Resolve output directory path if provided
    output_dir = args.output_dir
    if output_dir and not os.path.isabs(output_dir):
        # Make relative to src directory
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), output_dir)
    
    visualize_actant_journeys(data_dir, output_dir)
