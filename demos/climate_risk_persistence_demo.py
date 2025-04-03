#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Climate Risk Persistence Demo

This script demonstrates processing climate risk data through the Habitat Evolution
persistence layer to detect patterns, establish relationships, and make sense of
the climate risk information.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Habitat Evolution components
from src.habitat_evolution.core.services.event_bus import LocalEventBus, Event
from src.habitat_evolution.adaptive_core.emergence.vector_tonic_persistence_connector import create_connector
from src.habitat_evolution.adaptive_core.persistence.factory import create_repositories
from src.habitat_evolution.pattern_aware_rag.persistence.arangodb.connection_manager import ArangoDBConnectionManager

class ClimateRiskPersistenceDemo:
    """Demo for processing climate risk data through the persistence layer."""
    
    def __init__(self):
        """Initialize the demo with necessary components."""
        self.event_bus = LocalEventBus()
        self.connection_manager = ArangoDBConnectionManager()
        self.db = self.connection_manager.get_db()
        
        # Create repositories
        self.repositories = create_repositories(self.db)
        
        # Create persistence connector
        self.connector = create_connector(
            event_bus=self.event_bus,
            db=self.db,
            field_state_repository=self.repositories["field_state_repository"],
            pattern_repository=self.repositories["pattern_repository"],
            relationship_repository=self.repositories["relationship_repository"],
            topology_repository=self.repositories["topology_repository"]
        )
        
        # Data paths
        self.data_dir = Path(__file__).parent / "data" / "climate_risk"
        
        # Statistics for reporting
        self.stats = {
            "frames_processed": 0,
            "observations_processed": 0,
            "patterns_detected": 0,
            "relationships_detected": 0,
            "field_states_updated": 0
        }
    
    def load_json_file(self, filename):
        """Load a JSON file from the data directory."""
        file_path = self.data_dir / filename
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def process_observation_frames(self):
        """Process observation frames to establish field states."""
        logger.info("Processing observation frames...")
        
        # Load observation frames
        frames_data = self.load_json_file("observation_frames.json")
        frames = frames_data.get("observation_frames", [])
        
        for frame in frames:
            frame_id = f"frame_{frame['name']}"
            
            # Create field state from frame
            field_state = {
                "id": frame_id,
                "name": frame["name"],
                "description": frame["description"],
                "resonance_centers": frame["resonance_centers"],
                "dimensionality": frame["field_properties"]["dimensionality"],
                "density_centers": frame["field_properties"]["density_centers"],
                "flow_vectors": frame["field_properties"]["flow_vectors"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Process field state through connector
            previous_state = {}  # No previous state for new frames
            self.connector.on_field_state_change(
                field_id=frame_id,
                previous_state=previous_state,
                new_state=field_state,
                metadata={"source": "climate_risk_demo"}
            )
            
            # Create topology from flow vectors
            topology = {
                "field_id": frame_id,
                "nodes": [center for center in frame["resonance_centers"]],
                "edges": [
                    {
                        "source": vector["source"],
                        "target": vector["target"],
                        "weight": vector["strength"]
                    }
                    for vector in frame["field_properties"]["flow_vectors"]
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            # Process topology through connector
            self.connector.on_topology_change(
                field_id=frame_id,
                previous_topology={},
                new_topology=topology,
                metadata={"source": "climate_risk_demo"}
            )
            
            self.stats["frames_processed"] += 1
            self.stats["field_states_updated"] += 1
        
        logger.info(f"Processed {self.stats['frames_processed']} observation frames")
    
    def process_regional_observations(self):
        """Process regional observations to detect patterns and relationships."""
        logger.info("Processing regional observations...")
        
        # Load regional observations
        observations_data = self.load_json_file("regional_observations.json")
        observations = observations_data.get("observations", [])
        
        for obs in observations:
            # Process as pattern if it has observed entities
            if "observed_entity_a" in obs and "observed_entity_b" in obs:
                pattern_id = obs["id"]
                
                # Create pattern from observation
                pattern_data = {
                    "id": pattern_id,
                    "name": f"{obs['observed_entity_a']} - {obs['observed_entity_b']}",
                    "phenomenon": obs["observed_phenomenon"],
                    "vector": [
                        obs["context"]["vector_properties"]["x"],
                        obs["context"]["vector_properties"]["y"],
                        obs["context"]["vector_properties"]["z"],
                        obs["context"]["vector_properties"]["w"],
                        obs["context"]["vector_properties"]["v"]
                    ],
                    "harmonic_properties": obs["context"]["harmonic_properties"],
                    "confidence": obs["context"]["confidence"],
                    "tonic_value": obs["context"]["tonic_value"],
                    "perspective": obs["context"]["perspective"],
                    "timestamp": obs["timestamp"],
                    "raw_data": obs.get("raw_data", {})
                }
                
                # Process pattern through connector
                self.connector.on_pattern_detected(
                    pattern_id=pattern_id,
                    pattern_data=pattern_data,
                    metadata={"source": "climate_risk_demo"}
                )
                
                self.stats["patterns_detected"] += 1
            
            # Process as relationship if it has source, predicate, target
            elif "source" in obs and "predicate" in obs and "target" in obs:
                relationship_id = obs["id"]
                
                # Create source pattern if not exists
                source_id = f"entity_{uuid.uuid4()}"
                source_pattern = {
                    "id": source_id,
                    "name": obs["source"],
                    "vector": [
                        obs["context"]["vector_properties"]["x"],
                        obs["context"]["vector_properties"]["y"],
                        obs["context"]["vector_properties"]["z"],
                        obs["context"]["vector_properties"]["w"],
                        obs["context"]["vector_properties"]["v"]
                    ],
                    "confidence": obs["context"]["confidence"],
                    "timestamp": obs["timestamp"]
                }
                
                # Create target pattern if not exists
                target_id = f"entity_{uuid.uuid4()}"
                target_pattern = {
                    "id": target_id,
                    "name": obs["target"],
                    "vector": [
                        obs["context"]["vector_properties"]["x"] * 0.9,  # Slightly different vector
                        obs["context"]["vector_properties"]["y"] * 1.1,
                        obs["context"]["vector_properties"]["z"] * 0.95,
                        obs["context"]["vector_properties"]["w"] * 1.05,
                        obs["context"]["vector_properties"]["v"] * 0.98
                    ],
                    "confidence": obs["context"]["confidence"] * 0.9,  # Slightly lower confidence
                    "timestamp": obs["timestamp"]
                }
                
                # Process patterns through connector
                self.connector.on_pattern_detected(
                    pattern_id=source_id,
                    pattern_data=source_pattern,
                    metadata={"source": "climate_risk_demo"}
                )
                
                self.connector.on_pattern_detected(
                    pattern_id=target_id,
                    pattern_data=target_pattern,
                    metadata={"source": "climate_risk_demo"}
                )
                
                # Create relationship data
                relationship_data = {
                    "id": relationship_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "type": obs["predicate"],
                    "strength": obs["context"]["tonic_value"],
                    "harmonic_properties": obs["context"]["harmonic_properties"],
                    "perspective": obs["context"]["perspective"],
                    "timestamp": obs["timestamp"]
                }
                
                # Process relationship through connector
                self.connector.on_pattern_relationship_detected(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_data=relationship_data,
                    metadata={"source": "climate_risk_demo"}
                )
                
                self.stats["patterns_detected"] += 2
                self.stats["relationships_detected"] += 1
            
            self.stats["observations_processed"] += 1
        
        logger.info(f"Processed {self.stats['observations_processed']} observations")
    
    def verify_persistence(self):
        """Verify that data was correctly persisted."""
        logger.info("Verifying persistence...")
        
        # Check patterns
        pattern_count = len(self.repositories["pattern_repository"].find_all())
        logger.info(f"Found {pattern_count} patterns in the database")
        
        # Check relationships
        relationship_count = len(self.repositories["relationship_repository"].find_all())
        logger.info(f"Found {relationship_count} relationships in the database")
        
        # Check field states
        field_state_count = len(self.repositories["field_state_repository"].find_all())
        logger.info(f"Found {field_state_count} field states in the database")
        
        # Check topology
        topology_count = len(self.repositories["topology_repository"].find_all())
        logger.info(f"Found {topology_count} topologies in the database")
    
    def run(self):
        """Run the complete demo."""
        logger.info("Starting Climate Risk Persistence Demo")
        
        try:
            # Process observation frames first to establish field states
            self.process_observation_frames()
            
            # Process regional observations to detect patterns and relationships
            self.process_regional_observations()
            
            # Verify persistence
            self.verify_persistence()
            
            # Report statistics
            logger.info("Demo completed successfully")
            logger.info(f"Statistics: {json.dumps(self.stats, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error running demo: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    demo = ClimateRiskPersistenceDemo()
    demo.run()
