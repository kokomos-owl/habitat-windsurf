"""
Semantic Memory Persistence for Habitat Evolution.

This module implements the persistence layer for the elastic semantic memory system,
enabling storage and retrieval of entity and predicate quality states, transition histories,
and vector field snapshots.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
import networkx as nx
import numpy as np
from pathlib import Path

class SemanticMemoryPersistence:
    """
    Persistence layer for the elastic semantic memory system.
    
    This class enables storage and retrieval of entity and predicate quality states,
    transition histories, and vector field snapshots, creating a complete
    RAG↔Evolution↔Persistence loop.
    """
    
    def __init__(self, base_dir: str = None, logger=None):
        """
        Initialize the semantic memory persistence layer.
        
        Args:
            base_dir: Base directory for storing persistence files
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up base directory for persistence
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Default to a 'persistence' directory in the current working directory
            self.base_dir = Path(os.getcwd()) / 'persistence'
        
        # Create directories if they don't exist
        self.entity_dir = self.base_dir / 'entities'
        self.predicate_dir = self.base_dir / 'predicates'
        self.field_dir = self.base_dir / 'vector_field'
        self.network_dir = self.base_dir / 'entity_network'
        
        os.makedirs(self.entity_dir, exist_ok=True)
        os.makedirs(self.predicate_dir, exist_ok=True)
        os.makedirs(self.field_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)
        
        self.logger.info(f"Initialized semantic memory persistence at {self.base_dir}")
    
    def save_entity_quality_state(self, entity_quality: Dict[str, str], 
                                 entity_confidence: Dict[str, float],
                                 quality_transition_history: Dict[str, List[Dict]]) -> str:
        """
        Save entity quality states and transition history.
        
        Args:
            entity_quality: Dictionary mapping entities to quality states
            entity_confidence: Dictionary mapping entities to confidence scores
            quality_transition_history: Dictionary mapping entities to lists of transition events
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"entity_quality_state_{timestamp}.json"
        filepath = self.entity_dir / filename
        
        data = {
            'entity_quality': entity_quality,
            'entity_confidence': entity_confidence,
            'quality_transition_history': quality_transition_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved entity quality state to {filepath}")
        return str(filepath)
    
    def save_predicate_quality_state(self, predicate_quality: Dict[str, str],
                                    predicate_confidence: Dict[str, float],
                                    domain_predicate_specialization: Dict[Tuple[str, str], Dict[str, float]],
                                    quality_transition_history: Dict[str, List[Dict]]) -> str:
        """
        Save predicate quality states and transition history.
        
        Args:
            predicate_quality: Dictionary mapping predicates to quality states
            predicate_confidence: Dictionary mapping predicates to confidence scores
            domain_predicate_specialization: Dictionary mapping domain pairs to predicate specializations
            quality_transition_history: Dictionary mapping predicates to lists of transition events
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predicate_quality_state_{timestamp}.json"
        filepath = self.predicate_dir / filename
        
        # Convert tuple keys to strings for JSON serialization
        domain_spec_serializable = {str(k): v for k, v in domain_predicate_specialization.items()}
        
        data = {
            'predicate_quality': predicate_quality,
            'predicate_confidence': predicate_confidence,
            'domain_predicate_specialization': domain_spec_serializable,
            'quality_transition_history': quality_transition_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved predicate quality state to {filepath}")
        return str(filepath)
    
    def save_vector_field_snapshot(self, field_metrics: Dict[str, float],
                                  field_stability: float,
                                  field_coherence: float,
                                  field_density: Dict[str, float]) -> str:
        """
        Save a snapshot of the vector field state.
        
        Args:
            field_metrics: Dictionary of field metrics
            field_stability: Current field stability value
            field_coherence: Current field coherence value
            field_density: Dictionary mapping regions to density values
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vector_field_snapshot_{timestamp}.json"
        filepath = self.field_dir / filename
        
        data = {
            'field_metrics': field_metrics,
            'field_stability': field_stability,
            'field_coherence': field_coherence,
            'field_density': field_density,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved vector field snapshot to {filepath}")
        return str(filepath)
    
    def save_entity_network(self, entity_network: nx.DiGraph) -> str:
        """
        Save the entity network graph.
        
        Args:
            entity_network: NetworkX DiGraph representing the entity network
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"entity_network_{timestamp}.json"
        filepath = self.network_dir / filename
        
        # Convert NetworkX graph to serializable format
        data = {
            'nodes': [],
            'edges': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add nodes with attributes
        for node, attrs in entity_network.nodes(data=True):
            node_data = {'id': node}
            node_data.update(attrs)
            # Convert non-serializable values to strings
            for k, v in node_data.items():
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    node_data[k] = str(v)
            data['nodes'].append(node_data)
        
        # Add edges with attributes
        for source, target, attrs in entity_network.edges(data=True):
            edge_data = {'source': source, 'target': target}
            edge_data.update(attrs)
            # Convert non-serializable values to strings
            for k, v in edge_data.items():
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    edge_data[k] = str(v)
            data['edges'].append(edge_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved entity network to {filepath}")
        return str(filepath)
    
    def load_entity_quality_state(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Load entity quality states and transition history.
        
        Args:
            filepath: Path to the file to load, or None to load the most recent
            
        Returns:
            Dictionary containing loaded entity quality data
        """
        if filepath is None:
            # Find the most recent file
            files = list(self.entity_dir.glob("entity_quality_state_*.json"))
            if not files:
                self.logger.warning("No entity quality state files found")
                return {}
            
            filepath = str(max(files, key=os.path.getctime))
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded entity quality state from {filepath}")
        return data
    
    def load_predicate_quality_state(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Load predicate quality states and transition history.
        
        Args:
            filepath: Path to the file to load, or None to load the most recent
            
        Returns:
            Dictionary containing loaded predicate quality data
        """
        if filepath is None:
            # Find the most recent file
            files = list(self.predicate_dir.glob("predicate_quality_state_*.json"))
            if not files:
                self.logger.warning("No predicate quality state files found")
                return {}
            
            filepath = str(max(files, key=os.path.getctime))
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded predicate quality state from {filepath}")
        return data
    
    def load_vector_field_snapshot(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a vector field snapshot.
        
        Args:
            filepath: Path to the file to load, or None to load the most recent
            
        Returns:
            Dictionary containing loaded vector field data
        """
        if filepath is None:
            # Find the most recent file
            files = list(self.field_dir.glob("vector_field_snapshot_*.json"))
            if not files:
                self.logger.warning("No vector field snapshot files found")
                return {}
            
            filepath = str(max(files, key=os.path.getctime))
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded vector field snapshot from {filepath}")
        return data
    
    def load_entity_network(self, filepath: Optional[str] = None) -> nx.DiGraph:
        """
        Load an entity network graph.
        
        Args:
            filepath: Path to the file to load, or None to load the most recent
            
        Returns:
            NetworkX DiGraph representing the entity network
        """
        if filepath is None:
            # Find the most recent file
            files = list(self.network_dir.glob("entity_network_*.json"))
            if not files:
                self.logger.warning("No entity network files found")
                return nx.DiGraph()
            
            filepath = str(max(files, key=os.path.getctime))
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create a new graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_data in data.get('nodes', []):
            node_id = node_data.pop('id')
            G.add_node(node_id, **node_data)
        
        # Add edges with attributes
        for edge_data in data.get('edges', []):
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            G.add_edge(source, target, **edge_data)
        
        self.logger.info(f"Loaded entity network from {filepath}")
        return G
    
    def get_latest_state(self) -> Dict[str, Any]:
        """
        Get the latest state of the entire semantic memory.
        
        Returns:
            Dictionary containing the latest entity quality, predicate quality,
            vector field, and entity network data
        """
        entity_quality = self.load_entity_quality_state()
        predicate_quality = self.load_predicate_quality_state()
        vector_field = self.load_vector_field_snapshot()
        
        # Load entity network as serializable data
        entity_network = self.load_entity_network()
        network_data = {
            'node_count': entity_network.number_of_nodes(),
            'edge_count': entity_network.number_of_edges(),
            'nodes': list(entity_network.nodes()),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'entity_quality': entity_quality,
            'predicate_quality': predicate_quality,
            'vector_field': vector_field,
            'entity_network': network_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_complete_state(self, entity_quality: Dict[str, str],
                           entity_confidence: Dict[str, float],
                           entity_transition_history: Dict[str, List[Dict]],
                           predicate_quality: Dict[str, str],
                           predicate_confidence: Dict[str, float],
                           domain_predicate_specialization: Dict[Tuple[str, str], Dict[str, float]],
                           predicate_transition_history: Dict[str, List[Dict]],
                           field_metrics: Dict[str, float],
                           field_stability: float,
                           field_coherence: float,
                           field_density: Dict[str, float],
                           entity_network: nx.DiGraph) -> Dict[str, str]:
        """
        Save a complete snapshot of the semantic memory state.
        
        Args:
            entity_quality: Dictionary mapping entities to quality states
            entity_confidence: Dictionary mapping entities to confidence scores
            entity_transition_history: Dictionary mapping entities to lists of transition events
            predicate_quality: Dictionary mapping predicates to quality states
            predicate_confidence: Dictionary mapping predicates to confidence scores
            domain_predicate_specialization: Dictionary mapping domain pairs to predicate specializations
            predicate_transition_history: Dictionary mapping predicates to lists of transition events
            field_metrics: Dictionary of field metrics
            field_stability: Current field stability value
            field_coherence: Current field coherence value
            field_density: Dictionary mapping regions to density values
            entity_network: NetworkX DiGraph representing the entity network
            
        Returns:
            Dictionary mapping component names to saved file paths
        """
        entity_path = self.save_entity_quality_state(
            entity_quality, entity_confidence, entity_transition_history
        )
        
        predicate_path = self.save_predicate_quality_state(
            predicate_quality, predicate_confidence, 
            domain_predicate_specialization, predicate_transition_history
        )
        
        field_path = self.save_vector_field_snapshot(
            field_metrics, field_stability, field_coherence, field_density
        )
        
        network_path = self.save_entity_network(entity_network)
        
        # Create a manifest file with references to all components
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_filename = f"semantic_memory_manifest_{timestamp}.json"
        manifest_filepath = self.base_dir / manifest_filename
        
        manifest = {
            'entity_quality_path': entity_path,
            'predicate_quality_path': predicate_path,
            'vector_field_path': field_path,
            'entity_network_path': network_path,
            'timestamp': datetime.now().isoformat(),
            'entity_count': len(entity_quality),
            'predicate_count': len(predicate_quality),
            'relationship_count': entity_network.number_of_edges()
        }
        
        with open(manifest_filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Saved complete semantic memory state with manifest at {manifest_filepath}")
        
        return {
            'entity_quality': entity_path,
            'predicate_quality': predicate_path,
            'vector_field': field_path,
            'entity_network': network_path,
            'manifest': str(manifest_filepath)
        }
