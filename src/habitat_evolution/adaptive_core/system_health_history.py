"""System health history persistence with Neo4j and NetworkX integration."""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import networkx as nx
from neo4j import GraphDatabase

class SystemHealthHistory:
    """Persistent store for system health metrics with visualization support."""
    
    def __init__(self, 
                 history_dir: str,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "habitat"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Neo4j connection
        self._neo4j_driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
        
    def record_health_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """Record a system health snapshot with timestamp."""
        timestamp = datetime.now().isoformat()
        snapshot_id = f"health_snapshot_{timestamp}"
        
        # Add to NetworkX graph
        self.graph.add_node(
            snapshot_id,
            type="health_snapshot",
            timestamp=timestamp,
            **snapshot
        )
        
        # Save to JSON file
        snapshot_file = self.history_dir / f"{snapshot_id}.json"
        with open(snapshot_file, 'w') as f:
            json.dump({
                'id': snapshot_id,
                'timestamp': timestamp,
                'data': snapshot
            }, f, indent=2)
            
        # Export to Neo4j
        self._export_to_neo4j(snapshot_id, snapshot)
        
        return snapshot_id
    
    def record_health_transition(self, 
                               from_snapshot: str,
                               to_snapshot: str,
                               metrics: Dict[str, float]) -> None:
        """Record a transition between health snapshots."""
        # Add to NetworkX graph
        self.graph.add_edge(
            from_snapshot,
            to_snapshot,
            type="health_transition",
            timestamp=datetime.now().isoformat(),
            **metrics
        )
        
        # Export to Neo4j
        self._export_transition_to_neo4j(from_snapshot, to_snapshot, metrics)
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific health snapshot."""
        snapshot_file = self.history_dir / f"{snapshot_id}.json"
        if snapshot_file.exists():
            with open(snapshot_file) as f:
                return json.load(f)
        return None
    
    def export_networkx_graph(self, output_file: str) -> None:
        """Export the health history as a NetworkX graph file."""
        nx.write_gexf(self.graph, output_file)
    
    def _export_to_neo4j(self, snapshot_id: str, snapshot: Dict[str, Any]) -> None:
        """Export health snapshot to Neo4j."""
        with self._neo4j_driver.session() as session:
            # Create snapshot node
            session.run("""
                CREATE (s:SystemHealthSnapshot {
                    id: $id,
                    timestamp: $timestamp,
                    tonic: $tonic,
                    pulse: $pulse,
                    resonance: $resonance,
                    tension: $tension
                })
            """, {
                'id': snapshot_id,
                'timestamp': snapshot.get('timestamp', datetime.now().isoformat()),
                'tonic': snapshot.get('tonic', {}),
                'pulse': snapshot.get('pulse', {}),
                'resonance': snapshot.get('resonance', {}),
                'tension': snapshot.get('tension', {})
            })
            
            # Create dimension nodes and relationships
            for dim, metrics in snapshot.get('dimensions', {}).items():
                session.run("""
                    MATCH (s:SystemHealthSnapshot {id: $snapshot_id})
                    CREATE (d:Dimension {
                        type: $dim_type,
                        boundary_tension: $boundary_tension,
                        window_state: $window_state
                    })
                    CREATE (s)-[:HAS_DIMENSION]->(d)
                """, {
                    'snapshot_id': snapshot_id,
                    'dim_type': dim,
                    'boundary_tension': metrics.get('boundary_tension', 0.0),
                    'window_state': metrics.get('window_state', 'CLOSED')
                })
    
    def _export_transition_to_neo4j(self, 
                                  from_id: str, 
                                  to_id: str,
                                  metrics: Dict[str, float]) -> None:
        """Export health transition to Neo4j."""
        with self._neo4j_driver.session() as session:
            session.run("""
                MATCH (from:SystemHealthSnapshot {id: $from_id})
                MATCH (to:SystemHealthSnapshot {id: $to_id})
                CREATE (from)-[:TRANSITIONS_TO {
                    timestamp: $timestamp,
                    metrics: $metrics
                }]->(to)
            """, {
                'from_id': from_id,
                'to_id': to_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
    
    def get_history_graph(self) -> nx.DiGraph:
        """Get the NetworkX graph of the health history."""
        return self.graph
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
