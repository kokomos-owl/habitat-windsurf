"""
Topology manager for detecting, persisting, and retrieving topology constructs.

This module provides a central manager for working with topology states,
including detecting topological features, persisting them to Neo4j,
serializing/deserializing states, and tracking topology evolution.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from neo4j import GraphDatabase

from habitat_evolution.pattern_aware_rag.topology.models import (
    TopologyState, FrequencyDomain, Boundary, ResonancePoint, FieldMetrics
)
from habitat_evolution.pattern_aware_rag.topology.detector import TopologyDetector

logger = logging.getLogger(__name__)


class TopologyManager:
    """Manages the detection, persistence, and retrieval of topology constructs."""
    
    def __init__(self, neo4j_driver=None, persistence_mode=True):
        """
        Initialize the topology manager.
        
        Args:
            neo4j_driver: Neo4j driver for database persistence
            persistence_mode: Whether to persist topology states to Neo4j
        """
        self.detector = TopologyDetector()
        self.neo4j_driver = neo4j_driver
        self.persistence_mode = persistence_mode
        self.current_state = None
        self.state_history = []  # Limited history of recent states
    
    def analyze_patterns(self, patterns, learning_windows, time_period) -> TopologyState:
        """
        Analyze patterns and windows to detect topology.
        
        Args:
            patterns: List of pattern objects
            learning_windows: List of learning window objects
            time_period: Dictionary with 'start' and 'end' keys for the analysis period
            
        Returns:
            Detected topology state
        """
        logger.info(f"Analyzing topology for {len(patterns)} patterns and {len(learning_windows)} windows")
        
        # Use the detector to analyze topology
        self.current_state = self.detector.analyze_topology(patterns, learning_windows, time_period)
        
        # Add to history (limiting to 10 states)
        self.state_history.append(self.current_state)
        if len(self.state_history) > 10:
            self.state_history.pop(0)
        
        # Persist if enabled
        if self.persistence_mode and self.neo4j_driver:
            self.persist_to_neo4j(self.current_state)
        
        return self.current_state
    
    def persist_to_neo4j(self, state: TopologyState) -> bool:
        """
        Persist topology state to Neo4j.
        
        Args:
            state: Topology state to persist
            
        Returns:
            True if successful, False otherwise
        """
        if not self.neo4j_driver:
            logger.warning("No Neo4j driver provided, skipping persistence")
            return False
            
        try:
            with self.neo4j_driver.session() as session:
                # Create topology state node
                session.run("""
                    MERGE (ts:TopologyState {id: $id})
                    SET ts.timestamp = $timestamp,
                        ts.coherence = $coherence,
                        ts.entropy = $entropy
                """, 
                    id=state.id,
                    timestamp=state.timestamp.timestamp(),
                    coherence=state.field_metrics.coherence,
                    entropy=state.field_metrics.entropy
                )
                
                # Create frequency domains
                for domain_id, domain in state.frequency_domains.items():
                    session.run("""
                        MERGE (fd:FrequencyDomain {id: $id})
                        SET fd.dominantFrequency = $dominantFrequency,
                            fd.bandwidth = $bandwidth,
                            fd.phaseCoherence = $phaseCoherence,
                            fd.lastUpdated = $lastUpdated
                        WITH fd
                        MATCH (ts:TopologyState {id: $state_id})
                        MERGE (ts)-[:CONTAINS]->(fd)
                    """, 
                        id=domain_id,
                        dominantFrequency=domain.dominant_frequency,
                        bandwidth=domain.bandwidth,
                        phaseCoherence=domain.phase_coherence,
                        lastUpdated=domain.last_updated.timestamp(),
                        state_id=state.id
                    )
                    
                    # Connect patterns to domains
                    for pattern_id in domain.pattern_ids:
                        session.run("""
                            MATCH (fd:FrequencyDomain {id: $domain_id})
                            MERGE (p:Pattern {id: $pattern_id})
                            MERGE (p)-[:RESONATES_IN]->(fd)
                        """,
                            domain_id=domain_id,
                            pattern_id=pattern_id
                        )
                
                # Create boundaries
                for boundary_id, boundary in state.boundaries.items():
                    session.run("""
                        MERGE (b:Boundary {id: $id})
                        SET b.sharpness = $sharpness,
                            b.permeability = $permeability,
                            b.stability = $stability,
                            b.dimensionality = $dimensionality,
                            b.lastUpdated = $lastUpdated
                        WITH b
                        MATCH (ts:TopologyState {id: $state_id})
                        MERGE (ts)-[:CONTAINS]->(b)
                    """, 
                        id=boundary_id,
                        sharpness=boundary.sharpness,
                        permeability=boundary.permeability,
                        stability=boundary.stability,
                        dimensionality=boundary.dimensionality,
                        lastUpdated=boundary.last_updated.timestamp(),
                        state_id=state.id
                    )
                    
                    # Connect domains to boundaries
                    if len(boundary.domain_ids) == 2:
                        domain1, domain2 = boundary.domain_ids
                        session.run("""
                            MATCH (fd1:FrequencyDomain {id: $domain1})
                            MATCH (fd2:FrequencyDomain {id: $domain2})
                            MATCH (b:Boundary {id: $boundary_id})
                            MERGE (fd1)-[:BOUNDED_BY]->(b)
                            MERGE (fd2)-[:BOUNDED_BY]->(b)
                        """,
                            domain1=domain1,
                            domain2=domain2,
                            boundary_id=boundary_id
                        )
                
                # Create resonance points
                for point_id, point in state.resonance_points.items():
                    session.run("""
                        MERGE (r:ResonancePoint {id: $id})
                        SET r.strength = $strength,
                            r.stability = $stability,
                            r.attractorRadius = $attractorRadius,
                            r.lastUpdated = $lastUpdated
                        WITH r
                        MATCH (ts:TopologyState {id: $state_id})
                        MERGE (ts)-[:CONTAINS]->(r)
                    """, 
                        id=point_id,
                        strength=point.strength,
                        stability=point.stability,
                        attractorRadius=point.attractor_radius,
                        lastUpdated=point.last_updated.timestamp(),
                        state_id=state.id
                    )
                    
                    # Connect patterns to resonance points
                    for pattern_id, weight in point.contributing_pattern_ids.items():
                        session.run("""
                            MATCH (r:ResonancePoint {id: $point_id})
                            MERGE (p:Pattern {id: $pattern_id})
                            MERGE (p)-[:CONTRIBUTES_TO {weight: $weight}]->(r)
                        """,
                            point_id=point_id,
                            pattern_id=pattern_id,
                            weight=weight
                        )
                
            logger.info(f"Successfully persisted topology state {state.id} to Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Error persisting topology state to Neo4j: {e}")
            return False
    
    def load_from_neo4j(self, state_id=None, time_point=None) -> Optional[TopologyState]:
        """
        Load topology state from Neo4j.
        
        Args:
            state_id: ID of the state to load, or None for latest
            time_point: Timestamp to find nearest state, or None for latest
            
        Returns:
            Loaded topology state, or None if not found
        """
        if not self.neo4j_driver:
            logger.warning("No Neo4j driver provided, cannot load from Neo4j")
            return None
            
        try:
            with self.neo4j_driver.session() as session:
                # Query parameters
                params = {}
                
                # Build query based on parameters
                if state_id:
                    query = """
                        MATCH (ts:TopologyState {id: $state_id})
                        RETURN ts
                    """
                    params['state_id'] = state_id
                elif time_point:
                    if isinstance(time_point, datetime):
                        time_point = time_point.timestamp()
                    query = """
                        MATCH (ts:TopologyState)
                        WITH ts, abs(ts.timestamp - $time_point) AS diff
                        ORDER BY diff ASC
                        LIMIT 1
                        RETURN ts
                    """
                    params['time_point'] = time_point
                else:
                    query = """
                        MATCH (ts:TopologyState)
                        ORDER BY ts.timestamp DESC
                        LIMIT 1
                        RETURN ts
                    """
                
                # Execute query
                result = session.run(query, **params)
                record = result.single()
                
                if not record:
                    logger.warning("No topology state found in Neo4j")
                    return None
                    
                ts_node = record['ts']
                state_id = ts_node['id']
                
                # Create topology state
                state = TopologyState(
                    id=state_id,
                    timestamp=datetime.fromtimestamp(ts_node['timestamp']),
                    field_metrics=FieldMetrics(
                        coherence=ts_node.get('coherence', 0.0),
                        entropy=ts_node.get('entropy', 0.0)
                    )
                )
                
                # Load frequency domains
                domains_result = session.run("""
                    MATCH (ts:TopologyState {id: $state_id})-[:CONTAINS]->(fd:FrequencyDomain)
                    RETURN fd
                """, state_id=state_id)
                
                for record in domains_result:
                    fd_node = record['fd']
                    domain = FrequencyDomain(
                        id=fd_node['id'],
                        dominant_frequency=fd_node.get('dominantFrequency', 0.0),
                        bandwidth=fd_node.get('bandwidth', 0.0),
                        phase_coherence=fd_node.get('phaseCoherence', 0.0),
                        last_updated=datetime.fromtimestamp(fd_node.get('lastUpdated', 0))
                    )
                    
                    # Load pattern IDs for this domain
                    patterns_result = session.run("""
                        MATCH (p:Pattern)-[:RESONATES_IN]->(fd:FrequencyDomain {id: $domain_id})
                        RETURN p.id AS pattern_id
                    """, domain_id=domain.id)
                    
                    for p_record in patterns_result:
                        domain.pattern_ids.add(p_record['pattern_id'])
                    
                    state.frequency_domains[domain.id] = domain
                
                # Load boundaries
                boundaries_result = session.run("""
                    MATCH (ts:TopologyState {id: $state_id})-[:CONTAINS]->(b:Boundary)
                    RETURN b
                """, state_id=state_id)
                
                for record in boundaries_result:
                    b_node = record['b']
                    boundary = Boundary(
                        id=b_node['id'],
                        sharpness=b_node.get('sharpness', 0.0),
                        permeability=b_node.get('permeability', 0.0),
                        stability=b_node.get('stability', 0.0),
                        dimensionality=b_node.get('dimensionality', 0),
                        last_updated=datetime.fromtimestamp(b_node.get('lastUpdated', 0))
                    )
                    
                    # Load domain IDs for this boundary
                    domains_result = session.run("""
                        MATCH (fd:FrequencyDomain)-[:BOUNDED_BY]->(b:Boundary {id: $boundary_id})
                        RETURN fd.id AS domain_id
                    """, boundary_id=boundary.id)
                    
                    domain_ids = []
                    for d_record in domains_result:
                        domain_ids.append(d_record['domain_id'])
                    
                    if len(domain_ids) == 2:
                        boundary.domain_ids = tuple(domain_ids)
                    
                    state.boundaries[boundary.id] = boundary
                
                # Load resonance points
                points_result = session.run("""
                    MATCH (ts:TopologyState {id: $state_id})-[:CONTAINS]->(r:ResonancePoint)
                    RETURN r
                """, state_id=state_id)
                
                for record in points_result:
                    r_node = record['r']
                    point = ResonancePoint(
                        id=r_node['id'],
                        strength=r_node.get('strength', 0.0),
                        stability=r_node.get('stability', 0.0),
                        attractor_radius=r_node.get('attractorRadius', 0.0),
                        last_updated=datetime.fromtimestamp(r_node.get('lastUpdated', 0))
                    )
                    
                    # Load contributing patterns for this point
                    patterns_result = session.run("""
                        MATCH (p:Pattern)-[c:CONTRIBUTES_TO]->(r:ResonancePoint {id: $point_id})
                        RETURN p.id AS pattern_id, c.weight AS weight
                    """, point_id=point.id)
                    
                    for p_record in patterns_result:
                        point.contributing_pattern_ids[p_record['pattern_id']] = p_record['weight']
                    
                    state.resonance_points[point.id] = point
                
                self.current_state = state
                logger.info(f"Successfully loaded topology state {state_id} from Neo4j")
                return state
                
        except Exception as e:
            logger.error(f"Error loading topology state from Neo4j: {e}")
            return None
    
    def serialize_current_state(self) -> Optional[str]:
        """
        Serialize current state to JSON.
        
        Returns:
            JSON string of current state, or None if no state
        """
        if not self.current_state:
            logger.warning("No current topology state to serialize")
            return None
            
        return self.current_state.to_json()
    
    def load_from_serialized(self, json_str: str) -> Optional[TopologyState]:
        """
        Load state from serialized JSON.
        
        Args:
            json_str: JSON string to deserialize
            
        Returns:
            Loaded topology state, or None if invalid
        """
        try:
            self.current_state = TopologyState.from_json(json_str)
            return self.current_state
        except Exception as e:
            logger.error(f"Error deserializing topology state: {e}")
            return None
    
    def get_topology_diff(self, from_state_id: str, to_state_id: str = None) -> Dict[str, Any]:
        """
        Get difference between two topology states.
        
        Args:
            from_state_id: ID of the first state
            to_state_id: ID of the second state, or None for current state
            
        Returns:
            Dictionary of differences
        """
        # Load the first state
        from_state = None
        for state in self.state_history:
            if state.id == from_state_id:
                from_state = state
                break
                
        if not from_state and self.neo4j_driver:
            from_state = self.load_from_neo4j(state_id=from_state_id)
            
        if not from_state:
            logger.error(f"Could not find topology state {from_state_id}")
            return {}
            
        # Get the second state
        to_state = self.current_state
        if to_state_id:
            for state in self.state_history:
                if state.id == to_state_id:
                    to_state = state
                    break
                    
            if to_state.id != to_state_id and self.neo4j_driver:
                to_state = self.load_from_neo4j(state_id=to_state_id)
                
        if not to_state:
            logger.error(f"Could not find topology state {to_state_id}")
            return {}
            
        # Calculate diff
        return to_state.diff(from_state)
    
    def get_topology_evolution(self, time_period: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get evolution of topology over time.
        
        Args:
            time_period: Dictionary with 'start' and 'end' keys for the analysis period
            
        Returns:
            List of topology states in the period
        """
        if not self.neo4j_driver:
            logger.warning("No Neo4j driver provided, cannot get topology evolution")
            return []
            
        try:
            with self.neo4j_driver.session() as session:
                # Convert time period to timestamps
                start_time = time_period.get('start')
                end_time = time_period.get('end')
                
                if isinstance(start_time, datetime):
                    start_time = start_time.timestamp()
                if isinstance(end_time, datetime):
                    end_time = end_time.timestamp()
                
                # Query topology states in the period
                result = session.run("""
                    MATCH (ts:TopologyState)
                    WHERE ts.timestamp >= $start_time AND ts.timestamp <= $end_time
                    RETURN ts.id AS id, ts.timestamp AS timestamp
                    ORDER BY ts.timestamp ASC
                """, start_time=start_time, end_time=end_time)
                
                # Get state IDs
                state_ids = [record['id'] for record in result]
                
                # Load each state
                states = []
                for state_id in state_ids:
                    state = self.load_from_neo4j(state_id=state_id)
                    if state:
                        states.append({
                            'id': state.id,
                            'timestamp': state.timestamp,
                            'coherence': state.field_metrics.coherence,
                            'entropy': state.field_metrics.entropy,
                            'domain_count': len(state.frequency_domains),
                            'boundary_count': len(state.boundaries),
                            'resonance_count': len(state.resonance_points)
                        })
                
                return states
                
        except Exception as e:
            logger.error(f"Error getting topology evolution: {e}")
            return []
