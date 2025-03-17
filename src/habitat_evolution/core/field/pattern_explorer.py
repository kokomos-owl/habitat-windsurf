"""Pattern Explorer for Vector + Tonic-Harmonic patterns.

This module provides a Pattern Explorer that integrates the Field Navigator with
the Neo4j pattern schema to enable seamless pattern exploration and visualization.
It leverages eigenspace navigation capabilities and dimensional resonance detection
to provide a rich exploration experience.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import json
import logging
from pathlib import Path

from habitat_evolution.core.field.field_navigator import FieldNavigator
from habitat_evolution.core.field.neo4j_pattern_schema import Neo4jPatternSchema
from habitat_evolution.core.field.cypher_query_library import CypherQueryExecutor


class PatternExplorer:
    """Pattern Explorer for Vector + Tonic-Harmonic patterns.
    
    This class integrates the Field Navigator with the Neo4j pattern schema to enable
    seamless pattern exploration and visualization. It provides methods for exploring
    patterns, dimensions, communities, and their relationships.
    """
    
    def __init__(self, field_navigator: Optional[FieldNavigator] = None,
                neo4j_config: Optional[Dict[str, str]] = None):
        """Initialize the Pattern Explorer.
        
        Args:
            field_navigator: Field Navigator instance (optional)
            neo4j_config: Neo4j configuration (optional)
        """
        self.navigator = field_navigator or FieldNavigator()
        self.neo4j_schema = None
        self.query_executor = None
        
        if neo4j_config:
            self._setup_neo4j(neo4j_config)
            
        self.logger = logging.getLogger(__name__)
    
    def _setup_neo4j(self, config: Dict[str, str]):
        """Set up Neo4j connection and schema.
        
        Args:
            config: Neo4j configuration with uri, username, password, and database
        """
        try:
            uri = config.get("uri", "bolt://localhost:7687")
            username = config.get("username", "neo4j")
            password = config.get("password", "password")
            database = config.get("database", "neo4j")
            
            self.neo4j_schema = Neo4jPatternSchema(uri, username, password, database)
            self.query_executor = CypherQueryExecutor(uri, username, password, database)
            
            # Create schema if it doesn't exist
            self.neo4j_schema.create_schema()
            self.logger.info("Neo4j schema created successfully")
        except Exception as e:
            self.logger.error(f"Failed to set up Neo4j: {str(e)}")
            self.neo4j_schema = None
            self.query_executor = None
