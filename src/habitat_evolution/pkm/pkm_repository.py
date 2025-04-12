"""
Pattern Knowledge Medium (PKM) Repository for Habitat Evolution.

This module provides a repository for storing, retrieving, and managing PKM files,
which encapsulate patterns detected by the Habitat Evolution system along with
their relationships, metadata, and user attribution.

PKM files serve as a socialized knowledge medium that enables collaborative
pattern-aware knowledge building across users and domains.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

from src.habitat_evolution.infrastructure.persistence.arangodb.arangodb_connection import ArangoDBConnection
from src.habitat_evolution.core.pattern import PatternState, PatternQualityAnalyzer
from src.habitat_evolution.adaptive_core.id.adaptive_id import AdaptiveID

logger = logging.getLogger(__name__)

class PKMFile:
    """
    Pattern Knowledge Medium (PKM) file that encapsulates patterns and their relationships.
    
    A PKM file contains:
    1. A unique identifier
    2. The patterns it encapsulates
    3. Relationships between those patterns
    4. Metadata about the patterns and relationships
    5. User attribution
    6. Creation and modification timestamps
    7. References to original source documents
    """
    
    def __init__(
        self,
        pkm_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        creator_id: Optional[str] = None,
        source_documents: Optional[List[Dict[str, Any]]] = None,
        patterns: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a PKM file.
        
        Args:
            pkm_id: Unique identifier for the PKM file (generated if not provided)
            title: Title of the PKM file
            description: Description of the PKM file
            creator_id: Identifier of the user who created the PKM file
            source_documents: List of source documents referenced by the PKM file
            patterns: List of patterns encapsulated in the PKM file
            relationships: List of relationships between patterns
            metadata: Additional metadata about the PKM file
        """
        self.pkm_id = pkm_id or str(uuid.uuid4())
        self.title = title or f"PKM-{self.pkm_id[:8]}"
        self.description = description or ""
        self.creator_id = creator_id or "system"
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.source_documents = source_documents or []
        self.patterns = patterns or []
        self.relationships = relationships or []
        self.metadata = metadata or {}
        self.version = 1
        self.contributors = set([self.creator_id])
        
        # Create an AdaptiveID for this PKM file
        self.adaptive_id = AdaptiveID(
            base_concept=f"pkm_{self.title.lower().replace(' ', '_')}",
            creator_id=self.creator_id
        )
        
        logger.info(f"Created PKM file: {self.title} (ID: {self.pkm_id})")
    
    def add_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Add a pattern to the PKM file.
        
        Args:
            pattern: Pattern to add
        """
        # Check if the pattern already exists
        for existing_pattern in self.patterns:
            if existing_pattern.get("id") == pattern.get("id"):
                logger.info(f"Pattern {pattern.get('id')} already exists in PKM file")
                return
        
        # Add the pattern
        self.patterns.append(pattern)
        self.updated_at = datetime.now().isoformat()
        self.version += 1
        logger.info(f"Added pattern {pattern.get('id')} to PKM file")
    
    def add_relationship(self, relationship: Dict[str, Any]) -> None:
        """
        Add a relationship to the PKM file.
        
        Args:
            relationship: Relationship to add
        """
        # Check if the relationship already exists
        for existing_relationship in self.relationships:
            if existing_relationship.get("id") == relationship.get("id"):
                logger.info(f"Relationship {relationship.get('id')} already exists in PKM file")
                return
        
        # Add the relationship
        self.relationships.append(relationship)
        self.updated_at = datetime.now().isoformat()
        self.version += 1
        logger.info(f"Added relationship {relationship.get('id')} to PKM file")
    
    def add_contributor(self, contributor_id: str) -> None:
        """
        Add a contributor to the PKM file.
        
        Args:
            contributor_id: Identifier of the contributor
        """
        self.contributors.add(contributor_id)
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Added contributor {contributor_id} to PKM file")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the PKM file to a dictionary.
        
        Returns:
            Dictionary representation of the PKM file
        """
        return {
            "pkm_id": self.pkm_id,
            "title": self.title,
            "description": self.description,
            "creator_id": self.creator_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_documents": self.source_documents,
            "patterns": self.patterns,
            "relationships": self.relationships,
            "metadata": self.metadata,
            "version": self.version,
            "contributors": list(self.contributors),
            "adaptive_id": self.adaptive_id.to_dict() if self.adaptive_id else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PKMFile':
        """
        Create a PKM file from a dictionary.
        
        Args:
            data: Dictionary representation of a PKM file
            
        Returns:
            PKM file instance
        """
        pkm_file = cls(
            pkm_id=data.get("pkm_id"),
            title=data.get("title"),
            description=data.get("description"),
            creator_id=data.get("creator_id"),
            source_documents=data.get("source_documents"),
            patterns=data.get("patterns"),
            relationships=data.get("relationships"),
            metadata=data.get("metadata")
        )
        
        # Set additional attributes
        pkm_file.created_at = data.get("created_at", pkm_file.created_at)
        pkm_file.updated_at = data.get("updated_at", pkm_file.updated_at)
        pkm_file.version = data.get("version", 1)
        pkm_file.contributors = set(data.get("contributors", [pkm_file.creator_id]))
        
        # Create AdaptiveID if not present
        if not pkm_file.adaptive_id and data.get("adaptive_id"):
            pkm_file.adaptive_id = AdaptiveID.from_dict(data.get("adaptive_id"))
        
        return pkm_file
    
    def to_json(self) -> str:
        """
        Convert the PKM file to a JSON string.
        
        Returns:
            JSON string representation of the PKM file
        """
        # Convert set to list for JSON serialization
        data = self.to_dict()
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PKMFile':
        """
        Create a PKM file from a JSON string.
        
        Args:
            json_str: JSON string representation of a PKM file
            
        Returns:
            PKM file instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class PKMRepository:
    """
    Repository for storing, retrieving, and managing PKM files.
    
    This repository provides methods for:
    1. Creating new PKM files
    2. Retrieving PKM files by ID
    3. Searching for PKM files by various criteria
    4. Updating existing PKM files
    5. Managing relationships between PKM files
    6. Exporting and importing PKM files
    """
    
    def __init__(self, arangodb_connection: ArangoDBConnection):
        """
        Initialize a PKM repository.
        
        Args:
            arangodb_connection: Connection to ArangoDB
        """
        self.arangodb = arangodb_connection
        self.collection_name = "pkm_files"
        self.relationship_collection_name = "pkm_relationships"
        
        # Ensure collections exist
        self._ensure_collections_exist()
        
        logger.info("Initialized PKM repository")
    
    def _ensure_collections_exist(self) -> None:
        """Ensure that the necessary collections exist in ArangoDB."""
        # Create PKM files collection if it doesn't exist
        if not self.arangodb.collection_exists(self.collection_name):
            self.arangodb.create_collection(self.collection_name)
            logger.info(f"Created collection: {self.collection_name}")
        
        # Create PKM relationships collection if it doesn't exist
        if not self.arangodb.collection_exists(self.relationship_collection_name):
            self.arangodb.create_collection(self.relationship_collection_name, edge=True)
            logger.info(f"Created edge collection: {self.relationship_collection_name}")
    
    def create_pkm_file(self, pkm_file: PKMFile) -> str:
        """
        Create a new PKM file in the repository.
        
        Args:
            pkm_file: PKM file to create
            
        Returns:
            ID of the created PKM file
        """
        # Convert PKM file to dictionary
        pkm_dict = pkm_file.to_dict()
        
        # Add _key field for ArangoDB
        pkm_dict["_key"] = pkm_file.pkm_id
        
        # Create document in ArangoDB
        result = self.arangodb.create_document(self.collection_name, pkm_dict)
        
        logger.info(f"Created PKM file in repository: {pkm_file.title} (ID: {pkm_file.pkm_id})")
        
        return pkm_file.pkm_id
    
    def get_pkm_file(self, pkm_id: str) -> Optional[PKMFile]:
        """
        Retrieve a PKM file by ID.
        
        Args:
            pkm_id: ID of the PKM file to retrieve
            
        Returns:
            PKM file if found, None otherwise
        """
        # Get document from ArangoDB
        document = self.arangodb.get_document(self.collection_name, pkm_id)
        
        if not document:
            logger.warning(f"PKM file not found: {pkm_id}")
            return None
        
        # Convert document to PKM file
        pkm_file = PKMFile.from_dict(document)
        
        logger.info(f"Retrieved PKM file: {pkm_file.title} (ID: {pkm_file.pkm_id})")
        
        return pkm_file
    
    def update_pkm_file(self, pkm_file: PKMFile) -> bool:
        """
        Update an existing PKM file in the repository.
        
        Args:
            pkm_file: PKM file to update
            
        Returns:
            True if the update was successful, False otherwise
        """
        # Convert PKM file to dictionary
        pkm_dict = pkm_file.to_dict()
        
        # Add _key field for ArangoDB
        pkm_dict["_key"] = pkm_file.pkm_id
        
        # Update document in ArangoDB
        result = self.arangodb.update_document(self.collection_name, pkm_file.pkm_id, pkm_dict)
        
        if result:
            logger.info(f"Updated PKM file in repository: {pkm_file.title} (ID: {pkm_file.pkm_id})")
            return True
        else:
            logger.warning(f"Failed to update PKM file: {pkm_file.pkm_id}")
            return False
    
    def delete_pkm_file(self, pkm_id: str) -> bool:
        """
        Delete a PKM file from the repository.
        
        Args:
            pkm_id: ID of the PKM file to delete
            
        Returns:
            True if the deletion was successful, False otherwise
        """
        # Delete document from ArangoDB
        result = self.arangodb.delete_document(self.collection_name, pkm_id)
        
        if result:
            logger.info(f"Deleted PKM file from repository: {pkm_id}")
            return True
        else:
            logger.warning(f"Failed to delete PKM file: {pkm_id}")
            return False
    
    def search_pkm_files(self, query: Dict[str, Any]) -> List[PKMFile]:
        """
        Search for PKM files by various criteria.
        
        Args:
            query: Dictionary of search criteria
            
        Returns:
            List of matching PKM files
        """
        # Convert query to AQL filter
        filter_conditions = []
        for key, value in query.items():
            if isinstance(value, str):
                filter_conditions.append(f'doc.{key} == "{value}"')
            else:
                filter_conditions.append(f'doc.{key} == {value}')
        
        filter_str = " && ".join(filter_conditions)
        
        # Create AQL query
        aql_query = f"""
        FOR doc IN {self.collection_name}
        FILTER {filter_str}
        RETURN doc
        """
        
        # Execute query
        results = self.arangodb.execute_query(aql_query)
        
        # Convert results to PKM files
        pkm_files = [PKMFile.from_dict(doc) for doc in results]
        
        logger.info(f"Found {len(pkm_files)} PKM files matching query: {query}")
        
        return pkm_files
    
    def create_pkm_relationship(self, from_pkm_id: str, to_pkm_id: str, relationship_type: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create a relationship between two PKM files.
        
        Args:
            from_pkm_id: ID of the source PKM file
            to_pkm_id: ID of the target PKM file
            relationship_type: Type of relationship
            metadata: Additional metadata about the relationship
            
        Returns:
            ID of the created relationship
        """
        # Create relationship document
        relationship_id = str(uuid.uuid4())
        relationship = {
            "_key": relationship_id,
            "_from": f"{self.collection_name}/{from_pkm_id}",
            "_to": f"{self.collection_name}/{to_pkm_id}",
            "type": relationship_type,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Create document in ArangoDB
        result = self.arangodb.create_document(self.relationship_collection_name, relationship)
        
        logger.info(f"Created PKM relationship: {from_pkm_id} -> {to_pkm_id} ({relationship_type})")
        
        return relationship_id
    
    def get_related_pkm_files(self, pkm_id: str, relationship_type: Optional[str] = None) -> List[Tuple[PKMFile, str]]:
        """
        Get PKM files related to a given PKM file.
        
        Args:
            pkm_id: ID of the PKM file
            relationship_type: Optional type of relationship to filter by
            
        Returns:
            List of tuples containing related PKM files and their relationship types
        """
        # Create filter for relationship type
        type_filter = f'FILTER edge.type == "{relationship_type}"' if relationship_type else ""
        
        # Create AQL query for outgoing relationships
        aql_query = f"""
        FOR vertex, edge IN 1..1 OUTBOUND "{self.collection_name}/{pkm_id}" {self.relationship_collection_name}
        {type_filter}
        RETURN {{pkm_file: vertex, relationship_type: edge.type}}
        """
        
        # Execute query
        results = self.arangodb.execute_query(aql_query)
        
        # Convert results to PKM files and relationship types
        related_pkm_files = [(PKMFile.from_dict(result["pkm_file"]), result["relationship_type"]) for result in results]
        
        logger.info(f"Found {len(related_pkm_files)} PKM files related to {pkm_id}")
        
        return related_pkm_files
    
    def export_pkm_file(self, pkm_id: str, file_path: str) -> bool:
        """
        Export a PKM file to a JSON file.
        
        Args:
            pkm_id: ID of the PKM file to export
            file_path: Path to the output file
            
        Returns:
            True if the export was successful, False otherwise
        """
        # Get PKM file
        pkm_file = self.get_pkm_file(pkm_id)
        
        if not pkm_file:
            logger.warning(f"PKM file not found: {pkm_id}")
            return False
        
        # Convert PKM file to JSON
        json_str = pkm_file.to_json()
        
        # Write JSON to file
        try:
            with open(file_path, "w") as f:
                f.write(json_str)
            
            logger.info(f"Exported PKM file to {file_path}: {pkm_file.title} (ID: {pkm_file.pkm_id})")
            return True
        except Exception as e:
            logger.error(f"Error exporting PKM file: {e}")
            return False
    
    def import_pkm_file(self, file_path: str) -> Optional[str]:
        """
        Import a PKM file from a JSON file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            ID of the imported PKM file if successful, None otherwise
        """
        # Read JSON from file
        try:
            with open(file_path, "r") as f:
                json_str = f.read()
            
            # Convert JSON to PKM file
            pkm_file = PKMFile.from_json(json_str)
            
            # Create PKM file in repository
            pkm_id = self.create_pkm_file(pkm_file)
            
            logger.info(f"Imported PKM file from {file_path}: {pkm_file.title} (ID: {pkm_id})")
            return pkm_id
        except Exception as e:
            logger.error(f"Error importing PKM file: {e}")
            return None


def create_pkm_from_claude_response(
    response: Dict[str, Any],
    query: str,
    source_documents: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    creator_id: Optional[str] = None
) -> PKMFile:
    """
    Create a PKM file from a Claude API response.
    
    Args:
        response: Claude API response
        query: Original query sent to Claude
        source_documents: Source documents referenced by the response
        patterns: Patterns used to generate the response
        creator_id: Identifier of the creator
        
    Returns:
        Created PKM file
    """
    # Extract response text
    response_text = response.get("response", "")
    
    # Create title from query (don't truncate for testing)
    title = f"PKM: {query}"
    
    # Create description from response
    description = response_text[:200] + "..." if len(response_text) > 200 else response_text
    
    # Create metadata
    metadata = {
        "query": query,
        "response_id": response.get("query_id"),
        "timestamp": response.get("timestamp"),
        "model": response.get("model"),
        "tokens_used": response.get("tokens_used")
    }
    
    # Create PKM file
    pkm_file = PKMFile(
        title=title,
        description=description,
        creator_id=creator_id or "system",
        source_documents=source_documents,
        patterns=patterns,
        metadata=metadata
    )
    
    # Add response as a pattern
    response_pattern = {
        "id": str(uuid.uuid4()),
        "type": "claude_response",
        "content": response_text,
        "metadata": metadata,
        "created_at": datetime.now().isoformat()
    }
    
    pkm_file.add_pattern(response_pattern)
    
    logger.info(f"Created PKM file from Claude response: {pkm_file.title} (ID: {pkm_file.pkm_id})")
    
    return pkm_file
