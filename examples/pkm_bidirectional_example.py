"""
Example demonstrating the PKM Bidirectional Integration.

This example shows how to use the PKM Bidirectional Integration to:
1. Create a pattern-driven knowledge repository
2. Generate queries from patterns
3. Capture knowledge in PKM files
4. Create relationships between patterns and knowledge

To run this example, you need to install the Habitat Evolution package in development mode:
    pip install -e .

Or use the PYTHONPATH environment variable:
    PYTHONPATH=$PYTHONPATH:$(pwd) python examples/pkm_bidirectional_example.py
"""

import logging
import json
import uuid
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

# Add the project root to the Python path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import from the package
try:
    from habitat_evolution.pkm import (
        PKMFile, 
        PKMRepository, 
        create_pkm_from_claude_response,
        create_pkm_repository,
        create_pkm_bidirectional_integration
    )
    PACKAGE_IMPORTS_WORK = True
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported from habitat_evolution package")
except ImportError:
    # Fall back to direct imports
    try:
        from src.habitat_evolution.pkm.pkm_repository import PKMFile, PKMRepository, create_pkm_from_claude_response
        from src.habitat_evolution.pkm.pkm_bidirectional_integration import PKMBidirectionalIntegration
        from src.habitat_evolution.pkm.pkm_factory import create_pkm_repository, create_pkm_bidirectional_integration
        PACKAGE_IMPORTS_WORK = True
        logger = logging.getLogger(__name__)
        logger.info("Successfully imported from src.habitat_evolution package")
    except ImportError:
        PACKAGE_IMPORTS_WORK = False
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.warning("Could not import from habitat_evolution package. Using standalone implementation.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple classes to demonstrate the concept
class Pattern:
    """A pattern detected in the system."""
    
    def __init__(self, pattern_id: str, pattern_type: str, content: str, quality: float = 0.0, metadata: Dict[str, Any] = None):
        self.id = pattern_id
        self.type = pattern_type
        self.content = content
        self.quality = quality
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the pattern to a dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "quality": self.quality,
            "metadata": self.metadata,
            "created_at": self.created_at
        }

class PKMFile:
    """A Pattern Knowledge Medium file."""
    
    def __init__(self, pkm_id: str, title: str, content: str, patterns: List[Pattern], creator_id: str = "system"):
        self.id = pkm_id
        self.title = title
        self.content = content
        self.patterns = patterns
        self.creator_id = creator_id
        self.created_at = datetime.now().isoformat()
        self.relationships = []
    
    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to the PKM file."""
        self.patterns.append(pattern)
    
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str) -> None:
        """Add a relationship to the PKM file."""
        self.relationships.append({
            "source_id": source_id,
            "target_id": target_id,
            "type": relationship_type
        })

class PatternRepository:
    """A repository for patterns."""
    
    def __init__(self):
        self.patterns = {}
    
    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to the repository."""
        self.patterns[pattern.id] = pattern
        logger.info(f"Added pattern to repository: {pattern.id}")
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern from the repository."""
        return self.patterns.get(pattern_id)
    
    def get_patterns(self, filter_criteria: Dict[str, Any] = None) -> List[Pattern]:
        """Get patterns from the repository."""
        if not filter_criteria:
            return list(self.patterns.values())
        
        # Simple filtering
        result = []
        for pattern in self.patterns.values():
            match = True
            for key, value in filter_criteria.items():
                if getattr(pattern, key, None) != value:
                    match = False
                    break
            if match:
                result.append(pattern)
        return result

class PKMRepository:
    """A repository for PKM files."""
    
    def __init__(self):
        self.pkm_files = {}
    
    def add_pkm_file(self, pkm_file: PKMFile) -> None:
        """Add a PKM file to the repository."""
        self.pkm_files[pkm_file.id] = pkm_file
        logger.info(f"Added PKM file to repository: {pkm_file.id}")
    
    def get_pkm_file(self, pkm_id: str) -> Optional[PKMFile]:
        """Get a PKM file from the repository."""
        return self.pkm_files.get(pkm_id)
    
    def create_relationship(self, source_id: str, target_id: str, relationship_type: str) -> None:
        """Create a relationship between PKM files."""
        source = self.get_pkm_file(source_id)
        target = self.get_pkm_file(target_id)
        
        if source and target:
            source.add_relationship(source_id, target_id, relationship_type)
            logger.info(f"Created relationship: {source_id} -> {target_id} ({relationship_type})")

class EventService:
    """A simple event service."""
    
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish an event."""
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
        logger.info(f"Published event: {event_type}")

class BidirectionalFlowService:
    """A service for bidirectional flow between patterns and knowledge."""
    
    def __init__(self, pattern_repository: PatternRepository, pkm_repository: PKMRepository, event_service: EventService, max_depth: int = 3):
        self.pattern_repository = pattern_repository
        self.pkm_repository = pkm_repository
        self.event_service = event_service
        self.max_depth = max_depth
        self.current_depth = 0
        
        # Register event handlers
        self.event_service.subscribe("pattern.detected", self._handle_pattern_event)
        self.event_service.subscribe("pkm.created", self._handle_pkm_created_event)
    
    def _handle_pattern_event(self, event_data: Dict[str, Any]) -> None:
        """Handle a pattern event."""
        # Check if we've reached the maximum depth
        if self.current_depth >= self.max_depth:
            logger.info(f"Reached maximum depth ({self.max_depth}), stopping pattern processing")
            return
            
        pattern_id = event_data.get("id")
        pattern = self.pattern_repository.get_pattern(pattern_id)
        
        if pattern and pattern.quality > 0.7:
            # Generate a query from the pattern
            query = self._generate_query_from_pattern(pattern)
            
            if query:
                logger.info(f"Generated query from pattern {pattern_id}: {query}")
                
                # Create a PKM file from the query
                pkm_id = str(uuid.uuid4())
                pkm_file = PKMFile(
                    pkm_id=pkm_id,
                    title=f"PKM: {query}",
                    content=f"This PKM file was generated from the pattern: {pattern.content}",
                    patterns=[pattern],
                    creator_id="system"
                )
                
                # Add a response pattern
                response_pattern = Pattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="claude_response",
                    content=f"Mock response to: {query}",
                    quality=0.9,
                    metadata={"source": "claude"}
                )
                pkm_file.add_pattern(response_pattern)
                
                # Add the PKM file to the repository
                self.pkm_repository.add_pkm_file(pkm_file)
                
                # Increment the depth before publishing the PKM created event
                self.current_depth += 1
                
                # Publish a PKM created event
                self.event_service.publish("pkm.created", {
                    "pkm_id": pkm_id,
                    "pattern_id": pattern_id,
                    "query": query,
                    "depth": self.current_depth
                })
    
    def _handle_pkm_created_event(self, event_data: Dict[str, Any]) -> None:
        """Handle a PKM created event."""
        # Check if we've reached the maximum depth
        depth = event_data.get("depth", 0)
        if depth >= self.max_depth:
            logger.info(f"Reached maximum depth ({self.max_depth}), stopping PKM processing")
            # Reset depth for future pattern chains
            self.current_depth = 0
            return
            
        pkm_id = event_data.get("pkm_id")
        pkm_file = self.pkm_repository.get_pkm_file(pkm_id)
        
        if pkm_file:
            # Extract patterns from the PKM file
            for pattern in pkm_file.patterns:
                if pattern.type == "claude_response":
                    # Create a new pattern from the response
                    new_pattern = Pattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="semantic",
                        content=f"Insight from {pkm_file.title}: {pattern.content[:50]}...",
                        quality=0.8,
                        metadata={"source": pkm_id}
                    )
                    
                    # Add the pattern to the repository
                    self.pattern_repository.add_pattern(new_pattern)
                    
                    # Publish a pattern detected event
                    self.event_service.publish("pattern.detected", new_pattern.to_dict())
    
    def _generate_query_from_pattern(self, pattern: Pattern) -> str:
        """Generate a query from a pattern."""
        if pattern.type == "semantic":
            return f"What are the implications of {pattern.content}?"
        elif pattern.type == "statistical":
            return f"What does the statistical pattern '{pattern.content}' indicate?"
        else:
            return f"What insights can be derived from the pattern: {pattern.content}?"
    
    def publish_pattern(self, pattern: Pattern) -> None:
        """Publish a pattern."""
        # Add the pattern to the repository
        self.pattern_repository.add_pattern(pattern)
        
        # Publish a pattern detected event
        self.event_service.publish("pattern.detected", pattern.to_dict())

def main():
    """Run the PKM bidirectional example."""
    # If package imports work, use the actual implementation
    if PACKAGE_IMPORTS_WORK:
        try:
            # Use the factory method to create the PKM bidirectional integration
            logger.info("Creating PKM bidirectional integration using package implementation")
            pkm_bidirectional = create_pkm_bidirectional_integration(
                db_config={
                    "host": "localhost",
                    "port": 8529,
                    "username": "root",
                    "password": "habitat",
                    "database_name": "habitat_evolution"
                },
                creator_id="example_user"
            )
            
            # Use the actual implementation
            run_with_package_implementation(pkm_bidirectional)
            return
        except Exception as e:
            logger.error(f"Error using package implementation: {e}")
            logger.info("Falling back to standalone implementation")
    
    # Create repositories and services for standalone implementation
    pattern_repository = PatternRepository()
    pkm_repository = PKMRepository()
    event_service = EventService()
    
    # Create bidirectional flow service with a maximum depth of 3
    bidirectional_flow = BidirectionalFlowService(
        pattern_repository=pattern_repository,
        pkm_repository=pkm_repository,
        event_service=event_service,
        max_depth=3
    )
    
    # Run with standalone implementation
    run_with_standalone_implementation(bidirectional_flow, pattern_repository, pkm_repository)
    
def run_with_package_implementation(pkm_bidirectional):
    """Run the example using the actual package implementation."""
    # Sample patterns for Boston Harbor
    boston_harbor_patterns = [
        {
            "id": "pattern-1",
            "type": "semantic",
            "content": "Sea level rise in Boston Harbor",
            "quality": 0.9,
            "metadata": {
                "confidence": 0.9,
                "source": "climate_risk_assessment_2023.pdf"
            },
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "pattern-2",
            "type": "statistical",
            "content": "9-21 inches of sea level rise by 2050",
            "quality": 0.85,
            "metadata": {
                "confidence": 0.85,
                "source": "boston_harbor_measurements.csv"
            },
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "pattern-3",
            "type": "semantic",
            "content": "Infrastructure vulnerability in coastal areas",
            "quality": 0.8,
            "metadata": {
                "confidence": 0.8,
                "source": "infrastructure_vulnerability_report.pdf"
            },
            "created_at": datetime.now().isoformat()
        }
    ]
    
    # Example 1: Generate a query from a pattern
    pattern = boston_harbor_patterns[0]
    query = pkm_bidirectional.generate_query_from_patterns([pattern])
    logger.info(f"Generated query from pattern: {query}")
    
    # Example 2: Process a query with patterns as context
    pkm_id = pkm_bidirectional.process_query_with_patterns(
        query="What are the impacts of sea level rise on Boston Harbor infrastructure by 2050?",
        patterns=boston_harbor_patterns
    )
    
    if pkm_id:
        logger.info(f"Created PKM file with ID: {pkm_id}")
        
        # Get the PKM file
        pkm_file = pkm_bidirectional.pkm_repository.get_pkm_file(pkm_id)
        
        if pkm_file:
            logger.info(f"PKM file title: {pkm_file.title}")
            
            # Find the Claude response pattern
            response_pattern = None
            for pattern in pkm_file.patterns:
                if pattern.get("type") == "claude_response":
                    response_pattern = pattern
                    break
            
            if response_pattern:
                logger.info(f"Response content: {response_pattern.get('content')[:200]}...")
    
    # Example 3: Create a relationship between patterns
    relationship = {
        "source_id": boston_harbor_patterns[0]["id"],
        "target_id": boston_harbor_patterns[1]["id"],
        "type": "correlation",
        "properties": {
            "strength": 0.8,
            "description": "Correlation between sea level rise and measurements"
        }
    }
    
    # Publish the relationship
    pkm_bidirectional.bidirectional_flow_service.publish_relationship(relationship)
    logger.info(f"Published relationship: {relationship['source_id']} -> {relationship['target_id']}")
    
    # Example 4: Demonstrate bidirectional flow
    # First, publish patterns to the bidirectional flow service
    for pattern in boston_harbor_patterns:
        pkm_bidirectional.bidirectional_flow_service.publish_pattern(pattern)
        logger.info(f"Published pattern: {pattern['id']}")
    
    logger.info("PKM bidirectional example complete (package implementation)")

def run_with_standalone_implementation(bidirectional_flow, pattern_repository, pkm_repository):
    """Run the example using the standalone implementation."""
    # Sample patterns for Boston Harbor
    boston_harbor_patterns = [
        Pattern(
            pattern_id="pattern-1",
            pattern_type="semantic",
            content="Sea level rise in Boston Harbor",
            quality=0.9,
            metadata={
                "confidence": 0.9,
                "source": "climate_risk_assessment_2023.pdf"
            }
        ),
        Pattern(
            pattern_id="pattern-2",
            pattern_type="statistical",
            content="9-21 inches of sea level rise by 2050",
            quality=0.85,
            metadata={
                "confidence": 0.85,
                "source": "boston_harbor_measurements.csv"
            }
        ),
        Pattern(
            pattern_id="pattern-3",
            pattern_type="semantic",
            content="Infrastructure vulnerability in coastal areas",
            quality=0.8,
            metadata={
                "confidence": 0.8,
                "source": "infrastructure_vulnerability_report.pdf"
            }
        )
    ]
    
    # Publish patterns to demonstrate bidirectional flow
    logger.info("Publishing patterns to demonstrate bidirectional flow")
    for pattern in boston_harbor_patterns:
        bidirectional_flow.publish_pattern(pattern)
    
    # The bidirectional flow will automatically:
    # 1. Detect patterns
    # 2. Generate queries from patterns
    # 3. Create PKM files from queries
    # 4. Extract new patterns from PKM files
    # 5. Continue the cycle
    
    # Show the PKM files created
    logger.info("\nPKM files created:")
    for pkm_id, pkm_file in pkm_repository.pkm_files.items():
        logger.info(f"  - {pkm_file.title} (ID: {pkm_id})")
        logger.info(f"    Content: {pkm_file.content}")
        logger.info(f"    Patterns: {len(pkm_file.patterns)}")
        logger.info(f"    Relationships: {len(pkm_file.relationships)}")
    
    # Show the patterns detected
    logger.info("\nPatterns in repository:")
    for pattern_id, pattern in pattern_repository.patterns.items():
        logger.info(f"  - {pattern.type}: {pattern.content} (ID: {pattern_id})")
    
    logger.info("\nPKM bidirectional example complete (standalone implementation)")

if __name__ == "__main__":
    main()
