"""
Harmonic Climate Risk Data Processor

This module processes climate risk data through the harmonic I/O system,
allowing for natural pattern evolution without preset transformations and actants.

It integrates with the harmonic I/O service to ensure that data processing
operations don't disrupt the natural evolution of eigenspaces and pattern detection.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional, Generator, Tuple
from datetime import datetime
import uuid
from pathlib import Path

from ..adaptive_core.io.harmonic_io_service import HarmonicIOService, OperationType
from ..field.harmonic_field_io_bridge import HarmonicFieldIOBridge
from ..adaptive_core.transformation.actant_journey_tracker import ActantJourney, ActantJourneyPoint, DomainTransition
from ..adaptive_core.id.adaptive_id import AdaptiveID
from ..field.field_navigator import FieldNavigator
from ..field.semantic_boundary_detector import SemanticBoundaryDetector

logger = logging.getLogger(__name__)


class HarmonicClimateProcessor:
    """
    Process climate risk data through the harmonic I/O system.
    
    This processor allows climate risk data to flow through the system
    without preset transformations and actants, enabling natural pattern
    evolution guided by the harmonic I/O service.
    """
    
    def __init__(self, 
                 io_service: HarmonicIOService,
                 data_dir: str,
                 field_navigator: Optional[FieldNavigator] = None,
                 boundary_detector: Optional[SemanticBoundaryDetector] = None):
        """
        Initialize the harmonic climate processor.
        
        Args:
            io_service: Harmonic I/O service to use for scheduling
            data_dir: Directory containing climate risk data
            field_navigator: Optional field navigator for field integration
            boundary_detector: Optional boundary detector for field integration
        """
        self.io_service = io_service
        self.data_dir = Path(data_dir)
        self.field_navigator = field_navigator
        self.boundary_detector = boundary_detector
        
        # Create field I/O bridge if field components are provided
        if field_navigator or boundary_detector:
            self.field_bridge = HarmonicFieldIOBridge(io_service)
            
            if field_navigator:
                self.field_bridge.register_with_field_navigator(field_navigator)
                
            if boundary_detector:
                self.field_bridge.register_with_semantic_boundary_detector(boundary_detector)
        
        # Track discovered entities
        self.discovered_entities = set()
        self.discovered_domains = set()
        self.discovered_relationships = set()
        
        # Track processing metrics
        self.processing_metrics = {
            "files_processed": 0,
            "entities_discovered": 0,
            "domains_discovered": 0,
            "relationships_discovered": 0,
            "start_time": None,
            "end_time": None
        }
    
    def process_data(self):
        """
        Process all climate risk data in the data directory.
        
        This method discovers and processes all relevant data files
        in the data directory, allowing patterns to emerge naturally
        without preset transformations.
        
        Returns:
            Processing metrics dictionary
        """
        self.processing_metrics["start_time"] = datetime.now().isoformat()
        
        # Discover data files
        data_files = self._discover_data_files()
        logger.info(f"Discovered {len(data_files)} data files to process")
        
        # Process each file
        for file_path in data_files:
            self._process_file(file_path)
            self.processing_metrics["files_processed"] += 1
            
        self.processing_metrics["end_time"] = datetime.now().isoformat()
        self.processing_metrics["entities_discovered"] = len(self.discovered_entities)
        self.processing_metrics["domains_discovered"] = len(self.discovered_domains)
        self.processing_metrics["relationships_discovered"] = len(self.discovered_relationships)
        
        return self.processing_metrics
    
    def _discover_data_files(self) -> List[Path]:
        """
        Discover climate risk data files in the data directory.
        
        Returns:
            List of paths to data files
        """
        data_files = []
        
        # Walk through the data directory
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = Path(root) / file
                
                # Check if file is a data file we can process
                if self._is_processable_file(file_path):
                    data_files.append(file_path)
        
        return data_files
    
    def _is_processable_file(self, file_path: Path) -> bool:
        """
        Check if a file is processable.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is processable, False otherwise
        """
        # Check file extension
        extension = file_path.suffix.lower()
        
        if extension in ['.csv', '.json', '.jsonl', '.txt']:
            return True
            
        return False
    
    def _process_file(self, file_path: Path):
        """
        Process a single data file.
        
        Args:
            file_path: Path to the data file
        """
        logger.info(f"Processing file: {file_path}")
        
        # Determine file type and process accordingly
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            self._process_csv_file(file_path)
        elif extension in ['.json', '.jsonl']:
            self._process_json_file(file_path)
        elif extension == '.txt':
            self._process_text_file(file_path)
    
    def _process_csv_file(self, file_path: Path):
        """
        Process a CSV data file.
        
        Args:
            file_path: Path to the CSV file
        """
        # Determine domain from file path
        domain_id = self._extract_domain_from_path(file_path)
        self.discovered_domains.add(domain_id)
        
        # Read CSV file
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Process each row
            for row in reader:
                # Schedule processing through harmonic I/O
                self._schedule_data_processing(domain_id, row)
    
    def _process_json_file(self, file_path: Path):
        """
        Process a JSON data file.
        
        Args:
            file_path: Path to the JSON file
        """
        # Determine domain from file path
        domain_id = self._extract_domain_from_path(file_path)
        self.discovered_domains.add(domain_id)
        
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if file is JSON Lines format
            if file_path.suffix.lower() == '.jsonl':
                # Process each line as a separate JSON object
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self._schedule_data_processing(domain_id, data)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON line in {file_path}")
            else:
                # Process as a single JSON object or array
                try:
                    data = json.load(f)
                    
                    # Handle both single objects and arrays
                    if isinstance(data, list):
                        for item in data:
                            self._schedule_data_processing(domain_id, item)
                    else:
                        self._schedule_data_processing(domain_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON file {file_path}")
    
    def _process_text_file(self, file_path: Path):
        """
        Process a text data file.
        
        Args:
            file_path: Path to the text file
        """
        # Determine domain from file path
        domain_id = self._extract_domain_from_path(file_path)
        self.discovered_domains.add(domain_id)
        
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
            # Process text as a single data item
            data = {
                "text": text,
                "file_name": file_path.name,
                "timestamp": datetime.now().isoformat()
            }
            
            self._schedule_data_processing(domain_id, data)
    
    def _extract_domain_from_path(self, file_path: Path) -> str:
        """
        Extract domain ID from file path.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Domain ID string
        """
        # Use parent directory name as domain
        domain = file_path.parent.name
        
        # If parent is the data directory itself, use file name without extension
        if domain == self.data_dir.name:
            domain = file_path.stem
            
        return domain
    
    def _schedule_data_processing(self, domain_id: str, data: Dict[str, Any]):
        """
        Schedule data processing through harmonic I/O.
        
        Args:
            domain_id: Domain ID
            data: Data dictionary
        """
        # Extract entity ID if possible
        entity_id = self._extract_entity_id(data)
        
        if entity_id:
            self.discovered_entities.add(entity_id)
            
            # Create data context for harmonic scheduling
            data_context = {
                "domain_id": domain_id,
                "entity_id": entity_id,
                "data_type": "climate_risk",
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "stability": self._calculate_data_stability(data)
            }
            
            # Schedule processing operation
            self.io_service.schedule_operation(
                OperationType.PROCESS.value,
                self,
                "_process_data_item",
                (domain_id, entity_id, data),
                {},
                data_context
            )
    
    def _extract_entity_id(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Extract entity ID from data.
        
        Args:
            data: Data dictionary
            
        Returns:
            Entity ID string, or None if not found
        """
        # Try various common ID fields
        for key in ["id", "ID", "entity_id", "name", "Name", "identifier"]:
            if key in data and data[key]:
                return str(data[key])
                
        # If no ID field found, generate a UUID
        return str(uuid.uuid4())
    
    def _calculate_data_stability(self, data: Dict[str, Any]) -> float:
        """
        Calculate stability metric for data.
        
        This method estimates how stable/reliable the data is
        based on its completeness, consistency, etc.
        
        Args:
            data: Data dictionary
            
        Returns:
            Stability value (0.0 to 1.0)
        """
        # Start with base stability
        stability = 0.5
        
        # Adjust based on data completeness
        if data:
            # Count non-empty fields
            non_empty = sum(1 for v in data.values() if v)
            completeness = non_empty / max(1, len(data))
            
            # Adjust stability based on completeness
            stability += 0.3 * completeness
            
        # Adjust based on presence of timestamp
        if "timestamp" in data or "date" in data or "time" in data:
            stability += 0.1
            
        # Adjust based on presence of confidence/certainty fields
        if any(key in data for key in ["confidence", "certainty", "probability", "likelihood"]):
            confidence_value = next((data[key] for key in ["confidence", "certainty", "probability", "likelihood"] 
                                   if key in data and isinstance(data[key], (int, float))), None)
            
            if confidence_value is not None:
                # Normalize to 0-1 range if needed
                if confidence_value > 1:
                    confidence_value = confidence_value / 100.0
                    
                stability += 0.1 * confidence_value
        
        # Ensure stability is in valid range
        return max(0.1, min(0.9, stability))
    
    def _process_data_item(self, domain_id: str, entity_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single data item.
        
        This method is called by the harmonic I/O service when
        a data processing operation is executed.
        
        Args:
            domain_id: Domain ID
            entity_id: Entity ID
            data: Data dictionary
            
        Returns:
            Processing result dictionary
        """
        # Extract relationships from data
        relationships = self._extract_relationships(domain_id, entity_id, data)
        
        # Create adaptive ID for entity if not already exists
        adaptive_id = self._get_or_create_adaptive_id(domain_id, entity_id, data)
        
        # Update adaptive ID with new data
        self._update_adaptive_id(adaptive_id, domain_id, data)
        
        # Return processing result
        return {
            "domain_id": domain_id,
            "entity_id": entity_id,
            "relationships": relationships,
            "adaptive_id": adaptive_id.to_dict() if adaptive_id else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_relationships(self, domain_id: str, entity_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relationships from data.
        
        Args:
            domain_id: Domain ID
            entity_id: Entity ID
            data: Data dictionary
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Look for references to other entities
        for key, value in data.items():
            # Skip non-string values and common fields
            if not isinstance(value, str) or key in ["id", "ID", "timestamp", "date", "time"]:
                continue
                
            # Check if value looks like a reference to another entity
            if len(value) > 3 and "_" in value:
                target_entity_id = value
                relationship_type = key
                
                # Record relationship
                relationship = {
                    "id": str(uuid.uuid4()),
                    "source_domain_id": domain_id,
                    "source_entity_id": entity_id,
                    "target_entity_id": target_entity_id,
                    "relationship_type": relationship_type,
                    "timestamp": datetime.now().isoformat()
                }
                
                relationships.append(relationship)
                self.discovered_relationships.add((entity_id, relationship_type, target_entity_id))
        
        return relationships
    
    def _get_or_create_adaptive_id(self, domain_id: str, entity_id: str, data: Dict[str, Any]) -> AdaptiveID:
        """
        Get or create adaptive ID for entity.
        
        Args:
            domain_id: Domain ID
            entity_id: Entity ID
            data: Data dictionary
            
        Returns:
            AdaptiveID instance
        """
        # In a real implementation, we would check if an adaptive ID already exists
        # For this example, we'll create a new one
        
        # Determine base concept from data
        base_concept = entity_id
        
        # Check for name fields that might provide a better base concept
        for key in ["name", "Name", "title", "Title"]:
            if key in data and isinstance(data[key], str):
                base_concept = data[key]
                break
        
        # Create adaptive ID
        adaptive_id = AdaptiveID(
            base_concept=base_concept,
            creator_id="harmonic_climate_processor",
            weight=1.0,
            confidence=self._calculate_data_stability(data),
            uncertainty=0.2
        )
        
        # Add initial temporal context
        adaptive_id.update_temporal_context(
            "creation_time",
            datetime.now().isoformat(),
            "initialization"
        )
        
        # Add domain context
        adaptive_id.update_temporal_context(
            "domain",
            domain_id,
            "domain_assignment"
        )
        
        return adaptive_id
    
    def _update_adaptive_id(self, adaptive_id: AdaptiveID, domain_id: str, data: Dict[str, Any]):
        """
        Update adaptive ID with new data.
        
        Args:
            adaptive_id: AdaptiveID to update
            domain_id: Domain ID
            data: Data dictionary
        """
        # Update with data fields
        for key, value in data.items():
            # Skip complex values
            if isinstance(value, (str, int, float, bool)):
                adaptive_id.update_temporal_context(
                    key,
                    value,
                    "data_update"
                )
        
        # Update confidence based on data stability
        stability = self._calculate_data_stability(data)
        adaptive_id.confidence = (adaptive_id.confidence + stability) / 2
        
        # Update uncertainty
        adaptive_id.uncertainty = max(0.1, adaptive_id.uncertainty - 0.05)


def create_climate_processor(data_dir: str) -> Tuple[HarmonicClimateProcessor, HarmonicIOService]:
    """
    Create a harmonic climate processor with all necessary components.
    
    Args:
        data_dir: Directory containing climate risk data
        
    Returns:
        Tuple of (processor, io_service)
    """
    # Create harmonic I/O service
    io_service = HarmonicIOService(base_frequency=0.2, harmonics=3)
    io_service.start()
    
    # Create field components
    from ..field.topological_field_analyzer import TopologicalFieldAnalyzer
    field_analyzer = TopologicalFieldAnalyzer()
    field_navigator = FieldNavigator(field_analyzer=field_analyzer)
    boundary_detector = SemanticBoundaryDetector(field_navigator=field_navigator)
    
    # Create processor
    processor = HarmonicClimateProcessor(
        io_service=io_service,
        data_dir=data_dir,
        field_navigator=field_navigator,
        boundary_detector=boundary_detector
    )
    
    return processor, io_service
