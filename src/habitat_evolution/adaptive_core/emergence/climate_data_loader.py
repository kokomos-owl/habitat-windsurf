"""
Climate Data Loader

This module loads and processes climate risk data from text documents,
extracting semantic relationships that can be used to test emergent pattern detection.
"""

import os
import re
from typing import Dict, List, Any, Tuple, Set, Optional
from datetime import datetime
import logging
import json
import random

class ClimateDataLoader:
    """
    Loads and processes climate risk data from text documents.
    
    This class extracts semantic relationships from climate risk documents,
    identifying subjects, predicates, and objects that can be used to test
    emergent pattern detection without imposing predefined structures.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the climate data loader.
        
        Args:
            data_dir: Directory containing climate risk data files
        """
        self.data_dir = data_dir
        self.files = []
        self.extracted_data = {}
        self.relationships = []
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Load available files
        self._load_available_files()
    
    def _load_available_files(self) -> None:
        """Load available climate risk data files."""
        if not os.path.exists(self.data_dir):
            self.logger.error(f"Data directory does not exist: {self.data_dir}")
            return
        
        self.files = [f for f in os.listdir(self.data_dir) 
                     if f.endswith('.txt') and os.path.isfile(os.path.join(self.data_dir, f))]
        
        self.logger.info(f"Found {len(self.files)} climate risk data files")
    
    def load_file(self, filename: str) -> Dict[str, Any]:
        """
        Load and process a climate risk data file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Extracted data from the file
        """
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"File does not exist: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract document metadata
        metadata = self._extract_metadata(content, filename)
        
        # Extract sections
        sections = self._extract_sections(content)
        
        # Extract relationships
        relationships = self._extract_relationships(content, metadata)
        
        # Store extracted data
        extracted_data = {
            "metadata": metadata,
            "sections": sections,
            "relationships": relationships
        }
        
        self.extracted_data[filename] = extracted_data
        self.relationships.extend(relationships)
        
        return extracted_data
    
    def load_all_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Load and process all available climate risk data files.
        
        Returns:
            Dictionary mapping filenames to extracted data
        """
        for filename in self.files:
            self.load_file(filename)
        
        return self.extracted_data
    
    def _extract_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from document content.
        
        Args:
            content: Document content
            filename: Name of the file
            
        Returns:
            Extracted metadata
        """
        metadata = {
            "filename": filename,
            "processed_at": datetime.now().isoformat()
        }
        
        # Extract document ID if available
        id_match = re.search(r'Document ID:\s*([A-Z0-9-]+)', content)
        if id_match:
            metadata["document_id"] = id_match.group(1)
        
        # Extract location
        location_match = re.search(r'([A-Za-z\s\']+),\s*Massachusetts', content)
        if location_match:
            metadata["location"] = location_match.group(1).strip()
        else:
            # Try alternative patterns
            alt_patterns = [
                r'Climate Risk Assessment - ([A-Za-z\s\']+)',
                r'Temporal Analysis of Climate Impacts - ([A-Za-z\s\']+)',
                r'CLIMATE RISK ASSESSMENT – ([A-Za-z\s\']+)'
            ]
            
            for pattern in alt_patterns:
                match = re.search(pattern, content)
                if match:
                    metadata["location"] = match.group(1).strip()
                    break
        
        # Extract time periods if available
        time_periods = []
        
        # Look for explicit time periods
        period_matches = re.findall(r'(\d{4})-(\d{4})', content)
        if period_matches:
            for start, end in period_matches:
                time_periods.append({
                    "start_year": int(start),
                    "end_year": int(end)
                })
        
        # Look for projected periods
        projection_matches = [
            (r'Mid-Century\s*(?:Projections)?\s*\((\d{4})\)', "mid-century"),
            (r'Late-Century\s*(?:Scenarios)?\s*\((\d{4})\)', "late-century"),
            (r'Near-Term\s*(?:Projections)?\s*\((\d{4})-(\d{4})\)', "near-term"),
            (r'Short-Term\s*\((\d{4})-(\d{4})\)', "short-term"),
            (r'Mid-Term\s*\((\d{4})-(\d{4})\)', "mid-term"),
            (r'Long-Term\s*\((\d{4})-(\d{4})\)', "long-term")
        ]
        
        for pattern, period_type in projection_matches:
            matches = re.search(pattern, content)
            if matches:
                if len(matches.groups()) == 1:
                    time_periods.append({
                        "type": period_type,
                        "year": int(matches.group(1))
                    })
                elif len(matches.groups()) == 2:
                    time_periods.append({
                        "type": period_type,
                        "start_year": int(matches.group(1)),
                        "end_year": int(matches.group(2))
                    })
        
        if time_periods:
            metadata["time_periods"] = time_periods
        
        return metadata
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract sections from document content.
        
        Args:
            content: Document content
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        # Look for section headers
        section_matches = re.finditer(r'(?:SECTION\s+\d+:|^)([A-Z\s]+)(?:\n|:)', content, re.MULTILINE)
        
        section_positions = []
        for match in section_matches:
            section_name = match.group(1).strip()
            if section_name and len(section_name) > 3:  # Avoid short matches
                section_positions.append((match.start(), section_name))
        
        # Extract section content
        for i, (pos, name) in enumerate(section_positions):
            start = pos + len(name) + 1
            end = section_positions[i+1][0] if i < len(section_positions) - 1 else len(content)
            
            section_content = content[start:end].strip()
            sections[name] = section_content
        
        # If no sections found, try alternative approach
        if not sections:
            # Look for subsection headers
            subsection_matches = re.finditer(r'^(\d+\.\d+)\s+([A-Za-z\s]+)', content, re.MULTILINE)
            
            for match in subsection_matches:
                section_id = match.group(1)
                section_name = match.group(2).strip()
                
                # Find the end of this subsection
                section_start = match.end()
                next_match = re.search(r'^(\d+\.\d+)\s+', content[section_start:], re.MULTILINE)
                
                if next_match:
                    section_end = section_start + next_match.start()
                else:
                    section_end = len(content)
                
                section_content = content[section_start:section_end].strip()
                sections[f"{section_id} {section_name}"] = section_content
        
        return sections
    
    def _extract_relationships(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract semantic relationships from document content.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Define patterns to extract relationships
        patterns = [
            # SLR impacts
            (r'(Sea level rise|SLR)(?:\s+\w+){0,3}\s+(impacts|affects|threatens|erodes|inundates)\s+([A-Za-z\s]+)',
             "impacts"),
            
            # Community responses
            (r'(Communities|Residents|People)(?:\s+\w+){0,3}\s+(adapt|respond|prepare|relocate|protect)\s+(?:to|against|for)?\s+([A-Za-z\s]+)',
             "responds_to"),
            
            # Infrastructure vulnerability
            (r'(Infrastructure|Buildings|Roads|Facilities)(?:\s+\w+){0,3}\s+(vulnerable to|threatened by|at risk from|damaged by)\s+([A-Za-z\s]+)',
             "vulnerable_to"),
            
            # Ecosystem changes
            (r'(Ecosystems|Habitats|Marshes|Beaches)(?:\s+\w+){0,3}\s+(change|migrate|adapt|decline|disappear)\s+(?:due to|because of|from)?\s+([A-Za-z\s]+)',
             "changes_due_to"),
            
            # Policy actions
            (r'(Government|Policymakers|Officials|Planners)(?:\s+\w+){0,3}\s+(implement|develop|create|establish|fund)\s+([A-Za-z\s]+)',
             "implements"),
            
            # Budget allocations
            (r'(Budgets|Funds|Resources|Investments)(?:\s+\w+){0,3}\s+(allocated|directed|provided|spent)\s+(?:for|on|to)\s+([A-Za-z\s]+)',
             "allocated_for"),
            
            # Cultural impacts
            (r'(Cultural|Historic|Archaeological|Indigenous)(?:\s+\w+){0,3}\s+(sites|resources|practices|values)(?:\s+\w+){0,3}\s+(threatened|impacted|affected|lost)\s+(?:by|due to|from)\s+([A-Za-z\s]+)',
             "impacted_by"),
             
            # Cascade relationships
            (r'([A-Za-z\s]+)\s+→\s+([A-Za-z\s]+)\s+→\s+([A-Za-z\s]+)',
             "cascades_to")
        ]
        
        # Extract relationships using patterns
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                
                if rel_type == "cascades_to" and len(groups) == 3:
                    # Handle cascade relationships
                    first_step = groups[0].strip()
                    second_step = groups[1].strip()
                    third_step = groups[2].strip()
                    
                    # Create two relationships for the cascade
                    relationships.append({
                        "source": first_step,
                        "predicate": "leads_to",
                        "target": second_step,
                        "context": {
                            "relationship_type": "cascade_first_step",
                            "full_cascade": f"{first_step} → {second_step} → {third_step}",
                            "document": metadata.get("filename", ""),
                            "location": metadata.get("location", "")
                        }
                    })
                    
                    relationships.append({
                        "source": second_step,
                        "predicate": "leads_to",
                        "target": third_step,
                        "context": {
                            "relationship_type": "cascade_second_step",
                            "full_cascade": f"{first_step} → {second_step} → {third_step}",
                            "document": metadata.get("filename", ""),
                            "location": metadata.get("location", "")
                        }
                    })
                
                elif rel_type == "impacted_by" and len(groups) == 4:
                    # Handle cultural impacts with more complex pattern
                    cultural_type = groups[0].strip()
                    resource_type = groups[1].strip()
                    impact_type = groups[2].strip()
                    cause = groups[3].strip()
                    
                    source = f"{cultural_type} {resource_type}"
                    predicate = f"impacted_by"
                    
                    relationships.append({
                        "source": source,
                        "predicate": predicate,
                        "target": cause,
                        "context": {
                            "relationship_type": "cultural_impact",
                            "impact_type": impact_type,
                            "document": metadata.get("filename", ""),
                            "location": metadata.get("location", "")
                        }
                    })
                
                elif len(groups) == 3:
                    # Handle standard three-part relationships
                    source = groups[0].strip()
                    predicate = rel_type
                    target = groups[2].strip()
                    
                    relationships.append({
                        "source": source,
                        "predicate": predicate,
                        "target": target,
                        "context": {
                            "relationship_type": rel_type,
                            "document": metadata.get("filename", ""),
                            "location": metadata.get("location", "")
                        }
                    })
        
        # Extract temporal relationships if time periods are available
        if "time_periods" in metadata:
            time_patterns = [
                (r'By\s+(\d{4})(?:,)?\s+([A-Za-z\s]+)(?:\s+\w+){0,3}\s+(increase|decrease|rise|fall|change)\s+(?:by|to)?\s+([A-Za-z0-9\s\.%]+)',
                 "changes_to"),
                (r'In\s+the\s+(\d{4}s)(?:,)?\s+([A-Za-z\s]+)(?:\s+\w+){0,3}\s+(average|reached|experienced)\s+([A-Za-z0-9\s\.%]+)',
                 "experienced")
            ]
            
            for pattern, rel_type in time_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    groups = match.groups()
                    
                    if len(groups) == 4:
                        time_period = groups[0].strip()
                        source = groups[1].strip()
                        predicate = rel_type
                        target = groups[3].strip()
                        
                        relationships.append({
                            "source": source,
                            "predicate": predicate,
                            "target": target,
                            "context": {
                                "relationship_type": "temporal",
                                "time_period": time_period,
                                "document": metadata.get("filename", ""),
                                "location": metadata.get("location", "")
                            }
                        })
        
        return relationships
    
    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """
        Get all extracted relationships.
        
        Returns:
            List of all relationships extracted from all files
        """
        return self.relationships
    
    def generate_observation_data(self, batch_size: int = 5, 
                                 randomize: bool = True) -> List[Dict[str, Any]]:
        """
        Generate observation data for testing.
        
        Args:
            batch_size: Number of relationships per observation
            randomize: Whether to randomize the relationships
            
        Returns:
            List of observation data dictionaries
        """
        if not self.relationships:
            self.logger.warning("No relationships extracted yet")
            return []
        
        # Copy relationships to avoid modifying the original
        relationships = self.relationships.copy()
        
        # Randomize if requested
        if randomize:
            random.shuffle(relationships)
        
        # Create batches
        observations = []
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            
            observation = {
                "predicates": []
            }
            
            for rel in batch:
                observation["predicates"].append({
                    "subject": rel["source"],
                    "verb": rel["predicate"],
                    "object": rel["target"],
                    "context": rel["context"]
                })
            
            observations.append(observation)
        
        return observations
    
    def generate_evolution_data(self, steps: int = 5) -> List[Dict[str, Any]]:
        """
        Generate data that shows pattern evolution over time.
        
        Args:
            steps: Number of evolution steps
            
        Returns:
            List of observation data dictionaries showing evolution
        """
        if not self.relationships:
            self.logger.warning("No relationships extracted yet")
            return []
        
        # Find relationships with the same source and target but different predicates
        source_target_pairs = {}
        for rel in self.relationships:
            key = f"{rel['source']}_{rel['target']}"
            if key not in source_target_pairs:
                source_target_pairs[key] = []
            source_target_pairs[key].append(rel)
        
        # Filter to pairs with multiple predicates
        evolving_pairs = {k: v for k, v in source_target_pairs.items() if len(v) > 1}
        
        # If no evolving pairs found, create synthetic ones
        if not evolving_pairs:
            self.logger.info("No natural evolution found, creating synthetic examples")
            
            # Create synthetic evolution examples
            evolution_examples = []
            
            # Example 1: Community response evolution
            evolution_examples.append([
                {"source": "Coastal communities", "predicate": "threatened_by", "target": "sea level rise"},
                {"source": "Coastal communities", "predicate": "responds_to", "target": "sea level rise"},
                {"source": "Coastal communities", "predicate": "adapts_to", "target": "sea level rise"},
                {"source": "Coastal communities", "predicate": "resilient_to", "target": "sea level rise"}
            ])
            
            # Example 2: Infrastructure vulnerability evolution
            evolution_examples.append([
                {"source": "Infrastructure", "predicate": "vulnerable_to", "target": "flooding"},
                {"source": "Infrastructure", "predicate": "damaged_by", "target": "flooding"},
                {"source": "Infrastructure", "predicate": "retrofitted_against", "target": "flooding"},
                {"source": "Infrastructure", "predicate": "protected_from", "target": "flooding"}
            ])
            
            # Create observations from synthetic examples
            observations = []
            for example in evolution_examples:
                for i, rel in enumerate(example):
                    # Create context with step information
                    context = {
                        "relationship_type": "evolution",
                        "evolution_step": i + 1,
                        "total_steps": len(example)
                    }
                    
                    # Add to observation
                    observation = {
                        "predicates": [{
                            "subject": rel["source"],
                            "verb": rel["predicate"],
                            "object": rel["target"],
                            "context": context
                        }]
                    }
                    
                    observations.append(observation)
            
            return observations
        
        # Create observations from real evolving pairs
        observations = []
        for key, rels in evolving_pairs.items():
            if len(rels) >= steps:
                selected_rels = rels[:steps]
            else:
                # Repeat some relationships to reach desired steps
                selected_rels = rels + [random.choice(rels) for _ in range(steps - len(rels))]
            
            for i, rel in enumerate(selected_rels):
                # Create context with step information
                context = rel["context"].copy()
                context.update({
                    "evolution_step": i + 1,
                    "total_steps": steps
                })
                
                # Add to observation
                observation = {
                    "predicates": [{
                        "subject": rel["source"],
                        "verb": rel["predicate"],
                        "object": rel["target"],
                        "context": context
                    }]
                }
                
                observations.append(observation)
        
        return observations
