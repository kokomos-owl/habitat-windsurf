"""
Integration tests for knowledge graph evolution with coherence-adherence validation.
Environment supports evolution of structure-meaning relationships and preparation 
of coherent-adherent Seeds.
"""

import os
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4
import asyncio
import logging
from contextlib import asynccontextmanager

# Core imports
from core.state_processor import StateProcessor
from core.db_interface import DBInterface
from core.llm_interface import LLMInterface
from core.observable_interface import ObservableInterface
from core.rag_controller import RAGController

# Pattern and evolution imports
from adaptive_core.pattern_core import PatternCore
from adaptive_core.knowledge_coherence import KnowledgeCoherence
from meaning_evolution.structure_meaning_evolution import StructureMeaningEvolution

from config.settings import Settings

# Constants for validation
MIN_NODES_EXPECTED = 2
MIN_RELATIONSHIPS_EXPECTED = 1
REQUIRED_ENV_VARS = ['NEO4J_URI', 'MONGODB_URI', 'ANTHROPIC_API_KEY']

class CoherenceThresholds:
    """Defines thresholds for coherence validation across different contexts."""
    
    # Core thresholds
    BASE = 0.3                # Base coherence score required
    MIN_NODES = 2            # Minimum nodes for valid structure
    MIN_RELATIONSHIPS = 1    # Minimum relationships for valid structure
    
    # Context-specific thresholds
    DOCUMENT = 0.4           # Higher threshold for document ingestion
    USER = 0.25             # Lower threshold for user associations
    PATTERN = 0.35          # Medium threshold for pattern matching
    
    # Validation levels
    POC = "poc"             # Basic validation for POC
    STRICT = "strict"       # Stricter validation for production
    
    @classmethod
    def get_threshold(cls, origin_type: str, validation_level: str = POC) -> float:
        """Get appropriate threshold based on origin type and validation level."""
        base = {
            "document_ingestion": cls.DOCUMENT,
            "user_association": cls.USER,
            "pattern_observation": cls.PATTERN
        }.get(origin_type, cls.BASE)
        
        # Increase threshold for strict validation
        return base * 1.2 if validation_level == cls.STRICT else base


class InterfaceTypes:
    """Registry of supported interfaces and their requirements."""
    
    # Core interfaces
    CLAUDE = "claude"
    OLLAMA = "ollama"
    HABITAT = "habitat"
    
    # Interface requirements
    REQUIREMENTS = {
        CLAUDE: {
            "required_fields": ["model", "response"],
            "capabilities": ["text_generation", "embedding"],
            "version": "1.0"
        },
        OLLAMA: {
            "required_fields": ["response"],
            "capabilities": ["text_generation"],
            "version": "1.0"
        },
        HABITAT: {
            "required_fields": ["nodes", "relationships"],
            "capabilities": ["graph_operations"],
            "version": "1.0"
        }
    }
    
    @classmethod
    def validate_interface(cls, interface_type: str, interaction_result: Dict[str, Any]) -> Dict[str, bool]:
        """Validate interface type and its requirements."""
        try:
            # Check if interface type is registered
            if interface_type not in cls.REQUIREMENTS:
                return {"valid": False, "error": f"Unregistered interface type: {interface_type}"}
                
            requirements = cls.REQUIREMENTS[interface_type]
            
            # Validate required fields
            missing_fields = [
                field for field in requirements["required_fields"]
                if field not in interaction_result
            ]
            
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields for {interface_type}: {', '.join(missing_fields)}"
                }
                
            return {
                "valid": True,
                "interface": interface_type,
                "version": requirements["version"],
                "capabilities": requirements["capabilities"]
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}


class CoherenceAdherenceValidator:
    """Validates coherent-adherent relationships and prepares for Seed generation."""
    
    def __init__(self):
        self.valid_origin_types = ["document_ingestion", "user_association"]
        
    async def validate_interface_adherence(
        self,
        interface_type: str,
        interaction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates that interface interactions maintain adherence.
        Critical for both immediate validation and future Seed wrapping.
        """
        try:
            # Validate interface type and requirements
            validation_result = InterfaceTypes.validate_interface(interface_type, interaction_result)
            if not validation_result["valid"]:
                return validation_result
                
            # Additional interface-specific validation
            if interface_type == InterfaceTypes.CLAUDE:
                is_valid = isinstance(interaction_result["response"], str)
            elif interface_type == InterfaceTypes.OLLAMA:
                is_valid = isinstance(interaction_result["response"], str)
            elif interface_type == InterfaceTypes.HABITAT:
                is_valid = (isinstance(interaction_result["nodes"], list) and
                          isinstance(interaction_result["relationships"], list))
            else:
                is_valid = False
                
            return {
                "valid": is_valid,
                "interface": interface_type,
                "version": validation_result["version"],
                "capabilities": validation_result["capabilities"]
            }
            
        except Exception as e:
            logger.error(f"Interface adherence validation failed: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def validate_state_space(self, state_space: Dict[str, Any]) -> Dict[str, bool]:
        """Validate state space structure and properties."""
        try:
            # Required fields
            required_fields = {
                "origin_type": str,
                "origin_id": str,
                "state_type": str,
                "timestamp": str,
                "adaptive_context": dict
            }
            
            # Check required fields and types
            for field, expected_type in required_fields.items():
                if field not in state_space:
                    return {"valid": False, "error": f"Missing required field: {field}"}
                if not isinstance(state_space[field], expected_type):
                    return {"valid": False, "error": f"Invalid type for {field}"}
                    
            # Validate origin_type
            if state_space["origin_type"] not in self.valid_origin_types:
                return {"valid": False, "error": "Invalid origin_type"}
                
            # Validate timestamp format
            try:
                datetime.fromisoformat(state_space["timestamp"])
            except ValueError:
                return {"valid": False, "error": "Invalid timestamp format"}
                
            # Conditional fields based on origin_type
            if state_space["origin_type"] == "user_association":
                if "user_id" not in state_space:
                    return {"valid": False, "error": "Missing user_id for user_association"}
                if "association_type" not in state_space:
                    return {"valid": False, "error": "Missing association_type for user_association"}
                    
            elif state_space["origin_type"] == "document_ingestion":
                if "document_metadata" not in state_space:
                    return {"valid": False, "error": "Missing document_metadata for document_ingestion"}
                    
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
            
    async def check_knowledge_coherence(
        self,
        current_state: Dict[str, Any],
        proposed_change: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates state-space coherence for graph evolution.
        Ensures state transitions can support Seed generation.
        """
        try:
            # Validate structure
            if not all(key in proposed_change for key in ["state_id", "structural_elements", "state_space", "metadata"]):
                return {"coherent": False, "error": "Invalid state update structure"}
                
            # Validate state space
            state_space_validation = self.validate_state_space(proposed_change["state_space"])
            if not state_space_validation["valid"]:
                return {"coherent": False, "error": f"Invalid state space: {state_space_validation['error']}"}

            # Extract structural elements and metadata
            nodes = proposed_change["structural_elements"]["nodes"]
            relationships = proposed_change["structural_elements"]["relationships"]
            validation_level = proposed_change["metadata"]["validation_level"]
            origin_type = proposed_change["state_space"]["origin_type"]
            
            # Get appropriate threshold
            threshold = CoherenceThresholds.get_threshold(origin_type, validation_level)
            
            # Basic structural validation
            has_nodes = len(nodes) >= CoherenceThresholds.MIN_NODES
            has_relationships = len(relationships) >= CoherenceThresholds.MIN_RELATIONSHIPS
            
            if not (has_nodes and has_relationships):
                return {
                    "coherent": False,
                    "error": f"Insufficient structure: needs {CoherenceThresholds.MIN_NODES}+ nodes and {CoherenceThresholds.MIN_RELATIONSHIPS}+ relationships"
                }
                
            # Calculate coherence score based on origin type
            if origin_type == "document_ingestion":
                coherence_score = self._calculate_document_coherence(nodes, relationships)
            elif origin_type == "user_association":
                coherence_score = self._calculate_user_coherence(nodes, relationships)
            else:
                coherence_score = self._calculate_base_coherence(nodes, relationships)
                
            is_coherent = coherence_score >= threshold
            
            return {
                "coherent": is_coherent,
                "score": coherence_score,
                "threshold": threshold,
                "validation_level": validation_level
            }
            
        except Exception as e:
            logger.error(f"Knowledge coherence check failed: {str(e)}")
            return {"coherent": False, "error": str(e)}

    def _calculate_base_coherence(self, nodes: List[Dict], relationships: List[Dict]) -> float:
        """Calculate base coherence score."""
        node_score = min(1.0, len(nodes) / (CoherenceThresholds.MIN_NODES * 2)) * 0.6
        rel_score = min(1.0, len(relationships) / (CoherenceThresholds.MIN_RELATIONSHIPS * 2)) * 0.4
        return node_score + rel_score
        
    def _calculate_document_coherence(self, nodes: List[Dict], relationships: List[Dict]) -> float:
        """Calculate document-specific coherence score."""
        base_score = self._calculate_base_coherence(nodes, relationships)
        
        # Additional document-specific checks
        has_content_nodes = any(n.get("type") == "content" for n in nodes)
        has_metadata = any(n.get("type") == "metadata" for n in nodes)
        has_semantic_rels = any(r.get("type") == "semantic" for r in relationships)
        
        doc_score = sum([
            0.2 if has_content_nodes else 0,
            0.1 if has_metadata else 0,
            0.1 if has_semantic_rels else 0
        ])
        
        return min(1.0, base_score + doc_score)
        
    def _calculate_user_coherence(self, nodes: List[Dict], relationships: List[Dict]) -> float:
        """Calculate user association coherence score."""
        base_score = self._calculate_base_coherence(nodes, relationships)
        
        # Additional user-specific checks
        has_user_nodes = any(n.get("type") == "user" for n in nodes)
        has_interaction = any(r.get("type") == "interaction" for r in relationships)
        
        user_score = sum([
            0.15 if has_user_nodes else 0,
            0.15 if has_interaction else 0
        ])
        
        return min(1.0, base_score + user_score)

    def verify_environment():
        """Verify all required environment variables are set."""
        missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {missing}"
            )
            
class TestHabitatEvolution:
    """
    Test suite for Habitat knowledge evolution with coherence-adherence validation.
    Supports:
    1. Document ingestion and graph creation.
    2. RAG enhancement with coherence validation.
    3. Pattern observation ensuring state-space validity.
    4. Evolution supporting seed preparation.
    """
    
    @pytest.fixture(scope="class")
    async def setup_environment(self):
        """Setup and cleanup test environment."""
        verify_environment()
        
        try:
            # Ensure required paths exist
            ensure_volume_paths()  # Assumed utility function
            yield
            
        finally:
            await cleanup_test_data()  # Assumed utility function
            logger.info("Test environment cleaned up")
    
    @pytest.fixture
    async def setup_components(self, setup_environment):
        """Initialize and configure test components with coherence validation."""
        settings = Settings()
        
        try:
            # Initialize core components
            db = DBInterface(settings)
            llm = LLMInterface(settings)
            observable = ObservableInterface(settings)
            rag_controller = RAGController(settings, llm)
            
            # Initialize validation and evolution components
            coherence_validator = CoherenceAdherenceValidator()
            pattern_core = PatternCore(
                timestamp_service=MockTimestampService(),
                event_manager=MockEventManager(),
                version_service=MockVersionService()
            )
            
            components = {
                'db': db,
                'llm': llm,
                'observable': observable,
                'rag_controller': rag_controller,
                'pattern_core': pattern_core,
                'coherence': coherence_validator,
                'processor': StateProcessor(  # Changed from BidirectionalProcessor
                    validator=coherence_validator,
                    rag_controller=rag_controller,
                    db_interface=db,
                    state_coordinator=state_coordinator
                )
            }
            
            # Verify critical components
            await self._verify_db_connection(db)
            await self._verify_llm_service(llm)
            
            yield components
            
        except Exception as e:
            logger.error(f"Failed to setup components: {str(e)}")
            raise
        
        finally:
            # Cleanup
            for component in components.values():
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
            logger.info("Components cleaned up")

    async def _verify_db_connection(self, db: DBInterface) -> None:
        """Verify database connections are working."""
        try:
            await db.mongo_client.admin.command('ping')
            await db.neo4j_driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {str(e)}")

    async def _verify_llm_service(self, llm: LLMInterface) -> None:
        """Verify LLM service is accessible."""
        try:
            result = await llm.process_text("Test connection.")
            if not result or "response" not in result:
                raise ConnectionError("LLM service test failed")
        except Exception as e:
            raise ConnectionError(f"LLM service verification failed: {str(e)}")

    async def _verify_graph_minimum(self, state: Dict[str, Any]) -> None:
        """Verify graph meets minimum requirements for coherence."""
        if state["node_count"] < MIN_NODES_EXPECTED:
            raise ValueError(
                f"Graph has fewer than {MIN_NODES_EXPECTED} nodes: {state['node_count']}"
            )
        if state["relationship_count"] < MIN_RELATIONSHIPS_EXPECTED:
            raise ValueError(
                f"Graph has fewer than {MIN_RELATIONSHIPS_EXPECTED} relationships: " \
                f"{state['relationship_count']}"
            )
            
    def _create_test_document(self, doc_id: str) -> Dict[str, Any]:
        """Create test document with verifiable structure-meaning relationships."""
        timestamp = datetime.utcnow().isoformat()
        
        return {
            "doc_id": doc_id,
            "content": {
                "text": "Test document for climate impact analysis",
                "metadata": {
                    "source": "climate_assessment",
                    "timestamp": timestamp
                }
            },
            "structural_elements": {
                "nodes": [
                    {
                        "id": f"concept_1_{doc_id}",
                        "type": "concept",
                        "properties": {
                            "name": "climate_risk",
                            "confidence": 0.8
                        }
                    },
                    {
                        "id": f"concept_2_{doc_id}",
                        "type": "concept",
                        "properties": {
                            "name": "impact_assessment",
                            "confidence": 0.7
                        }
                    }
                ],
                "relationships": [
                    {
                        "source": f"concept_1_{doc_id}",
                        "target": f"concept_2_{doc_id}",
                        "type": "requires",
                        "properties": {
                            "confidence": 0.75,
                            "timestamp": timestamp
                        }
                    }
                ]
            },
            "metadata": {
                "domain": "climate",
                "timestamp": timestamp,
                "confidence": 0.8
            }
        }

    async def test_document_ingestion(self, setup_components):
        """Test document ingestion with coherence-adherence validation."""
        components = setup_components
        
        try:
            # Get initial state
            initial_state = await components['db'].get_graph_state()
            
            # Create test document
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            # Define adaptive context aligned with state space requirements
            adaptive_context = {
                "id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "type": "document_ingestion"
            }
            
            # Process state update
            state_result = await components['processor'].process_state_update(
                origin_type="document_ingestion",
                origin_id=doc_id,
                update={
                    "structural_elements": {
                        "nodes": document["structural_elements"]["nodes"],
                        "relationships": document["structural_elements"]["relationships"]
                    }, 
                    },
                adaptive_context=adaptive_context
            )
            
            # Validate state identity
            expected_state_id = f"state_{adaptive_context['id']}_document_ingestion_{doc_id}"
            assert state_result["state_id"] == expected_state_id, "State identity mismatch"
            
            # Verify state space exactly as defined in state_processor
            state_space = state_result["graph_updates"]["state_space"]
            assert state_space["origin_type"] == "document_ingestion"
            assert state_space["origin_id"] == doc_id
            assert state_space["state_type"] == "document_ingestion"
            assert "timestamp" in state_space
            assert state_space["adaptive_context"] == adaptive_context
            
            # Validate coherence
            assert state_result["coherence_result"]["coherent"], "State evolution failed coherence validation"
            
            # Get post-ingestion state and verify minimum requirements
            post_state = await components['db'].get_graph_state()
            await self._verify_graph_minimum(post_state)
            
            return {
                "success": True,
                "state_id": state_result["state_id"],
                "coherence_result": state_result["coherence_result"]
            }
            
        except Exception as e:
            logger.error(f"State evolution test failed: {str(e)}")
            raise
        
    async def test_user_association(self, setup_components):
        """
        Test user association state patterns (POC demonstration only).
        Validates state space can support future user interaction patterns.
        """
        components = setup_components
        
        try:
            # Simulate user discovering state in document
            user_id = str(uuid4())
            adaptive_context = {
                "id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "type": "user_association",
                "association_pattern": "discovery"  # Future: received, created
            }
            
            # Validate state space can handle user associations
            state_result = await components['processor'].process_state_update(
                origin_type="user_association",
                origin_id=user_id,
                update=None,  # No structural changes in POC
                adaptive_context=adaptive_context
            )
            
            # Verify state space supports user interaction patterns
            state_space = state_result["graph_updates"]["state_space"]
            assert state_space["user_id"] == user_id
            assert state_space["association_type"] == "discovery"
            
            # Validate coherence maintained with user state
            assert state_result["coherence_result"]["coherent"]
            
            return {
                "success": True,
                "state_id": state_result["state_id"],
                "note": "POC demonstration - user interaction patterns reserved for future implementation"
            }
            
        except Exception as e:
            logger.error(f"User association pattern test failed: {str(e)}")
            raise

    async def test_rag_enhancement(self, setup_components):
        """Test RAG enhancement with interface adherence validation."""
        components = setup_components
        
        try:
            # Create initial document state
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            # Process RAG enhancement with state tracking
            interface_type = "claude"  # Or get from configuration
            rag_adaptive_context = {
                "id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "type": "rag_enhancement",
                "interface_type": interface_type,  # Track specific LLM interface
                "domain_context": {
                    "domain": "climate",
                    "interface": interface_type
                }
            }
            
            # Process through state processor
            rag_state = await components['processor'].process_rag_enhancement(
                state_id=f"state_{rag_adaptive_context['id']}_rag_{doc_id}",
                document={
                    "doc_id": doc_id,
                    "content": document["content"],
                    "metadata": document["metadata"]
                },
                adaptive_context=rag_adaptive_context
            )
            
            # Validate interface adherence for specific LLM
            llm_adherence = await components['validator'].validate_interface_adherence(
                interface_type=interface_type,
                interaction_result=rag_state["rag_result"]
            )
            assert llm_adherence["valid"], f"RAG enhancement failed {interface_type} interface adherence"
            
            # Validate coherence
            assert rag_state["coherence_result"]["coherent"], "RAG enhancement failed coherence validation"
            
            return {
                "success": True,
                "state_id": rag_state["state_id"],
                "rag_coherence": rag_state["coherence_result"]["score"],
                "llm_adherence": llm_adherence["valid"],
                "interface_type": interface_type
            }
            
        except Exception as e:
            logger.error(f"RAG enhancement test failed: {str(e)}")
            raise
        
    async def test_pattern_observation(self, setup_components):
        """Test pattern observation with coherence-adherence validation."""
        components = setup_components
        
        try:
            # Create and process initial document
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            # Get document structure through state processor
            doc_state = await components['processor'].process_state_update(
                origin_type="document_ingestion",
                origin_id=doc_id,
                update=document,
                adaptive_context={
                    "id": str(uuid4()),
                    "type": "document_ingestion",
                    "timestamp": datetime.utcnow().isoformat(),
                    "domain": "climate"  # Domain context starts here
                }
            )
            
            # Get RAG context for semantic enhancement
            rag_result = await components['rag_controller'].process_document(
                document={
                    "doc_id": doc_id,
                    "content": document["content"],
                    "metadata": document["metadata"]
                },
                domain_context={
                    "domain": "climate",
                    "pattern_context": {
                        "domain_specific_coherence": True,
                        "domain_patterns": ["climate_change", "environmental_impact"]
                    }
                }
            )
            
            # Observe patterns with combined structural and semantic
            pattern = await components['pattern_core'].observe_evolution(
                structural_change=doc_state["graph_updates"]["structural"],
                semantic_change=doc_state["graph_updates"]["semantic"],
                evidence=rag_result["evidence"],
                domain_context={  # Pattern-specific domain context
                    "domain": "climate",
                    "coherence_threshold": 0.3,
                    "pattern_type": "domain_specific"
                }
            )
            
            # Validate pattern coherence with domain context
            coherence_result = await components['validator'].check_knowledge_coherence(
                current_state={"patterns": [], "domain": "climate"},
                proposed_change={"patterns": [pattern], "domain": "climate"}
            )
            assert coherence_result["coherent"], "Pattern observation failed coherence validation"
            
            # Process pattern state with full context
            pattern_state = await components['processor'].process_state_update(
                origin_type="pattern_observation",
                origin_id=pattern["pattern_id"],
                update={
                    "pattern": pattern,
                    "evidence": rag_result["evidence"],
                    "domain_context": {
                        "domain": "climate",
                        "pattern_type": "domain_specific"
                    }
                },
                adaptive_context={
                    "id": str(uuid4()),
                    "type": "pattern_observation",
                    "timestamp": datetime.utcnow().isoformat(),
                    "domain": "climate"
                }
            )
            
            return {
                "success": True,
                "pattern_id": pattern["pattern_id"],
                "coherence_score": coherence_result["score"],
                "domain": "climate",
                "state_id": pattern_state["state_id"]
            }
            
        except Exception as e:
            logger.error(f"Pattern observation test failed: {str(e)}")
            raise

    async def test_knowledge_evolution(self, setup_components):
        """Test knowledge evolution with full coherence-adherence validation."""
        components = setup_components
        
        try:
            # Get initial state
            initial_state = await components['db'].get_graph_state()
            
            # Create and process document through pipeline
            doc_id = str(uuid4())
            document = self._create_test_document(doc_id)
            
            # Document processing with validation
            doc_result = await components['processor'].process_state_update(
                origin_type="document_ingestion",
                origin_id=doc_id,
                update=document,
                adaptive_context={
                    "id": str(uuid4()),
                    "type": "document_ingestion",
                    "timestamp": datetime.utcnow().isoformat(),
                    "domain": "climate"  # Domain context starts here
                }
            )
            
            # RAG enhancement with validation
            rag_result = await components['rag_controller'].process_document(
                document={
                    "doc_id": doc_id,
                    "content": document["content"],
                    "metadata": document["metadata"]
                },
                domain_context={"domain": "climate"}
            )
            
            # Pattern observation
            pattern = await components['pattern_core'].observe_evolution(
                structural_change=doc_result["graph_updates"]["structural"],
                semantic_change=doc_result["graph_updates"]["semantic"],
                evidence=rag_result["evidence"]
            )
            
            # Evolution context preparation
            evolution_context = {
                "structural": rag_result["rag_output"]["graph_updates"]["structural"],
                "semantic": rag_result["rag_output"]["graph_updates"]["semantic"],
                "evidence": pattern
            }
            
            # Process evolution
            evolution_result = await components['evolution'].evolve(
                data=evolution_context,
                context=evolution_context
            )
            
            # Validate evolved state
            coherence_result = await components['validator'].check_knowledge_coherence(
                current_state=initial_state,
                proposed_change=evolution_result.evolved_data
            )
            assert coherence_result["coherent"], "Knowledge evolution failed coherence validation"
            
            # Update state for seed preparation
            await components['processor'].state_coordinator.update_state({
                "timestamp": datetime.utcnow().isoformat(),
                "doc_id": doc_id,
                "state_type": "evolution_complete",
                "pattern_id": pattern["pattern_id"],
                "coherence_result": coherence_result,
                "evolution_id": evolution_result.evolution_id
            })
            
            return {
                "success": True,
                "doc_id": doc_id,
                "pattern_id": pattern["pattern_id"],
                "evolution_id": evolution_result.evolution_id,
                "coherence_score": coherence_result["score"]
            }
            
        except Exception as e:
            logger.error(f"Knowledge evolution test failed: {str(e)}")
            raise
        
    async def _prepare_seed_state(
        self,
        components: Dict[str, Any],
        state_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare state data for potential Seed creation.
        Validates state coherence and interface adherence.
        """
        try:
            # Basic state validation
            if not state_data.get("doc_id") or not state_data.get("state_type"):
                raise ValueError("Invalid state data for Seed preparation")

            # Validate state coherence
            coherence_result = await components['validator'].check_knowledge_coherence(
                current_state=state_data,
                proposed_change=state_data  # Validating state integrity
            )
            
            if not coherence_result["coherent"]:
                raise ValueError("State failed coherence validation for Seed preparation")

            return {
                "id": f"seed_{state_data['doc_id']}",
                "timestamp": datetime.utcnow().isoformat(),
                "state_type": state_data["state_type"],
                "graph_state": state_data,
                "coherence_score": coherence_result["score"],
                "state_space": state_data["state_space"],  # Add explicit state space tracking
                "adaptive_context": state_data["adaptive_context"]  # Add explicit adaptive context
            }

        except Exception as e:
            logger.error(f"Seed state preparation failed: {str(e)}")
            raise

    async def test_state_validation(self, setup_components):
        """Test state validation for coherence-adherence and Seed preparation."""
        components = setup_components
        
        try:
            # Process test document through evolution
            evolution_result = await self.test_knowledge_evolution(components)
            assert evolution_result["success"], "Evolution failed during state validation"

            # Get current state from state coordinator
            current_state = await components['processor'].state_coordinator.get_current_state()

            # Validate state for Seed preparation
            seed_state = await self._prepare_seed_state(
                components,
                current_state
            )

            # Validate interface adherence for each required interface
            for interface in ["claude", "ollama", "habitat"]:
                adherence_result = await components['validator'].validate_interface_adherence(
                    interface_type=interface,
                    interaction_result=current_state
                )
                assert adherence_result["valid"], f"State failed {interface} interface adherence"

            return {
                "success": True,
                "seed_id": seed_state["id"],
                "coherence_score": seed_state["coherence_score"],
                "timestamp": seed_state["timestamp"]
            }

        except Exception as e:
            logger.error(f"State validation test failed: {str(e)}")
            raise

    async def test_component_health(self, setup_components):
        """Verify health of core components with coherence validation."""
        components = setup_components
        health_status = {}
        
        try:
            # Check DB connection
            await self._verify_db_connection(components['db'])
            health_status['db'] = "healthy"
            
            # Check LLM service
            await self._verify_llm_service(components['llm'])
            health_status['llm'] = "healthy"
            
            # Check RAG metrics
            rag_metrics = components['rag_controller'].get_metrics()
            assert "queries" in rag_metrics, "RAG metrics unavailable"
            health_status['rag'] = "healthy"
            
            # Verify graph minimum state
            graph_state = await components['db'].get_graph_state()
            await self._verify_graph_minimum(graph_state)
            health_status['graph'] = "healthy"
            
            # Verify coherence validator
            coherence_result = await components['validator'].check_knowledge_coherence(
                current_state=graph_state,
                proposed_change=graph_state
            )
            health_status['coherence'] = "healthy" if coherence_result["coherent"] else "unhealthy"
            
            return {
                "success": True,
                "health_status": health_status,
                "graph_state": graph_state,
                "coherence_status": coherence_result
            }
            
        except Exception as e:
            logger.error(f"Component health check failed: {str(e)}")
            health_status['error'] = str(e)
            return {
                "success": False,
                "health_status": health_status
            }

if __name__ == "__main__":
    verify_environment()
    pytest.main(["-v", __file__]) 