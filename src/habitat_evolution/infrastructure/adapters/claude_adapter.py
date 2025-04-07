"""
Claude API adapter for Habitat Evolution.

This module provides an adapter for integrating with Anthropic's Claude API,
enabling pattern-aware RAG to leverage Claude's capabilities for enhanced
pattern extraction and analysis.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import Anthropic SDK
import anthropic

# Import metrics collector and cache
from src.habitat_evolution.infrastructure.metrics.claude_api_metrics import claude_metrics
from src.habitat_evolution.infrastructure.adapters.claude_cache import claude_cache

logger = logging.getLogger(__name__)


class ClaudeAdapter:
    """
    Adapter for Anthropic's Claude API.
    
    This adapter provides methods for interacting with Claude, enabling
    pattern-aware RAG to leverage Claude's capabilities for enhanced
    pattern extraction and analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Claude adapter.
        
        Args:
            api_key: Optional API key for Claude (if None, will look for ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        # Check if we should use mock responses
        self.use_mock = not self.api_key
        
        if self.use_mock:
            logger.warning("No API key found. Using mock responses for testing.")
            self.client = None
        else:
            # Initialize the Anthropic client
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("Initialized Anthropic client")
            
        logger.info(f"Initialized ClaudeAdapter (use_mock: {self.use_mock})")
        
    async def process_query(
        self, query: str, context: Dict[str, Any], patterns: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query using Claude, with optional context and patterns.
        
        Args:
            query: The query to process
            context: Optional context to include with the query
            patterns: Optional patterns to include with the query
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            Dict containing the processed query and Claude's response
        """
        if self.use_mock:
            return self._mock_process_query(query, context, patterns)
            
        # Check cache if enabled
        if use_cache:
            cache_key = claude_cache.get_query_cache_key(query, context, patterns)
            cache_hit, cached_response = claude_cache.get_from_cache(cache_key)
            
            if cache_hit and cached_response:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_response
        
        try:
            # Start timing the API call
            start_time = time.time()
            
            # Format the patterns for Claude
            patterns_text = self._format_patterns_for_prompt(patterns)
            
            # Format context for Claude if available
            context_text = ""
            if context:
                context_text = "\nContext:\n" + json.dumps(context, indent=2)
            
            # Create the system prompt
            system_prompt = f"""
            You are an assistant that helps answer questions based on provided patterns and context.
            
            The following patterns have been identified as relevant to the query:
            {patterns_text}
            
            Use these patterns to inform your response, but don't explicitly mention them unless asked.
            Your response should be helpful, accurate, and concise.
            """
            
            # Create the user message
            user_message = f"Query: {query}{context_text}"
            
            # Call the Claude API
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract the response text
            response_text = response.content[0].text
            
            # Create the result
            result = {
                "query_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "response": response_text,
                "patterns": patterns,
                "model": "claude-3-opus-20240229",
                "tokens_used": response.usage.output_tokens
            }
            
            # Track metrics
            claude_metrics.track_query(query, result, response_time_ms)
            
            # Save to cache if enabled
            if use_cache:
                cache_key = claude_cache.get_query_cache_key(query, context, patterns)
                claude_cache.save_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error processing query with Claude: {e}")
            error_result = {
                "error": str(e),
                "status": "error",
                "query_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Track the error in metrics
            claude_metrics.track_error("query", str(e), {"query": query})
            
            return error_result
            
    async def process_document(self, document: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
        """
        Process a document with Claude.
        
        This method takes a document and processes it through Claude to
        extract patterns and insights.
        
        Args:
            document: The document to process
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            A dictionary containing the extracted patterns and additional information
        """
        if self.use_mock:
            return await self._mock_process_document(document)
            
        # Check cache if enabled
        if use_cache:
            cache_key = claude_cache.get_document_cache_key(document)
            cache_hit, cached_response = claude_cache.get_from_cache(cache_key)
            
            if cache_hit and cached_response:
                logger.info(f"Cache hit for document: {document.get('id', 'unknown')}")
                return cached_response
            
        try:
            # Start timing the API call
            start_time = time.time()
            
            # Extract the document content
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            doc_id = document.get("id", "unknown")
            title = document.get("title", "Untitled Document")
            
            # Create the system prompt
            system_prompt = """
            You are an assistant that helps extract patterns from documents related to climate risk.
            
            A pattern is a recurring structure, theme, or concept that appears in the document.
            Patterns should be specific, well-defined, and clearly present in the document.
            
            For each pattern you identify, provide:
            1. A short name or title
            2. A brief description
            3. Evidence from the document
            4. A quality assessment (hypothetical, emergent, or stable)
            
            Format your response as JSON with the following structure:
            {
                "patterns": [
                    {
                        "name": "Pattern Name",
                        "description": "Pattern Description",
                        "evidence": "Evidence from document",
                        "quality_state": "emergent"
                    },
                    ...
                ]
            }
            
            Ensure your response is valid JSON that can be parsed programmatically.
            """
            
            # Create the user message with metadata if available
            metadata_text = ""
            if metadata:
                metadata_text = f"\n\nDocument Metadata: {json.dumps(metadata, indent=2)}"
                
            user_message = f"Document ID: {doc_id}{metadata_text}\n\nDocument Content:\n{content}"
            
            # Call the Claude API
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract the response text
            response_text = response.content[0].text
            
            # Parse the patterns from the response
            try:
                # Extract JSON from the response text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    response_data = json.loads(json_str)
                    
                    # Extract patterns from the response
                    extracted_patterns = response_data.get("patterns", [])
                    
                    # Add IDs and document reference to patterns
                    patterns = []
                    for i, pattern in enumerate(extracted_patterns):
                        pattern_id = f"pattern-{doc_id}-{i}"
                        patterns.append({
                            "id": pattern_id,
                            "name": pattern.get("name", f"Pattern {i+1}"),
                            "description": pattern.get("description", "No description"),
                            "evidence": pattern.get("evidence", "No evidence"),
                            "quality_state": pattern.get("quality_state", "hypothetical"),
                            "source_document": doc_id
                        })
                else:
                    # Fallback if JSON parsing fails
                    logger.warning(f"Could not extract JSON from Claude response for document {doc_id}")
                    patterns = [{
                        "id": f"pattern-{doc_id}-fallback",
                        "name": "Fallback Pattern",
                        "description": "Pattern extracted without proper JSON structure",
                        "evidence": "Response did not contain valid JSON",
                        "quality_state": "hypothetical",
                        "source_document": doc_id
                    }]
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from Claude response: {e}")
                patterns = [{
                    "id": f"pattern-{doc_id}-error",
                    "name": "Error Pattern",
                    "description": "Error parsing pattern from Claude response",
                    "evidence": f"JSON parsing error: {str(e)}",
                    "quality_state": "hypothetical",
                    "source_document": doc_id
                }]
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Create the result
            result = {
                "patterns": patterns,
                "model": "claude-3-opus-20240229",
                "tokens_used": response.usage.output_tokens,
                "status": "success",
                "document_id": doc_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Track metrics
            claude_metrics.track_document(document, result, response_time_ms)
            
            # Save to cache if enabled
            if use_cache:
                cache_key = claude_cache.get_document_cache_key(document)
                claude_cache.save_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error processing document with Claude: {e}")
            error_result = {
                "error": str(e),
                "status": "error",
                "document_id": document.get("id", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Track the error in metrics
            claude_metrics.track_error("document", str(e), {"document_id": document.get("id", "unknown")})
            
            return error_result
            
    def _format_patterns_for_prompt(self, patterns: List[Dict[str, Any]]) -> str:
        """
        Format patterns for inclusion in a Claude prompt.
        
        Args:
            patterns: The patterns to format
            
        Returns:
            A formatted string containing the patterns
        """
        if not patterns:
            return "No relevant patterns found."
            
        formatted_patterns = []
        
        for i, pattern in enumerate(patterns):
            pattern_text = f"""
            Pattern {i+1}: {pattern.get('name', f'Pattern {i+1}')}
            Description: {pattern.get('description', 'No description available')}
            Quality: {pattern.get('quality_state', 'unknown')}
            """
            formatted_patterns.append(pattern_text)
            
        return "\n".join(formatted_patterns)
        
    async def _mock_process_query(self, query: str, context: Dict[str, Any], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a mock response for a query.
        
        This method is used for testing when no Claude API key is available.
        
        Args:
            query: The query to process
            context: The context for the query
            patterns: The patterns to leverage
            
        Returns:
            A dictionary containing the mock response
        """
        logger.info(f"Generating mock response for query: {query}")
        
        # Generate a unique query ID
        query_id = str(uuid.uuid4())
        
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Check if the query is about Habitat Evolution
        if "habitat" in query.lower() or "evolution" in query.lower():
            response = """
            Habitat Evolution is a system designed to detect and evolve coherent patterns, while enabling the observation of semantic change across the system. It's built on the principles of pattern evolution and co-evolution.
            
            The system includes several key components:
            1. A bidirectional flow system that enables communication between components
            2. Pattern-aware RAG for enhanced retrieval and generation
            3. ArangoDB for pattern persistence and relationship management
            4. Pattern evolution tracking to improve pattern quality over time
            
            This creates a complete functional loop from document ingestion through processing, persistence, and retrieval, with user interactions driving pattern evolution.
            """
        elif "pattern" in query.lower():
            response = """
            Patterns in Habitat Evolution are recurring structures, themes, or concepts that are detected and evolved over time. They go through quality states (hypothetical, emergent, stable) based on usage and feedback.
            
            Patterns are stored in ArangoDB and can form relationships with other patterns, creating a knowledge graph that evolves over time. The bidirectional flow system ensures that patterns are updated based on new information and user feedback.
            
            Pattern-aware RAG leverages these patterns to enhance retrieval and generation, providing more contextually relevant responses to user queries.
            """
        elif "dissonance" in query.lower() or "constructive dissonance" in query.lower():
            response = """
            Constructive dissonance in Habitat Evolution refers to the productive tension between patterns that can lead to new insights and pattern emergence. It's a key concept in the system's ability to evolve and adapt over time.
            
            When a query or document contains elements that challenge or contradict established patterns, but in a way that could lead to new insights, the system detects this as constructive dissonance. This dissonance is then used to drive pattern evolution, potentially creating new patterns or updating existing ones.
            
            The system includes a ConstructiveDissonanceService that calculates dissonance potential for patterns and queries, and an AccretiveWeedingService that uses dissonance metrics to determine which patterns to prune and which to preserve despite low usage.
            """
        elif "relational accretion" in query.lower():
            response = """
            Relational accretion in Habitat Evolution is an approach to modeling queries as actants in the system. Instead of projecting patterns onto queries, the system observes how queries interact with existing patterns and allows significance to emerge naturally.
            
            Queries gradually accrete relationships and significance through their interactions with the pattern space, starting with minimal "sign-ificance" and developing a distinct identity over time. This approach aligns with core Habitat Evolution principles of emergence over imposition, co-evolution, contextual reinforcement, and adaptive learning.
            
            The implementation includes services like SignificanceAccretionService, ClaudeBaselineService, and AccretivePatternRAG, which work together to enable this relational accretion approach.
            """
        else:
            # Use patterns to inform the response if available
            pattern_info = ""
            if patterns:
                pattern_info = "Based on the relevant patterns, I can tell you that:"
                for pattern in patterns[:3]:  # Limit to top 3 patterns
                    name = pattern.get("name", "Unknown pattern")
                    description = pattern.get("description", "No description available")
                    pattern_info += f"\n- {name}: {description}"
            
            response = f"""
            I understand you're asking about "{query}". {pattern_info}
            
            The Habitat Evolution system is designed to process queries like yours by retrieving relevant patterns from its knowledge base and generating responses that leverage those patterns. The system evolves over time based on usage and feedback, improving its ability to provide accurate and helpful responses.
            
            If you have more specific questions about Habitat Evolution or its components, feel free to ask!
            """
            
        # Create the result with the same structure as the real API response
        result = {
            "query_id": query_id,
            "timestamp": timestamp,
            "query": query,
            "context": context,
            "patterns": patterns,
            "response": response.strip(),
            "model": "claude-3-opus-20240229-mock",
            "tokens_used": 0
        }
        
        return result
        
    async def _mock_process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mock patterns for a document.
        
        This method is used for testing when no Claude API key is available.
        
        Args:
            document: The document to process
            
        Returns:
            A dictionary containing the mock patterns
        """
        logger.info(f"Generating mock patterns for document: {document.get('id', 'unknown')}")
        
        # Extract the document content and metadata
        content = document.get("content", "").lower()
        doc_id = document.get("id", f"doc-{uuid.uuid4().hex[:8]}")
        title = document.get("title", "Untitled Document")
        
        # Generate patterns based on document content
        patterns = []
        
        if "habitat" in content or "evolution" in content:
            patterns.append({
                "id": f"pattern-{doc_id}-1",
                "name": "Habitat Evolution Concept",
                "description": "The core concept of Habitat Evolution as a pattern evolution system",
                "evidence": "References to Habitat Evolution in the document",
                "quality_state": "emergent",
                "source_document": doc_id
            })
            
        if "pattern" in content:
            patterns.append({
                "id": f"pattern-{doc_id}-2",
                "name": "Pattern Concept",
                "description": "The concept of patterns as recurring structures or themes",
                "evidence": "References to patterns in the document",
                "quality_state": "hypothetical",
                "source_document": doc_id
            })
            
        if "bidirectional" in content or "flow" in content:
            patterns.append({
                "id": f"pattern-{doc_id}-3",
                "name": "Bidirectional Flow",
                "description": "The concept of bidirectional communication between components",
                "evidence": "References to bidirectional flow in the document",
                "quality_state": "emergent",
                "source_document": doc_id
            })
            
        if "rag" in content or "retrieval" in content:
            patterns.append({
                "id": f"pattern-{doc_id}-4",
                "name": "Pattern-Aware RAG",
                "description": "Enhanced retrieval and generation using patterns",
                "evidence": "References to RAG or retrieval in the document",
                "quality_state": "stable",
                "source_document": doc_id
            })
            
        if "dissonance" in content or "constructive" in content:
            patterns.append({
                "id": f"pattern-{doc_id}-5",
                "name": "Constructive Dissonance",
                "description": "The concept of productive tension between patterns leading to emergence",
                "evidence": "References to dissonance or constructive tension in the document",
                "quality_state": "hypothetical",
                "source_document": doc_id
            })
            
        if "relational" in content or "accretion" in content:
            patterns.append({
                "id": f"pattern-{doc_id}-6",
                "name": "Relational Accretion",
                "description": "Modeling queries as actants that accrete significance through interactions",
                "evidence": "References to relational accretion in the document",
                "quality_state": "emergent",
                "source_document": doc_id
            })
            
        # If no specific patterns were found, add a generic one
        if not patterns:
            patterns.append({
                "id": f"pattern-{doc_id}-generic",
                "name": f"Generic Pattern: {title}",
                "description": "A generic pattern extracted from the document",
                "evidence": "The document content",
                "quality_state": "hypothetical",
                "source_document": doc_id
            })
            
        # Create the result
        result = {
            "patterns": patterns,
            "model": "claude-3-opus-20240229-mock",
            "tokens_used": 0,
            "status": "success"
        }
        
        return result
        
    async def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using Claude.
        
        This method is provided for backward compatibility with code that
        expects a generate_text method.
        
        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        if self.use_mock:
            return await self._mock_generate_text(prompt, max_tokens)
            
        try:
            # Call the Claude API
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the response text
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Claude: {e}")
            return await self._mock_generate_text(prompt, max_tokens)
    
    async def _mock_generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate mock text for a prompt.
        
        This method is used for testing when no Claude API key is available.
        
        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        logger.info(f"Generating mock text for prompt: {prompt[:50]}...")
        
        # Check if the prompt is asking for JSON
        if "JSON" in prompt or "json" in prompt:
            if "dissonance" in prompt.lower():
                return """
                {
                    "dissonance_detected": true,
                    "dissonance_zones": [
                        {
                            "text": "climate adaptation strategies",
                            "pattern_id": "pattern-1",
                            "potential": 0.85
                        },
                        {
                            "text": "economic impacts",
                            "pattern_id": "pattern-2",
                            "potential": 0.72
                        }
                    ]
                }
                """
            elif "interaction" in prompt.lower() or "analyze" in prompt.lower():
                return """
                {
                    "interaction_strength": 0.78,
                    "pattern_reinforcement": 0.65,
                    "emergent_properties": ["adaptive response", "contextual awareness"],
                    "transition_points": ["climate risk assessment", "adaptation planning"],
                    "dissonance_zones": ["economic vs. environmental priorities"]
                }
                """
            else:
                return """
                {
                    "enhanced_query": "How do climate change impacts affect coastal communities?",
                    "semantic_dimensions": {
                        "specificity": 0.75,
                        "complexity": 0.68,
                        "domain_relevance": 0.82
                    },
                    "potential_domains": ["climate_science", "coastal_management", "community_resilience"]
                }
                """
        
        # Default responses based on prompt content
        if "habitat" in prompt.lower() or "evolution" in prompt.lower():
            return "Habitat Evolution is a system designed to detect and evolve coherent patterns, while enabling the observation of semantic change across the system."
        elif "pattern" in prompt.lower():
            return "Patterns in Habitat Evolution represent recurring structures, themes, or concepts that are detected and evolved over time based on usage and feedback."
        elif "enhance" in prompt.lower() or "query" in prompt.lower():
            return "The enhanced query focuses on climate adaptation strategies for coastal communities facing sea level rise and extreme weather events."
        else:
            return "I've analyzed the information provided and can offer insights based on the patterns and context available in the Habitat Evolution system."
