"""
User-facing API for pattern evolution system.
"""
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ...core.pattern.types import Pattern, FieldState
from ...core.pattern.evolution import FieldDrivenPatternManager
from ...core.quality.analyzer import PatternQualityAnalyzer
from ...adapters.document.processor import DocumentProcessor
from ...adapters.pattern.transformer import PatternTransformer

class DocumentInput(BaseModel):
    """Document input model."""
    content: str
    context: Optional[Dict[str, Any]] = None

class PatternResponse(BaseModel):
    """Pattern response model."""
    id: str
    coherence: float
    energy: float
    state: str
    metrics: Dict[str, float]
    relationships: List[str]

class FieldStateResponse(BaseModel):
    """Field state response model."""
    gradients: Dict[str, float]
    patterns: List[PatternResponse]
    timestamp: float

class PatternAPI:
    """API for pattern evolution system."""
    
    def __init__(self):
        self.pattern_manager = FieldDrivenPatternManager()
        self.quality_analyzer = PatternQualityAnalyzer()
        self.document_processor = DocumentProcessor()
        self.pattern_transformer = PatternTransformer()
        
        # Create FastAPI app
        self.app = FastAPI(title="Pattern Evolution API")
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.post("/process_document",
                      response_model=FieldStateResponse)
        async def process_document(document: DocumentInput):
            """Process document for patterns."""
            try:
                # Extract features
                features = self.document_processor.process_document(
                    document.content
                )
                
                # Convert to field state
                field_state = self.document_processor.to_field_state(features)
                
                # Create patterns from features
                patterns = []
                for p in features.patterns:
                    pattern = {
                        'id': f"pattern_{len(patterns)}",
                        'coherence': p['coherence'],
                        'energy': p['energy'],
                        'state': 'ACTIVE',
                        'metrics': {},
                        'relationships': []
                    }
                    patterns.append(pattern)
                
                # Evolve patterns
                evolved = await self.pattern_manager.evolve_patterns(
                    field_state,
                    patterns
                )
                
                # Transform response
                return FieldStateResponse(
                    gradients=field_state.gradients.__dict__,
                    patterns=[PatternResponse(**p) for p in evolved],
                    timestamp=field_state.timestamp
                )
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing document: {str(e)}"
                )
        
        @self.app.get("/pattern/{pattern_id}",
                     response_model=PatternResponse)
        async def get_pattern(pattern_id: str):
            """Get pattern by ID."""
            try:
                # Get pattern (implement storage later)
                pattern = {
                    'id': pattern_id,
                    'coherence': 0.5,
                    'energy': 0.5,
                    'state': 'ACTIVE',
                    'metrics': {},
                    'relationships': []
                }
                
                return PatternResponse(**pattern)
                
            except Exception as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"Pattern not found: {pattern_id}"
                )
        
        @self.app.post("/evolve_patterns",
                      response_model=List[PatternResponse])
        async def evolve_patterns(patterns: List[Pattern],
                                field_state: FieldState):
            """Evolve patterns in field."""
            try:
                # Evolve patterns
                evolved = await self.pattern_manager.evolve_patterns(
                    field_state,
                    patterns
                )
                
                return [PatternResponse(**p) for p in evolved]
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error evolving patterns: {str(e)}"
                )
