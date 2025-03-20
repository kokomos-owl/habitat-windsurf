"""
FastAPI implementation for the Topology Framework.

This module provides a RESTful API for interacting with the topology framework,
allowing other components to query topology states, analyze patterns, and
track topology evolution.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import json
import logging

from habitat_evolution.pattern_aware_rag.topology.models import (
    FrequencyDomain, Boundary, ResonancePoint, FieldMetrics, TopologyState, TopologyDiff
)
from habitat_evolution.pattern_aware_rag.topology.manager import TopologyManager
from habitat_evolution.pattern_aware_rag.learning.pattern_id import PatternID
from habitat_evolution.pattern_aware_rag.learning.learning_control import LearningWindow


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Topology Framework API",
    description="API for interacting with the Pattern Topology Framework",
    version="0.1.0"
)

# Pydantic models for API requests and responses
class TimeRange(BaseModel):
    """Time range for topology analysis."""
    start: datetime
    end: datetime = Field(default_factory=datetime.now)


class PatternReference(BaseModel):
    """Reference to a pattern."""
    pattern_id: str


class LearningWindowReference(BaseModel):
    """Reference to a learning window."""
    window_id: str


class TopologyAnalysisRequest(BaseModel):
    """Request for topology analysis."""
    pattern_ids: List[str]
    window_ids: Optional[List[str]] = None
    time_range: TimeRange


class TopologyStateResponse(BaseModel):
    """Response containing topology state."""
    id: str
    timestamp: datetime
    frequency_domains_count: int
    boundaries_count: int
    resonance_points_count: int
    field_coherence: float
    state_json: Optional[str] = None


class TopologyDiffResponse(BaseModel):
    """Response containing topology diff."""
    from_state_id: str
    to_state_id: str
    added_domains: List[str]
    removed_domains: List[str]
    modified_domains: List[str]
    added_boundaries: List[str]
    removed_boundaries: List[str]
    modified_boundaries: List[str]
    added_resonance_points: List[str]
    removed_resonance_points: List[str]
    modified_resonance_points: List[str]
    field_metrics_changes: Dict[str, Dict[str, float]]


# Dependency for getting the topology manager
def get_topology_manager():
    """Get the topology manager instance."""
    # In a real application, this would be a singleton or retrieved from a dependency injection container
    return TopologyManager(persistence_mode=True)


# API endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Topology Framework API"}


@app.get("/topology/states", response_model=List[TopologyStateResponse], tags=["Topology"])
async def get_topology_states(
    limit: int = Query(10, ge=1, le=100),
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Get recent topology states."""
    try:
        states = manager.get_recent_states(limit)
        return [
            TopologyStateResponse(
                id=state.id,
                timestamp=state.timestamp,
                frequency_domains_count=len(state.frequency_domains),
                boundaries_count=len(state.boundaries),
                resonance_points_count=len(state.resonance_points),
                field_coherence=state.field_metrics.coherence
            )
            for state in states
        ]
    except Exception as e:
        logger.error(f"Error getting topology states: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topology/states/{state_id}", response_model=TopologyStateResponse, tags=["Topology"])
async def get_topology_state(
    state_id: str,
    include_full_state: bool = False,
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Get a specific topology state by ID."""
    try:
        state = manager.get_state_by_id(state_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Topology state {state_id} not found")
        
        response = TopologyStateResponse(
            id=state.id,
            timestamp=state.timestamp,
            frequency_domains_count=len(state.frequency_domains),
            boundaries_count=len(state.boundaries),
            resonance_points_count=len(state.resonance_points),
            field_coherence=state.field_metrics.coherence
        )
        
        if include_full_state:
            response.state_json = state.to_json()
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topology state {state_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topology/diff/{from_state_id}/{to_state_id}", response_model=TopologyDiffResponse, tags=["Topology"])
async def get_topology_diff(
    from_state_id: str,
    to_state_id: str,
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Get the difference between two topology states."""
    try:
        # Load the states
        from_state = manager.get_state_by_id(from_state_id)
        if not from_state:
            raise HTTPException(status_code=404, detail=f"Topology state {from_state_id} not found")
        
        to_state = manager.get_state_by_id(to_state_id)
        if not to_state:
            raise HTTPException(status_code=404, detail=f"Topology state {to_state_id} not found")
        
        # Calculate diff
        diff = to_state.diff(from_state)
        
        # Convert to response model
        return TopologyDiffResponse(
            from_state_id=from_state_id,
            to_state_id=to_state_id,
            added_domains=list(diff.get("added_domains", {}).keys()),
            removed_domains=list(diff.get("removed_domains", {}).keys()),
            modified_domains=list(diff.get("modified_domains", {}).keys()),
            added_boundaries=list(diff.get("added_boundaries", {}).keys()),
            removed_boundaries=list(diff.get("removed_boundaries", {}).keys()),
            modified_boundaries=list(diff.get("modified_boundaries", {}).keys()),
            added_resonance_points=list(diff.get("added_resonance_points", {}).keys()),
            removed_resonance_points=list(diff.get("removed_resonance_points", {}).keys()),
            modified_resonance_points=list(diff.get("modified_resonance_points", {}).keys()),
            field_metrics_changes=diff.get("field_metrics_changes", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topology diff between {from_state_id} and {to_state_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topology/analyze", response_model=TopologyStateResponse, tags=["Topology"])
async def analyze_topology(
    request: TopologyAnalysisRequest,
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Analyze patterns to generate a new topology state."""
    try:
        # Get patterns and windows
        # In a real application, these would be retrieved from a registry or database
        patterns = [PatternID(pattern_id) for pattern_id in request.pattern_ids]
        
        windows = []
        if request.window_ids:
            # In a real application, these would be retrieved from a registry or database
            windows = [LearningWindow(window_id) for window_id in request.window_ids]
        
        # Analyze patterns
        time_period = {
            "start": request.time_range.start,
            "end": request.time_range.end
        }
        
        state = manager.analyze_patterns(patterns, windows, time_period)
        
        # Return response
        return TopologyStateResponse(
            id=state.id,
            timestamp=state.timestamp,
            frequency_domains_count=len(state.frequency_domains),
            boundaries_count=len(state.boundaries),
            resonance_points_count=len(state.resonance_points),
            field_coherence=state.field_metrics.coherence,
            state_json=state.to_json()
        )
    except Exception as e:
        logger.error(f"Error analyzing topology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topology/frequency-domains", tags=["Topology Components"])
async def get_frequency_domains(
    state_id: Optional[str] = None,
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Get frequency domains from the current or specified topology state."""
    try:
        state = None
        if state_id:
            state = manager.get_state_by_id(state_id)
            if not state:
                raise HTTPException(status_code=404, detail=f"Topology state {state_id} not found")
        else:
            state = manager.current_state
            if not state:
                raise HTTPException(status_code=404, detail="No current topology state available")
        
        # Convert to dict for JSON serialization
        domains = {}
        for domain_id, domain in state.frequency_domains.items():
            domains[domain_id] = {
                "id": domain.id,
                "dominant_frequency": domain.dominant_frequency,
                "bandwidth": domain.bandwidth,
                "phase_coherence": domain.phase_coherence,
                "center_coordinates": domain.center_coordinates,
                "radius": domain.radius,
                "pattern_ids": list(domain.pattern_ids)
            }
        
        return domains
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting frequency domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topology/boundaries", tags=["Topology Components"])
async def get_boundaries(
    state_id: Optional[str] = None,
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Get boundaries from the current or specified topology state."""
    try:
        state = None
        if state_id:
            state = manager.get_state_by_id(state_id)
            if not state:
                raise HTTPException(status_code=404, detail=f"Topology state {state_id} not found")
        else:
            state = manager.current_state
            if not state:
                raise HTTPException(status_code=404, detail="No current topology state available")
        
        # Convert to dict for JSON serialization
        boundaries = {}
        for boundary_id, boundary in state.boundaries.items():
            boundaries[boundary_id] = {
                "id": boundary.id,
                "domain_ids": boundary.domain_ids,
                "sharpness": boundary.sharpness,
                "permeability": boundary.permeability,
                "stability": boundary.stability,
                "dimensionality": boundary.dimensionality,
                "coordinates": boundary.coordinates
            }
        
        return boundaries
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting boundaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topology/resonance-points", tags=["Topology Components"])
async def get_resonance_points(
    state_id: Optional[str] = None,
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Get resonance points from the current or specified topology state."""
    try:
        state = None
        if state_id:
            state = manager.get_state_by_id(state_id)
            if not state:
                raise HTTPException(status_code=404, detail=f"Topology state {state_id} not found")
        else:
            state = manager.current_state
            if not state:
                raise HTTPException(status_code=404, detail="No current topology state available")
        
        # Convert to dict for JSON serialization
        points = {}
        for point_id, point in state.resonance_points.items():
            points[point_id] = {
                "id": point.id,
                "coordinates": point.coordinates,
                "strength": point.strength,
                "stability": point.stability,
                "attractor_radius": point.attractor_radius,
                "contributing_pattern_ids": point.contributing_pattern_ids
            }
        
        return points
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting resonance points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topology/field-metrics", tags=["Topology Components"])
async def get_field_metrics(
    state_id: Optional[str] = None,
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Get field metrics from the current or specified topology state."""
    try:
        state = None
        if state_id:
            state = manager.get_state_by_id(state_id)
            if not state:
                raise HTTPException(status_code=404, detail=f"Topology state {state_id} not found")
        else:
            state = manager.current_state
            if not state:
                raise HTTPException(status_code=404, detail="No current topology state available")
        
        # Convert to dict for JSON serialization
        metrics = {
            "coherence": state.field_metrics.coherence,
            "energy_density": state.field_metrics.energy_density,
            "adaptation_rate": state.field_metrics.adaptation_rate,
            "homeostasis_index": state.field_metrics.homeostasis_index,
            "entropy": state.field_metrics.entropy
        }
        
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting field metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topology/export", tags=["Topology"])
async def export_topology_state(
    state_id: Optional[str] = None,
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Export a topology state as JSON."""
    try:
        state = None
        if state_id:
            state = manager.get_state_by_id(state_id)
            if not state:
                raise HTTPException(status_code=404, detail=f"Topology state {state_id} not found")
        else:
            state = manager.current_state
            if not state:
                raise HTTPException(status_code=404, detail="No current topology state available")
        
        return json.loads(state.to_json())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting topology state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topology/import", tags=["Topology"])
async def import_topology_state(
    state_json: Dict[str, Any],
    manager: TopologyManager = Depends(get_topology_manager)
):
    """Import a topology state from JSON."""
    try:
        # Convert dict to JSON string
        json_str = json.dumps(state_json)
        
        # Load state
        state = manager.load_from_serialized(json_str)
        
        return {
            "message": f"Successfully imported topology state {state.id}",
            "state_id": state.id
        }
    except Exception as e:
        logger.error(f"Error importing topology state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add event handlers
@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Topology Framework API starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Topology Framework API shutting down")


# Main entry point
def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_api()
