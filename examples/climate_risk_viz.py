"""Example visualization of Martha's Vineyard climate risk patterns."""

from datetime import datetime, timedelta
from src.flow_representation.core.flow_state import FlowState, FlowDimensions
from src.flow_representation.visualization.flow_representation import FlowRepresentation

def create_climate_risk_visualization():
    """Create visualization of climate risk patterns."""
    flow_rep = FlowRepresentation()
    
    # Create precipitation pattern states
    precip_states = [
        FlowDimensions(
            stability=0.85,  # High stability in measurements
            coherence=0.9,   # Strong temporal progression
            emergence_rate=0.7,  # 55% increase observed
            cross_pattern_flow=0.8,  # Links to flooding
            energy_state=0.75,  # Increasing intensity
            adaptation_rate=0.6   # Clear return period changes
        ),
        FlowDimensions(
            stability=0.87,
            coherence=0.92,
            emergence_rate=0.75,
            cross_pattern_flow=0.85,
            energy_state=0.8,
            adaptation_rate=0.65
        )
    ]
    
    # Create drought pattern states
    drought_states = [
        FlowDimensions(
            stability=0.7,   # Moderate stability
            coherence=0.85,  # Clear probability progression
            emergence_rate=0.8,  # 8.5% → 13% → 26%
            cross_pattern_flow=0.9,  # Strong link to wildfire
            energy_state=0.7,
            adaptation_rate=0.8
        ),
        FlowDimensions(
            stability=0.75,
            coherence=0.9,
            emergence_rate=0.85,
            cross_pattern_flow=0.95,
            energy_state=0.8,
            adaptation_rate=0.85
        )
    ]
    
    # Create wildfire pattern states
    wildfire_states = [
        FlowDimensions(
            stability=0.8,   # Based on Fire Weather Index
            coherence=0.85,  # Clear percentage increases
            emergence_rate=0.7,  # 44% increase by mid-century
            cross_pattern_flow=0.9,  # Strong link to drought
            energy_state=0.85,  # 94% increase by late-century
            adaptation_rate=0.75
        ),
        FlowDimensions(
            stability=0.85,
            coherence=0.9,
            emergence_rate=0.8,
            cross_pattern_flow=0.95,
            energy_state=0.9,
            adaptation_rate=0.8
        )
    ]
    
    # Add states to visualization
    base_time = datetime.now()
    patterns = [
        ("precipitation", precip_states),
        ("drought", drought_states),
        ("wildfire", wildfire_states)
    ]
    
    for pattern_name, states in patterns:
        for i, dimensions in enumerate(states):
            state = FlowState(
                dimensions=dimensions,
                pattern_id=pattern_name,
                timestamp=base_time + timedelta(hours=i),
                related_patterns=[p[0] for p in patterns if p[0] != pattern_name]
            )
            flow_rep.add_state(state)
    
    # Create visualization
    fig = flow_rep.create_visualization()
    
    # Show different dimension views
    for dimension in ["coherence", "cross_pattern_flow", "emergence_rate"]:
        flow_rep.set_active_dimension(dimension)
        fig = flow_rep.create_visualization()
        fig.write_html(f"mv_climate_risk_{dimension}.html")
    
    # Toggle between abstract and geographic views
    flow_rep.toggle_choropleth_mode()
    fig = flow_rep.create_visualization()
    fig.write_html("mv_climate_risk_geographic.html")

if __name__ == "__main__":
    create_climate_risk_visualization()
